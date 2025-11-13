"""
This file is inspired by the code from https://github.com/ML-GSAI/SMDM
"""

import json
import os
import re
import sys
from typing import Any, Dict, List, Tuple, Union

import accelerate
import hydra
import numpy as np
import torch
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.loggers.evaluation_tracker import EvaluationTracker
from lm_eval.utils import make_table
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    PreTrainedTokenizer,
)

from datasets import Dataset
from scripts.utils import (
    load_model_from_ckpt_dir_path,
    maybe_add_missing_special_tokens,
    register_useful_resolvers,
    set_seed,
)
from src.utils import fsspec_exists, fsspec_mkdirs


DEFAULT_CHAT_TEMPLATE = """{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}Please reason step by step, and put your final answer within $\\boxed{}$. {{ message['content'] }}{% elif message['role'] == 'assistant' %}{{ eos_token }}Answer: {{ message['content'] }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ eos_token }}Answer: {% endif %}"""


class LMEvalHarnessModel(LM):

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        generated_samples_output_path: str,
        tokenizer: PreTrainedTokenizer,
        pretrained_model_revision: str | None = None,
        load_ema_weights: bool = False,
        ckpt_file: str = "best-rank0.pt",  # best-rank0.pt or latest-rank0.pt
        gen_kwargs: Any | None = None,
        accelerator: accelerate.Accelerator | None = None,
        throughput_run: bool = False,
        throughput_samples: int = 100,
        throughput_warmup: int = 100,
        model_config_overrides: dict[str, Any] | None = None,
        use_tokenizer_chat_template: bool = True,
    ):
        """
        Args:
            pretrained_model_name_or_path (str): Path to ckpt dir or HF model repo.
            generated_samples_output_path (str): Path to generated samples dir.
            tokenizer (str): Tokenizer name or path.
            pretrained_model_revision (Optional[str]): Revision (e.g., commit id)
                passed to `.from_pretrained` model instantiation.
            load_ema_weights (bool): Whether to load ema weights (for local ckpts).
            ckpt_file (str): Name of ckpt file (for local ckpts).
            gen_kwargs (dict): Generator kwargs.
                Ideally this should be passed via `lm_eval.evaluator.simple_evaluate`,
                however this method expects `gen_kwargs` as string with comma-separated
                arguments, which is not compatible in our hydra framework.
            throughput_run (bool): Whether to run the evaluation throughput.
            model_config_overrides (dict[str, Any]): Model config overrides.
            use_tokenizer_chat_template (bool): Whether to use the tokenizer's chat template.
                If True, the tokenizer's chat template will be used.
                If False, the default chat template will be used.
        """
        super().__init__()
        self.generated_samples_output_path = generated_samples_output_path
        if not fsspec_exists(self.generated_samples_output_path):
            fsspec_mkdirs(self.generated_samples_output_path)
        self.accelerator = accelerator
        if self.accelerator is not None:
            device = self.accelerator.device
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._rank = 0
            self._world_size = 1
        self.device = torch.device(f"{device}")

        model_config_overrides = (
            {} if model_config_overrides is None else model_config_overrides
        )
        if fsspec_exists(os.path.join(pretrained_model_name_or_path, "config.yaml")):
            model = load_model_from_ckpt_dir_path(
                path_to_ckpt_dir=pretrained_model_name_or_path,
                load_ema_weights=load_ema_weights,
                ckpt_file=ckpt_file,
                device=self.device,
                **model_config_overrides,
            )
        else:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    pretrained_model_name_or_path,
                    trust_remote_code=True,
                    revision=pretrained_model_revision,
                    **model_config_overrides,
                )
            except:  # Model not compatible with CausalLM
                model = AutoModelForMaskedLM.from_pretrained(
                    pretrained_model_name_or_path,
                    trust_remote_code=True,
                    revision=pretrained_model_revision,
                    **model_config_overrides,
                )
        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = maybe_add_missing_special_tokens(tokenizer)
        self.gen_kwargs = gen_kwargs
        self.throughput_run = throughput_run
        self.throughput_warmup = throughput_warmup
        self.throughput_samples = throughput_samples
        self.use_tokenizer_chat_template = use_tokenizer_chat_template
        if self.use_tokenizer_chat_template:
            self.chat_template_string = self.tokenizer.chat_template
            if self.chat_template_string is None:
                raise ValueError("Tokenizer chat template is not set")
            if not isinstance(self.chat_template_string, str):
                raise ValueError(f"Tokenizer chat template is not a string; got {type(self.chat_template_string)}")
        else:
            self.chat_template_string = DEFAULT_CHAT_TEMPLATE

    @staticmethod
    def prepare_requests(
        requests: List[Instance], tokenizer: PreTrainedTokenizer
    ) -> Dataset:
        """
        Convert lm_eval requests to a HuggingFace Dataset with tokenization.
        
        Args:
            requests: List of Instance objects from lm_eval, where each instance
                has .args containing (prefix, target) tuple and .doc containing document metadata.
            tokenizer: Tokenizer to use for encoding the prefix text.
        
        Returns:
            Dataset with 'prefix_text', 'prefix', 'target', and 'answer' columns, formatted as torch tensors.
        """
        def _tokenize(e: Dict[str, Any]) -> Dict[str, Any]:
            # The evaluation harness has already formatted the text with chat template
            ctx = e["prefix"]
            prefix_tokens = tokenizer(ctx, add_special_tokens=False)["input_ids"]
            return {
                "prefix_text": ctx,
                "prefix": prefix_tokens,
                "target": e["target"],
                "answer": e["answer"],
            }
        
        data = [
            {
                "prefix": req.args[0],
                "target": req.args[1],
                "answer": req.doc["answer"],
            }
            for req in requests
        ]
        ds = Dataset.from_list(data)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")
        return ds

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
        raise NotImplementedError

    def loglikelihood_rolling(self, requests) -> List[float]:
        raise NotImplementedError

    @property
    def tokenizer_name(self):
        return self.tokenizer.name_or_path

    def chat_template(self, chat_template: Union[bool, str] = False) -> str:
        print("chat_template called with chat_template arg: ", chat_template)
        if chat_template == False:
            raise ValueError("Usage of chat templates is required.")
        if chat_template == True:
            return self.chat_template_string
        raise NotImplementedError(f"Named chat template not supported: {chat_template}")

    def apply_chat_template(
        self, chat_history: List[Dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        import pdb; pdb.set_trace()
        result = self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            # Template should already include special tokens; see:
            # https://huggingface.co/docs/transformers/main/en/chat_templating#using-applychattemplate
            add_special_tokens=False,  
        )
        return result

    @staticmethod
    def parse_predicted_result(
        result: str, until_tokens: List[str], eos_token: str
    ) -> Tuple[str, str | None]:
        # Split on until tokens
        # TODO: why are these extra tokens not part of the stopping criteria already?
        for until in until_tokens + ["<|eot_id|>", eos_token]:
            result = result.split(until)[0]
        
        # Extract boxed answer if present
        # TODO: fix these hacks
        predicted_ans = None
        if "boxed{" in result:
            predicted_ans = result.split("boxed{")[1].split("}")[0]
            result = result.split("boxed{")[0] + "#### " + predicted_ans
            result = result.replace("$\\", "")
        
        return result, predicted_ans

    @staticmethod
    def parse_ground_truth_result(answer: str) -> str:
        # TODO: Is this just a gsm8k thing or what's the point of this?
        return answer.split("### ")[1]

    def process_requests(
        self, ds: Dataset
    ) -> List[Dict[str, str]]:
        res = []
        correct, total = 0, 0
        
        # Initialize throughput tracker
        tracker = ThroughputTracker(
            enabled=self.throughput_run,
            warmup=self.throughput_warmup,
            max_samples=self.throughput_samples,
            output_path=self.generated_samples_output_path,
            rank=self.rank,
        )
        
        for i, elem in tqdm(
            enumerate(ds), desc="Generating", total=len(ds), disable=(self.rank != 0)
        ):
            tracker.current_idx = i
            
            # Check if we should exit early for throughput benchmarking
            if tracker.should_exit():
                tracker.save_and_exit()
            
            # Start timing
            start_event, end_event = tracker.start_timing()
            
            # Generate sample(s)
            samples = self.model.generate(
                inputs=elem["prefix"][None, ...].to(self.device),
                # tokenizer=self.tokenizer,  # Uncomment for debugging
                **self.gen_kwargs,
            )
            
            # Record timing
            tracker.record_timing(
                start_event, end_event, samples.numel() - elem["prefix"].numel()
            )
            
            # Decode result
            raw_result = self.tokenizer.decode(samples[0, len(elem["prefix"]) :])
            
            # Parse predicted result
            result, predicted_ans = self.parse_predicted_result(
                raw_result, elem["target"]["until"], self.tokenizer.eos_token
            )
            
            # Parse ground truth
            ground_truth_ans = self.parse_ground_truth_result(elem["answer"])
            
            if self.rank == 0:
                print("=" * 20)
                print("Prefix: ", elem["prefix_text"])
                print("Raw prediction: ", raw_result)
                print("Parsed prediction: ", result)
                print("Predicted answer: ", predicted_ans)
                print("Raw ground truth: ", elem["answer"])
                print("Parsed ground truth: ", ground_truth_ans)
                print("=" * 20, end="\n\n")

            # Log accuracy
            if predicted_ans is not None and ground_truth_ans == predicted_ans:
                correct += 1
            total += 1

            res.append(
                {
                    "prefix": elem["prefix_text"],
                    "result": result,
                }
            )
            
            # Print stats
            if self.rank == 0:
                print(f"\nAccuracy: {correct}/{total} = {correct / total:.2%}\n")
                stats_str = tracker.get_stats_string()
                if stats_str:
                    print(stats_str)
        
        return res


    def save_results(self, results: List[Dict[str, str]]) -> None:
        samples_path = f"{self.generated_samples_output_path}/rank{self.rank}"
        with open(f"{samples_path}.json", "w") as f:
            json.dump(
                results,
                f,  # type: ignore
                indent=2,
            )

    def generate_until(self, requests, **generation_kwargs) -> List[str]:
        ds = self.prepare_requests(requests, self.tokenizer)
        results = self.process_requests(ds)
        self.save_results(results)
        print(f"RANK {self.rank} completed!")
        return [r["result"] for r in results]


class ThroughputTracker:
    """Tracks throughput metrics during generation."""
    
    def __init__(
        self,
        enabled: bool,
        warmup: int,
        max_samples: int,
        output_path: str,
        rank: int,
    ):
        """
        Initialize throughput tracker.
        
        Args:
            enabled: Whether throughput tracking is enabled (for benchmark runs).
            warmup: Number of warmup iterations before collecting throughput stats.
            max_samples: Maximum number of samples to collect for throughput run.
            output_path: Path to save throughput results.
            rank: Process rank for distributed settings.
        """
        self.enabled = enabled
        self.warmup = warmup
        self.max_samples = max_samples
        self.output_path = output_path
        self.rank = rank
        self.throughputs: List[float] = []
        self.current_idx = 0
    
    def start_timing(self) -> Tuple[torch.cuda.Event | None, torch.cuda.Event | None]:
        """
        Start timing for current iteration.
        
        Returns:
            Tuple of (start_event, end_event) if rank 0, else (None, None).
        """
        if self.rank == 0:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            return start_event, end_event
        return None, None
    
    def record_timing(
        self,
        start_event: torch.cuda.Event | None,
        end_event: torch.cuda.Event | None,
        num_tokens: int,
    ) -> None:
        """
        Record throughput for current iteration.
        
        Args:
            start_event: CUDA event marking start of generation.
            end_event: CUDA event marking end of generation.
            num_tokens: Number of tokens generated.
        """
        if self.rank == 0 and start_event is not None and end_event is not None:
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time_s = start_event.elapsed_time(end_event) / 1000
            tput = num_tokens / elapsed_time_s
            if self.current_idx >= self.warmup:
                self.throughputs.append(tput)
    
    def should_exit(self) -> bool:
        """
        Check if we've collected enough samples for throughput run.
        
        Returns:
            True if we should exit the generation loop.
        """
        return self.enabled and self.current_idx >= self.max_samples + self.warmup
    
    def save_and_exit(self) -> None:
        """Save throughput results and exit the program."""
        tputs_path = f"{self.output_path}/throughput-rank{self.rank}"
        with open(f"{tputs_path}.json", "w") as f:
            json.dump(
                {
                    "throughput_mean": np.mean(self.throughputs),
                    "throughput_std": np.std(self.throughputs),
                    "throughput_all": self.throughputs,
                },
                f,
                indent=2,
            )
        sys.exit(0)
    
    def get_stats_string(self) -> str:
        """
        Get throughput statistics as a formatted string.
        
        Returns:
            Formatted string with throughput stats, or empty string if no stats available.
        """
        if self.current_idx >= self.warmup and self.throughputs:
            return f"Thput (tok/s): {np.mean(self.throughputs):0.2f} +/- {np.std(self.throughputs):0.2f}"
        elif self.throughputs:
            return f"Thput (tok/s): {self.throughputs[-1]:0.2f}"
        return ""

@hydra.main(version_base=None, config_path="../../configs", config_name="eval_config")
def main(cfg: DictConfig) -> None:
    accelerator = accelerate.Accelerator()
    accelerator = accelerate.Accelerator() if accelerator.num_processes > 1 else None
    set_seed(cfg.seed)
    model = hydra.utils.instantiate(cfg.task.model, accelerator=accelerator)
    results = hydra.utils.call(cfg.task, model=model)
    if results is not None and (
        accelerator is None or accelerator.local_process_index == 0
    ):
        samples = results.pop("samples")
        evaluation_tracker = EvaluationTracker(output_path=cfg.output_path)
        evaluation_tracker.save_results_aggregated(results=results, samples=samples)
        for task_name, config in results["configs"].items():
            evaluation_tracker.save_results_samples(
                task_name=task_name, samples=samples[task_name]
            )
        print(make_table(results))
        metrics_f = f"{cfg.task.model.generated_samples_output_path}/metrics.txt"
        with open(metrics_f, "w") as f:
            f.write(make_table(results))
        if "groups" in results:
            print(make_table(results, "groups"))


if __name__ == "__main__":
    register_useful_resolvers()
    main()
