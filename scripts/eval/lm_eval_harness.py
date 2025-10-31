"""
This file is inspired by the code from https://github.com/ML-GSAI/SMDM
"""

import json
import logging
import os
import sys
from typing import Any, Dict, List, Tuple

import accelerate
import hydra
import numpy as np
import torch
from tabulate import tabulate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.loggers.evaluation_tracker import EvaluationTracker
from lm_eval.utils import make_table
from omegaconf import DictConfig, OmegaConf
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

log = logging.getLogger(__name__)

# ------------------------------------------------------------------------------------------------ #
# Eval Harness
# ------------------------------------------------------------------------------------------------ #

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
                # TODO: What is going on with this comment?  simple_evaluate can take a dict for gen_kwargs..
                Ideally this should be passed via `lm_eval.evaluator.simple_evaluate`,
                however this method expects `gen_kwargs` as string with comma-separated
                arguments, which is not compatible in our hydra framework.
            throughput_run (bool): Whether to run the evaluation throughput.
            model_config_overrides (dict[str, Any]): Model config overrides.
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

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        raise NotImplementedError

    @property
    def tokenizer_name(self):
        return self.tokenizer.name_or_path

    def apply_chat_template(
        self, chat_history: List[Dict[str, str]], add_generation_prompt: bool = True
    ):
        return self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    def process_requests(
        self, ds: Dataset
    ) -> Dataset:
        res = []
        
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
            
            # Generate sample
            assert elem["prefix_ids"].ndim == 1
            inputs = elem["prefix_ids"][None, ...].to(self.device)
            generations = self.model.generate(inputs=inputs, **self.gen_kwargs)
            
            # Record timing
            tracker.record_timing(
                start_event, end_event, generations.numel() - inputs.numel()
            )
            
            # Decode sample
            assert generations.shape[0] == 1
            generation = self.tokenizer.decode(generations[0, inputs.shape[1] :])
            
            # Extract prefix and answer
            prefix, answer = elem["prefix_text"], elem["answer"]

            if self.rank == 0:
                table_data = [
                    ["Example", i],
                    ["Prefix", prefix],
                    ["Generation", generation],
                    ["Answer", answer],
                ]
                log.debug("\n" + tabulate(table_data, tablefmt="grid", maxcolwidths=[20, 100]) + "\n")
            
            res.append(
                {
                    "prefix_text": prefix,
                    "generation": generation,
                    "answer": answer,
                }
            )
            
        # Log throughput stats
        if self.rank == 0:
            stats_str = tracker.get_stats_string()
            if stats_str:
                log.info(stats_str)
        
        return Dataset.from_list(res)

    def save_generations(self, results: Dataset) -> None:
        samples_path = f"{self.generated_samples_output_path}/rank{self.rank}.jsonl"
        log.info(f"Saving results to {samples_path}")
        results.to_json(samples_path)

    @staticmethod
    def prepare_requests(
        requests: List[Instance], tokenizer: PreTrainedTokenizer
    ) -> Dataset:
        data = []
        for req in requests:
            prefix_text, target_text = req.args[0], req.args[1]
            prefix_tokens = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
            data.append(
                {
                    "prefix_text": prefix_text,
                    "prefix_ids": prefix_tokens,
                    "target": target_text,
                    "answer": req.doc["answer"],
                }
            )
        ds = Dataset.from_list(data)
        ds = ds.with_format("torch")
        return ds

    def generate_until(self, requests: List[Instance], **generation_kwargs: Any) -> List[str]:
        ds = self.prepare_requests(requests, self.tokenizer)
        generations = self.process_requests(ds)
        self.save_generations(generations)
        log.info(f"RANK {self.rank} completed!")
        return [r["generation"] for r in generations]


# ------------------------------------------------------------------------------------------------ #
# Utilities
# ------------------------------------------------------------------------------------------------ #

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
        self.enabled = enabled
        self.warmup = warmup
        self.max_samples = max_samples
        self.output_path = output_path
        self.rank = rank
        self.throughputs: List[float] = []
        self.current_idx = 0
    
    def start_timing(self) -> Tuple[torch.cuda.Event | None, torch.cuda.Event | None]:
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
        if self.rank == 0 and start_event is not None and end_event is not None:
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time_s = start_event.elapsed_time(end_event) / 1000
            tput = num_tokens / elapsed_time_s
            if self.current_idx >= self.warmup:
                self.throughputs.append(tput)
    
    def should_exit(self) -> bool:
        return self.enabled and self.current_idx >= self.max_samples + self.warmup
    
    def save_and_exit(self) -> None:
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
        if self.current_idx >= self.warmup and self.throughputs:
            return f"Thput (tok/s): {np.mean(self.throughputs):0.2f} +/- {np.std(self.throughputs):0.2f}"
        elif self.throughputs:
            return f"Thput (tok/s): {self.throughputs[-1]:0.2f}"
        return ""

def show_config(cfg: DictConfig) -> None:
    log.debug("Hydra config:")
    log.debug(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=4, sort_keys=True))

def save_results(results: Dict[str, Any], cfg: DictConfig) -> None:
    # Save generations, answers, parsed results and other context
    if "samples" in results:
        samples = results.pop("samples")

        # - It is crucial that the either the output path for EvaluationTracker ends exactly in ".json"
        # or that the model being evaluated has `model_args` so that the tracker can automatically infer a model name from; see:
        # https://github.com/EleutherAI/lm-evaluation-harness/blob/68b03658ace40fa93221518b30096485a2387c58/lm_eval/loggers/evaluation_tracker.py#L68-L81
        # This fails when an LM subclass like LMEvalHarnessModel loads a model manually and bypasses the LM eval interface
        # for doing so via `model` and `model_args` for `simple_evaluate`.
        # - This code uses a .json suffix then to avoid ultimately calling .joinpath on a None object for the inferred model name here:
        # https://github.com/EleutherAI/lm-evaluation-harness/blob/68b03658ace40fa93221518b30096485a2387c58/lm_eval/loggers/evaluation_tracker.py#L239-L241
        # LM eval also buries the trace for this error and only reports the cryptic exception message: 
        # `TypeError('expected str, bytes or os.PathLike object, not NoneType')` in that case.
        tracker_path = f"{cfg.task.model.generated_samples_output_path}/tracker.json"

        log.info(f"Saving evaluation tracker results to {tracker_path!r}")
        evaluation_tracker = EvaluationTracker(output_path=tracker_path)
        evaluation_tracker.save_results_aggregated(results=results, samples=samples)
        for task_name, config in results["configs"].items():
            evaluation_tracker.save_results_samples(
                task_name=task_name, samples=samples[task_name]
            )
    else:
        log.info(
            "No samples found to save for task; set `log_samples` to "
            "True to enable this, otherwise only metrics will be available."
        )
    
    # Show metrics by group
    if "groups" in results:
        log.info(f"Metrics by group:\n{make_table(results, 'groups')}")

    # Show and save overall metrics
    metrics_table = make_table(results)
    log.info(f"Metrics:\n{metrics_table}")
    metrics_path = f"{cfg.task.model.generated_samples_output_path}/metrics.txt"
    log.info(f"Saving metrics to {metrics_path!r}")
    with open(metrics_path, "w") as f:
        f.write(metrics_table)


# ------------------------------------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------------------------------------ #

@hydra.main(version_base=None, config_path="../../configs", config_name="eval_config")
def main(cfg: DictConfig) -> None:
    show_config(cfg)
    accelerator = accelerate.Accelerator()
    # TODO: why is this being reinitialized?
    accelerator = accelerate.Accelerator() if accelerator.num_processes > 1 else None
    rank = accelerator.local_process_index if accelerator is not None else 0
    set_seed(cfg.seed)
    model = hydra.utils.instantiate(cfg.task.model, accelerator=accelerator)
    # Using _convert_="all" is necessary to ensure that dict LM eval task 
    # configs are not passed to simple_evaluate as OmegaConf containers.
    results = hydra.utils.call(cfg.task, model=model, _convert_="all")
    if results is not None and rank == 0:
        save_results(results, cfg)

if __name__ == "__main__":
    register_useful_resolvers()
    main()
