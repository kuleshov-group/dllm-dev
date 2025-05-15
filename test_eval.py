import os

import hydra
from composer.models import HuggingFaceModel
from composer.utils import dist, reproducibility
from omegaconf import OmegaConf
from streaming import StreamingDataset
from transformers import AutoTokenizer
import yaml

from scripts.utils import (
    load_model_from_ckpt_dir_path,
    maybe_add_missing_special_tokens,
    register_useful_resolvers,
)

register_useful_resolvers()


MODEL_DIR = "/home/ubuntu/runs/owt-block4-bs128-keepbottom21-keepevery1-causalencfalse-max1000000ba-lr1e-5-warmup1000ba-gc1.0-wd1e-5-bd3lm_qwen600M_v3"

def main(eval_dataset_name):
    config = OmegaConf.load(os.path.join(MODEL_DIR, "config.yaml"))
    # config.eval_dataloader.num_workers=128
    
    reproducibility.seed_all(config.seed)

    config.composer.trainer.autoresume = False
    config.composer.trainer.save_folder = "/home/ubuntu/trash"

    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer.pretrained_model_name_or_path,
        trust_remote_code=True,
    )
    tokenizer = maybe_add_missing_special_tokens(tokenizer)

    model = load_model_from_ckpt_dir_path(
        path_to_ckpt_dir=MODEL_DIR,
        load_ema_weights=False,
        ckpt_file="best-rank0.pt",
        verbose=True,
    ).to("cuda")

    model = HuggingFaceModel(
        model=model,
        tokenizer=tokenizer,
        metrics=list(hydra.utils.instantiate(config.metrics).values()),
    )

    if eval_dataset_name != "owt_eval":
        with open(f"configs/dataset/{eval_dataset_name}.yaml", "r") as file:
            eval_dataset = yaml.safe_load(file)

        eval_dataset = hydra.utils.instantiate(
            eval_dataset,
            tokenizer=tokenizer,
        )
    else:
        eval_dataset = hydra.utils.instantiate(
            config.eval_dataset,
            tokenizer=tokenizer,
        )

    collator = hydra.utils.instantiate(config.collator, tokenizer=tokenizer)
    eval_sampler = (
        dist.get_sampler(eval_dataset, shuffle=False, drop_last=False)
        if not isinstance(eval_dataset, StreamingDataset)
        else None
    )

    eval_dataloader = hydra.utils.instantiate(
        config.eval_dataloader,
        _convert_="partial",
        dataset=eval_dataset,
        collate_fn=collator,
        sampler=eval_sampler,
    )

    trainer = hydra.utils.instantiate(
        config.composer.trainer,
        _convert_="all",
        model=model,
        eval_dataloader=eval_dataloader,
        loggers=None,
    )
    metrics = trainer.eval()
    print(metrics)
    print('EVAL DATASET', eval_dataset_name)


if __name__ == "__main__":
    for eval_dataset_name in ["owt_eval", "ptb", "wikitext2", "lm1b-qwen", "lambada", "ag_news", "scientific-papers-pubmed", "scientific-papers-arxiv"]:
        main(eval_dataset_name)
