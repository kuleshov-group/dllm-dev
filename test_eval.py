import os

import hydra
import torch
from omegaconf import OmegaConf
from streaming import StreamingDataset
from composer.utils import dist, reproducibility
from composer.models import HuggingFaceModel
from transformers import AutoTokenizer
from scripts.utils import register_useful_resolvers, load_model_from_ckpt_dir_path

register_useful_resolvers()

CKPT_DIR = "/share/kuleshov/ma2238/runs/dllm-dev/"

RUN_NAME = "/home/ubuntu/runs/dllm-dev/gsm8k-block4-bs96-keep2-causalencfalse-max20000ba-lr1e-4-warmup1000ba-gc1.0-wd1e-5-bd3_phi_v3"
config = OmegaConf.load(os.path.join(CKPT_DIR, RUN_NAME, "config.yaml"))

ckpt_file = (
    f"{RUN_NAME}/checkpoints/latest-rank0.pt"
)

tokenizer = AutoTokenizer.from_pretrained(
    config.tokenizer.pretrained_model_name_or_path,
    trust_remote_code=True,
    use_fast=False,
)

# model = load_model_from_ckpt_dir_path(
#     path_to_ckpt_dir=RUN_NAME,
#     load_ema_weights=False,
# ).to('cuda')
model = hydra.utils.instantiate(
    config.model,
    _convert_="all",
)
model = HuggingFaceModel(
    model=model,
    tokenizer=tokenizer,
    metrics=list(hydra.utils.instantiate(config.metrics).values()),
)

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
callbacks = hydra.utils.instantiate(config.composer.callbacks)

trainer = hydra.utils.instantiate(
    config.composer.trainer,
    _convert_="all",
    model=model,
    eval_dataloader=eval_dataloader,
    callbacks=list(callbacks.values()),
    load_path=ckpt_file,
)
import ipdb ; ipdb.set_trace()
metrics = trainer.eval()
