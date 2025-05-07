import os

import hydra
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from scripts.utils import register_useful_resolvers

register_useful_resolvers()

CKPT_DIR = "/share/kuleshov/ma2238/runs/dllm-dev/"

# RUN_NAME="gsm8k-block4-bs96-keep4-causalencfalse-max20000ba-lr1e-4-warmup1000ba-gc1.0-wd1e-5-e2d2_phi_v1"
RUN_NAME = "gsm8k-block4-bs96-keep2-causalencfalse-max20000ba-lr1e-4-warmup1000ba-gc1.0-wd1e-5-bd3_phi_v1"

# E2D2
base_config = OmegaConf.load(os.path.join(CKPT_DIR, RUN_NAME, "config.yaml"))
overrides = [
    "composer.loggers=null",
    "model.config.sampler_config.config.block_size=4",
    "model.config.sampler_config.config.first_hitting=true",
    "model.config.sampler_config.config.use_x0_pred=true",
    "model.config.sampler_config.config.greedy=true",
    "model.config.sampler_config.config.low_confidence_remasking=true",
]
config = OmegaConf.merge(base_config, OmegaConf.from_dotlist(overrides))

ckpt_file = (
    f"/share/kuleshov/ma2238/runs/dllm-dev/${RUN_NAME}/checkpoints/best-rank0.pt"
)

model = hydra.utils.instantiate(
    config.model,
    _convert_="all",
).to("cuda")
model.eval()

ckpt = torch.load(ckpt_file, weights_only=False)
state_dict = ckpt["state"]["model"]
torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, "model.")
model.load_state_dict(state_dict)

tokenizer = AutoTokenizer.from_pretrained(
    config.tokenizer.pretrained_model_name_or_path,
    trust_remote_code=True,
    use_fast=False,
)
prompt = tokenizer(
    "<|im_end|>Please reason step by step, and put your final answer within \\boxed{}. Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?<|im_end|>Answer:",
    return_tensors="pt",
    padding=True,
    truncation=True,
).to("cuda")

"""
Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.
She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.
\\boxed{18}
"""

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()

samples, NFEs = model.generate(
    batch_size=1, max_seq_len=512, context=prompt["input_ids"]
)

end.record()
torch.cuda.synchronize()
print(f"Time taken: {start.elapsed_time(end)} ms")
print(tokenizer.decode(samples[0], skip_special_tokens=True))
