import os

import hydra
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from scripts.utils import register_useful_resolvers

register_useful_resolvers()

### LOAD CHECKPOINT
CKPT_DIR = "/share/kuleshov/ma2238/runs/dllm-dev/"

# RUN_NAME="gsm8k-block4-bs96-keep4-causalencfalse-max20000ba-lr1e-4-warmup1000ba-gc1.0-wd1e-5-e2d2_phi_v1"
RUN_NAME = "/home/ubuntu/runs/dllm-dev/gsm8k-block4-bs96-keep2-causalencfalse-max20000ba-lr1e-4-warmup1000ba-gc1.0-wd1e-5-bd3_phi_v3"
base_config = OmegaConf.load(os.path.join(CKPT_DIR, RUN_NAME, "config.yaml"))
sampler_overrides = [
    "composer.loggers=null",
    "model.config.sampler_config.block_size=4",
    "model.config.sampler_config.first_hitting=true",
    "model.config.sampler_config.use_x0_pred=true",
    "model.config.sampler_config.greedy=true",
    "model.config.sampler_config.low_confidence_remasking=true",
    "model.config.sampler_config.disable_cache=false",
    "model.config.sampler_config.kv_caching=false",
    "model.config.sampler_config.min_t=1e-5",
    "model.config.sampler_config.shift_logits=true",
    "model.config.sampler_config.top_p=0.85",
    "model.config.sampler_config.num_steps=1000",
    "model.config.sampler_config.pad_context=false",
]
config = OmegaConf.merge(base_config, OmegaConf.from_dotlist(sampler_overrides))

ckpt_file = (
    f"{RUN_NAME}/checkpoints/best-rank0.pt"
)

model = hydra.utils.instantiate(
    config.model,
    _convert_="all",
)
model.eval()

ckpt = torch.load(ckpt_file, weights_only=False)
state_dict = ckpt["state"]["model"]
torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, "model.")
model.load_state_dict(state_dict, strict=False)
model.to('cuda')


### PREPARE SAMPLING
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
).to(model.device)

"""
Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.
She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.
\\boxed{18}
"""


### SAMPLE
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()

samples, NFEs = model.generate(
    batch_size=1, max_length=512, context=prompt["input_ids"], tokenizer=tokenizer
)
end.record()
torch.cuda.synchronize()
print(f"Time taken: {start.elapsed_time(end)} ms")
print(tokenizer.decode(samples[0], skip_special_tokens=True))
