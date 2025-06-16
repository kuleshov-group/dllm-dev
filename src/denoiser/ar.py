import copy
from typing import Any

import torch
from transformers import (
    GenerationConfig,
    LogitsProcessorList,
    PreTrainedTokenizer,
    StoppingCriteriaList,
)
from transformers.cache_utils import Cache
from transformers.generation.utils import GenerateOutput

from src.denoiser.base import (
    Denoiser,
    DenoiserConfig,
    DenoiserInput,
    LossAndNllOutput,
)


def preprocess_attention_mask(attention_mask, dtype):
    min_dtype = torch.finfo(dtype).min
    attention_mask = torch.where(
        (attention_mask == 0.0).bool(),  # type: ignore
        min_dtype,
        0.0,
    ).to(dtype)
    return attention_mask


class ARConfig(DenoiserConfig):
    """Configuration class for autoregressive (AR) models."""

    model_type = "ar"
    auto_map = {
        "AutoConfig": "ar.ARConfig",
        "AutoModel": "ar.AR",
        "AutoModelForCausalLM": "ar.AR",
    }

    def __init__(
        self,
        length: int | None = None,
        backbone_config: dict[str, Any] | None = None,
        tokenization_config: dict[str, Any] | None = None,
        noise_config: None = None,
        time_conditioned_backbone: bool | None = None,
        **kwargs,
    ):
        super().__init__(
            length=length,
            backbone_config=backbone_config,
            noise_config=noise_config,
            tokenization_config=tokenization_config,
            time_conditioned_backbone=time_conditioned_backbone,
            **kwargs,
        )


class AR(Denoiser):
    """Denoiser class for autoregressive (AR) models."""

    config_class = ARConfig

    def __init__(
        self,
        config: ARConfig,
    ):
        super().__init__(config)
        self.block_size = 1
        static_mask = self._decoder_block_mask(
            b=None,
            h=None,
            q_idx=torch.arange(self.config.length - self.block_size)[:, None],
            kv_idx=torch.arange(self.config.length - self.block_size)[None, :],
            block_size=self.block_size,
            seq_length=self.config.length - self.block_size,
            mask_decoder=False,
        )
        self.register_buffer(
            "static_attention_mask",
            static_mask,
        )

    # noinspection PyUnusedLocal
    @staticmethod
    def _decoder_block_mask(
        b,
        h,
        q_idx,
        kv_idx,
        block_size: int | None = None,
        seq_length: int | None = None,
        mask_decoder: bool = False,
    ) -> torch.Tensor:
        del b, h

        # Compute block indices
        block_q = q_idx // block_size
        block_kv = (kv_idx % seq_length) // block_size

        # **1. Offset Block-Causal Mask (M_OBC) **
        return block_q >= block_kv

    def _prepare_inputs(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor | None = None,
        context_mask: torch.FloatTensor | None = None,
        t: torch.FloatTensor | None = None,
        past_key_values: Cache | None = None,
    ) -> DenoiserInput:
        # Prepare inputs for autoregressive model
        labels = copy.deepcopy(input_ids[..., self.block_size :])[..., None]
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if context_mask is None:
            context_mask = torch.zeros_like(attention_mask)
        attention_mask = attention_mask[..., : -self.block_size]
        context_mask = context_mask[..., : -self.block_size]
        input_ids = input_ids[..., : -self.block_size]
        decoder_attention_mask = (
            self.static_attention_mask[None, ...]
            & attention_mask[:, None, :]
            & attention_mask[..., None]
        )
        decoder_attention_mask = preprocess_attention_mask(
            decoder_attention_mask[:, None], dtype=torch.float
        )
        return DenoiserInput(
            xt=input_ids,  # type: ignore
            x0=labels,  # type: ignore
            attention_mask=decoder_attention_mask,  # type: ignore
            context_mask=context_mask,
            tokens_mask=attention_mask * (1 - context_mask),
            past_key_values=past_key_values,
        )

    def _compute_loss(
        self,
        model_output: torch.FloatTensor,
        denoiser_inputs: DenoiserInput,
        **kwargs: Any,
    ) -> LossAndNllOutput:
        # Shift labels
        loss = -torch.gather(model_output, -1, denoiser_inputs.x0).squeeze(-1)

        nlls = loss * denoiser_inputs.tokens_mask
        count = denoiser_inputs.tokens_mask.sum()

        batch_nll = nlls.sum()
        token_nll = batch_nll / count

        return LossAndNllOutput(loss=token_nll, nlls=nlls)  # type: ignore

    @torch.no_grad()
    def generate(
        self,
        inputs: torch.LongTensor | None = None,
        generation_config: GenerationConfig | None = None,
        logits_processor: LogitsProcessorList | None = None,
        stopping_criteria: StoppingCriteriaList | None = None,
        max_length: int | None = None,
        max_new_tokens: int | None = None,
        batch_size: int | None = None,
        device: str | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
        **kwargs,
    ) -> GenerateOutput | torch.LongTensor:
        outputs = self.backbone.model.generate(
            inputs=inputs,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )

        if tokenizer is not None:
            print(tokenizer.batch_decode(outputs))
        # Decode output
        return outputs
