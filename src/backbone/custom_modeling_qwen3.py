from typing import Callable, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from transformers import AttentionInterface
from transformers.models.qwen3.modeling_qwen3 import (
    Cache,
    FlashAttentionKwargs,
    Qwen3Attention,
    Qwen3Config,
    Qwen3DecoderLayer,
    Qwen3ForCausalLM,
    Qwen3Model,
    eager_attention_forward,
    rotate_half,
)
from transformers.processing_utils import Unpack
from transformers.utils import logging

logger = logging.get_logger(__name__)


def custom_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1, q_start_idx=0):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos[..., q_start_idx:, :]) + (
        rotate_half(q) * sin[..., q_start_idx:, :]
    )
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


@torch.compile(fullgraph=True, mode="max-autotune-no-cudagraphs")
def compiled_flex_attention(
    query_states, key_states, value_states, attention_mask, **kwargs
):
    return flex_attention(
        query_states, key_states, value_states, block_mask=attention_mask, **kwargs
    )


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim) to
    (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def custom_flex_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Union[torch.Tensor, "BlockMask"],
    scaling: Optional[float] = None,
    softcap: Optional[float] = None,
    head_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    block_mask = None
    causal_mask = None
    if isinstance(attention_mask, BlockMask):
        block_mask = attention_mask
    else:
        causal_mask = attention_mask

    if causal_mask is not None:
        causal_mask = causal_mask[:, :, :, : key.shape[-2]]

    def score_mod(score, batch_idx, head_idx, q_idx, kv_idx):
        if softcap is not None:
            score = softcap * torch.tanh(score / softcap)
        if causal_mask is not None:
            score = score + causal_mask[batch_idx][0][q_idx][kv_idx]
        if head_mask is not None:
            score = score + head_mask[batch_idx][head_idx][0][0]
        return score

    enable_gqa = True
    num_local_query_heads = query.shape[1]

    # When running TP this helps:
    if not ((num_local_query_heads & (num_local_query_heads - 1)) == 0):
        key = repeat_kv(key, query.shape[1] // key.shape[1])
        value = repeat_kv(value, query.shape[1] // value.shape[1])
        enable_gqa = False

    kernel_options = kwargs.get("kernel_options", None)
    attn_output, attention_weights = compiled_flex_attention(
        query,
        key,
        value,
        score_mod=score_mod,
        block_mask=block_mask,
        enable_gqa=enable_gqa,
        scale=scaling,
        kernel_options=kernel_options,
        # Last time checked on PyTorch == 2.5.1: Flex Attention always computes the lse
        # regardless.
        # For simplification, we thus always return it as no additional computations are
        # introduced.
        return_lse=True,
    )
    # lse is returned in float32
    attention_weights = attention_weights.to(value.dtype)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attention_weights


class CustomAttentionInterface(AttentionInterface):
    def __init__(self):
        super().__init__()
        self._local_mapping.update(
            {"custom_flex_attention": custom_flex_attention_forward}
        )


CUSTOM_ALL_ATTENTION_FUNCTIONS: CustomAttentionInterface = CustomAttentionInterface()


class CustomQwen3Attention(Qwen3Attention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__(config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        q_start_idx: int = 0,  # > 0: decoder pass w/encoder inputs in hidden_states
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_input_shape = hidden_states[:, q_start_idx:, :].shape[:-1]
        query_hidden_shape = (*query_input_shape, -1, self.head_dim)

        query_states = self.q_norm(
            self.q_proj(hidden_states[:, q_start_idx:, ...]).view(query_hidden_shape)
        ).transpose(1, 2)
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = custom_apply_rotary_pos_emb(
            query_states, key_states, cos, sin, q_start_idx=q_start_idx
        )

        if past_key_value is not None:
            # sin and cos are specific to RoPE models
            # cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # NOTE: downcast for flex-attention compatibility
        # TODO: Maybe upcast qk to v dtype instead?
        query_states, key_states = (
            query_states.to(value_states.dtype),
            key_states.to(value_states.dtype),
        )

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get(
                "output_attentions", False
            ):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention`"
                    "does not support `output_attentions=True`."
                    " Falling back to eager attention."
                    "This warning can be removed using the argument "
                    '`attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = CUSTOM_ALL_ATTENTION_FUNCTIONS[
                    "custom_flex_attention"
                    # self.config._attn_implementation
                ]

        # TODO: Ensure we're using flash_attn even though len(q) != len(k)
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*query_input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class CustomQwen3DecoderLayer(Qwen3DecoderLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__(config, layer_idx=layer_idx)
        self.self_attn = CustomQwen3Attention(config=config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        q_start_idx: int = 0,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states[:, q_start_idx:, ...]

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            q_start_idx=q_start_idx,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class CustomQwen3Model(Qwen3Model):
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [
                CustomQwen3DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        # Initialize weights and apply final processing
        self.post_init()


class CustomQwen3ForCausalLM(Qwen3ForCausalLM):
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        # Initialize a new model with custom layers
        self.model = CustomQwen3Model(config)

        # Initialize weights and apply final processing
        self.post_init()
