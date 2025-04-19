from typing import Any, Union

import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
from transformers.utils import logging

try:
    from torch.nn.attention.flex_attention import BlockMask
except ModuleNotFoundError:
    BlockMask = None


logger = logging.get_logger(__name__)


class Decoder(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(
        self,
        input_ids: Tensor,
        noise: Tensor | None,
        attention_mask: Union[Tensor, BlockMask],
        **kwargs: Any,
    ) -> Tensor:
        x = self.net(
            input_ids,
            noise,
            attention_mask=attention_mask,
            skip_embedding=True,
            **kwargs,
        )
        return x


class LlamaAsEncoderDecoder(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        num_layers_for_encoder: int,
        decoder_net: nn.Module,
    ):
        super().__init__()
        llama = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=True
        )  # TODO: try something like this:, attn_implementation="flex")
        self.num_layers_for_encoder = num_layers_for_encoder
        self.encoder = llama.model.layers[:num_layers_for_encoder]
        self.norm = llama.model.norm
        self.rotary_emb = llama.model.rotary_emb
        self.vocab_embed = llama.model.embed_tokens
        self.lm_head = llama.lm_head
        decoder = Decoder(decoder_net)
        self.decoder = decoder.net.blocks

    def forward(
        self,
        decoder_input_ids: Tensor,
        encoder_input_ids: Tensor,
        encoder_attention_mask: Union[Tensor, BlockMask],
        decoder_attention_mask: Union[Tensor, BlockMask],
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tensor:
        # TODO: not sure how to use this yet...
        # if use_cache and past_key_values is None:
        #     past_key_values = DynamicCache()
        # if cache_position is None:
        #     past_seen_tokens = (
        #         past_key_values.get_seq_length() if past_key_values is not None else 0
        #     )
        #     cache_position = torch.arange(
        #         past_seen_tokens,
        #         past_seen_tokens + inputs_embeds.shape[1],
        #         device=inputs_embeds.device,
        #     )
        # if position_ids is None:
        #     position_ids = cache_position.unsqueeze(0)
        #
        # causal_mask = self.llama._update_causal_mask(
        #     attention_mask,
        #     inputs_embeds,
        #     cache_position,
        #     past_key_values,
        #     output_attentions,
        # )
        n = encoder_input_ids.shape[-1]

        # Encode clean tokens
        encoder_hidden_states = self.vocab_embed(encoder_input_ids)
        encoder_position_ids = torch.arange(
            encoder_input_ids.shape[-1], device=encoder_input_ids.device
        ).unsqueeze(0)
        encoder_position_embeddings = self.rotary_emb(
            encoder_hidden_states, encoder_position_ids
        )
        encoder_attention_mask = encoder_attention_mask.to(
            encoder_hidden_states.dtype
        ).unsqueeze(1)
        decoder_attention_mask = decoder_attention_mask.to(
            encoder_hidden_states.dtype
        ).unsqueeze(1)
        for encoder_layer in self.encoder:
            encoder_hidden_states = encoder_layer(
                encoder_hidden_states,
                encoder_position_ids=encoder_position_ids,
                past_key_value=past_key_values,
                attention_mask=encoder_attention_mask,
                output_attentions=False,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=encoder_position_embeddings,
            )[0]
        # Run decoder with xattn to clean tokens
        decoder_inputs_embeds = self.vocab_embed(decoder_input_ids)
        decoder_hidden_states = torch.cat(
            (decoder_inputs_embeds, encoder_hidden_states), dim=1
        )

        decoder_position_ids = torch.cat(
            (
                torch.arange(
                    decoder_inputs_embeds.shape[1], device=decoder_input_ids.device
                ),
                torch.arange(
                    encoder_hidden_states.shape[1], device=decoder_input_ids.device
                ),
            ),
            dim=-1,
        ).unsqueeze(0)
        decoder_position_embeddings = self.rotary_emb(
            decoder_hidden_states, decoder_position_ids
        )
        decoder_position_embeddings = (
            decoder_position_embeddings[0][:, :, None, None, :],
            decoder_position_embeddings[1][:, :, None, None, :],
        )
        for decoder_layer in self.decoder:
            decoder_hidden_states = decoder_layer(
                decoder_hidden_states,
                decoder_position_embeddings,
                attention_mask=decoder_attention_mask,
            )
            decoder_hidden_states[:, n:] = encoder_hidden_states

        # Only keep logits for masked tokens
        decoder_hidden_states = decoder_hidden_states[:, :n]
        decoder_hidden_states = self.norm(decoder_hidden_states)

        # Use the same LM head from LLaMA
        return self.lm_head(decoder_hidden_states)
