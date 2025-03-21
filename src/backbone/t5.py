import math
from typing import Any

import einops
import hydra
import inspect
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing import Any, Dict, Union

try:
  from torch.nn.attention.flex_attention import BlockMask
except:
  BlockMask = None

class Encoder(nn.Module):
    def __init__(self, net: nn.Module, encoder_hidden_dim: int, decoder_hidden_dim: int):
        super().__init__()
        self.net = net
        self.postprocess = nn.Linear(encoder_hidden_dim, decoder_hidden_dim)

    def forward(self, input_ids: Tensor, noise: Tensor | None, 
                attention_mask: Union[Tensor, BlockMask], **kwargs: Any) -> Tensor:
        _, enc_hidden_states = self.net(
            input_ids, noise, attention_mask=attention_mask, output_hidden_states=True, **kwargs)
        x0 = enc_hidden_states[-1]
        x0 = self.postprocess(x0)
        return x0

class Decoder(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, input_ids: Tensor, noise: Tensor | None,
                attention_mask: Union[Tensor, BlockMask], **kwargs: Any) -> Tensor:
        x = self.net(
            input_ids, noise, attention_mask=attention_mask, skip_embedding=True, **kwargs)
        return x

class T5(nn.Module):
    def __init__(
        self,
        mask_token_id: int,
        encoder_hidden_dim: int,
        decoder_hidden_dim: int,
        encoder_net: nn.Module,
        decoder_net: nn.Module,
    ):
        super().__init__()
        self.mask_token_id = mask_token_id
        self.encoder = Encoder(encoder_net, encoder_hidden_dim, decoder_hidden_dim)
        self.decoder = Decoder(decoder_net)
        self.time_conditioned_modules = (
            "noise" in inspect.getfullargspec(self.encoder.net.forward).args
        )
        self.vocab_embed = self.encoder.net.get_input_embeddings()

    def forward(self, input_ids: Tensor, noise: Tensor,
                encoder_attention_mask: Union[Tensor, BlockMask],
                decoder_attention_mask: Union[Tensor, BlockMask], **kwargs: Any) -> Tensor:
        """Forward pass for DIT model.

        Args:
            input_ids: Input ids of shape (batch_size, sequence_length)
        """
        mask_ids = torch.full_like(input_ids, self.mask_token_id)
        m_enc = self.vocab_embed(mask_ids)
        if self.time_conditioned_modules:
            x0_enc = self.encoder(
                input_ids, noise, encoder_attention_mask, **kwargs)
            decoder_input = torch.cat((m_enc, x0_enc), dim=1)
            decoder_out = self.decoder(
                decoder_input, noise, decoder_attention_mask, **kwargs)
            x = decoder_out[:, :m_enc.shape[1]]  # only keep logits for masked tokens
        else:
            x0_enc = self.encoder(
                input_ids, encoder_attention_mask,**kwargs)
            decoder_input = torch.cat((m_enc, x0_enc), dim=-1)
            decoder_out = self.decoder(
                decoder_input, decoder_attention_mask, **kwargs)
            x = decoder_out[:, :m_enc.shape[1]]  # only keep logits for masked tokens
        return x