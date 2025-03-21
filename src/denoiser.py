import copy
import inspect
import sys
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, Union

import hydra.utils
import torch
from tqdm import tqdm
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

try:
  from torch.nn.attention.flex_attention import flex_attention, BlockMask, create_block_mask
except:
  flex_attention, BlockMask = None, None

# Add the local directory (enables hydra.utils.instantiate for local imports)
sys.path.append(str(Path(__file__).resolve().parent))

# noinspection PyUnresolvedReferences
# Local imports not used, but added here so that HF push_to_hub adds them to model repo
# noinspection PyUnresolvedReferences
from src.backbone.dit import DIT  # noqa: F401
from src.noise_schedule.noise_schedules import (  # noqa: F401
    CosineNoise,
    ExpNoise,
    LogarithmicNoise,
    LinearNoise,
    StaggeredNoise,
)


@dataclass
class DenoiserInput(OrderedDict):
    """Input to the denoiser model."""

    xt: torch.Tensor  # (B, L) Tensor of token_ids
    x0: Optional[torch.Tensor] = None  # (B, L) Tensor of token_ids (not used in gen.)
    attention_mask: Optional[torch.Tensor] = None  # (B, L)
    t: Optional[torch.Tensor] = None  # (B,)
    move_chance: Optional[torch.Tensor] = None  # (B,) | (B, 1) | (B, 1, 1)
    loss_scaling: Optional[torch.Tensor] = None  # (B,) | (B, 1) | (B, 1, 1)
    # Placeholder in case future experiments require different inputs
    kwargs: dict[str, Any] | None = None


@dataclass
class LossAndNllOutput(OrderedDict):
    """Loss output for denoiser models."""

    loss: torch.Tensor
    nlls: torch.Tensor


@dataclass
class DenoiserOutput(ModelOutput):
    """Output of the denoiser model."""

    model_output: torch.Tensor
    logits: torch.Tensor | None = None
    tokens_mask: torch.Tensor | None = None
    loss: torch.Tensor | None = None
    nlls: torch.Tensor | None = None
    # Placeholder in case future models produce different outputs
    output_kwargs: dict[str, Any] | None = None


class DenoiserConfig(PretrainedConfig):
    """Configuration class for Denoiser models.

    This class is used to initialize the model and contains all the necessary
    parameters for the model's architecture.
    """

    model_type = "denoiser"

    def __init__(
        self,
        length: int | None = None,
        backbone_config: dict[str, Any] | None = None,
        noise_config: dict[str, Any] | None = None,
        tokenization_config: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        for v in [
            "vocab_size",
            "mask_token_id",
            "pad_token_id",
            "bos_token_id",
            "eos_token_id",
            "pad_vocab_size_multiple",
        ]:
            if tokenization_config is not None and (
                not hasattr(self, v) or hasattr(tokenization_config, v)
            ):
                setattr(self, v, tokenization_config.get(v, None))
            else:
                setattr(self, v, None)
        self.backbone_config = backbone_config
        self.noise_config = noise_config
        self.length = length


class Denoiser(ABC, PreTrainedModel):
    """Abstract base class for denoising models.

    This class defines the interface for AR, Diffusion, and Flow-based parametrizations.
    """

    config_class = DenoiserConfig

    def __init__(
        self,
        config: DenoiserConfig,
    ):
        """
        Initialize the Denoiser with a configuration and optional dataset type.

        Parameters:
            config (Any): Configuration object for the model.
        """
        super().__init__(config)
        self.config = config
        self.vocab_size = config.vocab_size
        self.mask_token_id = config.mask_token_id
        self.pad_token_id = config.pad_token_id
        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id
        self.backbone = hydra.utils.instantiate(config.backbone_config)
        self.noise_schedule = (
            hydra.utils.instantiate(config.noise_config)
            if config.noise_config is not None
            else None
        )
        self.time_conditioned_backbone = (
            "noise" in inspect.getfullargspec(self.backbone.forward).args
        )

    @abstractmethod
    def _prepare_inputs(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor | None = None,
        t: torch.FloatTensor | None = None,
    ) -> DenoiserInput:
        """
        Prepare inputs for the model.

        Parameters:
            input_ids (torch.Tensor): Input tensor to the model.
            attention_mask (Optional[torch.Tensor]): Attention mask for the model.
            t (Optional[torch.Tensor]): Time step for the model.

        Returns:
            Denoiser inputs.
        """
        raise NotImplementedError("Denoiser subclasses must implement _prepare_inputs")

    @abstractmethod
    def _compute_loss(
        self, model_output: torch.Tensor, denoiser_inputs: DenoiserInput, **kwargs: Any
    ) -> LossAndNllOutput:
        """
        Compute the loss for the denoising model.

        Parameters:
            model_output (torch.Tensor): Output tensor from self.forward.
            denoiser_inputs (DenoiserInput): Inputs passed to the denoiser model.

        Returns:
            LossAndNllOutput: loss (torch.FloatTensor) and nlls (torch.FloatTensor).
        """
        raise NotImplementedError("Denoiser subclasses must implement _compute_loss")

    def _forward(
        self,
        backbone_output: torch.Tensor,
        denoiser_inputs: DenoiserInput,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Forward pass for the denoiser model returns probabilities over denoised
        sequence.

        Some classes may need to override this method.

        Parameters:
            backbone_output (torch.Tensor): Output tensor from the backbone model.
            denoiser_inputs (DenoiserInput): Inputs passed to the denoiser model.

        Returns:
            Model outputs (torch.Tensor).
        """
        return torch.log_softmax(backbone_output, dim=-1)

    def _backbone_forward(self, denoiser_inputs: DenoiserInput, **kwargs: Any):
        """Forward pass for the backbone model (should return logits).

        Some classes may need to override this method.

        Parameters:
            denoiser_inputs (DenoiserInput): Inputs passed to the denoiser model.

        Returns:
            Backbone output (torch.Tensor).
        """
        if self.time_conditioned_backbone:
            return self.backbone(
                denoiser_inputs.xt,
                attention_mask=denoiser_inputs.attention_mask,
                noise=(1-denoiser_inputs.move_chance),
            )
        return self.backbone(
            denoiser_inputs.xt, attention_mask=denoiser_inputs.attention_mask, **kwargs
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor | None = None,
        compute_loss: bool | None = True,
        **kwargs,
    ) -> DenoiserOutput:
        """
        Perform a forward pass through the denoising model and
        (optionally) compute the loss.

        Parameters:
            input_ids (torch.Tensor): Input tensor to the model.
            attention_mask (Optional[torch.Tensor]): Attention mask for the model.
            compute_loss (Optional[bool]): Flag to compute loss.

        Returns:
            DenoiserOutput
        """
        t = kwargs.pop("t", None)
        denoiser_inputs = self._prepare_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            t=t,
        )

        backbone_output = self._backbone_forward(denoiser_inputs, **kwargs)
        if isinstance(backbone_output, ModelOutput) and hasattr(
            backbone_output, "logits"
        ):
            backbone_output = backbone_output.logits
        model_output = self._forward(
            backbone_output,
            denoiser_inputs,
            **kwargs,
        )

        if compute_loss:
            loss_and_nll = self._compute_loss(
                model_output=model_output, denoiser_inputs=denoiser_inputs, **kwargs
            )
            loss = loss_and_nll.loss
            nlls = loss_and_nll.nlls
        else:
            loss, nlls = None, None
        return DenoiserOutput(
            model_output=model_output,
            logits=backbone_output,
            tokens_mask=denoiser_inputs.attention_mask,
            loss=loss,
            nlls=nlls,
        )

    @staticmethod
    def _sample_categorical(categorical_probs):
        """Helper function to sample from a categorical distribution."""
        categorical_probs = categorical_probs.to(torch.float64)
        gumbel_norm = (1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()).to(
            categorical_probs.dtype
        )
        return (categorical_probs / gumbel_norm).argmax(dim=-1)

    @abstractmethod
    def generate_samples(  # TODO: clean up signature and docstring
        self,
        batch_size: int,
        max_seq_len: int,
        num_steps: int,
        nucleus_p: float = 1.0,
        eps: float = 1e-5,
        device: str | None = None,
        disable_cache: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generates sample from denoising model.
        # TODO: will need to enable infilling / starting from partially noised sequences

        Args:
            batch_size (int): Batch size.
            max_seq_len (int): Maximum sequence length.
            num_steps (int): Number of sampling steps.
            nucleus_p (float, optional): Nucleus sampling probability.
                Defaults to 1.0 (i.e., no nucleus sampling)
            eps (float, optional): Minimum value for t. Defaults to 1e-5.
            device (str | None, optional): Device to use for computation.
                Defaults to None, which will select cuda (if available).
            disable_cache (bool, optional): Whether to disable caching.
                Defaults to False.
        Returns:
            torch.Tensor: Generated samples of token_ids (B, L).
        """
        raise NotImplementedError


class ARConfig(DenoiserConfig):
    """Configuration class for autoregressive (AR) models."""

    model_type = "ar"
    auto_map = {
        "AutoConfig": "denoiser.ARConfig",
        "AutoModel": "denoiser.AR",
    }

    def __init__(
        self,
        length: int | None = None,
        backbone_config: dict[str, Any] | None = None,
        tokenization_config: dict[str, Any] | None = None,
        noise_config: None = None,
        **kwargs,
    ):
        super().__init__(
            length=length,
            backbone_config=backbone_config,
            noise_config=noise_config,
            tokenization_config=tokenization_config,
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
        self.time_conditioned_backbone = False

    def _prepare_inputs(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor | None = None,
        t: torch.FloatTensor | None = None,
    ) -> DenoiserInput:
        # Prepare inputs for autoregressive model
        labels = copy.deepcopy(input_ids[..., 1:])[..., None]
        input_ids = input_ids[..., :-1]
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.float)
        elif attention_mask.shape != input_ids.shape:
            attention_mask = attention_mask[..., :-1]

        return DenoiserInput(
            xt=input_ids,
            x0=labels,
            attention_mask=attention_mask,
        )

    def _compute_loss(
        self, model_output: torch.Tensor, denoiser_inputs: DenoiserInput, **kwargs: Any
    ) -> LossAndNllOutput:
        # Shift labels
        loss = -torch.gather(model_output, -1, denoiser_inputs.x0).squeeze(-1)

        nlls = loss * denoiser_inputs.attention_mask
        count = denoiser_inputs.attention_mask.sum()

        batch_nll = nlls.sum()
        token_nll = batch_nll / count

        return LossAndNllOutput(loss=token_nll, nlls=nlls)

    def generate_samples(
        self,
        batch_size: int,
        max_seq_len: int,
        num_steps: int,
        nucleus_p: float = 1.0,
        eps: float = 1e-5,
        device: str | None = None,
        disable_cache: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        # TODO
        raise NotImplementedError


class D3PMConfig(DenoiserConfig):
    """Configuration class for D3PM models."""

    model_type = "d3pm"
    auto_map = {
        "AutoConfig": "denoiser.D3PMConfig",
        "AutoModel": "denoiser.D3PM",
    }

    def __init__(
        self,
        length: int | None = None,
        backbone_config: dict[str, Any] | None = None,
        noise_config: dict[str, Any] | None = None,
        tokenization_config: dict[str, Any] | None = None,
        T: int = 1000,
        diffusion_type: Literal["absorbing", "uniform"] = "absorbing",
        **kwargs,
    ):
        super().__init__(
            length, backbone_config, noise_config, tokenization_config, **kwargs
        )
        self.diffusion_type = diffusion_type
        self.T = T


class D3PM(Denoiser):
    """Denoiser class for D3PM models.

    This class implements the Denoiser interface for D3PM models.
    """

    config_class = D3PMConfig

    def __init__(self, config: D3PMConfig):
        super().__init__(config)
        self.T = config.T
        self.diffusion_type = config.diffusion_type

    def _sample_q_xt(self, x0: torch.Tensor, alpha_t: torch.Tensor) -> torch.Tensor:
        move_indices = torch.rand(*x0.shape, device=x0.device) < (1.0 - alpha_t)
        if self.diffusion_type == "absorbing":
            return torch.where(move_indices, self.mask_token_id, x0)
        if self.diffusion_type == "uniform":
            uniform_tensor = torch.randint(
                0, self.vocab_size, x0.shape, device=x0.device
            )
            return torch.where(move_indices, uniform_tensor, x0)
        raise NotImplementedError(
            f"Diffusion type '{self.diffusion_type}' not implemented."
        )

    def _prepare_inputs(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor | None = None,
        t: torch.FloatTensor | None = None,
    ):
        # Prepare inputs for D3PM model
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.float)

        if t is None:
            t = torch.rand(input_ids.shape[0], device=input_ids.device)
        loss_scaling, move_chance = self.noise_schedule(t)
        if move_chance.ndim == 1:
            move_chance = move_chance[..., None]
            loss_scaling = loss_scaling[..., None]
        xt = self._sample_q_xt(input_ids, move_chance)

        return DenoiserInput(
            xt=xt,
            x0=input_ids,
            attention_mask=attention_mask,
            t=t,
            move_chance=move_chance,
            loss_scaling=loss_scaling,
        )

    def _compute_loss(
        self, model_output: torch.Tensor, denoiser_inputs: DenoiserInput, **kwargs: Any
    ) -> LossAndNllOutput:
        raise NotImplementedError

    def _sample_prior(self, device, batch_size, length):
        """Samples from prior / limiting distribution."""
        if self.diffusion_type == "absorbing_state":
            return self.mask_token_id * torch.ones(
                (batch_size, length), dtype=torch.int64, device=device
            )
        if self.diffusion_type == "uniform":
            return torch.randint(
                0,
                self.vocab_size,
                (batch_size, length),
                device=device,
                dtype=torch.int64,
            )
        raise NotImplementedError(
            f"Diffusion type '{self.diffusion_type}' not implemented."
        )

    def _compute_posterior(
        self,
        x: torch.Tensor,
        xt: torch.Tensor,
        move_chance_t: torch.Tensor,
        move_chance_s: torch.Tensor,
    ) -> torch.Tensor:
        """Computes posterior / approximate posterior q(x_s | x_t, x),
            where x represents clean sequence (as one-hots) or the output of the
            denoising model.

        Args:
            x (torch.Tensor): True (one-hot) / predicted clean signal (B, L, V).
            xt (torch.Tensor): Noised signal at time t (B, L).
            move_chance_t (torch.Tensor): Noise schedule parameter at time t (B, 1, 1).
            move_chance_s (torch.Tensor): Noise schedule parameter at time s (B, 1, 1).
        """
        alpha_t, alpha_s = 1 - move_chance_t, 1 - move_chance_s
        if self.diffusion_type == "absorbing_state":
            q_xs = x * (alpha_s - alpha_t)
            q_xs[..., self.mask_token_id] = 1 - alpha_s[..., 0]
            q_xs /= 1 - alpha_t
            return q_xs

        alpha_ts = alpha_t / alpha_s
        d_alpha = alpha_s - alpha_t
        xt_one_hot = torch.nn.functional.one_hot(x, self.vocab_size)
        limiting_distribution = torch.ones_like(xt_one_hot) / self.vocab_size
        if self.diffusion_type == "uniform":
            return (
                alpha_t * self.vocab_size * x * xt_one_hot
                + (alpha_ts - alpha_t) * xt_one_hot
                + d_alpha * x
                + (1 - alpha_ts) * (1 - alpha_s) * limiting_distribution
            ) / (
                alpha_t * self.vocab_size * torch.gather(x, -1, xt[..., None])
                + (1 - alpha_t)
            )
        raise NotImplementedError(
            f"Diffusion type {self.diffusion_type} not implemented."
        )

    def _generate_unconditional(  # TODO add CBG and CFG generation
        self,
        move_chance_t: torch.Tensor,
        move_chance_s: torch.Tensor,
        nucleus_p: float,
        denoiser_inputs: DenoiserInput | None = None,
        cache: Dict[str, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if cache is None:
            backbone_output = self._backbone_forward(denoiser_inputs)
            if isinstance(backbone_output, ModelOutput) and hasattr(
                backbone_output, "logits"
            ):
                backbone_output = backbone_output.logits
            log_x_theta = self._forward(
                backbone_output,
                denoiser_inputs,
            )  # should be the log(x_\theta) with the shape of (B, Seq, Vocab)
            x_theta = log_x_theta.exp()
            if nucleus_p < 1:
                sorted_probs, sorted_indices = torch.sort(
                    x_theta, descending=True, dim=-1
                )
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                top_p_mask = cumulative_probs <= nucleus_p
                top_p_mask[..., 0] = True
                nucleus_probs = sorted_probs * top_p_mask
                nucleus_probs /= nucleus_probs.sum(dim=-1, keepdim=True)
                x_theta = torch.zeros_like(x_theta).scatter_(
                    -1, sorted_indices, nucleus_probs
                )
        else:
            x_theta = cache["x_theta"]
        q_xs = self._compute_posterior(x_theta, move_chance_t, move_chance_s)
        cache = {"x_theta": x_theta}
        return q_xs, cache

    def generate_samples(
        self,
        batch_size: int,
        length: int,
        num_steps: int,
        nucleus_p: float = 1.0,
        eps: float = 1e-5,
        device: str | None = None,
        disable_cache: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        device = (
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        xt = self._sample_prior(device, batch_size, length)
        timesteps = torch.linspace(1, eps, num_steps + 1, device=device)
        dt = (1 - eps) / num_steps
        pbar = tqdm(range(num_steps), desc="Sampling", leave=False)
        NFEs = 0
        cache = None
        for i in pbar:
            t = timesteps[i]
            if self.T > 0:
                t = (t * self.T).to(torch.int)
                t = t / self.T
                t += 1 / self.T
            if cache is None:
                NFEs += 1
            # t is 0-dim tensor, reshape to (1, 1, 1) for broadcasting
            move_chance_t, _ = self.noise_schedule(t)[None, None, None]
            move_chance_s, _ = self.noise_schedule(t - dt)[None, None, None]
            # prepare backbone inputs
            attention_mask = (
                torch.ones_like(xt, dtype=torch.float) if cache is None else None
            )
            denoiser_inputs = DenoiserInput(
                xt=xt, attention_mask=attention_mask, move_chance=move_chance_t
            )
            q_xs, cache = self._generate_unconditional(
                move_chance_t=move_chance_t,
                move_chance_s=move_chance_s,
                nucleus_p=nucleus_p,
                denoiser_inputs=denoiser_inputs,
                cache=cache,
            )
            xs = self._sample_categorical(q_xs)
            pbar.set_postfix(
                NFEs=NFEs,
                prob_check=(q_xs.sum() / xt.numel()).item(),
                nan_check=bool(q_xs.isnan().sum() > 0),
            )
            if not torch.allclose(xs, xt) or not disable_cache:
                cache = None
            xt = xs
        return xt


class MDLMConfig(D3PMConfig):
    """Configuration class for MDLM models."""

    model_type = "mdlm"
    auto_map = {
        "AutoConfig": "denoiser.MDLMConfig",
        "AutoModel": "denoiser.MDLM",
    }


class MDLM(D3PM):
    """Denoiser class for MDLM models."""

    config_class = MDLMConfig

    def __init__(self, config: MDLMConfig):
        super().__init__(config)
        self.neg_infinity = -1e12

    def _forward(
        self, backbone_output: torch.Tensor, denoiser_inputs: DenoiserInput, **kwargs
    ) -> torch.Tensor:
        # Zero-mask probability
        mask = (
            torch.arange(backbone_output.shape[-1], device=backbone_output.device)
            == self.mask_token_id
        ).view(1, 1, -1)  # unsqueeze for broadcast to (batch, seq_len, vocab_size)
        log_probs = torch.where(
            mask, backbone_output + self.neg_infinity, backbone_output
        )
        log_probs = log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)

        # Copy-over unmasked: For the log_probs of the unmasked tokens, set all values
        # to -infinity except for the indices corresponding to
        # the unmasked tokens.
        unmasked_indices = denoiser_inputs.xt != self.mask_token_id
        log_probs[unmasked_indices] = self.neg_infinity
        log_probs[unmasked_indices, denoiser_inputs.xt[unmasked_indices]] = 0
        return log_probs

    def _compute_loss(
        self, model_output: torch.Tensor, denoiser_inputs: DenoiserInput, **kwargs: Any
    ) -> LossAndNllOutput:
        log_p_theta = torch.gather(
            input=model_output, dim=-1, index=denoiser_inputs.x0[:, :, None]
        ).squeeze(-1)

        loss = -log_p_theta * denoiser_inputs.loss_scaling

        nlls = loss * denoiser_inputs.attention_mask
        count = denoiser_inputs.attention_mask.sum()

        batch_nll = nlls.sum()
        token_nll = batch_nll / count
        return LossAndNllOutput(loss=token_nll, nlls=nlls)



class AnyOrderMDLMConfig(MDLMConfig):
    """Configuration class for AnyOrder MDLM models."""

    model_type = "any_order_mdlm"
    auto_map = {
        "AutoConfig": "denoiser.AnyOrderMDLMConfig",
        "AutoModel": "denoiser.AnyOrderMDLM",
    }

    def __init__(
        self,
        length: int | None = None,
        backbone_config: dict[str, Any] | None = None,
        noise_config: dict[str, Any] | None = None,
        tokenization_config: dict[str, Any] | None = None,
        T: int = 1000,
        diffusion_type: str = "absorbing",  # absorbing, uniform, ...
        attn_backend: str = "sdpa", # sdpa, flex
        **kwargs,
    ):
        super().__init__(
            length, backbone_config, noise_config, tokenization_config, T, diffusion_type, **kwargs
        )
        self.attn_backend = attn_backend


class AnyOrderMDLM(MDLM):
    def __init__(self,config: AnyOrderMDLMConfig):
        super().__init__(config)
        mask = self._gen_mask()
        self.register_buffer("mask", mask)

        self.time_conditioned_backbone = (
            "noise" in inspect.getfullargspec(self.backbone.forward).args
        )
        self._validate_configuration()
        
    def _validate_configuration(self):
        assert self.noise_schedule.name in {'linear', 'staggered'}
        if self.config.attn_backend == 'flex' and flex_attention is None:
            raise ValueError("FlexAttention not available. Please install the latest version of PyTorch.")

    def _mask_fn(
            b: int,
            h: int,
            q_idx: torch.Tensor,
            kv_idx: torch.Tensor,
            n: int | None = None,
        ) -> Union[torch.Tensor, BlockMask]:
        """
        Constructs the encoder and decoder attention masks.
        
        Args:
            b, h: Batch and head indices (ignored for mask logic).
            q_idx, kv_idx: Query and Key indices.
            seq_len: Total sequence length.
        Returns:
            A boolean attention mask.
        """

        encoder_flag_q = (q_idx < n)
        encoder_flag_kv = (kv_idx < n)

        # Compute block indices
        idx_q = torch.where(encoder_flag_q == 1, (q_idx - n), q_idx)
        idx_kv = torch.where(encoder_flag_kv == 1, (kv_idx - n), kv_idx)

        # **1. Encoder Self-Attention **
        x0_self = (
            (idx_q >= idx_kv)
            & (encoder_flag_kv == 1) & (encoder_flag_kv == 1)
        )

        # **2. Decoder Self-Attention **
        xt_self = (
            (idx_q == idx_kv)
            & (encoder_flag_kv == 0) & (encoder_flag_kv == 0)
        )

        # **3. Decoder Cross-Attention **
        xt_cross = (
            (idx_q > idx_kv)
            & (encoder_flag_kv == 1) & (encoder_flag_kv == 0)
        )

        # **4. Combine Masks **
        return xt_self | xt_cross | x0_self
    

    def _gen_mask(self):
        if self.config.attn_backend == "flex":
            mask = create_block_mask(
                partial(self._mask_fn, n=self.config.length),
                B=None, H=None, Q_LEN=self.config.length*2, KV_LEN=self.config.length*2)
        else:
            mask = self._mask_fn(
                b=None, h=None, q_idx=torch.arange(self.config.length)[:, None], 
                kv_idx=torch.arange(self.config.length)[None, :], n=self.config.length)
        return mask
    
    def _sample_permutation(self, batch_dim: int) -> torch.Tensor:
        noise = torch.rand((batch_dim, self.config.length))
        ts = self.noise_schedule.inverse(noise)
        perms = (-ts).argsort(-1)  # sort by unmasking timestep (high to low)
        return perms
    
    def _permute_attention_mask(self, batch_dim: int, 
                                attention_mask: torch.Tensor) -> torch.Tensor:
        """Permute encoder AND decoder attention masks for any-order training."""
        permutation = self._sample_permutation(batch_dim).to(attention_mask.device)
        permutation = torch.cat((permutation, permutation + self.config.length + 1), dim=-1)
        permuted_attention_mask = attention_mask.gather(0, permutation)
        permuted_attention_mask = permuted_attention_mask.gather(1, permutation)
        return attention_mask
    
    def _prepare_inputs(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor | None = None,
        t: torch.FloatTensor | None = None,
    ):
        # Prepare inputs for AO-MDLM model
        attention_mask = self.mask

        if t is None:
            t = torch.rand(input_ids.shape[0], device=input_ids.device)
        loss_scaling, move_chance = self.noise_schedule(t)
        if move_chance.ndim == 1:
            move_chance = move_chance[..., None]
            loss_scaling = loss_scaling[..., None]
        xt = self._sample_q_xt(input_ids, move_chance)

        permuted_attention_mask = self._permute_attention_mask(
            input_ids.shape[0], attention_mask)
        encoder_attention_mask = permuted_attention_mask[:, :self.config.length]
        decoder_attention_mask = permuted_attention_mask[:, self.config.length:]

        return DenoiserInput(
            xt=xt,
            x0=input_ids,
            attention_mask=attention_mask,
            t=t,
            move_chance=move_chance,
            loss_scaling=loss_scaling,
            kwargs={
                "encoder_attention_mask": encoder_attention_mask,
                "decoder_attention_mask": decoder_attention_mask,
            },
        )
    
class BD3LMConfig(D3PMConfig):
    """Configuration class for BD3LM models."""
    model_type = "bd3lm"
    auto_map = {
        "AutoConfig": "denoiser.BD3LMConfig",
        "AutoModel": "denoiser.BD3LM",
    }

class BD3LM(MDLM):
    def __init__(self, config: MDLMConfig):
        super().__init__(config)
        self.block_size = config.block_size
        self.attn_backend = config.backbone_config.attn_backend
        if flex_attention is not None:
            raise ValueError("FlexAttention not available. Please install the latest version of PyTorch.")
        self._gen_mask()

    def _mask_fn(
            b: int,
            h: int,
            q_idx: torch.Tensor,
            kv_idx: torch.Tensor,
            block_size: Union[int, None] = None,
            n: int | None = None,
        ) -> Union[torch.Tensor, BlockMask]:
        """
        Constructs the specialized block diffusion attention mask for training
        composed of three masks:
        - **Block Diagonal Mask (M_BD)**: Self-attention within noised blocks
        - **Offset Block Causal Mask (M_OBC)**: Cross-attention for conditional context
        - **Block Causal Mask (M_BC)**: Attention to update x0

        Args:
            b, h: Batch and head indices (ignored for mask logic).
            q_idx, kv_idx: Query and Key indices.
            seq_len: Total sequence length.
            block_size: Defines the block structure.

        Returns:
            A boolean attention mask.
        """

        # Indicate whether token belongs to xt or x0
        x0_flag_q = (q_idx >= n)
        x0_flag_kv = (kv_idx >= n)

        # Compute block indices
        block_q = torch.where(x0_flag_q == 1,
                                (q_idx - n) // block_size,
                                q_idx // block_size)
        block_kv = torch.where(x0_flag_kv == 1,
                                (kv_idx - n) // block_size,
                                kv_idx // block_size)

        # **1. Block Diagonal Mask (M_BD) **
        block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)

        # **2. Offset Block-Causal Mask (M_OBC) **
        offset_block_causal = (
            (block_q > block_kv)
            & (x0_flag_kv == 1)
            & (x0_flag_q == 0)
        )

        # **3. Block-Causal Mask (M_BC) **
        block_causal = (block_q >= block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 1)

        # **4. Combine Masks **
        return block_diagonal | offset_block_causal | block_causal
    
    def _gen_mask(self):
        if self.attn_backend == "flex":
            self.mask = create_block_mask(
                partial(self._mask_fn, block_size=self.block_size, n=self.length),
                B=None, H=None, Q_LEN=self.length*2, KV_LEN=self.length*2)
        else:
            self.mask = self._mask_fn(
                b=None, h=None, q_idx=torch.arange(self.length)[:, None], 
                kv_idx=torch.arange(self.length)[None, :],
                block_size=self.block_size, n=self.length)

    def _prepare_inputs(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor | None = None,
        t: torch.FloatTensor | None = None,
    ) -> DenoiserInput:
        # Prepare inputs for BD3-LM
        labels = copy.deepcopy(input_ids[..., 1:])[..., None]
        input_ids = input_ids[..., :-1]
        if attention_mask is None:
            attention_mask = self.mask
       
        return DenoiserInput(
            xt=input_ids,
            x0=labels,
            attention_mask=attention_mask,
        )


    def _backbone_forward(self, denoiser_inputs: DenoiserInput, **kwargs: Any):
        """Forward pass for the backbone model (should return logits).

        Some classes may need to override this method.

        Parameters:
            denoiser_inputs (DenoiserInput): Inputs passed to the denoiser model.

        Returns:
            Backbone output (torch.Tensor).
        """
        x_full = torch.cat((denoiser_inputs.xt, denoiser_inputs.x0), dim=1)

        if self.time_conditioned_backbone:
            out = self.backbone(
                x_full,
                attention_mask=denoiser_inputs.attention_mask,
                noise=(1 - denoiser_inputs.move_chance),
            )
        out = self.backbone(
            x_full, attention_mask=denoiser_inputs.attention_mask, **kwargs
        )
        return out[:, :denoiser_inputs.xt.size(1)]

    def _generate_unconditional(
        self,
        move_chance_t: torch.Tensor,
        move_chance_s: torch.Tensor,
        nucleus_p: float,
        denoiser_inputs: DenoiserInput | None = None,
        cache: Dict[str, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError
    
# TODO
# class UDLM(D3PM):


# TODO
# class SEDD(Denoiser):


# TODO
# class DFM(Denoiser):
