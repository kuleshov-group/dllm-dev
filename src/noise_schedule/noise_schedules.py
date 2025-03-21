from abc import ABC, abstractmethod

import torch


class Noise(ABC):
    """
    Baseline forward method to get noise parameters at a timestep
    """

    def __call__(
        self, t: torch.Tensor | float
    ) -> tuple[torch.Tensor | float, torch.Tensor | float]:
        # Assume time goes from 0 to 1
        pass

    @abstractmethod
    def inverse(self, alpha_t: torch.Tensor) -> torch.Tensor:
        """
        Inverse function to compute the timestep t from the noise schedule param.
        """
        raise NotImplementedError("Inverse function not implemented")


class CosineNoise(Noise):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps
        self.name = "cosine"

    def __call__(self, t):
        cos = -(1 - self.eps) * torch.cos(t * torch.pi / 2)
        sin = -(1 - self.eps) * torch.sin(t * torch.pi / 2)
        move_chance = cos + 1
        alpha_t_prime = sin * torch.pi / 2
        return 1 - move_chance, alpha_t_prime


class ExpNoise(Noise):
    def __init__(self, exp=2, eps=1e-3):
        super().__init__()
        self.eps = eps
        self.exp = exp
        self.name = f"exp_{exp}"

    def __call__(self, t):
        move_chance = torch.pow(t, self.exp)
        move_chance = torch.clamp(move_chance, min=self.eps)
        alpha_t_prime = -(self.exp * torch.pow(t, self.exp - 1))
        return alpha_t_prime, 1 - move_chance


class LogarithmicNoise(Noise):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps
        self.name = "logarithmic"

    def __call__(self, t):
        move_chance = torch.log1p(t) / torch.log(torch.tensor(2.0))
        alpha_t_prime = -1 / (torch.log(torch.tensor(2.0)) * (1 + t))
        return 1 - move_chance, alpha_t_prime


class LinearNoise(Noise):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.name = "linear"

    def inverse(self, alpha_t):
        return 1 - alpha_t

    def __call__(self, t):
        alpha_t_prime = -1
        move_chance = t
        return 1 - move_chance, alpha_t_prime


class StaggeredNoise(Noise):
    def __init__(self, config, eps=1e-3):
        super().__init__()
        self.eps = eps
        self.init_schedule(config.scale_conf, config.n)

    def init_schedule(self, scale_conf, n, num_blocks=None):
        """Configure staggered schedule"""
        if num_blocks is None:
            num_blocks = 1
        n = n // num_blocks
        self.scale_conf = scale_conf
        self.scale = torch.ones(1, n).repeat(1, num_blocks) * scale_conf
        self.loc = -torch.linspace(0, 1 - 1 / scale_conf, n).flip(0)
        self.loc = self.loc[None, :].repeat(1, num_blocks)

        last_token_max_t = 1 / scale_conf
        self.max_lookahead = (-self.loc < last_token_max_t)[:, :n].sum().item()

    def inverse(self, alpha_t):
        move_chance = 1 - alpha_t
        scale, loc = self.scale.to(move_chance.device), self.loc.to(move_chance.device)
        t = (move_chance / scale) - loc
        return t

    def __call__(self, t):
        scale, loc = self.scale.to(t.device), self.loc.to(t.device)
        move_chance = (t + loc) * scale
        move_chance = move_chance.clamp(0, 1)

        # alpha_t_prime is zero outside of unmasking interval
        alpha_t_prime = (move_chance != 0) * (move_chance != 1) * scale
        # edge case: move_chance_t = 1 but move_chance_s < 1
        alpha_t_prime += (t == (1 / scale - loc)) * scale

        # TODO double check edge case when s == n
        return 1 - move_chance, alpha_t_prime
