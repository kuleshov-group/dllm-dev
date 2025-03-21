from abc import ABC, abstractmethod

import torch


class Noise(ABC):
    """
    Baseline forward method to get the total + rate of noise at a timestep
    """

    def __call__(
        self, t: torch.Tensor | float
    ) -> tuple[torch.Tensor | float, torch.Tensor | float]:
        # Assume time goes from 0 to 1
        return self.compute_loss_scaling_and_move_chance(t)

    @abstractmethod
    def compute_loss_scaling_and_move_chance(
      self, t: torch.Tensor | float) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the loss scaling and move chance at a given timestep t.
        """
        pass

    @abstractmethod
    def inverse(self, move_chance: torch.Tensor) -> torch.Tensor:
        """
        Inverse function to compute the timestep t from the move chance.
        """
        raise NotImplementedError("Inverse function not implemented")

class CosineNoise(Noise):
  def __init__(self, eps=1e-3):
    super().__init__()
    self.eps = eps
    self.name = 'cosine'

  def compute_loss_scaling_and_move_chance(self, t):
    cos = - (1 - self.eps) * torch.cos(t * torch.pi / 2)
    sin = - (1 - self.eps) * torch.sin(t * torch.pi / 2)
    move_chance = cos + 1
    loss_scaling = sin / (move_chance + self.eps) * torch.pi / 2
    return loss_scaling, move_chance

class ExpNoise(Noise):
  def __init__(self, exp=2, eps=1e-3):
    super().__init__()
    self.eps = eps
    self.exp = exp
    self.name = f'exp_{exp}'
  
  def compute_loss_scaling_and_move_chance(self, t):
    move_chance = torch.pow(t, self.exp)
    move_chance = torch.clamp(move_chance, min=self.eps)
    loss_scaling = - (self.exp * torch.pow(t, self.exp-1)) / move_chance
    return loss_scaling, move_chance

class LogarithmicNoise(Noise):
  def __init__(self, eps=1e-3):
    super().__init__()
    self.eps = eps
    self.name = 'logarithmic'

  def compute_loss_scaling_and_move_chance(self, t):
    move_chance = torch.log1p(t) / torch.log(torch.tensor(2.0))
    loss_scaling = - 1 / (move_chance * torch.log(torch.tensor(2.0)) * (1 + t))
    return loss_scaling, move_chance

class LinearNoise(Noise):
  def __init__(self, eps=1e-3):
    super().__init__()
    self.name = 'linear'

  def inverse(self, move_chance):
    return move_chance
    
  def compute_loss_scaling_and_move_chance(self, t):
    loss_scaling = 1 / t
    return loss_scaling, t
  

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
    self.loc =  - torch.linspace(0, 1 - 1 / scale_conf, n).flip(0)
    self.loc = self.loc[None, :].repeat(1, num_blocks)

    last_token_max_t = 1 / scale_conf
    self.max_lookahead = (-self.loc < last_token_max_t)[:, :n].sum().item()

  def inverse(self, move_chance):
    scale, loc = self.scale.to(move_chance.device), self.loc.to(move_chance.device)
    t = (move_chance / scale) - loc
    return t
  
  def compute_loss_scaling_and_move_chance(self, t):
    scale, loc = self.scale.to(t.device), self.loc.to(t.device)
    move_chance = (t + loc) * scale
    move_chance = move_chance.clamp(0, 1)

    # at_prime is zero outside of unmasking interval
    at_prime = ((move_chance != 0) * (move_chance != 1) * scale)
    # edge case: move_chance_t = 1 but move_chance_s < 1
    at_prime += (t == (1/scale - loc)) * scale

    # TODO double check edge case when s == n
    
    loss_scale = at_prime / move_chance
    loss_scale = torch.nan_to_num(loss_scale, nan=0)
    return loss_scale, move_chance