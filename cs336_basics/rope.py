import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int
from torch import Tensor

class RoPE(nn.Module):
   def __init__(self, theta: float,
                      d_k: int,
                      max_seq_len: int,
                      device:torch.device | None = None):
      super().__init__()
      self.theta = theta
      self.d_k = d_k
      self.max_seq_len = max_seq_len
      assert d_k % 2 == 0

   def forward(self, x: Float[Tensor, "... sequence_length d_k"], 
                     token_positions: Float[Tensor, "... sequence_length"]) -> Float[Tensor, "... sequence_length d_k"]:
      d_k = self.d_k
      i = torch.arange(d_k // 2)
      theta = self.theta ** (-2 * i / d_k)
      angles = token_positions.unsqueeze(-1) * theta

      cos = torch.cos(angles)
      sin = torch.sin(angles)

      x_even = x[..., 0::2]
      x_odd  = x[..., 1::2]

      out_even = x_even * cos - x_odd * sin
      out_odd  = x_even * sin + x_odd * cos

      out = torch.stack([out_even, out_odd], dim=-1)
      out = out.flatten(-2)

      return out

if __name__ == "__main__":
   r = RoPE(100, 16, 32)
   x = torch.tensor([i for i in range(16)])
   y = torch.tensor([i for i in range(16)])
   r(x, y)
