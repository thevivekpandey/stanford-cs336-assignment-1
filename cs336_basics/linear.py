import torch
import torch.nn as nn
from einops import rearrange, einsum
import math

class Linear(nn.Module):
   def __init__(self, in_features: int, 
                      out_features: int,
                      device: torch.device | None = None,
                      dtype: torch.dtype | None = None):
      super().__init__()
      self.W = nn.Parameter(torch.empty(out_features, in_features, device=device))
      std = math.sqrt(2 / (in_features + out_features))
      nn.init.trunc_normal_(self.W, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)

   def forward(self, x: torch.Tensor) -> torch.Tensor:
      return einsum(self.W, x, "... d_out d_in, ... d_in -> ... d_out")

if __name__ == "__main__":
   x = Linear(10, 10)
