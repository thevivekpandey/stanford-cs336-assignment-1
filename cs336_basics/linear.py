import torch
import torch.nn as nn
from einops import rearrange, einsum

class Linear(nn.Module):
   def __init__(self, in_features: int, 
                      out_features: int,
                      device: torch.device | None = None,
                      dtype: torch.dtype | None = None):
      super().__init__()
      self.W = nn.Parameter(torch.empty(out_features, in_features))
      nn.init.trunc_normal_(self.W, mean=0.0, std=1, a=-3.0, b=3.0)

   def forward(self, x: torch.Tensor) -> torch.Tensor:
      return einsum(self.W, x, "... d_out d_in, ... d_in -> ... d_out")

if __name__ == "__main__":
   x = Linear(10, 10)
