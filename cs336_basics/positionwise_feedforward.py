import torch
import torch.nn as nn
from einops import rearrange, einsum

class SwiGLU(nn.Module):
   def __init__(self, d_model: int,
                      d_ff: int,
                      device: torch.device | None=None,
                      dtype: torch.dtype | None=None):
      super().__init__()
      self.W1 = nn.Parameter(torch.empty(d_ff, d_model, device=device))
      self.W2 = nn.Parameter(torch.empty(d_model, d_ff, device=device))
      self.W3 = nn.Parameter(torch.empty(d_ff, d_model, device=device))

      nn.init.trunc_normal_(self.W1, mean=0.0, std=1, a=-3.0, b=3.0)
      nn.init.trunc_normal_(self.W2, mean=0.0, std=1, a=-3.0, b=3.0)
      nn.init.trunc_normal_(self.W3, mean=0.0, std=1, a=-3.0, b=3.0)

   def silu(self, x: torch.Tensor) -> torch.Tensor:
      return x * torch.sigmoid(x)

   def forward(self, x: torch.Tensor) -> torch.Tensor:
      # Compute W2 (SiLU(W1x) * W3x)
      W1x = einsum(self.W1, x, "d_ff d_model, ... d_model -> ... d_ff") 
      silu = self.silu(W1x)
      W3x = einsum(self.W3, x, "d_ff d_model, ... d_model -> ... d_ff") 
      had_prod = silu * W3x
      return einsum(self.W2, had_prod, "d_model d_ff, ... d_ff -> ... d_model")

if __name__ == "__main__":
   x = SwiGLU(3, 10)
   x(torch.tensor([1., 2, 3]))
