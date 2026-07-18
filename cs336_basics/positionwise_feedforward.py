import torch
import torch.nn as nn
from einops import rearrange, einsum
from linear import Linear

class SwiGLU(nn.Module):
   def __init__(self, d_model: int,
                      d_ff: int,
                      device: torch.device | None=None,
                      dtype: torch.dtype | None=None):
      super().__init__()

      self.W1 = Linear(d_model, d_ff, device=device)
      self.W2 = Linear(d_ff, d_model, device=device)
      self.W3 = Linear(d_model, d_ff, device=device)

   def silu(self, x: torch.Tensor) -> torch.Tensor:
      return x * torch.sigmoid(x)

   def forward(self, x: torch.Tensor) -> torch.Tensor:
      # Compute W2 (SiLU(W1x) * W3x)
      W1x = self.W1(x)
      silu = self.silu(W1x)
      W3x = self.W3(x)
      had_prod = silu * W3x
      return self.W2(had_prod)

class SiLU(nn.Module):
   def __init__(self, d_model: int,
                      d_ff: int,
                      device: torch.device | None=None,
                      dtype: torch.dtype | None=None):
      super().__init__()

      self.W1 = Linear(d_model, d_ff, device=device)
      self.W2 = Linear(d_ff, d_model, device=device)

   def silu(self, x: torch.Tensor) -> torch.Tensor:
      return x * torch.sigmoid(x)

   def forward(self, x: torch.Tensor) -> torch.Tensor:
      # Compute W2 (SiLU(W1x))
      W1x = self.W1(x)
      silu = self.silu(W1x)
      return self.W2(silu)

if __name__ == "__main__":
   x = SwiGLU(3, 10)
   x(torch.tensor([1., 2, 3]))
