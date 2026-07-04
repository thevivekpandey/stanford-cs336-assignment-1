import torch
import torch.nn as nn

class RMSNorm(nn.Module):
   def __init__(self, d_model: int,
                      eps: float = 1e-5,
                      device=None,
                      dtype=None):
      super().__init__()
      self.eps = eps
      self.d_model = d_model
      self.G = nn.Parameter(torch.ones(d_model))

   def forward(self, x: torch.Tensor) -> torch.Tensor:
      in_dtype = x.dtype
      x = x.to(torch.float32)
      rms = torch.sqrt((x ** 2).sum(dim=-1, keepdim=True) / self.d_model + self.eps)
      result = (x / rms) * self.G
      return result.to(in_dtype)

if __name__ == "__main__":
   x = RMSNorm(3)
   x.load_state_dict({"G": torch.tensor([0.1, 0.2, 0.3])})
   print(x(torch.tensor([[1., 2, 3], [4, 5, 6]])))
