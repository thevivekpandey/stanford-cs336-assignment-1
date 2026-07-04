import torch
import torch.nn as nn
from einops import rearrange, einsum
from softmax import Softmax

class ScaledDotProductAttention(nn.Module):
   def __init__(self):
      super().__init__()

   def forward(self, Q: torch.Tensor,
                     K: torch.Tensor,
                     V: torch.Tensor,
                     mask: torch.Tensor | None = None) -> torch.Tensor:

      d_k = Q.shape[-1]
      P = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
      if mask is not None:
         mask = mask.bool()
         mask = ~mask
         r = mask * -1e9

         while r.dim() < P.dim():
            r = r.unsqueeze(0)
         Q = P + r
      else:
         Q = P

      scaled = Q / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
      S = Softmax()(scaled, dim=-1)
      A = einsum(S, V, "... queries keys , ... keys d_v -> ... queries d_v")
      return A

if __name__ == "__main__":
   Q = torch.Tensor([[1, 2, 3], [4, 5, 6]])
   K = torch.Tensor([[1, 2, 3], [4, 5, 6]])
   V = torch.Tensor([[1, 2, 3], [4, 5, 6]])
   mask = torch.Tensor([[True, True], [True, False]])
   sdpa = ScaledDotProductAttention()
   sdpa(Q, K, V, mask=mask)

