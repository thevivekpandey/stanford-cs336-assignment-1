import torch
import torch.nn as nn

class Embedding(nn.Module):
   def __init__(self, num_embeddings: int, 
                      embedding_dim: int,
                      device: torch.device | None = None,
                      dtype: torch.dtype | None = None):
      super().__init__()
      self.E = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
      nn.init.trunc_normal_(self.E, mean=0.0, std=1, a=-3.0, b=3.0)

   def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
      x = torch.stack([self.E[t] for t in token_ids], dim=0)
      return x

if __name__ == "__main__":
   x = Embedding(10, 10)
