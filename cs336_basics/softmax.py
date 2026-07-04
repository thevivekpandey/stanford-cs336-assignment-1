import torch
import torch.nn as nn

class Softmax(nn.Module):
   def __init__(self):
      super().__init__()

   def forward(self, in_features: torch.Tensor, dim: int) -> torch.Tensor:
      M = in_features.amax(dim=dim)
      M = M.unsqueeze(dim=dim)
      U = in_features - M
      E = torch.exp(U)
      S = E.sum(dim=dim)
      S = S.unsqueeze(dim)
      return E / S

if __name__ == "__main__":
   x = torch.Tensor([[1, 2, 3], [3, 5, 7]])
   print(softmax()(x, 0))
   
      
