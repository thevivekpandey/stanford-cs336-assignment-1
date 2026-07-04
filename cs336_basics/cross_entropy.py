from jaxtyping import Bool, Float, Int
import torch
from torch import Tensor

def cross_entropy(logits: Float[Tensor, " ... vocab_size"],
                  targets: Int[Tensor, " ... "]):
   amax = torch.max(logits, dim=-1, keepdim=True).values
   N = logits - amax
   S = torch.exp(N).sum(dim=-1)
   ids = targets.unsqueeze(1)
   d = logits.dim()
   selected = N.gather(int(d) - 1, ids)
   selected = selected.squeeze(-1)
   
   L = torch.log(S) - selected
   return L.mean()

if __name__ == "__main__":
   logits = Tensor([[1, 2, 3], [4, 1, 3]])
   targets = torch.tensor([0, 1])
   ent = cross_entropy(logits, targets)
   print(ent)
