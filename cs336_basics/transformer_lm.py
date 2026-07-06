import torch
import torch.nn as nn
from einops import rearrange, einsum
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch.nn.init import trunc_normal_
from transformer_block import TransformerBlock
from embedding import Embedding
from rmsnorm import RMSNorm
from linear import Linear
from softmax import Softmax

class TransformerLM(nn.Module):
   def __init__(self, 
                vocab_size, 
                context_length, 
                d_model,
                num_layers, 
                num_heads, 
                d_ff,
                device,
                rope):
      super().__init__()
      self.embedding = Embedding(vocab_size, d_model, device)
      self.tblocks = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, device, rope) for _ in range(num_layers)])
      self.norm = RMSNorm(d_model, device=device)
      self.linear = Linear(d_model, vocab_size, device)
      self.softmax = Softmax()

   def forward(self, in_indices: Int[Tensor, " batch_size sequence_length"]) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
      x = self.embedding(in_indices)
      for tblock in self.tblocks:
         x = tblock(x)
      x = self.norm(x)
      x = self.linear(x)
      
      return x

if __name__ == "__main__":
   vocab_size = 50257
   context_length = 1024
   num_layers = 48
   d_model = 1600
   num_heads = 25
   d_ff = 4288
   tlm = TransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, None)
   
