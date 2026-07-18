import torch
import torch.nn as nn
from einops import rearrange, einsum
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch.nn.init import trunc_normal_
from rmsnorm import RMSNorm
from multihead_self_attention import MultiheadSelfAttention
from positionwise_feedforward import SwiGLU, SiLU

class TransformerBlock(nn.Module):
   def __init__(self, d_model, 
                      num_heads, 
                      d_ff, 
                      device,
                      rope):
      super().__init__()
      self.d_model = d_model
      self.num_heads = num_heads
      self.d_ff = d_ff
      self.rope = rope

      self.rms1 = RMSNorm(d_model, device=device)
      self.mha = MultiheadSelfAttention(d_model, num_heads, device, rope)
      self.rms2 = RMSNorm(d_model, device=device)
      #self.ff = SwiGLU(d_model, d_ff, device)
      self.ff = SiLU(d_model, d_ff, device)

   def forward(self, in_features: Float[Tensor, " batch sequence_length d_model"]) -> Float[Tensor, " batch sequence_length d_model"]:

      first_part = in_features + self.mha(self.rms1(in_features))
      return first_part + self.ff(self.rms2(first_part))
