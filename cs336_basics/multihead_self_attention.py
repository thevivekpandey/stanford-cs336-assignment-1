import torch
import torch.nn as nn
from einops import rearrange, einsum
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch.nn.init import trunc_normal_
from scaled_dot_product_attention import ScaledDotProductAttention

class MultiheadSelfAttentionSaved(nn.Module):
   def __init__(self, d_model, num_heads, rope=None):
      super().__init__()
      self.d_model = d_model
      self.num_heads = num_heads
      self.rope = rope

      self.W_q = nn.Parameter(torch.empty(d_model, d_model))
      self.W_k = nn.Parameter(torch.empty(d_model, d_model))
      self.W_v = nn.Parameter(torch.empty(d_model, d_model))
      self.W_o = nn.Parameter(torch.empty(d_model, d_model))

      trunc_normal_(self.W_q, mean=0.0, std=1, a=-3.0, b=3.0)
      trunc_normal_(self.W_k, mean=0.0, std=1, a=-3.0, b=3.0)
      trunc_normal_(self.W_v, mean=0.0, std=1, a=-3.0, b=3.0)
      trunc_normal_(self.W_o, mean=0.0, std=1, a=-3.0, b=3.0)

   def forward(self, x):
      d_model = self.d_model
      num_heads = self.num_heads
      heads_dim = d_model // num_heads

      # INCORRECT APPROACH (commented out):
      # w_q_split = self.W_q.reshape(d_model, num_heads, heads_dim)
      # w_k_split = self.W_k.reshape(d_model, num_heads, heads_dim)
      # w_v_split = self.W_v.reshape(d_model, num_heads, heads_dim)

      # WHY THE ABOVE IS INCORRECT:
      # The weight matrix W_q has shape (d_model, d_model) where:
      # - Rows represent OUTPUT dimensions
      # - Columns represent INPUT dimensions
      # In multi-head attention, we want to split the OUTPUT space into multiple heads.
      # The above reshape incorrectly treats each row as having num_heads separate transformations,
      # when actually we want to split the output dimension (rows) into num_heads groups.
      # This fundamentally misunderstands how the projection matrix should be partitioned.

      # INCORRECT EINSUM (commented out):
      # q = einsum(w_q_split, x, "d_model nh hd, ... seq_len d_model -> ... nh seq_len hd")
      # k = einsum(w_k_split, x, "d_model nh hd, ... seq_len d_model -> ... nh seq_len hd")
      # v = einsum(w_v_split, x, "d_model nh hd, ... seq_len d_model -> ... nh seq_len hd")

      # WHY THE EINSUM IS INCORRECT:
      # This einsum pattern assumes the weight tensor's first dimension matches with the input's
      # last dimension (d_model), but the reshaping above doesn't correctly organize the weights
      # for this operation. It treats the INPUT dimension as being split by heads, not the OUTPUT.

      # CORRECT APPROACH:
      # We need to reshape the weight matrices so that the OUTPUT dimension is split into heads
      # W_q: (d_model, d_model) -> (num_heads, heads_dim, d_model)
      # This means each head gets a (heads_dim, d_model) weight matrix

      # First reshape to (num_heads * heads_dim, d_model) = (d_model, d_model)
      # Then view as (num_heads, heads_dim, d_model)
      w_q_split = self.W_q.T.reshape(d_model, num_heads, heads_dim).permute(1, 2, 0)
      w_k_split = self.W_k.T.reshape(d_model, num_heads, heads_dim).permute(1, 2, 0)
      w_v_split = self.W_v.T.reshape(d_model, num_heads, heads_dim).permute(1, 2, 0)

      # Now w_q_split has shape (num_heads, heads_dim, d_model)
      # Each head has its own (heads_dim, d_model) projection matrix

      # Correct einsum: for each head, multiply its projection matrix with the input
      q = einsum(x, w_q_split, "... seq_len d_model, nh hd d_model -> ... nh seq_len hd")
      k = einsum(x, w_k_split, "... seq_len d_model, nh hd d_model -> ... nh seq_len hd")
      v = einsum(x, w_v_split, "... seq_len d_model, nh hd d_model -> ... nh seq_len hd")
      seq_len = x.shape[-2]
      mask = torch.tril(torch.ones(seq_len, seq_len)).bool()

      sdpa = ScaledDotProductAttention()
      attention = sdpa(q, k, v, mask)
      attention = rearrange(attention, "batch nh seq_len dim_head -> batch seq_len (nh dim_head)")
      mha = einsum(attention, self.W_o, "... sequence_length d_model, d_model d_model -> ... sequence_length d_model")
      return mha

class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, rope=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.rope = rope

        # Initialize weight parameters
        self.W_q = nn.Parameter(torch.empty(d_model, d_model))
        self.W_k = nn.Parameter(torch.empty(d_model, d_model))
        self.W_v = nn.Parameter(torch.empty(d_model, d_model))
        self.W_o = nn.Parameter(torch.empty(d_model, d_model))

        # Initialize with truncated normal
        trunc_normal_(self.W_q, mean=0.0, std=1, a=-3.0, b=3.0)
        trunc_normal_(self.W_k, mean=0.0, std=1, a=-3.0, b=3.0)
        trunc_normal_(self.W_v, mean=0.0, std=1, a=-3.0, b=3.0)
        trunc_normal_(self.W_o, mean=0.0, std=1, a=-3.0, b=3.0)

    def forward(self, x: Float[Tensor, "... sequence_length d_model"]) -> Float[Tensor, "... sequence_length d_model"]:
        d_model = self.d_model
        num_heads = self.num_heads
        heads_dim = d_model // num_heads
        seq_len = x.shape[-2]

        # Step 1: Apply linear projections in batch from d_model => d_model
        # The weight matrices are (d_model, d_model) with PyTorch convention:
        # rows = output features, columns = input features
        # So we need x @ W.T for the projection
        q = einsum(x, self.W_q.T, "... seq_len d_model, d_model d_model_out -> ... seq_len d_model_out")
        k = einsum(x, self.W_k.T, "... seq_len d_model, d_model d_model_out -> ... seq_len d_model_out")
        v = einsum(x, self.W_v.T, "... seq_len d_model, d_model d_model_out -> ... seq_len d_model_out")

        # Step 2: Reshape to split the d_model dimension into num_heads separate heads
        # From (..., seq_len, d_model) to (..., num_heads, seq_len, heads_dim)
        q = rearrange(q, "... seq_len (nh hd) -> ... nh seq_len hd", nh=num_heads, hd=heads_dim)
        k = rearrange(k, "... seq_len (nh hd) -> ... nh seq_len hd", nh=num_heads, hd=heads_dim)
        v = rearrange(v, "... seq_len (nh hd) -> ... nh seq_len hd", nh=num_heads, hd=heads_dim)

        token_positions = torch.arange(seq_len)

        if self.rope is not None:
           q = self.rope(q, token_positions)
           k = self.rope(k, token_positions)

        # Step 3: Create causal mask for self-attention
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))

        # Step 4: Apply scaled dot-product attention
        sdpa = ScaledDotProductAttention()
        attention = sdpa(q, k, v, mask)

        # Step 5: Concatenate heads
        # From (..., num_heads, seq_len, heads_dim) to (..., seq_len, d_model)
        attention = rearrange(attention, "... nh seq_len hd -> ... seq_len (nh hd)")

        # Step 6: Apply output projection
        output = einsum(attention, self.W_o.T, "... seq_len d_model, d_model d_model_out -> ... seq_len d_model_out")

        return output

if __name__ == "__main__":
   msa = MultiheadSelfAttention(16, 4) #d_model, nh
   msa(torch.randn(1, 6, 16)) #seq_len, d_model


