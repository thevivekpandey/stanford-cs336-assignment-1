from tokenizer import Tokenizer
import torch
import torch.nn as nn
from rope import RoPE
from transformer_lm import TransformerLM
from softmax import Softmax

def get_next_token(model: nn.Module, 
                   input_tensor, 
                   temperature: float, 
                   top_p: int):
   logits = model(input_tensor)[-1]
   print(f"logits.shape = {logits.shape}")
   probs = Softmax()(logits / temperature, dim=-1)
   threshold = torch.topk(logits, top_p, dim=-1).values[-1]
   print(threshold)
   filtered_logits = logits.masked_fill(logits < threshold, float('-inf'))
   new_logits = Softmax()(filtered_logits, dim=-1)
   return torch.multinomial(probs, num_samples=1)
 
def decode(input_prompt: str,
           model: nn.Module,
           max_tokens: int,
           temperature: float,
           top_p: int,
           tokenizer: Tokenizer,
           context_length: int):
    input_tokens = tokenizer.encode(input_prompt)
    input_tensor = torch.tensor(input_tokens, dtype=torch.int32)
    print(input_tensor)
    token = get_next_token(model, input_tensor, temperature, top_p)
    count = 1
    while count < max_tokens and tokenizer.vocab[token.item()] != b'<|endoftext|>':
       print("SEE")
       print(input_tensor)
       print(token)
       input_tensor = torch.cat([input_tensor, token])
       count += 1

if __name__ == "__main__":
   tokenizer = Tokenizer.from_files("vocab.pkl", 
                                    "merges.pkl", 
                                    ["<|endoftext|>"])

   vocab_size = 10000
   context_length = 32
   d_model = 32
   num_layers = 8
   num_heads = 2
   d_ff = 96
   theta = 10000
   max_seq_len = 100
   rope = RoPE(theta, d_model // num_heads, max_seq_len, "cpu")
   model = TransformerLM(vocab_size,
                         context_length,
                         d_model,
                         num_layers,
                         num_heads,
                         d_ff,
                         rope)
   temperature = 1.0
   top_p = 3
   decode("Hello sir!", model, max_seq_len, temperature, top_p, tokenizer, context_length)

