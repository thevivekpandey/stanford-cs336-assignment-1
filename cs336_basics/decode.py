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
   logits = logits / temperature
   threshold = torch.topk(logits, top_p, dim=-1).values[-1]
   filtered_logits = logits.masked_fill(logits < threshold, float('-inf'))
   probs = Softmax()(filtered_logits, dim=-1)
   return torch.multinomial(probs, num_samples=1)

def decode(input_prompt: str,
           model: nn.Module,
           max_tokens: int,
           temperature: float,
           top_p: int,
           tokenizer: Tokenizer,
           context_length: int,
           device: str):
    model.eval()
    input_tokens = tokenizer.encode(input_prompt)
    input_tensor = torch.tensor(input_tokens, dtype=torch.long).to(device)
    generated = []
    with torch.no_grad():
       for _ in range(max_tokens):
          context = input_tensor[-context_length:]
          token = get_next_token(model, context, temperature, top_p).to(input_tensor.dtype)
          if tokenizer.vocab[token.item()] == b'<|endoftext|>':
             break
          generated.append(token.item())
          input_tensor = torch.cat([input_tensor, token])
    completion = tokenizer.decode(generated)
    print(input_prompt + completion)
    return completion

if __name__ == "__main__":
   device = "cuda"
   checkpoint_path = "checkpoints/checkpoint_step_4900.pt"

   tokenizer = Tokenizer.from_files("vocab.pkl",
                                    "merges.pkl",
                                    ["<|endoftext|>"])

   # Architecture must match the checkpoint (see training_together.py defaults).
   vocab_size = 10000
   context_length = 256
   d_model = 512
   num_layers = 4
   num_heads = 16
   d_ff = 1344
   theta = 10000.0
   max_seq_len = 1024

   rope = RoPE(theta, d_model // num_heads, max_seq_len, device)
   model = TransformerLM(vocab_size,
                         context_length,
                         d_model,
                         num_layers,
                         num_heads,
                         d_ff,
                         device,
                         rope)

   checkpoint = torch.load(checkpoint_path, map_location=device)
   model.load_state_dict(checkpoint["model_state_dict"])
   print(f"Loaded checkpoint {checkpoint_path} (step {checkpoint['step']})")

   temperature = 1.5
   top_p = 10
   decode("Tom and Lily were playing with their toys", model, 512, temperature, top_p, tokenizer, context_length, device)
