import numpy as np
import argparse
from tokenizer import Tokenizer
from transformer_lm import TransformerLM
from rope import RoPE
from adamw import AdamW
from data_loading import data_loading
from cross_entropy import cross_entropy

class Train:
    def __init__(self,
                 data: np.ndarray,
                 d_model: int,
                 num_layers: int,
                 context_size: int,
                 batch_size: int,
                 num_steps: int,
                 vocab_size: int,
                 d_ff: int,
                 num_heads: int,
                 theta: float,
                 max_seq_len: int,
                 device: str):
        self.data = data
        self.d_model = d_model
        self.num_layers = num_layers
        self.context_size = context_size
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.vocab_size = vocab_size
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.device = device

    def train(self):
       print("Starting training")
       TRAIN_VAL_SPLIT = 0.9
       train_data_points = int(TRAIN_VAL_SPLIT * len(self.data))
       train_data = self.data[:train_data_points]
       val_data = self.data[train_data_points:]

       rope = RoPE(self.theta, self.d_model // self.num_heads, self.max_seq_len, self.device)
       model = TransformerLM(self.vocab_size,
                             self.context_size,
                             self.d_model,
                             self.num_layers,
                             self.num_heads,
                             self.d_ff,
                             rope)
       optimizer = AdamW(model.parameters())
       train_losses, val_losses, train_accs, val_accs = [], [], [], []
       for epoch in range(num_steps):
          model.train()

          input_batch, target_batch = data_loading(train_data,
                                              self.batch_size,
                                              self.context_size,
                                              self.device)
          optimizer.zero_grad()
          logits = model(input_batch)
          loss = cross_entropy(logits, target_batch)
          print(f"epoch {epoch}: loss = {loss.item()}")
          loss.backward()
          optimizer.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_model", type=int, default=32, required=False)
    parser.add_argument("--num_layers", type=int, default=4, required=False)
    parser.add_argument("--context_size", type=int, default=256, required=False)
    parser.add_argument("--batch_size", type=int, default=32, required=False)
    parser.add_argument("--num_steps", type=int, default=10, required=False)
    parser.add_argument("--vocab_size", type=int, default=50257, required=False)
    parser.add_argument("--d_ff", type=int, default=64, required=False)
    parser.add_argument("--num_heads", type=int, default=2, required=False)
    parser.add_argument("--theta", type=float, default=10000.0, required=False)
    parser.add_argument("--max_seq_len", type=int, default=4, required=False)
    parser.add_argument("--device", type=str, default="cpu", required=False)
    args = parser.parse_args()

    d_model = args.d_model
    num_layers = args.num_layers
    context_size = args.context_size
    batch_size = args.batch_size
    num_steps = args.num_steps
    vocab_size = args.vocab_size
    d_ff = args.d_ff
    num_heads = args.num_heads
    theta = args.theta
    max_seq_len = args.max_seq_len
    device = args.device

    tokenizer = Tokenizer.from_files("vocab.pkl", "merges.pkl", ["<|endoftext|>"])
    iterator = tokenizer.encode_iterable(open('TinyStoriesV2-GPT4-valid.txt'))
    arr = np.fromiter(iterator, dtype=int)
    t = Train(arr, d_model, num_layers, context_size, batch_size, num_steps, vocab_size, d_ff, num_heads, theta, max_seq_len, device)
    t.train()
