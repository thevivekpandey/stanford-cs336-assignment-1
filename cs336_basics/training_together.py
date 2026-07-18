import sys
import os
import time
import numpy as np
import argparse
from tokenizer import Tokenizer
from transformer_lm import TransformerLM
from rope import RoPE
from adamw import AdamW, grad_norm
from data_loading import data_loading
from cross_entropy import cross_entropy
from learning_rate_schedule import learning_rate_schedule
from gradient_clipping import gradient_clipping

import torch
import wandb

class Train:
    def __init__(self,
                 train_data: np.ndarray,
                 val_data: np.ndarray,
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
                 lr: float,
                 min_lr: float,
                 warmup_iter: int,
                 final_cosine_iter: int,
                 max_grad_norm: float,
                 device: str,
                 max_train_seconds: float | None = None):
        self.train_data = train_data
        self.val_data = val_data
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
        self.lr = lr
        self.min_lr = min_lr
        self.warmup_iter = warmup_iter
        self.final_cosine_iter = final_cosine_iter
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.max_train_seconds = max_train_seconds

    def train(self):
       print("Starting training")
       train_data = self.train_data
       val_data = self.val_data
       print(f"train tokens: {len(train_data)}, val tokens: {len(val_data)}")

       rope = RoPE(self.theta, self.d_model // self.num_heads, self.max_seq_len, self.device)
       model = TransformerLM(self.vocab_size,
                             self.context_size,
                             self.d_model,
                             self.num_layers,
                             self.num_heads,
                             self.d_ff,
                             self.device,
                             rope)
       optimizer = AdamW(model.parameters(), lr=self.lr)
       DIR = "checkpoints"
       os.makedirs(DIR, exist_ok=True)
       start_time = time.time()
       step = 0
       while True:
          if self.max_train_seconds is not None and time.time() - start_time >= self.max_train_seconds:
              print(f"step {step}: reached time budget of {self.max_train_seconds}s; stopping")
              break
          if self.max_train_seconds is None and step >= self.num_steps:
              break
          model.train()

          #lr = learning_rate_schedule(step,
          #                            self.lr,
          #                            self.min_lr,
          #                            self.warmup_iter,
          #                            self.final_cosine_iter)
          #for group in optimizer.param_groups:
          #    group["lr"] = lr

          input_batch, target_batch = data_loading(train_data,
                                              self.batch_size,
                                              self.context_size,
                                              self.device)
          optimizer.zero_grad()
          logits = model(input_batch)
          loss = cross_entropy(logits, target_batch)
          print(f"step {step}: loss = {loss.item()}")
          loss.backward()

          norm = grad_norm(model.parameters())
          gradient_clipping(model.parameters(), self.max_grad_norm)
          optimizer.step()

          metrics = {
              "step": step,
              "train_loss": loss.item(),
              "learning_rate": optimizer.param_groups[0]["lr"],
              "grad_norm": norm.item(),
              "wall_clock_time": time.time() - start_time,
          }

          if step % 10 == 0:
              model.eval()
              with torch.no_grad():
                  input_batch, target_batch = data_loading(val_data,
                                                     self.batch_size,
                                                     self.context_size,
                                                     self.device)
                  logits = model(input_batch)
                  val_loss = cross_entropy(logits, target_batch)
              print(f"step {step}: val loss = {val_loss.item()}")
              metrics["val_loss"] = val_loss.item()

          wandb.log(metrics, step=step)

          #if step % 100 == 0:
          #    checkpoint_path = os.path.join(DIR, f"checkpoint_step_{step}.pt")
          #    torch.save({
          #        "step": step,
          #        "model_state_dict": model.state_dict(),
          #        "optimizer_state_dict": optimizer.state_dict(),
          #    }, checkpoint_path)
          #    print(f"step {step}: saved checkpoint to {checkpoint_path}")

          step += 1

       wandb.finish()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_model", type=int, default=512, required=False)
    parser.add_argument("--num_layers", type=int, default=4, required=False)
    parser.add_argument("--context_size", type=int, default=256, required=False)
    parser.add_argument("--batch_size", type=int, default=128 , required=False)
    parser.add_argument("--num_steps", type=int, default=5000, required=False)
    parser.add_argument("--vocab_size", type=int, default=10000, required=False)
    parser.add_argument("--d_ff", type=int, default=2048, required=False)
    parser.add_argument("--num_heads", type=int, default=16, required=False)
    parser.add_argument("--theta", type=float, default=10000.0, required=False)
    parser.add_argument("--max_seq_len", type=int, default=1024, required=False)
    parser.add_argument("--lr", type=float, default=1e-3, required=False)
    parser.add_argument("--min_lr", type=float, default=1e-5, required=False)
    parser.add_argument("--max_grad_norm", type=float, default=1.0, required=False)
    parser.add_argument("--device", type=str, default="cuda", required=False)
    parser.add_argument("--group", type=str, default=None, required=False)
    parser.add_argument("--max_train_seconds", type=float, default=None, required=False, help="If set, train for this many wall-clock seconds regardless of --num_steps")
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
    lr = args.lr
    min_lr = args.min_lr
    max_grad_norm = args.max_grad_norm
    device = args.device

    warmup_iter = num_steps // 10 
    final_cosine_iter = num_steps - num_steps // 10

    wandb.init(
        entity="thevivekpandey-personal",
        project="stanford-cs336",
        name=f"silu",
        group=args.group,
        config={
            **vars(args),
            "warmup_iter": warmup_iter,
            "final_cosine_iter": final_cosine_iter,
        },
    )

    train_data = np.load("TinyStories-train.npy", mmap_mode="r")
    val_data = np.load("TinyStories-valid.npy", mmap_mode="r")
    t = Train(train_data, val_data, d_model, num_layers, context_size, batch_size, num_steps, vocab_size, d_ff, num_heads, theta, max_seq_len, lr, min_lr, warmup_iter, final_cosine_iter, max_grad_norm, device, args.max_train_seconds)
    t.train()
