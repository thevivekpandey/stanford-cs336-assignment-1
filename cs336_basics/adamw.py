from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

def grad_norm(parameters):
    """Global L2 norm over all parameter gradients (for logging/monitoring)."""
    return torch.sqrt(sum(p.grad.detach().pow(2).sum()
                          for p in parameters
                          if p.grad is not None))

class AdamW(torch.optim.Optimizer):
    def __init__(self, 
                 params, 
                 lr=1e-3, 
                 weight_decay=0.01, 
                 betas=(0.9, 0.999), 
                 eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, 
                    "weight_decay": weight_decay,
                    "betas": betas,
                    "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable]=None):
        loss = None if closure is None else closer()
        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]

                t = state.get("t", 1)
                m = state.get("m", 0.0)
                v = state.get("v", 0.0)

                grad = p.grad.data

                num = math.sqrt(1 - betas[1] ** t)
                den = 1 - betas[0] ** t
                lr_t = lr * num / den

                #This is where we update the weight: part 1
                p.data -= lr * weight_decay * p.data
                
                m = betas[0] * m + (1 - betas[0]) * grad
                v = betas[1] * v + (1 - betas[1]) * grad * grad

                #This is where we update the weight: part 2
                p.data -= lr_t * m / (torch.sqrt(v) + eps)

                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
