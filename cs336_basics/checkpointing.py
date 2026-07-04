import os
import typing
import torch
def save_checkpoint(model: torch.nn.Module, 
                    optimizer: torch.optim.Optimizer, 
                    iteration: int,
                    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    obj = {}
    obj["weights"] = model.state_dict()
    obj["optimizer_state"] = optimizer.state_dict()
    obj["iter"] = iteration
    torch.save(obj, out)

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer):
    obj = torch.load(src)
    model.load_state_dict(obj["weights"])
    optimizer.load_state_dict(obj["optimizer_state"])
    return obj["iter"]
