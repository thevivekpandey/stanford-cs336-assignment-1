import numpy as np
import torch

def data_loading(x: np.ndarray,
                 batch_size: int,
                 context_length: int,
                 device: str):   
   start_indices = np.random.randint(low=0, high=len(x) - context_length, size=(batch_size,))
   offsets = np.arange(context_length)
   indices = start_indices[:, None] + offsets[None, :]
   target_indices = indices + 1
   a = torch.tensor(x[indices].astype(np.int64))
   b = torch.tensor(x[target_indices].astype(np.int64))
   a = a.to(device)
   b = b.to(device)
   return a, b
   

if __name__ == "__main__":
    x = np.array([i for i in range(100)])
    ret = data_loading(x, 10, 10, "abc")
    print(ret)
