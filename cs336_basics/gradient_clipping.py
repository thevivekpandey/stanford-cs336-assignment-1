import math
def gradient_clipping(parameters, M):
   eps = 1e-6
   s = 0
   for param in parameters:
      if param.grad is None:
          continue
      s += (param.grad ** 2).sum()
   s = math.sqrt(s)
   if s < M:
      return
   factor = M / (s + eps)
   for param in parameters:
      if param.grad is None:
          continue
      param.grad = param.grad * factor
