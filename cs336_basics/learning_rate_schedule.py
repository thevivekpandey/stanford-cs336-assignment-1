import math
def learning_rate_schedule(t: int,
                           max_lr: float,
                           min_lr: float,
                           warmup_iter: int,
                           final_cosine_iter: int):
    if t < warmup_iter:
        return (t / warmup_iter) * max_lr
    elif t <= final_cosine_iter:
        frac = (t - warmup_iter) / (final_cosine_iter - warmup_iter)
        return min_lr + 0.5 * (1 + math.cos(frac * math.pi)) * (max_lr - min_lr)
    else:
        return min_lr

