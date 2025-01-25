# utils.py

import math

def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    """
    Cosine decay learning rate scheduler with linear warmup.
    """
    # 1) linear warmup
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    
    # 2) clamp LR to min_lr after max_steps
    if step > max_steps:
        return min_lr
    
    # 3) in between, do a cosine decay
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)
