"""Learning-rate schedule: linear warmup, then cosine decay to min_lr_ratio × peak."""
import math


def lr_at(step: int, total_steps: int, peak: float, warmup_ratio: float, min_lr_ratio: float) -> float:
    warmup = max(1, int(total_steps * warmup_ratio)) if warmup_ratio > 0 else 0
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return peak * (min_lr_ratio + (1 - min_lr_ratio) * cosine)
