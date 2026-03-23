import math


def cosine_lr(
    step: int,
    warmup_steps: int,
    total_steps: int,
    max_lr: float,
    min_lr: float = 0.0,
) -> float:
    """Linear warmup then cosine decay."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    progress = min(progress, 1.0)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def wsd_lr(
    step: int,
    warmup_steps: int,
    stable_steps: int,
    decay_steps: int,
    max_lr: float,
    min_lr: float = 0.0,
) -> float:
    """
    Warmup-Stable-Decay schedule (SmolLM handbook default for all phases).
      [0, warmup_steps)                      — linear warmup
      [warmup_steps, warmup_steps+stable)    — constant max_lr
      [warmup_steps+stable, ...)             — cosine decay to min_lr
    """
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step < warmup_steps + stable_steps:
        return max_lr
    decay_step = step - warmup_steps - stable_steps
    progress = min(decay_step / decay_steps, 1.0)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
