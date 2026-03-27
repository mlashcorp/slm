# OLMo Learning Rate Schedules

Source: https://github.com/allenai/OLMo/blob/main/olmo/optim.py

## Overview

OLMo implements several learning rate schedulers with warmup support. The key schedules used in OLMo 2 are:

## Scheduler Types

### 1. CosWithWarmup (Cosine with Warmup)

```python
@dataclass
class CosWithWarmup(Scheduler):
    """
    Cosine decay with warmup.
    
    lr = base_lr * 0.5 * (1 + cos(pi * (step - warmup) / (T - warmup)))
    """
    num_warmup_steps: int = 0
    num_cycles: float = 0.5  # Number of cosine cycles
    
    def get_lr(self, step: int) -> float:
        if step < self.num_warmup_steps:
            # Linear warmup
            return self.base_lr * (step / self.num_warmup_steps)
        else:
            # Cosine decay
            progress = (step - self.num_warmup_steps) / (self.num_steps - self.num_warmup_steps)
            return self.base_lr * 0.5 * (1 + cos(pi * progress))
```

### 2. LinearWithWarmup

```python
@dataclass  
class LinearWithWarmup(Scheduler):
    """
    Linear decay with warmup.
    
    lr = base_lr * (1 - step / T)
    """
    num_warmup_steps: int = 0
    
    def get_lr(self, step: int) -> float:
        if step < self.num_warmup_steps:
            return self.base_lr * (step / self.num_warmup_steps)
        else:
            return self.base_lr * max(0.0, 1 - step / self.num_steps)
```

### 3. InvSqrtWithWarmup (Inverse Square Root)

```python
@dataclass
class InvSqrtWithWarmup(Scheduler):
    """
    Inverse square root decay with warmup.
    
    lr = base_lr / sqrt(max(step, warmup))
    
    This is the classic transformer schedule from "Attention Is All You Need".
    """
    num_warmup_steps: int = 0
    
    def get_lr(self, step: int) -> float:
        if step < self.num_warmup_steps:
            return self.base_lr * (step / self.num_warmup_steps)
        else:
            return self.base_lr / sqrt(step)
```

### 4. MaxScheduler

```python
@dataclass
class MaxScheduler(Scheduler):
    """
    Takes the minimum of two schedulers at each step.
    Useful for combining warmup with decay.
    """
    scheduler1: Scheduler
    scheduler2: Scheduler
    
    def get_lr(self, step: int) -> float:
        return min(self.scheduler1.get_lr(step), self.scheduler2.get_lr(step))
```

### 5. CosLinearEnvelope

```python
@dataclass
class CosLinearEnvelope(Scheduler):
    """
    WSD (Warmup-Stable-Decay) schedule used in OLMo 2.
    
    1. Warmup: Linear warmup from 0 to base_lr
    2. Stable: Constant at base_lr
    3. Decay: Linear or cosine decay to min_lr
    
    This is similar to what SmolLM uses (trapezoidal schedule).
    """
    num_warmup_steps: int = 0
    num_stable_steps: int = 0
    min_lr: float = 0.0
    
    def get_lr(self, step: int) -> float:
        if step < self.num_warmup_steps:
            # Warmup phase
            return self.base_lr * (step / self.num_warmup_steps)
        elif step < self.num_warmup_steps + self.num_stable_steps:
            # Stable phase
            return self.base_lr
        else:
            # Decay phase
            decay_steps = self.num_steps - self.num_warmup_steps - self.num_stable_steps
            progress = (step - self.num_warmup_steps - self.num_stable_steps) / decay_steps
            return self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + cos(pi * progress))
```

### 6. BoltOnWarmupScheduler

```python
class BoltOnWarmupScheduler:
    """
    Wraps any scheduler with an additional warmup phase.
    Useful for fine-tuning or stage 2 training.
    """
    def __init__(self, scheduler: Scheduler, num_warmup_steps: int):
        self.scheduler = scheduler
        self.num_warmup_steps = num_warmup_steps
    
    def get_lr(self, step: int) -> float:
        if step < self.num_warmup_steps:
            return self.scheduler.base_lr * (step / self.num_warmup_steps)
        else:
            return self.scheduler.get_lr(step - self.num_warmup_steps)
```

## OLMo 2 Schedule Configuration

From `configs/official-1124/OLMo2-7B-stage1.yaml`:

```yaml
optimizer:
  type: adamw
  lr: 3.0e-4
  betas: [0.9, 0.95]
  eps: 1.0e-8
  weight_decay: 0.1
  decay_norm_and_bias: true
  decay_embeddings: false

scheduler:
  type: linear_with_warmup
  num_warmup_steps: 2000
  num_steps: 500000  # Total training steps
```

## Gradient Clipping

OLMo supports two clipping modes:

### 1. Global Fixed Clipping

```python
# Clip gradients by global norm
for group in optimizer.param_groups:
    max_grad_norm = group.get("max_grad_norm")
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(group["params"], max_grad_norm)
```

### 2. Adaptive Clipping

```python
# Clip based on exponential moving average of gradient norms
for group in optimizer.param_groups:
    max_norm_ratio = group.get("max_grad_norm_ratio")
    if max_norm_ratio is not None:
        # Adaptive clipping based on grad norm history
        max_allowed_norm = max_norm_ratio * grad_norm_exp_avg
        clip_coef = max_allowed_norm / (grad_norm + 1e-6)
        if clip_coef < 1:
            p.grad.mul_(clip_coef)
```

## Comparison to SLM Schedule

| Feature | OLMo 2 | SLM |
|---------|--------|-----|
| Base LR | 3e-4 | 6e-4 (Phase 1) |
| Warmup | 2000 steps | 2000 steps |
| Schedule | Linear decay | WSD (trapezoidal) |
| Decay phase | Linear to 0 | Linear to 3e-5 (Phase 1) |
| Stable phase | None | Yes (before decay) |
| Min LR | 0 | 3e-5 |

## Relevant Patterns for SLM

1. **Bolt-on warmup** — Add warmup when continuing from checkpoint
2. **Adaptive clipping** — Track gradient norm EMA for stability
3. **Scheduler composition** — Combine multiple schedules
4. **Decay norm and bias** — Don't decay LayerNorm params

## Citation

See OLMo 2 paper for schedule ablations.
