# OLMo Training Loop

Source: https://github.com/allenai/OLMo/blob/main/scripts/train.py

## Overview

OLMo uses a distributed training script with support for both DDP and FSDP strategies.

## Key Components

### 1. Configuration Loading

```python
from olmo.config import TrainConfig

def main(cfg: TrainConfig) -> None:
    # Ensure run name set
    if cfg.run_name is None:
        raise OLMoConfigurationError("--run_name is required")
```

### 2. Device Setup

```python
# Set CUDA device
if torch.cuda.is_available():
    torch.cuda.set_device(f"cuda:{get_local_rank()}")
    torch.cuda.empty_cache()
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
```

### 3. Batch Size Calculation

```python
# Fill configuration options
cfg.device_train_batch_size = cfg.global_train_batch_size // get_world_size()
cfg.device_train_grad_accum = cfg.device_train_batch_size // cfg.device_train_microbatch_size
```

### 4. Model Initialization

```python
from olmo.model import OLMo

# Initialize model
olmo_model = OLMo(cfg.model)

# Log parameter counts
log.info(f"Total number of parameters: {olmo_model.num_params():,d}")
log.info(f"Number of non-embedding parameters: {olmo_model.num_params(include_embedding=False):,d}")

# Set activation checkpointing
olmo_model.set_activation_checkpointing(cfg.activation_checkpointing)
```

### 5. Distributed Strategy

```python
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# DDP
if cfg.distributed_strategy == DistributedStrategy.ddp:
    dist_model = DDP(olmo_model.to(device), find_unused_parameters=cfg.ddp.find_unused_params)

# FSDP
elif cfg.distributed_strategy == DistributedStrategy.fsdp:
    wrap_policy = olmo_model.get_fsdp_wrap_policy(cfg.fsdp.wrapping_strategy)
    dist_model = FSDP(
        olmo_model,
        param_init_fn=param_init_fn,
        use_orig_params=True,
        auto_wrap_policy=wrap_policy,
        sharding_strategy=cfg.fsdp.sharding_strategy,
    )
```

### 6. Data Loader Construction

```python
from olmo.data import build_train_dataloader

train_loader = build_train_dataloader(cfg)
```

### 7. Evaluator Construction

```python
from olmo.eval import build_evaluators

evaluators = build_evaluators(cfg, device)
```

### 8. Trainer Initialization

```python
from olmo.train import Trainer

trainer = Trainer(
    cfg=cfg,
    epoch=0,
    model=dist_model,
    optim=optim,
    scheduler=scheduler,
    train_loader=train_loader,
    evaluators=evaluators,
    device=device,
)
```

### 9. Training Loop

```python
# Main training loop
while trainer.epoch < cfg.num_epochs:
    trainer.train()
    trainer.eval()
    trainer.epoch += 1
```

## Key Differences from SLM

| Aspect | OLMo | SLM |
|--------|------|-----|
| Distributed | FSDP/DDP support | Single GPU |
| Config | YAML + dataclass | Hardcoded in train.py |
| Checkpointing | Every 1000 steps | End of phase |
| WandB | Integrated | Integrated |
| Evaluation | Inline evaluators | Separate eval.py |

## Relevant Patterns for SLM

1. **Grad accumulation calculation** — `device_train_grad_accum = batch_size // microbatch_size`
2. **Activation checkpointing** — `model.set_activation_checkpointing()`
3. **Seed all** — `seed_all(cfg.seed)` for reproducibility
4. **Peak memory logging** — Track before/after model wrapping

## Citation

See OLMo 2 paper for full methodology.
