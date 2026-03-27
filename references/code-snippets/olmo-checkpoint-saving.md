# OLMo Checkpoint Saving

Source: https://github.com/allenai/OLMo/blob/main/olmo/train.py

## Overview

OLMo supports multiple checkpoint types for distributed training:

1. **Sharded** — Each GPU saves its own shard (fast, minimal memory)
2. **Unsharded** — Full model saved on rank 0 (portable, slower)
3. **Ephemeral** — Temporary checkpoints for fault tolerance

## Checkpoint Types

```python
from olmo.config import CheckpointType

class CheckpointType(Enum):
    sharded = "sharded"          # Per-GPU shards
    unsharded = "unsharded"      # Full model on rank 0
    sharded_ephemeral = "ephemeral"  # Temporary checkpoints
```

## Save Checkpoint Flow

```python
def save_checkpoint(
    self,
    checkpoint_type: CheckpointType = CheckpointType.sharded
) -> Tuple[PathOrStr, Optional[PathOrStr]]:
    if checkpoint_type == CheckpointType.sharded:
        result = self.save_sharded_checkpoint()
    elif checkpoint_type == CheckpointType.unsharded:
        result = self.save_unsharded_checkpoint()
    elif checkpoint_type == CheckpointType.sharded_ephemeral:
        result = self.save_ephemeral_checkpoint()
    else:
        raise NotImplementedError(checkpoint_type)
    
    gc_cuda()  # Garbage collect CUDA memory
    return result
```

## Sharded Checkpoint

```python
def save_sharded_checkpoint(self) -> Tuple[PathOrStr, Optional[PathOrStr]]:
    checkpointer = build_sharded_checkpointer(self.cfg)
    result = self._save_checkpoint(checkpointer, CheckpointType.sharded)
    self.last_sharded_checkpoint_step = self.global_step
    return result
```

## Unsharded Checkpoint (Portable)

```python
def save_unsharded_checkpoint(self) -> Tuple[PathOrStr, Optional[PathOrStr]]:
    checkpointer = FullCheckpointer(self.cfg)
    result = self._save_checkpoint(checkpointer, CheckpointType.unsharded)
    self.last_unsharded_checkpoint_step = self.global_step
    return result
```

## Trainer State Saved

```python
trainer_state = {
    "epoch": self.epoch,
    "global_step": self.global_step,
    "global_train_tokens_seen": self.global_train_tokens_seen,
    "best_eval_loss": self.best_eval_loss,
    "best_eval_step": self.best_eval_step,
    "world_size": get_world_size(),
    "seed": self.cfg.seed,
    "scheduler": self.scheduler.state_dict(),
}
```

## Checkpoint Contents

Each checkpoint contains:

| Component | Description |
|-----------|-------------|
| `model.pt` | Model weights (sharded or full) |
| `optim.pt` | Optimizer state (momentum, variance) |
| `trainer.pt` | Training progress (step, epoch, tokens seen) |
| `config.yaml` | Training configuration |
| `latest` | Symlink to latest checkpoint |

## Configuration (OLMo2-1B)

```yaml
save_folder: http://olmo-data.org/checkpoints/OLMo-small/${run_name}
save_interval: 1000                      # Save every 1000 steps
save_interval_ephemeral: 1000           # Ephemeral checkpoints
save_num_checkpoints_to_keep: 0        # Keep all (0 = unlimited)
save_interval_unsharded: 1000          # Full checkpoints
save_num_unsharded_checkpoints_to_keep: -1  # Keep all
```

## Checkpoint Cleanup

```python
def remove_checkpoint(self, idx: int = 0, checkpoint_type: CheckpointType = CheckpointType.sharded):
    if checkpoint_type == CheckpointType.sharded:
        self.remove_sharded_checkpoint(idx=idx)
    elif checkpoint_type == CheckpointType.unsharded:
        self.remove_unsharded_checkpoint(idx=idx)

def _remove_sharded_checkpoint(self, idx: int, checkpoints: List[Path]):
    oldest_checkpoint = checkpoints.pop(idx)
    barrier()  # Sync across ranks
    if get_fs_local_rank() == 0 and oldest_checkpoint.is_dir():
        shutil.rmtree(oldest_checkpoint, ignore_errors=True)
    # Update symlink
    latest_path = Path(self.cfg.save_folder) / "latest"
    if latest_path.resolve() == oldest_checkpoint.resolve():
        latest_path.unlink()
    barrier()
```

## Restore Checkpoint

```python
def restore_checkpoint(
    self,
    load_path: PathOrStr,
    *,
    checkpoint_type: Optional[CheckpointType] = None,
    load_optimizer_state: bool = True,
    load_trainer_state: bool = True,
):
    # Zero gradients to avoid gathering them
    self.optim.zero_grad(set_to_none=True)
    
    checkpointer = build_sharded_checkpointer(self.cfg)
    trainer_state = checkpointer.restore_checkpoint(
        load_path,
        self.dist_model,
        self.optim,
        load_optimizer_state=load_optimizer_state,
    )
    
    if load_trainer_state:
        self.load_trainer_state_dict(trainer_state)
    
    barrier()
```

## Comparison to SLM

| Feature | OLMo | SLM |
|---------|------|-----|
| Checkpoint types | Sharded/Unsharded/Ephemeral | Single type |
| Save frequency | Every 1000 steps | End of phase |
| Optimizer state | Saved by default | Saved |
| Trainer state | Full (tokens, best loss) | Basic (step, epoch) |
| Cleanup | Automatic with config | Manual |
| HF conversion | Separate script | Built into save |

## Relevant Patterns for SLM

1. **Trainer state** — Track tokens seen, best eval loss, scheduler state
2. **Latest symlink** — Point to most recent checkpoint
3. **Barrier sync** — Ensure all ranks finish saving before continuing
4. **GC CUDA** — Call `gc_cuda()` after saving to free memory
5. **Zero grad before restore** — Avoid gradient gathering issues

## HF Conversion

OLMo converts to HuggingFace format separately:

```bash
python scripts/convert_to_hf.py \
    --checkpoint_path checkpoints/step500000 \
    --output_path hf_checkpoints/olmo2-1b
```

## Citation

See OLMo 2 paper for checkpoint and fault tolerance details.
