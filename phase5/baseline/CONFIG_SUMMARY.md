# Phase 5 Baseline Configuration Summary

## Quick Start

```bash
# Launch the 124M compute-optimal baseline
bash phase5/baseline/launch_baseline.sh
```

## What Changed

### GPU Upgrade: RTX 4090 → RTX 5090
- **RTX 4090:** 24GB VRAM, ~$0.35/hr
- **RTX 5090:** 32GB VRAM, ~$0.40-0.50/hr
- **Benefit:** More VRAM headroom, faster training (~80-100K tok/sec vs ~60-80K)

### Model Size Correction
- **Previous plan:** depth=8 (~35M params, miscalculated)
- **Corrected:** depth=24 (~124M params)
- **Why:** Nanochat auto-calculates dimensions from depth

### Data Strategy Simplified
- **Previous:** 50/50 python-edu/climbmix (python-edu metadata-only, not usable)
- **Current:** ClimbMix-only (25 shards, has ~10-20% Python content)
- **Why:** python-edu from smollm-corpus is metadata-only, requires S3 download script
- **Future:** Add dedicated Python data in Stage 2 mixture sweep

### Compute-Optimal Token Budget
- **Target:** 12× parameter-to-data ratio (nanochat default)
- **124M params × 12 = 1.49B tokens**
- **25 ClimbMix shards = ~1.53B tokens**
- **Training steps:** ~715 (auto-calculated by nanochat)

## Configuration Files

### Canary (Quick Validation)
```
phase5/canary/
├── vast_canary_config.yaml       # RTX 5090 config
├── remote_nanochat_canary.sh     # 300-step training
└── launch_canary.sh              # Launcher
```

### Baseline (Full Training)
```
phase5/baseline/
├── vast_baseline_config.yaml      # RTX 5090 config
├── remote_nanochat_baseline.sh    # 715-step training (NEW)
├── launch_baseline.sh             # Launcher (NEW)
└── README.md                      # Documentation
```

## Training Comparison

| Aspect | Canary | Baseline |
|--------|--------|----------|
| **GPU** | RTX 5090 | RTX 5090 |
| **Model** | depth=8 (~35M) | depth=24 (~124M) |
| **Data** | ClimbMix (1 shard) | ClimbMix (25 shards) |
| **Tokens** | ~61M | ~1.53B |
| **Steps** | 300 | ~715 |
| **Time** | ~30 min | ~5-6 hours |
| **Cost** | ~$0.25 | ~$2.33 |
| **Purpose** | Pipeline validation | Compute-optimal baseline |

## Expected Outcomes

### Canary (300 steps, depth=8)
- Validate data download works
- Validate training pipeline
- Check for OOM or divergence
- Quick feedback loop (~30 min)

### Baseline (715 steps, depth=24)
- **val_bpb:** < 2.0 on ClimbMix validation
- **CORE metric:** > 0.15 (small model baseline)
- **Training stability:** No divergence, smooth loss curve
- **Checkpoint quality:** Usable for downstream tasks

## Next Steps

1. **Run canary first** (optional, validates pipeline)
   ```bash
   bash phase5/canary/launch_canary.sh
   ```

2. **Launch baseline** (main experiment)
   ```bash
   bash phase5/baseline/launch_baseline.sh
   ```

3. **Monitor training**
   - W&B: https://wandb.ai/slm-phase5-baseline
   - Wait ~5-6 hours for completion

4. **Evaluate results**
   - Check val_bpb trajectory
   - Review CORE metric
   - Compare against targets

5. **Plan Stage 2 experiments**
   - Data mixture sweep (Python % optimization)
   - Architecture enhancements (AST-FIM, CAG, TOP)
   - Synthetic data augmentation

## Cost Tracking

| Run | GPU | Duration | Rate | Cost |
|-----|-----|----------|------|------|
| Canary | RTX 5090 | 0.5 hr | $0.45/hr | $0.23 |
| Baseline | RTX 5090 | 5.8 hr | $0.40/hr | $2.33 |
| **Total** | | **6.3 hr** | | **~$2.56** |

## Troubleshooting

### python-edu not available
The `smollm-corpus` python-edu subset only contains metadata, not actual code content. We're using ClimbMix which has ~10-20% Python content mixed in.

### Want to add Python data?
Options for Stage 2:
1. Download python-edu from Software Heritage S3 (requires finding download script)
2. Use codeparrot/github-code (requires HF_TOKEN, ~7M Python files)
3. Use bigcode/the-stack (requires HF_TOKEN, gated dataset)

### Training diverges
- Check learning rate scaling
- Verify batch size
- Review val_bpb trajectory (should decrease monotonically)
- Reduce depth if needed (try depth=16 or depth=12)

## References

- [Nanochat Repository](https://github.com/karpathy/nanochat)
- [ClimbMix-400B Dataset](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle)
- [Nanochat Speedrun](https://github.com/karpathy/nanochat/blob/main/runs/speedrun.sh)
