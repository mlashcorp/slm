# Phase 5 Canary Run

**Objective:** Establish baseline for 124M Python coding model using nanochat architecture with mixed data (python-edu + climbmix).

## Configuration

| Parameter | Value |
|-----------|-------|
| **Model** | Nanochat 124M (depth=8, context=1024) |
| **Data** | ClimbMix-only (1 shard for validation) |
| **Steps** | 300 |
| **Batch size** | 2048 tokens |
| **Token budget** | ~630M tokens |
| **Save interval** | Every 100 steps |
| **GPU** | RTX 5090 (32GB VRAM) |

## Data Strategy

Both datasets are downloaded **directly on the remote instance** from HuggingFace:
- **python-edu**: From `HuggingFaceTB/smollm-corpus` (python code)
- **climbmix**: From nanochat's built-in downloader (general text)

The dataloader iterates through all parquet files in the unified directory, effectively creating a 50/50 mixture by shard count.

**Schema requirement:** Both sources use the same schema - single `text` column in parquet format.

## Files

```
phase5/canary/
├── README.md                      # This file
├── vast_canary_config.yaml        # Vast.ai instance config
├── remote_nanochat_canary.sh      # Main training script (runs on remote)
├── remote_download_data.sh        # Data downloader (runs on remote)
└── launch_canary.sh               # Local launcher
```

## How to Run

```bash
# From local machine
bash phase5/canary/launch_canary.sh
```

This will:
1. Search for available GPU on Vast.ai (RTX 5090, ~$0.40-0.50/hr)
2. Launch instance and upload code
3. Download ClimbMix data directly on remote (~5-10 min)
4. Train tokenizer on ClimbMix data
5. Run 300 training steps
6. Push checkpoints to HuggingFace Hub (if HF_TOKEN set)

## Evaluation

**During training (logged to W&B):**
- Every 100 steps: `val_bpb` (general), `python_val_bpb`
- Every 200 steps: CORE metric

**Final checkpoint:**
- HumanEval pass@1
- MBPP pass@1
- SAFIM score (syntax-aware FIM)
- NaturalCodeBench score (real queries)

## Monitoring

- **W&B:** https://wandb.ai/mlashcorp/slm-phase5-canary
- **Checkpoints:** https://huggingface.co/mlashcorp/slm-phase5-checkpoints

## Expected Runtime & Cost

- **Data download:** ~5-10 minutes
- **Training:** ~30-60 minutes (300 steps)
- **Total cost:** ~$0.20-0.30 (at $0.40-0.50/hr, RTX 5090)

## Next Steps After Canary

1. Review metrics in W&B
2. Compare against targets:
   - `python_val_bpb` < 2.0
   - `val_bpb` stable (no catastrophic forgetting)
   - HumanEval pass@1 > 0% (sanity check)
3. Proceed to Stage 2 mixture sweep (50/50, 65/35, 80/20, 90/10)

## Troubleshooting

**Data download fails:**
- Check HuggingFace API access
- Verify disk space on remote (need ~2GB for both datasets)

**Training OOM:**
- Reduce `--device-batch-size` in training script
- Reduce `--max-seq-len`

**Checkpoints not uploading:**
- Verify HF_TOKEN is set in `.env`
- Check HuggingFace repo permissions
