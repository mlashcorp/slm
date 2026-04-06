# Phase 5 Baseline — nanochat 124M on ClimbMix

**Objective:** Establish a compute-optimal baseline using the nanochat architecture (depth=12) trained on ClimbMix-400B.

## Active Run

| Field | Value |
|-------|-------|
| **Instance** | Vast.ai RTX 3090 (`34238966`) |
| **SSH** | `ssh -p 38966 root@ssh9.vast.ai` |
| **W&B** | https://wandb.ai/mlashcorp/nanochat/runs/xljut36a |
| **Run ID** | `phase5_baseline_124m_d12` |
| **Status** | Training — ~26% complete at step 187/715 |

## Configuration

| Parameter | Value |
|-----------|-------|
| **Model** | nanochat depth=12 (~124M params) |
| **Data** | ClimbMix-400B — 25 train shards + 1 val (~1.53B tokens) |
| **Steps** | 715 (`--num-iterations 715`) |
| **Batch size** | 2,048 tokens (`--total-batch-size 2048`) |
| **Context length** | 1,024 tokens |
| **Vocab size** | 8,192 (BPE, trained on ClimbMix) |
| **Window pattern** | `L` (full attention, no sliding window) |
| **torch.compile** | disabled (`DISABLE_TORCH_COMPILE=1`) |
| **GPU** | RTX 3090 24GB @ ~$0.11/hr |

## Model Architecture (depth=12)

| Param | Value |
|-------|-------|
| Layers | 12 |
| Model dim | 768 |
| Attention heads | 6 |
| KV heads | 6 |
| ~Parameters | ~124M |
| FLOPs/token | ~1.07e+09 |

**Note:** depth=24 produces ~856M params (not 124M). depth=12 is the correct choice for ~124M scale, matching GPT-2 small. Nanochat auto-calculates all other dimensions from depth.

## Data

**Dataset:** `karpathy/climbmix-400b-shuffle`
- 25 train shards + 1 validation shard (shard 6542)
- ~1.53B tokens total (25 × ~61M tokens/shard)
- ~2.3GB on disk
- Contains mixed web text with ~10-20% Python content
- Downloaded directly on remote — no R2 or local disk required

**Why ClimbMix only (for now)?**
- `smollm-corpus` python-edu is metadata-only; actual code requires Software Heritage S3 hydration
- ClimbMix has enough Python to establish a meaningful baseline
- Dedicated Python data will be added in the Stage 2 mixture sweep

## Evaluation Schedule

| Frequency | Metric |
|-----------|--------|
| Every 100 steps | `val_bpb` (validation bits-per-byte) |
| Every 200 steps | CORE metric (DCLM benchmark) |
| Final checkpoint | Full eval: HumanEval, MBPP, SAFIM, NaturalCodeBench |

## Observed Performance

| Metric | Value |
|--------|-------|
| **Throughput** | ~20K tok/sec (RTX 3090, no compile, SDPA) |
| **MFU** | ~18-19% (SDPA fallback — no FlashAttention 3) |
| **Loss at step 0** | 3.265 bpb |
| **Loss at step ~185** | ~6.11 (cross-entropy, warmup complete) |
| **GPU power** | ~340W / 350W |

**Note on MFU:** The RTX 3090 does not support FlashAttention 3. Nanochat falls back to PyTorch SDPA which gives ~18% MFU vs ~45% with FA3. For future runs, use an H100 or upgrade to a host with FA3 support.

## Files

```
phase5/baseline/
├── README.md                           # This file
├── vast_baseline_config.yaml           # Vast.ai instance config (RTX 3090)
├── remote_nanochat_baseline.sh         # Training script (runs on remote)
├── launch_baseline.sh                  # Local launcher
└── CONFIG_SUMMARY.md                   # Quick reference
```

## How to Run

```bash
# From local machine (loads credentials from .env)
bash phase5/baseline/launch_baseline.sh
```

The launcher will:
1. Search for RTX 3090 on Vast.ai, skipping broken CDI hosts
2. rsync code to `/workspace/slm`
3. Download 25 ClimbMix shards on remote (~5-10 min)
4. Train tokenizer on ClimbMix data (~1 min)
5. Run 715 training steps (~60 min on RTX 3090)
6. Save checkpoint at step 500 and final

## Known Issues / Lessons Learned

| Issue | Fix Applied |
|-------|-------------|
| `smollm-corpus` python-edu has no `text` column | Use ClimbMix only for baseline |
| `depth=24` → ~856M params (not 124M) | Use `depth=12` for ~124M |
| `torch.compile` hangs on this host | `DISABLE_TORCH_COMPILE=1` |
| `window_pattern=SSSL` breaks without FA3 | `--window-pattern L` |
| CDI device injection fails on some Vast.ai hosts | `launch.py` now retries up to 20 offers, fast-fails on CDI error |
| W&B API key lost in `nohup` subshell | Run `wandb login <key>` on remote; persists via `~/.netrc` |
| `NANOCHAT_BASE_DIR` not exported to subshell | Set explicitly in every `nohup`/`tmux` command |

## Cost

| Component | Duration | Rate | Cost |
|-----------|----------|------|------|
| Data download + tokenizer | ~15 min | $0.11/hr | ~$0.03 |
| Training (715 steps) | ~60 min | $0.11/hr | ~$0.11 |
| **Total** | **~75 min** | **$0.11/hr** | **~$0.14** |

## Next Steps After Baseline

1. Review W&B: loss curve, val_bpb, CORE metric
2. Run final evals: HumanEval, MBPP, SAFIM, NaturalCodeBench
3. Stage 2 — data mixture sweep:
   - Add python-edu (hydrate from Software Heritage S3 via `scripts/download_python_edu.py`)
   - Sweep ratios: 100/0, 80/20, 65/35, 50/50 Python/ClimbMix
4. Stage 3 — architecture experiments (AST-FIM, CAG, TOP)

## References

- [nanochat](https://github.com/karpathy/nanochat)
- [ClimbMix-400B](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle)
- [nanochat speedrun.sh](https://github.com/karpathy/nanochat/blob/main/runs/speedrun.sh)
- [Chinchilla scaling laws](https://arxiv.org/abs/2203.15556)
