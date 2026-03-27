# Phase 0 Scratchpad: NanoChat GPT-2 Reproduction

## Goal

Recreate Karpathy's nanochat approach to achieve GPT-2 capability (CORE score > 0.26) in ~3-5 hours on RTX 5090.

## Key Decisions

1. ✅ **New files** - Clean separation from Phase 1-3
2. ✅ **New tokenizer** - Train on DCLM (vocab=32768)
3. ✅ **Validate with d12** - Quick 100-step test before full run
4. ✅ **FP8 training** - Enable for speed
5. ✅ **Same W&B project** - Use `phase0_d{depth}` prefix

## Architecture Comparison

| Feature | Phase 1 (SmolLM2) | Phase 0 (NanoChat) |
|---------|-------------------|-------------------|
| Activation | GELU | ReLU² |
| QK Normalization | No | Yes (RMSNorm) |
| Value Embeddings | No | Yes (ResFormer-style) |
| Sliding Window | No | Yes (SSSL pattern) |
| Tied Embeddings | Yes | No (untied) |
| Norm After Embed | No | Yes |
| Per-layer Scalars | No | Yes (resid_lambdas, x0_lambdas) |
| Smear Gate | No | Yes (bigram-like) |
| Backout | No | Yes |
| Softcap Logits | No | Yes (±15) |
| Optimizer | AdamW | MuonAdamW |
| Precision | BF16 | FP8 (optional) |

## Model Sizes

| Config | Depth | Hidden | Heads | Params | Training Tokens | Target |
|--------|-------|--------|-------|--------|-----------------|--------|
| d12 | 12 | 384 | 6 | ~12M | ~100M | Quick validation |
| d24 | 24 | 512 | 4 | ~50M | 2-5B | GPT-2 level |
| d26 | 26 | 640 | 5 | ~60M | 3-5B | Beat GPT-2 |

## Training Configuration

### Hyperparameters (from nanochat)

```yaml
# Optimizer - AdamW params
unembedding_lr: 0.008
embedding_lr: 0.2
scalar_lr: 0.5

# Optimizer - Muon params
matrix_lr: 0.02
momentum: 0.95
weight_decay: 0.28

# Training
warmup_steps: 40
warmdown_ratio: 0.65
final_lr_frac: 0.05
target_param_data_ratio: 8

# Compute
fp8: true
compile: true
```

### Learning Rate Schedule

```
Phase 1: Warmup (40 steps)
  LR: 0 → peak (linear)

Phase 2: Stable (remaining steps - warmdown)
  LR: peak (constant)

Phase 3: Warmdown (65% of training from end)
  LR: peak → peak * 0.05 (linear)
```

### Momentum Schedule (Muon)

```
Steps 0-400: 0.85 → 0.97 (warmup)
Steps 400-warmdown: 0.97 (stable)
Warmdown: 0.97 → 0.90 (decay)
```

## File Structure

```
slm/
├── src/
│   ├── model/
│   │   ├── nanochat_config.py    # Config (d12, d24, d26)
│   │   └── nanochat_gpt.py       # Model implementation
│   ├── training/
│   │   ├── fp8.py                 # FP8 Linear layer
│   │   └── muon_optimizer.py      # MuonAdamW optimizer
│   └── data/
│       └── bos_bestfit.py         # BOS-aligned best-fit loader
├── scripts/
│   ├── train_phase0.py            # Training script
│   ├── eval_core.py               # CORE metric evaluation
│   └── download_dclm.py           # DCLM download
├── data/
│   └── dclm/                      # DCLM dataset
└── checkpoints/
    └── phase0/
        └── d{depth}/
```

## Dataset: DCLM

### What is DCLM?

Data Comp for Language Models (DCLM) is a high-quality pretraining dataset used by nanochat.

### Download

```bash
# Download 8 shards for tokenizer training (~2B chars)
python scripts/download_dclm.py -n 8

# Download 170 shards for full training (~42B chars)
python scripts/download_dclm.py -n 170
```

### Data Loading: BOS-Aligned Best-Fit

Key properties:
- Every sequence starts with BOS token
- Best-fit cropping: find largest doc that fits
- 100% utilization (no padding)
- ~35% tokens cropped (but all trained on)

Algorithm:
```
for each row:
    while pos < row_capacity:
        # Find largest doc that fits
        best = max(docs, key=len) if len(max) <= remaining
        if best:
            row[pos:pos+len(best)] = best
        else:
            # Crop shortest to fill
            row[pos:] = shortest[:remaining]
```

## FP8 Training

### How it Works

FP8 training uses 8-bit floating point for matmuls:

1. **Forward**: `output = input @ weight.T`
   - Quantize input and weight to `float8_e4m3fn` (high precision)
   - Call `torch._scaled_mm` (cuBLAS FP8 kernel)
   - Dequantize output

2. **Backward**: Two matmuls
   - `grad_input = grad_output @ weight` (e5m2 for gradients)
   - `grad_weight = grad_output.T @ input`

### FP8 Data Types

| Type | Exponent | Mantissa | Range | Usage |
|------|----------|----------|-------|-------|
| `float8_e4m3fn` | 4 bits | 3 bits | [-448, 448] | Input, weight |
| `float8_e5m2` | 5 bits | 2 bits | [-57344, 57344] | Gradients |

### Hardware Support

- **H100+ (Hopper)**: Native FP8 support
- **RTX 5090 (Blackwell)**: Native FP8 support (SM 100+)
- **Fallback**: Use BF16 if FP8 not available

## Muon Optimizer

### Key Components

1. **Nesterov Momentum**: Smooth gradients
2. **Polar Express**: Orthogonalize update (5 Newton-Schulz iterations)
3. **Variance Reduction**: Per-neuron adaptive LR
4. **Cautious Weight Decay**: Only when gradient and param have same sign

### Parameter Groups

```python
param_groups = [
    # AdamW: embeddings, scalars
    dict(kind='adamw', params=lm_head, lr=0.008),
    dict(kind='adamw', params=embeddings, lr=0.2),
    dict(kind='adamw', params=scalars, lr=0.5),

    # Muon: matrix params (same shape for stacking)
    dict(kind='muon', params=matrices, lr=0.02, momentum=0.95),
]
```

### Why Muon?

- Faster convergence for transformers
- Better scaling properties
- Used in nanochat's GPT-2 reproduction

## CORE Metric

### What is CORE?

DCLM benchmark suite for evaluating model quality:
- HellaSwag
- PIQA
- ARC (Easy and Challenge)
- Winogrande
- MMLU

### Baseline Scores

| Model | CORE Score | Parameters |
|-------|------------|------------|
| GPT-2 (1.5B) | 0.2565 | 1.5B |
| NanoChat d24 | 0.2585 | ~50M |
| NanoChat d26 | 0.2626 | ~60M |

### Evaluation

```bash
# Full evaluation with lm-eval
lm_eval --model hf --model_args path/to/model --tasks hellaswag,piqa,arc_easy,arc_challenge,winogrande,mmlu

# Custom evaluation
python scripts/eval_core.py --checkpoint checkpoints/phase0/d24/final.pt
```

## Quick Start

### 1. Install Dependencies

```bash
pip install tiktoken lm-eval
```

### 2. Download Data

```bash
python scripts/download_dclm.py -n 170
```

### 3. Quick Validation (d12, 100 steps)

```bash
python scripts/train_phase0.py \
    --depth 12 \
    --num-iterations 100 \
    --batch-size 4 \
    --wandb
```

### 4. Full Training (d24)

```bash
python scripts/train_phase0.py \
    --depth 24 \
    --fp8 \
    --compile \
    --batch-size 16 \
    --wandb
```

## Expected Timeline

| Phase | Tokens | Time (RTX 5090) | CORE Target |
|-------|--------|-----------------|-------------|
| Validation (d12) | 100M | ~5 min | N/A |
| d24 (2B tokens) | 2B | ~5 hours | >0.20 |
| d24 (5B tokens) | 5B | ~12 hours | >0.26 |

## Known Issues

### 1. Memory Error with torch.compile

The compiled kernels may cause memory corruption on some systems.

**Workaround**: Disable compile
```bash
python scripts/train_phase0.py --depth 12 --no-compile
```

### 2. FP8 Not Supported

**Error**: "FP8 not supported"

**Solution**: Ensure RTX 5090 or H100+ GPU. Fallback to BF16.

### 3. Dataset Not Found

**Error**: "Data directory not found"

**Solution**: Download DCLM first
```bash
python scripts/download_dclm.py -n 8
```

## Success Criteria

| Milestone | Target | Verification |
|-----------|--------|--------------|
| Model compiles | No errors | `python -c "from src.model.nanochat_gpt import GPT"` |
| Forward pass | Output shape correct | Test script |
| Optimizer step | Loss decreases | 10-step test |
| d12 validation | Loss < 3.0 in 100 steps | W&B |
| d24 full run | CORE > 0.26 | lm-eval |

## Reference Links

- **NanoChat repo**: https://github.com/karpathy/nanochat
- **NanoChat paper**: arXiv:2502.02737 (SmolLM2)
- **Muon optimizer**: https://kellerjordan.github.io/posts/muon/
- **DCLM dataset**: https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0-parquet
- **Polar Express**: https://arxiv.org/pdf/2505.16932

## Notes

- The NanoChat architecture is significantly different from SmolLM2
- Key innovations: QK norm, value embeddings, smear gate, backout
- FP8 training requires H100+ or RTX 5090 (Blackwell)
- Muon optimizer is critical for fast convergence
- Training with data:param ratio of 8 (vs 20 Chinchilla optimal)

## Next Steps

1. Test on RTX 5090 with CUDA
2. Debug any torch.compile issues
3. Run d12 validation (100 steps)
4. If successful, proceed to d24 full training
5. Evaluate CORE metric and compare to nanochat leaderboard
