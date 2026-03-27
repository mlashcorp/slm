# Phase 0 Scratchpad: NanoChat GPT-2 Reproduction

## Goal

Recreate Karpathy's nanogpt approach with modern improvements to achieve GPT-2 capability (CORE score > 0.26) in ~3-5 hours on RTX 5090.

## Reference

**Base**: https://github.com/karpathy/nanogpt — the original minimal GPT-2 reimplementation this builds on
**Muon optimizer**: https://kellerjordan.github.io/posts/muon/
**Polar Express**: https://arxiv.org/pdf/2505.16932
**Dataset**: `data/python_edu_hydrated_train_split` (7.6M rows, converted to parquet)

---

## Architecture: What We Changed vs karpathy/nanogpt

karpathy/nanogpt is a clean GPT-2 baseline: LayerNorm, absolute position embeddings, GELU, tied weights, AdamW.
Below is what we changed and added on top.

| Feature | karpathy/nanogpt | Ours | Status |
|---------|-----------------|------|--------|
| Norm type | LayerNorm (learned) | RMSNorm (no params) | ✅ |
| Position encoding | Absolute (wpe) | RoPE | ✅ |
| Activation | GELU | ReLU² | ✅ |
| Tied embeddings | Yes | No (untied) | ✅ |
| Norm after embed | No | Yes | ✅ |
| QK normalization | No | Yes (RMSNorm on Q, K) | ✅ |
| Sliding window attn | No | Yes (SSSL pattern) | ✅ |
| Value embeddings | No | Yes (ResFormer-style) | ✅ |
| VE gate | — | `2 * sigmoid(linear(cat([x[:6], ve[:6]])))` | ✅ |
| Per-layer resid_lambdas | No | `(L, 2)` — attn + MLP | ✅ |
| Per-layer post_lambdas | No | `(L, 2)` — attn + MLP | ✅ |
| Per-layer x0_lambdas | No | `(L,)` — blend initial embed | ✅ |
| Per-layer sa_lambdas | No | `(L, 2)` — scale QKV + O | ✅ |
| Smear gate | No | Yes (bigram-like mixing) | ✅ |
| Attention output gate | No | `y * sigmoid(linear(x[:12]))` per head | ✅ |
| Skip connection | No | Layer 3 → 6, layer 6 skips attn | ✅ |
| Bigram embeddings | No | Hash-based, injected per layer | ✅ |
| Backout | No | Subtract cached mid-stream at ~64% depth | ✅ |
| Softcap logits | No | `23 * sigmoid((logits + 5) / 7.5)` | ✅ |
| Optimizer | AdamW | MuonAdamW (Muon for matrices) | ✅ |
| Adam odd-steps | No | Adam only on odd steps | ✅ |
| Polar Express cushion | — | `1 + 2e-2` | ✅ |
| Dropout | Yes (0.0 for pretrain) | No | ✅ |
| Vocab size | 50304 | 50304 (r50k_base) | ✅ |

### Deferred (Phase 5)

| Feature | Notes |
|---------|-------|
| YaRN RoPE | Half-truncated, base `(1/1024)^linspace`, `attn_scale=0.1` |
| Paired head layers | Adjacent heads attend to interleaved doubled sequences |
| MTP | Multi-token prediction heads with `[1.0, 0.5, 0.25]` weights |
| Mantissa tracking | uint16 mantissa buffer for BF16 precision in Muon |

---

## Implementation Status

All Phase 1–4 items from the plan are complete. End-to-end smoke test passes.

```
✅ Bug: tokenizer cl100k_base → r50k_base, vocab_size 32768 → 50304
✅ Bug: doc_iter nonlocal in refill_buffer()
✅ Bug: initial_lr stored after optimizer creation
✅ Bug: --compile / --wandb use BooleanOptionalAction
✅ Arch: VE gate uses cat([x[:6], ve[:6]]) with 2× multiplier
✅ Arch: backout_layer = round(depth * 7/11)
✅ Arch: resid_lambdas (L,2), post_lambdas (L,2)
✅ Arch: logit softcap = 23 * sigmoid((x+5)/7.5)
✅ Arch: Polar Express cushion 1.02
✅ Feature: sa_lambdas
✅ Feature: attention output gate (init zeros)
✅ Feature: skip connection + attention-free layer
✅ Feature: bigram embeddings
✅ Training: Adam odd-steps-only
✅ Training: final_lr_frac 0.15
✅ Training: momentum peak 0.95, warmup 300 steps, cooldown to 0.85
```

---

## Training Configuration

```yaml
# Optimizer — AdamW
unembedding_lr: 0.008
embedding_lr:   0.2
scalar_lr:      0.5

# Optimizer — Muon
matrix_lr:      0.02
momentum:       0.95         # warms up 0.85→0.95 over 300 steps
weight_decay:   0.28
beta2:          0.9

# Schedule
warmup_steps:   40
warmdown_ratio: 0.65
final_lr_frac:  0.15         # final LR = 15% of peak
```

---

## Model Sizes

| Config | Depth | Hidden | Heads | Params | Training Tokens | Target |
|--------|-------|--------|-------|--------|-----------------|--------|
| d12 | 12 | 384 | 6 | ~12M | ~100M | Quick validation |
| d24 | 24 | 512 | 4 | ~50M | 2-5B | GPT-2 level |
| d26 | 26 | 640 | 5 | ~60M | 3-5B | Beat GPT-2 |

---

## File Structure

```
slm/
├── src/
│   ├── model/
│   │   ├── nanochat_config.py
│   │   └── nanochat_gpt.py
│   ├── training/
│   │   ├── fp8.py
│   │   └── muon_optimizer.py
│   └── data/
│       └── bos_bestfit.py
├── scripts/
│   ├── train_phase0.py
│   ├── eval_core.py
│   └── download_dclm.py
├── data/
│   ├── dclm/                         # parquet shards for training
│   ├── python_edu_hydrated_train_split/
│   └── python_edu_hydrated_val_split/
└── checkpoints/
    └── phase0/d{depth}/
```

---

## Data

Currently using `python_edu` converted to parquet (`data/dclm/`).
For a proper pretraining run, download DCLM:

```bash
python scripts/download_dclm.py -n 170   # ~42B chars, 170 shards
```

---

## Quick Start

```bash
# Smoke test (10 steps)
.venv/bin/python scripts/train_phase0.py \
    --depth 12 --num-iterations 10 --batch-size 2 \
    --no-wandb --no-compile --data-dir data/dclm

# Validation (d12, 100 steps)
.venv/bin/python scripts/train_phase0.py \
    --depth 12 --num-iterations 100 --batch-size 4 \
    --no-wandb --data-dir data/dclm

# Full training (d24)
.venv/bin/python scripts/train_phase0.py \
    --depth 24 --fp8 --batch-size 16 \
    --wandb --data-dir data/dclm
```

---

## CORE Metric

Evaluated on: HellaSwag, PIQA, ARC-Easy, ARC-Challenge, Winogrande, MMLU

| Model | CORE | Params |
|-------|------|--------|
| GPT-2 (1.5B) | 0.2565 | 1.5B |
| Target d24 | >0.26 | ~50M |

```bash
python scripts/eval_core.py --checkpoint checkpoints/phase0/d24/final.pt
```

---

## Expected Timeline (RTX 5090)

| Run | Tokens | Time | Purpose |
|-----|--------|------|---------|
| d12, 100 steps | ~400K | <5 min | Sanity check |
| d24, 2B tokens | 2B | ~5 hours | First CORE eval |
| d24, 5B tokens | 5B | ~12 hours | Target CORE > 0.26 |
