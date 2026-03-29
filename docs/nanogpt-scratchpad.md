# Phase 0: Pure nanoGPT Implementation

## Goal

Train a GPT-2 (124M) model from scratch using Karpathy's nanoGPT implementation on OpenWebText dataset.

## Reference

**Base**: https://github.com/karpathy/nanogpt - The original minimal GPT-2 reimplementation

---

## Implementation

This is a **1:1 copy** of Karpathy's nanoGPT, with no modifications:

| Feature | Implementation |
|---------|----------------|
| Norm | LayerNorm (learnable weight + bias) |
| Position | Absolute (wpe embedding) |
| Activation | GELU |
| Embeddings | Tied (wte = lm_head) |
| Attention | Fused QKV (c_attn) |
| Bias | Optional (config.bias) |
| Dropout | Optional (config.dropout) |
| Optimizer | AdamW (fused on CUDA) |
| LR Schedule | Cosine with warmup |
| Weight Decay | 0.1 (2D params only) |
| Gradient Clipping | 1.0 |
| Data | Binary memmap (train.bin, val.bin) |

---

## File Structure

```
slm/
├── src/model/
│   └── nanogpt.py          # Pure nanoGPT model
├── data/openwebtext/
│   ├── prepare.py          # Data preparation script
│   ├── train.bin           # Training tokens (memmap)
│   ├── val.bin             # Validation tokens (memmap)
│   └── meta.pkl            # Metadata (vocab_size, tokenizer)
├── config/
│   ├── train_gpt2.py       # Full training config (600K steps)
│   └── train_gpt2_quick.py # Quick test config (10K steps)
├── scripts/
│   └── train_nanogpt.py    # Training script
└── checkpoints/nanogpt/
    └── ckpt.pt             # Checkpoints
```

---

## Quick Start

### 1. Prepare Data

```bash
python data/openwebtext/prepare.py
```

This will:
- Download OpenWebText from HuggingFace
- Tokenize with GPT-2 BPE (tiktoken)
- Split into train/val
- Save as binary memmap files

### 2. Quick Test (10K steps, ~2-4 hours)

```bash
python scripts/train_nanogpt.py config/train_gpt2_quick.py
```

### 3. Full Training (600K steps, ~weeks)

```bash
python scripts/train_nanogpt.py config/train_gpt2.py
```

---

## Model Sizes

| Config | n_layer | n_head | n_embd | Params |
|--------|---------|--------|--------|--------|
| GPT-2 | 12 | 12 | 768 | 124M |
| GPT-2 Medium | 24 | 16 | 1024 | 350M |
| GPT-2 Large | 36 | 20 | 1280 | 774M |
| GPT-2 XL | 48 | 25 | 1600 | 1558M |

---

## Hyperparameters (GPT-2)

```yaml
# Model
n_layer: 12
n_head: 12
n_embd: 768
dropout: 0.0
bias: true

# Training
batch_size: 64
block_size: 1024
gradient_accumulation_steps: 1

# Optimizer
learning_rate: 6e-4
weight_decay: 0.1
beta1: 0.9
beta2: 0.95
grad_clip: 1.0

# Schedule
warmup_iters: 2000
max_iters: 600000
lr_decay_iters: 600000
min_lr: 6e-5

# Eval
eval_interval: 1000
eval_iters: 200
log_interval: 10
```

---

## Expected Results

| Run | Tokens | Target Val Loss | Time (RTX 5090) |
|-----|--------|-----------------|-----------------|
| Quick test | ~655M | ~4.0 | ~2-4 hours |
| Full training | ~39B | ~2.85 | ~weeks |

Note: Original GPT-2 training used 300B tokens on 8x A100 for ~4 days.

---

## W&B Project

- Project: `nanogpt`
- Run names: `gpt2-124M`, `gpt2-124M-quick`
