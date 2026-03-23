# Architecture Design

> **Goal:** Train a small language model specialized on Python code, using two phases:
> a fast iteration/proxy model (~360M) to ablate decisions, and a full target model (~2B).
>
> **Hardware:** NVIDIA RTX 5090, 32 GB VRAM, single GPU.

---

## Table of Contents

1. [Project Scope](#1-project-scope)
2. [Why Python-Only](#2-why-python-only)
3. [Phase 1 — Iteration Model (~360M)](#3-phase-1--iteration-model-360m)
4. [Phase 2 — Target Model (~2B)](#4-phase-2--target-model-2b)
5. [Component Rationale](#5-component-rationale)
   - [Attention: GQA](#attention-grouped-query-attention-gqa)
   - [Activation: SwiGLU](#activation-swiglu)
   - [Normalization: RMSNorm](#normalization-rmsnorm-pre-norm)
   - [Positional Encoding: RoPE](#positional-encoding-rope)
   - [Training Objective: FIM](#training-objective-fill-in-middle-fim)
   - [Tokenizer: Custom BPE](#tokenizer-custom-bpe)
   - [Tied Embeddings](#tied-embeddings)
6. [VRAM Budget](#6-vram-budget)
7. [Comparison with Existing Models](#7-comparison-with-existing-models)
8. [References](#8-references)

---

## 1. Project Scope

This project trains a Python code language model from scratch using a two-phase approach drawn from the SmolLM3 training playbook:

- **Phase 1** runs fast ablations on a ~360M proxy model (1/6 of target size, ~0.5–1% of target token budget). Every architecture, data, and hyperparameter decision is validated here before committing to Phase 2.
- **Phase 2** trains the target ~2B model applying all validated decisions. This is the maximum size trainable from scratch on 32 GB VRAM with the required memory tricks.

Both phases use the **same architecture family** (Llama-style decoder-only transformer). Only scale differs, so ablation findings transfer directly.

> **Key principle (from SmolLM3 playbook):** "Successful LLM training is 70% data curation, 20% iteration speed, and 10% architectural innovation." Architecture is deliberately conventional — the investment is in data quality.

---

## 2. Why Python-Only

**The phi-1 result** ([arXiv:2306.11644](https://arxiv.org/abs/2306.11644)) is the primary motivation. Microsoft trained a 1.3B model on ~7B tokens of curated + synthetic Python data and achieved **50.6% HumanEval pass@1**, outperforming multilingual code models trained on 100–700× more tokens:

| Model | Params | HumanEval pass@1 | Training tokens |
|---|---|---|---|
| **phi-1** | **1.3B** | **50.6%** | **~50B effective (7B unique)** |
| phi-1-small | 350M | 45.0% | ~50B effective |
| Qwen2.5-Coder-1.5B | 1.5B | 43.9% | 5.5T |
| DeepSeek-Coder-1.3B | 1.3B | 34.8% | 2T |
| StarCoder2-3B | 3B | 31.7% | 3.3T |
| CodeGemma-2B | 2B | 31.1% | +500B continued pretraining |

**Data quality dominates data quantity for code models.** A Python-only model with curated + synthetic data is the highest-ROI path for a single-GPU project.

---

## 3. Phase 1 — Iteration Model (~360M)

### Config

Reference: SmolLM2-360M ([config.json](https://huggingface.co/HuggingFaceTB/SmolLM2-360M/resolve/main/config.json)) — a proven, publicly available Llama-style config at this scale.

| Parameter | Value |
|---|---|
| **Total parameters** | **~360M** |
| `num_hidden_layers` | 32 |
| `hidden_size` | 960 |
| `num_attention_heads` | 15 |
| `num_key_value_heads` | 5 (GQA, ratio 3:1) |
| `intermediate_size` | 2,560 |
| `vocab_size` | 49,152 (SmolLM2 BPE tokenizer) |
| `max_position_embeddings` | 2,048 |
| `rope_theta` | 100,000 |
| Activation | SiLU (≈ SwiGLU gate) |
| Normalization | RMSNorm, pre-norm |
| Bias terms | None |
| Tied embeddings | Yes |

### Optimizer & Schedule

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 5 × 10⁻⁴ |
| β₁ / β₂ | 0.9 / 0.95 |
| Weight decay | 0.1 |
| Gradient clip | 1.0 |
| LR schedule | Cosine with 1,000-step warmup |
| Precision | BF16 mixed precision |
| Gradient accumulation | Yes — target ~500K token effective batch |

### Training Objective

50% standard causal language modeling (CLM) + 50% Fill-in-Middle (FIM).
FIM uses both PSM (Prefix–Suffix–Middle) and SPM (Suffix–Prefix–Middle) modes at equal ratio.

### Purpose of Phase 1

Phase 1 is not a standalone deliverable — it is an ablation instrument. Use it to lock in decisions for Phase 2:

- [ ] Optimal FIM ratio (50/50 baseline; test 30/70 and 70/30)
- [ ] Data mixture (Stack-Edu Python ratio vs synthetic textbook ratio)
- [ ] LR schedule (cosine vs WSD)
- [ ] Tokenizer (SmolLM2 49K BPE vs custom 32K Python BPE)
- [ ] Effective batch size sensitivity
- [ ] Evaluation cadence (HumanEval pass@1 vs validation loss correlation)

### VRAM & Speed

| Metric | Value |
|---|---|
| Static VRAM (weights + optimizer + grads) | ~5.7 GB |
| Total VRAM with activations (batch=8, seq=2048) | ~10–12 GB |
| Memory tricks required | None |
| Estimated throughput | ~40–50K tokens/sec on RTX 5090 |
| Training time for 10B tokens | ~2–3 days |

---

## 4. Phase 2 — Target Model (~2B)

### Config

Custom config. No clean off-the-shelf Llama-style 2B model exists — Gemma-2 2B uses GELU + interleaved local/global attention (architecturally distinct); Qwen2.5 jumps from 1.5B to 3B. Derived by scaling SmolLM2-1.7B ([config.json](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B/resolve/main/config.json)) with GQA applied.

| Parameter | Value |
|---|---|
| **Total parameters** | **~2.0B** |
| `num_hidden_layers` | 32 |
| `hidden_size` | 2,048 |
| `num_attention_heads` | 16 |
| `num_key_value_heads` | 2 (GQA, ratio 8:1) |
| `intermediate_size` | 8,192 (4× hidden) |
| `vocab_size` | 32,768–49,152 (custom Python BPE) |
| `max_position_embeddings` | 4,096 (extend to 8,192 in stage 2) |
| `rope_theta` | 500,000 |
| Activation | SwiGLU |
| Normalization | RMSNorm, pre-norm |
| Bias terms | None |
| Tied embeddings | Yes |

### Parameter Count Breakdown

| Component | Parameters |
|---|---|
| Token embeddings (32K vocab × 2048 hidden, tied) | ~65.5M |
| Attention per layer: Q (2048×2048) + K (2048×256) + V (2048×256) + O (2048×2048) | ~9.4M |
| FFN per layer: gate (2048×8192) + up (2048×8192) + down (8192×2048) | ~50.3M |
| RMSNorm per layer (2 × 2048, negligible) | ~0.1M |
| **Per layer total** | **~59.8M** |
| **32 layers** | ~1,913M |
| **Embeddings** | ~65.5M |
| **Total** | **~1,978M ≈ 2.0B** |

### Optimizer & Schedule

| Parameter | Value |
|---|---|
| Optimizer | AdamW 8-bit (bitsandbytes `AdamW8bit`) — **required** |
| Learning rate | 3 × 10⁻⁴ |
| β₁ / β₂ | 0.9 / 0.95 |
| Weight decay | 0.1 |
| Gradient clip | 1.0 |
| LR schedule | Cosine with 2,000-step warmup (or WSD — decide in Phase 1) |
| Precision | BF16 mixed precision |
| Gradient checkpointing | **Required** |
| Flash Attention 2 | **Required** |
| `torch.compile` | Yes — 15–30% throughput gain |
| Gradient accumulation | Yes — target ~1M token effective batch |

### Training Objective

Same as Phase 1: 50% CLM + 50% FIM (PSM + SPM, equal split). Exact ratio locked in via Phase 1 ablations.

### Data

| Source | Tokens | Mix |
|---|---|---|
| Stack-Edu Python ([HuggingFaceTB/stack-edu](https://huggingface.co/datasets/HuggingFaceTB/stack-edu)) | 21.8B | 80% |
| Synthetic Python textbooks + exercises (phi-1 style, generated) | 1–2B | 15% |
| Python docs, PEPs, PyPI | < 1B | 5% |
| **Total unique** | **~24B** | — |

Train for 3–5 epochs ≈ **75–120B effective tokens** (phi-1 equivalent).

### Context Length Strategy

| Stage | Context | Rationale |
|---|---|---|
| Pretraining (bulk) | 4,096 | Covers ~95% of Python files; 16× cheaper per seq than 16K |
| Stage 2 extension | 8,192 | Multi-class/multi-function context; NTK-aware RoPE scaling |

### VRAM Budget

| Component | Memory |
|---|---|
| Weights BF16 (2B × 2 bytes) | ~4.0 GB |
| Gradients BF16 | ~4.0 GB |
| 8-bit AdamW states (2B × 2 bytes) | ~4.0 GB |
| Activations with gradient checkpointing (batch=2, seq=4096) | ~6–10 GB |
| **Total** | **~18–22 GB / 32 GB** |

Without 8-bit AdamW (FP32): 2B × 12 bytes optimizer = 24 GB optimizer alone — OOM. Both 8-bit AdamW and gradient checkpointing are required, not optional.

### Training Time Estimate (RTX 5090, ~20K tokens/sec)

| Token budget | Wall time |
|---|---|
| 10B tokens | ~6 days |
| 50B tokens | ~29 days |
| 100B tokens | ~58 days |

Target: **75–100B effective tokens** → estimated **~50–55% HumanEval pass@1**.

---

## 5. Component Rationale

### Attention: Grouped-Query Attention (GQA)

Standard multi-head attention (MHA) creates one KV pair per query head. GQA groups multiple query heads to share a single KV head, reducing KV cache memory proportionally.

**Why GQA over MHA:**
- KV cache during inference scales as `num_kv_heads × seq_len × head_dim × layers`. GQA-8 (16Q/2KV) reduces inference KV cache by 8× vs MHA with no measured quality loss at this scale.
- Every major code model from 2024 onward uses GQA: StarCoder2 (GQA-2), Qwen2.5-Coder (GQA-7), Llama 3.2 (GQA-4).
- Training memory is not affected — gradients still flow through all Q heads independently.

**Why GQA-8 for Phase 2 (16Q / 2KV heads):**
StarCoder2 and Qwen2.5-Coder both use 2 KV heads at the 1–3B scale. Ablations show no quality degradation vs MHA or GQA-4 at these sizes. 2 KV heads is the most aggressive reduction with confirmed quality parity. [[arXiv:2305.13245](https://arxiv.org/abs/2305.13245)]

**Why not MLA (Multi-Head Latent Attention)?**
MLA (DeepSeek-V2) compresses KV via low-rank projection. Its benefits are most pronounced at very large scales (≥70B) or extreme context lengths (>32K). At 2B scale, MLA adds implementation complexity with negligible gains over GQA. [[Raschka SotA survey, Feb 2026](https://magazine.sebastianraschka.com/p/a-dream-of-spring-for-open-weight)]

---

### Activation: SwiGLU

SwiGLU replaces the standard FFN's two-matrix structure with a gated three-matrix variant:

```
FFN(x) = (xW₁ ⊙ SiLU(xW₃)) · W₂
```

The gate (`SiLU(xW₃)`) selects which activations pass through, improving gradient flow and expressiveness.

**Why SwiGLU:**
- Outperforms GELU and ReLU on both language and code tasks across model scales.
- Adopted universally in 2024+ models: Llama 2/3, Qwen2.5, SmolLM2, DeepSeek-Coder, StarCoder2.
- phi-1 (2023) used GELU (pre-SwiGLU adoption) — switching to SwiGLU is a free quality improvement.

**Intermediate size formula:** SwiGLU uses 3 weight matrices instead of 2, so the intermediate dimension is reduced to keep parameter count equivalent: `intermediate = (2/3) × 4 × hidden` rounded to a multiple of 64.
- Phase 1: (2/3 × 4 × 960) = 2,560 ✓
- Phase 2: uses 8,192 (4× hidden) — slightly larger than formula suggests, matching SmolLM2-1.7B practice.

---

### Normalization: RMSNorm (Pre-Norm)

RMSNorm simplifies LayerNorm by removing the mean-centering step:

```
RMSNorm(x) = x / RMS(x) · γ,  where RMS(x) = sqrt(mean(x²))
```

**Why RMSNorm over LayerNorm:**
- ~10% faster than LayerNorm (no mean computation).
- Empirically more stable during pretraining at scale.
- Standard in all 2024+ Llama-family models.

**Why pre-norm (norm before attention/FFN) over post-norm:**
- Post-norm (original transformer, GPT-2) requires careful LR warmup to avoid gradient explosion at initialization.
- Pre-norm is more stable and allows higher learning rates. Standard since GPT-3.

---

### Positional Encoding: RoPE

Rotary Position Embedding encodes position by rotating query and key vectors in 2D subspaces:

```
RoPE(q, pos) = q ⊗ exp(i · pos · θ^(-2k/d))  for each dimension pair k
```

**Why RoPE over learned absolute positions (GPT-2 style):**
- Generalizes to context lengths beyond training length (with appropriate θ scaling).
- Relative position information is preserved in the dot-product (`q·k` encodes relative distance).
- Enables context extension via NTK-aware scaling without full retraining.
- Python files have high structural variability (short functions to long class definitions) — relative position awareness matters.

**θ values:**
- Phase 1: θ = 100,000 (SmolLM2 standard for 2K context)
- Phase 2: θ = 500,000 (Llama 3 style; better extrapolation for 4K–8K context)

Higher θ spreads the rotational frequencies across a wider range, allowing the model to distinguish positions further apart.

---

### Training Objective: Fill-in-Middle (FIM)

Standard causal LM predicts the next token given all preceding tokens. FIM extends this by also training the model to predict a middle span given its prefix and suffix:

```
PSM mode:  <PRE> prefix_tokens <SUF> suffix_tokens <MID> infill_tokens
SPM mode:  <SUF> suffix_tokens <PRE> prefix_tokens <MID> infill_tokens
```

**Why FIM for a code model:**
- IDE-style code completion is a middle-insertion task (cursor is inside a file, not at the end). A CLM-only model cannot do this.
- FIM at 50% rate, measured across StarCoder, DeepSeek-Coder, and CodeGemma: **no degradation to standard generation quality** while adding full completion capability.
- phi-1 did not use FIM — switching to FIM is a direct capability upgrade for free.
- PSM and SPM modes are both included to prevent the model from learning a spurious dependence on the token order of the sentinel tokens. [[arXiv:2207.14255](https://arxiv.org/abs/2207.14255)]

**FIM special tokens (add to vocabulary):**
- `<|fim_prefix|>` — marks start of prefix
- `<|fim_suffix|>` — marks start of suffix
- `<|fim_middle|>` — marks start of infill target
- `<|endoftext|>` — document separator

---

### Tokenizer: Custom BPE

Byte-level BPE (BBPE) trained on the Python corpus.

**Why not reuse an existing tokenizer:**

| Tokenizer | Vocab | Issue |
|---|---|---|
| Llama 3 | 128,256 | Embedding table = 128K × 2048 × 2 bytes = **500 MB** — 25% of the total 2B model |
| Qwen2.5 | 151,646 | Same issue — even larger |
| GPT-2 | 50,257 | Not code-optimized; fragments common Python tokens |
| SmolLM2 | 49,152 | General-purpose; acceptable for Phase 1 only |

A custom 32K–49K vocabulary trained on Python source code:
- Keeps the embedding table to ~130–200 MB (manageable fraction of 2B model)
- Learns Python-specific tokens: `def`, `class`, `import`, `self`, `__init__`, indentation patterns, common library names (`numpy`, `torch`, `os.path`)
- No unknown tokens (byte fallback in BBPE)

**Phase 1:** Reuse SmolLM2 49K tokenizer to skip tokenizer development during ablations.
**Phase 2:** Train custom BPE on Stack-Edu Python corpus. Include FIM special tokens in vocabulary.

---

### Tied Embeddings

Input token embeddings (shape: `vocab_size × hidden`) and output projection (shape: `hidden × vocab_size`) share the same weight matrix.

**Why tie embeddings:**
- Saves `vocab_size × hidden` parameters ≈ 65–100M params for a 32K–49K vocab at hidden=2048.
- That is 3–5% of total parameters for Phase 2 — non-trivial.
- Used in every sub-2B model surveyed (SmolLM2, Qwen2.5-Coder ≤1.5B, phi-1).
- Linguistic justification: semantically similar tokens should have similar input representations and similar output logit distributions — sharing weights enforces this symmetry.

---

## 6. VRAM Budget

### The 16-bytes-per-parameter rule

Full mixed-precision training with FP32 AdamW requires:

| Component | Bytes / param |
|---|---|
| Model weights (BF16) | 2 |
| Gradients (BF16) | 2 |
| AdamW momentum m (FP32) | 4 |
| AdamW variance v (FP32) | 4 |
| Master weight copy (FP32) | 4 |
| **Static total** | **16** |

Source: [Lyceum Technology](https://lyceum.technology/magazine/predict-vram-usage-pytorch-model/), [Modal](https://modal.com/blog/how-much-vram-need-fine-tuning)

### Phase 1 (~360M)

| Config | Static | +Activations (batch=8, seq=2048) | Total |
|---|---|---|---|
| BF16 + FP32 AdamW | 5.7 GB | ~4–6 GB | ~10–12 GB |

No memory tricks needed. Large batch headroom.

### Phase 2 (~2B)

| Config | Static | +Activations | Total |
|---|---|---|---|
| BF16 + FP32 AdamW (baseline) | **32 GB** | OOM | OOM |
| + 8-bit AdamW | ~12 GB | ~8–10 GB | ~20–22 GB |
| + 8-bit AdamW + grad checkpointing | ~12 GB | ~4–6 GB | **~16–18 GB** |

8-bit AdamW (bitsandbytes) quantizes optimizer states from FP32 (8 bytes: m + v) to INT8 (2 bytes: m + v), saving ~8 GB. ~98% accuracy retention confirmed. [[ROCm blog](https://rocm.blogs.amd.com/artificial-intelligence/bnb-8bit/README.html)]

Gradient checkpointing discards intermediate activations during the forward pass and recomputes them during backprop. Reduces activation memory by 50–80% at ~20% compute overhead. [[MLWorks](https://medium.com/mlworks/gradient-checkpointing-the-unsung-hero-of-llm-training-ac2bbe5d4396)]

---

## 7. Comparison with Existing Models

| Model | Params | Layers | Hidden | Q/KV heads | Intermediate | Context | Vocab | HumanEval |
|---|---|---|---|---|---|---|---|---|
| **Phase 1 (ours)** | **~360M** | **32** | **960** | **15/5** | **2,560** | **2K** | **49K** | target: ablation |
| SmolLM2-360M | 360M | 32 | 960 | 15/5 | 2,560 | 8K | 49K | — |
| phi-1-small | 350M | 24 | 1,024 | 16/16 (MHA) | 4,096 | 2K | 51K | 45% |
| Qwen2.5-Coder-0.5B | 494M | 24 | 896 | 14/2 | 4,864 | 32K | 152K | 28% |
| **Phase 2 (ours)** | **~2.0B** | **32** | **2,048** | **16/2** | **8,192** | **4K→8K** | **32–49K** | **target: ~50–55%** |
| SmolLM2-1.7B | 1.71B | 24 | 2,048 | 32/32 (MHA) | 8,192 | 8K | 49K | 22.6% |
| phi-1 | 1.3B | 24 | 2,048 | 32/32 (MHA) | 8,192 | 2K | 51K | 50.6% |
| DeepSeek-Coder-1.3B | 1.3B | 24 | 2,048 | 16/16 (MHA) | — | 16K | — | 34.8% |
| Qwen2.5-Coder-1.5B | 1.54B | 28 | 1,536 | 12/2 | 8,960 | 32K | 152K | 43.9% |
| Qwen2.5-3B | 3.09B | 36 | 2,048 | 16/2 | 11,008 | 32K | 152K | 52.4% |

Our Phase 2 config sits between phi-1 (1.3B, Python-only, 50.6%) and Qwen2.5-3B (multilingual, 52.4%) in both scale and expected performance. With Python-only data and FIM, the target is 50–55% HumanEval from 2B parameters.

---

## 8. References

| Topic | Reference |
|---|---|
| phi-1 (data quality strategy) | [arXiv:2306.11644](https://arxiv.org/abs/2306.11644) |
| FIM training objective | [arXiv:2207.14255](https://arxiv.org/abs/2207.14255) |
| GQA paper | [arXiv:2305.13245](https://arxiv.org/abs/2305.13245) |
| StarCoder2 / The Stack v2 | [arXiv:2402.19173](https://arxiv.org/abs/2402.19173) |
| Qwen2.5-Coder | [arXiv:2409.12186](https://arxiv.org/abs/2409.12186) |
| SmolLM2 | [arXiv:2502.02737](https://arxiv.org/abs/2502.02737) |
| SmolLM2-360M config | [HuggingFace config.json](https://huggingface.co/HuggingFaceTB/SmolLM2-360M/resolve/main/config.json) |
| SmolLM2-1.7B config | [HuggingFace config.json](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B/resolve/main/config.json) |
| SmolLM3 training playbook | [HuggingFace blog](https://huggingface.co/blog/smollm3) |
| Proxy model ablation research | [arXiv:2512.24503](https://arxiv.org/html/2512.24503) |
| 8-bit AdamW (bitsandbytes) | [HF docs](https://huggingface.co/docs/bitsandbytes/main/en/optimizers) |
| VRAM formula | [Lyceum Technology](https://lyceum.technology/magazine/predict-vram-usage-pytorch-model/) |
| VRAM fine-tuning reference | [Modal blog](https://modal.com/blog/how-much-vram-need-fine-tuning) |
| Gradient checkpointing | [MLWorks/Medium](https://medium.com/mlworks/gradient-checkpointing-the-unsung-hero-of-llm-training-ac2bbe5d4396) |
| SotA architecture survey (Feb 2026) | [Raschka — A Dream of Spring](https://magazine.sebastianraschka.com/p/a-dream-of-spring-for-open-weight) |
| Stack-Edu Python dataset | [HuggingFaceTB/stack-edu](https://huggingface.co/datasets/HuggingFaceTB/stack-edu) |
| The Stack v2 | [bigcode/the-stack-v2](https://huggingface.co/datasets/bigcode/the-stack-v2) |
| NVIDIA Blackwell training guide | [NVIDIA blog](https://developer.nvidia.com/blog/train-an-llm-on-an-nvidia-blackwell-desktop-with-unsloth-and-scale-it/) |
