# SLM Project — Research Scratchpad

> **Goals:**
> 1. Understand the current state of the art — techniques, architectures, training recipes, and best practices for building small language models.
> 2. Achieve the best possible performance within the available hardware budget.
>
> **Hardware:** NVIDIA RTX 5090, 32 GB VRAM (single GPU, consumer hardware)
>
> This document synthesizes key findings from primary resources on SLM training.

---

## Table of Contents

1. [HuggingFace Smol Training Playbook](#1-huggingface-smol-training-playbook)
2. [Giles Thomas — LLM from Scratch (Part 28)](#2-giles-thomas--llm-from-scratch-part-28)
3. [karpathy/nanochat](#3-karpathynanochat)
4. [rasbt/reasoning-from-scratch](#4-rasbtreasoningfrom-scratch)
5. [Raschka — A Dream of Spring: 10 Open-Weight Architectures Jan-Feb 2026](#5-raschka--a-dream-of-spring-10-open-weight-architectures-jan-feb-2026)
6. [Python Code SLM Research — Architecture & Data](#6-python-code-slm-research--architecture--data)
7. [Synthesis & Scope Definition](#7-synthesis--scope-definition)

---

## 1. HuggingFace Smol Training Playbook

**Source:** https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook
**Authors:** 12 HF researchers (Loubna Ben Allal, Lewis Tunstall, et al.)
**Published:** Oct 30, 2025
**Model studied:** SmolLM3 — 3B params, 11T tokens, multilingual, 128k context

### Should You Train From Scratch?

Valid reasons:
- Research hypothesis testing (e.g. "does this optimizer scale?")
- Domain specificity (DNA, legal, custom hardware)
- Safety/governance requirements
- Open-source ecosystem gap (underserved languages, on-device sizes)

Rule of thumb: spend days trying Qwen3/Gemma 3 post-training first. If you can't reach goals → consider pretraining.

### Key Philosophy

> "Successful LLM training is 70% data curation, 20% iteration speed, and 10% architectural innovation."

- Iteration cadence beats architectural novelty (train every 2–3 months)
- Small teams (2–3 people) can pretrain Llama-3-scale models
- Ablations + debugging = 50%+ of total compute cost

### Architecture Decisions

| Component | Recommended Choice | Notes |
|---|---|---|
| Attention | GQA-4 (32 query, 8 KV heads) | Same perf as MHA, 4× smaller KV cache |
| Activation | SwiGLU | Gated FFN, standard in 2025 |
| Layer norm | RMSNorm | More stable than LayerNorm |
| Bias terms | None | Modern trend |
| Embedding | Tied (input = output projection) | 10–18% param savings for small models |
| Positional | RoPE for ≤4k context; NoPE hybrid for long context | |
| Document masking | Enabled | Prevents attention across unrelated packed docs |

**Baseline architectures to copy:** Llama 3.2 (1B/3B), Qwen3 (0.6B–32B)

### Ablation Config (1B model, reference)

```yaml
model:
  hidden_size: 2048
  num_hidden_layers: 16
  num_attention_heads: 32
  num_key_value_heads: 8       # GQA-4
  intermediate_size: 8192
  max_position_embeddings: 4096
  rope_theta: 50000.0
  tie_word_embeddings: true

optimizer:
  learning_rate: 0.0005
  lr_warmup_steps: 2000
  lr_decay_steps: 18000
  lr_decay_style: cosine
  adam_beta1: 0.9
  adam_beta2: 0.95
  clip_grad: 1.0
```

### Data Recommendations

| Dataset | Domain | Notes |
|---|---|---|
| FineWeb-Edu | Web (education-filtered) | Better than raw Common Crawl |
| DCLM | Web (large-scale filtered) | Alternative to FineWeb |
| FineMath | Math | Mathematical reasoning |
| Stack-Edu-Python | Code (Python) | Coding/reasoning tasks |
| FinePDFs | Long-form PDF | ~2× longer than web docs |

SmolLM3 ablation mix: **FineWeb-Edu 70% / Stack-Edu-Python 20% / FineMath 10%**

### Training Hyperparameters

- LR: 2–4 × 10⁻⁴ for 1–3B models
- Warmup: 2000–2500 steps
- Schedule: Cosine or WSD (Warmup-Stable-Decay)
- Global batch size: ~1–4M tokens
- Chinchilla optimal: 20 tokens/param — but SmolLM3 intentionally overtrains at 3.7× for quality

### Ablation Benchmarks

Prioritize monotonic, low-noise, above-random tasks:
- MMLU, ARC, HellaSwag, WinoGrande, CommonSenseQA, GSM8K, HumanEval

### Training Frameworks

| Framework | Notes |
|---|---|
| Nanotron | Used by SmolLM3, StarCoder; moderate complexity |
| TorchTitan | Newer, leaner (~7k core LoC) |
| Megatron-LM | Battle-tested, complex |
| DeepSpeed | ZeRO pioneer, complex |

### Rules of Engagement

1. Validate eval suite before training (reproduce published results first)
2. Change one variable at a time
3. Train ≥35–45B tokens for reliable 1B ablation signal
4. Be paranoid about infrastructure (TP bugs silently corrupt training)
5. Version-control everything; small bugs compound over trillions of tokens

---

## 2. Giles Thomas — LLM from Scratch (Part 28)

**Source:** https://www.gilesthomas.com/2025/12/llm-from-scratch-28-training-a-base-model-from-scratch
**Model:** 163M params (GPT-2 small architecture)
**Hardware:** Single RTX 3090 (24 GB VRAM)

### Architecture (GPT-2 small, from Raschka's book)

| Param | Value |
|---|---|
| Vocab size | 50,257 (GPT-2 tokenizer) |
| Context length | 1,024 tokens |
| Embedding dim | 768 |
| Attention heads | 12 |
| Layers | 12 |
| Dropout | 0.1 |
| QKV bias | Disabled |
| Weight tying | Disabled |

### Dataset & Tokenization Strategy

- Compared FineWeb-10B vs FineWeb-Edu
- Problem with naive truncation: ~29% of tokens lost
- Solution: **concatenate documents with EOS delimiter** into a continuous stream, then split into fixed-length sequences

### Training Setup

- Tokens: ~3.2B (Chinchilla-optimal for 163M params = 20 × 163M)
- Optimizer: AdamW (lr=4×10⁻⁴, weight_decay=0.1)
- Batch size: 6 sequences
- Mixed precision: AMP + TF32 tensor cores

### Performance Optimizations

| Technique | Speedup |
|---|---|
| TF32 tensor cores | ~22% |
| AMP (bfloat16/float16) | ~48% combined with TF32 |
| Throughput | ~12,600 → ~19,921 tokens/sec |
| Training time | ~44–48 hours for 3.2B tokens |

### Results

| Model | Validation Loss |
|---|---|
| OpenAI GPT-2 small (original) | ~3.50 |
| Author's FineWeb | 3.94 |
| Author's FineWeb-Edu | 3.693 |
| Author's FineWeb-Edu (2× training) | 3.661 |

Gap with OpenAI weights attributed to missing:
- Cosine LR scheduling
- Gradient accumulation for larger effective batch
- Weight tying, QKV bias
- More training data

### Key Takeaways

- A single RTX 3090 can train a 163M model in ~2 days
- Data concatenation with EOS is better than truncation
- FineWeb-Edu > FineWeb for downstream quality
- Mixed precision is essential for consumer GPU training

---

## 3. karpathy/nanochat

**Source:** https://github.com/karpathy/nanochat
**Author:** Andrej Karpathy
**Goal:** Full pipeline (pretraining → SFT → RL → eval → inference) on a single GPU node

### Philosophy

- Single complexity knob: `--depth` (number of transformer layers)
- All other hyperparameters (width, heads, LR, training duration, weight decay) computed automatically for Chinchilla-optimal results
- Rejects giant config files and framework complexity
- "A single, cohesive, minimal, readable, hackable codebase"

### Specs

| Property | Value |
|---|---|
| Model size | ~1.6B params (GPT-2 capability at depth 24–26) |
| Hardware | 8× H100 (or A100) node |
| Precision | bfloat16 on modern NVIDIA; float32 elsewhere |
| Cost | ~$15–48 per full training run |
| Speedrun time | ~1.65–3 hours to GPT-2 capability on 8×H100 |

### Pipeline

Tokenization → Pretraining → SFT → RL → Evaluation → Inference

Usage:
```bash
bash runs/speedrun.sh          # Full training pipeline
python -m scripts.chat_web     # ChatGPT-like interface
```

### Key Contribution

- Proves the full LLM pipeline (not just pretraining) costs ~$48
- Automated hyperparameter selection from a single `--depth` integer
- Good reference for a minimal, hackable end-to-end system

---

## 4. rasbt/reasoning-from-scratch

**Source:** https://github.com/rasbt/reasoning-from-scratch
**Author:** Sebastian Raschka
**Book:** *Build A Reasoning Model (From Scratch)* (Manning, ISBN 9781633434677)

### Goal

Start from a pre-trained base LLM, add reasoning capabilities step by step in PyTorch.
Mirrors DeepSeek R1 and GPT-4 Thinking approaches, scaled for consumer hardware.

### Base Model

**Qwen3** (pre-trained) — focus is post-training, not foundation building.

### Techniques Covered

| Technique | Description |
|---|---|
| Chain-of-thought prompting | Inference-time scaling |
| Self-consistency | Multiple reasoning paths, majority vote |
| Self-refinement | Iterative generation improvement |
| GRPO | Group Relative Policy Optimization (RL training) |
| Model distillation | Transfer reasoning from larger to smaller model |
| DeepSeek-V3/Olmo3/GDPO-style RL | Advanced RL variants |

### Evaluation Methods

- MMLU
- MATH-500
- LLM-as-a-judge

### Structure

8 chapters: text generation fundamentals → evaluation → inference scaling → RL → distillation

### Key Insight

This repo is about **post-training** (reasoning on top of a base model), not pretraining from scratch. Relevant to the later phases of this project.

---

## 5. Raschka — A Dream of Spring: 10 Open-Weight Architectures Jan-Feb 2026

**Source:** https://magazine.sebastianraschka.com/p/a-dream-of-spring-for-open-weight
**Author:** Sebastian Raschka
**Focus:** Architectural analysis of 10 major open-weight LLM releases from Jan-Feb 2026

> "Modeling performance is likely not attributed to the architecture design itself but rather the dataset quality and training recipes." — Raschka
> However, architectural choices significantly impact inference efficiency and deployment feasibility.

### Models Reviewed

| Model | Size | Key Architecture |
|---|---|---|
| Arcee AI Trinity Large | 400B (13B active) | MoE, alternating local-global attention (3:1, 4096 window), QK-Norm, gated attention, depth-scaled RMSNorm |
| Moonshot AI Kimi K2.5 | 1T | DeepSeek V3 architecture, native multimodal via early fusion, trained on 15T mixed visual+text tokens |
| StepFun Step 3.5 Flash | 196B MoE (11B active) | Multi-Token Prediction (MTP-3) at train and inference, 100 tok/s at 128k ctx vs DeepSeek V3.2's 33 tok/s |
| Qwen3-Coder-Next | 80B (3B active) | Gated DeltaNet + Gated Attention hybrid (3:1), 262k native context |
| z.AI GLM-5 | 744B (40B active) | Adopted DeepSeek MLA, DeepSeek Sparse Attention for long contexts |
| MiniMax M2.5 | 230B | Classic GQA only — no hybrid attention; strong coding despite architectural simplicity |
| Nanbeige 4.1 | 3B | Similar to Qwen3 4B / Llama 3.2 3B; no weight tying; gains mainly from post-training |
| Qwen3.5 | 397B-A17B | Gated DeltaNet hybrid adopted from -Next; adds multimodal; improved agentic coding |
| Ant Group Ling 2.5 / Ring 2.5 | 1T each | Lightning Attention (recurrent linear) + MLA; 3.5× higher throughput than Kimi K2 at 32k seq length |
| Cohere Tiny Aya | 3.35B | Parallel transformer blocks (attention + MLP simultaneously); dropped QK-Norm; multilingual focus; non-commercial only |

### Update: Sarvam 30B and 105B (March 6, 2026)

- **30B:** Standard Grouped Query Attention
- **105B:** DeepSeek-style Multi-Head Latent Attention (MLA)
- 105B matches gpt-oss 120B / Qwen3-Next on benchmarks
- Sarvam 30B shows 20–40% throughput improvement over Qwen3-30B
- Strong performance on Indian language benchmarks (90% preference over alternatives)

### Key Architectural Trends (SotA as of Feb 2026)

**Attention mechanisms — diversifying beyond standard scaled dot-product:**
- **Multi-Head Latent Attention (MLA)** — DeepSeek-originated; adopted by GLM-5, Sarvam 105B; compresses KV cache via low-rank projection
- **Hybrid linear + standard attention** — Gated DeltaNet alternating with softmax attention (Qwen3-Coder-Next, Qwen3.5); 3:1 ratio linear-to-standard
- **Lightning Attention** — linear recurrent attention alternative; used in Ling 2.5 for 3.5× throughput gain at long sequences
- **Sliding window / local-global** — 3:1 local:global ratio with 4096 window (Arcee Trinity)
- **GQA** — still the baseline standard; MiniMax M2.5 uses it alone successfully

**Efficiency techniques gaining adoption:**
- **Multi-Token Prediction (MTP)** — applied at both train and inference time (StepFun Flash); improves throughput without accuracy loss
- **QK-Norm** — adds training stability; Arcee uses it; Cohere deliberately dropped it to preserve long-context performance
- **Gated attention mechanisms** — reduces attention sinks; paired with depth-scaled RMSNorm initialization
- **DeepSeek Sparse Attention** — emerging for long-context efficiency

**MoE patterns:**
- Active parameter ratios typically 3–11B out of total; inference cost dominated by active params
- Alternating attention types (local/global or linear/standard) are the new normal at scale

### Relevance to This Project

**For our 1B dense model (Phase 2):**
- **GQA remains the right choice** for our scale — MLA benefits are most pronounced at very large scale or extreme context lengths
- **MTP is worth investigating** — even if not at 3-token, MTP-2 could improve our training efficiency
- **Hybrid attention** (e.g. some DeltaNet layers) is emerging but adds complexity; defer to Phase 3 unless evidence shows clear wins at 1B scale
- **The MiniMax M2.5 result is validation**: classic GQA-only architecture can be competitive if data and training recipe are strong — architecture alone is not the bottleneck at our scale
- **QK-Norm** is worth adding for training stability — low cost, clear benefit
- **Parallel attention + MLP blocks** (Cohere Aya approach) can increase throughput; worth benchmarking vs sequential

### Images

42 images from the article are stored in `docs/images/` (img_01 through img_42). These contain architecture diagrams, benchmark comparison tables, and attention mechanism illustrations for each of the 10 models.

---

## 6. Python Code SLM Research — Architecture & Data

**Focus:** What architecture and data choices are optimal for a 1B-scale model trained exclusively on Python code?
**Sources:** phi-1 (arXiv:2306.11644), StarCoder2 (arXiv:2402.19173), Qwen2.5-Coder (arXiv:2409.12186), DeepSeek-Coder, CodeGemma (arXiv:2406.11409), SmolLM2 (arXiv:2502.02737), TinyLlama, HF Stack-Edu

---

### Existing Small Code Models — Benchmark Comparison

| Model | Params | HumanEval pass@1 | MBPP pass@1 | Context | Attention | Training tokens |
|---|---|---|---|---|---|---|
| phi-1 (Microsoft) | 1.3B | **50.6%** | **55.5%** | 2,048 | MHA | ~50B effective (7B unique) |
| phi-1-small | 350M | 45.0% | — | 2,048 | MHA | — |
| Qwen2.5-Coder-1.5B | 1.5B | 43.9% | 59.2% | 32K | GQA-2 | 5.5T |
| Qwen2.5-Coder-0.5B | 0.5B | 28.0% | 40.4% | 32K | GQA-2 | 5.5T |
| DeepSeek-Coder-1.3B | 1.3B | 34.8% | — | 16,384 | MHA | 2T |
| CodeGemma-2B | 2B | 31.1% | 43.6% | 8,192 | MHA | +500B |
| StarCoder2-3B | 3B | 31.7% | — | 16K (SW) | GQA-2 | 3.3T |
| SmolLM2-1.7B | 1.7B | 22.6% | — | 8,192 | GQA | 11T |
| TinyLlama-1.1B | 1.1B | — | — | 4,096 | GQA | 3T |

**Key insight:** phi-1 at 1.3B achieves 50.6% HumanEval with only ~7B unique tokens of curated Python data. Qwen2.5-Coder-1.5B trained on 5.5T tokens reaches only 43.9%. **Data quality dominates data quantity for code models.** A Python-only model with curated + synthetic data should significantly outperform a multilingual code model at the same parameter count.

---

### Recommended Architecture for 1B Python Code SLM

Correlated across phi-1, Qwen2.5-Coder, DeepSeek-Coder, StarCoder2, and the Raschka SotA survey:

| Component | Choice | Rationale |
|---|---|---|
| Parameters | ~1.1–1.3B | phi-1 and DeepSeek-Coder-1.3B scale; fits RTX 5090 with headroom |
| Layers | 24 | phi-1, DeepSeek-Coder-1.3B; Qwen2.5-Coder-0.5B also uses 24 |
| Hidden size | 2,048 | Standard at this scale |
| Query heads | 16 | Standard for hidden=2048 |
| KV heads (GQA) | 2 | StarCoder2, Qwen2.5-Coder standard; 8× KV cache reduction vs MHA |
| FFN intermediate | 5,504 (≈ 8/3 × 2048, ×64 multiple) | SwiGLU formula |
| Activation | SwiGLU | Universal 2024+ standard; outperforms GELU for code |
| Normalization | RMSNorm (pre-norm) | Universal 2024+ standard; more stable than LayerNorm |
| Positional encoding | RoPE (θ=500,000) | Long-context friendly; Llama 3 style |
| Vocabulary | 32K–49K byte-level BPE | Trained on Python corpus; code-specific tokens; keeps embedding table small |
| Context length | 4,096 (extend to 8K in stage 2) | Covers ~95% of Python files; 4K training is 16× cheaper than 16K per sequence |
| Training objective | **CLM 50% + FIM 50%** (PSM/SPM split) | FIM proven to improve both code completion and generation quality |
| Tied embeddings | Yes | ~10–15% param savings; all sub-2B models use this |
| Flash Attention 2 | Yes | Mandatory on Blackwell; O(n) memory |
| Bias terms | None | Modern standard |
| QK-Norm | Optional | Arcee Trinity uses it for stability; low cost |

**Why FIM (Fill-in-Middle)?** StarCoder, DeepSeek-Coder, CodeGemma all use FIM at 50% rate with PSM (Prefix-Suffix-Middle) and SPM (Suffix-Prefix-Middle) split. It directly improves IDE-style code completion without hurting generation quality. PSM: `<PRE> prefix <SUF> suffix <MID> infill`. Ref: arXiv:2207.14255.

**Why GQA-2 instead of GQA-4?** StarCoder2 and Qwen2.5-Coder use 2 KV heads rather than the 4 used by larger models. At 1B scale with 16 Q heads, 2 KV heads gives 8× KV cache reduction. Ablations show no quality loss vs MHA at this scale.

---

### Tokenizer

| Choice | Vocab size | Tradeoff |
|---|---|---|
| Byte-level BPE on Python corpus | 32K–49,152 | **Recommended**: no UNK tokens; handles all Unicode/special chars; embedding table ≈ 100–160M params; code-specific tokens |
| Llama 3 tokenizer | 128,256 | Multilingual-optimized; 25% fewer tokens for code but embedding table costs ~510M params at hidden=2048 — nearly half the model |
| Qwen2.5 tokenizer | 151,646 | Same issue: large embedding table dominates param budget for a 1B model |
| GPT-2 tokenizer | 50,257 | Acceptable but not code-optimized; legacy |

**Verdict:** For a Python-only 1B model, a custom 32K–49K BPE tokenizer trained on your corpus is optimal. Large vocabularies (100K+) are efficient per-token but the embedding table cost is prohibitive at 1B scale. Code-specific special tokens to include: `<|fim_prefix|>`, `<|fim_suffix|>`, `<|fim_middle|>`, `<|endoftext|>`.

---

### Data Sources

**phi-1's CodeTextbook is not publicly available.** Microsoft never released it. No clean equivalent exists on HuggingFace. Synthetic data generation is off the table.

The ready-to-use solution is already ideal:

| Dataset | URL | Tokens | Download size | License | Agreement | Notes |
|---|---|---|---|---|---|---|
| **smollm-corpus python-edu** | [HuggingFaceTB/smollm-corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) | **4B** | **644 MB** | ODC-BY | None | Score ≥4, content included, used to train SmolLM-360M. **Primary source.** |
| **FineWeb-Edu 10B sample** | [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | 10B | 28.5 GB | ODC-BY | None | English web text filtered for educational quality. English NL component. |
| stack-edu Python full | [HuggingFaceTB/stack-edu](https://huggingface.co/datasets/HuggingFaceTB/stack-edu) | 21.8B | ~3 GB | ODC-BY | None | Needs Software Heritage S3 pipeline for content — not trivial. Skip for now. |

**Note on stack-edu Python full:** The dataset stores SWHIDs (Software Heritage identifiers), not raw code. Extracting content requires a non-trivial S3 pipeline. `smollm-corpus python-edu` is the pre-extracted version — use that instead.

**Why include English NL (50/50 mix):**
A 2024 controlled study ([arXiv:2408.10914](https://arxiv.org/abs/2408.10914)) compared training strategies:
- Mixed 50% code + 50% text → **best code AND best NL reasoning**
- Code-only → 86% relative drop in world knowledge, weaker reasoning
- phi-1 appears code-only but its synthetic textbooks are written in English — NL is embedded in the data

**Decided data mix:**

| Phase | Source | Tokens | Mix | Download |
|---|---|---|---|---|
| Both | smollm-corpus python-edu | 4B | 50% | 644 MB |
| Both | FineWeb-Edu (subset, 4B tokens sampled) | 4B | 50% | ~11 GB (subset of 28.5 GB) |
| **Total unique** | | **~8B** | — | **~12 GB** |

Train 2 epochs = **~16B effective tokens**. Chinchilla-optimal for 360M is ~7B; for 2B is ~40B. This is a practical compromise given single-GPU time constraints.

**Throughput reality (RTX 5090):**

| Model | Est. tokens/sec | 8B tokens | 16B tokens |
|---|---|---|---|
| 360M | ~40–50K | ~44–56 hours (~2 days) | ~88–112 hours (~4 days) |
| 2B | ~5–8K | ~14–22 days | ~28–44 days |

Training time formula: `6 × N_params × N_tokens / (TFLOPS × MFU)`. RTX 5090 at 209 BF16 TFLOPS, ~30% MFU.

---

### Context Length Strategy

| Stage | Context | Rationale |
|---|---|---|
| Pretraining (bulk) | 4,096 | Covers ~95% of Python files; 16× cheaper per seq than 16K |
| Mid-training extension | 8,192 | Cover multi-class/multi-function files; apply YaRN scaling |
| Optional long-context | 16,384+ | Project-level context; only if needed |

Training at 4K vs 32K context costs 64× less compute per token (attention is O(n²)). Start at 4K and extend in a short second stage using NTK-aware RoPE scaling.

---

### VRAM Budget (1.2B model, seq_len=4096, batch=4, grad checkpointing)

| Component | Memory |
|---|---|
| Weights BF16 | ~2.4 GB |
| Gradients BF16 | ~2.4 GB |
| AdamW FP32 states | ~9.6 GB |
| Activations (grad checkpointing) | ~4–6 GB |
| **Total** | **~18–20 GB / 32 GB** |

Comfortable margin — can increase batch size or sequence length.

---

### Training Time Estimate (single RTX 5090, ~20K tokens/sec)

| Tokens | Time |
|---|---|
| 10B | ~6 days |
| 50B | ~29 days |
| 100B | ~58 days |

Target: **75–100B effective tokens** (phi-1 equivalent). Expected result: **~45–55% HumanEval pass@1**.

---

### Training Schedule

#### WSD (Warmup-Stable-Decay)

Introduced by MiniCPM ([arXiv:2404.06395](https://arxiv.org/abs/2404.06395)). Divides training into three phases:

```
Phase 1 — Warmup   (1–2% of steps):  LR rises 0 → peak
Phase 2 — Stable   (80–85% of steps): LR held constant at peak
Phase 3 — Decay    (10–15% of steps): LR anneals to zero
```

Key advantage over cosine decay: **stable-phase checkpoints are reusable**. You can fork multiple decay runs with different data mixes from the same checkpoint, compare them, and keep the best — without retraining. With cosine the LR is always changing so checkpoints are not interchangeable.

The decay phase is also where you inject the highest-quality data. MiniCPM found mixing high-quality data *during decay* outperformed putting it in post-training SFT.

#### OLMo-2 two-stage recipe (best fully open recipe)

AllenAI released full training details for OLMo-2 ([arXiv:2501.00656](https://arxiv.org/abs/2501.00656), [HuggingFace](https://huggingface.co/allenai/OLMo-2-1124-7B)):

- **Stage 1 (~90% of budget):** Broad data mix at constant LR — DCLM web 88%, code 2%, papers/math/Wikipedia
- **Stage 2 (~10% of budget):** High-quality annealing mix — upweight best sources, LR anneals to zero
- **Model souping:** Train 2–3 Stage 2 variants with different mixes, average weights — free quality improvement

OLMo-2 architecture improvements over OLMo-1 match our choices exactly: RMSNorm, QK-Norm, RoPE.

#### Recommended schedule for this project

**Phase 1 (360M iteration model):**
```
Single stage, cosine LR (simpler for ablations)
Data: Stack-Edu Python 70% + English NL 15% + Synthetic 15%
Goal: validate pipeline, FIM ratio, data mix — not peak quality
```

**Phase 2 (2B target model):**
```
Stage 1 — WSD Stable (~85% of tokens, ~64–85B tokens)
  LR: constant 3×10⁻⁴ after 2K-step warmup
  Data: Stack-Edu Python 70% / English NL 15% / Synthetic textbooks 15%

Stage 2 — WSD Decay (~15% of tokens, ~11–19B tokens)
  LR: anneals from 3×10⁻⁴ → 0
  Data: upweight quality — Synthetic textbooks 40% / Python docs+PEPs 30% / Stack-Edu top-scored 30%
  Optional: model soup — run 2 decay variants, average weights
```

#### Data ordering within a stage

DeepSeek-Coder sorts files by **dependency order** (imports resolved first) before packing into sequences. This means the model sees library definitions before the code that uses them — reduces confusion from forward references. Worth implementing in the data pipeline.

### Key References

| Resource | URL |
|---|---|
| phi-1 paper | arXiv:2306.11644 |
| Code in pretraining study | arXiv:2408.10914 |
| WSD schedule (MiniCPM) | arXiv:2404.06395 |
| OLMo-2 paper | arXiv:2501.00656 |
| OLMo-2 HuggingFace | huggingface.co/allenai/OLMo-2-1124-7B |
| StarCoder2 / The Stack v2 | arXiv:2402.19173 |
| Qwen2.5-Coder | arXiv:2409.12186 |
| DeepSeek-Coder | github.com/deepseek-ai/DeepSeek-Coder |
| CodeGemma | arXiv:2406.11409 |
| SmolLM2 | arXiv:2502.02737 |
| FIM training | arXiv:2207.14255 |
| Stack-Edu Python dataset | huggingface.co/datasets/HuggingFaceTB/stack-edu |
| The Stack v2 | huggingface.co/datasets/bigcode/the-stack-v2 |

---

## 7. Synthesis & Scope Definition

### What This Project Is

Train a small language model from scratch, **specialized on Python code**, implementing and iterating on the full pipeline: data curation → tokenizer → pretraining → (optionally) instruction tuning / FIM.

**Domain focus: Python code only** — motivated by phi-1's result showing that a 1.3B Python-only model with curated + synthetic data (7B unique tokens) outperformed multilingual code models 20–30× larger. This is the highest-ROI path for a single-GPU project.

### Hardware Budget — RTX 5090 (32 GB VRAM)

The RTX 5090 unlocks significantly more headroom than the RTX 3090 used in Giles Thomas's guide. Key capabilities:

**VRAM budget breakdown for training (mixed precision, AdamW):**

| Model size | Weights (bf16) | Optimizer states (fp32 AdamW) | Gradients (bf16) | Approx. total (no activations) |
|---|---|---|---|---|
| 125M | 0.25 GB | 1.0 GB | 0.25 GB | ~1.5 GB |
| 500M | 1.0 GB | 4.0 GB | 1.0 GB | ~6 GB |
| 1B | 2.0 GB | 8.0 GB | 2.0 GB | ~12 GB |
| 3B | 6.0 GB | 24.0 GB | 6.0 GB | ~36 GB ⚠️ tight |

**Comfortable training target: 1B parameters** (leaves ~20 GB for activations + batch)
- With gradient checkpointing: can push to ~2B
- 3B is possible with gradient checkpointing + small batch + gradient accumulation, but tight

**RTX 5090 architecture notes (Blackwell):**
- 32 GB GDDR7 (vs. 24 GB on RTX 3090)
- Higher memory bandwidth than 4090 (~1.8 TB/s)
- FP8 tensor core support (potential additional speedup vs. bf16)
- Flash Attention 2/3 compatible
- `torch.compile` + CUDA graphs will help significantly

### Architecture Decision

Both phases use identical architecture family (Llama-style: GQA, SwiGLU, RMSNorm, RoPE, FIM) and the same dataset. Scale is the only difference. No ablations — use published defaults throughout.

---

**Phase 1 — ~360M (validate pipeline, get a working model fast)**

Reference config: SmolLM2-360M ([config.json](https://huggingface.co/HuggingFaceTB/SmolLM2-360M/resolve/main/config.json)) — proven architecture trained on this exact dataset.

| Decision | Value | Rationale |
|---|---|---|
| Params | **~360M** | SmolLM2-360M config; fast to train |
| Layers | 32 | SmolLM2-360M config |
| Hidden size | 960 | SmolLM2-360M config |
| Q / KV heads | 15 / 5 (GQA-3) | SmolLM2-360M config |
| Intermediate (SwiGLU) | 2,560 | SmolLM2-360M config |
| Vocab | 49,152 (SmolLM2 BPE tokenizer) | Reuse existing; skip tokenizer work |
| Context length | 2,048 | Fast training; covers most Python functions |
| RoPE θ | 100,000 | SmolLM2 standard |
| Training objective | CLM 50% + FIM 50% (PSM/SPM) | Standard code model objective |
| Dataset | smollm-corpus python-edu 50% + FineWeb-Edu 50% | 8B unique tokens total |
| Optimizer | AdamW (lr=5×10⁻⁴, β₁=0.9, β₂=0.95, wd=0.1) | Standard for 360M |
| LR schedule | Cosine with 1K warmup | Simple, no mid-training data changes |
| Training tokens | 16B (2 epochs over 8B unique) | ~4 days on RTX 5090 @ 40–50K tok/s |
| Static VRAM | ~5.7 GB | No memory tricks needed |

---

**Phase 2 — Target Model: ~2B**

Custom config derived from Qwen2.5-3B/SmolLM2-1.7B at reduced scale. No clean off-the-shelf Llama-style 2B exists (Gemma-2B uses GELU+interleaved attention; Qwen2.5 jumps from 1.5B to 3B). [[verified against configs](https://huggingface.co/Qwen/Qwen2.5-3B/resolve/main/config.json)]

| Decision | Value | Rationale |
|---|---|---|
| Params | **~2.0B** | Max practical size on 32 GB with required tricks |
| Layers | 32 | Balances depth vs compute |
| Hidden size | 2,048 | Same as SmolLM2-1.7B; parameter-efficient |
| Q / KV heads | 16 / 2 (GQA-8) | 8× KV cache reduction; Qwen2.5-Coder / StarCoder2 standard |
| Intermediate (SwiGLU) | 8,192 (4× hidden) | SmolLM2-1.7B standard |
| Vocab | Custom 32K–49K byte-level BPE on Python corpus | Avoids 100K+ embedding overhead; see Section 6 |
| Context length | 4,096 (extend to 8K in stage 2) | Covers ~95% of Python files |
| RoPE θ | 500,000 | Long-context friendly; Llama 3 style |
| Tied embeddings | Yes | ~10–15% param savings |
| Training objective | CLM 50% + FIM 50% (PSM/SPM) | Code completion critical |
| Dataset (Stage 1) | smollm-corpus python-edu 50% + FineWeb-Edu 50% | Same as Phase 1; 8B unique tokens |
| Dataset (Stage 2 decay) | smollm-corpus python-edu 80% + FineWeb-Edu 20% | Upweight Python quality in decay phase |
| Optimizer | AdamW 8-bit (lr=3×10⁻⁴, β₁=0.9, β₂=0.95, wd=0.1) | Standard for 2B |
| **8-bit AdamW (bitsandbytes)** | **Required** | Saves ~8 GB vs FP32; 2B × 16B = 32 GB without it |
| **Gradient checkpointing** | **Required** | Activation memory otherwise OOM |
| LR schedule | **WSD** — stable at 3×10⁻⁴ for Stage 1, decay to 0 in Stage 2 | Reusable checkpoints; upweight Python in decay phase |
| Training tokens | 16B (2 epochs over 8B unique) — extend if time allows | ~28–44 days on RTX 5090 @ 5–8K tok/s |
| Flash Attention 2 | Yes | Mandatory on Blackwell |
| torch.compile | Yes | 15–30% throughput gain |
| Static VRAM | ~20 GB with 8-bit AdamW + grad checkpointing | Feasible on 32 GB |

**Expected target metric:** ~35–45% HumanEval pass@1. Lower than phi-1 (50.6%) because: smaller 8B token budget vs phi-1's 50B effective tokens; no synthetic textbook data. Can improve by training more epochs if time allows.

---

**Techniques to implement (both phases):**
- [ ] Flash Attention 2
- [ ] Gradient checkpointing
- [ ] `torch.compile` + CUDA graphs
- [ ] FP8 training (Phase 2 only; Blackwell-native)
- [ ] GQA (Phase 1: GQA-3, Phase 2: GQA-8)
- [ ] RoPE with high base frequency (θ=500K for Phase 2)
- [ ] FIM training objective (PSM + SPM modes, 50% rate)
- [ ] 8-bit AdamW via bitsandbytes (Phase 2 required)
- [ ] WSD learning rate schedule (Phase 2)
- [ ] NTK-aware RoPE context extension (Phase 2 stage 2, optional)

**Phase 3 — Post-Training (Optional)**

- SFT on Python instruction data (OSS-Instruct, Magicoder)
- GRPO-based reasoning (rasbt/reasoning-from-scratch)
- Evaluate: HumanEval pass@1, MBPP, MultiPL-E Python

### Key References for Implementation

| Topic | Reference |
|---|---|
| Phase 1 model config | [SmolLM2-360M config.json](https://huggingface.co/HuggingFaceTB/SmolLM2-360M/resolve/main/config.json) |
| Phase 2 model config (closest) | [SmolLM2-1.7B config.json](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B/resolve/main/config.json) |
| **Primary dataset** | [HuggingFaceTB/smollm-corpus python-edu](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) |
| **English NL dataset** | [HuggingFaceFW/fineweb-edu sample-10BT](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) |
| FIM training | [arXiv:2207.14255](https://arxiv.org/abs/2207.14255) |
| 8-bit optimizer | [HF bitsandbytes docs](https://huggingface.co/docs/bitsandbytes/main/en/optimizers) |
| VRAM formula | [Lyceum Tech](https://lyceum.technology/magazine/predict-vram-usage-pytorch-model/) |
| Minimal training loop reference | Giles Thomas [Part 28](https://www.gilesthomas.com/2025/12/llm-from-scratch-28-training-a-base-model-from-scratch) |
| SmolLM3 training playbook | [HuggingFace blog](https://huggingface.co/blog/smollm3) |

### Decisions Made

- [x] **No ablations.** Use published defaults throughout. Architecture is settled; data mix is settled.
- [x] **No synthetic data generation.** phi-1's CodeTextbook is not publicly available and we are not generating it ourselves.
- [x] **Primary dataset:** `smollm-corpus python-edu` (644 MB, 4B tokens, score ≥4). Ready to use, no agreement.
- [x] **English NL component:** FineWeb-Edu (50/50 split). Pure code-only causes 86% drop in reasoning ([arXiv:2408.10914](https://arxiv.org/abs/2408.10914)).
- [x] **Token budget:** 8B unique tokens × 2 epochs = 16B effective tokens. ~4 days for 360M, ~28–44 days for 2B.
- [x] **Tokenizer:** SmolLM2 49K BPE + 4 FIM tokens added at load time = **49156 vocab**. FIM tokens are not in the base SmolLM2 vocab and must be injected via `add_special_tokens()`. All model configs use `vocab_size=49156`.
- [x] **FIM ratio:** 50% CLM / 50% FIM — standard, no ablation needed.
- [x] **LR schedule:** Cosine for Phase 1. WSD for Phase 2 (stable → decay with upweighted Python).
- [x] **Two-stage training (Phase 2):** Stage 1 50/50 mix → Stage 2 decay upweights python-edu.
- [x] **Data ordering:** Sort files by dependency order before packing (DeepSeek-Coder approach).

### Open Questions

- [ ] **Training framework:** Implement from scratch (learning value) or use existing framework (Nanotron/TorchTitan) for throughput? Decision needed before starting Phase 1.
- [ ] **FP8 training:** RTX 5090 Blackwell supports natively — benchmark BF16 vs FP8 throughput for Phase 2.
- [ ] **Context extension:** Short NTK-scaled 8K stage after Phase 2 pretraining — worth the extra time?
- [ ] **Evaluation cadence:** Run HumanEval pass@1 every N checkpoints. What N? Automated with LightEval.
- [ ] **Epochs for Phase 2:** Start with 2 epochs (~28–44 days). If results are below target, add more. Accept iterative approach.

---

## 8. Implementation Log

### Plan Updates (pre-implementation review vs SmolLM handbook)

Before writing any code, the plan was reviewed against the SmolLM handbook. Material differences corrected:

| Issue | Old plan | Corrected |
|---|---|---|
| LR Phase 1 (135M) | 6e-4 | **1e-3** (handbook default) |
| LR Phase 2 (360M) | 5e-4 | **1e-3** (handbook default) |
| LR schedule Phase 1+2 | cosine | **WSD** (all phases use WSD per handbook) |
| Weight decay | uniform (incl. embeddings) | **exclude embeddings** — param group split |
| Intra-document masking | missing | **added** — `build_doc_mask` in transformer; pack/loader emit `doc_ids` |
| Gradient norm logging | missing | **added** — log `gnorm` each step; early warning for loss spikes |
| Validation set | missing | **added** — 1% python-edu holdout, val loss at each checkpoint |

The scratchpad "Decisions Made" entry `LR schedule: Cosine for Phase 1. WSD for Phase 2` is now superseded — WSD is used for all three phases.

---

### Step 1 — Environment Setup (2026-03-21)

**GPU / CUDA:**
- RTX 5090 confirmed present; CUDA driver 580.95.05, CUDA runtime 12.8 (via torch)
- **No CUDA toolkit installed** (`nvcc` missing) — cannot compile CUDA extensions from source
- This blocks source-build of flash-attn; pre-built wheels required

**Python / torch:**
- Python 3.13 in venv
- torch **2.8.0** installed (initially 2.10.0, downgraded — see flash-attn below)
- bitsandbytes 0.49.2 ✓

**flash-attn:**
- No pre-built wheel exists for torch 2.10 / Python 3.13 (flash-attn 2.8.3 tops out at torch 2.8)
- Installed: `flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp313-cp313-linux_x86_64.whl`
- Required downgrading torch to **2.8.0** (torch 2.10 wheel caused undefined symbol `c10_cuda_check_implementation`)
- `requirements.txt` pins `torch==2.8.0` to lock this

**Additional dependency discovered:**
- `accelerate` is required by lm-eval-harness HuggingFace model backend but not listed in original requirements
- Added to `requirements.txt`

**Upgrade path:** When a flash-attn wheel for torch >2.8 / cp313 is released, can unpin torch and re-test.

---

### Step 2 — Eval Harness Setup (2026-03-21)

**lm-eval-harness CLI breaking changes** (version installed: lm-eval ≥0.4.3):

| Old (plan) | New (actual) |
|---|---|
| `lm_eval --model ...` | `lm_eval run --model ...` |
| `--allow_code_execution` | `--confirm_run_unsafe_code` |
| (not needed) | `HF_ALLOW_CODE_EVAL=1` env var required |

`scripts/evaluate.py` updated to use correct CLI. Sets `HF_ALLOW_CODE_EVAL=1` automatically.

**Baseline result — SmolLM2-135M on HumanEval:**

```
pass@1 = 0.0  (greedy decoding, n=1)
```

Expected: greedy pass@1 for small models is typically 0 or near-0. Official SmolLM2-135M HumanEval numbers use temperature sampling with n=200 samples. Our trained models will be evaluated the same way (greedy, n=1) for a fair apples-to-apples comparison across checkpoints.

---

### Step 3 — Dataset Download (2026-03-21)

| Dataset | Expected (plan) | Actual |
|---|---|---|
| python-edu examples | ~4M | **7.7M** |
| python-edu size | 644 MB | **947 MB** |
| fineweb-edu examples | ~10M | 9.7M ✓ |
| fineweb-edu size | 28 GB | **46 GB** |

python-edu is nearly 2× the expected example count. Token budget calculations in the plan assumed the smaller figure. Actual unique token count will be higher — the 16B effective token budget may be achievable in fewer epochs, or we can run longer for better coverage.

**Impact on step counts:** Plan configs set `total_steps` based on 16B tokens / tokens-per-step. These are still valid targets; we just have more data available than assumed.

---

### Step 4 — Tokenizer (2026-03-21)

**Critical finding: FIM tokens not in SmolLM2 vocab.**

The plan stated "FIM tokens are already in its vocabulary" — this is **incorrect** for all SmolLM2 variants (135M, 360M, 1.7B). None contain `<|fim_prefix|>`, `<|fim_suffix|>`, `<|fim_middle|>`, or `<|fim_pad|>`.

**Fix:** `load_tokenizer()` adds them as special tokens via `add_special_tokens()`.

**Downstream impact — vocab size changes from 49152 → 49156:**
- All model configs must use `vocab_size=49156`
- The plan's config blocks reference `vocab_size: 49152` (from SmolLM2-135M config.json) — must be updated when implementing `src/model/config.py`

Also updated "Decisions Made":
- ~~SmolLM2 49K BPE~~ → SmolLM2 49K BPE + 4 FIM tokens = **49156 vocab**
