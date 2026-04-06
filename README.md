# SLM — Python-Specialized Small Language Model

A from-scratch implementation of a Python-specialized language model trained in three phases, following the [SmolLM training handbook](https://huggingface.co/blog/smollm). No training frameworks — plain PyTorch throughout.

## Architecture

Llama-style decoder-only transformer with:

- **GQA** (Grouped Query Attention) for efficient KV cache
- **SwiGLU** FFN
- **RMSNorm** pre-norm
- **RoPE** positional encoding
- **FIM** (Fill-in-Middle) training objective — 50% of examples rearranged in PSM or SPM format
- **Intra-document attention masking** — packed sequences block cross-document attention via `doc_ids`
- **BF16** mixed precision, **Flash Attention 2** (optional), **torch.compile**

## Training Phases

| Phase | Size | Tokens | Schedule | Time (RTX 5090) |
|-------|------|--------|----------|-----------------|
| 1 | 135M | 16B | WSD, lr=1e-3 | ~2 days |
| 2 | 360M | 16B | WSD, lr=1e-3 | ~4 days |
| 3 | 2B   | 16B | WSD, lr=3e-4 | ~28–44 days |

All phases use the **SmolLM2 49K BPE tokenizer** extended with 4 FIM tokens (vocab size 49,156).

## Phase 5 — nanochat Baseline (Active)

A parallel experiment track using [karpathy/nanochat](https://github.com/karpathy/nanochat) as the training framework to rapidly iterate on architecture and data mixture ideas before committing them to the full Llama-based pipeline.

**Current run:** nanochat depth=12 (~124M params), 715 steps on ClimbMix-400B (25 shards, ~1.53B tokens).

| Field | Value |
|-------|-------|
| **W&B** | https://wandb.ai/mlashcorp/nanochat/runs/xljut36a |
| **Instance** | Vast.ai RTX 3090 @ $0.11/hr |
| **Details** | `phase5/baseline/README.md` |

**Planned experiments (ranked by expected gain / complexity):**

1. Data mixture sweep — add python-edu, sweep Python % ratios
2. AST-FIM — syntax-aware fill-in-middle objective
3. CAG — contextual action gating (free retrieval decision)
4. TOP — token order prediction auxiliary objective
5. XMoE — sparse expert routing (50% compute reduction)

## Project Layout

```
slm/
├── src/
│   ├── model/
│   │   ├── config.py        # ModelConfig — phase1_135m / phase2_360m / phase3_2b
│   │   ├── norm.py          # RMSNorm
│   │   ├── rope.py          # RoPE (precompute_freqs_cis, apply_rotary_emb)
│   │   ├── attention.py     # GroupedQueryAttention
│   │   ├── mlp.py           # SwiGLU FFN
│   │   ├── block.py         # TransformerBlock (pre-norm + residual + grad ckpt)
│   │   └── transformer.py   # Full model — embeddings, N blocks, LM head
│   ├── data/
│   │   ├── tokenizer.py     # Load SmolLM2 tokenizer, inject FIM tokens
│   │   ├── fim.py           # FIM transformation (PSM + SPM)
│   │   ├── pack.py          # Greedy sequence packing with doc_ids
│   │   └── loader.py        # MixedPackedDataset + make_dataloader
│   └── training/
│       ├── schedule.py      # cosine_lr, wsd_lr (warmup-stable-decay)
│       ├── checkpoint.py    # save/load — includes HF LlamaForCausalLM remapping
│       └── trainer.py       # Training loop — grad accum, BF16, WSD, logging
├── scripts/
│   ├── download_data.py         # Download FineWeb-Edu (metadata + text)
│   ├── download_python_edu.py   # Hydrate python-edu from Software Heritage S3
│   ├── train.py                 # Entry point — phase 1/2/3
│   └── evaluate.py              # lm-eval-harness wrapper (HumanEval + MBPP)
├── tests/                       # 35 tests, all passing
└── evals/                       # Baseline results
```

## Setup

**Requirements:** Python 3.11+, CUDA 12+, ~50 GB disk for data.

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.8.0
pip install flash-attn --no-build-isolation   # or use pre-built wheel
pip install -r requirements.txt
pip install -e .
```

## Data

The training mix is 50/50 python-edu and FineWeb-Edu.

**FineWeb-Edu** (220B tokens, general text):
```bash
python scripts/download_data.py
```

**python-edu** (4B tokens, high-quality Python code): the HuggingFace dataset stores only blob IDs; source code must be fetched from the Software Heritage S3 bucket (public, anonymous access):

```bash
# Quick sample (~5 min, good for end-to-end testing)
python scripts/download_python_edu.py --sample 150000 --output ./data/python_edu_hydrated

# Full dataset (~6 hrs from inside AWS us-east-1, longer from outside)
python scripts/download_python_edu.py --output ./data/python_edu_hydrated

# Resume an interrupted download
python scripts/download_python_edu.py --output ./data/python_edu_hydrated --resume
```

## Training

```bash
# Phase 1 — 135M proxy model (~2 days)
python scripts/train.py --phase 1

# Resume from checkpoint
python scripts/train.py --phase 1 --resume ./checkpoints/phase1/step-00020000

# Phase 2 — 360M intermediate (~4 days, optional)
python scripts/train.py --phase 2

# Phase 3 — 2B target (~28-44 days)
python scripts/train.py --phase 3
```

Expected output (Phase 1, after warmup):
```
step    50 | loss 9.19 | lr 9.00e-04 | gnorm 2.38 | 22.6K tok/s
step   100 | loss 7.83 | lr 1.00e-03 | gnorm 1.10 | 28.3K tok/s
```

Initial loss should be near `ln(49156) ≈ 10.8` and decrease steadily. The WSD schedule holds LR constant for the bulk of training then applies a cosine decay in the final ~13% of steps.

**W&B logging** is disabled by default. To enable:

1. Get your API key from **wandb.ai → Settings → API keys**, then login:
   ```bash
   # Interactive (prompts for key)
   wandb login

   # Non-interactive
   wandb login <your-api-key>

   # Or via env var (no persistent storage)
   export WANDB_API_KEY=<your-api-key>
   ```
2. Set `"wandb_enabled": True` in the relevant phase config in `scripts/train.py`.

## Evaluation

```bash
python scripts/evaluate.py \
  --checkpoint ./checkpoints/phase1/step-00061035 \
  --output ./evals/phase1_final.json
```

Checkpoints are saved in HuggingFace `LlamaForCausalLM` format so lm-eval-harness can load them directly.

**Baseline** (SmolLM2-135M, no Python specialization):

| Task | pass@1 |
|------|--------|
| HumanEval | see `evals/baseline_smollm2_135m_*.json` |

## Tests

```bash
pytest tests/ -v   # 35 tests
```

## Key Design Decisions

**Why FIM?** Fill-in-Middle training lets the model learn to complete code given both prefix and suffix context — critical for IDE autocompletion use cases.

**Why intra-document masking?** Greedy packing concatenates multiple documents into one sequence. Without `doc_ids`, tokens in document N can attend to tokens in document N-1, leaking context across document boundaries.

**Why WSD over cosine?** The Warmup-Stable-Decay schedule (SmolLM handbook default) keeps LR constant for most of training, which is more stable than continuous cosine decay. The sharp decay phase at the end also makes it easy to continue training from any checkpoint — just run another WSD cycle.

**Why 49,156 vocab size?** SmolLM2's base tokenizer has 49,152 tokens but does not include FIM special tokens. We add `<|fim_prefix|>`, `<|fim_suffix|>`, `<|fim_middle|>`, `<|fim_pad|>` at load time, extending the vocab to 49,156.
