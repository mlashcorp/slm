# Dolmino Mix 1124 Dataset

**HuggingFace:** https://huggingface.co/datasets/allenai/dolmino-mix-1124  
**License:** ODC-BY

## Overview

Dolmino Mix 1124 is a **high-quality, domain-focused data mixture** used in the **Stage 2 (annealing phase)** of OLMo 2 training. It significantly improves model capabilities across many downstream task benchmarks when introduced via late-stage curriculum training.

## Purpose

Used for:
- **Late-stage curriculum training** (annealing phase of pretraining)
- **Domain-focused fine-tuning** after initial large-scale pretraining
- Improving downstream task benchmarks

## Subsets

The dataset contains multiple subsets:

| Subset | Rows | Purpose |
|--------|------|---------|
| `dclm` | ~63.6M | High-quality web content |
| `flan` | - | Instruction data |
| `math` | ~4.7M | Mathematical content |
| `pes2o` | ~95.9M | Academic papers |
| `stackexchange` | ~2.47M | Q&A content |
| `wiki` | ~3.24M | Wikipedia content |

## Key Features

- **Educational scoring** — Fineweb-edu-classifier scores applied
- **Quality filtering** — Higher quality threshold than Dolma v1
- **Domain-specific** — Focused on math, code, academic content
- **Token repetitions** — Deduplication metrics tracked

## Usage in OLMo 2

1. **Stage 1:** Train on OLMo-mix-1124 (large-scale web data) → ~4T tokens
2. **Stage 2:** Continue training on Dolmino-mix-1124 (high-quality data) → 50-100B tokens
3. **Model Soup:** Average checkpoints from multiple seeds trained on Stage 2

## Relation to SLM Project

This is similar to the **educational Python filtering** approach used in SmolLM's python-edu dataset, where a classifier is used to select high-quality educational content from a larger corpus.

## Citation

See Dolma citation for the underlying methodology. Dolmino-mix-1124 is an extension of the Dolma pipeline for late-stage curriculum training.
