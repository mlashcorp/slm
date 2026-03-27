# OLMo References

This folder contains reference materials from the Allen AI OLMo (Open Language Model) project for use in developing the SLM (Small Language Model) project.

## Directory Structure

```
references/
├── papers/                    # Paper abstracts and summaries
│   ├── olmo2-abstract.md     # OLMo 2 paper summary
│   └── olmo3-abstract.md     # OLMo 3 paper summary
├── configs/                   # Training configuration files
│   ├── olmo2-1b-stage1.yaml  # 1B model stage 1 (large-scale pretraining)
│   ├── olmo2-1b-stage2-seed42.yaml  # 1B model stage 2 (high-quality data)
│   ├── olmo2-7b-stage1.yaml  # 7B model stage 1
│   ├── olmo2-7b-stage2-seed42.yaml  # 7B model stage 2
│   └── olmo2-13b-stage1.yaml # 13B model stage 1
├── data/                      # Dataset documentation
│   ├── dolma-dataset-card.md # Dolma 3T token pretraining corpus
│   └── dolmino-mix-1124-card.md  # High-quality stage 2 data
└── code-snippets/             # Code reference implementations
    ├── olmo-training-loop.md     # Training loop structure
    ├── olmo-lr-schedule.md       # Learning rate schedules (WSD, cosine)
    └── olmo-checkpoint-saving.md # Checkpoint management
```

## Key Takeaways for SLM

### Architecture
- OLMo 2 uses: GQA, SwiGLU, RMSNorm, RoPE (same as SLM)
- 1B model: 16 layers, 2048 hidden, 16 heads, 8 KV heads (GQA ratio 2)

### Training Schedule
- Two-stage training: Stage 1 (web data) → Stage 2 (high-quality data)
- WSD-style schedule with stable phase before decay
- Linear warmup for 2000 steps
- LR: 4e-4 for 1B, 3e-4 for 7B

### Data
- Dolma: 3T tokens, diverse sources (web, code, papers, books)
- Dolmino-mix: High-quality subset for stage 2 (50-100B tokens)
- Educational scoring for quality filtering

### Checkpointing
- Save every 1000 steps
- Track: tokens seen, best eval loss, scheduler state
- Symlink to latest checkpoint

## External Links

- OLMo GitHub: https://github.com/allenai/OLMo
- OLMo-core: https://github.com/allenai/OLMo-core
- Dolma Dataset: https://huggingface.co/datasets/allenai/dolma
- Dolmino-mix: https://huggingface.co/datasets/allenai/dolmino-mix-1124
- OLMo 2 Paper: https://arxiv.org/abs/2501.00656
- OLMo 3 Paper: https://arxiv.org/abs/2512.13961
- Playground: https://playground.allenai.org

## Last Updated

2026-03-24
