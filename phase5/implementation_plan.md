# Phase 5 Implementation Plan

**Goal:** Optimize a 124M Python coding LLM along three axes — data mixture, synthetic
data, and learning curriculum — under a fixed token/compute budget using Karpathy's
nanochat as the training backbone.

---

## Stage 0 — Infra & Baseline Freeze (DONE)

All infrastructure is validated and frozen.

- E2E smoke flow: download → R2 → Vast → SSH, all working.
- Canary run completed: depth=8, ctx=1024, 300 steps, val bpb=2.23.
- Launcher (`Opencode-Remote/ml/launch.py`) with SSH readiness, R2 sync, offer fallback.
- Scripts: `phase5/e2e/`, `phase5/canary/`.

**Frozen baseline recipe:**
- `phase5/canary/vast_canary_config.yaml` — immutable baseline.
- Model: depth=8, context=1024, Muon+AdamW optimizer, warmup/cosine schedule.
- Eval: nanochat CORE metric + val bpb every 100 steps.

---

## Stage 1 — Replay Floor

**Status:** Not started.  
**Priority:** Implement before any Stage A sweep run.

### What

Always keep a small fixed fraction of general/non-Python tokens in every training
step, regardless of the target Python/general ratio.

### Why

Without a replay floor, high Python ratios (90/10, 80/20) can silently cause
catastrophic forgetting of general linguistic priors. This contaminates mixture
comparisons. At zero extra compute cost, a 3% floor prevents this.

Reference: `2502.06042` (Scaling Laws for Forgetting with Pretraining Data Injection).

### Implementation

**File to create:** `phase5/data/mixed_loader.py`

```python
class MixedShardLoader:
    """
    Wraps nanochat's parquets_iter_batched with a configurable
    multi-source interleaver.

    Args:
        sources: list of (parquet_dir, weight) tuples.
        replay_floor: minimum fraction of any single source (default 0.03).
        split: "train" or "val".
    """
```

The loader:
1. Accepts a list of `(shard_dir, weight)` pairs.
2. Normalises weights so they sum to 1.
3. Enforces `replay_floor`: no source falls below this fraction after normalisation.
4. At each step, samples the next document from a source chosen proportionally.

**Config knob added to all YAML configs:**

```yaml
data:
  sources:
    - dir: "phase5/e2e/nanochat/base_data_climbmix"
      weight: 0.80
    - dir: "phase5/e2e/nanochat/python_edu"
      weight: 0.20
  replay_floor: 0.03
```

**Training script integration:**  
- Pass `replay_floor` to `MixedShardLoader` at init.  
- Log actual per-source sampling fraction to W&B every 500 steps.

**Success criterion:** Training log confirms each source is sampled within ±0.5% of
target fraction at every checkpoint step.

---

## Stage 2 — Mixture Sweep (Stage A)

**Status:** Not started.  
**Depends on:** Stage 1 complete.

### What

4 training runs, identical in every way except the Python/general data ratio.

| Run ID | Python % | General % | Notes |
|--------|----------|-----------|-------|
| A1     | 50       | 50        | Equal split baseline |
| A2     | 65       | 35        | Moderately Python-heavy |
| A3     | 80       | 20        | Strongly Python-heavy |
| A4     | 90       | 10        | Near-maximum Python (replay floor still enforced) |

Python bucket: `python-edu` shards from `HuggingFaceTB/smollm-corpus`.  
General bucket: nanochat `climbmix` shards.

### Implementation

**Files to create:**

```
phase5/sweeps/stage_a/
  run_A1.yaml
  run_A2.yaml
  run_A3.yaml
  run_A4.yaml
  launch_sweep_a.sh
  remote_train_sweep.sh
```

`run_A{n}.yaml` structure:

```yaml
run_id: "phase5_A3_80_20"
data:
  sources:
    - dir: "python_edu"
      weight: 0.80
    - dir: "base_data_climbmix"
      weight: 0.20
  replay_floor: 0.03
model:
  depth: 8
  max_seq_len: 1024
training:
  num_iterations: 2000
  device_batch_size: 2
  total_batch_size: 2048
  save_every: 500
  eval_every: 200
  eval_tokens: 65536
```

`remote_train_sweep.sh` reads the YAML and calls `scripts.base_train` via the
`MixedShardLoader`. W&B run names encode the ratio (e.g. `phase5_A3_80_20`).

### Evaluation Protocol (fixed for all A runs)

1. `val_bpb` on nanochat ClimbMix val shard (general quality).
2. `python_val_bpb` on held-out Python-edu val shard (Python quality).
3. nanochat CORE metric at final step.
4. Log all three to W&B for each run.

### Promotion Criterion

Top 2 runs by `python_val_bpb` proceed to Stage 3.

---

## Stage 3 — AST-FIM Objective

**Status:** Not started.  
**Depends on:** Stage 2 complete (winner known).

### What

Replace random-span fill-in-the-middle (FIM) with syntax-aware masking where spans
align to Python AST units (functions, classes, blocks).

Reference: `2506.00204` (Structure-Aware Fill-in-the-Middle Pretraining for Code).

### Why

Random FIM masks arbitrary byte ranges, which creates many semantically incoherent
span boundaries. Masking complete syntactic units (a whole function body, a class,
an if-block) produces training examples closer to real IDE edit-style tasks, which
is the primary use case for a Python coding model.

### Implementation

**File to create:** `phase5/data/ast_fim.py`

```python
def ast_fim_transform(source: str, p_fim: float = 0.5, p_ast: float = 0.9) -> str:
    """
    Given a Python source string, return a FIM-transformed version.

    With probability p_fim, apply FIM transformation.
    Within FIM, with probability p_ast, select an AST-aligned span.
    With probability (1 - p_ast), fall back to random span (robustness).

    Returns string in <PRE>...<SUF>...<MID>... format.

    AST node priority order:
      1. FunctionDef / AsyncFunctionDef (body only, not signature)
      2. ClassDef (body only)
      3. For / While / With / If block body
      4. Any expression statement

    Falls back to raw string if ast.parse() fails (e.g. partial snippets).
    """
```

**Config flags added to training YAML:**

```yaml
fim:
  enabled: true
  p_fim: 0.50        # fraction of documents that get FIM treatment
  p_ast: 0.90        # fraction of FIM docs that use AST-aligned spans
  fallback_random: true
```

**Ablation runs (2 runs, on best Stage A mixture):**

| Run ID | FIM | p_fim | p_ast | Notes |
|--------|-----|-------|-------|-------|
| B0     | off | —     | —     | Stage A winner, no FIM (control) |
| B1     | on  | 0.50  | 0.90  | AST-FIM |

**Promotion criterion:** If B1 `python_val_bpb` < B0, AST-FIM is adopted into all
subsequent stages. Otherwise, no FIM.

---

## Stage 4 — Synthetic Data Ablations

**Status:** Not started.  
**Depends on:** Stage 3 complete (best mixture + FIM decision known).

### What

Add synthetic Python examples to the training mix at varying proportions and compare
against the no-synthetic baseline.

Reference: `2512.00884` (Active Synthetic Data Generation).

### Synthetic types

1. **Static pool (Phase 1):** Pre-generated offline. Prompt types:
   - Function synthesis from docstring.
   - Docstring generation from function body.
   - Bug-fix pairs (introduce a bug, ask to fix it).
   - Unit test completion.

2. **Active loop (Phase 2):** After N training steps, run inference on a seed set,
   identify high-loss examples, generate targeted synthetic continuations, add to
   next epoch's mix. Requires a teacher model and a generation/filter pipeline.

### Implementation

**Files to create:**

```
phase5/data/synthetic_gen.py   — teacher-model generation pipeline
phase5/data/synthetic_filter.py — quality gates (syntax, execution, style)
phase5/sweeps/stage_b/
  run_B0.yaml   (0% synthetic)
  run_B1.yaml   (10% synthetic)
  run_B2.yaml   (20% synthetic)
  run_B3.yaml   (30% synthetic)
  launch_sweep_b.sh
```

`synthetic_gen.py`:

```python
def generate_static_pool(
    seed_file: str,
    output_dir: str,
    teacher: str = "gpt-4o-mini",
    n_per_type: int = 5000,
    max_teacher_tokens: int = 5_000_000,
) -> None:
    """
    Generate a static synthetic pool.
    Hard stops at max_teacher_tokens to cap spend.
    Writes output as parquet, one file per type.
    """
```

Quality gates in `synthetic_filter.py`:
1. `ast.parse()` must succeed (valid Python).
2. `py_compile.compile()` must succeed (no syntax errors).
3. Length check: 50 < tokens < 2048.
4. Dedup: exact hash dedup against training set.

**Ablation design:**

| Run ID | Synthetic % | Type | Notes |
|--------|-------------|------|-------|
| B0     | 0           | —    | Control (best Stage A+FIM config) |
| B1     | 10          | static | Static pool |
| B2     | 20          | static | Static pool |
| B3     | 30          | static | Static pool |

Active loop comparison: run after B0–B3 to validate whether iterative refresh beats
the best static pool under equal teacher-token budget.

**Teacher budget cap:** Set hard cap in `synthetic_gen.py`. Default: `5M tokens`.
Stop when marginal `python_val_bpb` improvement < 0.001 per 1M tokens.

**Promotion criterion:** Best synthetic proportion by `python_val_bpb` proceeds to
Stage 5.

---

## Stage 5 — Curriculum Ablations

**Status:** Not started.  
**Depends on:** Stage 4 complete.

### What

Compare static token ordering vs two soft curriculum strategies.

References:
- `2505.11643` (easy-to-hard on GPT-2 124M).
- `2504.08165` (BabyLM): curriculum rarely dominates at small scale; treat as ablation.

### Difficulty proxy for Python

Computed offline per document using `ast.parse()`:

```
difficulty_score = nesting_depth + log(num_ast_nodes + 1)
```

Bucketed into 3 tiers:
- Easy: score < P33 (short snippets, docstrings, basic tests).
- Medium: P33–P66 (function synthesis, basic algorithms).
- Hard: score > P66 (bugfix, multi-function, longer context).

### Curriculum strategies

| Run ID | Strategy | Description |
|--------|----------|-------------|
| C0     | Static   | No ordering (control, same as Stage 4 winner) |
| C1     | Soft easy→hard | Anneal difficulty weight from easy-heavy to hard-heavy linearly over training |
| C2     | Cyclical | Repeat easy→hard cycle every 500 steps with full mix replay |

### Implementation

**File to create:** `phase5/data/curriculum.py`

```python
def compute_difficulty(source: str) -> float:
    """Returns a scalar difficulty score for a Python source string."""

def build_difficulty_index(shard_dir: str, output_path: str) -> None:
    """Pre-compute and cache difficulty scores for all docs in a shard dir."""

class CurriculumSampler:
    """
    Wraps MixedShardLoader with a difficulty-aware sampling schedule.
    strategy: "static" | "easy_to_hard" | "cyclical"
    """
```

**Files to create:**

```
phase5/sweeps/stage_c/
  run_C0.yaml
  run_C1.yaml
  run_C2.yaml
  launch_sweep_c.sh
```

**Promotion criterion:** Best strategy by `python_val_bpb`. If C0 (static) wins,
curriculum is dropped. Only proceed to Stage 6 with curriculum if gain > 0.005 bpb.

---

## Stage 6 — Confirmation Runs

**Status:** Not started.  
**Depends on:** Stage 5 complete.

### What

Take the top 2 combined configs and run at full token budget with ≥2 seeds each.

**Full token budget:** Compute-optimal for 124M parameters.  
Using nanochat's scaling law: `tokens = target_param_data_ratio × scaling_params`.  
At depth=12 (124M), this is approximately **1.5B tokens**.

### Runs

| Run ID | Config | Seed | Notes |
|--------|--------|------|-------|
| D1a    | Best combined | 42 | |
| D1b    | Best combined | 7  | Seed variance check |
| D2a    | Second best   | 42 | |
| D2b    | Second best   | 7  | Seed variance check |

### Success criterion

- Seed variance (D1a vs D1b): `python_val_bpb` delta < 0.01.
- D1 vs D2 gap significant: delta > 0.02 bpb.
- Final checkpoint passes sanity evals:
  - CORE metric > canary baseline.
  - Python val bpb < 2.0 (target).
  - No catastrophic general regression (general val bpb < canary + 0.05).

---

## Run Naming Convention

```
phase5_{stage}{run_id}_{python_pct}_{general_pct}[_{fim}][_{synth_pct}][_{curriculum}]
```

Examples:
- `phase5_A3_80_20`
- `phase5_B1_80_20_astfim_synth10`
- `phase5_C1_80_20_astfim_synth10_e2h`

---

## Evaluation Protocol (fixed across all stages)

| Metric | When | Notes |
|--------|------|-------|
| `val_bpb` | Every 200 steps | nanochat ClimbMix val shard |
| `python_val_bpb` | Every 200 steps | Held-out Python-edu val shard |
| CORE metric | Every 500 steps | nanochat's eval bundle |
| Peak memory | End of run | |
| Total training time | End of run | |
| tok/sec | Per step | GPU efficiency |

All logged to W&B. All eval settings identical across runs.

---

## Open Questions (decide before Stage 2)

1. **Python bucket corpus:** Use `python-edu` (smollm-corpus) or add Stack-v2 Python
   slices? Current plan: `python-edu` only for simplicity. Revisit in Stage 4 if
   synthetic data adds pressure for more diversity.

2. **Teacher for synthetic:** External API (GPT-4o-mini) vs self-generated via
   rejection sampling? Default plan: external API with hard token budget cap.
   If budget is tight, switch to self-generation from a stronger open checkpoint.

3. **FIM tokens:** nanochat's tokenizer currently uses BOS-only special tokens.
   AST-FIM requires `<PRE>/<SUF>/<MID>` tokens. Options:
   - Retrain tokenizer with FIM tokens (clean but expensive).
   - Reuse existing unused token IDs (fast but hacky).
   - Use sentinel byte sequences that the BPE tokenizer will encode consistently.
   Decision needed before Stage 3.

---

## Files Summary

```
phase5/
  implementation_plan.md        ← this file
  scratchpad.md                 ← research notes and paper signals
  canary/                       ← Stage 0 baseline
  e2e/                          ← smoke test harness
  data/
    mixed_loader.py             ← Stage 1: replay floor + multi-source mixing
    ast_fim.py                  ← Stage 3: AST-aligned FIM transform
    synthetic_gen.py            ← Stage 4: static synthetic generation
    synthetic_filter.py         ← Stage 4: quality gates
    curriculum.py               ← Stage 5: difficulty scoring + curriculum sampler
  sweeps/
    stage_a/                    ← Stage 2: mixture sweep configs + launchers
    stage_b/                    ← Stage 4: synthetic ablation configs + launchers
    stage_c/                    ← Stage 5: curriculum ablation configs + launchers
  results/                      ← per-run manifests and summary CSVs (gitignored)
```
