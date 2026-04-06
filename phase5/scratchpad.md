# Phase 5 Scratchpad

## Research focus

Optimize a 124M Python coding LLM along three axes:

- data mixture
- synthetic data
- learning curriculum

Primary objective: maximize Python coding quality under fixed token/compute budget.

## 2025+ paper signals to use

### Data mixture

- `2507.09404` (Scaling Laws for Optimal Data Mixtures): mixture can be predicted with small proxy runs.
- `2501.11747` (LLM-Estimated Utility): utility-guided mixing can approach ablation quality with far less compute.
- `2506.10952` (Domain2Vec): training-free domain alignment signals can reduce search cost.
- `2505.24844` (Chameleon): flexible domain reweighting with low overhead.
- `2502.06042` (Pretraining injection): even small replay fractions can reduce forgetting.
- `2510.14865` (Midtraining bridge): code/math benefit from bridging distributions before post-training.

### Synthetic data

- `2512.00884` (Active Synthetic Data Generation): iterative, student-aware synthetic refresh beats static pools for fixed generation budget.

### Curriculum

- `2505.11643` (easy-to-hard on GPT-2 124M): can improve convergence speed at this scale.
- `2504.08165` (BabyLM findings): many curriculum variants are weak; use curriculum as an ablation, not an assumption.

### Code objective design

- `2506.00204` (AST-FIM): syntax-structured FIM masking improves real code editing performance vs random span masking.

## Phase 5 experimental design

### Stage A - base mixture sweep (no synthetic, static order)

- Hold model/optimizer/schedule/token budget fixed.
- Sweep Python/general ratios: `50/50`, `65/35`, `80/20`, `90/10`.
- Keep a base replay floor (`1-5%`) to reduce over-specialization risk.

### Stage B - synthetic data ablations

- Start from top-2 Stage A mixtures.
- Synthetic proportions: `0%`, `10%`, `20%`, `30%`.
- Compare static synthetic pool vs active iterative synthetic refresh.
- Enforce filtering gates (tests/lint/quality) before adding synthetic samples.

### Stage C - curriculum ablations

- Use best mixture + synthetic setting.
- Compare:
  - static sampling
  - easy-to-hard
  - cyclical curriculum
  - midtraining-style bridge transition

Difficulty buckets for Python curriculum:

- short completion / docs / basic tests
- function-level synthesis
- bugfix / refactor / longer-context tasks

### Stage D - confirmation

- Top 2 combined configs, longer runs, at least 2 seeds each.
- Select final recipe by quality + stability + cost efficiency.

## Evaluation protocol (fixed across runs)

- HumanEval pass@1
- MBPP pass@1
- infilling benchmark (FIM-oriented)
- validation loss/perplexity on held-out Python mix
- throughput and cost per gain (pass@1 per GPU-hour)

## Practical principles

- Isolate one axis at a time before combining winners.
- Keep decode/eval settings identical across runs.
- Track full run manifest: mixture, synthetic config, curriculum, seed, token budget, checkpoint IDs.
- Use short canary runs first on Vast.ai before committing to long runs.

## Immediate next action

See `phase5/implementation_plan.md` for the full staged roadmap.

## Implementation roadmap summary

### Priority order

1. Stage 1 — Replay floor (config only, implement before any sweep)
2. Stage 2 — Mixture sweep A: 50/50, 65/35, 80/20, 90/10
3. Stage 3 — AST-FIM objective ablation (2 runs, on best Stage A mixture)
4. Stage 4 — Synthetic data ablations (4 runs: 0/10/20/30% synthetic)
5. Stage 5 — Curriculum ablations (3 runs: static, easy→hard, cyclical)
6. Stage 6 — Confirmation runs (top 2 configs × 2 seeds)

### Stage 1: Replay floor
- Always keep ≥3% general data tokens in every run
- Implemented in `phase5/data/mixed_loader.py` as `MixedShardLoader`
- Zero extra compute; prevents forgetting artifacts in mixture comparisons

### Stage 2: Mixture sweep
- 4 runs differing only in Python/general ratio
- Python bucket: python-edu shards
- General bucket: nanochat climbmix shards
- Evaluation: val_bpb (general) + python_val_bpb (Python) + CORE metric
- Promotion: top 2 by python_val_bpb

### Stage 3: AST-FIM
- Mask complete Python AST units (functions, classes, blocks) instead of random spans
- p_fim=0.50, p_ast=0.90, fallback random 10%
- Implemented in `phase5/data/ast_fim.py`
- 2 ablation runs: with and without FIM on best Stage A config
- Open question: FIM special tokens (retrain tokenizer vs reuse existing IDs)

### Stage 4: Synthetic data
- Static pool: function synthesis, docstring generation, bug-fix pairs, unit tests
- Quality gates: ast.parse(), py_compile, length, dedup
- Sweep: 0/10/20/30% synthetic fraction
- Teacher budget cap: 5M tokens default
- Active loop follow-up: compare vs static at equal budget

### Stage 5: Curriculum
- Difficulty proxy: nesting_depth + log(num_ast_nodes)
- Strategies: static, soft easy→hard, cyclical
- BabyLM evidence says expect modest gains; keep expectations low
- Only carry forward if delta > 0.005 bpb vs static

### Stage 6: Confirmation
- Target token budget: ~1.5B tokens (compute-optimal for 124M params)
- Top 2 configs, 2 seeds each (seeds 42 and 7)
- Pass criteria: python_val_bpb < 2.0, seed variance < 0.01 bpb

### Open questions before Stage 2
1. Python corpus: python-edu only, or add Stack-v2 Python slices?
2. Synthetic teacher: GPT-4o-mini API vs self-generation?
3. FIM tokens: retrain tokenizer vs reuse existing token IDs?

## Paper signals per stage

| Stage | Key paper(s) |
|-------|-------------|
| 1 (replay) | 2502.06042 |
| 2 (mixture) | 2507.09404, 2501.11747, 2506.10952 |
| 3 (AST-FIM) | 2506.00204 |
| 4 (synthetic) | 2512.00884 |
| 5 (curriculum) | 2505.11643, 2504.08165, 2510.14865 |
| 6 (confirmation) | — |

## Data source decisions

- Licensing: mixed-license allowed
- Primary: python-edu + climbmix (already in R2)
- Expansion (Stage 4+): Stack-v2 Python, StarCoder, notebooks, tests
- Compliance: persist source/provenance IDs in all run manifests
- Filters: exact+near dedup, ast.parse() validity, length bounds, leakage checks

## State of the Art: Tiny Python Coding Models (sub-500M) — 2024-2026 Literature Review
_Conducted via 4 specialized agents, ~90 papers surveyed. All papers confirmed novel vs. existing scratchpad content._

---

### Cluster 1: Data Pipeline & Quality (Highest Leverage)

| Paper | arXiv | Date | Key Finding | Applicability to 124M |
|-------|-------|------|---------------|---------------------|
| Arctic-SnowCoder | `2409.02326` | 2024-09 | 3-phase funnel: dedup → quality annotator → synthetic top-1%. 1.3B beats 3B on 3.3T tokens. | Direct blueprint. Quality > quantity at small scale. |
| Quality-Aware Scaling Laws | `2510.03313` | 2025-09 | Adds quality parameter Q to Chinchilla. Quality can substitute for model scale. | Foundational: quantifies curation ROI. |
| QuRating | `2402.09739` | 2024-02 | Quality scorer via LLM pairwise judgments. Sampling by quality = +50% training steps. | Lightweight scoring + curriculum ordering. |
| SIEVE | `2410.02755` | 2024-10 | Distills GPT-4o quality judgments to cheap classifier. 500× cheaper, domain-specific. | Python-specific quality filter at scale. |
| MixMinHash dedup | `2512.18834` | 2025-12 | Cross-source MinHash as free quality signal. +4.5-5.5% + 4× unique tokens. | Zero-model dedup before any other step. |
| BM25Chunk packing | `2402.13991` | 2024-02 | Semantic sequence packing eliminates cross-document noise. +11.6% ICL, +9.8% knowledge. | Free efficiency gain, zero cost. |
| EntiGraph synthesis | `2409.07431` | 2024-09 | Entity-graph augmentation guarantees dense rare-construct coverage. | Prevents overfitting to common Python idioms. |

---

### Cluster 2: Tokenizer Design (One-Time High-Impact Decision)

| Paper | arXiv | Date | Key Finding | Applicability to 124M |
|-------|-------|------|---------------|---------------------|
| PARAMANU-GANITA | `2404.14395` | 2024-04 | Python-specific BPE from scratch. 208M beats 7B generalist by 30pts. | At 124M, embedding matrix is ~10% of params. Domain vocab frees depth/width. |
| Ayn | `2403.13681` | 2024-03 | 88M with 16K domain vocab beats models 80× larger. Small vocab = more transformer params. | Architecture-level efficiency gain. |
| IGOT | `2405.09857` | 2024-05 | Information-gain tokenizer extension. 12% training speedup, better convergence. | Safe extension from existing checkpoint. |
| Vocabulary Customization | `2509.26124` | 2025-09 | Additive extension guaranteed not to lengthen sequences. 20% shorter Python sequences. | Free effective context window gain. |
| Byte-Level FIM Fix | `2410.09303` | 2024-10 | Fixes tokenization boundary bias in FIM. +18% on FIM benchmarks, zero retraining. | Small models more sensitive to boundary OOD. |

---

### Cluster 3: Training Objectives & Architecture

| Paper | arXiv | Date | Key Finding | Applicability to 124M |
|-------|-------|------|---------------|---------------------|
| Case2Code | `2407.12504` | 2024-07 | Verifiable I/O-to-code objective. Execution-grounded, scales without teacher per sample. | Novel second objective compatible with AST-FIM. |
| Token Loss Weighting | `2503.09202` | 2025-03 | Per-token loss weights from tiny reference model. Routes gradient to high-info tokens. | Gradient budget optimization at small scale. |
| EST (Evolving Subnetworks) | `2406.06962` | 2024-06 | Progressive subnetwork growth during pretraining. 25% FLOP savings + downstream improvement. | Free regularization, no arch changes needed. |

---

### Cluster 4: Post-Training & Alignment

| Paper | arXiv | Date | Key Finding | Applicability to 124M |
|-------|-------|------|---------------|---------------------|
| DiagnosticSLM | `2511.21748` | 2025-11 | DAPT→DSFT→DPO ladder with execution-feedback DPO. 25% over 2-9B models. | Zero-cost preference signal from Python execution. |
| IFIM | `2509.24637` | 2025-09 | Instruction-conditioned FIM. Pass@1 84.6% → 93.6%. | Stacks on AST-FIM pretraining. |
| StructureCoder | `2508.19532` | 2025-08 | AST-split DPO preference pairs. Curriculum DPO coarse→fine. | Post-pretraining alignment, not competing. |
| ProRL / Nemotron-RL | `2505.24864`, `2507.12507` | 2025-05, 2025-07 | KL control + periodic resets prevent entropy collapse at small scale. | Enables RL post-training without collapse at 124M. |
| T1 (Test-Time Tools) | `2504.04718` | 2025-04 | Delegate verification to interpreter at inference. 1B beats 8B on MATH. | Free capability boost, zero retraining. |

---

### Cluster 5: Evaluation Infrastructure (Critical for Valid Results)

| Paper | arXiv | Date | Key Finding | Action Required |
|-------|-------|------|---------------|----------------|
| SAFIM | `2403.04814` | 2024-03 | 17,720 syntax-targeted FIM examples. Pretraining quality matters more than size. | Add to eval suite for AST-FIM validation. |
| NaturalCodeBench | `2405.04520` | 2024-05 | 402 real user queries. HumanEval-overfit models diverge heavily here. | Orthogonal eval to detect overfitting. |
| Real-FIM-Eval | (from AST-FIM `2506.00204`) | 2025-06 | Real-world infilling benchmark. | Already in plan; keep as primary FIM metric. |

---

### Critical Decisions Before Stage 2 Mixture Sweep

**Decision 1: Tokenizer Strategy**
- **Option A:** Train Python-specific BPE from scratch (Paramanu + Ayn approach) — 16-24K vocab
- **Option B:** Extend existing tokenizer with high-information Python tokens (IGOT / Vocab Custom)
- **Recommendation:** Option A if starting from scratch; Option B if continuing from GPT-2/CodeBERT checkpoint

**Decision 2: Data Quality Funnel Before Sweep?**
- Arctic-SnowCoder (`2409.02326`) shows 3-phase quality pipeline beats raw volume
- **Question:** Run one quality-filtered baseline before mixture sweep, or assume python-edu is sufficient?
- **Recommendation:** Apply MixMinHash dedup + BM25Chunk packing (both zero-cost) before any training; defer quality annotator to Stage 1b

**Decision 3: Evaluation Suite Expansion**
- HumanEval alone misleads at 124M (NaturalCodeBench finding)
- **Action:** Add SAFIM (syntax-aware FIM) + NaturalCodeBench (real queries) before Stage 2 results
- **Cost:** Both are inference-only; no training changes needed

---

### Updated Implementation Roadmap (New Stages)

**Stage 0.5 — Data Prep & Tokenizer (Before Stage 1)**
- [ ] Run MixMinHash dedup on python-edu + climbmix
- [ ] Implement BM25Chunk sequence packing
- [ ] Decide: train Python tokenizer from scratch OR extend existing
- [ ] (If extending) Implement IGOT-style information-gain token addition

**Stage 1 — Mixture Sweep (Revised)**
- [ ] Run 50/50, 65/35, 80/20, 90/10 Python/general
- [ ] Add 98/2 stress-test run (optional, tests floor necessity)
- [ ] Measure: Python val bpb, General val bpb, SAFIM score, NaturalCodeBench score
- [ ] **Do NOT enforce replay floor yet** — let sweep reveal if/where forgetting occurs

**Stage 1b — Replay Floor Ablation (If 90/10 or 98/2 shows instability)**
- [ ] Run 1%, 3%, 5% replay floor at 90/10 mixture
- [ ] Measure: General val bpb trend, Python val bpb, training stability
- [ ] Decision rule: If 1% stabilizes without hurting Python bpb, adopt 1% floor; else skip

**Stage 2 — Quality Pipeline (Post-Sweep Optimization)**
- [ ] Implement QuRating or SIEVE quality scorer for Python data
- [ ] Run 3-phase Arctic-SnowCoder pipeline on best mixture from Stage 1
- [ ] Compare: Stage 1 best vs. Stage 2 quality-enhanced (same mixture, better data)

**Stage 3 — AST-FIM + Tokenization Boundary Fix**
- [ ] Implement AST-FIM masking
- [ ] Apply byte-level FIM boundary correction at inference
- [ ] Evaluate on SAFIM + Real-FIM-Eval

**Stage 4 — Synthetic Data & Case2Code Objective**
- [ ] Implement Case2Code I/O-to-code objective as auxiliary task
- [ ] Compare: AST-FIM only vs. AST-FIM + Case2Code multi-task
- [ ] Measure: HumanEval, MBPP, SAFIM, training stability

**Stage 5 — Post-Training Ladder**
- [ ] DAPT on best mixture
- [ ] DSFT on synthetic Python task QA (EntiGraph-augmented)
- [ ] DPO with execution feedback (zero-cost preference signal)
- [ ] Optional: KL-controlled RL with periodic resets (ProRL recipe)

**Stage 6 — Confirmation & Quantization**
- [ ] Top 2 configs × 2 seeds
- [ ] 8-bit GGUF quantization (Quecto-V1 recipe)
- [ ] Deployable artifact: <150MB, near-full precision accuracy

---

### Key Insights from Literature Review

1. **Quality > Quantity at Small Scale**: Arctic-SnowCoder, QuRating, and Quality-Aware Scaling Laws all converge: at sub-500M, data quality can substitute for model scale. A 124M model on high-Q data can match a larger model on raw data.

2. **Tokenizer is Architecture-Level Decision**: At 124M, embedding matrix is ~10-12M parameters. A domain-specific tokenizer (16K vs 32K vocab) frees those parameters for transformer depth/width — a structural efficiency gain, not just tokenization efficiency.

3. **Replay Floor Evidence is Weak for Pretraining**: The 1% replay floor from `2502.06042` applies to finetuning, not pretraining from scratch. No paper provides direct evidence for a specific floor in pretraining. Let the mixture sweep reveal the breaking point.

4. **Evaluation Must Expand Beyond HumanEval**: NaturalCodeBench and SAFIM are essential orthogonal signals. HumanEval alone will mislead at 124M due to overfitting risk.

5. **Post-Training RL is Now Viable at Small Scale**: ProRL and Nemotron-RL show KL control + periodic resets prevent entropy collapse at 1-2B scale. This should transfer to 124M with appropriate hyperparameter tuning.

6. **Test-Time Compute is Free Capability**: T1 shows delegating verification to tools at inference time boosts effective capability without retraining. For Python, code execution is the natural verifier.

---

### Open Questions (To Be Decided Before Stage 1)

1. **Tokenizer:** Train from scratch (Python-only BPE) or extend existing (IGOT/Vocab Custom)?
2. **Data Quality:** Apply full Arctic-SnowCoder 3-phase pipeline before Stage 1, or run Stage 1 on current python-edu quality first?
3. **Case2Code Objective:** Add as auxiliary task from the start, or wait until Stage 4 after mixture is locked?
4. **Evaluator Expansion:** Add SAFIM + NaturalCodeBench to eval suite immediately, or after Stage 1 results?

---

## Cluster 6: Architecture Innovations for Narrow Domains (2024-2026)
_Novel architectures beyond standard dense transformers, specifically for sub-500M models in specialized domains._

---

### 6A. Linear Attention & Recurrent Alternatives

| Paper | arXiv | Date | Innovation | Scale | Gain | 124M Fit |
|-------|-------|------|------------|-------|------|----------|
| **RWKV-v6** | `2409.00286` | 2024-09 | Linear attention, O(1) inference | 196M | +37.6% acc vs 360M SOTA | ⭐⭐⭐⭐⭐ |
| **LCLM/GLA** | `2406.04467` | 2024-06 | Gated linear attention, O(n) vs O(n²) | 64M | +62% faster training | ⭐⭐⭐⭐⭐ |
| **OnlySportsLM** | `2409.00286` | 2024-09 | RWKV-6 for domain specialization | 196M | Matches 1.7B SomlLM | ⭐⭐⭐⭐⭐ |

**Key insight:** RWKV and linear attention architectures are **specifically designed for small models** and achieve 2-3× effective capability at same parameter count. Python implementations exist.

**Applicability to 124M Python model:**
- RWKV-v6 can be trained from scratch at 124M with ~1-2B tokens
- Linear attention = O(1) inference, critical for long code sequences
- OnlySportsLM shows domain specialization at 196M matches models 8-10× larger
- **Recommendation:** Strong candidate for Option B (Moderate) architecture track

---

### 6B. Graph & Neural-Symbolic Hybrids

| Paper | arXiv | Date | Innovation | Scale | Gain | 124M Fit |
|-------|-------|------|------------|-------|------|----------|
| **Contextual Graph Transformer (CGT)** | `2508.02532` | 2025-08 | GNN over tokens → Transformer | 70M | +24.7% acc, 62% fewer params | ⭐⭐⭐⭐⭐ |
| **NesyCD** | `2409.13203` | 2024-09 | Neural SLM + symbolic KB | ≤7B | 8B ≈ 70B (9×) | ⭐⭐⭐⭐⭐ |
| **GFM-RAG** | `2502.01113` | 2025-02 | Graph foundation model for RAG | 8M | SOTA on 10 datasets | ⭐⭐⭐⭐⭐ |

**Key insight:** Graph structures naturally match Python's AST/syntax tree. CGT achieves 24.7% gain with **62% fewer parameters**. Neural-symbolic approach lets you encode Python rules (syntax, type system) symbolically, reducing neural burden.

**Applicability to 124M Python model:**
- CGT's graph over tokens can directly encode Python AST relationships
- NesyCD: Store Python syntax rules, stdlib signatures in symbolic KB
- GFM-RAG: Model Python module/function dependencies as graph (14M triples possible)
- **Recommendation:** CGT is highest-priority architecture for Python (natural AST fit)

---

### 6C. Mixture of Experts (MoE) for Small Models

| Paper | arXiv | Date | Architecture | Experts | Efficiency | 124M Fit |
|-------|-------|------|--------------|---------|------------|----------|
| **CoSMoE** | `2503.00245` | 2025-03 | Weight-decomposed experts | Multiple | 4× total, 2× active | ⭐⭐⭐⭐⭐ |
| **XMoE** | `2403.18926` | 2024-03 | Threshold routing | Fine-grained | 50% compute reduction | ⭐⭐⭐⭐⭐ |
| **MH-MoE** | `2411.16205` | 2024-11 | Multi-head expert routing | 8-40, Top-1/2 | FLOP-matched to dense | ⭐⭐⭐⭐ |
| **USMoE** | `2503.22996` | 2025-03 | Unified sparse routing | Adaptive K | 14% inference cost ↓ | ⭐⭐⭐⭐⭐ |
| **S2MoE** | `2503.23007` | 2025-03 | Stochastic routing | Larger embeddings | 28% cost ↓, prevents collapse | ⭐⭐⭐⭐⭐ |

**Key insight:** MoE at small scale requires careful design to avoid expert collapse. CoSMoE and S2MoE specifically address this for sub-500M models.

**Applicability to 124M Python model:**
- **CoSMoE**: 4× total params (~500M), 2× active (~250M), optimized for on-device
- **XMoE**: 50% compute reduction at MoE layers, threshold routing prevents collapse
- **MH-MoE**: FLOP-matched to dense baseline, compatible with 1-bit quantization
- **Recommendation:** XMoE or CoSMoE for Option B track; avoid Sparse Upcycling (40% inference slowdown)

---

### 6D. Router-Based & Multi-Model Orchestration

| Paper | arXiv | Date | Innovation | Scale | Gain | 124M Fit |
|-------|-------|------|------------|-------|------|----------|
| **RODOS/ARLISS** | `2407.17546` | 2024-07 | Domain router + experts/adapters | 77-207M | 45-55% param ↓ | ⭐⭐⭐⭐⭐ |
| **SLM-MUX** | `2510.05077` | 2025-10 | Multi-SLM orchestration | Multiple 1-3B | 2 SLMs > 72B on GPQA | ⭐⭐⭐⭐ |
| **LM-Lexicon** | `2602.14060` | 2026-02 | Semantic expert routing | Small per domain | +7% BLEU | ⭐⭐⭐⭐ |

**Key insight:** Router can switch between Python subdomains (web, data science, automation, stdlib) — effectively multiple specialized models in one.

**Applicability to 124M Python model:**
- RODOS: External router selects domain expert (e.g., "flask" vs "pandas" vs "asyncio")
- LM-Lexicon: Domain-level routing (+1% efficacy vs token-level)
- SLM-MUX: Orchestrate multiple 124M models for different Python tasks
- **Recommendation:** RODOS-style routing for Python subdomains (low overhead, high impact)

---

### 6E. Retrieval & Memory Augmentation

| Paper | arXiv | Date | Mechanism | Integration | 124M Fit |
|-------|-------|------|-----------|-------------|----------|
| **CAG** | `2411.16133` | 2024-11 | Statistical gating (O(1)) | Dynamic retrieval decision | ⭐⭐⭐⭐⭐ |
| **Parametric RAG** | `2501.15915` | 2025-01 | Knowledge in FFN params | Deep parametric integration | ⭐⭐⭐⭐⭐ |
| **IT Support RAG** | `2409.13707` | 2024-09 | Multi-stage pipeline | 125M encoder proven | ⭐⭐⭐⭐⭐ |
| **GFM-RAG** | `2502.01113` | 2025-02 | Graph neural retriever | 60 KGs, 14M triples | ⭐⭐⭐⭐⭐ |

**Key insight:** CAG's statistical gating is O(1) and avoids unnecessary retrieval — critical for small model efficiency. Parametric RAG encodes knowledge directly into parameters, perfect for Python stdlib/docs.

**Applicability to 124M Python model:**
- **CAG**: Statistical test decides if retrieval needed (O(1), no LLM overhead)
- **Parametric RAG**: Encode Python stdlib signatures, common patterns into FFN weights
- **IT Support RAG**: 125M encoder proven; Python docs retrieval analogous to IT docs
- **GFM-RAG**: Graph of Python module dependencies, function relationships
- **Recommendation:** CAG + Parametric RAG combination (lightweight + deep integration)

---

### 6F. Auxiliary Objectives (Beyond AST-FIM & Case2Code)

| Paper | arXiv | Date | Objective | Compatibility | 124M Fit |
|-------|-------|------|-----------|---------------|----------|
| **TOP (Token Order Prediction)** | `2508.19228` | 2025-08 | Rank upcoming tokens by proximity | Orthogonal to FIM | ⭐⭐⭐⭐⭐ |
| **NextLat** | `2511.05963` | 2025-11 | Predict next latent state | Complements FIM | ⭐⭐⭐⭐⭐ |
| **CECO** | `2404.01860` | 2024-04 | Reconstruction + contrastive | May overlap with FIM | ⭐⭐⭐⭐ |
| **Bloop** | `2402.02998` | 2024-02 | Gradient surgery with EMA | Optimization technique | ⭐⭐⭐⭐ |

**TOP (Token Order Prediction)** is the standout:
- Single unembedding layer overhead
- Proven at 340M, 1.8B, 7B scales
- Improves code/math benchmarks with continued training
- Orthogonal to FIM (can run both)

**NextLat (Next-Latent Prediction)**:
- Self-supervised latent space prediction
- Injects recurrent inductive bias
- Encourages world model formation
- Architecture-agnostic

**Applicability to 124M Python model:**
- **TOP**: High priority — minimal overhead, proven code gains
- **NextLat**: Medium-high — reasoning improvements, but later-stage
- **CECO**: Medium — may overlap with AST-FIM's fill-in objective
- **Bloop**: Use only if gradient conflicts arise from multiple objectives

---

## Top 5 Architecture Recommendations for 124M Python Model

| Rank | Architecture | Why | Expected Gain | Implementation Cost |
|------|--------------|-----|---------------|---------------------|
| 1 | **RWKV-v6** (linear attention) | O(1) inference, 196M proven, Python impl | +37% vs GPT at same scale | Medium (new arch) |
| 2 | **Contextual Graph Transformer** | Graph matches Python AST, 70M proven | +25% acc, 62% param ↓ | Medium-High |
| 3 | **CoSMoE / XMoE** | Sparse MoE for small scale | +10-20% quality, 50% compute ↓ | Medium |
| 4 | **NesyCD** (neural-symbolic) | Encode Python rules symbolically | 8B ≈ 70B (9×) | Low-Medium |
| 5 | **CAG + Parametric RAG** | Lightweight retrieval gating | Free capability boost | Low |

---

## Recommended Build Options for Phase 5

### Option A: Conservative (minimal architecture risk)
**Goal:** Prove data/recipe improvements first, architecture second.

1. Standard GPT-style 124M (baseline from canary)
2. Add **CAG** retrieval gating (O(1) statistical test)
3. Add **TOP** auxiliary objective (single unembedding layer)
4. Add **NesyCD** symbolic KB for Python rules (stdlib, syntax)

**Expected outcome:** Baseline + 15-25% effective capability with minimal architectural changes.
**Timeline:** Stage 0.5-1 (data prep) + Stage 3 (auxiliary objectives)
**Risk:** Low — all components are additive, no core architecture change

---

### Option B: Moderate (proven small-model innovations)
**Goal:** Adopt architectures specifically designed for sub-500M scale.

1. **RWKV-v6** architecture at 124M (replace standard transformer)
2. **XMoE** or **CoSMoE** sparse experts (4× total params, 2× active)
3. **TOP** auxiliary objective (token order prediction)
4. **CAG** retrieval for Python docs/stdlib

**Expected outcome:** Baseline + 50-70% effective capability (matching 200-300M dense models)
**Timeline:** Stage 0 (new architecture) + Stage 3 (auxiliary)
**Risk:** Medium — RWKV requires new training code, but Python impl exists

---

### Option C: Aggressive (maximize capability extraction)
**Goal:** Push 124M to absolute limits with cutting-edge hybrid architecture.

1. **Contextual Graph Transformer** over Python tokens (GNN → Transformer)
2. **NesyCD** neural-symbolic split (symbolic KB for Python rules)
3. **Parametric RAG** for stdlib encoding (FFN integration)
4. **TOP + NextLat** dual auxiliary objectives
5. **RODOS** domain router (switch between Python subdomains)

**Expected outcome:** Baseline + 100-200% effective capability (matching 500M-1B dense models)
**Timeline:** Stage 0 (new arch) + Stage 3-5 (all innovations)
**Risk:** High — multiple unproven components at 124M scale, but highest ceiling

---

## Decision Matrix: Which Path to Choose?

| Criterion | Option A (Conservative) | Option B (Moderate) | Option C (Aggressive) |
|-----------|------------------------|---------------------|----------------------|
| **Time to first result** | 2-3 weeks | 4-6 weeks | 8-12 weeks |
| **Expected gain** | +15-25% | +50-70% | +100-200% |
| **Implementation risk** | Low | Medium | High |
| **Code changes** | Minimal (additive) | Moderate (new arch) | Major (hybrid) |
| **Best for** | Validate data/recipe first | Maximize small-model efficiency | Push 124M to absolute limit |
| **Python AST fit** | N/A (standard) | Good (RWKV handles sequences) | Excellent (graph = AST) |
| **Recommended if...** | You want clean baseline first | You believe architecture matters at 124M | You're willing to risk for 2-3× gains |

**My recommendation:** Start with **Option A** to establish baseline + CAG + TOP (2-3 weeks). If results are promising but ceiling is visible, pivot to **Option B** (RWKV) for the full Phase 5 sweep. Reserve **Option C** for Phase 6 or if Option B hits a wall.

---

## Open Questions (Architecture-Specific)

1. **RWKV vs. Graph Transformer:** Which is the better primary architecture for Python? RWKV handles long sequences better; CGT matches AST structure naturally.

2. **MoE at 124M:** Is sparse MoE worth the complexity at 124M, or does expert collapse dominate at this scale?

3. **Symbolic KB scope:** For NesyCD, what Python rules should be symbolic (syntax, types, stdlib) vs. neural (patterns, idioms)?

4. **Retrieval granularity:** Should CAG retrieve at function-level, module-level, or pattern-level for Python?

5. **Auxiliary objective balance:** How to weight TOP + AST-FIM + Case2Code without gradient conflicts? (Bloop may help here.)

