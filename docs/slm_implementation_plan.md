# SLM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a Python-specialized small language model in three phases: a 135M proxy model (Phase 1, ~2 days) for fast pipeline validation, an optional 360M intermediate model (Phase 2, ~4 days), and a 2B target model (Phase 3, ~28–44 days). Evals established before training begins.

**Architecture:** Llama-style decoder-only transformer with GQA, SwiGLU, RMSNorm pre-norm, RoPE, and FIM training objective. Phase 1 mirrors SmolLM2-135M config exactly. Phase 3 scales to 2B with 8-bit AdamW + gradient checkpointing required. All phases reuse the SmolLM2 49K BPE tokenizer.

**Tech Stack:** PyTorch (plain, no framework), Flash Attention 2, bitsandbytes (8-bit AdamW), lm-eval-harness (HumanEval+/MBPP+), HuggingFace datasets + tokenizers, torch.compile, BF16 mixed precision.

---

## File Map

```
slm/
├── src/
│   ├── model/
│   │   ├── config.py          # ModelConfig dataclass for Phase 1 and Phase 2
│   │   ├── norm.py            # RMSNorm
│   │   ├── rope.py            # RoPE positional encoding
│   │   ├── attention.py       # GQA attention layer
│   │   ├── mlp.py             # SwiGLU FFN
│   │   ├── block.py           # TransformerBlock (attention + FFN + norms)
│   │   └── transformer.py     # Full model: embedding + N blocks + head
│   ├── data/
│   │   ├── tokenizer.py       # Load SmolLM2 tokenizer, add FIM tokens
│   │   ├── fim.py             # FIM transformation (PSM + SPM modes)
│   │   ├── pack.py            # Greedy sequence packing to context length
│   │   └── loader.py          # Dataset streaming + DataLoader
│   ├── training/
│   │   ├── schedule.py        # WSD LR (all phases) + cosine helper used by WSD decay
│   │   ├── checkpoint.py      # Save and load checkpoints
│   │   └── trainer.py         # Training loop, gradient accumulation, logging
│   └── eval/
│       └── run_eval.py        # Thin wrapper: save checkpoint → run lm-eval-harness
├── configs/
│   ├── phase1_135m.yaml       # All Phase 1 hyperparameters
│   ├── phase2_360m.yaml       # Optional Phase 2 hyperparameters
│   └── phase3_2b.yaml         # All Phase 3 hyperparameters
├── scripts/
│   ├── download_data.py       # Download smollm-corpus python-edu + FineWeb-Edu
│   ├── train.py               # Entry point: load config → Trainer.train()
│   └── evaluate.py            # Entry point: load checkpoint → run_eval.py
├── tests/
│   ├── test_model.py          # Shape checks for each model component
│   ├── test_fim.py            # FIM transformation correctness
│   └── test_pack.py           # Sequence packing correctness
└── requirements.txt
```

---

## Phase 0 — Environment & Evals First

### Task 0.1: Install dependencies

- [ ] **Step 1: Create requirements.txt**

```
torch>=2.4.0
transformers>=4.40.0
datasets>=2.18.0
tokenizers>=0.19.0
bitsandbytes>=0.43.0
flash-attn>=2.5.0
lm-eval>=0.4.3
pyyaml
tqdm
wandb
```

- [ ] **Step 2: Install**

```bash
pip install -r requirements.txt
```

- [ ] **Step 3: Verify Flash Attention installed correctly**

```bash
python -c "import flash_attn; print(flash_attn.__version__)"
```

Expected: version string printed (≥2.5.0). If this fails, Flash Attention needs to be built from source for Blackwell: `pip install flash-attn --no-build-isolation`

- [ ] **Step 4: Verify bitsandbytes can see the GPU**

```bash
python -c "import bitsandbytes as bnb; print(bnb.__version__)"
```

- [ ] **Step 5: Create package structure and pyproject.toml**

```bash
mkdir -p src/model src/data src/training src/eval
touch src/__init__.py src/model/__init__.py src/data/__init__.py src/training/__init__.py src/eval/__init__.py
```

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"

[project]
name = "slm"
version = "0.1.0"

[tool.setuptools.packages.find]
where = ["."]
include = ["src*"]
```

Then install in editable mode so all `from src.*` imports resolve without `PYTHONPATH`:

```bash
pip install -e .
```

---

### Task 0.2: Set up eval harness (SmolLM playbook: evals first)

This task sets up and validates lm-eval-harness against a known public model before writing any training code. The goal is to confirm the eval pipeline works and establish baseline numbers to compare against.

- [ ] **Step 1: Verify lm-eval-harness is installed and working**

```bash
lm_eval --help
```

- [ ] **Step 2: Run HumanEval against SmolLM2-135M as a sanity check**

This will take ~5–10 minutes and confirms the eval pipeline end-to-end. SmolLM2-135M is the same architecture as our Phase 1 target so this also gives us a direct comparison baseline.

```bash
lm_eval \
  --model hf \
  --model_args pretrained=HuggingFaceTB/SmolLM2-135M \
  --tasks humaneval \
  --allow_code_execution \
  --num_fewshot 0 \
  --output_path ./evals/baseline_smollm2_135m.json
```

- [ ] **Step 3: Create eval wrapper script**

Create `scripts/evaluate.py`:

```python
"""
Run lm-eval-harness against a checkpoint directory.

Usage:
  python scripts/evaluate.py --checkpoint ./checkpoints/step-10000 --output ./evals/step-10000.json
"""
import argparse
import subprocess
import sys

TASKS = "humaneval,mbpp"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint dir (HF format)")
    parser.add_argument("--output", required=True, help="Path to output JSON")
    parser.add_argument("--tasks", default=TASKS)
    args = parser.parse_args()

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={args.checkpoint}",
        "--tasks", args.tasks,
        "--allow_code_execution",
        "--num_fewshot", "0",
        "--output_path", args.output,
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Commit**

```bash
git add requirements.txt scripts/evaluate.py evals/
git commit -m "feat: add eval harness setup and baseline SmolLM2-135M results"
```

---

## Phase 1 — Data Pipeline

### Task 1.1: Download datasets

- [ ] **Step 1: Create download script**

Create `scripts/download_data.py`:

```python
"""
Download smollm-corpus python-edu and FineWeb-Edu sample-10BT.
Saves to ./data/ as Parquet shards.

Usage: python scripts/download_data.py
"""
from datasets import load_dataset
import os

os.makedirs("./data/python_edu", exist_ok=True)
os.makedirs("./data/fineweb_edu", exist_ok=True)

print("Downloading smollm-corpus python-edu...")
ds = load_dataset(
    "HuggingFaceTB/smollm-corpus",
    "python-edu",
    split="train",
    streaming=False,
)
ds.save_to_disk("./data/python_edu")
print(f"python-edu: {len(ds):,} examples")

print("Downloading FineWeb-Edu sample-10BT...")
fw = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name="sample-10BT",
    split="train",
    streaming=False,
)
fw.save_to_disk("./data/fineweb_edu")
print(f"fineweb-edu: {len(fw):,} examples")
```

- [ ] **Step 2: Run the download** (this may take 30–60 minutes depending on connection)

```bash
python scripts/download_data.py
```

Expected output:
```
python-edu: ~4,000,000 examples
fineweb-edu: ~10,000,000 examples
```

- [ ] **Step 3: Verify data on disk**

```bash
du -sh data/python_edu data/fineweb_edu
```

Expected: python-edu ~644 MB, fineweb-edu ~28 GB.

- [ ] **Step 4: Commit the download script** (not the data)

```bash
echo "data/" >> .gitignore
git add scripts/download_data.py .gitignore
git commit -m "feat: add dataset download script"
```

---

### Task 1.2: Tokenizer setup

- [ ] **Step 1: Write test**

Create `tests/test_tokenizer.py`:

```python
from src.data.tokenizer import load_tokenizer, FIM_TOKENS

def test_fim_tokens_in_vocab():
    tok = load_tokenizer()
    for token in FIM_TOKENS.values():
        assert token in tok.get_vocab(), f"{token} not in vocab"

def test_roundtrip():
    tok = load_tokenizer()
    text = "def hello():\n    return 42\n"
    ids = tok.encode(text)
    assert tok.decode(ids) == text

def test_vocab_size():
    tok = load_tokenizer()
    assert tok.vocab_size == 49152
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_tokenizer.py -v
```

Expected: ImportError or AttributeError — `src/data/tokenizer.py` doesn't exist yet.

- [ ] **Step 3: Implement tokenizer.py**

Create `src/data/tokenizer.py`:

```python
from transformers import AutoTokenizer

FIM_TOKENS = {
    "prefix": "<|fim_prefix|>",
    "suffix": "<|fim_suffix|>",
    "middle": "<|fim_middle|>",
    "pad": "<|fim_pad|>",
}

def load_tokenizer(model_id: str = "HuggingFaceTB/SmolLM2-360M"):
    """Load SmolLM2 tokenizer. FIM tokens are already in its vocabulary."""
    tok = AutoTokenizer.from_pretrained(model_id)
    # Verify FIM tokens exist — SmolLM2 includes them
    for name, token in FIM_TOKENS.items():
        if token not in tok.get_vocab():
            raise ValueError(f"Expected FIM token {token!r} not found in tokenizer vocab. "
                             "Check that you are using the SmolLM2 tokenizer.")
    return tok
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_tokenizer.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/data/tokenizer.py tests/test_tokenizer.py
git commit -m "feat: add tokenizer loader with FIM token validation"
```

---

### Task 1.3: FIM transformation

50% of training examples are rearranged as Fill-in-Middle. Each example either stays as CLM (causal left-to-right) or is transformed into PSM or SPM format.

- [ ] **Step 1: Write test**

Create `tests/test_fim.py`:

```python
import random
from src.data.fim import apply_fim, FimMode

def test_clm_passthrough():
    tokens = [1, 2, 3, 4, 5]
    result = apply_fim(tokens, mode=None, prefix_id=10, suffix_id=11, middle_id=12)
    assert result == tokens

def test_psm_structure():
    # PSM: <PRE> prefix <SUF> suffix <MID> middle
    # Use token IDs that do NOT overlap with the sequence (0-19)
    tokens = list(range(20))
    prefix_id, suffix_id, middle_id = 100, 101, 102
    result = apply_fim(tokens, mode=FimMode.PSM, prefix_id=prefix_id,
                       suffix_id=suffix_id, middle_id=middle_id)
    assert result[0] == prefix_id
    assert suffix_id in result
    assert middle_id in result
    # All original tokens still present
    original_in_result = [t for t in result if t not in (prefix_id, suffix_id, middle_id)]
    assert sorted(original_in_result) == sorted(tokens)

def test_spm_structure():
    tokens = list(range(20))
    prefix_id, suffix_id, middle_id = 100, 101, 102
    result = apply_fim(tokens, mode=FimMode.SPM, prefix_id=prefix_id,
                       suffix_id=suffix_id, middle_id=middle_id)
    assert result[0] == suffix_id
    assert prefix_id in result
    assert middle_id in result

def test_fim_rate():
    random.seed(42)
    tokens = list(range(50))
    prefix_id, suffix_id, middle_id = 10, 11, 12
    fim_count = 0
    N = 1000
    for _ in range(N):
        result = apply_fim(tokens, fim_rate=0.5, prefix_id=prefix_id,
                           suffix_id=suffix_id, middle_id=middle_id)
        if result[0] in (prefix_id, suffix_id):
            fim_count += 1
    # Should be approximately 50%, allow ±5%
    assert 0.45 < fim_count / N < 0.55
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_fim.py -v
```

- [ ] **Step 3: Implement fim.py**

Create `src/data/fim.py`:

```python
import random
from enum import Enum
from typing import Optional

class FimMode(Enum):
    PSM = "psm"  # Prefix-Suffix-Middle
    SPM = "spm"  # Suffix-Prefix-Middle

def apply_fim(
    tokens: list[int],
    fim_rate: float = 0.5,
    prefix_id: int = None,
    suffix_id: int = None,
    middle_id: int = None,
    mode: Optional[FimMode] = None,  # None = auto-sample
) -> list[int]:
    """
    Apply Fill-in-Middle transformation to a token sequence.

    With probability fim_rate, pick a random split point and rearrange as:
      PSM: [prefix_id, prefix_tokens, suffix_id, suffix_tokens, middle_id, middle_tokens]
      SPM: [suffix_id, suffix_tokens, prefix_id, prefix_tokens, middle_id, middle_tokens]

    With probability 1 - fim_rate, return tokens unchanged (CLM).
    """
    if mode is None:
        if random.random() >= fim_rate:
            return tokens
        mode = random.choice([FimMode.PSM, FimMode.SPM])

    if mode is None:
        return tokens

    n = len(tokens)
    if n < 4:
        return tokens  # too short to split meaningfully

    # Pick two split points: prefix|middle|suffix
    i = random.randint(1, n - 2)
    j = random.randint(i + 1, n - 1)

    prefix = tokens[:i]
    middle = tokens[i:j]
    suffix = tokens[j:]

    if mode == FimMode.PSM:
        return [prefix_id] + prefix + [suffix_id] + suffix + [middle_id] + middle
    else:  # SPM
        return [suffix_id] + suffix + [prefix_id] + prefix + [middle_id] + middle
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_fim.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/data/fim.py tests/test_fim.py
git commit -m "feat: add FIM transformation (PSM + SPM modes, 50% rate)"
```

---

### Task 1.4: Sequence packing

Short documents are concatenated (with an `<|endoftext|>` separator) to fill the context window. This avoids wasting compute on padding.

- [ ] **Step 1: Write test**

Create `tests/test_pack.py`:

```python
from src.data.pack import pack_sequences

def test_packs_to_exact_length():
    seqs = [[1, 2, 3], [4, 5], [6, 7, 8, 9], [10]]
    sep = 0
    packed = pack_sequences(seqs, max_len=6, sep_id=sep)
    for chunk in packed:
        assert len(chunk) == 6

def test_no_data_loss():
    seqs = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    sep = 0
    packed = pack_sequences(seqs, max_len=6, sep_id=sep)
    all_tokens = [t for chunk in packed for t in chunk if t != sep]
    assert sorted(all_tokens) == list(range(1, 10))

def test_long_sequence_truncated():
    seqs = [list(range(100))]
    packed = pack_sequences(seqs, max_len=10, sep_id=0)
    for chunk in packed:
        assert len(chunk) == 10
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_pack.py -v
```

- [ ] **Step 3: Implement pack.py**

Create `src/data/pack.py`:

```python
from typing import Iterator

def pack_sequences(
    sequences: list[list[int]],
    max_len: int,
    sep_id: int,
) -> list[tuple[list[int], list[int]]]:
    """
    Greedily pack token sequences into fixed-length chunks.
    Each sequence is separated by sep_id. Long sequences are truncated.
    Returns only complete chunks of exactly max_len tokens.

    Returns list of (tokens, doc_ids) tuples. doc_ids[i] is the document
    index within the packed chunk for token i — used to build intra-document
    attention masks (tokens in different documents must not attend to each other).
    """
    chunks = []
    current: list[int] = []
    current_doc_ids: list[int] = []
    doc_idx = 0

    for seq in sequences:
        # Truncate if single sequence is longer than max_len
        if len(seq) > max_len:
            seq = seq[:max_len]

        # Add separator before sequence (except if current is empty)
        if current:
            to_add = [sep_id] + seq
            doc_ids_to_add = [doc_idx] + [doc_idx + 1] * len(seq)
            doc_idx += 1
        else:
            to_add = seq
            doc_ids_to_add = [doc_idx] * len(seq)

        if len(current) + len(to_add) <= max_len:
            current.extend(to_add)
            current_doc_ids.extend(doc_ids_to_add)
        else:
            # Flush current if non-empty, then start new chunk
            if current:
                if len(current) == max_len:
                    chunks.append((current, current_doc_ids))
                doc_idx += 1
                current = seq
                current_doc_ids = [doc_idx] * len(seq)
            else:
                current = seq
                current_doc_ids = [doc_idx] * len(seq)

        if len(current) == max_len:
            chunks.append((current, current_doc_ids))
            current = []
            current_doc_ids = []
            doc_idx += 1

    # Drop the last incomplete chunk (no padding)
    return chunks


def pack_dataset_streaming(
    dataset,
    tokenizer,
    max_len: int,
    fim_rate: float = 0.5,
    text_field: str = "text",
) -> Iterator[tuple[list[int], list[int]]]:
    """
    Stream a HuggingFace dataset, tokenize, apply FIM, and yield (tokens, doc_ids) pairs.
    doc_ids[i] is the document index for token i within the packed chunk.
    """
    from src.data.fim import apply_fim
    from src.data.tokenizer import FIM_TOKENS

    vocab = tokenizer.get_vocab()
    prefix_id = vocab[FIM_TOKENS["prefix"]]
    suffix_id = vocab[FIM_TOKENS["suffix"]]
    middle_id = vocab[FIM_TOKENS["middle"]]
    sep_id = tokenizer.eos_token_id

    token_buffer: list[int] = []
    doc_id_buffer: list[int] = []
    doc_idx = 0

    for example in dataset:
        text = example[text_field]
        tokens = tokenizer.encode(text, add_special_tokens=False)
        tokens = apply_fim(tokens, fim_rate=fim_rate,
                           prefix_id=prefix_id, suffix_id=suffix_id, middle_id=middle_id)
        tokens = tokens + [sep_id]
        doc_ids = [doc_idx] * len(tokens)
        doc_idx += 1

        token_buffer.extend(tokens)
        doc_id_buffer.extend(doc_ids)

        while len(token_buffer) >= max_len:
            yield token_buffer[:max_len], doc_id_buffer[:max_len]
            token_buffer = token_buffer[max_len:]
            doc_id_buffer = doc_id_buffer[max_len:]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_pack.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/data/pack.py tests/test_pack.py
git commit -m "feat: add greedy sequence packing with FIM integration"
```

---

### Task 1.5: DataLoader

- [ ] **Step 1: Implement loader.py**

Create `src/data/loader.py`:

```python
"""
Mixed dataloader: streams python-edu and fineweb-edu at a 50/50 ratio,
applies FIM, packs to context length, and yields batches.
"""
import random
from pathlib import Path
from datasets import load_from_disk
from torch.utils.data import IterableDataset, DataLoader
import torch
from src.data.pack import pack_dataset_streaming
from src.data.tokenizer import load_tokenizer


class MixedPackedDataset(IterableDataset):
    def __init__(
        self,
        python_edu_path: str,
        fineweb_edu_path: str,
        max_len: int,
        fim_rate: float = 0.5,
        seed: int = 42,
    ):
        self.python_edu_path = python_edu_path
        self.fineweb_edu_path = fineweb_edu_path
        self.max_len = max_len
        self.fim_rate = fim_rate
        self.seed = seed
        self.tokenizer = load_tokenizer()

    def __iter__(self):
        # Offset seed per worker so each worker produces a distinct data stream
        worker_info = torch.utils.data.get_worker_info()
        seed = self.seed + (worker_info.id if worker_info is not None else 0)
        rng = random.Random(seed)
        py_ds = load_from_disk(self.python_edu_path)
        fw_ds = load_from_disk(self.fineweb_edu_path)

        py_iter = iter(py_ds.shuffle(seed=self.seed))
        fw_iter = iter(fw_ds.shuffle(seed=self.seed))

        py_buf: list[int] = []
        fw_buf: list[int] = []

        sep_id = self.tokenizer.eos_token_id
        vocab = self.tokenizer.get_vocab()

        from src.data.tokenizer import FIM_TOKENS
        from src.data.fim import apply_fim
        prefix_id = vocab[FIM_TOKENS["prefix"]]
        suffix_id = vocab[FIM_TOKENS["suffix"]]
        middle_id = vocab[FIM_TOKENS["middle"]]

        py_doc_buf: list[int] = []
        fw_doc_buf: list[int] = []
        py_doc_idx = 0
        fw_doc_idx = 0

        def next_tokens(it, tok_buf, doc_buf, doc_idx, field):
            if len(tok_buf) >= self.max_len:
                chunk = tok_buf[:self.max_len]
                doc_chunk = doc_buf[:self.max_len]
                return chunk, doc_chunk, tok_buf[self.max_len:], doc_buf[self.max_len:], doc_idx
            try:
                ex = next(it)
                toks = self.tokenizer.encode(ex[field], add_special_tokens=False)
                toks = apply_fim(toks, fim_rate=self.fim_rate,
                                 prefix_id=prefix_id, suffix_id=suffix_id, middle_id=middle_id)
                toks = toks + [sep_id]
                tok_buf.extend(toks)
                doc_buf.extend([doc_idx] * len(toks))
                doc_idx += 1
            except StopIteration:
                pass
            return None, None, tok_buf, doc_buf, doc_idx

        while True:
            # 50/50 mix: alternate sources
            source = rng.choice(["python", "fineweb"])

            if source == "python":
                chunk = None
                while chunk is None and py_iter is not None:
                    chunk, doc_chunk, py_buf, py_doc_buf, py_doc_idx = next_tokens(
                        py_iter, py_buf, py_doc_buf, py_doc_idx, "text")
                    if chunk is None and len(py_buf) < self.max_len:
                        break
                if chunk is None:
                    break
            else:
                chunk = None
                while chunk is None and fw_iter is not None:
                    chunk, doc_chunk, fw_buf, fw_doc_buf, fw_doc_idx = next_tokens(
                        fw_iter, fw_buf, fw_doc_buf, fw_doc_idx, "text")
                    if chunk is None and len(fw_buf) < self.max_len:
                        break
                if chunk is None:
                    break

            yield {
                "input_ids": torch.tensor(chunk, dtype=torch.long),
                "doc_ids": torch.tensor(doc_chunk, dtype=torch.long),
            }


def make_dataloader(
    python_edu_path: str,
    fineweb_edu_path: str,
    max_len: int,
    batch_size: int,
    fim_rate: float = 0.5,
    num_workers: int = 4,
    seed: int = 42,
    val_split: float = 0.01,  # hold out 1% of python-edu as validation
) -> tuple[DataLoader, DataLoader]:
    """Returns (train_loader, val_loader). Val is ~1% of python-edu, no FIM."""
    py_ds = load_from_disk(python_edu_path)
    split = py_ds.train_test_split(test_size=val_split, seed=seed)
    train_py_path = python_edu_path + "_train_split"
    val_py_path = python_edu_path + "_val_split"
    split["train"].save_to_disk(train_py_path)
    split["test"].save_to_disk(val_py_path)

    train_dataset = MixedPackedDataset(
        python_edu_path=train_py_path,
        fineweb_edu_path=fineweb_edu_path,
        max_len=max_len,
        fim_rate=fim_rate,
        seed=seed,
    )
    val_dataset = MixedPackedDataset(
        python_edu_path=val_py_path,
        fineweb_edu_path=fineweb_edu_path,
        max_len=max_len,
        fim_rate=0.0,   # no FIM on validation — cleaner perplexity signal
        seed=seed,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader
```

- [ ] **Step 2: Smoke test the dataloader manually**

```bash
python -c "
from src.data.loader import make_dataloader
dl = make_dataloader('./data/python_edu', './data/fineweb_edu', max_len=2048, batch_size=4)
batch = next(iter(dl))
print('input_ids shape:', batch['input_ids'].shape)   # expect [4, 2048]
print('doc_ids shape:  ', batch['doc_ids'].shape)     # expect [4, 2048]
print('Unique docs in seq 0:', batch['doc_ids'][0].unique().numel())  # expect several
"
```

- [ ] **Step 3: Commit**

```bash
git add src/data/loader.py
git commit -m "feat: add mixed 50/50 dataloader with streaming and FIM"
```

---

## Phase 2 — Model Architecture

### Task 2.1: Model config

- [ ] **Step 1: Implement config.py**

Create `src/model/config.py`:

```python
from dataclasses import dataclass

@dataclass
class ModelConfig:
    # Dimensions
    vocab_size: int = 49152
    hidden_size: int = 960
    intermediate_size: int = 2560
    num_hidden_layers: int = 32
    # Attention
    num_attention_heads: int = 15
    num_key_value_heads: int = 5   # GQA: groups = num_attention_heads / num_key_value_heads
    head_dim: int = 64             # hidden_size / num_attention_heads
    # Context
    max_position_embeddings: int = 2048
    rope_theta: float = 100_000.0
    # Misc
    rms_norm_eps: float = 1e-5
    tie_embeddings: bool = True
    use_flash_attn: bool = False   # Set True for Phase 2
    use_gradient_checkpointing: bool = False  # Set True for Phase 2

    @classmethod
    def phase1_135m(cls) -> "ModelConfig":
        """SmolLM2-135M architecture. Mirrors HuggingFaceTB/SmolLM2-135M config.json exactly.
        ~1 day to train 16B tokens on RTX 5090. Primary iteration/validation model."""
        return cls(
            vocab_size=49152,
            hidden_size=576,
            intermediate_size=1536,
            num_hidden_layers=30,
            num_attention_heads=9,
            num_key_value_heads=3,
            head_dim=64,
            max_position_embeddings=2048,
            rope_theta=100_000.0,
        )

    @classmethod
    def phase2_360m(cls) -> "ModelConfig":
        """SmolLM2-360M architecture. Optional intermediate phase.
        ~4 days to train 16B tokens on RTX 5090. Skip if Phase 1 results are clean."""
        return cls(
            vocab_size=49152,
            hidden_size=960,
            intermediate_size=2560,
            num_hidden_layers=32,
            num_attention_heads=15,
            num_key_value_heads=5,
            head_dim=64,
            max_position_embeddings=2048,
            rope_theta=100_000.0,
        )

    @classmethod
    def phase3_2b(cls) -> "ModelConfig":
        """Custom 2B config derived from SmolLM2-1.7B at 2B scale with GQA-8.
        ~28-44 days on RTX 5090. Requires 8-bit AdamW + gradient checkpointing."""
        return cls(
            vocab_size=49152,
            hidden_size=2048,
            intermediate_size=8192,
            num_hidden_layers=32,
            num_attention_heads=16,
            num_key_value_heads=2,
            head_dim=128,
            max_position_embeddings=4096,
            rope_theta=500_000.0,
            use_flash_attn=True,
            use_gradient_checkpointing=True,
        )
```

- [ ] **Step 2: Verify param counts match expectations**

```bash
python -c "
from src.model.config import ModelConfig

def count_params(cfg):
    embed = cfg.vocab_size * cfg.hidden_size
    # Q + K + V + O projections (o_proj has same size as q_proj)
    attn = cfg.hidden_size * (2 * cfg.num_attention_heads + 2 * cfg.num_key_value_heads) * cfg.head_dim
    ffn = 3 * cfg.hidden_size * cfg.intermediate_size
    return embed + cfg.num_hidden_layers * (attn + ffn)

print(f'Phase 1 (135M): {count_params(ModelConfig.phase1_135m())/1e6:.0f}M')   # expect ~135M
print(f'Phase 2 (360M): {count_params(ModelConfig.phase2_360m())/1e6:.0f}M')   # expect ~360M
print(f'Phase 3  (2B):  {count_params(ModelConfig.phase3_2b())/1e6:.0f}M')     # expect ~2000M
"
```

- [ ] **Step 3: Commit**

```bash
git add src/model/config.py
git commit -m "feat: add ModelConfig for Phase 1 (135M), Phase 2 (360M), Phase 3 (2B)"
```

---

### Task 2.2: RMSNorm

- [ ] **Step 1: Write test**

In `tests/test_model.py`:

```python
import torch
from src.model.norm import RMSNorm

def test_rmsnorm_shape():
    norm = RMSNorm(64)
    x = torch.randn(2, 16, 64)
    out = norm(x)
    assert out.shape == x.shape

def test_rmsnorm_scale():
    norm = RMSNorm(4)
    # All-ones input → output = gamma (1.0 by default)
    x = torch.ones(1, 1, 4)
    out = norm(x)
    assert torch.allclose(out, torch.ones_like(out), atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_model.py::test_rmsnorm_shape tests/test_model.py::test_rmsnorm_scale -v
```

- [ ] **Step 3: Implement norm.py**

Create `src/model/norm.py`:

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_model.py::test_rmsnorm_shape tests/test_model.py::test_rmsnorm_scale -v
```

---

### Task 2.3: RoPE

- [ ] **Step 1: Write test**

Add to `tests/test_model.py`:

```python
from src.model.rope import precompute_freqs_cis, apply_rotary_emb

def test_rope_output_shape():
    seq_len, head_dim = 16, 64
    freqs = precompute_freqs_cis(head_dim, seq_len, theta=10000.0)
    q = torch.randn(2, seq_len, 4, head_dim)
    k = torch.randn(2, seq_len, 4, head_dim)
    q_rot, k_rot = apply_rotary_emb(q, k, freqs)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape

def test_rope_preserves_norm():
    # RoPE is a rotation — it preserves vector norms
    seq_len, head_dim = 8, 32
    freqs = precompute_freqs_cis(head_dim, seq_len, theta=10000.0)
    q = torch.randn(1, seq_len, 2, head_dim)
    k = torch.randn(1, seq_len, 2, head_dim)
    q_rot, k_rot = apply_rotary_emb(q, k, freqs)
    assert torch.allclose(q.norm(dim=-1), q_rot.norm(dim=-1), atol=1e-5)
```

- [ ] **Step 2: Implement rope.py**

Create `src/model/rope.py`:

```python
import torch

def precompute_freqs_cis(
    head_dim: int,
    seq_len: int,
    theta: float = 10_000.0,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Precompute RoPE frequency complex exponentials.
    Returns complex tensor of shape (seq_len, head_dim // 2).
    """
    assert head_dim % 2 == 0
    half = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, freqs)  # (seq_len, half)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE to query and key tensors.
    q, k: (batch, seq_len, n_heads, head_dim)
    freqs_cis: (seq_len, head_dim // 2) complex
    """
    # Reshape to complex
    q_ = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    k_ = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
    # Broadcast freqs: (1, seq_len, 1, head_dim // 2)
    freqs = freqs_cis.unsqueeze(0).unsqueeze(2)
    q_rot = torch.view_as_real(q_ * freqs).flatten(-2)
    k_rot = torch.view_as_real(k_ * freqs).flatten(-2)
    return q_rot.to(q.dtype), k_rot.to(k.dtype)
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_model.py -v -k "rope"
```

---

### Task 2.4: GQA Attention

- [ ] **Step 1: Write test**

Add to `tests/test_model.py`:

```python
from src.model.attention import GroupedQueryAttention
from src.model.config import ModelConfig
from src.model.rope import precompute_freqs_cis

def test_gqa_output_shape():
    cfg = ModelConfig.phase1_135m()
    attn = GroupedQueryAttention(cfg)
    batch, seq = 2, 16
    x = torch.randn(batch, seq, cfg.hidden_size)
    freqs = precompute_freqs_cis(cfg.head_dim, seq, theta=cfg.rope_theta)
    out = attn(x, freqs)
    assert out.shape == (batch, seq, cfg.hidden_size)

def test_gqa_causal_mask():
    cfg = ModelConfig.phase1_135m()
    attn = GroupedQueryAttention(cfg)
    batch, seq = 1, 8
    x = torch.randn(batch, seq, cfg.hidden_size)
    freqs = precompute_freqs_cis(cfg.head_dim, seq, theta=cfg.rope_theta)
    # Perturb token at position 4 — positions 0-3 output should not change
    x2 = x.clone()
    x2[0, 4] += 10.0
    out1 = attn(x, freqs)
    out2 = attn(x2, freqs)
    # Causal: output at positions 0-3 must be identical
    assert torch.allclose(out1[0, :4], out2[0, :4], atol=1e-4)
```

- [ ] **Step 2: Implement attention.py**

Create `src/model/attention.py`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.config import ModelConfig
from src.model.rope import apply_rotary_emb

class GroupedQueryAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.use_flash = config.use_flash_attn
        # n_groups = n_heads / n_kv_heads
        assert self.n_heads % self.n_kv_heads == 0
        self.n_groups = self.n_heads // self.n_kv_heads

        self.q_proj = nn.Linear(self.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,                        # (batch, seq, hidden)
        freqs_cis: torch.Tensor,                # (seq, head_dim // 2) complex
        attn_mask: torch.Tensor | None = None,  # (batch, 1, seq, seq) bool, True = attend
    ) -> torch.Tensor:
        B, S, _ = x.shape

        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim)

        q, k = apply_rotary_emb(q, k, freqs_cis)

        if self.use_flash:
            from flash_attn import flash_attn_func
            # flash_attn expects (B, S, H, D)
            # Expand KV heads to match Q heads
            k = k.repeat_interleave(self.n_groups, dim=2)
            v = v.repeat_interleave(self.n_groups, dim=2)
            # attn_mask not directly supported by flash_attn_func; for doc masking
            # fall back to sdpa when a custom mask is provided
            if attn_mask is not None:
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
                out = out.transpose(1, 2)
            else:
                out = flash_attn_func(q, k, v, causal=True)
        else:
            # Standard scaled dot-product attention (PyTorch 2.0+)
            # Transpose to (B, H, S, D)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            # Expand KV heads for GQA
            k = k.repeat_interleave(self.n_groups, dim=1)
            v = v.repeat_interleave(self.n_groups, dim=1)
            # attn_mask=None → is_causal=True; attn_mask provided → mask already encodes causality
            if attn_mask is not None:
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            else:
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            out = out.transpose(1, 2)  # back to (B, S, H, D)

        out = out.reshape(B, S, self.n_heads * self.head_dim)
        return self.o_proj(out)
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_model.py -v -k "gqa"
```

---

### Task 2.5: SwiGLU FFN

- [ ] **Step 1: Write test**

Add to `tests/test_model.py`:

```python
from src.model.mlp import SwiGLU

def test_swiglu_shape():
    mlp = SwiGLU(hidden_size=960, intermediate_size=2560)
    x = torch.randn(2, 16, 960)
    out = mlp(x)
    assert out.shape == x.shape
```

- [ ] **Step 2: Implement mlp.py**

Create `src/model/mlp.py`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    """
    SwiGLU FFN: FFN(x) = (x·W_gate ⊙ SiLU(x·W_up)) · W_down
    Three weight matrices, no bias.
    """
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

- [ ] **Step 3: Run test**

```bash
pytest tests/test_model.py::test_swiglu_shape -v
```

---

### Task 2.6: TransformerBlock

- [ ] **Step 1: Write test**

Add to `tests/test_model.py`:

```python
from src.model.block import TransformerBlock
from src.model.rope import precompute_freqs_cis

def test_block_shape():
    cfg = ModelConfig.phase1_135m()
    block = TransformerBlock(cfg)
    batch, seq = 2, 16
    x = torch.randn(batch, seq, cfg.hidden_size)
    freqs = precompute_freqs_cis(cfg.head_dim, seq, theta=cfg.rope_theta)
    out = block(x, freqs)
    assert out.shape == x.shape
```

- [ ] **Step 2: Implement block.py**

Create `src/model/block.py`:

```python
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from src.model.config import ModelConfig
from src.model.norm import RMSNorm
from src.model.attention import GroupedQueryAttention
from src.model.mlp import SwiGLU

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.use_grad_ckpt = config.use_gradient_checkpointing
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = GroupedQueryAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)

    def _forward(self, x: torch.Tensor, freqs_cis: torch.Tensor,
                 attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), freqs_cis, attn_mask=attn_mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor,
                attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.use_grad_ckpt and self.training:
            return checkpoint(self._forward, x, freqs_cis, attn_mask, use_reentrant=False)
        return self._forward(x, freqs_cis, attn_mask)
```

- [ ] **Step 3: Run test**

```bash
pytest tests/test_model.py::test_block_shape -v
```

---

### Task 2.7: Full Transformer model

- [ ] **Step 1: Write test**

Add to `tests/test_model.py`:

```python
from src.model.transformer import Transformer

def test_transformer_forward_shape():
    cfg = ModelConfig.phase1_135m()
    model = Transformer(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 16))
    logits = model(input_ids)
    assert logits.shape == (2, 16, cfg.vocab_size)

def test_transformer_param_count():
    cfg = ModelConfig.phase1_135m()
    model = Transformer(cfg)
    total = sum(p.numel() for p in model.parameters())
    # Expect ~135M. Allow ±10%.
    assert 120_000_000 < total < 150_000_000, f"Got {total:,} params"

def test_tied_embeddings():
    cfg = ModelConfig.phase1_135m()
    model = Transformer(cfg)
    assert model.embed_tokens.weight is model.lm_head.weight, "Embeddings not tied"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_model.py -v -k "transformer"
```

- [ ] **Step 3: Implement transformer.py**

Create `src/model/transformer.py`:

```python
import torch
import torch.nn as nn
from src.model.config import ModelConfig
from src.model.norm import RMSNorm
from src.model.block import TransformerBlock
from src.model.rope import precompute_freqs_cis

class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # Precompute RoPE freqs up to max context length
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(config.head_dim, config.max_position_embeddings, theta=config.rope_theta),
            persistent=False,
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    @staticmethod
    def build_doc_mask(doc_ids: torch.Tensor) -> torch.Tensor:
        """
        Build a causal attention mask that also blocks cross-document attention.
        doc_ids: (batch, seq_len) — integer document index per token.
        Returns float mask of shape (batch, 1, seq_len, seq_len):
          0.0 where attention is allowed (same doc + causal), -inf elsewhere.
        """
        B, S = doc_ids.shape
        # same_doc[b, i, j] = True if token i and j belong to the same document
        same_doc = doc_ids.unsqueeze(2) == doc_ids.unsqueeze(1)  # (B, S, S)
        # causal[i, j] = True if j <= i
        causal = torch.ones(S, S, device=doc_ids.device, dtype=torch.bool).tril()
        mask = same_doc & causal.unsqueeze(0)  # (B, S, S)
        # Convert to additive float mask: 0 where allowed, -inf elsewhere
        float_mask = torch.zeros(B, 1, S, S, device=doc_ids.device)
        float_mask.masked_fill_(~mask.unsqueeze(1), float("-inf"))
        return float_mask

    def forward(
        self,
        input_ids: torch.Tensor,              # (batch, seq_len)
        doc_ids: torch.Tensor | None = None,  # (batch, seq_len) for intra-doc masking
    ) -> torch.Tensor:
        B, S = input_ids.shape
        x = self.embed_tokens(input_ids)
        freqs = self.freqs_cis[:S]

        attn_mask = self.build_doc_mask(doc_ids) if doc_ids is not None else None

        for layer in self.layers:
            x = layer(x, freqs, attn_mask=attn_mask)

        x = self.norm(x)
        return self.lm_head(x)  # (batch, seq_len, vocab_size)

    def num_parameters(self, trainable_only: bool = True) -> int:
        params = self.parameters() if not trainable_only else filter(lambda p: p.requires_grad, self.parameters())
        return sum(p.numel() for p in params)
```

- [ ] **Step 4: Run all model tests**

```bash
pytest tests/test_model.py -v
```

All tests should pass. The param count test is the critical one — it confirms the architecture matches spec.

- [ ] **Step 5: Commit all model code**

```bash
git add src/model/
git commit -m "feat: implement full transformer architecture (GQA, SwiGLU, RMSNorm, RoPE)"
```

---

## Phase 3 — Training

### Task 3.1: LR schedules

- [ ] **Step 1: Implement schedule.py**

Create `src/training/schedule.py`:

```python
import math

def cosine_lr(step: int, warmup_steps: int, total_steps: int, max_lr: float, min_lr: float = 0.0) -> float:
    """Cosine decay with linear warmup. Used for Phase 1."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def wsd_lr(step: int, warmup_steps: int, stable_steps: int, decay_steps: int, max_lr: float, min_lr: float = 0.0) -> float:
    """
    Warmup-Stable-Decay schedule. Used for Phase 2.
    - [0, warmup_steps): linear warmup
    - [warmup_steps, warmup_steps + stable_steps): constant max_lr
    - [warmup_steps + stable_steps, ...): cosine decay to min_lr
    """
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step < warmup_steps + stable_steps:
        return max_lr
    decay_step = step - warmup_steps - stable_steps
    progress = min(decay_step / decay_steps, 1.0)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
```

- [ ] **Step 2: Quick sanity check**

```bash
python -c "
from src.training.schedule import cosine_lr, wsd_lr
# Phase 1: 16B tokens at 500K batch = 32000 steps
for step in [0, 500, 1000, 16000, 32000]:
    lr = cosine_lr(step, warmup_steps=1000, total_steps=32000, max_lr=5e-4)
    print(f'step {step:6d}: lr={lr:.2e}')
"
```

---

### Task 3.2: Checkpoint save/load

- [ ] **Step 1: Implement checkpoint.py**

Create `src/training/checkpoint.py`:

```python
import torch
from pathlib import Path
from transformers import PretrainedConfig, AutoConfig

def save_checkpoint(
    model,
    optimizer,
    step: int,
    loss: float,
    config,
    output_dir: str,
):
    """Save model weights, optimizer state, and training metadata."""
    path = Path(output_dir) / f"step-{step:08d}"
    path.mkdir(parents=True, exist_ok=True)

    torch.save({
        "step": step,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }, path / "checkpoint.pt")

    # Also save HF-compatible weights for lm-eval-harness
    save_hf_format(model, config, str(path))
    print(f"Saved checkpoint to {path}")


def load_checkpoint(path: str, model, optimizer=None):
    """Load checkpoint. Returns step number."""
    ckpt = torch.load(Path(path) / "checkpoint.pt", map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt["step"]


def save_hf_format(model, config, output_dir: str):
    """
    Save in HuggingFace format so lm-eval-harness can load it directly.
    Uses LlamaForCausalLM config schema (compatible with our architecture).
    """
    from transformers import LlamaConfig
    import json

    hf_config = LlamaConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        max_position_embeddings=config.max_position_embeddings,
        rope_theta=config.rope_theta,
        rms_norm_eps=config.rms_norm_eps,
        tie_word_embeddings=config.tie_embeddings,
        hidden_act="silu",
        torch_dtype="bfloat16",
    )
    hf_config.save_pretrained(output_dir)

    # Remap our weight names to LlamaForCausalLM weight names.
    # Our module hierarchy already matches HF Llama closely; only top-level
    # embedding/norm need a "model." prefix, and the final norm key needs
    # careful handling to avoid corrupting per-layer norm keys.
    state_dict = model.state_dict()
    remapped = {}
    for k, v in state_dict.items():
        new_k = k
        # Top-level embedding: embed_tokens.weight → model.embed_tokens.weight
        if new_k.startswith("embed_tokens."):
            new_k = "model." + new_k
        # Top-level final norm: norm.weight → model.norm.weight
        # Use exact prefix match to avoid touching input_layernorm / post_attention_layernorm
        elif new_k == "norm.weight":
            new_k = "model.norm.weight"
        # Transformer layers: layers.N.* → model.layers.N.*
        elif new_k.startswith("layers."):
            new_k = "model." + new_k
        # lm_head stays as-is
        remapped[new_k] = v

    torch.save(remapped, Path(output_dir) / "pytorch_model.bin")
```

> **Note:** The weight remapping in `save_hf_format` maps our module names to LlamaForCausalLM names so lm-eval can load the checkpoint via `--model hf`. Verify this mapping once a first checkpoint is saved.

- [ ] **Step 2: Commit**

```bash
git add src/training/schedule.py src/training/checkpoint.py
git commit -m "feat: add LR schedules (cosine + WSD) and checkpoint save/load"
```

---

### Task 3.3: Training loop (Phase 1)

- [ ] **Step 1: Implement trainer.py**

Create `src/training/trainer.py`:

```python
"""
Training loop for all phases: Phase 1 (135M), Phase 2 (360M), Phase 3 (2B).
Handles gradient accumulation, LR schedule, checkpointing, and logging.
"""
import time
import torch
import torch.nn.functional as F
from src.training.schedule import cosine_lr, wsd_lr
from src.training.checkpoint import save_checkpoint


class Trainer:
    def __init__(self, model, dataloader, config: dict, device: str = "cuda",
                 val_dataloader=None):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.cfg = config
        self.device = device

        # Optimizer: split param groups — embeddings get no weight decay
        # (matches SmolLM handbook: weight decay applied to non-embedding params only)
        decay_params = [p for n, p in model.named_parameters() if "embed" not in n and p.requires_grad]
        no_decay_params = [p for n, p in model.named_parameters() if "embed" in n and p.requires_grad]
        param_groups = [
            {"params": decay_params, "weight_decay": config["weight_decay"]},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        if config.get("use_8bit_adam", False):
            import bitsandbytes as bnb
            self.optimizer = bnb.optim.AdamW8bit(
                param_groups,
                lr=config["lr"],
                betas=(config["beta1"], config["beta2"]),
            )
        else:
            self.optimizer = torch.optim.AdamW(
                param_groups,
                lr=config["lr"],
                betas=(config["beta1"], config["beta2"]),
            )

        # torch.compile for throughput
        if config.get("compile", True):
            self.model = torch.compile(self.model)

        # BF16 does not need GradScaler (same dynamic range as FP32, no underflow risk)
        self.global_step = 0

    def _get_lr(self) -> float:
        cfg = self.cfg
        if cfg.get("schedule") == "wsd":
            return wsd_lr(
                self.global_step,
                warmup_steps=cfg["warmup_steps"],
                stable_steps=cfg["stable_steps"],
                decay_steps=cfg["decay_steps"],
                max_lr=cfg["lr"],
            )
        return cosine_lr(
            self.global_step,
            warmup_steps=cfg["warmup_steps"],
            total_steps=cfg["total_steps"],
            max_lr=cfg["lr"],
        )

    def _set_lr(self, lr: float):
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    @torch.no_grad()
    def _eval_loss(self, max_batches: int = 50) -> float:
        """Estimate validation loss over up to max_batches batches."""
        if self.val_dataloader is None:
            return float("nan")
        self.model.eval()
        total, count = 0.0, 0
        for batch in self.val_dataloader:
            if count >= max_batches:
                break
            input_ids = batch["input_ids"].to(self.device)
            doc_ids = batch.get("doc_ids")
            if doc_ids is not None:
                doc_ids = doc_ids.to(self.device)
            inputs = input_ids[:, :-1].contiguous()
            targets = input_ids[:, 1:].contiguous()
            doc_ids_shifted = doc_ids[:, :-1].contiguous() if doc_ids is not None else None
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = self.model(inputs, doc_ids=doc_ids_shifted)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            total += loss.item()
            count += 1
        self.model.train()
        return total / count if count > 0 else float("nan")

    def train(self):
        cfg = self.cfg
        model = self.model
        model.train()

        grad_accum = cfg.get("grad_accum_steps", 1)
        log_every = cfg.get("log_every", 100)
        save_every = cfg.get("save_every", 1000)
        total_steps = cfg["total_steps"]

        tokens_seen = 0
        t0 = time.time()
        micro_step = 0          # independent counter — never resets, drives accumulation

        for batch in self.dataloader:
            if self.global_step >= total_steps:
                break

            input_ids = batch["input_ids"].to(self.device)
            doc_ids = batch.get("doc_ids")
            if doc_ids is not None:
                doc_ids = doc_ids.to(self.device)
            # Shift: inputs = [:-1], targets = [1:]
            inputs = input_ids[:, :-1].contiguous()
            targets = input_ids[:, 1:].contiguous()
            # Trim doc_ids to match shifted sequence length
            doc_ids_shifted = doc_ids[:, :-1].contiguous() if doc_ids is not None else None

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(inputs, doc_ids=doc_ids_shifted)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                )
                loss = loss / grad_accum

            loss.backward()
            tokens_seen += inputs.numel()
            micro_step += 1

            if micro_step % grad_accum == 0:
                lr = self._get_lr()
                self._set_lr(lr)

                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1

                if self.global_step % log_every == 0:
                    elapsed = time.time() - t0
                    tok_per_sec = tokens_seen / elapsed
                    # Log grad_norm: sustained rise here precedes loss spikes by ~100s of steps
                    print(f"step {self.global_step:6d} | loss {loss.item() * grad_accum:.4f} | "
                          f"lr {lr:.2e} | gnorm {grad_norm:.3f} | {tok_per_sec/1000:.1f}K tok/s")

                if self.global_step % save_every == 0:
                    val_loss = self._eval_loss()
                    print(f"  → val_loss {val_loss:.4f} at step {self.global_step}")
                    save_checkpoint(
                        model=model,
                        optimizer=self.optimizer,
                        step=self.global_step,
                        loss=loss.item() * grad_accum,
                        config=self.model.config if hasattr(self.model, "config") else cfg,
                        output_dir=cfg["output_dir"],
                    )

        print(f"Training complete. Total steps: {self.global_step}, tokens seen: {tokens_seen:,}")
```

- [ ] **Step 2: Create train entry point**

Create `scripts/train.py`:

```python
"""
Entry point for training.

Usage:
  python scripts/train.py --phase 1
  python scripts/train.py --phase 2 --resume ./checkpoints/step-00010000
"""
import argparse
import torch
from src.model.config import ModelConfig
from src.model.transformer import Transformer
from src.data.loader import make_dataloader
from src.training.trainer import Trainer

# Token budget: 16B effective tokens for all phases
# Steps = 16B / (batch_size * seq_len * grad_accum)
#
# Phase 1: 16 * 2048 * 8 = 262,144 tokens/step → 16B / 262,144 = 61,035 steps
# Phase 2: 8  * 2048 * 8 = 131,072 tokens/step → 16B / 131,072 = 122,070 steps
# Phase 3: 2  * 4096 * 16 = 131,072 tokens/step → 16B / 131,072 = 122,070 steps

PHASE1_CONFIG = {
    # Phase 1: 135M — fast iteration, ~2 days at ~80-100K tok/s
    # LR 1e-3 and WSD schedule match SmolLM handbook defaults for 135M
    "lr": 1e-3,
    "beta1": 0.9,
    "beta2": 0.95,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "warmup_steps": 1000,
    "stable_steps": 52_035,     # bulk of training at stable LR
    "decay_steps": 8_000,       # WSD cosine decay to min_lr
    "total_steps": 61_035,      # 16B tokens / (16 * 2048 * 8)
    "schedule": "wsd",
    "grad_accum_steps": 8,      # effective batch = 16 * 2048 * 8 = 262,144 tokens/step
    "use_8bit_adam": False,
    "compile": True,
    "log_every": 50,
    "save_every": 2000,
    "output_dir": "./checkpoints/phase1",
}

PHASE2_CONFIG = {
    # Phase 2: 360M — optional intermediate, ~4 days
    # LR 1e-3 and WSD schedule match SmolLM handbook defaults for 360M
    "lr": 1e-3,
    "beta1": 0.9,
    "beta2": 0.95,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "warmup_steps": 1000,
    "stable_steps": 111_070,    # bulk of training at stable LR
    "decay_steps": 10_000,      # WSD cosine decay to min_lr
    "total_steps": 122_070,     # 16B tokens / (8 * 2048 * 8)
    "schedule": "wsd",
    "grad_accum_steps": 8,      # effective batch = 8 * 2048 * 8 = 131,072 tokens/step
    "use_8bit_adam": False,
    "compile": True,
    "log_every": 50,
    "save_every": 2000,
    "output_dir": "./checkpoints/phase2",
}

PHASE3_CONFIG = {
    # Phase 3: 2B — target model, ~28-44 days
    "lr": 3e-4,
    "beta1": 0.9,
    "beta2": 0.95,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "warmup_steps": 2000,
    "stable_steps": 110_070,    # bulk of training at stable LR
    "decay_steps": 10_000,      # WSD decay phase (upweights python-edu)
    "total_steps": 122_070,     # 16B tokens / (2 * 4096 * 16)
    "schedule": "wsd",
    "grad_accum_steps": 16,     # effective batch = 2 * 4096 * 16 = 131,072 tokens/step
    "use_8bit_adam": True,
    "compile": True,
    "log_every": 50,
    "save_every": 2000,
    "output_dir": "./checkpoints/phase3",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    if args.phase == 1:
        model_cfg = ModelConfig.phase1_135m()
        train_cfg = PHASE1_CONFIG
        seq_len = 2048
        batch_size = 16           # 135M is small — can fit larger batch
    elif args.phase == 2:
        model_cfg = ModelConfig.phase2_360m()
        train_cfg = PHASE2_CONFIG
        seq_len = 2048
        batch_size = 8
    else:
        model_cfg = ModelConfig.phase3_2b()
        train_cfg = PHASE3_CONFIG
        seq_len = 4096
        batch_size = 2

    model = Transformer(model_cfg)
    print(f"Model parameters: {model.num_parameters()/1e6:.0f}M")

    if args.resume:
        from src.training.checkpoint import load_checkpoint
        load_checkpoint(args.resume, model)
        print(f"Resumed from {args.resume}")

    train_dl, val_dl = make_dataloader(
        python_edu_path="./data/python_edu",
        fineweb_edu_path="./data/fineweb_edu",
        max_len=seq_len,
        batch_size=batch_size,
    )

    trainer = Trainer(model, train_dl, train_cfg, val_dataloader=val_dl)
    trainer.train()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Commit**

```bash
git add src/training/trainer.py scripts/train.py
git commit -m "feat: add training loop and train.py entry point"
```

---

## Phase 4 — Smoke Tests & First Runs

### Task 4.1: Smoke test on tiny data

Before starting the full 16B token run, verify the training loop runs without errors for 50 steps.

- [ ] **Step 1: Run smoke test**

```bash
python scripts/train.py --phase 1
# Let it run for ~50 steps, verify:
# - Loss starts at ~log(49152) ≈ 10.8 and begins decreasing
# - tok/s printed (expect 40–50K for Phase 1)
# - No OOM errors
# Ctrl+C after 50 steps
```

Expected output (first few lines):
```
Model parameters: 135M
step     50 | loss 9.8xxx | lr 5.00e-05 | gnorm 1.xxx | 90.0K tok/s
step    100 | loss 8.5xxx | lr 1.00e-04 | gnorm 0.xxx | 92.5K tok/s
```

Note: LR is low during warmup. It reaches the max (1e-3) at step 1000 then stays flat until the WSD decay phase. The 135M model should run significantly faster than 360M (~80–100K tok/s vs ~40–50K tok/s).

- [ ] **Step 2: Verify checkpoint saves correctly**

```bash
ls ./checkpoints/phase1/
# Should see: step-00000500/checkpoint.pt and step-00000500/pytorch_model.bin
```

- [ ] **Step 3: Run eval on the (untrained) checkpoint to confirm eval pipeline works end-to-end**

```bash
python scripts/evaluate.py \
  --checkpoint ./checkpoints/phase1/step-00000500 \
  --output ./evals/phase1_step500_sanity.json
```

The score will be near 0 (model is untrained), but this confirms the eval pipeline can load the checkpoint format.

---

### Task 4.2: Phase 1 full training run

This is the Phase 1 run (~2 days at ~80–100K tok/s, 61K steps). Start it in a persistent terminal session (use `tmux` or `screen`).

- [ ] **Step 1: Start training in tmux**

```bash
tmux new -s slm_phase1
python scripts/train.py --phase 1 2>&1 | tee logs/phase1.log
# Detach: Ctrl+B, D
```

- [ ] **Step 2: Monitor training every few hours**

Check loss curve is descending and tok/s is stable:
```bash
grep "step" logs/phase1.log | tail -20
```

- [ ] **Step 3: Run eval at checkpoints**

Run after step 8000 (~8B tokens), 16000 (~16B tokens), and final:

```bash
python scripts/evaluate.py \
  --checkpoint ./checkpoints/phase1/step-00008000 \
  --output ./evals/phase1_step8000.json

python scripts/evaluate.py \
  --checkpoint ./checkpoints/phase1/step-00016000 \
  --output ./evals/phase1_step16000.json
```

Expected HumanEval pass@1 at completion: ~5–15% (135M, no synthetic data, 16B tokens). Primary goal is pipeline validation — the number itself matters less than confirming training is healthy and eval works.

---

## Phase 5 — Phase 3 (2B model)

> **Optional Phase 2 (360M):** After Phase 1 completes, run `python scripts/train.py --phase 2` for ~4 days to get an intermediate quality checkpoint before committing to the full 2B run. Skip if Phase 1 results look clean.

### Task 5.1: Verify Phase 3 VRAM fits

Before starting the 28–44 day Phase 3 run, confirm VRAM usage is within budget.

- [ ] **Step 1: VRAM dry run (1 forward + backward, no data loading)**

```bash
python -c "
import torch
from src.model.config import ModelConfig
from src.model.transformer import Transformer
import torch.nn.functional as F

cfg = ModelConfig.phase3_2b()
model = Transformer(cfg).cuda().bfloat16()
print(f'Params: {model.num_parameters()/1e9:.2f}B')
print(f'VRAM after model load: {torch.cuda.memory_allocated()/1e9:.1f} GB')

# Single forward/backward
x = torch.randint(0, cfg.vocab_size, (2, 4096)).cuda()
with torch.autocast('cuda', dtype=torch.bfloat16):
    logits = model(x[:, :-1])
    loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), x[:, 1:].reshape(-1))
loss.backward()
print(f'VRAM after forward+backward: {torch.cuda.memory_allocated()/1e9:.1f} GB')
# Expect: ~18-22 GB with grad checkpointing
"
```

If VRAM exceeds 28 GB, reduce batch_size in PHASE3_CONFIG from 2 to 1 and increase grad_accum_steps to 32.

- [ ] **Step 2: Start Phase 3 training**

```bash
tmux new -s slm_phase3
python scripts/train.py --phase 3 2>&1 | tee logs/phase3.log
```

Expected throughput: ~5–8K tok/s. At 5K tok/s, 16B tokens ≈ 37 days.

- [ ] **Step 3: Eval Phase 3 checkpoints**

Run eval every ~2B tokens (every ~4000 steps):

```bash
python scripts/evaluate.py \
  --checkpoint ./checkpoints/phase3/step-00004000 \
  --output ./evals/phase3_step4000.json
```

Target at completion: ~35–45% HumanEval pass@1.

---

## Multi-GPU Scaling (RunPod, future)

When scaling to multiple GPUs on RunPod, wrap the trainer in PyTorch FSDP. No model architecture changes needed.

- [ ] **Add FSDP wrapper to train.py** (when needed):

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from src.model.block import TransformerBlock
import functools

fsdp_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={TransformerBlock})
model = FSDP(model, auto_wrap_policy=fsdp_policy, mixed_precision=...)
```

This is straightforward to add since the model is plain `nn.Module` with no framework-specific abstractions.

---

## Eval Reference

| Checkpoint | Expected HumanEval pass@1 |
|---|---|
| SmolLM2-135M (baseline, general) | TBD (not published for code) |
| SmolLM2-360M (baseline, general) | ~22% |
| Phase 1 / 135M, step 8000 (~8B tokens) | ~3–8% |
| Phase 1 / 135M, final (16B tokens) | ~5–15% |
| Phase 2 / 360M, final (16B tokens, optional) | ~15–25% |
| Phase 3 / 2B, final (16B tokens) | ~35–45% |
| phi-1 (1.3B, 50B tokens, synthetic data) | 50.6% |
