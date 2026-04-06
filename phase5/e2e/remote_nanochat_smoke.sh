#!/usr/bin/env bash
set -euo pipefail

echo "=== Remote smoke job start ==="
date
nvidia-smi || true

export NANOCHAT_BASE_DIR="/workspace/datasets/phase5/e2e/nanochat"
DATA_DIR="${NANOCHAT_BASE_DIR}/base_data_climbmix"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "Dataset directory missing: ${DATA_DIR}" >&2
  exit 2
fi

echo "Dataset files:"
ls -lah "${DATA_DIR}"

NANOCHAT_DIR="/workspace/nanochat"
if [[ ! -d "${NANOCHAT_DIR}" ]]; then
  git clone https://github.com/karpathy/nanochat.git "${NANOCHAT_DIR}"
fi

cd "${NANOCHAT_DIR}"

python3 -m pip install -q --upgrade pip
python3 -m pip install -q \
  numpy \
  requests \
  pyarrow \
  filelock \
  wandb \
  tokenizers \
  tiktoken \
  rustbpe

echo "Checking data visibility through nanochat module"
python3 - <<'PY'
import os
os.environ.setdefault("NANOCHAT_BASE_DIR", "/workspace/datasets/phase5/e2e/nanochat")
from nanochat.dataset import list_parquet_files
files = list_parquet_files()
print(f"nanochat sees {len(files)} parquet files")
for f in files[:5]:
    print(f"  {f}")
PY

TOKENIZER_PATH="${NANOCHAT_BASE_DIR}/tokenizer/tokenizer.pkl"
if [[ ! -f "${TOKENIZER_PATH}" ]]; then
  echo "Tokenizer not found. Training tiny tokenizer for smoke run..."
  OMP_NUM_THREADS=1 python3 -m scripts.tok_train \
    --max-chars=5000000 \
    --doc-cap=4000 \
    --vocab-size=4096
fi

echo "Running tiny nanochat pretraining smoke (20 iters)"
OMP_NUM_THREADS=1 python3 -m scripts.base_train \
  --run="dummy" \
  --device-type="cuda" \
  --depth=4 \
  --max-seq-len=512 \
  --device-batch-size=1 \
  --total-batch-size=512 \
  --num-iterations=20 \
  --window-pattern="L" \
  --eval-every=-1 \
  --core-metric-every=-1 \
  --sample-every=-1 \
  --save-every=-1

echo "=== Remote smoke job done ==="
date
