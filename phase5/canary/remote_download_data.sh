#!/usr/bin/env bash
set -euo pipefail

echo "=== Remote canary data download start ==="
date

NANOCHAT_DIR="/workspace/nanochat"
NANOCHAT_BASE_DIR="/workspace/datasets/phase5/e2e/nanochat"
DATA_DIR="${NANOCHAT_BASE_DIR}/base_data"
TOKENIZER_PATH="${NANOCHAT_BASE_DIR}/tokenizer/tokenizer.pkl"

# Number of shards to download per source
NUM_SHARDS="${NUM_SHARDS:-1}"

# Mixture ratio: 80% python-edu, 20% climbmix
# For 1 shard each: 1 python shard + 1 climbmix shard
# The dataloader will sample uniformly from all shards, so ratio is controlled by shard count
PYTHON_SHARDS=1
CLIMBMIX_SHARDS=1

echo "Target directory: ${DATA_DIR}"
echo "Python-edu shards: ${PYTHON_SHARDS}"
echo "ClimbMix shards: ${CLIMBMIX_SHARDS}"
echo ""

# Clone nanochat if not exists
if [[ ! -d "${NANOCHAT_DIR}" ]]; then
    echo "Cloning nanochat repo..."
    git clone --depth 1 https://github.com/karpathy/nanochat.git "${NANOCHAT_DIR}"
fi

cd "${NANOCHAT_DIR}"

# Install dependencies
echo "Installing dependencies..."
python3 -m pip install -q --upgrade pip
python3 -m pip install -q \
    numpy \
    requests \
    pyarrow \
    filelock \
    huggingface_hub \
    datasets

# Create unified data directory
mkdir -p "${DATA_DIR}"

# Download climbmix shards (using nanochat's dataset module)
echo ""
echo "=== Downloading ClimbMix shards ==="
# Override the DATA_DIR temporarily for nanochat's downloader
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR}"
CLIMBMIX_TMP_DIR="${NANOCHAT_BASE_DIR}/base_data_climbmix"
mkdir -p "${CLIMBMIX_TMP_DIR}"

# Use nanochat's built-in downloader
python3 -m nanochat.dataset -n "${CLIMBMIX_SHARDS}" -w 4

# Copy climbmix shards to unified directory
echo "Copying ClimbMix shards to unified directory..."
cp "${CLIMBMIX_TMP_DIR}"/*.parquet "${DATA_DIR}/" 2>/dev/null || true

# Download python-edu shards from HuggingFace
echo ""
echo "=== Downloading Python-edu shards ==="

python3 << EOF
import os
import sys
from pathlib import Path

data_dir = "${DATA_DIR}"
num_shards = ${PYTHON_SHARDS}

print(f"Downloading {num_shards} python-edu shard(s) from HuggingFace...")
print(f"Target: {data_dir}")

try:
    from datasets import load_dataset
    import pyarrow.parquet as pq
    import pyarrow as pa
    
    # Load python-edu from smollm-corpus
    ds = load_dataset("HuggingFaceTB/smollm-corpus", "python-edu", split="train", trust_remote_code=True)
    print(f"Dataset loaded: {len(ds)} examples")
    
    # Save shards to unified directory
    os.makedirs(data_dir, exist_ok=True)
    
    # Split into shards and save
    shard_size = (len(ds) + num_shards - 1) // num_shards if num_shards > 0 else len(ds)
    
    for i in range(num_shards):
        start_idx = i * shard_size
        end_idx = min((i + 1) * shard_size, len(ds))
        shard_data = ds.select(range(start_idx, end_idx))
        
        texts = shard_data['text']
        table = pa.table({'text': texts})
        shard_path = os.path.join(data_dir, f'python_shard_{i:05d}.parquet')
        print(f"Writing {shard_path} ({len(texts)} examples)...")
        pq.write_table(table, shard_path)
    
    print(f"Successfully saved {num_shards} python-edu shard(s)")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

EOF

# List all shards in unified directory
echo ""
echo "=== Unified data directory ==="
ls -lh "${DATA_DIR}"

echo ""
echo "=== Download complete ==="
date
echo "=== Remote data download done ==="
