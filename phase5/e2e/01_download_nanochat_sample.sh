#!/usr/bin/env bash
set -euo pipefail

NANOCHAT_DIR="${NANOCHAT_DIR:-/home/ubuntu/workspace/code/nanochat}"
DATA_ROOT="${DATA_ROOT:-/home/ubuntu/workspace/data/nanochat-e2e}"
NUM_FILES="${NUM_FILES:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"

echo "[1/4] Preparing nanochat repo at ${NANOCHAT_DIR}"
if [[ ! -d "${NANOCHAT_DIR}" ]]; then
  git clone "https://github.com/karpathy/nanochat.git" "${NANOCHAT_DIR}"
fi

mkdir -p "${DATA_ROOT}"

echo "[2/4] Installing minimal downloader dependencies"
python3 -m pip install -q --upgrade pip
python3 -m pip install -q requests pyarrow filelock numpy
if ! python3 -c "import torch" >/dev/null 2>&1; then
  echo "Installing CPU torch for nanochat utilities"
  python3 -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu
fi

echo "[3/4] Downloading nanochat parquet shards"
echo "      num-files=${NUM_FILES} num-workers=${NUM_WORKERS}"
cd "${NANOCHAT_DIR}"
NANOCHAT_BASE_DIR="${DATA_ROOT}" \
  python3 -m nanochat.dataset -n "${NUM_FILES}" -w "${NUM_WORKERS}"

echo "[4/4] Done"
echo "Data directory: ${DATA_ROOT}/base_data_climbmix"
ls -lh "${DATA_ROOT}/base_data_climbmix"
