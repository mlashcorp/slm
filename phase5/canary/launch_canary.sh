#!/usr/bin/env bash
set -euo pipefail

ML_DIR="${ML_DIR:-/home/ubuntu/workspace/code/Opencode-Remote-/ml}"
CONFIG_PATH="${CONFIG_PATH:-/home/ubuntu/code/slm/phase5/canary/vast_canary_config.yaml}"
GPU="${GPU:-RTX 4090}"
MAX_PRICE="${MAX_PRICE:-0.50}"

if [[ ! -f "${ML_DIR}/.env" ]]; then
  echo "Missing ${ML_DIR}/.env" >&2
  exit 1
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Missing config: ${CONFIG_PATH}" >&2
  exit 1
fi

cd "${ML_DIR}"
set -a
# shellcheck source=/dev/null
source .env
set +a

echo "Dry run offer search (gpu=${GPU}, max-price=${MAX_PRICE})"
python3 launch.py --config "${CONFIG_PATH}" launch --dry-run --gpu "${GPU}" --max-price "${MAX_PRICE}"

echo "Launching canary job"
python3 launch.py --config "${CONFIG_PATH}" launch --gpu "${GPU}" --max-price "${MAX_PRICE}"
