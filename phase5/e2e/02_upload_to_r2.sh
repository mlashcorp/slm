#!/usr/bin/env bash
set -euo pipefail

ML_DIR="${ML_DIR:-/home/ubuntu/workspace/code/Opencode-Remote-/ml}"
DATA_ROOT="${DATA_ROOT:-/home/ubuntu/workspace/data/nanochat-e2e}"
SRC_DIR="${SRC_DIR:-${DATA_ROOT}/base_data_climbmix}"
R2_DEST="${R2_DEST:-r2:ml-datasets/phase5/e2e/nanochat/base_data_climbmix}"

if [[ ! -d "${SRC_DIR}" ]]; then
  echo "Source directory missing: ${SRC_DIR}" >&2
  echo "Run phase5/e2e/01_download_nanochat_sample.sh first." >&2
  exit 1
fi

if [[ ! -f "${ML_DIR}/.env" ]]; then
  echo "Missing ${ML_DIR}/.env" >&2
  exit 1
fi

echo "Loading R2 credentials from ${ML_DIR}/.env"
set -a
# shellcheck source=/dev/null
source "${ML_DIR}/.env"
set +a

export RCLONE_CONFIG_R2_TYPE="s3"
export RCLONE_CONFIG_R2_PROVIDER="Cloudflare"
export RCLONE_CONFIG_R2_ACCESS_KEY_ID="${R2_ACCESS_KEY_ID}"
export RCLONE_CONFIG_R2_SECRET_ACCESS_KEY="${R2_SECRET_ACCESS_KEY}"
export RCLONE_CONFIG_R2_ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"

echo "Uploading ${SRC_DIR} -> ${R2_DEST}"
rclone copy "${SRC_DIR}" "${R2_DEST}" --progress --transfers=8 --checkers=16

echo "Verifying uploaded files"
rclone ls "${R2_DEST}"

echo "Upload complete"
