#!/usr/bin/env bash
set -euo pipefail

echo "=== Remote canary job start ==="
date
nvidia-smi || true

NANOCHAT_BASE_DIR="/workspace/datasets/phase5/e2e/nanochat"
DATA_DIR="${NANOCHAT_BASE_DIR}/base_data"
TOKENIZER_PATH="${NANOCHAT_BASE_DIR}/tokenizer/tokenizer.pkl"

# Download data directly on remote (no R2 needed)
if [[ ! -d "${DATA_DIR}" ]] || [[ ! -f "${DATA_DIR}"/*.parquet ]]; then
    echo "Dataset not found. Downloading from HuggingFace..."
    bash /workspace/slm/phase5/canary/remote_download_data.sh
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
    rustbpe \
    huggingface_hub

if [[ ! -f "${TOKENIZER_PATH}" ]]; then
    echo "Training tokenizer on mixed data..."
    OMP_NUM_THREADS=1 python3 -m scripts.tok_train \
        --max-chars=10000000 \
        --doc-cap=4000 \
        --vocab-size=8192
fi

RUN_ID="${RUN_ID:-phase5_canary_baseline}"
HF_REPO="${HF_REPO:-mlashcorp/slm-phase5-checkpoints}"
CHECKPOINT_DIR="${NANOCHAT_BASE_DIR}/base_checkpoints/d8"
SAVE_EVERY=100
NUM_ITERATIONS=300

echo ""
echo "=== Training Configuration ==="
echo "Run ID: ${RUN_ID}"
echo "Data: 50/50 python-edu/climbmix (1 shard each)"
echo "Steps: ${NUM_ITERATIONS}"
echo "Save every: ${SAVE_EVERY}"
echo "Checkpoint dir: ${CHECKPOINT_DIR}"
echo ""

echo "Starting nanochat canary training..."
OMP_NUM_THREADS=1 python3 -m scripts.base_train \
    --run="${RUN_ID}" \
    --device-type="cuda" \
    --depth=8 \
    --max-seq-len=1024 \
    --device-batch-size=2 \
    --total-batch-size=2048 \
    --num-iterations=${NUM_ITERATIONS} \
    --window-pattern="L" \
    --eval-every=100 \
    --eval-tokens=65536 \
    --save-every=${SAVE_EVERY} \
    --sample-every=-1

echo "Training complete. Pushing checkpoints to HuggingFace Hub..."

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "WARNING: HF_TOKEN not set, skipping checkpoint push to HuggingFace Hub"
else
    cp -r /workspace/slm/phase5/checkpointing /workspace/checkpointing

    for STEP in $(seq ${SAVE_EVERY} ${SAVE_EVERY} ${NUM_ITERATIONS}); do
        MODEL_FILE="${CHECKPOINT_DIR}/model_$(printf '%06d' ${STEP}).pt"
        if [[ -f "${MODEL_FILE}" ]]; then
            echo "Pushing step ${STEP} to ${HF_REPO}..."
            python3 /workspace/checkpointing/hf_checkpoint.py push \
                --checkpoint-dir "${CHECKPOINT_DIR}" \
                --step ${STEP} \
                --run-id "${RUN_ID}" \
                --hf-repo "${HF_REPO}" \
                --hf-token "${HF_TOKEN}"
        else
            echo "WARNING: checkpoint file not found for step ${STEP}, skipping"
        fi
    done

    echo "All checkpoints pushed. List available:"
    python3 /workspace/checkpointing/hf_checkpoint.py list \
        --run-id "${RUN_ID}" \
        --hf-repo "${HF_REPO}" \
        --hf-token "${HF_TOKEN}"
fi

echo "=== Remote canary job done ==="
date
