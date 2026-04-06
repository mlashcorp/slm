#!/usr/bin/env bash
set -euo pipefail

echo "=== Remote baseline job start ==="
date
nvidia-smi || true

NANOCHAT_BASE_DIR="/workspace/datasets/phase5/baseline"
DATA_DIR="${NANOCHAT_BASE_DIR}/base_data_climbmix"
TOKENIZER_PATH="${NANOCHAT_BASE_DIR}/tokenizer/tokenizer.pkl"

# Download data directly on remote (no R2 needed)
# 25 shards of climbmix = ~1.53B tokens (compute-optimal for 124M model)
NUM_SHARDS=25

if [[ ! -d "${DATA_DIR}" ]] || [[ -z "$(ls -A ${DATA_DIR} 2>/dev/null)" ]]; then
    echo "Dataset not found. Downloading ${NUM_SHARDS} climbmix shards from HuggingFace..."
    
    NANOCHAT_DIR="/workspace/nanochat"
    if [[ ! -d "${NANOCHAT_DIR}" ]]; then
        git clone --depth 1 https://github.com/karpathy/nanochat.git "${NANOCHAT_DIR}"
    fi
    
    cd "${NANOCHAT_DIR}"
    
    echo "Installing dependencies..."
    python3 -m pip install -q --upgrade pip
    python3 -m pip install -q \
        numpy \
        requests \
        pyarrow \
        filelock \
        huggingface_hub \
        datasets
    
    # Download climbmix shards
    echo ""
    echo "=== Downloading ${NUM_SHARDS} ClimbMix shards ==="
    echo "This will download ~1.53B tokens (compute-optimal for 124M model)"
    export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR}"
    python3 -m nanochat.dataset -n ${NUM_SHARDS} -w 4
    
    echo ""
    echo "Dataset files:"
    ls -lh "${DATA_DIR}" || true
fi

echo "Dataset files:"
ls -lh "${DATA_DIR}" || true

NANOCHAT_DIR="/workspace/nanochat"
if [[ ! -d "${NANOCHAT_DIR}" ]]; then
    git clone --depth 1 https://github.com/karpathy/nanochat.git "${NANOCHAT_DIR}"
fi

cd "${NANOCHAT_DIR}"

echo "Installing training dependencies..."
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
    huggingface_hub \
    torchao

if [[ ! -f "${TOKENIZER_PATH}" ]]; then
    echo "Training tokenizer on ClimbMix data..."
    OMP_NUM_THREADS=1 python3 -m scripts.tok_train \
        --max-chars=10000000 \
        --doc-cap=4000 \
        --vocab-size=8192
fi

RUN_ID="${RUN_ID:-phase5_baseline_124m}"
HF_REPO="${HF_REPO:-mlashcorp/slm-phase5-checkpoints}"
CHECKPOINT_DIR="${NANOCHAT_BASE_DIR}/base_checkpoints/d12"
SAVE_EVERY=500
NUM_ITERATIONS=715

echo ""
echo "=== Training Configuration ==="
echo "Run ID: ${RUN_ID}"
echo "Model: ~124M params (depth=12)"
echo "Data: ClimbMix (25 shards = ~1.53B tokens)"
echo "Token budget: ~1.5B tokens"
echo "Steps: ${NUM_ITERATIONS}"
echo "Save every: ${SAVE_EVERY}"
echo "Checkpoint dir: ${CHECKPOINT_DIR}"
echo ""

echo "Starting nanochat baseline training (depth=12, ~124M params)..."
echo "This will take approximately 60-90 minutes on RTX 4090"
echo ""

OMP_NUM_THREADS=1 python3 -m scripts.base_train \
    --run="${RUN_ID}" \
    --device-type="cuda" \
    --depth=12 \
    --window-pattern="L" \
    --max-seq-len=1024 \
    --device-batch-size=2 \
    --total-batch-size=2048 \
    --num-iterations=${NUM_ITERATIONS} \
    --eval-every=100 \
    --eval-tokens=65536 \
    --save-every=${SAVE_EVERY} \
    --sample-every=-1 \
    --core-metric-every=200

echo "Training complete. Pushing checkpoints to HuggingFace Hub..."

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "WARNING: HF_TOKEN not set, skipping checkpoint push to HuggingFace Hub"
else
    cp -r /workspace/slm/phase5/checkpointing /workspace/checkpointing

    # Push all saved checkpoints directly using known save interval
    for STEP in $(seq ${SAVE_EVERY} ${SAVE_EVERY} ${NUM_ITERATIONS}); do
        # Try all known nanochat checkpoint naming patterns
        MODEL_FILE=""
        for pattern in \
            "${CHECKPOINT_DIR}/model_${STEP}.pt" \
            "${CHECKPOINT_DIR}/model_$(printf '%06d' ${STEP}).pt" \
            "${CHECKPOINT_DIR}/model_$(printf '%07d' ${STEP}).pt"; do
            if [[ -f "${pattern}" ]]; then
                MODEL_FILE="${pattern}"
                break
            fi
        done

        if [[ -n "${MODEL_FILE}" ]]; then
            echo "Pushing step ${STEP} to ${HF_REPO}..."
            python3 /workspace/checkpointing/hf_checkpoint.py push \
                --checkpoint-dir "${CHECKPOINT_DIR}" \
                --step ${STEP} \
                --run-id "${RUN_ID}" \
                --hf-repo "${HF_REPO}" \
                --hf-token "${HF_TOKEN}"
        else
            echo "WARNING: no checkpoint file found for step ${STEP} in ${CHECKPOINT_DIR}"
            echo "Available files:"
            ls -la "${CHECKPOINT_DIR}/" 2>/dev/null || echo "  (directory does not exist)"
        fi
    done

    echo "All checkpoints pushed. List available:"
    python3 /workspace/checkpointing/hf_checkpoint.py list \
        --run-id="${RUN_ID}" \
        --hf-repo="${HF_REPO}" \
        --hf-token="${HF_TOKEN}"
fi

echo "=== Remote baseline job done ==="
date
