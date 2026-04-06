#!/usr/bin/env bash
set -euo pipefail

echo "=== Phase 5 Baseline Launcher ==="
echo ""
echo "This will:"
echo "  1. Launch a Vast.ai instance with RTX 5090"
echo "  2. Download 25 ClimbMix shards (~1.53B tokens)"
echo "  3. Train 124M nanochat model (depth=24) for ~715 steps"
echo "  4. Evaluate on: val_bpb, CORE metric"
echo ""
echo "Config:"
echo "  - Model: 124M (depth=24, context=1024)"
echo "  - Data: ClimbMix 25 shards (~1.53B tokens)"
echo "  - Token budget: 12× params (compute-optimal)"
echo "  - Steps: ~715 (auto-calculated)"
echo "  - Batch: 2048 tokens"
echo ""
echo "Expected:"
echo "  - Training time: ~5-6 hours"
echo "  - Cost: ~2-3 USD (RTX 5090 @ ~0.40 USD/hr)"
echo ""

# Check credentials
if [[ -z "${VASTAI_API_KEY:-}" ]]; then
    echo "ERROR: VASTAI_API_KEY not set"
    exit 1
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "WARNING: HF_TOKEN not set - checkpoint upload to HuggingFace will be skipped"
fi

cd /home/ubuntu/workspace/code/Opencode-Remote-/ml

# Launch the baseline training
echo "Launching Vast.ai instance..."
python3 launch.py --config /home/ubuntu/code/slm/phase5/baseline/vast_baseline_config.yaml launch

echo ""
echo "Instance launched. Wait ~5-10 minutes for data download, then training will start."
echo "Monitor at: https://wandb.ai/slm-phase5-baseline"
