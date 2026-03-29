#!/bin/bash
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
cd /home/cortereal/workspace/code/slm
.venv/bin/python scripts/train_nanogpt.py config/train_gpt2_quick.py
