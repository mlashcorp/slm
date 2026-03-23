"""
Training entry point.

Usage:
  python scripts/train.py --phase 1
  python scripts/train.py --phase 2 --resume ./checkpoints/phase1/step-00061035
"""
import argparse
import torch
from src.model.config import ModelConfig
from src.model.transformer import Transformer
from src.data.loader import make_dataloader
from src.training.trainer import Trainer
from src.training.checkpoint import load_checkpoint

# Steps = 16B tokens / (batch_size * seq_len * grad_accum)
# Phase 1: 16 * 2048 * 8  = 262,144 tok/step → 61,035 steps
# Phase 2:  8 * 2048 * 8  = 131,072 tok/step → 122,070 steps
# Phase 3:  2 * 4096 * 16 = 131,072 tok/step → 122,070 steps

PHASE1_CONFIG = {
    "lr": 1e-3,
    "beta1": 0.9,
    "beta2": 0.95,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "warmup_steps": 1_000,
    "stable_steps": 52_035,
    "decay_steps": 8_000,
    "total_steps": 61_035,
    "schedule": "wsd",
    "grad_accum_steps": 8,
    "use_8bit_adam": False,
    "compile": True,
    "log_every": 50,
    "save_every": 2_000,
    "output_dir": "./checkpoints/phase1",
}

PHASE2_CONFIG = {
    "lr": 1e-3,
    "beta1": 0.9,
    "beta2": 0.95,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "warmup_steps": 1_000,
    "stable_steps": 111_070,
    "decay_steps": 10_000,
    "total_steps": 122_070,
    "schedule": "wsd",
    "grad_accum_steps": 8,
    "use_8bit_adam": False,
    "compile": True,
    "log_every": 50,
    "save_every": 2_000,
    "output_dir": "./checkpoints/phase2",
}

PHASE3_CONFIG = {
    "lr": 3e-4,
    "beta1": 0.9,
    "beta2": 0.95,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "warmup_steps": 2_000,
    "stable_steps": 110_070,
    "decay_steps": 10_000,
    "total_steps": 122_070,
    "schedule": "wsd",
    "grad_accum_steps": 16,
    "use_8bit_adam": True,
    "compile": True,
    "log_every": 50,
    "save_every": 2_000,
    "output_dir": "./checkpoints/phase3",
}

PHASE_CONFIGS = {1: PHASE1_CONFIG, 2: PHASE2_CONFIG, 3: PHASE3_CONFIG}
PHASE_MODELS = {
    1: (ModelConfig.phase1_135m, 2048, 16),
    2: (ModelConfig.phase2_360m, 2048, 8),
    3: (ModelConfig.phase3_2b,   4096, 2),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint dir to resume from")
    parser.add_argument("--python-edu-path", type=str,
                        default="./data/python_edu_hydrated")
    parser.add_argument("--fineweb-edu-path", type=str,
                        default="./data/fineweb_edu")
    args = parser.parse_args()

    model_fn, seq_len, batch_size = PHASE_MODELS[args.phase]
    model_cfg = model_fn()
    train_cfg = PHASE_CONFIGS[args.phase]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = Transformer(model_cfg)
    print(f"Model parameters: {model.num_parameters() / 1e6:.0f}M")

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model)
        print(f"Resumed from {args.resume} (step {start_step})")

    train_dl, val_dl = make_dataloader(
        python_edu_path=args.python_edu_path,
        fineweb_edu_path=args.fineweb_edu_path,
        max_len=seq_len,
        batch_size=batch_size,
    )

    trainer = Trainer(model, train_dl, train_cfg, device=device, val_dataloader=val_dl)
    trainer.global_step = start_step
    trainer.train()


if __name__ == "__main__":
    main()
