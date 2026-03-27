#!/usr/bin/env python
"""
Phase 0 Training Script: NanoChat GPT-2 Reproduction

This script trains a NanoChat GPT model following Karpathy's approach:
- ReLU^2 activation
- QK normalization
- Value embeddings
- Sliding window attention
- Muon optimizer
- FP8 training (optional)

Usage:
    # Quick validation with d12 (~5 min)
    python scripts/train_phase0.py --depth 12 --num-iterations 100

    # Full GPT-2 reproduction with d24 (~5 hours)
    python scripts/train_phase0.py --depth 24

Based on: https://github.com/karpathy/nanochat
"""
import os
import sys
import time
import math
import json
import argparse
from pathlib import Path
from dataclasses import asdict

import torch
import torch.nn.functional as F
import wandb

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.nanochat_config import NanoChatConfig
from src.model.nanochat_gpt import GPT
from src.training.muon_optimizer import MuonAdamW, setup_param_groups
from src.training.fp8 import Float8Linear, convert_to_float8_training, check_fp8_support
from src.data.bos_bestfit import create_data_loader, Tokenizer


# =============================================================================
# Learning Rate Schedule (Warmup-Const-Warmdown)
# =============================================================================

def get_lr_multiplier(step: int, warmup_steps: int, num_iterations: int, warmdown_ratio: float, final_lr_frac: float) -> float:
    """Compute learning rate multiplier.

    Args:
        step: Current step
        warmup_steps: Number of warmup steps
        num_iterations: Total number of iterations
        warmdown_ratio: Ratio of training for warmdown
        final_lr_frac: Final LR as fraction of initial

    Returns:
        LR multiplier (1.0 = peak LR)
    """
    warmdown_steps = round(warmdown_ratio * num_iterations)

    if step < warmup_steps:
        # Linear warmup
        return (step + 1) / warmup_steps
    elif step <= num_iterations - warmdown_steps:
        # Stable phase
        return 1.0
    else:
        # Linear warmdown
        progress = (num_iterations - step) / warmdown_steps
        return progress * 1.0 + (1 - progress) * final_lr_frac


def get_momentum(step: int, num_iterations: int, warmdown_ratio: float) -> float:
    """Compute momentum for Muon optimizer.

    Momentum warms up to 0.97, then warms down to 0.90 during LR warmdown.

    Args:
        step: Current step
        num_iterations: Total iterations
        warmdown_ratio: Ratio of training for warmdown

    Returns:
        Momentum value
    """
    warmdown_steps = round(warmdown_ratio * num_iterations)
    warmdown_start = num_iterations - warmdown_steps

    if step < 400:
        # Warmup to 0.97
        frac = step / 400
        return (1 - frac) * 0.85 + frac * 0.97
    elif step >= warmdown_start:
        # Warmdown to 0.90
        progress = (step - warmdown_start) / warmdown_steps
        return 0.97 * (1 - progress) + 0.90 * progress
    else:
        return 0.97


# =============================================================================
# Training Loop
# =============================================================================

def train(args):
    """Main training function."""

    # Device setup
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device) if args.device == "cuda" else None

    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'N/A'}")

    # Check FP8 support
    if args.fp8:
        fp8_supported, fp8_reason = check_fp8_support()
        print(f"FP8: {fp8_reason}")
        if not fp8_supported:
            print("Warning: FP8 not supported, falling back to bf16")
            args.fp8 = False

    # Model configuration
    if args.depth == 12:
        config = NanoChatConfig.d12()
    elif args.depth == 24:
        config = NanoChatConfig.d24()
    elif args.depth == 26:
        config = NanoChatConfig.d26()
    else:
        # Custom depth
        config = NanoChatConfig(depth=args.depth)

    # Override sequence length if specified
    if args.max_seq_len:
        config.sequence_len = args.max_seq_len

    print(f"\nModel config:")
    print(f"  depth: {config.depth}")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  num_heads: {config.num_heads}")
    print(f"  head_dim: {config.head_dim}")
    print(f"  vocab_size: {config.vocab_size}")

    # Parameter count
    param_counts = config.num_parameters()
    print(f"\nParameter counts:")
    for key, value in param_counts.items():
        print(f"  {key}: {value:,}")

    # Create model
    model = GPT(config)
    model.init_weights()
    model = model.to(device)

    print(f"  Total: {model.num_parameters():,} parameters")

    # FP8 conversion
    if args.fp8:
        print("\nConverting to FP8 training...")

        def fp8_filter(mod, fqn):
            if not isinstance(mod, torch.nn.Linear):
                return False
            if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                return False
            if min(mod.in_features, mod.out_features) < 128:
                return False
            return True

        model = convert_to_float8_training(model, module_filter_fn=fp8_filter)
        print("  FP8 conversion complete")

    # Compile model
    if args.compile:
        print("\nCompiling model...")
        model = torch.compile(model, dynamic=False)
        print("  Compilation complete")

    # Optimizer
    print("\nSetting up optimizer...")
    param_groups = setup_param_groups(
        model,
        matrix_lr=args.matrix_lr,
        embedding_lr=args.embedding_lr,
        unembedding_lr=args.unembedding_lr,
        scalar_lr=args.scalar_lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
    )

    optimizer = MuonAdamW(param_groups)

    print(f"  Parameter groups: {len(param_groups)}")
    for i, group in enumerate(param_groups):
        print(f"    Group {i}: kind={group['kind']}, lr={group.get('lr', 'N/A')}")

    # Data loader
    print("\nSetting up data loader...")
    data_dir = Path(args.data_dir)

    tokenizer = Tokenizer()  # Uses cl100k_base by default
    print(f"  Tokenizer vocab size: {tokenizer.get_vocab_size()}")

    train_loader = create_data_loader(
        batch_size=args.batch_size,
        seq_len=config.sequence_len,
        split="train",
        data_dir=data_dir,
        device=str(device),
    )

    # Calculate training horizon
    if args.num_iterations > 0:
        num_iterations = args.num_iterations
    elif args.target_tokens > 0:
        tokens_per_step = args.batch_size * config.sequence_len
        num_iterations = args.target_tokens // tokens_per_step
    else:
        # Default: use target_param_data_ratio
        scaling_params = param_counts["transformer_matrices"] + param_counts["lm_head"]
        target_tokens = args.target_param_data_ratio * scaling_params
        tokens_per_step = args.batch_size * config.sequence_len
        num_iterations = int(target_tokens // tokens_per_step)

    total_tokens = num_iterations * args.batch_size * config.sequence_len

    print(f"\nTraining configuration:")
    print(f"  Total iterations: {num_iterations:,}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Sequence length: {config.sequence_len}")
    print(f"  Tokens per step: {args.batch_size * config.sequence_len:,}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Tokens / scaling params: {total_tokens / scaling_params:.1f}")

    # W&B
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"phase0_d{args.depth}",
            config={
                "depth": config.depth,
                "hidden_size": config.hidden_size,
                "num_heads": config.num_heads,
                "batch_size": args.batch_size,
                "seq_len": config.sequence_len,
                "num_iterations": num_iterations,
                "total_tokens": total_tokens,
                "fp8": args.fp8,
                "matrix_lr": args.matrix_lr,
                "embedding_lr": args.embedding_lr,
            },
        )
    else:
        wandb.init(mode="disabled")

    # Training loop
    print("\nStarting training...")
    print(f"{'Step':>6} | {'Loss':>8} | {'LR':>8} | {'Mom':>6} | {'Tok/s':>8} | {'Time':>8}")
    print("-" * 60)

    model.train()
    step = 0
    total_time = 0.0

    # Prefetch first batch
    inputs, targets, _ = next(train_loader)

    while step < num_iterations:
        # Get LR and momentum
        lr_mult = get_lr_multiplier(
            step, args.warmup_steps, num_iterations,
            args.warmdown_ratio, args.final_lr_frac
        )
        momentum = get_momentum(step, num_iterations, args.warmdown_ratio)

        # Update optimizer params
        for group in optimizer.param_groups:
            group["lr"] = group.get("initial_lr", group.get("lr", 0.01)) * lr_mult
            if group["kind"] == "muon":
                group["momentum"] = momentum

        # Forward pass
        t0 = time.perf_counter()
        loss = model(inputs, targets)

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()
        model.zero_grad(set_to_none=True)

        t1 = time.perf_counter()
        dt = t1 - t0
        total_time += dt

        # Logging
        tokens_per_sec = args.batch_size * config.sequence_len / dt

        if step % args.log_interval == 0:
            print(f"{step:6d} | {loss.item():8.4f} | {lr_mult:8.4f} | {momentum:6.3f} | {tokens_per_sec:8.0f} | {dt*1000:6.1f}ms")

            if args.wandb:
                wandb.log({
                    "step": step,
                    "train/loss": loss.item(),
                    "train/lr_mult": lr_mult,
                    "train/momentum": momentum,
                    "train/tokens_per_sec": tokens_per_sec,
                    "train/time_per_step": dt,
                })

        # Next batch
        inputs, targets, _ = next(train_loader)
        step += 1

        # Checkpoint
        if args.save_every > 0 and step % args.save_every == 0 and step > 0:
            checkpoint_dir = Path(args.checkpoint_dir) / f"d{args.depth}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"step_{step:06d}.pt"

            torch.save({
                "step": step,
                "model_state_dict": model.state_dict() if not args.compile else model._orig_mod.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": asdict(config),
                "loss": loss.item(),
            }, checkpoint_path)

            print(f"  Saved checkpoint: {checkpoint_path}")

    # Final checkpoint
    checkpoint_dir = Path(args.checkpoint_dir) / f"d{args.depth}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "final.pt"

    torch.save({
        "step": step,
        "model_state_dict": model.state_dict() if not args.compile else model._orig_mod.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(config),
        "loss": loss.item(),
    }, checkpoint_path)

    print(f"\nFinal checkpoint saved: {checkpoint_path}")

    # Summary
    print(f"\nTraining complete!")
    print(f"  Total steps: {step:,}")
    print(f"  Total tokens: {step * args.batch_size * config.sequence_len:,}")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Average tokens/sec: {step * args.batch_size * config.sequence_len / total_time:.0f}")

    wandb.finish()


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 0: NanoChat GPT-2 Training")

    # Model
    parser.add_argument("--depth", type=int, default=24, help="Model depth (12, 24, 26)")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="Sequence length")

    # Training
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--num-iterations", type=int, default=-1, help="Number of iterations (-1 = auto)")
    parser.add_argument("--target-tokens", type=int, default=-1, help="Target tokens (-1 = auto)")
    parser.add_argument("--target-param-data-ratio", type=float, default=8.0, help="Tokens per scaling param")

    # Optimizer
    parser.add_argument("--matrix-lr", type=float, default=0.02, help="LR for matrix params (Muon)")
    parser.add_argument("--embedding-lr", type=float, default=0.2, help="LR for embeddings")
    parser.add_argument("--unembedding-lr", type=float, default=0.008, help="LR for lm_head")
    parser.add_argument("--scalar-lr", type=float, default=0.5, help="LR for scalars")
    parser.add_argument("--weight-decay", type=float, default=0.28, help="Weight decay (Muon)")
    parser.add_argument("--momentum", type=float, default=0.95, help="Momentum (Muon)")
    parser.add_argument("--warmup-steps", type=int, default=40, help="Warmup steps")
    parser.add_argument("--warmdown-ratio", type=float, default=0.65, help="Warmdown ratio")
    parser.add_argument("--final-lr-frac", type=float, default=0.05, help="Final LR fraction")

    # Compute
    parser.add_argument("--fp8", action="store_true", help="Enable FP8 training")
    parser.add_argument("--compile", action="store_true", default=True, help="torch.compile")
    parser.add_argument("--device", type=str, default="cuda", help="Device")

    # Data
    parser.add_argument("--data-dir", type=str, default="data/dclm", help="Data directory")

    # Logging
    parser.add_argument("--wandb", action="store_true", default=True, help="Enable W&B")
    parser.add_argument("--wandb-project", type=str, default="slm", help="W&B project")
    parser.add_argument("--log-interval", type=int, default=10, help="Log interval")

    # Checkpointing
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/phase0", help="Checkpoint directory")
    parser.add_argument("--save-every", type=int, default=-1, help="Save every N steps (-1 = only final)")

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
