#!/usr/bin/env python
"""
CORE Metric Evaluation for Phase 0.

Evaluates model on DCLM CORE benchmark tasks:
- HellaSwag
- PIQA
- ARC (Easy and Challenge)
- Winogrande
- MMLU

Based on nanochat's core_eval.py:
https://github.com/karpathy/nanochat/blob/master/nanochat/core_eval.py

The CORE metric is a normalized aggregate score used by nanochat for
comparing models. GPT-2 (1.5B) baseline is ~0.2565.
"""
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.nanochat_config import NanoChatConfig
from src.model.nanochat_gpt import GPT


# =============================================================================
# Task Definitions
# =============================================================================

CORE_TASKS = [
    "hellaswag",
    "piqa",
    "arc_easy",
    "arc_challenge",
    "winogrande",
    "mmlu",
]


@dataclass
class TaskResult:
    """Result for a single task."""
    task: str
    accuracy: float
    num_samples: int
    details: Optional[Dict] = None


# =============================================================================
# Model Loading
# =============================================================================

def load_model(checkpoint_path: Path, device: str = "cuda") -> tuple:
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model to

    Returns:
        (model, config) tuple
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create config from checkpoint
    config_dict = checkpoint.get("config", {})
    config = NanoChatConfig(**config_dict)

    # Create and load model
    model = GPT(config)
    model.init_weights()
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, config


# =============================================================================
# Evaluation Functions
# =============================================================================

@torch.no_grad()
def evaluate_log_likelihood(
    model: GPT,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Evaluate log likelihood of sequences.

    Args:
        model: GPT model
        input_ids: (batch, seq_len) input tokens
        attention_mask: (batch, seq_len) attention mask

    Returns:
        (batch,) log likelihoods
    """
    # Get logits
    logits = model(input_ids)  # (batch, seq_len, vocab)

    # Compute log probs
    log_probs = F.log_softmax(logits, dim=-1)

    # Gather log probs for target tokens (shifted by 1)
    targets = input_ids[:, 1:]  # (batch, seq_len - 1)
    log_probs = log_probs[:, :-1, :]  # (batch, seq_len - 1, vocab)

    # Gather
    gathered = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)  # (batch, seq_len - 1)

    # Mask padding
    mask = attention_mask[:, 1:].float()  # (batch, seq_len - 1)
    log_likelihood = (gathered * mask).sum(dim=1)  # (batch,)

    return log_likelihood


@torch.no_grad()
def evaluate_multiple_choice(
    model: GPT,
    prompts: List[str],
    choices_list: List[List[str]],
    tokenizer,
    device: str = "cuda",
) -> List[int]:
    """Evaluate multiple choice questions.

    Args:
        model: GPT model
        prompts: List of question prompts
        choices_list: List of choice lists for each question
        tokenizer: Tokenizer
        device: Device

    Returns:
        List of predicted choice indices
    """
    predictions = []

    for prompt, choices in zip(prompts, choices_list):
        best_idx = 0
        best_ll = float("-inf")

        for i, choice in enumerate(choices):
            # Combine prompt and choice
            text = f"{prompt} {choice}"

            # Tokenize
            tokens = tokenizer.encode([text], prepend=tokenizer.get_bos_token_id())[0]
            input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)

            # Evaluate log likelihood
            ll = evaluate_log_likelihood(model, input_ids, attention_mask)

            if ll.item() > best_ll:
                best_ll = ll.item()
                best_idx = i

        predictions.append(best_idx)

    return predictions


@torch.no_grad()
def evaluate_task(
    model: GPT,
    task_name: str,
    max_samples: int = 500,
    device: str = "cuda",
) -> TaskResult:
    """Evaluate model on a single task.

    This is a simplified evaluation that uses lm-eval-harness format.
    For full evaluation, use: lm_eval --model hf --model_args path/to/model --tasks <task>

    Args:
        model: GPT model
        task_name: Name of the task
        max_samples: Maximum samples to evaluate
        device: Device

    Returns:
        TaskResult
    """
    try:
        from lm_eval import evaluator
        from lm_eval.models.huggingface import HuggingFaceAutoLM
    except ImportError:
        print("Warning: lm-eval not installed. Install with: pip install lm-eval")
        return TaskResult(task_name, 0.0, 0, {"error": "lm-eval not installed"})

    # This is a placeholder - for full evaluation, use lm-eval directly
    # The actual implementation would need to wrap our model for lm-eval
    print(f"Note: Full evaluation requires lm-eval. Run with:")
    print(f"  lm_eval --model hf --model_args {model.config} --tasks {task_name}")

    return TaskResult(task_name, 0.0, 0, {"note": "use lm-eval for full evaluation"})


# =============================================================================
# CORE Score Calculation
# =============================================================================

def calculate_core_score(results: List[TaskResult]) -> float:
    """Calculate normalized CORE score.

    The CORE metric normalizes task accuracies to a 0-1 scale
    and averages them.

    Reference: GPT-2 (1.5B) baseline is ~0.2565

    Args:
        results: List of task results

    Returns:
        CORE score (0-1)
    """
    if not results:
        return 0.0

    # Simple average for now
    # Full implementation would normalize by task difficulty
    valid_results = [r for r in results if r.num_samples > 0]
    if not valid_results:
        return 0.0

    return sum(r.accuracy for r in valid_results) / len(valid_results)


# =============================================================================
# Main Evaluation
# =============================================================================

def evaluate(
    checkpoint_path: Path,
    tasks: List[str] = None,
    max_samples: int = 500,
    device: str = "cuda",
    output_path: Optional[Path] = None,
) -> Dict:
    """Evaluate model on CORE tasks.

    Args:
        checkpoint_path: Path to model checkpoint
        tasks: List of tasks (default: all CORE tasks)
        max_samples: Maximum samples per task
        device: Device
        output_path: Path to save results

    Returns:
        Dict with evaluation results
    """
    if tasks is None:
        tasks = CORE_TASKS

    print(f"Loading model from {checkpoint_path}...")
    model, config = load_model(checkpoint_path, device)

    print(f"\nModel: n_layer={config.n_layer}, n_embd={config.n_embd}")
    print(f"Parameters: {model.num_parameters():,}")

    print(f"\nEvaluating on {len(tasks)} tasks (max {max_samples} samples each)...")

    results = []
    for task in tasks:
        print(f"\n  {task}...")
        result = evaluate_task(model, task, max_samples, device)
        results.append(result)
        print(f"    Accuracy: {result.accuracy:.4f} ({result.num_samples} samples)")

    core_score = calculate_core_score(results)

    output = {
        "checkpoint": str(checkpoint_path),
        "model_params": model.num_parameters(),
        "tasks": {r.task: {"accuracy": r.accuracy, "num_samples": r.num_samples} for r in results},
        "core_score": core_score,
    }

    print(f"\n{'='*60}")
    print(f"CORE Score: {core_score:.4f}")
    print(f"{'='*60}")

    if output_path:
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {output_path}")

    return output


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 0: CORE Metric Evaluation")

    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--tasks", type=str, default="all", help="Tasks (comma-separated or 'all')")
    parser.add_argument("--max-samples", type=int, default=500, help="Max samples per task")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")

    args = parser.parse_args()

    # Parse tasks
    if args.tasks == "all":
        tasks = CORE_TASKS
    else:
        tasks = [t.strip() for t in args.tasks.split(",")]

    # Evaluate
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output) if args.output else None

    results = evaluate(
        checkpoint_path=checkpoint_path,
        tasks=tasks,
        max_samples=args.max_samples,
        device=args.device,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
