"""
Checkpoint watcher: polls a training checkpoint directory and runs lm-eval-harness
on each new checkpoint, logging results to the same W&B run as the trainer.

Run in a separate tmux pane alongside training:
python scripts/eval_watcher.py --phase 1
python scripts/eval_watcher.py --phase 1fp8
python scripts/eval_watcher.py --phase 2

The watcher checks every --poll-interval seconds (default 120) for new step-* directories,
runs evaluate.py on ones it hasn't seen yet, then logs metrics to W&B.

Benchmarks (matching SmolLM2-135M base model evaluation):
- Code: humaneval, mbpp (pass@1)
- Knowledge/Reasoning: hellaswag, piqa, arc_easy, arc_challenge, winogrande, openbookqa, commonsense_qa, mmlu
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import wandb

PHASE_OUTPUT_DIRS = {
    "1": "./checkpoints/phase1",
    "1fp8": "./checkpoints/phase1-fp8",
    "2": "./checkpoints/phase2",
    "3": "./checkpoints/phase3",
}

EVAL_OUTPUT_DIR = "./evals"

CODE_TASKS = "humaneval,mbpp"
GENERAL_TASKS = "hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa,commonsense_qa,mmlu"
ALL_TASKS = f"{CODE_TASKS},{GENERAL_TASKS}"

ALL_EVAL_TASKS = [
    "humaneval", "mbpp",
    "hellaswag", "piqa", "arc_easy", "arc_challenge",
    "winogrande", "openbookqa", "commonsense_qa", "mmlu",
]


def parse_results(json_path: Path) -> dict[str, float]:
    """Extract metrics from lm-eval output JSON."""
    with open(json_path) as f:
        data = json.load(f)
    results = data.get("results", {})
    metrics = {}
    for task, task_results in results.items():
        task_lower = task.lower()
        for key, value in task_results.items():
            if value is None:
                continue
            key_lower = key.lower()
            if task_lower in ("humaneval", "mbpp"):
                if key_lower.startswith("pass@1") and not key_lower.endswith("stderr"):
                    metrics[f"eval/{task_lower}/pass@1"] = value
            else:
                if key_lower in ("acc", "acc_norm") and not key_lower.endswith("stderr"):
                    metric_name = "acc_norm" if "acc_norm" in key_lower else "acc"
                    metrics[f"eval/{task_lower}/{metric_name}"] = value
    return metrics


def step_from_name(name: str) -> int:
    """Extract step number from checkpoint directory name like 'step-00002000'."""
    return int(name.replace("step-", ""))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=list(PHASE_OUTPUT_DIRS.keys()))
    parser.add_argument("--poll-interval", type=int, default=120,
        help="Seconds between checkpoint directory polls (default: 120)")
    parser.add_argument("--eval-output-dir", default=EVAL_OUTPUT_DIR)
    parser.add_argument("--tasks", default="all",
        help="Tasks to run: 'code', 'general', 'all', or comma-separated list (default: all)")
    parser.add_argument("--batch_size", type=int, default=1,
        help="Batch size for evaluation (default: 1)")
    args = parser.parse_args()

    if args.tasks == "code":
        tasks = CODE_TASKS
    elif args.tasks == "general":
        tasks = GENERAL_TASKS
    elif args.tasks == "all":
        tasks = ALL_TASKS
    else:
        tasks = args.tasks

    checkpoint_dir = Path(PHASE_OUTPUT_DIRS[args.phase])
    eval_output_dir = Path(args.eval_output_dir)
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    evaluated = set()
    run = None

    print(f"Watching {checkpoint_dir} every {args.poll_interval}s ...")
    print(f"Tasks: {tasks}")

    while True:
        # Try to connect to W&B run using the ID written by the trainer
        if run is None:
            run_id_path = checkpoint_dir / "wandb_run_id.txt"
            if run_id_path.exists():
                run_id = run_id_path.read_text().strip()
                run = wandb.init(
                    project="slm",
                    id=run_id,
                    resume="must",
                    mode="online",
                )
                print(f"Connected to W&B run {run_id}")

                # Define eval metrics with custom step metric to allow logging at any step
                # This resolves the "step must be monotonically increasing" error
                wandb.define_metric("eval/*", step_metric="eval/step")
                print("Defined eval/* metrics with step_metric=eval/step")

        # Find checkpoints not yet evaluated
        if checkpoint_dir.exists():
            checkpoints = sorted(
                [d for d in checkpoint_dir.iterdir()
                 if d.is_dir() and d.name.startswith("step-")],
                key=lambda d: step_from_name(d.name),
            )
            for ckpt in checkpoints:
                if ckpt.name in evaluated:
                    continue

                step = step_from_name(ckpt.name)
                eval_out = eval_output_dir / f"phase{args.phase}_{ckpt.name}.json"

                print(f"\n[step {step}] Running eval on {ckpt} ...")
                result = subprocess.run(
                    [sys.executable, "scripts/evaluate.py",
                     "--checkpoint", str(ckpt),
                     "--output", str(eval_out),
                     "--tasks", tasks,
                     "--batch_size", str(args.batch_size)],
                    capture_output=False,
                )

                evaluated.add(ckpt.name)

                if result.returncode != 0:
                    print(f"[step {step}] Eval failed (exit code {result.returncode}), skipping W&B log.")
                    continue

                # Find the actual output file (lm-eval appends timestamp to filename)
                candidates = list(eval_output_dir.glob(f"phase{args.phase}_{ckpt.name}*.json"))
                output_file = max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None

                if output_file is None:
                    print(f"[step {step}] Output JSON not found, skipping W&B log.")
                    continue

                metrics = parse_results(output_file)
                print(f"[step {step}] Results: {metrics}")

                if run is not None:
                    # Include eval/step so metrics are plotted against checkpoint step, not training step
                    log_data = {"eval/step": step, **metrics}
                    wandb.log(log_data)
                    print(f"[step {step}] Logged to W&B.")

        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
