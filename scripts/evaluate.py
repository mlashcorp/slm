"""
Run lm-eval-harness against a checkpoint directory.

Usage:
  python scripts/evaluate.py --checkpoint ./checkpoints/step-10000 --output ./evals/step-10000.json

Note: sets HF_ALLOW_CODE_EVAL=1 required for humaneval/mbpp code execution tasks.
"""
import argparse
import os
import subprocess
import sys

TASKS = "humaneval,mbpp"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint dir (HF format)")
    parser.add_argument("--output", required=True, help="Path to output JSON")
    parser.add_argument("--tasks", default=TASKS)
    args = parser.parse_args()

    env = {**os.environ, "HF_ALLOW_CODE_EVAL": "1"}
    cmd = [
        sys.executable, "-m", "lm_eval", "run",
        "--model", "hf",
        "--model_args", f"pretrained={args.checkpoint}",
        "--tasks", args.tasks,
        "--confirm_run_unsafe_code",
        "--num_fewshot", "0",
        "--output_path", args.output,
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)

if __name__ == "__main__":
    main()
