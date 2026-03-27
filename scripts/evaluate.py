"""
Run lm-eval-harness against a checkpoint directory.

Usage:
python scripts/evaluate.py --checkpoint ./checkpoints/step-10000 --output ./evals/step-10000.json
python scripts/evaluate.py --checkpoint ./checkpoints/step-10000 --output ./evals/step-10000.json --tasks all

Note: sets HF_ALLOW_CODE_EVAL=1 required for humaneval/mbpp code execution tasks.

Benchmarks (matching SmolLM2-135M base model evaluation):
- Code: humaneval, mbpp
- Knowledge/Reasoning: hellaswag, piqa, arc_easy, arc_challenge, winogrande, openbookqa, commonsense_qa, mmlu
"""
import argparse
import os
import subprocess
import sys

CODE_TASKS = "humaneval,mbpp"
GENERAL_TASKS = "hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa,commonsense_qa,mmlu"
ALL_TASKS = f"{CODE_TASKS},{GENERAL_TASKS}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint dir (HF format)")
    parser.add_argument("--output", required=True, help="Path to output JSON")
    parser.add_argument("--tasks", default="all", help="Tasks to run: 'code', 'general', 'all', or comma-separated list (default: all)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation (default: 1)")
    args = parser.parse_args()

    if args.tasks == "code":
        tasks = CODE_TASKS
    elif args.tasks == "general":
        tasks = GENERAL_TASKS
    elif args.tasks == "all":
        tasks = ALL_TASKS
    else:
        tasks = args.tasks

    env = {**os.environ, "HF_ALLOW_CODE_EVAL": "1"}
    cmd = [
        sys.executable, "-m", "lm_eval", "run",
        "--model", "hf",
        "--model_args", f"pretrained={args.checkpoint},dtype=bfloat16",
        "--tasks", tasks,
        "--confirm_run_unsafe_code",
        "--num_fewshot", "0",
        "--device", "cuda",
        "--batch_size", str(args.batch_size),
        "--output_path", args.output,
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)

if __name__ == "__main__":
    main()
