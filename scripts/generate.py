"""
Generate text from a saved checkpoint.

Usage:
  python scripts/generate.py --checkpoint ./checkpoints/phase1/step-00002000 --prompt "def fibonacci(n):"
  python scripts/generate.py --checkpoint ./checkpoints/phase1/step-00002000 --prompt "def fibonacci(n):" --max_new_tokens 200 --temperature 0.8
"""
import argparse
import torch
from transformers import AutoModelForCausalLM
from src.data.tokenizer import load_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint directory (HF format)")
    parser.add_argument("--prompt", required=True, help="Input prompt string")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0, help="0.0 = greedy")
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    tokenizer = load_tokenizer()
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    model.eval()

    inputs = tokenizer(args.prompt, return_tensors="pt").to(args.device)

    do_sample = args.temperature > 0.0
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=do_sample,
            temperature=args.temperature if do_sample else None,
            top_p=args.top_p if do_sample else None,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    completion = tokenizer.decode(new_tokens, skip_special_tokens=True)

    print(f"--- Prompt ---\n{args.prompt}")
    print(f"--- Completion ---\n{completion}")


if __name__ == "__main__":
    main()
