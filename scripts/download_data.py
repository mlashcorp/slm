"""
Download smollm-corpus python-edu and FineWeb-Edu sample-10BT.
Saves to ./data/ as Arrow shards for fast streaming.

Usage: python scripts/download_data.py
"""
from datasets import load_dataset
import os

os.makedirs("./data/python_edu", exist_ok=True)
os.makedirs("./data/fineweb_edu", exist_ok=True)

print("Downloading smollm-corpus python-edu...")
ds = load_dataset(
    "HuggingFaceTB/smollm-corpus",
    "python-edu",
    split="train",
    streaming=False,
)
ds.save_to_disk("./data/python_edu")
print(f"python-edu: {len(ds):,} examples")

print("Downloading FineWeb-Edu sample-10BT...")
fw = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name="sample-10BT",
    split="train",
    streaming=False,
)
fw.save_to_disk("./data/fineweb_edu")
print(f"fineweb-edu: {len(fw):,} examples")
