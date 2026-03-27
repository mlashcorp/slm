#!/usr/bin/env python
"""
Download DCLM (Data Comp for Language Models) dataset for Phase 0 training.

Based on nanochat's dataset.py:
https://github.com/karpathy/nanochat/blob/master/nanochat/dataset.py

DCLM is a high-quality pretraining dataset used in nanochat's GPT-2 reproduction.

Usage:
    python scripts/download_dclm.py -n 170  # Download 170 shards (~42B chars)
    python scripts/download_dclm.py -n 8    # Download 8 shards for tokenizer training
"""
import os
import argparse
import time
import requests
from multiprocessing import Pool
from pathlib import Path


# DCLM Baseline dataset on HuggingFace
# This is the dataset used by nanochat for pretraining
BASE_URL = "https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0-parquet/resolve/main"
MAX_SHARD = 1000  # Approximate number of shards in DCLM baseline

# Alternative: Use nanochat's ClimbMix (NVIDIA's curated dataset)
# BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
# MAX_SHARD = 6542


def index_to_filename(index: int) -> str:
    """Convert shard index to filename."""
    return f"shard_{index:05d}.parquet"


def get_data_dir() -> Path:
    """Get data directory for DCLM."""
    base_dir = Path(__file__).parent.parent / "data" / "dclm"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def download_single_file(index: int, data_dir: Path, base_url: str) -> bool:
    """Download a single shard file with retries.

    Args:
        index: Shard index
        data_dir: Directory to save files
        base_url: Base URL for downloads

    Returns:
        True if download succeeded
    """
    filename = index_to_filename(index)
    filepath = data_dir / filename

    # Skip if already exists
    if filepath.exists():
        print(f"Skipping {filename} (already exists)")
        return True

    url = f"{base_url}/{filename}"
    print(f"Downloading {filename}...")

    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            # Write to temp file first
            temp_path = filepath.with_suffix(".tmp")
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)

            # Move to final location
            temp_path.rename(filepath)
            print(f"Successfully downloaded {filename}")
            return True

        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")

            # Clean up partial files
            for path in [filepath.with_suffix(".tmp"), filepath]:
                if path.exists():
                    try:
                        path.unlink()
                    except:
                        pass

            if attempt < max_attempts:
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)

    print(f"Failed to download {filename} after {max_attempts} attempts")
    return False


def main():
    parser = argparse.ArgumentParser(description="Download DCLM dataset shards")
    parser.add_argument("-n", "--num-files", type=int, default=-1,
        help="Number of train shards to download (default: -1 for all)")
    parser.add_argument("-w", "--num-workers", type=int, default=4,
        help="Number of parallel download workers (default: 4)")
    parser.add_argument("--url", type=str, default=BASE_URL,
        help="Base URL for downloads")
    parser.add_argument("--max-shard", type=int, default=MAX_SHARD,
        help="Maximum shard index")
    args = parser.parse_args()

    data_dir = get_data_dir()
    print(f"Target directory: {data_dir}")

    # Determine which shards to download
    if args.num_files == -1:
        num_shards = args.max_shard
    else:
        num_shards = min(args.num_files, args.max_shard)

    ids_to_download = list(range(num_shards))

    # Always download the last shard as validation
    if args.max_shard not in ids_to_download:
        ids_to_download.append(args.max_shard)

    print(f"Downloading {len(ids_to_download)} shards using {args.num_workers} workers...")
    print()

    # Download in parallel
    with Pool(processes=args.num_workers) as pool:
        results = pool.starmap(
            download_single_file,
            [(i, data_dir, args.url) for i in ids_to_download]
        )

    # Report results
    successful = sum(1 for r in results if r)
    print()
    print(f"Done! Downloaded: {successful}/{len(ids_to_download)} shards to {data_dir}")

    if successful < len(ids_to_download):
        print("Some downloads failed. You can re-run this script to retry.")

    # List downloaded files
    downloaded = sorted(data_dir.glob("*.parquet"))
    print(f"Total shards in directory: {len(downloaded)}")


if __name__ == "__main__":
    main()
