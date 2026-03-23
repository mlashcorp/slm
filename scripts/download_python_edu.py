"""
High-throughput downloader for smollm-corpus python-edu.

Fetches actual Python source code from the Software Heritage S3 bucket
(public, anonymous access) using a ThreadPoolExecutor for I/O concurrency.

Usage:
    # Full dataset (~7.68M files, hours outside AWS)
    python scripts/download_python_edu.py

    # Quick end-to-end test (~5 min outside AWS)
    python scripts/download_python_edu.py --sample 10000

    # Custom concurrency
    python scripts/download_python_edu.py --sample 10000 --num-workers 128

    # Resume interrupted download (skips already-fetched blob_ids)
    python scripts/download_python_edu.py --resume

Options:
    --sample N        Download only the first N rows (default: full dataset)
    --num-workers N   Concurrent S3 threads (default: 256)
    --output DIR      Output directory (default: ./data/python_edu_hydrated)
    --resume          Skip blob_ids already present in output dataset
"""
import argparse
import gzip
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import botocore
from botocore.config import Config
from botocore.exceptions import ClientError
from datasets import Dataset, load_dataset, load_from_disk
from tqdm import tqdm

BUCKET = "softwareheritage"

# Anonymous access — no AWS credentials needed.
# max_pool_connections must exceed num_workers or boto3 serializes connections.
def make_s3_client(num_workers: int) -> boto3.client:
    return boto3.client(
        "s3",
        region_name="us-east-1",
        config=Config(
            signature_version=botocore.UNSIGNED,
            max_pool_connections=num_workers + 50,
            retries={"mode": "adaptive", "total_max_attempts": 10},
            connect_timeout=10,
            read_timeout=30,
            tcp_keepalive=True,
        ),
    )


def fetch_one(s3, blob_id: str) -> dict:
    key = f"content/{blob_id}"
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        with gzip.GzipFile(fileobj=obj["Body"]) as f:
            text = f.read().decode("utf-8", errors="ignore")
        return {"blob_id": blob_id, "text": text, "download_success": True}
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("NoSuchKey", "404"):
            return {"blob_id": blob_id, "text": "", "download_success": False}
        raise


def download(
    metadata_ds,
    output_dir: str,
    num_workers: int,
    resume: bool,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Resume: load already-fetched blob_ids so we can skip them
    done_ids: set[str] = set()
    resume_ds = None
    if resume and (output_path / "dataset_info.json").exists():
        print("Resuming — loading existing dataset...")
        resume_ds = load_from_disk(str(output_path))
        done_ids = set(resume_ds["blob_id"])
        print(f"  Already downloaded: {len(done_ids):,} files")

    blob_ids = metadata_ds["blob_id"]
    if done_ids:
        blob_ids = [b for b in blob_ids if b not in done_ids]
    total = len(blob_ids)
    print(f"Files to download: {total:,} (workers: {num_workers})")

    if total == 0:
        print("Nothing to do.")
        return

    s3 = make_s3_client(num_workers)

    rows: list[dict] = []
    failed = 0
    # Submit in batches so we don't pre-allocate millions of Future objects at once.
    BATCH = num_workers * 4

    with tqdm(total=total, unit="file", dynamic_ncols=True) as pbar:
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            for batch_start in range(0, total, BATCH):
                batch = blob_ids[batch_start : batch_start + BATCH]
                futures = {pool.submit(fetch_one, s3, bid): bid for bid in batch}
                for fut in as_completed(futures):
                    result = fut.result()
                    rows.append(result)
                    if not result["download_success"]:
                        failed += 1
                    pbar.update(1)
                    pbar.set_postfix(failed=failed, refresh=False)

    # Merge with any previously downloaded data
    new_ds = Dataset.from_list(rows)
    if resume_ds is not None:
        from datasets import concatenate_datasets
        # Drop reference to resume_ds before saving to avoid overwrite conflict
        merged = concatenate_datasets([resume_ds, new_ds])
        del resume_ds
    else:
        merged = new_ds

    success_ds = merged.filter(lambda x: x["download_success"])

    # Save to a temp path first, then replace, so we never overwrite the source
    import shutil
    tmp_path = output_path.parent / (output_path.name + "_tmp")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    success_ds.save_to_disk(str(tmp_path))
    if output_path.exists():
        shutil.rmtree(output_path)
    shutil.move(str(tmp_path), str(output_path))

    print(f"\nDone. {len(success_ds):,} files saved to {output_path}")
    print(f"Failed (NoSuchKey): {failed:,}")


def main():
    parser = argparse.ArgumentParser(description="Hydrate smollm-corpus python-edu from S3")
    parser.add_argument("--sample", type=int, default=None,
                        help="Download only the first N rows (omit for full dataset)")
    parser.add_argument("--num-workers", type=int, default=256,
                        help="Concurrent S3 threads (default: 256)")
    parser.add_argument("--output", type=str, default="./data/python_edu_hydrated",
                        help="Output directory (default: ./data/python_edu_hydrated)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip blob_ids already present in output dataset")
    args = parser.parse_args()

    print("Loading python-edu metadata from HuggingFace...")
    metadata_ds = load_dataset(
        "HuggingFaceTB/smollm-corpus",
        "python-edu",
        split="train",
        streaming=False,
    )

    if args.sample is not None:
        metadata_ds = metadata_ds.select(range(min(args.sample, len(metadata_ds))))
        print(f"Sample mode: {len(metadata_ds):,} files")
    else:
        print(f"Full dataset: {len(metadata_ds):,} files")

    download(
        metadata_ds=metadata_ds,
        output_dir=args.output,
        num_workers=args.num_workers,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
