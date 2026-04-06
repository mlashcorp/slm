"""
HuggingFace Hub checkpoint push/pull utilities for Phase 5.

Usage (push after training step):
    from phase5.checkpointing.hf_checkpoint import push_checkpoint, pull_checkpoint

    push_checkpoint(
        checkpoint_dir="/workspace/datasets/phase5/e2e/nanochat/base_checkpoints/d8",
        step=300,
        run_id="phase5_canary",
        hf_repo="mlashcorp/slm-phase5-checkpoints",
        hf_token=os.environ["HF_TOKEN"],
    )

Usage (pull to resume):
    local_dir = pull_checkpoint(
        run_id="phase5_canary",
        step=300,
        hf_repo="mlashcorp/slm-phase5-checkpoints",
        hf_token=os.environ["HF_TOKEN"],
        dest_dir="/workspace/datasets/phase5/e2e/nanochat/base_checkpoints/d8",
    )
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_api(hf_token: str):
    from huggingface_hub import HfApi
    return HfApi(token=hf_token)


def _repo_path(run_id: str, step: int) -> str:
    """Path inside the HF repo for a given run and step."""
    return f"{run_id}/step_{step:07d}"


# ---------------------------------------------------------------------------
# Push
# ---------------------------------------------------------------------------

def push_checkpoint(
    checkpoint_dir: str | Path,
    step: int,
    run_id: str,
    hf_repo: str,
    hf_token: str,
    private: bool = True,
    non_blocking: bool = False,
    extra_meta: Optional[dict] = None,
) -> None:
    """
    Upload a nanochat checkpoint directory to HuggingFace Hub.

    The checkpoint is stored at:
        {hf_repo}/{run_id}/step_{step:07d}/

    Uploads all files in checkpoint_dir that match:
        model_step_{step:07d}.pt
        optimizer_step_{step:07d}.pt
        meta_step_{step:07d}.json

    Also writes a manifest.json at the run root so you can list
    available checkpoints without downloading anything.

    Args:
        checkpoint_dir: local directory where nanochat saves checkpoints.
        step: training step number.
        run_id: human-readable run identifier (e.g. "phase5_A3_80_20").
        hf_repo: HuggingFace repo id (e.g. "mlashcorp/slm-phase5-checkpoints").
        hf_token: HuggingFace write token.
        private: whether to create the repo as private (default True).
        non_blocking: if True, upload runs in a background thread.
        extra_meta: additional metadata dict merged into the manifest entry.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    checkpoint_dir = Path(checkpoint_dir)

    # Ensure the repo exists
    api.create_repo(
        repo_id=hf_repo,
        repo_type="model",
        private=private,
        exist_ok=True,
    )

    # Collect checkpoint files for this step
    # Nanochat uses format: model_{step:06d}.pt, optim_{step:06d}_rank0.pt, meta_{step:06d}.json
    patterns = [
        f"model_{step:06d}.pt",
        f"optim_{step:06d}_rank0.pt",
        f"meta_{step:06d}.json",
    ]
    files_to_upload = []
    for pattern in patterns:
        p = checkpoint_dir / pattern
        if p.exists():
            files_to_upload.append(p)

    if not files_to_upload:
        print(f"[hf_checkpoint] WARNING: no checkpoint files found for step {step} in {checkpoint_dir}")
        return

    repo_subdir = _repo_path(run_id, step)
    print(f"[hf_checkpoint] Uploading {len(files_to_upload)} files to {hf_repo}/{repo_subdir}")

    for local_file in files_to_upload:
        path_in_repo = f"{repo_subdir}/{local_file.name}"
        future = api.upload_file(
            path_or_fileobj=str(local_file),
            path_in_repo=path_in_repo,
            repo_id=hf_repo,
            repo_type="model",
            run_as_future=non_blocking,
        )
        if not non_blocking:
            print(f"[hf_checkpoint]   uploaded {local_file.name} -> {path_in_repo}")

    # Read meta to build manifest entry
    meta_path = checkpoint_dir / f"meta_step_{step:07d}.json"
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    # Update manifest
    _update_manifest(
        api=api,
        hf_repo=hf_repo,
        run_id=run_id,
        step=step,
        files=[f.name for f in files_to_upload],
        meta=meta,
        extra_meta=extra_meta or {},
    )

    print(f"[hf_checkpoint] Done. Checkpoint step={step} pushed to {hf_repo}")


def _update_manifest(
    api,
    hf_repo: str,
    run_id: str,
    step: int,
    files: list[str],
    meta: dict,
    extra_meta: dict,
) -> None:
    """
    Maintain a JSON manifest at {run_id}/manifest.json listing all
    uploaded checkpoints so callers can list available steps cheaply.
    """
    import io
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    manifest_path_in_repo = f"{run_id}/manifest.json"

    # Try to load existing manifest
    existing_manifest = {"run_id": run_id, "checkpoints": []}
    try:
        local = hf_hub_download(
            repo_id=hf_repo,
            filename=manifest_path_in_repo,
            repo_type="model",
            token=api.token,
        )
        with open(local) as f:
            existing_manifest = json.load(f)
    except (EntryNotFoundError, Exception):
        pass  # first push, manifest doesn't exist yet

    # Remove existing entry for this step if present (idempotent)
    existing_manifest["checkpoints"] = [
        c for c in existing_manifest["checkpoints"] if c.get("step") != step
    ]

    # Add new entry
    entry = {
        "step": step,
        "uploaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": files,
        "repo_path": _repo_path(run_id, step),
        "val_bpb": meta.get("val_bpb"),
        "model_config": meta.get("model_config", {}),
    }
    entry.update(extra_meta)
    existing_manifest["checkpoints"].append(entry)
    existing_manifest["checkpoints"].sort(key=lambda c: c["step"])
    existing_manifest["latest_step"] = existing_manifest["checkpoints"][-1]["step"]

    manifest_bytes = json.dumps(existing_manifest, indent=2).encode()
    api.upload_file(
        path_or_fileobj=io.BytesIO(manifest_bytes),
        path_in_repo=manifest_path_in_repo,
        repo_id=hf_repo,
        repo_type="model",
        commit_message=f"Update manifest: step {step}",
    )


# ---------------------------------------------------------------------------
# Pull / Resume
# ---------------------------------------------------------------------------

def pull_checkpoint(
    run_id: str,
    step: int | None,
    hf_repo: str,
    hf_token: str,
    dest_dir: str | Path,
) -> Path:
    """
    Download a checkpoint from HuggingFace Hub to a local directory.

    If step is None, downloads the latest checkpoint recorded in the manifest.

    Returns the local checkpoint directory path.
    """
    from huggingface_hub import HfApi, hf_hub_download, list_repo_files
    from huggingface_hub.utils import EntryNotFoundError

    api = HfApi(token=hf_token)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Resolve step from manifest if not given
    if step is None:
        manifest_path = f"{run_id}/manifest.json"
        try:
            local = hf_hub_download(
                repo_id=hf_repo,
                filename=manifest_path,
                repo_type="model",
                token=hf_token,
            )
            with open(local) as f:
                manifest = json.load(f)
            step = manifest["latest_step"]
            print(f"[hf_checkpoint] Resolved latest step from manifest: {step}")
        except Exception as e:
            raise RuntimeError(
                f"Could not resolve latest step from manifest at {hf_repo}/{manifest_path}: {e}"
            )

    repo_subdir = _repo_path(run_id, step)
    print(f"[hf_checkpoint] Downloading checkpoint from {hf_repo}/{repo_subdir} -> {dest_dir}")

    # List files at that path in the repo
    all_files = list(list_repo_files(repo_id=hf_repo, repo_type="model", token=hf_token))
    ckpt_files = [f for f in all_files if f.startswith(repo_subdir + "/")]

    if not ckpt_files:
        raise FileNotFoundError(
            f"No checkpoint files found at {hf_repo}/{repo_subdir}. "
            f"Available paths: {set('/'.join(f.split('/')[:2]) for f in all_files)}"
        )

    for repo_file in ckpt_files:
        filename = Path(repo_file).name
        local_dest = dest_dir / filename
        print(f"[hf_checkpoint]   downloading {repo_file} -> {local_dest}")
        downloaded = hf_hub_download(
            repo_id=hf_repo,
            filename=repo_file,
            repo_type="model",
            token=hf_token,
            local_dir=str(dest_dir),
        )
        # hf_hub_download may nest under subdirs; move to dest_dir root
        downloaded_path = Path(downloaded)
        if downloaded_path != local_dest and downloaded_path.exists():
            shutil.move(str(downloaded_path), str(local_dest))

    print(f"[hf_checkpoint] Checkpoint step={step} downloaded to {dest_dir}")
    return dest_dir


# ---------------------------------------------------------------------------
# List available checkpoints for a run
# ---------------------------------------------------------------------------

def list_checkpoints(
    run_id: str,
    hf_repo: str,
    hf_token: str,
) -> list[dict]:
    """
    Return list of checkpoint metadata dicts for a run, sorted by step.
    Reads from the run's manifest.json without downloading model weights.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    manifest_path = f"{run_id}/manifest.json"
    try:
        local = hf_hub_download(
            repo_id=hf_repo,
            filename=manifest_path,
            repo_type="model",
            token=hf_token,
        )
        with open(local) as f:
            manifest = json.load(f)
        return manifest.get("checkpoints", [])
    except EntryNotFoundError:
        return []
    except Exception as e:
        print(f"[hf_checkpoint] WARNING: could not fetch manifest: {e}")
        return []


# ---------------------------------------------------------------------------
# CLI entrypoint for use in shell scripts
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HuggingFace checkpoint push/pull/list")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # push
    p = sub.add_parser("push", help="Upload a checkpoint to HF Hub")
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--hf-repo", required=True)
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))

    # pull
    p = sub.add_parser("pull", help="Download a checkpoint from HF Hub")
    p.add_argument("--run-id", required=True)
    p.add_argument("--step", type=int, default=None, help="Step to download (default: latest)")
    p.add_argument("--hf-repo", required=True)
    p.add_argument("--dest-dir", required=True)
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))

    # list
    p = sub.add_parser("list", help="List available checkpoints for a run")
    p.add_argument("--run-id", required=True)
    p.add_argument("--hf-repo", required=True)
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))

    args = parser.parse_args()

    if args.cmd == "push":
        push_checkpoint(
            checkpoint_dir=args.checkpoint_dir,
            step=args.step,
            run_id=args.run_id,
            hf_repo=args.hf_repo,
            hf_token=args.hf_token,
        )
    elif args.cmd == "pull":
        pull_checkpoint(
            run_id=args.run_id,
            step=args.step,
            hf_repo=args.hf_repo,
            hf_token=args.hf_token,
            dest_dir=args.dest_dir,
        )
    elif args.cmd == "list":
        ckpts = list_checkpoints(
            run_id=args.run_id,
            hf_repo=args.hf_repo,
            hf_token=args.hf_token,
        )
        if not ckpts:
            print("No checkpoints found.")
        for c in ckpts:
            print(f"  step={c['step']:7d}  val_bpb={c.get('val_bpb')}  uploaded={c.get('uploaded_at')}  files={c.get('files')}")
