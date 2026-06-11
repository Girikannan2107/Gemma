#!/usr/bin/env python3
"""
Phase 1 — Model Download
Downloads google/gemma-4-E4B-it from HuggingFace.

Gemma 4 is a gated model — you need:
  1. A HuggingFace account
  2. Accept the license at https://huggingface.co/google/gemma-4-E4B-it
  3. A read token: https://huggingface.co/settings/tokens

Usage:
    python download_model.py --token hf_xxxxxxxx
    python download_model.py --token hf_xxxxxxxx --cache_dir /path/to/large/disk

Why keep model separate from code: the uncompressed model is ~10 GB.
During Phase 2, compressed variants will be ~3–5 GB.
Keeping them in a separate directory avoids cluttering the repo.
"""

import os
import sys
import argparse
import time
from pathlib import Path


def human_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def download_model(model_id: str, local_dir: str, token: str):
    from huggingface_hub import snapshot_download, login, HfApi
    import huggingface_hub

    print(f"\n{'='*60}")
    print(f"  Downloading: {model_id}")
    print(f"  Destination: {local_dir}")
    print(f"  HF version : {huggingface_hub.__version__}")
    print(f"{'='*60}\n")

    # Login
    if token:
        login(token=token, add_to_git_credential=False)
        print("✓ Logged in to HuggingFace\n")
    else:
        print("⚠  No token provided — will fail for gated models.\n")

    # Check model exists and is accessible
    api = HfApi()
    try:
        info = api.model_info(model_id, token=token)
        print(f"Model info:")
        print(f"  ID      : {info.modelId}")
        print(f"  License : {info.cardData.get('license', 'unknown') if info.cardData else 'unknown'}")
        print(f"  Tags    : {info.tags[:5] if info.tags else []}")
        print()
    except Exception as e:
        print(f"⚠  Could not fetch model info: {e}")
        print("   Proceeding with download anyway...\n")

    Path(local_dir).mkdir(parents=True, exist_ok=True)

    # Download — skip TF and Flax weights to save disk space
    t0 = time.time()
    downloaded_path = snapshot_download(
        repo_id=model_id,
        token=token,
        local_dir=local_dir,
        ignore_patterns=[
            "*.msgpack",            # Flax weights
            "*.h5",                 # Keras weights
            "flax_model*",
            "tf_model*",
            "rust_model*",
        ],
    )
    elapsed = time.time() - t0

    # Report size
    all_files = list(Path(downloaded_path).rglob("*"))
    files = [f for f in all_files if f.is_file()]
    total_bytes = sum(f.stat().st_size for f in files)

    print(f"\n✓ Download complete in {elapsed:.0f}s")
    print(f"  Files     : {len(files)}")
    print(f"  Total size: {human_size(total_bytes)}")
    print(f"  Location  : {downloaded_path}\n")

    # List key files
    print("Key files:")
    for f in sorted(files):
        rel = f.relative_to(downloaded_path)
        if any(rel.name.endswith(ext) for ext in [".json", ".safetensors", ".bin", ".gguf"]):
            print(f"  {rel}  ({human_size(f.stat().st_size)})")

    return downloaded_path


def main():
    p = argparse.ArgumentParser(description="Download google/gemma-4-E4B-it")
    p.add_argument(
        "--model_id",
        default="google/gemma-4-E4B-it",
        help="HuggingFace model ID",
    )
    p.add_argument(
        "--local_dir",
        default="../models/gemma-4-E4B-it",
        help="Where to store the model (needs ~15 GB free)",
    )
    p.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN", ""),
        help="HuggingFace read token (or set HF_TOKEN env var)",
    )
    args = p.parse_args()

    if not args.token:
        print("ERROR: No HuggingFace token. Provide --token or set HF_TOKEN.")
        print("Get a token at: https://huggingface.co/settings/tokens")
        print("Accept the Gemma 4 license at: https://huggingface.co/google/gemma-4-E4B-it")
        sys.exit(1)

    download_model(args.model_id, args.local_dir, args.token)


if __name__ == "__main__":
    main()
