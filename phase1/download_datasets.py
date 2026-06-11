#!/usr/bin/env python3
"""
Phase 1 — Dataset Download
Downloads evaluation benchmarks and calibration data.

EVALUATION benchmarks (to measure the 100% baseline):
  - AI2D         : diagram multiple-choice QA  (~15k images)
  - MMBench      : general multimodal multiple-choice (~3k images)
  - TextVQA      : open-ended text-in-image QA (~5k val images)

CALIBRATION data (for Phase 2 AWQ/GPTQ quantization):
  - COCO val2017 : 5k diverse natural images + captions
    We use 512 image-caption pairs as the calibration corpus.
    Scientific reason: AWQ/GPTQ need a small representative dataset to
    compute activation statistics / Hessians. 512 samples is standard.
    The calibration set must be DISTINCT from the evaluation set.

Usage:
    python download_datasets.py --data_dir ../data
    python download_datasets.py --data_dir ../data --calibration_only
    python download_datasets.py --data_dir ../data --eval_only
"""

import os
import json
import argparse
import random
import urllib.request
import zipfile
import tarfile
from pathlib import Path
from typing import Optional


# ── helpers ───────────────────────────────────────────────────────────────────

def download_file(url: str, dest: Path, desc: str = "") -> Path:
    """Download a URL to dest with a simple progress indicator."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [skip] {dest.name} already exists")
        return dest

    label = desc or dest.name
    print(f"  Downloading {label}...", end="", flush=True)

    def _progress(block, block_size, total):
        if total > 0:
            pct = min(100, block * block_size * 100 // total)
            print(f"\r  Downloading {label}... {pct}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, _progress)
    size = dest.stat().st_size
    print(f"\r  ✓ {label} ({size / 1e6:.1f} MB)")
    return dest


def extract_zip(archive: Path, dest_dir: Path):
    print(f"  Extracting {archive.name}...")
    with zipfile.ZipFile(archive, "r") as z:
        z.extractall(dest_dir)
    print(f"  ✓ Extracted to {dest_dir}")


def extract_tar(archive: Path, dest_dir: Path):
    print(f"  Extracting {archive.name}...")
    with tarfile.open(archive, "r:gz") as t:
        t.extractall(dest_dir)
    print(f"  ✓ Extracted to {dest_dir}")


# ── evaluation benchmarks (HuggingFace datasets) ─────────────────────────────

def download_ai2d(data_dir: Path):
    """
    AI2D: ~15k diagram-based multiple-choice QA.
    Science diagrams with 4 answer choices.
    Why: Clean accuracy metric (A/B/C/D), strong proxy for
         multiple-choice evaluation used in the competition.
    HF: https://huggingface.co/datasets/lmms-lab/ai2d
    """
    print("\n[AI2D — Diagram QA, multiple choice]")
    try:
        from datasets import load_dataset
        ds = load_dataset("lmms-lab/ai2d", split="test", cache_dir=str(data_dir / "ai2d"))
        # Save a JSONL with: id, image_path, question, choices, answer
        out_path = data_dir / "ai2d" / "ai2d_test.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not out_path.exists():
            img_dir = data_dir / "ai2d" / "images"
            img_dir.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                for i, ex in enumerate(ds):
                    img_file = img_dir / f"{i:06d}.png"
                    if not img_file.exists():
                        ex["image"].save(img_file)
                    record = {
                        "id": i,
                        "image_path": str(img_file),
                        "question": ex.get("question", ""),
                        "choices": ex.get("options", []),
                        "answer": ex.get("answer", ""),
                    }
                    f.write(json.dumps(record) + "\n")
            print(f"  ✓ AI2D saved: {len(ds)} examples → {out_path}")
        else:
            print(f"  [skip] Already processed: {out_path}")
    except Exception as e:
        print(f"  ⚠  AI2D download failed: {e}")
        print("     Manual alt: https://huggingface.co/datasets/lmms-lab/ai2d")


def download_mmbench(data_dir: Path):
    """
    MMBench English test set (~3k examples).
    Multiple choice with A/B/C/D options covering diverse visual abilities.
    Why: Standard competition-style multimodal MC benchmark.
    HF: https://huggingface.co/datasets/lmms-lab/MMBench_EN
    """
    print("\n[MMBench — General multimodal multiple choice]")
    try:
        from datasets import load_dataset
        ds = load_dataset("lmms-lab/MMBench_EN", split="test", cache_dir=str(data_dir / "mmbench"))
        out_path = data_dir / "mmbench" / "mmbench_test.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not out_path.exists():
            img_dir = data_dir / "mmbench" / "images"
            img_dir.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                for i, ex in enumerate(ds):
                    img_file = img_dir / f"{i:06d}.png"
                    if hasattr(ex.get("image", None), "save") and not img_file.exists():
                        ex["image"].save(img_file)
                    record = {
                        "id": i,
                        "image_path": str(img_file),
                        "question": ex.get("question", ""),
                        "choices": [ex.get(k, "") for k in ["A", "B", "C", "D"] if ex.get(k)],
                        "answer": ex.get("answer", ""),
                    }
                    f.write(json.dumps(record) + "\n")
            print(f"  ✓ MMBench saved: {len(ds)} examples → {out_path}")
        else:
            print(f"  [skip] Already processed: {out_path}")
    except Exception as e:
        print(f"  ⚠  MMBench download failed: {e}")
        print("     Manual: https://huggingface.co/datasets/lmms-lab/MMBench_EN")


def download_textvqa(data_dir: Path):
    """
    TextVQA validation set — ~5k open-ended questions about text in images.
    Why: Open-ended metric (exact match, relaxed accuracy).
         Proxies the 'Open Question' category in the competition.
    HF: https://huggingface.co/datasets/lmms-lab/textvqa
    """
    print("\n[TextVQA — Open-ended text-in-image QA]")
    try:
        from datasets import load_dataset
        ds = load_dataset("lmms-lab/textvqa", split="validation", cache_dir=str(data_dir / "textvqa"))
        out_path = data_dir / "textvqa" / "textvqa_val.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not out_path.exists():
            img_dir = data_dir / "textvqa" / "images"
            img_dir.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                for i, ex in enumerate(ds):
                    img_file = img_dir / f"{i:06d}.png"
                    if hasattr(ex.get("image", None), "save") and not img_file.exists():
                        ex["image"].save(img_file)
                    record = {
                        "id": i,
                        "image_path": str(img_file),
                        "question": ex.get("question", ""),
                        "answers": ex.get("answers", []),
                    }
                    f.write(json.dumps(record) + "\n")
            print(f"  ✓ TextVQA saved: {len(ds)} examples → {out_path}")
        else:
            print(f"  [skip] Already processed: {out_path}")
    except Exception as e:
        print(f"  ⚠  TextVQA download failed: {e}")


# ── calibration data ──────────────────────────────────────────────────────────

def download_calibration_data(data_dir: Path, n_samples: int = 512):
    """
    COCO val2017: 5k natural images + captions.
    We sample 512 pairs as the AWQ/GPTQ calibration corpus.

    Why COCO for calibration?
      - Diverse image content (people, animals, objects, scenes)
      - Paired captions provide natural language context
      - Widely used in VLM quantization papers
      - Completely disjoint from our evaluation benchmarks

    Why 512 samples?
      - Standard in AWQ / GPTQ literature
      - Enough for accurate activation statistics
      - Fast enough to complete in ~30 min on a single GPU

    Direct download (no HF account needed):
      images: http://images.cocodataset.org/zips/val2017.zip  (~800 MB)
      captions: http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    """
    print(f"\n[COCO val2017 calibration data — {n_samples} samples]")
    cal_dir = data_dir / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)

    out_path = cal_dir / f"coco_calibration_{n_samples}.jsonl"
    if out_path.exists():
        print(f"  [skip] Already exists: {out_path}")
        return out_path

    # Download annotations first (small file)
    ann_zip = cal_dir / "annotations.zip"
    download_file(
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        ann_zip,
        "COCO annotations",
    )
    ann_dir = cal_dir / "annotations_raw"
    if not ann_dir.exists():
        extract_zip(ann_zip, ann_dir)

    # Load captions
    import json as _json
    ann_file = ann_dir / "annotations" / "captions_val2017.json"
    with open(ann_file) as f:
        coco_ann = _json.load(f)

    # Build image_id → captions map
    img_map = {img["id"]: img["file_name"] for img in coco_ann["images"]}
    from collections import defaultdict
    cap_map = defaultdict(list)
    for ann in coco_ann["annotations"]:
        cap_map[ann["image_id"]].append(ann["caption"])

    # Download images
    img_zip = cal_dir / "val2017.zip"
    download_file(
        "http://images.cocodataset.org/zips/val2017.zip",
        img_zip,
        "COCO val2017 images (~800 MB)",
    )
    img_dir = cal_dir / "val2017"
    if not img_dir.exists():
        extract_zip(img_zip, cal_dir)

    # Sample n_samples unique images
    random.seed(42)
    image_ids = list(img_map.keys())
    sampled_ids = random.sample(image_ids, min(n_samples, len(image_ids)))

    with open(out_path, "w") as f:
        for img_id in sampled_ids:
            fname = img_map[img_id]
            img_path = cal_dir / "val2017" / fname
            captions = cap_map[img_id]
            if not img_path.exists():
                continue
            record = {
                "image_path": str(img_path),
                "caption": captions[0] if captions else "",
                "all_captions": captions,
            }
            f.write(_json.dumps(record) + "\n")

    print(f"  ✓ Calibration corpus saved: {n_samples} samples → {out_path}")
    return out_path


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download eval + calibration datasets")
    parser.add_argument("--data_dir", default="../data", help="Root directory for datasets")
    parser.add_argument("--calibration_only", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--n_calib", type=int, default=512, help="Calibration samples")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nData root: {data_dir.resolve()}")

    if not args.calibration_only:
        download_ai2d(data_dir)
        download_mmbench(data_dir)
        download_textvqa(data_dir)

    if not args.eval_only:
        download_calibration_data(data_dir, n_samples=args.n_calib)

    print("\n✓ All datasets ready.\n")
    print("Dataset summary:")
    for p in sorted(data_dir.rglob("*.jsonl")):
        n = sum(1 for _ in open(p))
        print(f"  {p.relative_to(data_dir)}  ({n} examples)")


if __name__ == "__main__":
    main()
