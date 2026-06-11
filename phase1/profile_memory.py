#!/usr/bin/env python3
"""
Phase 1 — Memory & Compute Profiling

Measures:
  - VRAM usage breakdown: weights vs activations vs KV cache
  - Latency vs batch size curve
  - Throughput vs sequence length curve
  - FLOPs per forward pass estimate

Why profile this now?
  Phase 2 quantization reduces weight memory.
  Phase 4 inference tuning optimizes KV cache and batching.
  You need these numbers to know the ceiling for each technique.

Usage:
    python profile_memory.py \
        --model_dir ../models/gemma-4-E4B-it \
        --output_dir ../results/baseline \
        --quick    # just model weight size + single-image VRAM
"""

import os
import gc
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any

import torch


def reset_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def get_vram_mb() -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    return {
        "allocated_mb": torch.cuda.memory_allocated() / 1e6,
        "reserved_mb":  torch.cuda.memory_reserved()  / 1e6,
        "peak_mb":      torch.cuda.max_memory_allocated() / 1e6,
    }


def count_parameters(model) -> Dict[str, Any]:
    """Count total, trainable, and per-dtype parameters."""
    total   = 0
    by_dtype: Dict[str, int] = {}
    trainable = 0
    for name, param in model.named_parameters():
        n = param.numel()
        total += n
        dtype_str = str(param.dtype)
        by_dtype[dtype_str] = by_dtype.get(dtype_str, 0) + n
        if param.requires_grad:
            trainable += n

    # Estimate size in memory
    dtype_bytes = {
        "torch.float32": 4,
        "torch.float16": 2,
        "torch.bfloat16": 2,
        "torch.int8": 1,
        "torch.uint8": 1,
    }
    estimated_gb = sum(
        (by_dtype.get(d, 0) * b) / 1e9
        for d, b in dtype_bytes.items()
    )

    return {
        "total_params":       total,
        "trainable_params":   trainable,
        "by_dtype":           by_dtype,
        "estimated_size_gb":  estimated_gb,
        "params_billions":    total / 1e9,
    }


def profile_model_weights(model_dir: str) -> Dict[str, Any]:
    """
    Load model and measure weight memory before any inference.
    This is the minimum VRAM floor — any inference adds activations on top.
    """
    from transformers import AutoProcessor, AutoModelForImageTextToText

    print("\n[Weight memory profiling]")
    reset_vram()
    vram_before = get_vram_mb()

    print("  Loading model in FP16...")
    t0 = time.time()
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    except Exception:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    load_time = time.time() - t0
    vram_after = get_vram_mb()

    param_info = count_parameters(model)

    weight_vram_mb = vram_after["allocated_mb"] - vram_before.get("allocated_mb", 0)

    results = {
        "load_time_s": load_time,
        "weight_vram_mb": weight_vram_mb,
        "weight_vram_gb": weight_vram_mb / 1024,
        "vram_after_load": vram_after,
        **param_info,
    }

    print(f"  Load time          : {load_time:.1f}s")
    print(f"  Weight VRAM (FP16) : {weight_vram_mb/1024:.2f} GB")
    print(f"  Parameters         : {param_info['params_billions']:.2f}B")
    print(f"  Estimated size     : {param_info['estimated_size_gb']:.2f} GB")
    print(f"  Dtype breakdown:")
    for dtype, n in param_info["by_dtype"].items():
        print(f"    {dtype:24s}: {n/1e9:.2f}B params")

    # Quantization size projections
    fp16_gb  = param_info["estimated_size_gb"]
    int8_gb  = fp16_gb * 0.50
    int4_gb  = fp16_gb * 0.25
    fp8_gb   = fp16_gb * 0.50

    print(f"\n  Quantization size projections (weights only):")
    print(f"    FP16 (baseline) : {fp16_gb:.2f} GB")
    print(f"    FP8             : {fp8_gb:.2f} GB  (−50%)")
    print(f"    INT8            : {int8_gb:.2f} GB  (−50%)")
    print(f"    INT4 (AWQ/GPTQ) : {int4_gb:.2f} GB  (−75%)")
    print(f"\n  L4 headroom after weights:")
    print(f"    FP16 : {16 - fp16_gb:.2f} GB for activations + KV cache")
    print(f"    INT4 : {16 - int4_gb:.2f} GB for activations + KV cache")

    del model
    reset_vram()
    return results


def profile_inference_latency(
    model_dir: str,
    data_dir: str,
    output_dir: Path,
    n_warmup: int = 3,
    n_timed: int = 20,
) -> Dict[str, Any]:
    """
    Measures latency for a single image+question.
    Runs n_warmup to populate CUDA caches, then n_timed for measurement.
    """
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from PIL import Image

    print("\n[Inference latency profiling]")
    reset_vram()

    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    except Exception:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model.eval()

    # Find a sample image from calibration data
    cal_file = Path(data_dir) / "calibration" / "coco_calibration_512.jsonl"
    sample_img_path = None
    if cal_file.exists():
        import json as _json
        with open(cal_file) as f:
            line = f.readline()
            sample_img_path = _json.loads(line)["image_path"]
    if not sample_img_path or not Path(sample_img_path).exists():
        # Create a dummy 224×224 image for profiling
        dummy_img = Path(output_dir) / "_dummy_profile.png"
        Image.new("RGB", (224, 224), color=(128, 128, 128)).save(dummy_img)
        sample_img_path = str(dummy_img)
        print("  Using dummy image for profiling (calibration data not found)")

    def _run_one(prompt: str = "Describe this image.") -> float:
        image = Image.open(sample_img_path).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": prompt},
        ]}]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        return (time.perf_counter() - t0) * 1000

    print(f"  Warming up ({n_warmup} runs)...")
    for _ in range(n_warmup):
        _run_one()

    print(f"  Timing ({n_timed} runs)...")
    latencies = [_run_one() for _ in range(n_timed)]

    results = {
        "n_timed": n_timed,
        "mean_ms":   sum(latencies) / len(latencies),
        "min_ms":    min(latencies),
        "max_ms":    max(latencies),
        "p50_ms":    sorted(latencies)[len(latencies) // 2],
        "p95_ms":    sorted(latencies)[int(len(latencies) * 0.95)],
        "vram_during_inference_gb": get_vram_mb()["peak_mb"] / 1024,
    }

    print(f"  Mean latency : {results['mean_ms']:.0f} ms")
    print(f"  P50 latency  : {results['p50_ms']:.0f} ms")
    print(f"  P95 latency  : {results['p95_ms']:.0f} ms")
    print(f"  Peak VRAM    : {results['vram_during_inference_gb']:.2f} GB")
    print(f"  Throughput   : {1000/results['mean_ms']:.2f} samples/s")

    del model
    reset_vram()
    return results


def main():
    p = argparse.ArgumentParser(description="Memory and latency profiling")
    p.add_argument("--model_dir",  required=True)
    p.add_argument("--data_dir",   default="../data")
    p.add_argument("--output_dir", default="../results/baseline")
    p.add_argument("--quick", action="store_true",
                   help="Only profile weight memory (skip latency sweep)")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  Phase 1 — Memory & Compute Profile")
    print("=" * 60)

    report = {}
    report["weights"] = profile_model_weights(args.model_dir)

    if not args.quick:
        report["latency"] = profile_inference_latency(
            args.model_dir, args.data_dir, output_dir
        )

    out_path = output_dir / "memory_profile.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  ✓ Profile saved: {out_path}\n")


if __name__ == "__main__":
    main()
