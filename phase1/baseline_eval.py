#!/usr/bin/env python3
"""
Phase 1 — Baseline Evaluation with Energy Tracking

Runs google/gemma-4-E4B-it on three benchmarks, measuring:
  - Accuracy (multiple choice: exact match, open-ended: relaxed accuracy)
  - Energy consumption: kWh via CodeCarbon
  - GPU power: average W via NVML
  - Peak VRAM: GB
  - Latency: ms per image

This establishes the 100% reference point.
Every Phase 2+ compressed model will be compared against these numbers.
The competition requires staying above 80% of baseline accuracy.

Usage:
    # Full evaluation (all benchmarks, FP16):
    python baseline_eval.py \
        --model_dir ../models/gemma-4-E4B-it \
        --data_dir  ../data \
        --output_dir ../results/baseline \
        --device cuda

    # Fast sanity check (50 samples each):
    python baseline_eval.py \
        --model_dir ../models/gemma-4-E4B-it \
        --data_dir  ../data \
        --output_dir ../results/baseline \
        --max_samples 50

    # Dev machine (1080 Ti, 11 GB): add --load_8bit to fit in VRAM
    python baseline_eval.py \
        --model_dir ../models/gemma-4-E4B-it \
        --data_dir  ../data \
        --output_dir ../results/baseline \
        --load_8bit
"""

import os
import sys
import json
import time
import argparse
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

import torch
from PIL import Image
from tqdm import tqdm


# ── result container ──────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    benchmark: str
    n_samples: int
    accuracy: float           # 0.0–1.0
    avg_latency_ms: float
    peak_vram_gb: float
    energy_kwh: float
    avg_gpu_power_w: float
    co2_kg: float
    throughput_samples_per_sec: float
    errors: int = 0


# ── GPU power monitor ─────────────────────────────────────────────────────────

class GPUPowerMonitor:
    """
    Polls GPU power via NVML every 100ms in a background thread.
    Provides average power over the measurement window.
    This mirrors the competition's use of NVML for energy accounting.
    """
    def __init__(self, device_index: int = 0, poll_interval_s: float = 0.1):
        import pynvml
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        self.poll_interval = poll_interval_s
        self._readings: List[float] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._pynvml = pynvml

    def start(self):
        self._readings = []
        self._running = True
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self):
        while self._running:
            try:
                mw = self._pynvml.nvmlDeviceGetPowerUsage(self.handle)
                self._readings.append(mw / 1000.0)  # mW → W
            except Exception:
                pass
            time.sleep(self.poll_interval)

    def stop(self) -> float:
        """Stop polling and return average power in Watts."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._readings:
            return sum(self._readings) / len(self._readings)
        return 0.0


def get_peak_vram_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e9
    return 0.0


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(model_dir: str, load_8bit: bool = False, device: str = "cuda"):
    """
    Loads Gemma 4 E4B-it (vision-language model).
    Uses AutoProcessor + the appropriate AutoModel class.

    For FP16 on L4 (16 GB):  load normally
    For dev on 1080 Ti (11 GB): --load_8bit loads in INT8 via bitsandbytes
      Note: 8-bit loading is NOT the Phase 2 quantization — it's just to
      fit the baseline in 11 GB VRAM. Phase 2 will use AWQ/GPTQ properly.
    """
    from transformers import AutoProcessor, AutoModelForImageTextToText
    import transformers

    print(f"\n[Model loading]")
    print(f"  Path   : {model_dir}")
    print(f"  Device : {device}")
    print(f"  8-bit  : {load_8bit}  (dev convenience only, not the Phase 2 method)")

    torch.cuda.reset_peak_memory_stats()

    load_kwargs: Dict[str, Any] = {
        "torch_dtype": torch.float16,
        "device_map": "auto",
    }

    if load_8bit:
        # bitsandbytes 8-bit: halves memory with ~1% accuracy cost
        # Only for local testing — competition submits AWQ/GPTQ in Phase 2
        load_kwargs["load_in_8bit"] = True
        load_kwargs.pop("torch_dtype", None)

    t0 = time.time()
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)

    # Try specific Gemma4 class first, fall back to AutoModel
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_dir,
            trust_remote_code=True,
            **load_kwargs,
        )
    except Exception:
        # Gemma 4 may use a different class name — let transformers auto-detect
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            model_dir,
            trust_remote_code=True,
            **load_kwargs,
        )

    model.eval()
    load_time = time.time() - t0
    vram_after_load = get_peak_vram_gb()

    print(f"  Load time   : {load_time:.1f}s")
    print(f"  VRAM used   : {vram_after_load:.2f} GB")
    print(f"  Parameters  : {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    return model, processor


# ── inference ─────────────────────────────────────────────────────────────────

def run_inference(
    model,
    processor,
    image_path: str,
    prompt: str,
    max_new_tokens: int = 50,
    device: str = "cuda",
) -> str:
    """
    Run a single image+text → text inference.
    Returns the generated text (stripped of the prompt / special tokens).
    """
    image = Image.open(image_path).convert("RGB")

    # Build the conversation format Gemma 4 uses
    # Gemma 4 uses a chat template with <image> token
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": prompt},
            ],
        }
    ]

    # Processor handles both image encoding and text tokenization
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,         # greedy — deterministic
            temperature=1.0,
            top_p=0.95,
            top_k=64,                # Google's default params (from slide 11)
        )

    # Decode only the newly generated tokens
    n_input = inputs["input_ids"].shape[-1]
    generated = output_ids[0, n_input:]
    text = processor.decode(generated, skip_special_tokens=True).strip()
    return text


# ── per-benchmark evaluators ──────────────────────────────────────────────────

CHOICE_MAP = {"A": 0, "B": 1, "C": 2, "D": 3}


def build_mc_prompt(question: str, choices: List[str]) -> str:
    """Format a multiple-choice question in a way Gemma understands."""
    choice_str = "\n".join(
        f"{chr(65+i)}. {c}" for i, c in enumerate(choices)
    )
    return (
        f"Answer the following question by selecting one option (A, B, C, or D).\n\n"
        f"Question: {question}\n\n"
        f"{choice_str}\n\n"
        f"Answer with only the letter of the correct choice."
    )


def extract_mc_answer(response: str) -> str:
    """Extract A/B/C/D from model response."""
    response = response.strip().upper()
    for ch in ["A", "B", "C", "D"]:
        if response.startswith(ch):
            return ch
    # Try to find any letter in the response
    for ch in ["A", "B", "C", "D"]:
        if ch in response:
            return ch
    return "X"  # no valid answer found


def relaxed_accuracy(prediction: str, ground_truths: List[str]) -> float:
    """
    VQA-style relaxed accuracy: prediction is correct if it matches
    any of the ground truth answers (case-insensitive, stripped).
    Returns 1.0 if correct, 0.0 otherwise.
    """
    pred = prediction.lower().strip()
    for gt in ground_truths:
        if pred == gt.lower().strip():
            return 1.0
        # partial match: gt appears in pred
        if gt.lower().strip() in pred:
            return 1.0
    return 0.0


def evaluate_benchmark(
    model,
    processor,
    jsonl_path: Path,
    benchmark_name: str,
    benchmark_type: str,   # "multiple_choice" or "open_ended"
    output_dir: Path,
    max_samples: Optional[int] = None,
    device: str = "cuda",
    gpu_monitor: Optional[GPUPowerMonitor] = None,
) -> BenchmarkResult:
    """
    Runs evaluation on one benchmark JSONL file.
    Tracks energy via CodeCarbon and power via NVML.
    """
    from codecarbon import EmissionsTracker

    print(f"\n{'─'*60}")
    print(f"  Benchmark : {benchmark_name} ({benchmark_type})")
    print(f"  Data      : {jsonl_path}")

    # Load examples
    examples = []
    with open(jsonl_path) as f:
        for line in f:
            examples.append(json.loads(line))

    if max_samples:
        examples = examples[:max_samples]
    print(f"  Samples   : {len(examples)}")

    # Output file for per-example predictions
    pred_file = output_dir / f"{benchmark_name}_predictions.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Energy tracking
    tracker = EmissionsTracker(
        project_name=f"gemma4_baseline_{benchmark_name}",
        output_dir=str(output_dir),
        output_file=f"{benchmark_name}_emissions.csv",
        log_level="error",
        tracking_mode="process",   # track this process
        gpu_ids=[0],
        save_to_file=True,
        measure_power_secs=15,
    )

    # Reset VRAM stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    correct = 0
    total   = 0
    errors  = 0
    latencies_ms = []
    predictions = []

    # Start energy + power tracking
    tracker.start()
    if gpu_monitor:
        gpu_monitor.start()

    for ex in tqdm(examples, desc=benchmark_name, ncols=80):
        img_path = ex["image_path"]
        if not Path(img_path).exists():
            errors += 1
            continue

        try:
            if benchmark_type == "multiple_choice":
                prompt = build_mc_prompt(ex["question"], ex["choices"])
                t0 = time.perf_counter()
                response = run_inference(model, processor, img_path, prompt, max_new_tokens=10, device=device)
                latency_ms = (time.perf_counter() - t0) * 1000
                pred_letter = extract_mc_answer(response)
                is_correct  = (pred_letter == ex["answer"].strip().upper())
                correct += int(is_correct)
                predictions.append({
                    "id": ex["id"],
                    "question": ex["question"],
                    "answer": ex["answer"],
                    "prediction": pred_letter,
                    "raw_response": response,
                    "correct": is_correct,
                })
            else:  # open_ended
                prompt = f"Question: {ex['question']}\nAnswer briefly:"
                t0 = time.perf_counter()
                response = run_inference(model, processor, img_path, prompt, max_new_tokens=30, device=device)
                latency_ms = (time.perf_counter() - t0) * 1000
                score = relaxed_accuracy(response, ex.get("answers", []))
                correct += score
                predictions.append({
                    "id": ex["id"],
                    "question": ex["question"],
                    "answers": ex.get("answers", []),
                    "prediction": response,
                    "score": score,
                })

            latencies_ms.append(latency_ms)
            total += 1

        except Exception as e:
            errors += 1
            tqdm.write(f"  Error on example {ex.get('id', '?')}: {e}")

    # Stop trackers
    emissions_data = tracker.stop()
    avg_power_w = gpu_monitor.stop() if gpu_monitor else 0.0

    # Compute metrics
    accuracy       = correct / total if total > 0 else 0.0
    avg_latency_ms = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
    peak_vram_gb   = get_peak_vram_gb()
    energy_kwh     = emissions_data if isinstance(emissions_data, float) else 0.0
    co2_kg         = 0.0  # CodeCarbon also tracks this

    # Try to get detailed emissions
    try:
        import pandas as pd
        csv_path = output_dir / f"{benchmark_name}_emissions.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if not df.empty:
                row = df.iloc[-1]
                energy_kwh = float(row.get("energy_consumed", energy_kwh))
                co2_kg     = float(row.get("emissions", 0.0))
    except Exception:
        pass

    total_time_s = sum(latencies_ms) / 1000
    throughput   = total / total_time_s if total_time_s > 0 else 0.0

    result = BenchmarkResult(
        benchmark=benchmark_name,
        n_samples=total,
        accuracy=accuracy,
        avg_latency_ms=avg_latency_ms,
        peak_vram_gb=peak_vram_gb,
        energy_kwh=energy_kwh,
        avg_gpu_power_w=avg_power_w,
        co2_kg=co2_kg,
        throughput_samples_per_sec=throughput,
        errors=errors,
    )

    # Save predictions
    with open(pred_file, "w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")

    # Print result
    print(f"\n  Results:")
    print(f"    Accuracy          : {accuracy:.4f}  ({correct:.1f}/{total})")
    print(f"    Avg latency       : {avg_latency_ms:.0f} ms/sample")
    print(f"    Throughput        : {throughput:.2f} samples/s")
    print(f"    Peak VRAM         : {peak_vram_gb:.2f} GB")
    print(f"    Energy consumed   : {energy_kwh:.6f} kWh")
    print(f"    Avg GPU power     : {avg_power_w:.1f} W")
    print(f"    CO2 equivalent    : {co2_kg*1000:.2f} g")
    if errors:
        print(f"    Errors (skipped)  : {errors}")

    return result


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Baseline evaluation for google/gemma-4-E4B-it")
    p.add_argument("--model_dir",   required=True, help="Path to downloaded model")
    p.add_argument("--data_dir",    required=True, help="Path to datasets (from download_datasets.py)")
    p.add_argument("--output_dir",  default="../results/baseline")
    p.add_argument("--device",      default="cuda")
    p.add_argument("--max_samples", type=int, default=None, help="Limit per benchmark (None = full)")
    p.add_argument("--load_8bit",   action="store_true", help="Load in INT8 (dev only, for 11GB VRAM)")
    p.add_argument("--benchmarks",  nargs="+",
                   default=["ai2d", "mmbench", "textvqa"],
                   choices=["ai2d", "mmbench", "textvqa"],
                   help="Which benchmarks to run")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir   = Path(args.data_dir)

    print("\n" + "=" * 60)
    print("  Phase 1 — Baseline Evaluation")
    print("=" * 60)
    print(f"  Model dir  : {args.model_dir}")
    print(f"  Data dir   : {data_dir}")
    print(f"  Output dir : {output_dir}")
    print(f"  Max samples: {args.max_samples or 'full'}")
    print(f"  Benchmarks : {args.benchmarks}")

    # Load model once, reuse across benchmarks
    model, processor = load_model(args.model_dir, load_8bit=args.load_8bit, device=args.device)

    # GPU power monitor (shared across all benchmarks)
    gpu_monitor = None
    try:
        gpu_monitor = GPUPowerMonitor(device_index=0)
    except Exception as e:
        print(f"  ⚠  NVML power monitoring unavailable: {e}")

    # Benchmark registry
    BENCHMARKS = {
        "ai2d": {
            "path": data_dir / "ai2d"    / "ai2d_test.jsonl",
            "type": "multiple_choice",
        },
        "mmbench": {
            "path": data_dir / "mmbench" / "mmbench_test.jsonl",
            "type": "multiple_choice",
        },
        "textvqa": {
            "path": data_dir / "textvqa" / "textvqa_val.jsonl",
            "type": "open_ended",
        },
    }

    all_results = []
    for bname in args.benchmarks:
        info = BENCHMARKS[bname]
        if not info["path"].exists():
            print(f"\n⚠  {bname} data not found at {info['path']}")
            print(f"   Run:  python download_datasets.py --data_dir {data_dir}")
            continue

        result = evaluate_benchmark(
            model=model,
            processor=processor,
            jsonl_path=info["path"],
            benchmark_name=bname,
            benchmark_type=info["type"],
            output_dir=output_dir,
            max_samples=args.max_samples,
            device=args.device,
            gpu_monitor=gpu_monitor,
        )
        all_results.append(asdict(result))

    # ── save summary ──────────────────────────────────────────────────────────
    summary = {
        "model": "google/gemma-4-E4B-it",
        "precision": "int8_bnb" if args.load_8bit else "fp16",
        "hardware": "CUDA GPU",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": all_results,
        "aggregate": {},
    }

    if all_results:
        mc  = [r for r in all_results if r["benchmark"] in ("ai2d", "mmbench")]
        oe  = [r for r in all_results if r["benchmark"] == "textvqa"]
        mc_acc = sum(r["accuracy"] for r in mc) / len(mc)  if mc else None
        oe_acc = sum(r["accuracy"] for r in oe) / len(oe)  if oe else None
        total_energy = sum(r["energy_kwh"]  for r in all_results)
        avg_power    = sum(r["avg_gpu_power_w"] for r in all_results) / len(all_results)

        summary["aggregate"] = {
            "mc_accuracy_avg":  mc_acc,
            "oe_accuracy_avg":  oe_acc,
            "total_energy_kwh": total_energy,
            "avg_gpu_power_w":  avg_power,
            "threshold_80pct":  {
                "mc_must_exceed":  mc_acc * 0.80 if mc_acc else None,
                "oe_must_exceed":  oe_acc * 0.80 if oe_acc else None,
            },
        }

    summary_path = output_dir / "baseline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ── final printout ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  BASELINE SUMMARY")
    print("=" * 60)
    for r in all_results:
        print(f"  {r['benchmark']:12s}  accuracy={r['accuracy']:.4f}  "
              f"energy={r['energy_kwh']:.6f} kWh  "
              f"power={r['avg_gpu_power_w']:.0f}W  "
              f"latency={r['avg_latency_ms']:.0f}ms")

    if summary["aggregate"]:
        ag = summary["aggregate"]
        print(f"\n  MC accuracy avg  : {ag.get('mc_accuracy_avg', 'N/A')}")
        print(f"  OE accuracy avg  : {ag.get('oe_accuracy_avg', 'N/A')}")
        print(f"  Total energy     : {ag['total_energy_kwh']:.6f} kWh")
        print(f"  80% threshold    : MC≥{ag['threshold_80pct']['mc_must_exceed']:.4f}  "
              f"OE≥{ag['threshold_80pct']['oe_must_exceed']:.4f}")

    print(f"\n  Full results saved: {summary_path}")
    print("=" * 60 + "\n")
    print("  ✓ Phase 1 complete. Review baseline_summary.json then")
    print("    approve Phase 2 (quantization) to proceed.\n")


if __name__ == "__main__":
    main()
