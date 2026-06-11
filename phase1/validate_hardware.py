#!/usr/bin/env python3
"""
Phase 1 — Hardware Validation
Checks GPU, VRAM, CUDA, vLLM, llama-cpp compatibility.

Why this matters: The competition runs on NVIDIA L4 (Ada Lovelace, 16 GB).
Your dev machine (GTX 1080 Ti, Pascal, 11 GB) does NOT support:
  - BF16 natively
  - FP8 (Ada+ feature)
  - Flash Attention 2 (requires Ampere+)

This script prints exactly what each GPU supports so you know which
quantization options are testable locally vs only on L4.

Usage:
    python validate_hardware.py
"""

import sys
import json
import subprocess
from pathlib import Path

# ── graceful imports ──────────────────────────────────────────────────────────
def _try_import(module, pkg_hint=None):
    try:
        return __import__(module)
    except ImportError:
        hint = pkg_hint or module
        print(f"  [MISSING] {module} — install with: pip install {hint}")
        return None


print("\n" + "=" * 64)
print("  Phase 1 — Hardware Validation")
print("=" * 64)

# ── PyTorch / CUDA ────────────────────────────────────────────────────────────
torch = _try_import("torch")
if torch:
    print(f"\n[PyTorch]")
    print(f"  Version        : {torch.__version__}")
    print(f"  CUDA available : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version   : {torch.version.cuda}")
        try:
            import torch.backends.cudnn as cudnn
            print(f"  cuDNN version  : {cudnn.version()}")
        except Exception:
            pass
        print(f"  BF16 supported : {torch.cuda.is_bf16_supported()}")
        print(f"  Device count   : {torch.cuda.device_count()}")

# ── NVML / per-GPU details ────────────────────────────────────────────────────
pynvml = _try_import("pynvml", "nvidia-ml-py3")
if pynvml:
    try:
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        print(f"\n[NVML — {count} GPU(s)]")
        warnings = []
        for i in range(count):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name  = pynvml.nvmlDeviceGetName(h)
            mem   = pynvml.nvmlDeviceGetMemoryInfo(h)
            cc    = pynvml.nvmlDeviceGetCudaComputeCapability(h)
            try:
                power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(h) / 1000
            except Exception:
                power_limit = "N/A"

            vram_gb = mem.total / 1e9
            free_gb = mem.free  / 1e9
            cc_str  = f"{cc[0]}.{cc[1]}"

            # feature matrix
            supports_bf16    = cc[0] >= 8                   # Ampere+
            supports_fp8     = cc[0] >= 9                   # Ada/Hopper+
            supports_fa2     = cc[0] >= 8                   # Ampere+
            vram_ok_for_l4   = vram_gb >= 14.0              # L4 has 24 GB actually
            # NOTE: L4 has 24 GB (datasheet); competition slide says 16 GB
            # Treat 16 GB as minimum

            print(f"\n  GPU {i}: {name}")
            print(f"    VRAM            : {vram_gb:.1f} GB total / {free_gb:.1f} GB free")
            print(f"    Compute cap.    : {cc_str}")
            print(f"    Power limit     : {power_limit} W")
            print(f"    BF16 native     : {'✓' if supports_bf16 else '✗ (Pascal — use FP16 instead)'}")
            print(f"    FP8 native      : {'✓' if supports_fp8  else '✗ (needs Ada/Hopper)'}")
            print(f"    Flash Attn 2    : {'✓' if supports_fa2  else '✗ (needs Ampere+)'}")
            print(f"    VRAM ≥ 16 GB    : {'✓' if vram_ok_for_l4 else f'⚠  {vram_gb:.1f} GB < 16 GB (test only)'}")

            if not supports_bf16:
                warnings.append(
                    f"GPU {i} ({name}) is Pascal. AWQ/GPTQ will run in FP16, "
                    "not BF16. Results may differ from L4."
                )
            if vram_gb < 16:
                warnings.append(
                    f"GPU {i} ({name}) has {vram_gb:.1f} GB VRAM. "
                    "Full precision baseline may OOM — use --load-in-8bit for dev."
                )

        if warnings:
            print("\n  [WARNINGS]")
            for w in warnings:
                print(f"  ⚠  {w}")

        pynvml.nvmlShutdown()
    except Exception as e:
        print(f"  NVML error: {e}")

# ── System RAM / CPU ──────────────────────────────────────────────────────────
psutil = _try_import("psutil")
if psutil:
    mem  = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    print(f"\n[System]")
    print(f"  RAM total  : {mem.total / 1e9:.1f} GB")
    print(f"  RAM free   : {mem.available / 1e9:.1f} GB")
    print(f"  CPU cores  : {psutil.cpu_count(logical=False)} physical / {psutil.cpu_count()} logical")
    print(f"  Disk free  : {disk.free / 1e9:.1f} GB  (need ~30 GB for model + datasets)")
    if disk.free < 30e9:
        print("  ⚠  Low disk space — ensure ≥30 GB free before downloading model.")

# ── Inference engines ─────────────────────────────────────────────────────────
print(f"\n[Inference engines]")
try:
    import vllm
    v = vllm.__version__
    ok = v == "0.17.1"
    print(f"  vLLM        : {v}  {'✓' if ok else '⚠  competition uses 0.17.1'}")
except ImportError:
    print("  vLLM        : NOT INSTALLED — pip install vllm==0.17.1")

try:
    from llama_cpp import Llama  # noqa: F401
    print("  llama.cpp   : ✓ installed")
    # check GPU layers support
    import llama_cpp
    if hasattr(llama_cpp, "__cuda__"):
        print(f"  llama.cpp CUDA build: ✓")
    else:
        print("  llama.cpp CUDA build: unknown — ensure CMAKE_ARGS='-DGGML_CUDA=on'")
except ImportError:
    print("  llama.cpp   : NOT INSTALLED")

# ── Quantization toolkits ─────────────────────────────────────────────────────
print(f"\n[Quantization toolkits]")
for pkg, display in [("awq", "AutoAWQ"), ("auto_gptq", "AutoGPTQ")]:
    try:
        mod = __import__(pkg)
        v   = getattr(mod, "__version__", "installed")
        print(f"  {display:12s}: ✓  {v}")
    except ImportError:
        print(f"  {display:12s}: NOT INSTALLED")

# ── Energy tracking ───────────────────────────────────────────────────────────
print(f"\n[Energy tracking]")
try:
    import codecarbon
    print(f"  CodeCarbon  : ✓  {codecarbon.__version__}")
except ImportError:
    print("  CodeCarbon  : NOT INSTALLED — pip install codecarbon")

# ── transformers ─────────────────────────────────────────────────────────────
print(f"\n[Transformers]")
try:
    import transformers
    print(f"  transformers: ✓  {transformers.__version__}")
    # Check Gemma 4 support
    try:
        from transformers import AutoProcessor  # noqa: F401
        print("  AutoProcessor: ✓")
    except Exception:
        pass
except ImportError:
    print("  transformers: NOT INSTALLED")

print("\n" + "=" * 64)
print("  Validation complete. Resolve any ✗ / ⚠  before proceeding.")
print("=" * 64 + "\n")
