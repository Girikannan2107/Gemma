# Gemma 4 E4B-it — Compression Pipeline
## Resilient AI Challenge — Image-to-Text Category

**Model**: `google/gemma-4-E4B-it`
**Target hardware**: NVIDIA L4 (16 GB VRAM)
**Inference engines**: vLLM 0.17.1 · llama.cpp
**Accuracy floor**: ≥ 80% of baseline
**Energy metric**: CodeCarbon (CodeCarbon + NVML)

---

## Project Layout

```
gemma_compress/
├── phase1/
│   ├── setup_env.sh          # Conda + pip environment setup
│   ├── validate_hardware.py  # GPU/CUDA/vLLM validation
│   ├── download_model.py     # Download google/gemma-4-E4B-it
│   ├── download_datasets.py  # Download eval + calibration data
│   ├── baseline_eval.py      # Full evaluation with energy tracking
│   ├── profile_memory.py     # VRAM + RAM profiling
│   └── run_phase1.sh         # Orchestrates all of the above
├── configs/
│   └── eval_config.yaml      # Evaluation settings
└── results/
    └── baseline/             # JSON results written here
```

## Quick Start

```bash
cd phase1
bash setup_env.sh
conda activate gemma_compress
bash run_phase1.sh --token YOUR_HF_TOKEN
```

## Phase Summary

| Phase | Technique | Est. Energy Reduction | Est. Accuracy Impact |
|-------|-----------|----------------------|----------------------|
| 1 | Baseline | 0% | 0% (reference) |
| 2 | AWQ INT4 / GGUF Q4_K_M | 40–55% | 2–5% |
| 3 | Structured pruning (conditional) | 10–15% additional | 3–8% |
| 4 | Inference config tuning | 5–10% additional | 0% |
| **Total** | **Combined** | **55–70%** | **within 80% floor** |
