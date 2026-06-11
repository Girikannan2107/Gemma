#!/usr/bin/env bash
# =============================================================================
# Phase 1 — Environment Setup
# Tested on: Ubuntu 22.04, CUDA 12.x, Python 3.10
# Dev machine:  GTX 1080 Ti (11 GB) — limited FP8 / BF16 support
# Competition:  NVIDIA L4    (16 GB) — full Ada Lovelace feature set
# =============================================================================
set -euo pipefail

ENV_NAME="gemma_compress"
PYTHON_VERSION="3.10"

echo "==> Creating conda environment: ${ENV_NAME}"
conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
# shellcheck disable=SC1090
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "==> Installing PyTorch (CUDA 12.1 wheel)"
# CUDA 12.1 wheels work on CUDA 12.x hosts
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121

echo "==> Installing HuggingFace stack"
pip install \
    transformers>=4.47.0 \
    accelerate>=0.34.0 \
    huggingface_hub>=0.24.0 \
    datasets>=2.20.0 \
    safetensors \
    sentencepiece \
    tokenizers

echo "==> Installing evaluation libraries"
pip install \
    lm-eval==0.4.4 \
    rouge-score \
    sacrebleu \
    nltk \
    Pillow \
    opencv-python-headless

echo "==> Installing energy tracking"
pip install \
    codecarbon==2.5.0 \
    nvidia-ml-py3==7.352.0      # pynvml

echo "==> Installing inference engines"
# vLLM — pin to competition version
pip install vllm==0.17.1

# llama-cpp-python — CUDA build (enables GPU acceleration)
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

echo "==> Installing quantization toolkits (needed for Phase 2, install now)"
pip install autoawq          # AWQ INT4
pip install auto-gptq        # GPTQ INT4

echo "==> Installing utilities"
pip install \
    rich \
    tqdm \
    pyyaml \
    psutil \
    pandas \
    matplotlib \
    seaborn \
    wandb          # optional: experiment tracking

echo ""
echo "========================================"
echo "Environment setup complete."
echo "Activate with: conda activate ${ENV_NAME}"
echo "Then run:       python validate_hardware.py"
echo "========================================"
