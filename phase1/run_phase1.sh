#!/usr/bin/env bash
# =============================================================================
# Phase 1 — Master Orchestration Script
# Runs all Phase 1 steps in order.
#
# Usage:
#   bash run_phase1.sh --token hf_xxxxxxxx
#   bash run_phase1.sh --token hf_xxxxxxxx --max_samples 50  # fast sanity check
#   bash run_phase1.sh --token hf_xxxxxxxx --skip_download    # if model already exists
#
# On dev machine (GTX 1080 Ti, 11 GB), add: --dev_mode
# This forces --load_8bit on evaluation to fit in VRAM.
# =============================================================================
set -euo pipefail

# ── parse arguments ───────────────────────────────────────────────────────────
TOKEN=""
MAX_SAMPLES=""
SKIP_DOWNLOAD=false
DEV_MODE=false
MODEL_DIR="../models/gemma-4-E4B-it"
DATA_DIR="../data"
OUTPUT_DIR="../results/baseline"

while [[ $# -gt 0 ]]; do
    case $1 in
        --token)         TOKEN="$2";          shift 2 ;;
        --max_samples)   MAX_SAMPLES="$2";    shift 2 ;;
        --skip_download) SKIP_DOWNLOAD=true;  shift ;;
        --dev_mode)      DEV_MODE=true;       shift ;;
        --model_dir)     MODEL_DIR="$2";      shift 2 ;;
        --data_dir)      DATA_DIR="$2";       shift 2 ;;
        --output_dir)    OUTPUT_DIR="$2";     shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "======================================================"
echo "  Phase 1 — Full Pipeline"
echo "======================================================"
echo "  Token        : ${TOKEN:0:10}... (masked)"
echo "  Model dir    : $MODEL_DIR"
echo "  Data dir     : $DATA_DIR"
echo "  Output dir   : $OUTPUT_DIR"
echo "  Max samples  : ${MAX_SAMPLES:-full}"
echo "  Dev mode     : $DEV_MODE"
echo "======================================================"

# ── step 1: validate hardware ─────────────────────────────────────────────────
echo ""
echo "Step 1/5: Hardware validation"
python validate_hardware.py
echo "✓ Hardware validated"

# ── step 2: download model ────────────────────────────────────────────────────
if [ "$SKIP_DOWNLOAD" = false ]; then
    echo ""
    echo "Step 2/5: Model download"
    if [ -z "$TOKEN" ]; then
        echo "ERROR: --token required for model download"
        echo "  Get token: https://huggingface.co/settings/tokens"
        echo "  Accept license: https://huggingface.co/google/gemma-4-E4B-it"
        exit 1
    fi
    python download_model.py \
        --model_id  google/gemma-4-E4B-it \
        --local_dir "$MODEL_DIR" \
        --token     "$TOKEN"
    echo "✓ Model downloaded"
else
    echo ""
    echo "Step 2/5: Skipping download (--skip_download set)"
fi

# ── step 3: download datasets ─────────────────────────────────────────────────
echo ""
echo "Step 3/5: Dataset download"
python download_datasets.py \
    --data_dir "$DATA_DIR" \
    --n_calib  512
echo "✓ Datasets ready"

# ── step 4: memory profiling ──────────────────────────────────────────────────
echo ""
echo "Step 4/5: Memory profiling"
python profile_memory.py \
    --model_dir  "$MODEL_DIR" \
    --data_dir   "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR"
echo "✓ Memory profile saved"

# ── step 5: baseline evaluation ───────────────────────────────────────────────
echo ""
echo "Step 5/5: Baseline evaluation (this is the longest step)"

EVAL_ARGS="--model_dir $MODEL_DIR --data_dir $DATA_DIR --output_dir $OUTPUT_DIR"

if [ -n "$MAX_SAMPLES" ]; then
    EVAL_ARGS="$EVAL_ARGS --max_samples $MAX_SAMPLES"
fi

if [ "$DEV_MODE" = true ]; then
    echo "  [DEV MODE] Using --load_8bit to fit in 11 GB VRAM"
    EVAL_ARGS="$EVAL_ARGS --load_8bit"
fi

# shellcheck disable=SC2086
python baseline_eval.py $EVAL_ARGS

echo ""
echo "======================================================"
echo "  Phase 1 COMPLETE"
echo "======================================================"
echo ""
echo "  Results:    $OUTPUT_DIR/baseline_summary.json"
echo "  Memory:     $OUTPUT_DIR/memory_profile.json"
echo ""
echo "  Next steps:"
echo "    1. Review $OUTPUT_DIR/baseline_summary.json"
echo "    2. Note the 80% accuracy thresholds"
echo "    3. Note total energy_kwh — Phase 2 target: reduce by ≥40%"
echo "    4. Approve Phase 2 to begin quantization"
echo ""
