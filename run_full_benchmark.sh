#!/bin/bash
# =============================================================================
# GreenVLM — Full Benchmark Runner (resumable)
#
# Runs every model over all tasks and resolutions. Each model writes its own
# checkpoint (results/full/<model>/checkpoint.jsonl); every completed inference
# is persisted immediately. If the run is interrupted (Ctrl-C, crash, reboot),
# just re-run this script — it skips finished work and resumes where it stopped.
#
# Usage:
#   bash run_full_benchmark.sh                 # run all models
#   bash run_full_benchmark.sh qwen3-vl:8b     # run / resume a single model
# =============================================================================
set -u
cd "$(dirname "$0")"

# Use the project virtualenv python (has torch/transformers for the HF backend)
# regardless of whether it is activated in the caller's shell.
PY="python3"
if [ -x "../.venv/bin/python" ]; then PY="../.venv/bin/python"; fi
echo "Python      : $PY"

DATA_DIR="data/full_samples"
OUT_ROOT="results/full"
RESOLUTIONS="224 448 896"
TASKS="DocVQA ChartQA POPE COCO"

# Default model set (override by passing model names as arguments).
# Ollama-served models:
MODELS=(
  "qwen3-vl:8b"
  "gemma4:12b"
  "llava:7b"
  "ministral-3:8b"
  "minicpm-v:8b"
)
if [ "$#" -gt 0 ]; then
  MODELS=("$@")
fi

# HuggingFace-served models (not available on Ollama). Run via --backend hf.
# Set to empty to skip. InternVL is the dynamic-resolution doc contrast model.
HF_MODELS=(
  "OpenGVLab/InternVL3-8B-hf"
)
HF_QUANT="4bit"

if [ ! -f "$DATA_DIR/metadata.json" ] && [ ! -f "$DATA_DIR/multitask_samples_metadata.json" ]; then
  echo "[!] No dataset found in '$DATA_DIR'. Run:"
  echo "    python3 download_datasets.py --samples 250 --output $DATA_DIR"
  exit 1
fi

mkdir -p "$OUT_ROOT"
echo "Models      : ${MODELS[*]}"
echo "Data dir    : $DATA_DIR"
echo "Resolutions : $RESOLUTIONS"
echo "Output root : $OUT_ROOT"
echo "============================================================"

for M in "${MODELS[@]}"; do
  OUT="$OUT_ROOT/${M//:/_}"
  echo ""
  echo "############## MODEL: $M -> $OUT ##############"
  "$PY" main_benchmark.py \
      --model "$M" \
      --tasks $TASKS \
      --resolutions $RESOLUTIONS \
      --data-dir "$DATA_DIR" \
      --output-dir "$OUT" || echo "[!] $M exited non-zero (resume by re-running this script)."
  echo "############## DONE: $M ##############"
done

# HuggingFace backend models (skipped when invoking with explicit model args).
if [ "$#" -eq 0 ]; then
  for M in "${HF_MODELS[@]}"; do
    OUT="$OUT_ROOT/${M//\//_}"
    echo ""
    echo "############## MODEL (hf): $M -> $OUT ##############"
    "$PY" main_benchmark.py \
        --model "$M" \
        --backend hf --quant "$HF_QUANT" \
        --tasks $TASKS \
        --resolutions $RESOLUTIONS \
        --data-dir "$DATA_DIR" \
        --output-dir "$OUT" || echo "[!] $M exited non-zero (resume by re-running this script)."
    echo "############## DONE: $M ##############"
  done
fi

echo ""
echo "ALL_MODELS_COMPLETE"
