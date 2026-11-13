#!/bin/bash
# =============================================================================
# GreenVLM — Full Benchmark Runner (config-driven, resumable)
#
# Models come from configs/models.yaml. Each model writes its own checkpoint
# (<OUT_ROOT>/<model>/checkpoint.jsonl); every completed inference is persisted
# immediately. If the run is interrupted (Ctrl-C, crash, reboot), re-run this
# script with the SAME settings — it skips finished work and resumes.
#
# Settings (override via environment variables):
#   DATA_DIR    dataset dir           (default data/full_samples)
#   OUT_ROOT    results dir           (default results/full)
#   RESOLUTIONS square sizes          (default from models.yaml)
#   TASKS       task list             (default from models.yaml)
#   MAX_TOKENS  uniform token budget  (default from models.yaml)
#
# Examples:
#   bash run_full_benchmark.sh
#   DATA_DIR=/mnt/bigdisk/greenvlm/data/full_samples \
#   OUT_ROOT=/mnt/bigdisk/greenvlm/results/full bash run_full_benchmark.sh
#   bash run_full_benchmark.sh qwen3-vl:8b        # run/resume one model
# =============================================================================
set -u
cd "$(dirname "$0")"

# Use the project virtualenv python (has torch/transformers for the HF backend)
# regardless of whether it is activated in the caller's shell.
PY="python3"
if [ -x "../.venv/bin/python" ]; then PY="../.venv/bin/python"; fi

CONFIG="configs/models.yaml"

# Read defaults from the YAML (env vars win if set).
read_default() { "$PY" -c "import yaml,sys; print(yaml.safe_load(open('$CONFIG')).get('defaults',{}).get('$1',''))"; }
DATA_DIR="${DATA_DIR:-data/full_samples}"
OUT_ROOT="${OUT_ROOT:-results/full}"
RESOLUTIONS="${RESOLUTIONS:-$(read_default resolutions)}"
TASKS="${TASKS:-$(read_default tasks)}"
MAX_TOKENS="${MAX_TOKENS:-$(read_default max_tokens)}"

if [ ! -f "$DATA_DIR/metadata.json" ] && [ ! -f "$DATA_DIR/multitask_samples_metadata.json" ]; then
  echo "[!] No dataset found in '$DATA_DIR'. Run:"
  echo "    $PY download_datasets.py --samples 250 --output $DATA_DIR"
  exit 1
fi

mkdir -p "$OUT_ROOT"
echo "Python      : $PY"
echo "Config      : $CONFIG"
echo "Data dir    : $DATA_DIR"
echo "Output root : $OUT_ROOT"
echo "Resolutions : $RESOLUTIONS"
echo "Tasks       : $TASKS"
echo "Max tokens  : $MAX_TOKENS (uniform across all models)"
echo "============================================================"

# Emit "backend|name|quant" per model (filtered to CLI args if any were given).
WANT="$*"
MODEL_LINES=$("$PY" - "$CONFIG" "$WANT" <<'PYEOF'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
want = set(sys.argv[2].split()) if len(sys.argv) > 2 and sys.argv[2].strip() else None
for m in cfg.get("models", []):
    if want and m["name"] not in want:
        continue
    print(f"{m.get('backend','ollama')}|{m['name']}|{m.get('quant','4bit')}")
PYEOF
)

if [ -z "$MODEL_LINES" ]; then
  echo "[!] No matching models in $CONFIG."
  exit 1
fi

while IFS='|' read -r BACKEND NAME QUANT; do
  [ -z "$NAME" ] && continue
  OUT="$OUT_ROOT/$(echo "$NAME" | tr '/:' '__')"
  echo ""
  echo "############## MODEL: $NAME  (backend=$BACKEND) -> $OUT ##############"
  EXTRA=""
  if [ "$BACKEND" = "hf" ]; then EXTRA="--backend hf --quant $QUANT"; fi
  "$PY" main_benchmark.py \
      --model "$NAME" $EXTRA \
      --tasks $TASKS \
      --resolutions $RESOLUTIONS \
      --max-tokens "$MAX_TOKENS" \
      --data-dir "$DATA_DIR" \
      --output-dir "$OUT" || echo "[!] $NAME exited non-zero (resume by re-running this script)."
  echo "############## DONE: $NAME ##############"
done <<< "$MODEL_LINES"

echo ""
echo "ALL_MODELS_COMPLETE"
