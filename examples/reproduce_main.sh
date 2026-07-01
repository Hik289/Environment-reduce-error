#!/bin/bash
# Reproduce the main Stage-D registry layers.
# Usage: bash examples/reproduce_main.sh [path/to/cells_registry.csv]

set -euo pipefail

ENVPROBE_ROOT="${ENVPROBE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ENVPROBE_ROOT"

: "${OPENAI_API_KEY:?Please set OPENAI_API_KEY before running}"

REGISTRY="${1:-cells_registry.csv}"

mkdir -p experiments logs

python -m src.scripts.run_main \
  --registry "$REGISTRY" \
  --layer r3_stage_d \
  --prefix r3_stage_d \
  --parallel 8 2>&1 | tee logs/r3_stage_d_run.log

python -m src.scripts.run_main \
  --registry "$REGISTRY" \
  --layer r3_stage_d_ablation \
  --prefix r3_stage_d_ablation \
  --parallel 8 2>&1 | tee logs/r3_stage_d_ablation_run.log

python -m src.scripts.run_main \
  --registry "$REGISTRY" \
  --layer r3_stage_d_tw \
  --prefix r3_stage_d_tw \
  --parallel 6 2>&1 | tee logs/r3_stage_d_tw_run.log

echo ""
echo "Done. Episode metrics are under experiments/."
