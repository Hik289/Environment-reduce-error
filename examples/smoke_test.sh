#!/bin/bash
# Smoke test: 1 episode across the three EnvProbe worlds.
# Requires OPENAI_API_KEY because even no_probe uses the LLM agent.

set -euo pipefail

ENVPROBE_ROOT="${ENVPROBE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ENVPROBE_ROOT"

: "${OPENAI_API_KEY:?Please set OPENAI_API_KEY before running}"

mkdir -p experiments

python -m src.scripts.run_smoke \
  --envs ObjectStateWorld ToolDAGWorld GraphNavWorld \
  --stress S2 \
  --methods no_probe periodic_probe envprobe_simple \
  --episodes 1 \
  --prefix smoke

echo ""
echo "Smoke test complete. Check experiments/ for JSONL outputs."
