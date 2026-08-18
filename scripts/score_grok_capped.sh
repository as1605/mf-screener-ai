#!/usr/bin/env bash
# Score all four Grok sectors sequentially at a CPU duty-cycle cap.
# Usage: scripts/score_grok_capped.sh [YYYY-MM-DD] [percent]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATE="${1:-2026-08-15}"
PCT="${2:-60}"
cd "$ROOT"
mkdir -p logs results/experiments
for s in "Small Cap" "Mid Cap" "Multi Asset" "Total Market"; do
  safe="${s// /_}"
  echo "======== scoring $s (cap ${PCT}%) ========"
  "$ROOT/scripts/run_capped.sh" "$PCT" -- \
    .venv/bin/python "$ROOT/src/algorithms/${s}_Grok.py" --date "$DATE" \
    | tee "logs/score_${safe}.log"
done
echo "done all sectors"
