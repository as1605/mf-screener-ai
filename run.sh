#!/usr/bin/env bash
#
# run.sh: Fetch data → Run all algorithms → Compile results (Local runner).
# Usage:
#   ./run.sh                  # Run for current date
#   ./run.sh 2026-02-13       # Run for specific date
#

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"
ALGOS_DIR="$ROOT_DIR/src/algorithms"

cd "$ROOT_DIR"

RUN_DATE="${1:-}"
DATE_ARGS=()

if [[ -n "$RUN_DATE" ]]; then
  if ! [[ "$RUN_DATE" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    echo "Usage: $0 [YYYY-MM-DD]"
    echo "  Date must look like 2026-02-13"
    exit 1
  fi
  DATE_ARGS=(--date "$RUN_DATE")
fi

echo ""
echo "========================================"
echo "  1/3  FETCHING DATA"
echo "========================================"
if [[ ${#DATE_ARGS[@]} -gt 0 ]]; then
  python3 src/mf_data_provider.py "${DATE_ARGS[@]}"
else
  python3 src/mf_data_provider.py
fi

echo ""
echo "========================================"
echo "  2/3  RUNNING ALGORITHMS"
echo "========================================"
for script in "$ALGOS_DIR"/*.py; do
  if [[ -f "$script" ]]; then
    name="$(basename "$script")"
    echo ""
    echo ">>> $name"
    echo "----------------------------------------"
    if [[ ${#DATE_ARGS[@]} -gt 0 ]]; then
      python3 "$script" "${DATE_ARGS[@]}" || { echo "FAILED: $name"; exit 1; }
    else
      python3 "$script" || { echo "FAILED: $name"; exit 1; }
    fi
  fi
done

echo ""
echo "========================================"
echo "  3/3  COMPILING RESULTS"
echo "========================================"
python3 src/run.py --no-sheet

echo ""
echo "========================================"
echo "  DONE: fetch → algorithms → compile"
echo "========================================"
