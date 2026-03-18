#!/usr/bin/env bash
#
# run.sh: Fetch data → Run all algorithms → Compile & publish to Google Sheet.
# Run from project root (script cd's to root automatically).
#

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/" && pwd)"
ALGOS_DIR="$ROOT_DIR/src/algorithms"

cd "$ROOT_DIR"

echo ""
echo "========================================"
echo "  1/3  FETCHING ALL DATA"
echo "========================================"
python3 src/mf_data_provider.py

echo ""
echo "========================================"
echo "  2/3  RUNNING ALL ALGORITHMS"
echo "========================================"
for script in "$ALGOS_DIR"/*.py; do
  if [[ -f "$script" ]]; then
    name="$(basename "$script")"
    echo ""
    echo ">>> $name"
    echo "----------------------------------------"
    python3 "$script" || { echo "FAILED: $name"; exit 1; }
  fi
done

echo ""
echo "========================================"
echo "  3/3  COMPILING & PUBLISHING"
echo "========================================"
python3 run_publish.py

echo ""
echo "========================================"
echo "  DONE: fetch → algorithms → publish"
echo "========================================"
