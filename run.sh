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

# ---------------------------------------------------------------------------
# Python & Virtual Environment Setup
# ---------------------------------------------------------------------------
PYTHON_CMD=""
if command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_CMD="python"
else
  echo "Error: Neither python3 nor python was found in PATH." >&2
  exit 1
fi

VENV_DIR=""
for v in .venv venv env; do
  if [[ -d "$ROOT_DIR/$v" && -f "$ROOT_DIR/$v/bin/activate" ]]; then
    VENV_DIR="$ROOT_DIR/$v"
    break
  fi
done

if [[ -z "$VENV_DIR" ]]; then
  echo "No virtual environment found. Creating .venv with $PYTHON_CMD..."
  "$PYTHON_CMD" -m venv "$ROOT_DIR/.venv"
  VENV_DIR="$ROOT_DIR/.venv"
fi

echo "Using virtual environment: $VENV_DIR"
source "$VENV_DIR/bin/activate"

# Check if essential requirements are installed
if ! python -c "import pandas, gspread, numpy, scipy, dotenv" >/dev/null 2>&1; then
  echo "Installing dependencies from requirements.txt..."
  pip install -r "$ROOT_DIR/requirements.txt"
fi

# ---------------------------------------------------------------------------
# Pipeline Execution
# ---------------------------------------------------------------------------
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
  python src/mf_data_provider.py "${DATE_ARGS[@]}"
else
  python src/mf_data_provider.py
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
      python "$script" "${DATE_ARGS[@]}" || { echo "FAILED: $name"; exit 1; }
    else
      python "$script" || { echo "FAILED: $name"; exit 1; }
    fi
  fi
done

echo ""
echo "========================================"
echo "  3/3  COMPILING RESULTS"
echo "========================================"
python run.py --no-sheet

echo ""
echo "========================================"
echo "  DONE: fetch → algorithms → compile"
echo "========================================"
