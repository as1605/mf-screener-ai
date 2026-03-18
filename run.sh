#!/usr/bin/env bash
#
# run.sh: Fetch data → Run all algorithms → Compile & publish to Google Sheet.
# Run from project root (script cd's to root automatically).
#
# Usage:
#   ./run.sh                  # today's data folder; no git branch workflow
#   ./run.sh 2026-02-13       # branch date/2026-02-13:
#                             #   - new branch: created from origin/main (or main/master)
#                             #   - existing: merge main on top, then fetch/analyze/commit/push
#

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/" && pwd)"
ALGOS_DIR="$ROOT_DIR/src/algorithms"

cd "$ROOT_DIR"

RUN_DATE="${1:-}"
BRANCH=""
DATE_ARGS=()

if [[ -n "$RUN_DATE" ]]; then
  if ! [[ "$RUN_DATE" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    echo "Usage: $0 [YYYY-MM-DD]"
    echo "  Date must look like 2026-02-13"
    exit 1
  fi
  BRANCH="date/$RUN_DATE"
  DATE_ARGS=(--date "$RUN_DATE")

  if ! git rev-parse --git-dir >/dev/null 2>&1; then
    echo "Not a git repository; cannot use dated run (branch/commit/push)."
    exit 1
  fi

  echo ""
  echo "========================================"
  echo "  GIT: branch $BRANCH"
  echo "========================================"
  git fetch origin 2>/dev/null || true

  # Primary branch to merge from (prefer remote tracking)
  MAIN_REF=""
  if git rev-parse "origin/main" >/dev/null 2>&1; then
    MAIN_REF="origin/main"
  elif git rev-parse "origin/master" >/dev/null 2>&1; then
    MAIN_REF="origin/master"
  elif git rev-parse "main" >/dev/null 2>&1; then
    MAIN_REF="main"
  elif git rev-parse "master" >/dev/null 2>&1; then
    MAIN_REF="master"
  else
    echo "Could not find main or master (local or origin). Cannot run dated workflow."
    exit 1
  fi
  echo "  Integration branch: $MAIN_REF"

  LOCAL_BRANCH=$(git show-ref --verify --quiet "refs/heads/$BRANCH" && echo 1 || echo 0)
  REMOTE_BRANCH=$(git show-ref --verify --quiet "refs/remotes/origin/$BRANCH" && echo 1 || echo 0)

  if [[ "$LOCAL_BRANCH" == 1 ]]; then
    git checkout "$BRANCH"
    if git rev-parse "@{u}" >/dev/null 2>&1; then
      git pull --no-rebase --ff-only 2>/dev/null || git pull --no-rebase || true
    elif [[ "$REMOTE_BRANCH" == 1 ]]; then
      git pull --no-rebase origin "$BRANCH" 2>/dev/null || true
    fi
    echo "  Merging $MAIN_REF into $BRANCH ..."
    git merge "$MAIN_REF" --no-edit
  elif [[ "$REMOTE_BRANCH" == 1 ]]; then
    git checkout -b "$BRANCH" "origin/$BRANCH"
    echo "  Merging $MAIN_REF into $BRANCH ..."
    git merge "$MAIN_REF" --no-edit
  else
    echo "  Creating $BRANCH from $MAIN_REF ..."
    git checkout -b "$BRANCH" "$MAIN_REF"
  fi
fi

echo ""
echo "========================================"
echo "  1/3  FETCHING ALL DATA"
echo "========================================"
if [[ ${#DATE_ARGS[@]} -gt 0 ]]; then
  python3 src/mf_data_provider.py "${DATE_ARGS[@]}"
else
  python3 src/mf_data_provider.py
fi

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
    if [[ ${#DATE_ARGS[@]} -gt 0 ]]; then
      python3 "$script" "${DATE_ARGS[@]}" || { echo "FAILED: $name"; exit 1; }
    else
      python3 "$script" || { echo "FAILED: $name"; exit 1; }
    fi
  fi
done

echo ""
if [[ ${#DATE_ARGS[@]} -gt 0 ]]; then
  echo "========================================"
  echo "  3/3  COMPILING (Google Sheet skipped for dated run)"
  echo "========================================"
  python3 run_publish.py --no-sheet
else
  echo "========================================"
  echo "  3/3  COMPILING & PUBLISHING"
  echo "========================================"
  python3 run_publish.py
fi

echo ""
echo "========================================"
if [[ ${#DATE_ARGS[@]} -gt 0 ]]; then
  echo "  DONE: fetch → algorithms → compile (sheet unchanged)"
else
  echo "  DONE: fetch → algorithms → compile & publish"
fi
echo "========================================"

if [[ -n "$BRANCH" ]]; then
  echo ""
  echo "========================================"
  echo "  GIT: commit & push $BRANCH"
  echo "========================================"
  git add -A
  if git diff --cached --quiet; then
    echo "Nothing to commit (working tree clean after run)."
  else
    git commit -m "Screener run for $RUN_DATE"
  fi
  git push -u origin "$BRANCH"
  echo "Pushed $BRANCH"
fi
