#!/usr/bin/env bash
# Run a command at ~60% duty cycle (SIGSTOP/SIGCONT) so a laptop stays cool.
# Usage: scripts/run_capped.sh [percent] -- cmd args...
# Example: scripts/run_capped.sh 60 -- .venv/bin/python src/algorithms/Small\ Cap_Grok.py --date 2026-08-15
set -euo pipefail
PCT="${1:-60}"
shift
if [[ "${1:-}" == "--" ]]; then shift; fi
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 [percent] -- command..." >&2
  exit 2
fi
if ! [[ "$PCT" =~ ^[0-9]+$ ]] || (( PCT < 10 || PCT > 100 )); then
  echo "percent must be 10-100, got $PCT" >&2
  exit 2
fi

"$@" &
pid=$!
on=$(awk -v p="$PCT" 'BEGIN{printf "%.2f", p/100.0 * 0.5}')
off=$(awk -v p="$PCT" 'BEGIN{printf "%.2f", (1-p/100.0) * 0.5}')
echo "capped pid=$pid at ${PCT}% (on=${on}s off=${off}s): $*"

cleanup() {
  kill -CONT "$pid" 2>/dev/null || true
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}
trap cleanup INT TERM

while kill -0 "$pid" 2>/dev/null; do
  kill -CONT "$pid" 2>/dev/null || break
  sleep "$on"
  if ! kill -0 "$pid" 2>/dev/null; then break; fi
  kill -STOP "$pid" 2>/dev/null || break
  sleep "$off"
done
wait "$pid"
trap - INT TERM
exit $?
