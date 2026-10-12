#!/bin/bash
# Runs every fpu_test mode in its own process and checks the outcome
# recorded in DESIGN.md 4.2 / HANDOFF step 5. Run from the repo root.
# - mask, thread (H4), intdiv (H5): exit 0, 1/0 gives +Inf twice
# - nomask-small: EZeroDivide raised from inside C++ frames
# - nomask-large, pool-first: crash or hang (they show why Tch masks in
#   its initialization, before libtorch creates its pool); a clean run
#   means the behaviour changed and is reported as a failure
# libtorch's "[W..." warning lines are filtered out; a mode that does not
# finish within 30 s is killed and counts as a hang.
set -euo pipefail
bin=${1:-build/fpu_test}
out=$(mktemp)
trap 'rm -f "$out"' EXIT
failures=0
for mode in nomask-small nomask-large mask pool-first thread intdiv; do
  status=0
  timeout 30 "$bin" "$mode" >"$out" 2>&1 || status=$?
  grep -v '^\[W' "$out" | grep -v '^An unhandled exception\|^EZeroDivide: \|^EAccessViolation\|^  \$' || true
  if [ "$status" -eq 124 ]; then echo "  HUNG (killed after 30 s)"; else echo "  exit status: $status"; fi
  infs=$(grep -c 'ok, q\[0\] = +Inf' "$out" || true)
  case "$mode" in
    mask|thread|intdiv)
      if [ "$status" -eq 0 ] && [ "$infs" -eq 2 ]; then echo "  ok: masked, +Inf twice"
      else echo "  FAIL: expected exit 0 and +Inf twice"; failures=$((failures + 1)); fi ;;
    nomask-small)
      if grep -q 'Pascal exception EZeroDivide' "$out"; then echo "  ok: EZeroDivide raised"
      else echo "  FAIL: expected EZeroDivide"; failures=$((failures + 1)); fi ;;
    nomask-large|pool-first)
      if [ "$status" -ne 0 ]; then echo "  ok: crashed or hung, as recorded"
      else echo "  FAIL: ran cleanly unmasked (behaviour changed)"; failures=$((failures + 1)); fi ;;
  esac
  echo
done
if [ "$failures" -eq 0 ]; then echo OK; else echo "FAILED ($failures)"; exit 1; fi
