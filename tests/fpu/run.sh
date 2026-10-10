#!/bin/bash
# Runs every fpu_test mode in its own process and reports the exit status
# (crashes are expected for the nomask modes). Run from the repo root.
# libtorch's "[W..." warning lines are filtered out; a mode that does not
# finish within 30 s is reported as hung.
set -euo pipefail
bin=${1:-build/fpu_test}
out=$(mktemp)
trap 'rm -f "$out"' EXIT
for mode in nomask-small nomask-large mask pool-first thread intdiv; do
  status=0
  timeout 30 "$bin" "$mode" >"$out" 2>&1 || status=$?
  grep -v '^\[W' "$out" || true
  if [ "$status" -eq 124 ]; then echo "  HUNG (killed after 30 s)"; else echo "  exit status: $status"; fi
  echo
done
