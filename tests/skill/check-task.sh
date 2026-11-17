#!/usr/bin/env bash
# Check one answer of an H39 task (DESIGN 7.5): it must compile with the
# skill's build script and print exactly the task's expected.txt.
# Usage: check-task.sh <task dir> <answer.pas>
#   LIBTCH  libtch build directory (default: build/libtch)
# Prints PASS or FAIL with the reason; exit 0 on PASS.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
task=$(cd "$1" && pwd)
answer=$2
libtch=${LIBTCH:-$root/build/libtch}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
[ -f "$answer" ] || { echo "FAIL no answer file"; exit 1; }
if ! exe=$(TCH_FPC_ROOT=$root TCH_OUT=$work TCH_UNITS=$work/units \
    "$root/plugin/skills/tch-fpc/scripts/build-example.sh" "$answer" "$libtch" 2> "$work/compile.log"); then
  echo "FAIL does not compile: $(grep -E 'Error|Fatal' "$work/compile.log" | head -1)"
  exit 1
fi
if ! (cd "$root" && timeout 120 "$exe") > "$work/out.txt" 2> "$work/err.txt"; then
  echo "FAIL run failed: $(tail -1 "$work/err.txt")"
  exit 1
fi
if ! diff <(sed 's/[[:space:]]*$//' "$task/expected.txt") <(sed 's/[[:space:]]*$//' "$work/out.txt") > "$work/diff.txt"; then
  echo "FAIL output differs: $(grep -E '^[<>]' "$work/diff.txt" | head -2 | tr '\n' ' ')"
  exit 1
fi
echo "PASS"
