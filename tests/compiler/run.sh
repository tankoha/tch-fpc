#!/bin/bash
# Compiler acceptance tests (HANDOFF step 7). Run from the repo root.
# h*.pas must compile and run; xfail_*.pas must FAIL to compile;
# gen_big.sh builds a 2,657-method record and reports the compile time.
set -euo pipefail
out=build/compiler
mkdir -p "$out"
failures=0
for src in tests/compiler/h*.pas; do
  name=$(basename "$src" .pas)
  echo "=== $name"
  if fpc -FU"$out" "$src" -o"$out/$name" >"$out/$name.log" 2>&1; then
    "$out/$name"
  else
    grep -E 'Error|Fatal' "$out/$name.log" || true
    echo "  FAIL: did not compile"
    failures=$((failures + 1))
  fi
done
for src in tests/compiler/xfail_*.pas; do
  name=$(basename "$src" .pas)
  echo "=== $name (must not compile)"
  if fpc -FU"$out" "$src" -o"$out/$name" >"$out/$name.log" 2>&1; then
    echo "  FAIL: compiled"
    failures=$((failures + 1))
  else
    grep -E -o 'Error: .*' "$out/$name.log" | head -1
    echo "  ok: rejected as expected"
  fi
done
echo "=== big_record (2,657 methods via {\$I})"
tests/compiler/gen_big.sh "$out/big"
start=$(date +%s%N)
if fpc -FU"$out/big" "$out/big/big_record.pas" -o"$out/big/big_record" \
    >"$out/big/build.log" 2>&1; then
  end=$(date +%s%N)
  echo "  compiled in $(( (end - start) / 1000000 )) ms"
  "$out/big/big_record"
else
  grep -E 'Error|Fatal' "$out/big/build.log" || true
  echo "  FAIL: did not compile"
  failures=$((failures + 1))
fi
echo
if [ "$failures" -eq 0 ]; then echo OK; else echo "FAILED ($failures)"; exit 1; fi
