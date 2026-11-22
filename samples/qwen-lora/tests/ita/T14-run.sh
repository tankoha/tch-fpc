#!/usr/bin/env bash
# ITa of T14 (written by the harness owner; agents run it, they do not change it).
# Builds tests/ita/T14/loraoptim_check.pas against src/, runs it once from a
# fresh directory outside the project on the CPU, and writes
#   work/ita/T14/summary.tsv: case, OK|NG, value, limit, note
# (OK when value <= limit; the value is a difference from libtorch's AdamW
# (TOptimizer.AdamW) on the toy problem of tests/ut/LoraOptimFixture.pas),
# then the driver's exit code and stderr line count. Takes a
# few seconds.
# Usage: tests/ita/T14-run.sh            (no QLORA_* variable is needed)
#        tests/ita/T14-run.sh gpu        harness owner only: the two-GPU
#                                        cases 101-102 into work/ita/T14-gpu
# T14_RESULTS, T14_UNITS, T14_EXTRA_FU are for the harness owner's own check
# against a reference unit; agents leave them unset.
set -euo pipefail
P=$(cd "$(dirname "$0")/../.." && pwd)
root=$(cd "$P/../.." && pwd)
lib=${TCH_LIBTCH:-$root/build/libtch}
[ -d "$lib" ] || lib=$root/build/libtch-rocm
mode=${1:-cpu}
if [ "$mode" = gpu ]; then
  ids="101 102"
  R=${T14_RESULTS:-$P/work/ita/T14-gpu}
else
  export HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
  ids="001 002 003 004 005 006 007"
  R=${T14_RESULTS:-$P/work/ita/T14}
fi
U=${T14_UNITS:-$P/work/bin/T14}
B=$U/loraoptim_check
mkdir -p "$R" "$U"
rm -f "$R/summary.tsv" "$R/ours.tsv" "$R/errors.txt"
if ! fpc -B -Mobjfpc -Sh -O2 -vew -l- ${T14_EXTRA_FU:+-Fu"$T14_EXTRA_FU"} -Fu"$P/src" -Fu"$P/tests/ut" -Fu"$root/src" \
     -FU"$U" -Fl"$lib" -k-rpath="$lib" "$P/tests/ita/T14/loraoptim_check.pas" -o"$B" > "$R/build.log" 2>&1; then
  echo "build failed: $R/build.log" >&2
  exit 1
fi
w=$(mktemp -d)
cd "$w"
rc=0
if [ "$mode" = gpu ]; then
  "$B" gpu > "$R/ours.tsv" 2> "$R/errors.txt" || rc=$?
else
  "$B" > "$R/ours.tsv" 2> "$R/errors.txt" || rc=$?
fi
# libtorch's ROCm agent warns about the hidden GPUs on stderr; count the rest
grep -v -e 'agent.cpp' -e '^(null): No such file or directory$' "$R/errors.txt" > "$R/errors-other.txt" || true
for id in $ids; do
  line=$(awk -F'\t' -v id="ITa-T14-$id" '$1 == id' "$R/ours.tsv")
  if [ -z "$line" ]; then
    printf 'ITa-T14-%s\tNG\t-\t-\tno output from the driver\n' "$id" >> "$R/summary.tsv"
    continue
  fi
  printf '%s\n' "$line" | awk -F'\t' -v OFS='\t' '{ r = ($2 + 0 <= $3 + 0) ? "OK" : "NG"; print $1, r, $2, $3, $4 }' >> "$R/summary.tsv"
done
printf 'driver\texit %s, %s other stderr lines (%s)\n' "$rc" "$(wc -l < "$R/errors-other.txt")" "$R/errors-other.txt" >> "$R/summary.tsv"
cat "$R/summary.tsv"
