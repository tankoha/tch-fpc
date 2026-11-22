#!/usr/bin/env bash
# ITa of T10 (written by the harness owner; agents run it, they do not change it).
# Builds tests/ita/T10/blocks_check.pas against src/, runs it once from a
# fresh directory outside the project on the CPU, and writes
#   work/ita/T10/summary.tsv: case, OK|NG, value, limit, note
# (OK when value <= limit; the value is a max abs difference from the slow
# scalar version of tests/ut/QwenRefScalar.pas, or from an ordinary backward),
# then the driver's exit code and stderr line count. Takes a
# few seconds.
# Usage: QLORA_MODEL=<absolute path of the rehearsal GGUF> tests/ita/T10-run.sh
#        QLORA_MODEL=... tests/ita/T10-run.sh gpu   harness owner only: the GPU
#                                        case 101 into work/ita/T10-gpu
# T10_RESULTS, T10_UNITS, T10_EXTRA_FU are for the harness owner's own check
# against a reference unit; agents leave them unset.
set -euo pipefail
P=$(cd "$(dirname "$0")/../.." && pwd)
M=${QLORA_MODEL:?set QLORA_MODEL to the rehearsal GGUF (absolute path)}
root=$(cd "$P/../.." && pwd)
lib=${TCH_LIBTCH:-$root/build/libtch}
[ -d "$lib" ] || lib=$root/build/libtch-rocm
mode=${1:-cpu}
if [ "$mode" = gpu ]; then
  ids="101"
  R=${T10_RESULTS:-$P/work/ita/T10-gpu}
else
  export HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
  ids="001 002 003 004 005"
  R=${T10_RESULTS:-$P/work/ita/T10}
fi
U=${T10_UNITS:-$P/work/bin/T10}
B=$U/blocks_check
mkdir -p "$R" "$U"
rm -f "$R/summary.tsv" "$R/ours.tsv" "$R/errors.txt"
if ! fpc -B -Mobjfpc -Sh -O2 -vew -l- ${T10_EXTRA_FU:+-Fu"$T10_EXTRA_FU"} -Fu"$P/src" -Fu"$P/tests/ut" -Fu"$root/src" \
     -FU"$U" -Fl"$lib" -k-rpath="$lib" "$P/tests/ita/T10/blocks_check.pas" -o"$B" > "$R/build.log" 2>&1; then
  echo "build failed: $R/build.log" >&2
  exit 1
fi
w=$(mktemp -d)
cd "$w"
rc=0
if [ "$mode" = gpu ]; then
  "$B" "$M" gpu > "$R/ours.tsv" 2> "$R/errors.txt" || rc=$?
else
  "$B" "$M" > "$R/ours.tsv" 2> "$R/errors.txt" || rc=$?
fi
# libtorch's ROCm agent warns about the hidden GPUs on stderr; count the rest
grep -v -e 'agent.cpp' -e '^(null): No such file or directory$' "$R/errors.txt" > "$R/errors-other.txt" || true
for id in $ids; do
  line=$(awk -F'\t' -v id="ITa-T10-$id" '$1 == id' "$R/ours.tsv")
  if [ -z "$line" ]; then
    printf 'ITa-T10-%s\tNG\t-\t-\tno output from the driver\n' "$id" >> "$R/summary.tsv"
    continue
  fi
  printf '%s\n' "$line" | awk -F'\t' -v OFS='\t' '{ r = ($2 + 0 <= $3 + 0) ? "OK" : "NG"; print $1, r, $2, $3, $4 }' >> "$R/summary.tsv"
done
printf 'driver\texit %s, %s other stderr lines (%s)\n' "$rc" "$(wc -l < "$R/errors-other.txt")" "$R/errors-other.txt" >> "$R/summary.tsv"
cat "$R/summary.tsv"
