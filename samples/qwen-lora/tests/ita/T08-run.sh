#!/usr/bin/env bash
# ITa of T08 (written by the harness owner; agents run it, they do not change it).
# Builds tests/ita/T08/deltarule_check.pas against src/, runs it once from a
# fresh directory outside the project on the CPU, and writes
#   work/ita/T08/summary.tsv: case, OK|NG, value, limit, note
# (OK when value <= limit; the value is a max abs difference from the scalar
# Double reference of tests/ut/DeltaRuleFixture.pas or from hand values),
# then the driver's exit code and stderr line count. Takes about 10 s.
# Usage: tests/ita/T08-run.sh            (no QLORA_* variable is needed)
# T08_RESULTS, T08_UNITS, T08_EXTRA_FU are for the harness owner's own check
# against a reference unit; agents leave them unset.
set -euo pipefail
P=$(cd "$(dirname "$0")/../.." && pwd)
root=$(cd "$P/../.." && pwd)
lib=${TCH_LIBTCH:-$root/build/libtch}
[ -d "$lib" ] || lib=$root/build/libtch-rocm
export HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
R=${T08_RESULTS:-$P/work/ita/T08}
U=${T08_UNITS:-$P/work/bin/T08}
B=$U/deltarule_check
mkdir -p "$R" "$U"
rm -f "$R/summary.tsv" "$R/ours.tsv" "$R/errors.txt"
if ! fpc -B -Mobjfpc -Sh -O2 -vew -l- ${T08_EXTRA_FU:+-Fu"$T08_EXTRA_FU"} -Fu"$P/src" -Fu"$P/tests/ut" -Fu"$root/src" \
     -FU"$U" -Fl"$lib" -k-rpath="$lib" "$P/tests/ita/T08/deltarule_check.pas" -o"$B" > "$R/build.log" 2>&1; then
  echo "build failed: $R/build.log" >&2
  exit 1
fi
w=$(mktemp -d)
cd "$w"
rc=0
"$B" > "$R/ours.tsv" 2> "$R/errors.txt" || rc=$?
# libtorch's ROCm agent warns about the hidden GPUs on stderr; count the rest
grep -v -e 'agent.cpp' -e '^(null): No such file or directory$' "$R/errors.txt" > "$R/errors-other.txt" || true
for id in 001 002 003 004 005 006 007; do
  line=$(awk -F'\t' -v id="ITa-T08-$id" '$1 == id' "$R/ours.tsv")
  if [ -z "$line" ]; then
    printf 'ITa-T08-%s\tNG\t-\t-\tno output from the driver\n' "$id" >> "$R/summary.tsv"
    continue
  fi
  printf '%s\n' "$line" | awk -F'\t' -v OFS='\t' '{ r = ($2 + 0 <= $3 + 0) ? "OK" : "NG"; print $1, r, $2, $3, $4 }' >> "$R/summary.tsv"
done
printf 'driver\texit %s, %s other stderr lines (%s)\n' "$rc" "$(wc -l < "$R/errors-other.txt")" "$R/errors-other.txt" >> "$R/summary.tsv"
cat "$R/summary.tsv"
