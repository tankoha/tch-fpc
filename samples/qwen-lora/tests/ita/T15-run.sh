#!/usr/bin/env bash
# ITa of T15 (written by the harness owner; agents run it, they do not change it).
# Builds tests/ita/T15/data_check.pas against src/ and runs it on the CPU
# from a fresh directory outside the project with the rehearsal model's
# tokenizer and tests/ita/T15/records.jsonl: once `check` (cases 001, 002,
# 003, 005), twice `dump` in two processes (case 004: the batches of three
# epochs must be equal). Writes
#   work/ita/T15/summary.tsv: case, OK|NG, value, limit, note
# (OK when value <= limit), then the exit codes and the count of stderr
# lines other than the GPU warnings. Takes under a minute.
# Usage: QLORA_MODEL=<absolute path of the rehearsal GGUF> tests/ita/T15-run.sh
# T15_RESULTS, T15_UNITS, T15_EXTRA_FU are for the harness owner's own check
# against a reference unit; agents leave them unset.
set -euo pipefail
P=$(cd "$(dirname "$0")/../.." && pwd)
M=${QLORA_MODEL:?set QLORA_MODEL to the rehearsal GGUF (absolute path)}
root=$(cd "$P/../.." && pwd)
lib=${TCH_LIBTCH:-$root/build/libtch}
[ -d "$lib" ] || lib=$root/build/libtch-rocm
export HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
ids="001 002 003 004 005"
R=${T15_RESULTS:-$P/work/ita/T15}
U=${T15_UNITS:-$P/work/bin/T15}
mkdir -p "$R" "$U"
rm -f "$R/summary.tsv" "$R/ours.tsv" "$R/errors.txt" "$R/errors-other.txt" "$R/dump1.txt" "$R/dump2.txt"
src=$P/tests/ita/T15/data_check.pas
if ! fpc -B -Mobjfpc -Sh -O2 -vew -l- ${T15_EXTRA_FU:+-Fu"$T15_EXTRA_FU"} -Fu"$P/src" -Fu"$root/src" \
     -FU"$U" -Fl"$lib" -k-rpath="$lib" "$src" -o"$U/data_check" > "$R/build-data_check.log" 2>&1; then
  echo "build failed: $R/build-data_check.log" >&2
  exit 1
fi
w=$(mktemp -d)
cd "$w"
cp "$P/tests/ita/T15/records.jsonl" .
: > "$R/errors.txt"
rc1=0; rc2=0; rc3=0
"$U/data_check" "$M" records.jsonl check > "$R/ours.tsv" 2>> "$R/errors.txt" || rc1=$?
"$U/data_check" "$M" records.jsonl dump > "$R/dump1.txt" 2>> "$R/errors.txt" || rc2=$?
"$U/data_check" "$M" records.jsonl dump > "$R/dump2.txt" 2>> "$R/errors.txt" || rc3=$?
n=$(wc -l < "$R/dump1.txt")
if [ "$rc2" = 0 ] && [ "$rc3" = 0 ] && [ "$n" -gt 0 ] && cmp -s "$R/dump1.txt" "$R/dump2.txt"; then d=0; else d=1; fi
printf 'ITa-T15-004\t%s\t0\tthe batches of epochs 0-2 (%s lines), two processes: 0 when equal\n' "$d" "$n" >> "$R/ours.tsv"
# libtorch's ROCm agent warns about the hidden GPUs on stderr; count the rest
grep -v -e 'agent.cpp' -e '^(null): No such file or directory$' "$R/errors.txt" > "$R/errors-other.txt" || true
for id in $ids; do
  line=$(awk -F'\t' -v id="ITa-T15-$id" '$1 == id' "$R/ours.tsv")
  if [ -z "$line" ]; then
    printf 'ITa-T15-%s\tNG\t-\t-\tno output from the driver\n' "$id" >> "$R/summary.tsv"
    continue
  fi
  printf '%s\n' "$line" | awk -F'\t' -v OFS='\t' '{ r = ($2 + 0 <= $3 + 0) ? "OK" : "NG"; print $1, r, $2, $3, $4 }' >> "$R/summary.tsv"
done
printf 'driver\tdata_check exit codes: check %s, dump %s, dump %s; %s other stderr lines (%s)\n' \
  "$rc1" "$rc2" "$rc3" "$(wc -l < "$R/errors-other.txt")" "$R/errors-other.txt" >> "$R/summary.tsv"
rm -rf "$w"
cat "$R/summary.tsv"
