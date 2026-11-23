#!/usr/bin/env bash
# ITa of T16 (written by the harness owner; agents run it, they do not change it).
# Builds tests/ita/T16/ckpt_check.pas against src/ and runs it four times on
# the CPU from a fresh directory outside the project, each run its own
# process: full (10 updates in one go, a checkpoint after each), part1
# (5 updates, a checkpoint), part2 (a fresh run resumed from it, 5 more),
# compare. LoRA rank 16, scale 2 on the rehearsal model; the oracle of the
# resume is the uninterrupted run. Writes
#   work/ita/T16/summary.tsv: case, OK|NG, value, limit, note
# (OK when value <= limit), then the exit codes of the four runs and the
# count of stderr lines other than the GPU warnings. Takes a few minutes
# and needs about 0.5 GB in work/ita/T16/ckpt (deleted afterwards).
# Usage: QLORA_MODEL=<absolute path of the rehearsal GGUF> tests/ita/T16-run.sh
# T16_RESULTS, T16_UNITS, T16_EXTRA_FU are for the harness owner's own check
# against a reference unit; agents leave them unset.
set -euo pipefail
P=$(cd "$(dirname "$0")/../.." && pwd)
M=${QLORA_MODEL:?set QLORA_MODEL to the rehearsal GGUF (absolute path)}
root=$(cd "$P/../.." && pwd)
lib=${TCH_LIBTCH:-$root/build/libtch}
[ -d "$lib" ] || lib=$root/build/libtch-rocm
export HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
ids="001 002 003 004"
R=${T16_RESULTS:-$P/work/ita/T16}
U=${T16_UNITS:-$P/work/bin/T16}
mkdir -p "$R" "$U"
rm -rf "$R/summary.tsv" "$R/ours.tsv" "$R/errors.txt" "$R/errors-other.txt" "$R/ckpt"
src=$P/tests/ita/T16/ckpt_check.pas
if ! fpc -B -Mobjfpc -Sh -O2 -vew -l- ${T16_EXTRA_FU:+-Fu"$T16_EXTRA_FU"} -Fu"$P/src" -Fu"$root/src" \
     -FU"$U" -Fl"$lib" -k-rpath="$lib" "$src" -o"$U/ckpt_check" > "$R/build-ckpt_check.log" 2>&1; then
  echo "build failed: $R/build-ckpt_check.log" >&2
  exit 1
fi
w=$(mktemp -d)
cd "$w"
cp "$P/work/oracle/T11/prompts.jsonl" .
mkdir -p "$R/ckpt"
codes=""
: > "$R/ours.tsv"
: > "$R/errors.txt"
for mode in full part1 part2 compare; do
  rc=0
  "$U/ckpt_check" "$M" prompts.jsonl "$R/ckpt" "$mode" >> "$R/ours.tsv" 2>> "$R/errors.txt" || rc=$?
  codes="$codes $mode $rc"
done
# libtorch's ROCm agent warns about the hidden GPUs on stderr; count the rest
grep -v -e 'agent.cpp' -e '^(null): No such file or directory$' "$R/errors.txt" > "$R/errors-other.txt" || true
for id in $ids; do
  line=$(awk -F'\t' -v id="ITa-T16-$id" '$1 == id' "$R/ours.tsv")
  if [ -z "$line" ]; then
    printf 'ITa-T16-%s\tNG\t-\t-\tno output from the driver\n' "$id" >> "$R/summary.tsv"
    continue
  fi
  printf '%s\n' "$line" | awk -F'\t' -v OFS='\t' '{ r = ($2 + 0 <= $3 + 0) ? "OK" : "NG"; print $1, r, $2, $3, $4 }' >> "$R/summary.tsv"
done
printf 'driver\tckpt_check exit codes:%s; %s other stderr lines (%s)\n' \
  "$codes" "$(wc -l < "$R/errors-other.txt")" "$R/errors-other.txt" >> "$R/summary.tsv"
cp "$R/ckpt/full-loss.tsv" "$R/ckpt/part-loss.tsv" "$R/" 2>/dev/null || true
rm -rf "$w" "$R/ckpt"
cat "$R/summary.tsv"
