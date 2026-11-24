#!/usr/bin/env bash
# ITa of T18 (written by the harness owner; agents run it, they do not change it).
# Builds src/qlora_train.pas, src/qlora_merge.pas and tests/ita/T18/merge_check.pas
# against src/ and runs on the CPU from a fresh directory outside the
# project, with the rehearsal model, each program its own process:
#   train  qlora_train on tests/ita/T17/train.jsonl (8 short records), 4
#          epochs of one update (lr 1e-3, warm-up 1): ckpt-00000004
#   merge  qlora_merge MODEL ckpt-00000004 merged.gguf
# then merge_check (cases 001-005; DESIGN 8.15 with the floor criterion of
# 2026-11-25) and case 006: merged.gguf byte for byte the prototype's
# (its sha256 in work/oracle/T18/merged.sha256). The prompts are those of
# work/oracle/T11; the H47 reference (work/oracle/T18/expected.jsonl) is
# the bundled llama-server on an all-F32 copy of the prototype's merged
# file, taken once by the harness owner. Writes
#   work/ita/T18/summary.tsv: case, OK|NG, value, limit, note
# (OK when value <= limit), then the exit codes and the count of stderr
# lines other than the GPU warnings. Takes about 5 minutes.
# Usage: QLORA_MODEL=<absolute path of the rehearsal GGUF> tests/ita/T18-run.sh
# T18_RESULTS, T18_UNITS, T18_EXTRA_FU are for the harness owner's own check
# against a reference unit; agents leave them unset. With T18_EXTRA_FU the
# programs are built from copies in T18_UNITS: FPC looks for units next to
# the main source first, so a build in src/ would take src/'s units.
set -euo pipefail
P=$(cd "$(dirname "$0")/../.." && pwd)
M=${QLORA_MODEL:?set QLORA_MODEL to the rehearsal GGUF (absolute path)}
root=$(cd "$P/../.." && pwd)
lib=${TCH_LIBTCH:-$root/build/libtch}
[ -d "$lib" ] || lib=$root/build/libtch-rocm
export HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
O=$P/work/oracle/T18
ids="001 002 003 004 005"
R=${T18_RESULTS:-$P/work/ita/T18}
U=${T18_UNITS:-$P/work/bin/T18}
mkdir -p "$R" "$U"
rm -rf "$R/summary.tsv" "$R/ours.tsv" "$R/errors.txt" "$R/errors-other.txt" "$R/runs"
progs="$P/src/qlora_train.pas $P/src/qlora_merge.pas"
if [ -n "${T18_EXTRA_FU:-}" ]; then
  cp "$P/src/qlora_train.pas" "$P/src/qlora_merge.pas" "$U/"
  progs="$U/qlora_train.pas $U/qlora_merge.pas"
fi
for prog in $progs "$P/tests/ita/T18/merge_check.pas"; do
  name=$(basename "$prog" .pas)
  if ! fpc -B -Mobjfpc -Sh -O2 -vew -l- ${T18_EXTRA_FU:+-Fu"$T18_EXTRA_FU"} -Fu"$P/src" -Fu"$root/src" \
       -FU"$U" -Fl"$lib" -k-rpath="$lib" "$prog" -o"$U/$name" > "$R/build-$name.log" 2>&1; then
    echo "build failed: $R/build-$name.log" >&2
    exit 1
  fi
done
w=$(mktemp -d)
cd "$w"
cp "$P/tests/ita/T17/train.jsonl" .
mkdir -p "$R/runs"
: > "$R/errors.txt"
codes=""
run() {  # name, program, then its arguments
  local n=$1 p=$2; shift 2
  local rc=0
  "$U/$p" "$@" >> "$R/runs/out.txt" 2>> "$R/errors.txt" || rc=$?
  codes="$codes $n $rc"
}
run train qlora_train --epochs 4 --warmup 1 --lr 1e-3 --batches-per-update 1 --ckpt-every 100 \
  "$M" train.jsonl "$w/train"
run merge qlora_merge "$M" "$w/train/ckpt/ckpt-00000004" "$w/merged.gguf"
rc=0
"$U/merge_check" "$M" "$w/merged.gguf" "$w/train/ckpt/ckpt-00000004" train.jsonl \
  "$P/work/oracle/T11/prompts.jsonl" "$O/expected.jsonl" "$w" > "$R/ours.tsv" 2>> "$R/errors.txt" || rc=$?
codes="$codes check $rc"
# libtorch's ROCm agent warns about the hidden GPUs on stderr; count the rest
grep -v -e 'agent.cpp' -e '^(null): No such file or directory$' \
  "$R/errors.txt" > "$R/errors-other.txt" || true
for id in $ids; do
  line=$(awk -F'\t' -v id="ITa-T18-$id" '$1 == id' "$R/ours.tsv")
  if [ -z "$line" ]; then
    printf 'ITa-T18-%s\tNG\t-\t-\tno output from the driver\n' "$id" >> "$R/summary.tsv"
    continue
  fi
  printf '%s\n' "$line" | awk -F'\t' -v OFS='\t' '{ r = ($2 + 0 <= $3 + 0) ? "OK" : "NG"; print $1, r, $2, $3, $4 }' >> "$R/summary.tsv"
done
want=$(cat "$O/merged.sha256")
got=$( { sha256sum "$w/merged.gguf" 2>/dev/null || echo none; } | cut -d' ' -f1)
if [ "$got" = "$want" ]; then v=0; r=OK; else v=1; r=NG; fi
printf 'ITa-T18-006\t%s\t%s\t0\tmerged.gguf against the prototype'"'"'s (sha256 %s): 0 when byte-identical\n' \
  "$r" "$v" "${want:0:12}…" >> "$R/summary.tsv"
printf 'driver\texit codes:%s; %s other stderr lines (%s)\n' \
  "$codes" "$(wc -l < "$R/errors-other.txt")" "$R/errors-other.txt" >> "$R/summary.tsv"
rm -rf "$w"
cat "$R/summary.tsv"
