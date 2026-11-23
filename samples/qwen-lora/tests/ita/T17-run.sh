#!/usr/bin/env bash
# ITa of T17 (written by the harness owner; agents run it, they do not change it).
# Builds src/qlora_train.pas and tests/ita/T17/train_check.pas against src/
# and runs on the CPU from a fresh directory outside the project, with the
# rehearsal model and tests/ita/T17/{train,valid}.jsonl (8 and 4 short
# records), each run of qlora_train its own process:
#   ovf   --valid, 8 epochs of one update (lr 1e-3, warm-up 1)
#   full  2 epochs, budget 256 (5 updates per epoch), a checkpoint every 3
#   part  the same with --max-updates 3 (inside epoch 0), then --resume
#   nan   lr 1e30, budget 256 (several updates per epoch), 2 epochs
# then train_check (cases 001-006). Writes
#   work/ita/T17/summary.tsv: case, OK|NG, value, limit, note
# (OK when value <= limit), then the exit codes and the count of stderr
# lines other than the GPU warnings. Takes about 6.5 minutes.
# Usage: QLORA_MODEL=<absolute path of the rehearsal GGUF> tests/ita/T17-run.sh
# T17_RESULTS, T17_UNITS, T17_EXTRA_FU are for the harness owner's own check
# against a reference unit; agents leave them unset. With T17_EXTRA_FU the
# program is built from a copy in T17_UNITS: FPC looks for units next to the
# main source first, so a build in src/ would take src/'s units.
set -euo pipefail
P=$(cd "$(dirname "$0")/../.." && pwd)
M=${QLORA_MODEL:?set QLORA_MODEL to the rehearsal GGUF (absolute path)}
root=$(cd "$P/../.." && pwd)
lib=${TCH_LIBTCH:-$root/build/libtch}
[ -d "$lib" ] || lib=$root/build/libtch-rocm
export HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
ids="001 002 003 004 005 006"
R=${T17_RESULTS:-$P/work/ita/T17}
U=${T17_UNITS:-$P/work/bin/T17}
mkdir -p "$R" "$U"
rm -rf "$R/summary.tsv" "$R/ours.tsv" "$R/errors.txt" "$R/errors-other.txt" "$R/runs"
prog_train=$P/src/qlora_train.pas
if [ -n "${T17_EXTRA_FU:-}" ]; then
  cp "$prog_train" "$U/qlora_train.pas"
  prog_train=$U/qlora_train.pas
fi
for prog in "$prog_train" "$P/tests/ita/T17/train_check.pas"; do
  name=$(basename "$prog" .pas)
  if ! fpc -B -Mobjfpc -Sh -O2 -vew -l- ${T17_EXTRA_FU:+-Fu"$T17_EXTRA_FU"} -Fu"$P/src" -Fu"$root/src" \
       -FU"$U" -Fl"$lib" -k-rpath="$lib" "$prog" -o"$U/$name" > "$R/build-$name.log" 2>&1; then
    echo "build failed: $R/build-$name.log" >&2
    exit 1
  fi
done
w=$(mktemp -d)
cd "$w"
cp "$P/tests/ita/T17/train.jsonl" "$P/tests/ita/T17/valid.jsonl" .
mkdir -p "$R/runs"
: > "$R/errors.txt"
codes=""
run() {  # name, then the arguments of qlora_train
  local n=$1; shift
  local rc=0
  "$U/qlora_train" "$@" >> "$R/runs/out.txt" 2>> "$R/errors.txt" || rc=$?
  codes="$codes $n $rc"
  last=$rc
}
common="--warmup 1 --lr 1e-3 --batches-per-update 1 --ckpt-every 100"
run ovf --valid valid.jsonl --epochs 8 $common "$M" train.jsonl "$R/runs/ovf"
small="--epochs 2 --warmup 1 --lr 1e-3 --budget 256 --batches-per-update 1 --ckpt-every 3"
run full $small "$M" train.jsonl "$R/runs/full"
run part1 $small --max-updates 3 "$M" train.jsonl "$R/runs/part"
run part2 $small --resume "$M" train.jsonl "$R/runs/part"
run nan --epochs 2 --warmup 1 --lr 1e30 --budget 256 --batches-per-update 1 --ckpt-every 100 "$M" train.jsonl "$R/runs/nan"
mkdir -p "$R/runs/nan"
echo "$last" > "$R/runs/nan/exit"
rc=0
"$U/train_check" "$M" train.jsonl valid.jsonl "$R/runs" > "$R/ours.tsv" 2>> "$R/errors.txt" || rc=$?
codes="$codes check $rc"
# libtorch's ROCm agent warns about the hidden GPUs on stderr; the nan run's
# message is expected; count the rest
grep -v -e 'agent.cpp' -e '^(null): No such file or directory$' -e '^non-finite loss at update 2; stopped without updating$' \
  "$R/errors.txt" > "$R/errors-other.txt" || true
for id in $ids; do
  line=$(awk -F'\t' -v id="ITa-T17-$id" '$1 == id' "$R/ours.tsv")
  if [ -z "$line" ]; then
    printf 'ITa-T17-%s\tNG\t-\t-\tno output from the driver\n' "$id" >> "$R/summary.tsv"
    continue
  fi
  printf '%s\n' "$line" | awk -F'\t' -v OFS='\t' '{ r = ($2 + 0 <= $3 + 0) ? "OK" : "NG"; print $1, r, $2, $3, $4 }' >> "$R/summary.tsv"
done
printf 'driver\texit codes:%s; %s other stderr lines (%s)\n' \
  "$codes" "$(wc -l < "$R/errors-other.txt")" "$R/errors-other.txt" >> "$R/summary.tsv"
rm -rf "$w" "$R/runs/ovf/ckpt" "$R/runs/full/ckpt" "$R/runs/part/ckpt" "$R/runs/nan/ckpt"
cat "$R/summary.tsv"
