#!/usr/bin/env bash
# ITa of T21 (written by the harness owner; agents run it, they do not change it).
# Builds src/qlora_logprobs.pas and src/qlora_train.pas against src/ and runs
# them on the CPU with the rehearsal model, each run its own process:
#   lp32       qlora_logprobs --top 20 (default kind) on the T11 prompts
#   lp16a/b    the same with --top 100 --compute bf16, twice
#   tr32       qlora_train (default kind) on tests/ita/T17/train.jsonl, 4
#              epochs of one update, lr 1e-3, warm-up 1, a checkpoint each
#   tr16       the same with --compute bf16
#   mix        the same with --max-updates 2 (fp32), then --compute bf16
#              --resume
# then tests/ita/T21/check.bb (cases 001-006) against the oracle of the code
# before T21 (work/oracle/T21: lp-fp32.jsonl, train-fp32.tsv, made by the
# harness owner with make-oracle.sh). Writes
#   work/ita/T21/summary.tsv: case, OK|NG, value, limit, note
# (OK when value <= limit), then the exit codes and the count of stderr
# lines other than the GPU warnings. Takes about 9 minutes.
# Usage: QLORA_MODEL=<absolute path of the rehearsal GGUF> tests/ita/T21-run.sh
# T21_RESULTS, T21_UNITS, T21_EXTRA_FU are for the harness owner's own check
# against a reference unit; agents leave them unset. With T21_EXTRA_FU the
# programs are built from copies in T21_UNITS: FPC looks for units next to
# the main source first, so a build in src/ would take src/'s units.
set -euo pipefail
P=$(cd "$(dirname "$0")/../.." && pwd)
M=${QLORA_MODEL:?set QLORA_MODEL to the rehearsal GGUF (absolute path)}
root=$(cd "$P/../.." && pwd)
lib=${TCH_LIBTCH:-$root/build/libtch}
[ -d "$lib" ] || lib=$root/build/libtch-rocm
export HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
ids="001 002 003 004 005 006"
R=${T21_RESULTS:-$P/work/ita/T21}
U=${T21_UNITS:-$P/work/bin/T21}
O=$P/work/oracle/T21
PR=$P/work/oracle/T11/prompts.jsonl
TR=$P/tests/ita/T17/train.jsonl
mkdir -p "$R" "$U"
rm -rf "$R/summary.tsv" "$R/ours.tsv" "$R/errors.txt" "$R/errors-other.txt" "$R/runs"
for f in "$O/lp-fp32.jsonl" "$O/train-fp32.tsv" "$PR"; do
  [ -f "$f" ] || { echo "missing $f (the harness owner's oracle)" >&2; exit 1; }
done
progs="$P/src/qlora_logprobs.pas $P/src/qlora_train.pas"
if [ -n "${T21_EXTRA_FU:-}" ]; then
  cp $progs "$U/"
  progs="$U/qlora_logprobs.pas $U/qlora_train.pas"
fi
for prog in $progs; do
  name=$(basename "$prog" .pas)
  if ! fpc -B -Mobjfpc -Sh -O2 -vew -l- ${T21_EXTRA_FU:+-Fu"$T21_EXTRA_FU"} -Fu"$P/src" -Fu"$root/src" \
       -FU"$U" -Fl"$lib" -k-rpath="$lib" "$prog" -o"$U/$name" > "$R/build-$name.log" 2>&1; then
    echo "build failed: $R/build-$name.log" >&2
    exit 1
  fi
done
: > "$R/errors.txt"
mkdir -p "$R/runs"
codes=""
run() {  # name, program, arguments...
  local n=$1 p=$2; shift 2
  local rc=0
  "$U/$p" "$@" > "$R/runs/$n.out" 2>> "$R/errors.txt" || rc=$?
  codes="$codes $n $rc"
}
run lp32 qlora_logprobs --top 20 "$M" "$PR"
run lp16a qlora_logprobs --top 100 --compute bf16 "$M" "$PR"
run lp16b qlora_logprobs --top 100 --compute bf16 "$M" "$PR"
C=(--epochs 4 --warmup 1 --lr 1e-3 --batches-per-update 1000 --ckpt-every 1)
run tr32 qlora_train "${C[@]}" "$M" "$TR" "$R/runs/tr32"
run tr16 qlora_train "${C[@]}" --compute bf16 "$M" "$TR" "$R/runs/tr16"
run mix1 qlora_train "${C[@]}" --max-updates 2 "$M" "$TR" "$R/runs/mix"
run mix2 qlora_train "${C[@]}" --compute bf16 --resume "$M" "$TR" "$R/runs/mix"
rc=0
bb "$P/tests/ita/T21/check.bb" "$O" "$R/runs" > "$R/ours.tsv" 2>> "$R/errors.txt" || rc=$?
codes="$codes check $rc"
grep -v -e 'agent.cpp' -e '^(null): No such file or directory$' "$R/errors.txt" > "$R/errors-other.txt" || true
for id in $ids; do
  line=$(awk -F'\t' -v id="ITa-T21-$id" '$1 == id' "$R/ours.tsv")
  if [ -z "$line" ]; then
    printf 'ITa-T21-%s\tNG\t-\t-\tno output from the driver\n' "$id" >> "$R/summary.tsv"
    continue
  fi
  printf '%s\n' "$line" | awk -F'\t' -v OFS='\t' '{ r = ($2 + 0 <= $3 + 0) ? "OK" : "NG"; print $1, r, $2, $3, $4 }' >> "$R/summary.tsv"
done
printf 'driver\texit codes:%s; %s other stderr lines (%s)\n' \
  "$codes" "$(wc -l < "$R/errors-other.txt")" "$R/errors-other.txt" >> "$R/summary.tsv"
cat "$R/summary.tsv"
