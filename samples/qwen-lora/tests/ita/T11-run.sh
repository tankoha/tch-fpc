#!/usr/bin/env bash
# ITa of T11 (written by the harness owner; agents run it, they do not change it).
# Builds src/qlora_logprobs.pas and tests/ita/T11/logprobs_check.pas, runs
# qlora_logprobs --top 50 once from a fresh directory outside the project on
# the prompts of work/oracle/T11 (24 token prompts of 1-600 tokens, 3 text
# prompts) on the CPU, and compares with the reference made once by the
# harness owner (expected.jsonl: the bundled llama-server on an all-F32 copy
# of the rehearsal GGUF with an f32 KV cache; DESIGN 8.11 "H43 in detail").
# Writes
#   work/ita/T11/summary.tsv: case, OK|NG, value, limit, note
# (OK when value <= limit), then the exit codes and the count of stderr
# lines other than the GPU warnings. Takes about a minute.
# Usage: QLORA_MODEL=<absolute path of the rehearsal GGUF> tests/ita/T11-run.sh
#        QLORA_MODEL=... tests/ita/T11-run.sh gpu   harness owner only: case
#                       101 (--split 12 on cuda:0 / cuda:1 against the CPU)
#                       into work/ita/T11-gpu
# T11_RESULTS, T11_UNITS, T11_EXTRA_FU are for the harness owner's own check
# against a reference unit; agents leave them unset.
set -euo pipefail
P=$(cd "$(dirname "$0")/../.." && pwd)
M=${QLORA_MODEL:?set QLORA_MODEL to the rehearsal GGUF (absolute path)}
root=$(cd "$P/../.." && pwd)
lib=${TCH_LIBTCH:-$root/build/libtch}
[ -d "$lib" ] || lib=$root/build/libtch-rocm
O=$P/work/oracle/T11
mode=${1:-cpu}
if [ "$mode" = gpu ]; then
  ids="101"
  R=${T11_RESULTS:-$P/work/ita/T11-gpu}
else
  export HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
  ids="001 002 003 004"
  R=${T11_RESULTS:-$P/work/ita/T11}
fi
U=${T11_UNITS:-$P/work/bin/T11}
mkdir -p "$R" "$U"
rm -f "$R/summary.tsv" "$R/ours.tsv" "$R/ours.jsonl" "$R/errors.txt" "$R/cli-errors.txt"
for src in "$P/src/qlora_logprobs.pas" "$P/tests/ita/T11/logprobs_check.pas"; do
  b=$U/$(basename "$src" .pas)
  if ! fpc -B -Mobjfpc -Sh -O2 -vew -l- ${T11_EXTRA_FU:+-Fu"$T11_EXTRA_FU"} -Fu"$P/src" -Fu"$root/src" \
       -FU"$U" -Fl"$lib" -k-rpath="$lib" "$src" -o"$b" > "$R/build-$(basename "$src" .pas).log" 2>&1; then
    echo "build failed: $R/build-$(basename "$src" .pas).log" >&2
    exit 1
  fi
done
w=$(mktemp -d)
cd "$w"
cp "$O/prompts.jsonl" "$O/expected.jsonl" .
crc=0
rc=0
if [ "$mode" = gpu ]; then
  # the CPU run's output is the reference for case 101
  HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 "$U/qlora_logprobs" --top 50 "$M" prompts.jsonl > cpu.jsonl 2> cpu-errors.txt || crc=$?
  "$U/qlora_logprobs" --top 50 --split 12 "$M" prompts.jsonl > "$R/ours.jsonl" 2> "$R/cli-errors.txt" || crc=$?
  "$U/logprobs_check" "$M" "$R/ours.jsonl" cpu.jsonl prompts.jsonl gpu > "$R/ours.tsv" 2> "$R/errors.txt" || rc=$?
else
  "$U/qlora_logprobs" --top 50 "$M" prompts.jsonl > "$R/ours.jsonl" 2> "$R/cli-errors.txt" || crc=$?
  "$U/logprobs_check" "$M" "$R/ours.jsonl" expected.jsonl prompts.jsonl > "$R/ours.tsv" 2> "$R/errors.txt" || rc=$?
fi
# libtorch's ROCm agent warns about the hidden GPUs on stderr; count the rest
cat "$R/cli-errors.txt" "$R/errors.txt" | grep -v -e 'agent.cpp' -e '^(null): No such file or directory$' > "$R/errors-other.txt" || true
for id in $ids; do
  line=$(awk -F'\t' -v id="ITa-T11-$id" '$1 == id' "$R/ours.tsv")
  if [ -z "$line" ]; then
    printf 'ITa-T11-%s\tNG\t-\t-\tno output from the driver\n' "$id" >> "$R/summary.tsv"
    continue
  fi
  printf '%s\n' "$line" | awk -F'\t' -v OFS='\t' '{ r = ($2 + 0 <= $3 + 0) ? "OK" : "NG"; print $1, r, $2, $3, $4 }' >> "$R/summary.tsv"
done
printf 'driver\tqlora_logprobs exit %s, logprobs_check exit %s, %s other stderr lines (%s)\n' \
  "$crc" "$rc" "$(wc -l < "$R/errors-other.txt")" "$R/errors-other.txt" >> "$R/summary.tsv"
rm -rf "$w"
cat "$R/summary.tsv"
