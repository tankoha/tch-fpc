#!/usr/bin/env bash
# ITa of T19 (written by the harness owner; agents run it, they do not change it).
# Builds src/qlora_dataset.pas against src/ and runs it twice, each from a
# fresh directory outside the project, on the synthetic raw set
# tests/ita/T19/raw.jsonl (16 good records in four dialects, 17 with
# planted faults; made by make-raw.bb), with the evaluation tasks
# eval.jsonl, the example directory examples/, the tokenizer reference
# ref-tokens.jsonl (the bundled llama-server's /tokenize, one record
# altered; made by make-ref.bb) and the rehearsal model's tokenizer:
#   qlora_dataset --model M --ref-tokens ... --eval ... --leak-dir ...
#                 --valid-percent 25 raw.jsonl out
# then tests/ita/T19/check.bb (cases 001-006). Writes
#   work/ita/T19/summary.tsv: case, OK|NG, value, limit, note
# (OK when value <= limit), then the exit codes and the count of stderr
# lines other than the GPU warnings. Takes under a minute.
# Usage: QLORA_MODEL=<absolute path of the rehearsal GGUF> tests/ita/T19-run.sh
# T19_RESULTS, T19_UNITS, T19_EXTRA_FU are for the harness owner's own check
# against a reference unit; agents leave them unset. With T19_EXTRA_FU the
# program is built from a copy in T19_UNITS: FPC looks for units next to
# the main source first, so a build in src/ would take src/'s units.
set -euo pipefail
P=$(cd "$(dirname "$0")/../.." && pwd)
M=${QLORA_MODEL:?set QLORA_MODEL to the rehearsal GGUF (absolute path)}
root=$(cd "$P/../.." && pwd)
lib=${TCH_LIBTCH:-$root/build/libtch}
[ -d "$lib" ] || lib=$root/build/libtch-rocm
export HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
T=$P/tests/ita/T19
ids="001 002 003 004 005 006"
R=${T19_RESULTS:-$P/work/ita/T19}
U=${T19_UNITS:-$P/work/bin/T19}
mkdir -p "$R" "$U"
rm -rf "$R/summary.tsv" "$R/ours.tsv" "$R/errors.txt" "$R/errors-other.txt" "$R/out1" "$R/out2"
prog=$P/src/qlora_dataset.pas
if [ -n "${T19_EXTRA_FU:-}" ]; then
  cp "$prog" "$U/qlora_dataset.pas"
  prog=$U/qlora_dataset.pas
fi
if ! fpc -B -Mobjfpc -Sh -O2 -vew -l- ${T19_EXTRA_FU:+-Fu"$T19_EXTRA_FU"} -Fu"$P/src" -Fu"$root/src" \
     -FU"$U" -Fl"$lib" -k-rpath="$lib" "$prog" -o"$U/qlora_dataset" > "$R/build-qlora_dataset.log" 2>&1; then
  echo "build failed: $R/build-qlora_dataset.log" >&2
  exit 1
fi
: > "$R/errors.txt"
codes=""
for n in 1 2; do
  w=$(mktemp -d)
  cp "$T/raw.jsonl" "$T/eval.jsonl" "$T/ref-tokens.jsonl" "$w/"
  cp -r "$T/examples" "$w/examples"
  rc=0
  (cd "$w" && "$U/qlora_dataset" --model "$M" --ref-tokens ref-tokens.jsonl --eval eval.jsonl \
     --leak-dir examples --valid-percent 25 raw.jsonl out) >> "$R/runs-out.txt" 2>> "$R/errors.txt" || rc=$?
  codes="$codes run$n $rc"
  rm -rf "$R/out$n"
  cp -r "$w/out" "$R/out$n" 2>/dev/null || mkdir -p "$R/out$n"
  rm -rf "$w"
done
rc=0
bb "$T/check.bb" "$T/raw.jsonl" "$T/expected.tsv" "$R/out1" "$R/out2" > "$R/ours.tsv" 2>> "$R/errors.txt" || rc=$?
codes="$codes check $rc"
# libtorch's ROCm agent warns about the hidden GPUs on stderr; count the rest
grep -v -e 'agent.cpp' -e '^(null): No such file or directory$' \
  "$R/errors.txt" > "$R/errors-other.txt" || true
for id in $ids; do
  line=$(awk -F'\t' -v id="ITa-T19-$id" '$1 == id' "$R/ours.tsv")
  if [ -z "$line" ]; then
    printf 'ITa-T19-%s\tNG\t-\t-\tno output from the driver\n' "$id" >> "$R/summary.tsv"
    continue
  fi
  printf '%s\n' "$line" | awk -F'\t' -v OFS='\t' '{ r = ($2 + 0 <= $3 + 0) ? "OK" : "NG"; print $1, r, $2, $3, $4 }' >> "$R/summary.tsv"
done
printf 'driver\texit codes:%s; %s other stderr lines (%s)\n' \
  "$codes" "$(wc -l < "$R/errors-other.txt")" "$R/errors-other.txt" >> "$R/summary.tsv"
cat "$R/summary.tsv"
