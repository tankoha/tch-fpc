#!/usr/bin/env bash
# ITa of T20 (written by the harness owner; agents run it, they do not change it).
# Builds src/qlora_eval.pas and src/qlora_verify.pas against src/ and runs
# qlora_eval twice, each from a fresh directory outside the project, on the
# fixed records of tests/ita/T20: tasks.jsonl (eleven Pascal tasks in four
# dialects and three regression tasks, made by make-tasks.bb) and the
# replies of the rehearsal model before (qlora-m2-base) and after (qlora-m2)
# the M2 merge through Ollama (replies-before.jsonl, replies-after.jsonl,
# recorded once by the harness owner with ask.bb: two seeds, thinking on
# and off):
#   qlora_eval --speed-ratio 0.9 tasks.jsonl replies-before.jsonl replies-after.jsonl out
# then tests/ita/T20/check.bb (cases 001-005), which verifies every reply
# again with qlora_verify and recomputes the report. Writes
#   work/ita/T20/summary.tsv: case, OK|NG, value, limit, note
# (OK when value <= limit), then the exit codes and the count of stderr
# lines. Takes a few minutes (every reply is compiled three times).
# Usage: tests/ita/T20-run.sh
# T20_RESULTS, T20_UNITS, T20_EXTRA_FU are for the harness owner's own check
# against a reference unit; agents leave them unset. With T20_EXTRA_FU the
# program is built from a copy in T20_UNITS: FPC looks for units next to
# the main source first, so a build in src/ would take src/'s units.
set -euo pipefail
P=$(cd "$(dirname "$0")/../.." && pwd)
T=$P/tests/ita/T20
ids="001 002 003 004 005"
R=${T20_RESULTS:-$P/work/ita/T20}
U=${T20_UNITS:-$P/work/bin/T20}
mkdir -p "$R" "$U"
rm -rf "$R/summary.tsv" "$R/ours.tsv" "$R/errors.txt" "$R/errors-other.txt" "$R/out1" "$R/out2"
progs="$P/src/qlora_eval.pas $P/src/qlora_verify.pas"
if [ -n "${T20_EXTRA_FU:-}" ]; then
  cp "$P/src/qlora_eval.pas" "$P/src/qlora_verify.pas" "$U/"
  progs="$U/qlora_eval.pas $U/qlora_verify.pas"
fi
for prog in $progs; do
  name=$(basename "$prog" .pas)
  if ! fpc -B -Mobjfpc -Sh -O2 -vew -l- ${T20_EXTRA_FU:+-Fu"$T20_EXTRA_FU"} -Fu"$P/src" \
       -FU"$U" "$prog" -o"$U/$name" > "$R/build-$name.log" 2>&1; then
    echo "build failed: $R/build-$name.log" >&2
    exit 1
  fi
done
: > "$R/errors.txt"
codes=""
for n in 1 2; do
  w=$(mktemp -d)
  cp "$T/tasks.jsonl" "$T/replies-before.jsonl" "$T/replies-after.jsonl" "$w/"
  rc=0
  (cd "$w" && "$U/qlora_eval" --speed-ratio 0.9 tasks.jsonl replies-before.jsonl replies-after.jsonl out) \
    >> "$R/runs-out.txt" 2>> "$R/errors.txt" || rc=$?
  codes="$codes run$n $rc"
  rm -rf "$R/out$n"
  cp -r "$w/out" "$R/out$n" 2>/dev/null || mkdir -p "$R/out$n"
  rm -rf "$w"
done
w=$(mktemp -d)
rc=0
bb "$T/check.bb" "$T/tasks.jsonl" "$T/replies-before.jsonl" "$T/replies-after.jsonl" "$R/out1" "$R/out2" \
  "$U/qlora_verify" "$w" 0.9 > "$R/ours.tsv" 2>> "$R/errors.txt" || rc=$?
codes="$codes check $rc"
rm -rf "$w"
cp "$R/errors.txt" "$R/errors-other.txt"
for id in $ids; do
  line=$(awk -F'\t' -v id="ITa-T20-$id" '$1 == id' "$R/ours.tsv")
  if [ -z "$line" ]; then
    printf 'ITa-T20-%s\tNG\t-\t-\tno output from the driver\n' "$id" >> "$R/summary.tsv"
    continue
  fi
  printf '%s\n' "$line" | awk -F'\t' -v OFS='\t' '{ r = ($2 + 0 <= $3 + 0) ? "OK" : "NG"; print $1, r, $2, $3, $4 }' >> "$R/summary.tsv"
done
printf 'driver\texit codes:%s; %s stderr lines (%s)\n' \
  "$codes" "$(wc -l < "$R/errors-other.txt")" "$R/errors-other.txt" >> "$R/summary.tsv"
cat "$R/summary.tsv"
