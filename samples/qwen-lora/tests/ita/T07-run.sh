#!/usr/bin/env bash
# ITa of T07 (written by the harness owner; agents run it, they do not change it).
# Builds tests/ita/T07/chat_check.pas against src/, runs it once from a fresh
# directory outside the project on the conversations of work/oracle/T07
# (corpus.jsonl) and compares with the bundled llama-server's
# /apply-template and /tokenize under the 27B model's chat template
# (expected.jsonl, made once by the harness owner). Writes
#   work/ita/T07/summary.tsv: case, OK|NG, value, limit, note
# (the value is a count of records that differ; OK when value <= limit),
# then the driver's exit code and stderr line count; details.txt has one
# line per record. Takes a few seconds.
# Usage: QLORA_MODEL=<absolute path of the rehearsal GGUF> tests/ita/T07-run.sh
# T07_RESULTS, T07_UNITS, T07_EXTRA_FU are for the harness owner's own check
# against a reference unit; agents leave them unset.
set -euo pipefail
P=$(cd "$(dirname "$0")/../.." && pwd)
M=${QLORA_MODEL:?set QLORA_MODEL to the rehearsal GGUF (absolute path)}
O=$P/work/oracle/T07
R=${T07_RESULTS:-$P/work/ita/T07}
U=${T07_UNITS:-$P/work/bin/T07}
B=$U/chat_check
mkdir -p "$R" "$U"
rm -f "$R/summary.tsv" "$R/ours.tsv" "$R/details.txt" "$R/errors.txt"
if ! fpc -B -Mobjfpc -Sh -O2 -vew -l- ${T07_EXTRA_FU:+-Fu"$T07_EXTRA_FU"} -Fu"$P/src" \
     -FU"$U" "$P/tests/ita/T07/chat_check.pas" -o"$B" > "$R/build.log" 2>&1; then
  echo "build failed: $R/build.log" >&2
  exit 1
fi
w=$(mktemp -d)
trap 'rm -rf "$w"' EXIT
cd "$w"
cp "$O/corpus.jsonl" "$O/expected.jsonl" .
rc=0
"$B" "$M" corpus.jsonl expected.jsonl > "$R/ours.tsv" 2> "$R/errors.txt" || rc=$?
grep '^# ' "$R/ours.tsv" > "$R/details.txt" || true
for id in 001 002 003 004; do
  line=$(awk -F'\t' -v id="ITa-T07-$id" '$1 == id' "$R/ours.tsv")
  if [ -z "$line" ]; then
    printf 'ITa-T07-%s\tNG\t-\t-\tno output from the driver\n' "$id" >> "$R/summary.tsv"
    continue
  fi
  printf '%s\n' "$line" | awk -F'\t' -v OFS='\t' '{ r = ($2 + 0 <= $3 + 0) ? "OK" : "NG"; print $1, r, $2, $3, $4 }' >> "$R/summary.tsv"
done
printf 'driver\texit %s, %s stderr lines (%s)\n' "$rc" "$(wc -l < "$R/errors.txt")" "$R/errors.txt" >> "$R/summary.tsv"
cat "$R/summary.tsv"
