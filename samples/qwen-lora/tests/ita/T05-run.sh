#!/usr/bin/env bash
# ITa of T05, H46 (written by the harness owner; agents run it, they do not change it).
# Builds src/qlora_tokenize.pas and tests/ita/T05/roundtrip_check.pas, runs
# them from a fresh directory outside the project on the corpus of
# work/oracle/T05 (Pascal sources of five dialects, English and Japanese
# paragraphs, special tokens, boundary lines, 2000 random strings) and
# compares the token ids with the bundled llama-server's /tokenize
# (expected.tsv, made once by the harness owner). Writes
#   work/ita/T05/summary.tsv: case, OK|NG, diff lines, lines ours, lines oracle, note
# with the compared files and the diffs next to it (ITa-T05-NNN.*).
# Usage: QLORA_MODEL=<absolute path of the rehearsal GGUF> tests/ita/T05-run.sh
# Takes about half a minute. T05_RESULTS, T05_UNITS, T05_EXTRA_FU, T05_PROG_DIR are for
# the harness owner's own check against a reference unit; agents leave them unset.
set -euo pipefail
P=$(cd "$(dirname "$0")/../.." && pwd)
M=${QLORA_MODEL:?set QLORA_MODEL to the rehearsal GGUF (absolute path)}
O=$P/work/oracle/T05
R=${T05_RESULTS:-$P/work/ita/T05}
U=${T05_UNITS:-$P/work/bin/T05}
mkdir -p "$R" "$U"
rm -f "$R"/ITa-T05-*.* "$R/summary.tsv"
for prog in "${T05_PROG_DIR:-$P/src}/qlora_tokenize.pas" "$P/tests/ita/T05/roundtrip_check.pas"; do
  name=$(basename "$prog" .pas)
  if ! fpc -Mobjfpc -Sh -O2 -vew -l- ${T05_EXTRA_FU:+-Fu"$T05_EXTRA_FU"} -Fu"$P/src" -FU"$U" \
       "$prog" -o"$U/$name" > "$R/build-$name.log" 2>&1; then
    echo "build failed: $R/build-$name.log" >&2
    exit 1
  fi
done
w=$(mktemp -d)
trap 'rm -rf "$w"' EXIT
cd "$w"
: > "$R/errors.txt"
: > "$R/drivers.txt"
run() { # name, command...
  local name=$1 rc=0; shift
  "$@" > "$R/$name.out" 2>> "$R/errors.txt" || rc=$?
  printf 'driver %s\texit %s\n' "$name" "$rc" >> "$R/drivers.txt"
}
check() { # case expected-lines ours oracle note
  local id=$1 want=$2 a=$3 b=$4 note=$5 d la lb r
  diff "$a" "$b" > "$R/$id.diff" || true
  d=$(wc -l < "$R/$id.diff"); la=$(wc -l < "$a"); lb=$(wc -l < "$b")
  if [ "$d" -eq 0 ] && [ "$la" -eq "$want" ] && [ "$lb" -eq "$want" ]; then r=OK; else r=NG; fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$id" "$r" "$d" "$la" "$lb" "$note" >> "$R/summary.tsv"
}
kind() { grep -E "^$2[:#]" "$1" || true; }

# the corpus is copied to a bare relative name in the fresh directory
cp "$O/corpus.jsonl" corpus.jsonl
run tokenize "$U/qlora_tokenize" --jsonl "$M" corpus.jsonl
n=1
for k in pas en ja sp edge fuzz; do
  id=$(printf 'ITa-T05-%03d' "$n")
  kind "$R/tokenize.out" "$k" > "$R/$id.ours.tsv"
  kind "$O/expected.tsv" "$k" > "$R/$id.oracle.tsv"
  want=$(wc -l < "$R/$id.oracle.tsv")
  check "$id" "$want" "$R/$id.ours.tsv" "$R/$id.oracle.tsv" "token ids equal /tokenize: $k records"
  n=$((n + 1))
done

# 007: Decode(Encode(text)) = text for every record
run roundtrip "$U/roundtrip_check" "$M" corpus.jsonl
cp "$R/roundtrip.out" "$R/ITa-T05-007.ours.tsv"
cut -f1 "$O/expected.tsv" | sed 's/$/\tOK/' > "$R/ITa-T05-007.oracle.tsv"
check ITa-T05-007 "$(wc -l < "$O/expected.tsv")" "$R/ITa-T05-007.ours.tsv" "$R/ITa-T05-007.oracle.tsv" "decode of encode gives the text back"

# 008: plain mode on a whole source file (absolute path), equal to that record's ids
src=$(awk -F'\t' '$1 ~ /pascal-fpc-objfpc\/examples\/02_generics.pas$/ {print $1}' "$O/expected.tsv")
file=${src#pas:}; file=${file/#\~/$HOME}
run plain "$U/qlora_tokenize" "$M" "$file"
cp "$R/plain.out" "$R/ITa-T05-008.ours.tsv"
awk -F'\t' -v s="$src" '$1 == s {print $2}' "$O/expected.tsv" > "$R/ITa-T05-008.oracle.tsv"
check ITa-T05-008 1 "$R/ITa-T05-008.ours.tsv" "$R/ITa-T05-008.oracle.tsv" "plain mode: one file's ids equal /tokenize"

cat "$R/drivers.txt" >> "$R/summary.tsv"
printf 'stderr\t%s lines (%s)\n' "$(wc -l < "$R/errors.txt")" "$R/errors.txt" >> "$R/summary.tsv"
cat "$R/summary.tsv"
