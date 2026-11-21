#!/usr/bin/env bash
# ITa of T04 (written by the harness owner; agents run it, they do not change it).
# Builds tests/ita/T04/writer_check.pas against src/, runs it from a fresh
# directory outside the project and compares with the oracle of
# work/oracle/T04 (upstream ggml's own GGUF code and the T02 C reference):
#   work/ita/T04/summary.tsv: case, OK|NG, diff lines, lines ours, lines oracle, note
# with the compared files and the diffs next to it (ITa-T04-NNN.*).
# Usage: QLORA_MODEL=<absolute path of the rehearsal GGUF> tests/ita/T04-run.sh
# The projector file is taken from the same directory (mmproj-BF16.gguf).
# Takes about a minute; the written files (up to 0.6 GB) are deleted.
# T04_RESULTS, T04_UNITS, T04_EXTRA_FU are for the harness owner's own
# check against a reference unit; agents leave them unset.
set -euo pipefail
P=$(cd "$(dirname "$0")/../.." && pwd)
M=${QLORA_MODEL:?set QLORA_MODEL to the rehearsal GGUF (absolute path)}
MM=$(dirname "$M")/mmproj-BF16.gguf
O=$P/work/oracle/T04
R=${T04_RESULTS:-$P/work/ita/T04}
U=${T04_UNITS:-$P/work/bin/T04}
B=$U/writer_check
mkdir -p "$R" "$U"
rm -f "$R"/ITa-T04-*.* "$R/summary.tsv"
if ! fpc -Mobjfpc -Sh -O2 -vew -l- ${T04_EXTRA_FU:+-Fu"$T04_EXTRA_FU"} -Fu"$P/src" -FU"$U" \
     "$P/tests/ita/T04/writer_check.pas" -o"$B" > "$R/build.log" 2>&1; then
  echo "build failed: $R/build.log" >&2
  exit 1
fi
w=$(mktemp -d)
trap 'rm -rf "$w"' EXIT
cd "$w"
: > "$R/errors.txt"

expect() { awk -F'\t' -v c="$1" -v n="$2" '$1 == c && (n == "" || $4 == n) {print $2 "\t" $3}' "$O/expected.tsv"; }
digest() { if [ -f "$1" ]; then printf '%s\t%s\n' "$(sha256sum < "$1" | cut -d' ' -f1)" "$(stat -c %s "$1")"; else echo missing; fi; }
check() { # case expected-lines ours oracle note
  local id=$1 want=$2 a=$3 b=$4 note=$5 d la lb r
  diff "$a" "$b" > "$R/$id.diff" || true
  d=$(wc -l < "$R/$id.diff"); la=$(wc -l < "$a"); lb=$(wc -l < "$b")
  if [ "$d" -eq 0 ] && [ "$la" -eq "$want" ] && [ "$lb" -eq "$want" ]; then r=OK; else r=NG; fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$id" "$r" "$d" "$la" "$lb" "$note" >> "$R/summary.tsv"
}
run() { # name, args...: driver exit code goes to summary as a note line
  local name=$1 rc=0; shift
  "$B" "$@" > "$R/$name.layout.tsv" 2>> "$R/errors.txt" || rc=$?
  printf 'driver %s\texit %s\n' "$name" "$rc" >> "$R/drivers.txt"
}
: > "$R/drivers.txt"
nt=$(awk -F'\t' '$1 ~ /^[0-9]+$/' "$O/retype-layout.tsv" | wc -l)

# 001/002: identity copy of the rehearsal model (relative destination name)
run copy-model copy "$M" model-copy.gguf
"$O/gguf_meta" "$M" "$w/orig.meta" > "$R/ITa-T04-002.oracle.tsv"
cp "$R/copy-model.layout.tsv" "$R/ITa-T04-002.ours.tsv"
digest model-copy.gguf > "$R/ITa-T04-001.ours.tsv"
expect identity "$(basename "$M")" > "$R/ITa-T04-001.oracle.tsv"
rm -f model-copy.gguf
check ITa-T04-001 1 "$R/ITa-T04-001.ours.tsv" "$R/ITa-T04-001.oracle.tsv" "identity copy of the model: sha256 and size"
check ITa-T04-002 $((nt + 2)) "$R/ITa-T04-002.ours.tsv" "$R/ITa-T04-002.oracle.tsv" "planned layout of the copy equals upstream"

# 003: identity copy of the projector (absolute destination path)
run copy-mmproj copy "$MM" "$w/mmproj-copy.gguf"
digest "$w/mmproj-copy.gguf" > "$R/ITa-T04-003.ours.tsv"
expect identity "$(basename "$MM")" > "$R/ITa-T04-003.oracle.tsv"
rm -f "$w/mmproj-copy.gguf"
check ITa-T04-003 1 "$R/ITa-T04-003.ours.tsv" "$R/ITa-T04-003.oracle.tsv" "identity copy of the projector: sha256 and size"

# 004-006: 78 attention-side tensors retyped to Q8_0
run retype retype "$M" retyped.gguf "$O/retype-names.txt"
cp "$R/retype.layout.tsv" "$R/ITa-T04-004.ours.tsv"
cp "$O/retype-layout.tsv" "$R/ITa-T04-004.oracle.tsv"
check ITa-T04-004 $((nt + 2)) "$R/ITa-T04-004.ours.tsv" "$R/ITa-T04-004.oracle.tsv" "planned layout after retyping equals upstream"
digest retyped.gguf > "$R/ITa-T04-005.ours.tsv"
expect retype "" > "$R/ITa-T04-005.oracle.tsv"
check ITa-T04-005 1 "$R/ITa-T04-005.ours.tsv" "$R/ITa-T04-005.oracle.tsv" "retyped file: sha256 and size"
# upstream reads the written file back: its layout is the planned one
if [ -f retyped.gguf ]; then
  "$O/gguf_meta" retyped.gguf "$w/back.meta" > "$R/ITa-T04-006.ours.tsv" 2>> "$R/errors.txt" || true
else
  : > "$R/ITa-T04-006.ours.tsv"
fi
cp "$O/retype-layout.tsv" "$R/ITa-T04-006.oracle.tsv"
rm -f retyped.gguf
check ITa-T04-006 $((nt + 2)) "$R/ITa-T04-006.ours.tsv" "$R/ITa-T04-006.oracle.tsv" "upstream reads the retyped file back with that layout"

cat "$R/drivers.txt" >> "$R/summary.tsv"
printf 'stderr\t%s lines (%s)\n' "$(wc -l < "$R/errors.txt")" "$R/errors.txt" >> "$R/summary.tsv"
cat "$R/summary.tsv"
