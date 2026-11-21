#!/usr/bin/env bash
# ITa of T03 (written by the harness owner; agents run it, they do not change it).
# Builds tests/ita/T03/quantt_check.pas against src/, runs it once on the
# rehearsal model from a fresh directory outside the project, compares the
# sha256 of every tensor with the upstream C oracle (work/oracle/T02, the same as T02) and writes
#   work/ita/T03/summary.tsv: case, OK|NG, diff lines, lines ours, lines oracle, note
# with the compared files and the diffs next to it (ITa-T03-NNN.*).
# Usage: QLORA_MODEL=<absolute path of the rehearsal GGUF> tests/ita/T03-run.sh
# Takes about a minute. T03_RESULTS, T03_UNITS, T03_EXTRA_FU are for the harness
# owner's own check against a reference unit; agents leave them unset.
set -euo pipefail
P=$(cd "$(dirname "$0")/../.." && pwd)
M=${QLORA_MODEL:?set QLORA_MODEL to the rehearsal GGUF (absolute path)}
O=$P/work/oracle/T02
O1=$P/work/oracle/T01
root=$(cd "$P/../.." && pwd)
lib=${TCH_LIBTCH:-$root/build/libtch}
[ -d "$lib" ] || lib=$root/build/libtch-rocm
# the CPU only (as the UT): bit equality is what is checked here
export HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
R=${T03_RESULTS:-$P/work/ita/T03}
U=${T03_UNITS:-$P/work/bin/T03}
B=$U/quantt_check
mkdir -p "$R" "$U"
rm -f "$R"/ITa-T03-*.* "$R/summary.tsv"
if ! fpc -Mobjfpc -Sh -O2 -vew -l- ${T03_EXTRA_FU:+-Fu"$T03_EXTRA_FU"} -Fu"$P/src" -Fu"$P/../../src" -FU"$U" -Fl"$lib" -k-rpath="$lib" \
     "$P/tests/ita/T03/quantt_check.pas" -o"$B" > "$R/build.log" 2>&1; then
  echo "build failed: $R/build.log" >&2
  exit 1
fi
# the driver's SHA-256 must agree with sha256sum
if [ "$("$B" --selftest)" != "$(printf abc | sha256sum | cut -d' ' -f1)" ]; then
  echo "quant_check --selftest disagrees with sha256sum" >&2
  exit 1
fi
w=$(mktemp -d)
cd "$w"
rc=0
"$B" "$M" > "$R/ours.tsv" 2> "$R/errors.txt" || rc=$?
awk -F'\t' -v OFS='\t' '$1 == "deq" {print $2, $3, $4, $5}' "$R/ours.tsv" > "$R/deq.tsv"
awk -F'\t' -v OFS='\t' '$1 == "q8" {print $2, $3, $4, $5}' "$R/ours.tsv" > "$R/q8.tsv"

# the q8_0 tensors' own bytes in the file (independent of the code under test)
data=$(awk -F'\t' '$1 == "data_offset" {print $2}' "$O1/summary.txt")
: > "$R/q8_0-file.tsv"
while IFS=$'\t' read -r name type dims off bytes; do
  [ "$type" = q8_0 ] || continue
  n=1; for d in $dims; do n=$((n * d)); done
  s=$(dd if="$M" bs=1M iflag=skip_bytes,count_bytes skip=$((data + off)) count="$bytes" status=none | sha256sum | cut -d' ' -f1)
  printf '%s\t%s\t%s\t%s\n' "$name" "$type" "$n" "$s" >> "$R/q8_0-file.tsv"
done < "$O1/tensors.tsv"

check() { # case expected-lines ours oracle note
  local id=$1 want=$2 a=$3 b=$4 note=$5 d la lb r
  diff "$a" "$b" > "$R/$id.diff" || true
  d=$(wc -l < "$R/$id.diff"); la=$(wc -l < "$a"); lb=$(wc -l < "$b")
  if [ "$d" -eq 0 ] && [ "$la" -eq "$want" ] && [ "$lb" -eq "$want" ]; then r=OK; else r=NG; fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$id" "$r" "$d" "$la" "$lb" "$note" >> "$R/summary.tsv"
}
bytype() { awk -F'\t' -v t="$2" '$2 == t' "$1"; }
i=1
for t in f32:133 q8_0:36 q4_K:98 q5_K:36 q6_K:17; do
  id=ITa-T03-00$i
  bytype "$R/deq.tsv" "${t%%:*}" > "$R/$id.ours.tsv"
  bytype "$O/dequant-sha256.tsv" "${t%%:*}" > "$R/$id.oracle.tsv"
  check "$id" "${t##*:}" "$R/$id.ours.tsv" "$R/$id.oracle.tsv" "dequantise ${t%%:*}"
  i=$((i + 1))
done
check ITa-T03-006 284 "$R/q8.tsv" "$O/q8_0-sha256.tsv" "dequantise, then quantise to Q8_0"
bytype "$R/q8.tsv" q8_0 > "$R/ITa-T03-007.ours.tsv"
check ITa-T03-007 36 "$R/ITa-T03-007.ours.tsv" "$R/q8_0-file.tsv" "q8_0 re-quantised equals its bytes in the file"
printf 'driver\texit %s, %s stderr lines (%s)\n' "$rc" "$(wc -l < "$R/errors.txt")" "$R/errors.txt" >> "$R/summary.tsv"
cat "$R/summary.tsv"
