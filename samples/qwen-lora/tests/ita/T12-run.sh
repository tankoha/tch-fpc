#!/usr/bin/env bash
# ITa of T12 (written by the harness owner; agents run it, they do not change it).
# Builds tests/ita/T12/lora_check.pas against src/ and runs it once on the CPU
# from a fresh directory outside the project: LoRA rank 16, scale 2 (DESIGN
# 8.13) on the rehearsal model, against oracles that do not use the code
# under test for what they check (the tensor directory; a GGUF with the
# merged matrices W + s*B*A computed in Double; one backward through the
# whole graph). Writes
#   work/ita/T12/summary.tsv: case, OK|NG, value, limit, note
# (OK when value <= limit), then the exit code and the count of stderr lines
# other than the GPU warnings. Takes about a minute and needs about 1 GB in
# work/ita/T12 for a moment (the merged GGUF, deleted afterwards).
# Usage: QLORA_MODEL=<absolute path of the rehearsal GGUF> tests/ita/T12-run.sh
# T12_RESULTS, T12_UNITS, T12_EXTRA_FU are for the harness owner's own check
# against a reference unit; agents leave them unset.
set -euo pipefail
P=$(cd "$(dirname "$0")/../.." && pwd)
M=${QLORA_MODEL:?set QLORA_MODEL to the rehearsal GGUF (absolute path)}
root=$(cd "$P/../.." && pwd)
lib=${TCH_LIBTCH:-$root/build/libtch}
[ -d "$lib" ] || lib=$root/build/libtch-rocm
export HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
ids="001 002 003 004 005"
R=${T12_RESULTS:-$P/work/ita/T12}
U=${T12_UNITS:-$P/work/bin/T12}
mkdir -p "$R" "$U"
rm -f "$R/summary.tsv" "$R/ours.tsv" "$R/errors.txt" "$R/errors-other.txt"
src=$P/tests/ita/T12/lora_check.pas
if ! fpc -B -Mobjfpc -Sh -O2 -vew -l- ${T12_EXTRA_FU:+-Fu"$T12_EXTRA_FU"} -Fu"$P/src" -Fu"$root/src" \
     -FU"$U" -Fl"$lib" -k-rpath="$lib" "$src" -o"$U/lora_check" > "$R/build-lora_check.log" 2>&1; then
  echo "build failed: $R/build-lora_check.log" >&2
  exit 1
fi
w=$(mktemp -d)
cd "$w"
cp "$P/work/oracle/T11/prompts.jsonl" .
rc=0
"$U/lora_check" "$M" prompts.jsonl "$R" > "$R/ours.tsv" 2> "$R/errors.txt" || rc=$?
# libtorch's ROCm agent warns about the hidden GPUs on stderr; count the rest
grep -v -e 'agent.cpp' -e '^(null): No such file or directory$' "$R/errors.txt" > "$R/errors-other.txt" || true
for id in $ids; do
  line=$(awk -F'\t' -v id="ITa-T12-$id" '$1 == id' "$R/ours.tsv")
  if [ -z "$line" ]; then
    printf 'ITa-T12-%s\tNG\t-\t-\tno output from the driver\n' "$id" >> "$R/summary.tsv"
    continue
  fi
  printf '%s\n' "$line" | awk -F'\t' -v OFS='\t' '{ r = ($2 + 0 <= $3 + 0) ? "OK" : "NG"; print $1, r, $2, $3, $4 }' >> "$R/summary.tsv"
done
printf 'driver\tlora_check exit %s, %s other stderr lines (%s)\n' \
  "$rc" "$(wc -l < "$R/errors-other.txt")" "$R/errors-other.txt" >> "$R/summary.tsv"
rm -rf "$w"
cat "$R/summary.tsv"
