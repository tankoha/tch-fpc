#!/usr/bin/env bash
# ITa of T22 (written by the harness owner; agents run it, they do not change it).
# Builds src/qlora_verify.pas against src/ and verifies the Oxygene files of
# tests/ita/T22/files (four correct programs and planted faults, made by
# make-files.bb) with --dialect oxygene: EBuild in the Elements flatpak,
# then dotnet. Then runs the T19 ITa (tests/ita/T19-run.sh, the rehearsal
# model) with the same src/, whose raw record x-oxygene is now compiled,
# run and rejected for its planted wrong output. tests/ita/T22/check.bb
# gives cases 001-006. Writes
#   work/ita/T22/summary.tsv: case, OK|NG, value, limit, note
# (OK when value <= limit), then the exit codes and the count of stderr
# lines other than the GPU warnings. Takes about a minute.
# Usage: QLORA_MODEL=<absolute path of the rehearsal GGUF> tests/ita/T22-run.sh
# T22_RESULTS, T22_UNITS, T22_EXTRA_FU are for the harness owner's own check
# against a reference unit; agents leave them unset (T22_EXTRA_FU builds
# qlora_verify from a copy in T22_UNITS and is passed on to the T19 run as
# T19_EXTRA_FU).
set -euo pipefail
P=$(cd "$(dirname "$0")/../.." && pwd)
M=${QLORA_MODEL:?set QLORA_MODEL to the rehearsal GGUF (absolute path)}
root=$(cd "$P/../.." && pwd)
export HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
ids="001 002 003 004 005 006"
R=${T22_RESULTS:-$P/work/ita/T22}
U=${T22_UNITS:-$P/work/bin/T22}
O=$HOME/.cache/qlora-verify
mkdir -p "$R" "$U" "$O"
rm -rf "$R/summary.tsv" "$R/ours.tsv" "$R/errors.txt" "$R/errors-other.txt" "$R/verify.tsv" "$R/t19"
prog=$P/src/qlora_verify.pas
if [ -n "${T22_EXTRA_FU:-}" ]; then
  cp "$prog" "$U/qlora_verify.pas"
  prog=$U/qlora_verify.pas
fi
if ! fpc -B -Mobjfpc -Sh -O2 -vew -l- ${T22_EXTRA_FU:+-Fu"$T22_EXTRA_FU"} -Fu"$P/src" -Fu"$root/src" \
     -FU"$U" "$prog" -o"$U/qlora_verify" > "$R/build-qlora_verify.log" 2>&1; then
  echo "build failed: $R/build-qlora_verify.log" >&2
  exit 1
fi
: > "$R/errors.txt"
codes=""
before=$(ls "$O" | wc -l)
w=$(mktemp -d)
s=$(date +%s)
rc=0
(cd "$w" && "$U/qlora_verify" --dialect oxygene "$P/tests/ita/T22/files") > "$R/verify.tsv" 2>> "$R/errors.txt" || rc=$?
secs=$(( $(date +%s) - s ))
codes="$codes verify $rc"
after=$(ls "$O" | wc -l)
rm -rf "$w"
rc=0
env T19_RESULTS="$R/t19" T19_UNITS="$R/t19-units" ${T22_EXTRA_FU:+T19_EXTRA_FU="$T22_EXTRA_FU"} \
  QLORA_MODEL=$M bash "$P/tests/ita/T19-run.sh" > "$R/t19-run.txt" 2>> "$R/errors.txt" || rc=$?
codes="$codes t19 $rc"
rc=0
bb "$P/tests/ita/T22/check.bb" "$P/tests/ita/T22/expected.tsv" "$R/verify.tsv" "$((after - before))" \
  "$secs" "$R/t19/summary.tsv" "$R/t19/out1/rejects.jsonl" "$HOME" > "$R/ours.tsv" 2>> "$R/errors.txt" || rc=$?
codes="$codes check $rc"
grep -v -e 'agent.cpp' -e '^(null): No such file or directory$' "$R/errors.txt" > "$R/errors-other.txt" || true
for id in $ids; do
  line=$(awk -F'\t' -v id="ITa-T22-$id" '$1 == id' "$R/ours.tsv")
  if [ -z "$line" ]; then
    printf 'ITa-T22-%s\tNG\t-\t-\tno output from the driver\n' "$id" >> "$R/summary.tsv"
    continue
  fi
  printf '%s\n' "$line" | awk -F'\t' -v OFS='\t' '{ r = ($2 + 0 <= $3 + 0) ? "OK" : "NG"; print $1, r, $2, $3, $4 }' >> "$R/summary.tsv"
done
printf 'driver\texit codes:%s; %s other stderr lines (%s)\n' \
  "$codes" "$(wc -l < "$R/errors-other.txt")" "$R/errors-other.txt" >> "$R/summary.tsv"
cat "$R/summary.tsv"
