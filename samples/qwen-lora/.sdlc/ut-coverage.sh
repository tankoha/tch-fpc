#!/usr/bin/env bash
# :coverage-cmd of this project (run from the project root by sdlc.bb).
# Builds every tests/ut/*_test.pas with line information, runs each under
# kcov (lines of src/ only), merges the runs and writes coverage/lcov.info.
# Test failures do not stop the measurement; they are recorded in
# test-results/ut.md. A build failure exits non-zero.
#   TCH_LIBTCH  libtch build directory (default: ../../build/libtch, else
#               ../../build/libtch-rocm)
set -euo pipefail
shopt -s nullglob
proj=$(pwd)
root=$(cd ../.. && pwd)
lib=${TCH_LIBTCH:-$root/build/libtch}
[ -d "$lib" ] || lib=$root/build/libtch-rocm
export HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
work=$proj/work/ut
rm -rf "$work" coverage
mkdir -p "$work/units" "$work/bin" "$work/kcov" coverage
runs=()
for t in tests/ut/*_test.pas; do
  name=$(basename "$t" .pas)
  fpc -Mobjfpc -Sh -gl -gw -O- -vew -l- -Fu"$proj/src" -Fu"$root/src" -Fu"$proj/tests/ut" \
    -FU"$work/units" -Fl"$lib" -k-rpath="$lib" "$t" -o"$work/bin/$name" \
    > "$work/$name.build.log" 2>&1 || { cat "$work/$name.build.log"; exit 1; }
  kcov --include-path="$proj/src" "$work/kcov/$name" "$work/bin/$name" --all --format=plain \
    > "$work/$name.run.log" 2>&1 || echo "$name: tests failed (see work/ut/$name.run.log)"
  runs+=("$work/kcov/$name")
done
[ ${#runs[@]} -gt 0 ] || { echo "no tests/ut/*_test.pas"; exit 1; }
kcov --merge "$work/kcov/merged" "${runs[@]}" > /dev/null
bb "$proj/.sdlc/cobertura2lcov.bb" "$work/kcov/merged/kcov-merged/cobertura.xml" > coverage/lcov.info
