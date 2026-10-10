#!/bin/bash
# Builds the mock libtch (tests/mock/mock_tch.c + generated stubs) as
# build/mock/libtch.so and runs tests/mock/mock_test.pas,
# nn_mock_test.pas and jit_mock_test.pas against it. Needs gcc and fpc only; no
# libtorch. Run from the repo root.
set -euo pipefail
out=build/mock
mkdir -p "$out"
tests/mock/gen_stubs.sh csrc/libtch/torch_api_generated.h tests/mock/mock_tch.c "$out/mock_atg.c"
gcc -std=c11 -Wall -Wno-unused-parameter -shared -fPIC -Icsrc/libtch \
  tests/mock/mock_tch.c "$out/mock_atg.c" -o "$out/libtch.so"
for t in mock_test nn_mock_test jit_mock_test; do
  fpc -Mobjfpc -Sh -Fusrc -FU"$out" -Fl"$out" -k-rpath="$PWD/$out" \
    tests/mock/$t.pas -o"$out/$t" > "$out/build.log" 2>&1 \
    || { grep -E 'Error|Fatal' "$out/build.log"; exit 1; }
  "$out/$t"
done
