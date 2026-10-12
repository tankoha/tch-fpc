#!/bin/bash
# Builds and runs tests/gpu/gpu_test.pas against build/libtch-<variant>
# (DESIGN.md 4.8). Variant rocm (default), cuda, or cpu (build/libtch:
# prints SKIP). Only the discrete GPU is made visible, so CudaDeviceCount
# is 1 (the Raphael iGPU, gfx1036, is unsupported). Run from the repo root
# after building the variant's libtch.
set -euo pipefail
v=${1:-rocm}
case "$v" in
  cpu) lib=build/libtch ;;
  rocm|cuda) lib=build/libtch-$v ;;
  *) echo "usage: $0 [rocm|cuda|cpu]" >&2; exit 2 ;;
esac
out=build/gpu-$v
mkdir -p "$out"
fpc -Mobjfpc -Sh -Fusrc -FU"$out" -Fl"$lib" -k-rpath="$PWD/$lib" \
  tests/gpu/gpu_test.pas -o"$out/gpu_test" > "$out/build.log" 2>&1 \
  || { grep -E 'Error|Fatal' "$out/build.log"; exit 1; }
HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 "$out/gpu_test"
