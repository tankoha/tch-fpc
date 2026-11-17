#!/usr/bin/env bash
# Verify the tch-fpc skill's files with the real compiler and libtorch
# (DESIGN 7.4):
#   examples/NN_name.pas : compiles, runs, stdout equals NN_name.expected
#   pitfalls/pNN_*.pas   : does not compile; the log contains the text after
#                          "expect-error:" in the file's first comment
#   traps/tNN_*.pas      : compiles, runs, stdout equals tNN_*.expected
# Usage: check-examples.sh [name...]   (default: all; a name is a file stem)
#   LIBTCH  libtch build directory (default: build/libtch)
# Programs run from the repository root, so examples can read tests/jit/models.
set -euo pipefail
shopt -s nullglob
root=$(cd "$(dirname "$0")/../.." && pwd)
skill=$root/plugin/skills/tch-fpc
build=$skill/scripts/build-example.sh
libtch=${LIBTCH:-$root/build/libtch}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
export TCH_FPC_ROOT=$root TCH_OUT=$work/bin TCH_UNITS=$work/units

wanted() {
  [ $# -eq 1 ] && return 0
  local n=$1; shift
  for w in "$@"; do [ "$w" = "$n" ] && return 0; done
  return 1
}

fail=0
run_and_compare() {   # <kind> <src>
  local kind=$1 src=$2 name exe
  name=$(basename "${src%.pas}")
  if ! exe=$("$build" "$src" "$libtch" 2> "$work/$name.compile.log"); then
    echo "NG  $kind/$name: does not compile"
    grep -E 'Error|Fatal' "$work/$name.compile.log" | sed 's/^/      /' | head -5
    fail=1; return
  fi
  if ! (cd "$root" && "$exe") > "$work/$name.out" 2> "$work/$name.err"; then
    echo "NG  $kind/$name: run failed"
    sed 's/^/      /' "$work/$name.err" | tail -5
    fail=1; return
  fi
  if ! diff -u "${src%.pas}.expected" "$work/$name.out" > "$work/$name.diff"; then
    echo "NG  $kind/$name: output differs from .expected"
    sed 's/^/      /' "$work/$name.diff"
    fail=1; return
  fi
  echo "OK  $kind/$name"
}

for src in "$skill"/examples/*.pas; do
  wanted "$(basename "${src%.pas}")" "$@" || continue
  run_and_compare examples "$src"
done
for src in "$skill"/traps/*.pas; do
  wanted "$(basename "${src%.pas}")" "$@" || continue
  run_and_compare traps "$src"
done
for src in "$skill"/pitfalls/*.pas; do
  name=$(basename "${src%.pas}")
  wanted "$name" "$@" || continue
  expect=$(sed -n 's/.*expect-error: *\(.*[^ ]\) *}.*/\1/p; s/.*expect-error: *\(.*[^ ]\) *$/\1/p' "$src" | head -1)
  if [ -z "$expect" ]; then
    echo "NG  pitfalls/$name: no expect-error: line"; fail=1; continue
  fi
  if "$build" "$src" "$libtch" > /dev/null 2> "$work/$name.compile.log"; then
    echo "NG  pitfalls/$name: compiled, but must not"; fail=1; continue
  fi
  if ! grep -qF -- "$expect" "$work/$name.compile.log"; then
    echo "NG  pitfalls/$name: compiler message lacks: $expect"
    grep -E 'Error|Fatal' "$work/$name.compile.log" | sed 's/^/      /' | head -3
    fail=1; continue
  fi
  echo "OK  pitfalls/$name"
done
exit $fail
