#!/usr/bin/env bash
# Build one tch-fpc program: the build line the skill documents and the one
# tests/skill/check-examples.sh uses, so the two cannot drift apart.
#
# Usage: build-example.sh <file.pas> [<libtch build dir>]
#   TCH_FPC_ROOT  tch-fpc checkout (default: the checkout this script is in)
#   TCH_OUT       directory for the executable (default: build/examples)
#   TCH_UNITS     directory for compiled units (default: build/skill-units)
# The libtch build dir defaults to $TCH_FPC_ROOT/build/libtch. The
# executable finds libtch through an rpath, so no LD_LIBRARY_PATH is needed.
# Prints the path of the executable on success.
set -euo pipefail
src=${1:?usage: build-example.sh <file.pas> [<libtch build dir>]}
here=$(cd "$(dirname "$0")" && pwd)
root=${TCH_FPC_ROOT:-$(cd "$here/../../../.." && pwd)}
if [ ! -f "$root/src/Tch.pas" ]; then
  echo "build-example.sh: no tch-fpc checkout at $root; set TCH_FPC_ROOT" >&2
  exit 2
fi
libtch=$(cd "${2:-$root/build/libtch}" && pwd)
out=${TCH_OUT:-$root/build/examples}
units=${TCH_UNITS:-$root/build/skill-units}
mkdir -p "$out" "$units"
name=$(basename "${src%.*}")
fpc -Mobjfpc -Sh -vew -l- -Fu"$root/src" -FU"$units" -Fl"$libtch" \
  -k-rpath="$libtch" "$src" -o"$out/$name" >&2
echo "$out/$name"
