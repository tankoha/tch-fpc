# Building

## libtch (once per libtorch variant)
The C wrapper in `csrc/libtch` is built with CMake against an unpacked
libtorch (CPU: `libtorch-shared-with-deps-2.13.0+cpu.zip` from
download.pytorch.org):

```bash
cmake -S csrc/libtch -B build/libtch -DCMAKE_PREFIX_PATH="$PWD/libtorch" -DCMAKE_BUILD_TYPE=Release
cmake --build build/libtch -j "$(nproc)"
```

For ROCm or CUDA use another directory (`build/libtch-rocm`); DESIGN 4.8
has the variant without the vendor's CMake files (`TCH_LIBTORCH_DIR`).

## A program
`scripts/build-example.sh <file.pas> [<libtch dir>]` runs

```bash
fpc -Mobjfpc -Sh -vew -l- -Fu<checkout>/src -FU<units> -Fl<libtch> -k-rpath=<libtch> <file.pas> -o<out>/<name>
```

- `TCH_FPC_ROOT`: the checkout (default: the one the script is in; set it
  when the skill is installed as a plugin).
- `TCH_OUT`: where the executable goes (default `build/examples`).
  `TCH_UNITS`: where compiled units go (default `build/skill-units`); the
  first build compiles `Tch` (about 50,000 lines, a few seconds), later
  builds reuse it.
- The executable finds `libtch.so` and libtorch through the rpath. Do not
  set `LD_LIBRARY_PATH` globally: it overrides rpaths and can pick another
  libtorch (CPU vs ROCm) without an error until something differs.

## Your own project
Same flags: `-Fu` to the checkout's `src`, `-Fl` and `-k-rpath=` to the
libtch build directory, `{$mode objfpc}{$H+}` in every unit. Lazarus:
the same paths in Project Options, and the rpath as a linker option.

## Tests of the repository
`tests/mock/run.sh` and `tests/compiler/run.sh` need no libtorch; the
Pascal tests in `tests/*/` carry their build line in the header comment;
`tests/skill/check-examples.sh` checks this skill (`LIBTCH=<libtch dir>`).

Source: DESIGN 4.5, 4.8; `.github/workflows/ci.yml`.
