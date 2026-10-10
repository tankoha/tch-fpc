# Handoff

Japanese version: [HANDOFF_jp.md](HANDOFF_jp.md)

Read this first when resuming. Decisions and rationale: [DESIGN.md](DESIGN.md).

## State (2026-10-10, end of cloud design session 2)

- Design for all three layers written in DESIGN.md. Nothing built yet.
- Reference: tch-rs `4227b89b72059b0651ff83a38637693574e0a2d6`,
  libtorch 2.13.0.
- Decided: fork tch-rs libtch (keep C names, harden `PROTECT`, guard
  `extern "C"`), shared `libtch.so`, `TTensor` = record wrapping a refcounted
  interface, exceptions via `ETorchError`/`ETorchNativeError` after every
  call, nil-handle check before every non-optional tensor argument,
  snake_case generated ops (`&`-escaped where reserved) + CamelCase
  hand-written API, `out` parameters for multi-output ops, generated default
  parameters for trailing simple arguments, `TOptIntList` for
  `IntListOption`, Pascal emitter fed by a manifest from a forked gen.ml.
- License: MIT OR Apache-2.0, same as tch-rs (`LICENSE-MIT`,
  `LICENSE-APACHE` at the repo root). Vendored tch-rs files keep tch-rs's own
  license copies next to them.
- OCaml: `gen/gen.ml` is vendored as source (part of the fork). The OCaml
  compiler and opam packages (`yaml`, `base`, `stdio`, `dune`) are installed
  on the local machine to run it; they are a maintainer tool, needed only
  when regenerating for a new libtorch, never to build tch-fpc.
- Session 2 reviewed section 4 against tch-rs's sources and added 4.6 (nn /
  VarStore / optimizers / JIT as classes + `TVarPath` record, single
  `Forward(X, Train = False)`) and 4.7 (grad mode). Unverified FPC 3.2.2
  behaviour is marked **(H1)**–**(H11)** in DESIGN.md and listed below.

## Cloud session 3 (2026-10-10): parameter-name rule revised

- **Parameter names vs members, revised (H12).** The `_` suffix rule from
  local session 1 is replaced by `{$modeswitch duplicatelocals}` in
  `Tch.pas`; generated parameter and `out` names stay verbatim (`dim`,
  `indices`, `pow`), `NumDims` is `Dim` again, and reserved words use the
  same `&` escape as op names (`&out`, `&end`, `&to`); only `result`
  (3 ops) becomes `result_`. Measured on FPC 3.2.2 in the cloud container:
  `tests/compiler/h12_duplicatelocals.pas` and three `xfail_h12_*.pas`;
  the whole `tests/compiler/run.sh` passes. Details and the compiler-source
  check in DESIGN.md 4.4; README (en/jp) updated.
- Changed code: `src/Tch.pas` (switch, `Dim`, stand-ins use `dim`),
  `tests/ownership/ownership_test.pas` (`V.Dim`). The ownership test needs
  libtorch and was **not** re-run in the cloud; run it locally once
  (`Tch.pas` itself compiles).
- Still open from the hand-back below: items 1 (ops.tsv schema, tchgen.pas
  output per op shape) and 2 (error message format, `TKind` 27-45,
  `TORCH_WARN`, default-argument scope, `FromArray` Int32).

## Handing back to the cloud (2026-10-10, end of local session 1)

Steps 1-7 are done and every hypothesis H1-H11 is measured (results in
"Local progress" below and next to each (Hn) in DESIGN.md). The OCaml
toolchain for step 8 is installed locally. The user chose to design step 8
in the cloud. Asked of the next cloud session:

1. **Design the generator (step 8)** so that local work can be mechanical:
   - exact `ops.tsv` schema emitted by gen.ml (columns, escaping, how
     defaults, optional/list kinds, receiver and result shape are encoded);
   - `gen/tchgen.pas` structure and the exact Pascal it emits for each op
     shape (fixed-N, dynamic, scalar, nothing), with the stand-ins in
     `src/Tch.pas` (`add`, `cat`, `max_dim`, `ones`) as the reference
     bodies that already work;
   - naming: the parameter/out-name rule (DESIGN 4.4; revised in cloud
     session 3, see above), the `&` escapes, and the op-name check against
     our hand-written members (see the tch-rs notes under "Parameter names
     vs `TTensor` members").
2. **Decide the open questions the generator touches:** native error
   message format (`Message` vs `Detail`), `TKind` 27-45, `TORCH_WARN`
   handling (patch `at_autocast_*` and/or a warning handler), default
   arguments scope, and dropping the `array of Int32` `FromArray`
   overload (H2).
3. Update DESIGN/HANDOFF (en + jp) at the end, as usual.

Local state the design can rely on: `src/TchSys.pas`, `src/Tch.pas`
(minimal `TTensor`, `Dim`, `TchLiveHandles`), tests under `tests/`,
build commands in each test's header and in "Local progress".

## Local progress (2026-10-10)

- **Step 1 done.** FPC 3.2.2, g++ 15.2.0, CMake 4.2.3 (reads libtorch's
  CMake config fine; "kineto not found" warning is harmless), libtorch
  2.13.0+cpu (cxx11 ABI) in `libtorch/`. OCaml not installed yet (step 8).
- **Step 2 done.** `d81f48b` pristine import, `a15806e` fork patch
  (`git diff d81f48b a15806e` shows the patch). Build:
  `cmake -S csrc/libtch -B build/libtch -DCMAKE_PREFIX_PATH=$PWD/libtorch
  -DCMAKE_BUILD_TYPE=Release && cmake --build build/libtch -j 8` (~30 s).
  2657 `atg_*` exported, `ldd -r` clean. gen.ml patched too, so the
  `extern "C"` guard survives regeneration.
- **Step 3 done.** `tests/c/smoke.c` passes (build command in its header).
  Found while doing it: the C branch of `torch_api.h` needed `<stddef.h>`
  (`size_t`) and `<stdbool.h>` (`bool`, before C23); added to the patch.
  An error message is libtorch's text followed by a C++ origin/backtrace
  starting at `Exception raised from ...` (see open questions).

- **Step 4 done.** `tests/abi/abi_test.pas` passes (build command in its
  header; `build/abi_test nilarg` runs the nil-argument case). Measured:
  - C99 `bool` returns: declare as `Boolean`. FPC tests only `%al`
    (`testb %al,%al`), which is all the SysV ABI guarantees. **Do not use
    `LongBool` or ctypes' `cbool`** (a `LongBool`): FPC then tests the whole
    `%eax` (`orl %eax,%eax`). It happened to read correctly here because
    g++ left the upper bits zero, but that is not guaranteed. `Boolean`
    arguments (`at_autocast_set_enabled`) also work.
  - `c10::ScalarType` 0-26 match `TKind` exactly, Float8 23-26 included.
    libtorch 2.13 also has **27-45** (UInt16/32/64, UInt1-7, Int1-7,
    Float8_e8m0fnu, Float4_e2m1fn_x2); `atg_empty` creates all of them on
    CPU and `at_scalar_type` round-trips (see open questions).
  - `uint8_t x_null` as `Byte` and `int8_t` layout as `ShortInt` (incl.
    -1 = nullopt) behave as intended (`atg_argmax`, `atg_to_sparse`).
  - `external 'tch'` + `-Flbuild/libtch -k-rpath=$PWD/build/libtch` finds
    `libtch.so` at run time; libtorch is found through libtch's RUNPATH.
  - nil tensor to a non-optional argument (`atg_add(@S, nil, A)`): FPC's
    SIGSEGV handler turns it into a catchable `EAccessViolation`, raised
    from inside C++ frames, and the process keeps running. That looks
    recoverable but is not safe (no C++ unwinding, libtorch state
    unknown), so the nil-handle check before every call stays necessary.
  - libtorch writes `TORCH_WARN` messages straight to stderr (e.g. the
    deprecated autocast API on every `at_autocast_*` call, quantized and
    ComplexHalf creation). See open questions.

- **Step 5 done.** `tests/fpu/fpu_test.pas` + `tests/fpu/run.sh` (one
  process per mode, 30 s timeout). FPC default MXCSR=$1900, x87CW=$1372;
  after `SetExceptionMask(all)` $1F80/$137F.
  - Unmasked, 4 elements: `EZeroDivide` raised from inside C++ frames.
  - Unmasked, 10^7 elements: unhandled exceptions in the OpenMP workers,
    then a **hang** (and corrupted Pascal state on the main thread).
  - Masked first: 1/0 gives +Inf for both sizes.
  - Pool created unmasked, masked afterwards: still hangs. Confirms that
    the mask belongs in `Tch`'s `initialization`.
  - **(H4) resolved:** a `TThread` started after the mask is masked too;
    small and parallel ops in it give Inf.
  - **(H5) resolved:** the mask survives a caught integer `EDivByZero`.
  - DESIGN.md 4.2 updated accordingly.

- **Step 6 done.** First code: `src/TchSys.pas` (hand-written externals,
  plus stand-ins for the few `atg_*` the tests need) and `src/Tch.pas`
  (`TTensor`/`TTensorHandle`, `TchCheck`, nil-argument check, FPU mask in
  `initialization`, `TchLiveHandles` counter; stand-in ops `add`, `mul`,
  `cat`, `max_dim`, `ones` in the generator's shape). Test:
  `tests/ownership/ownership_test.pas`, all passing.
  - `B := A; A := Default(TTensor)` keeps B alive; `A := A + B` correct.
  - 10^6 x `D := (A + B) * C`: handle count and RSS flat (~1.5 s total).
  - **(H1) resolved:** temporaries live until their slot is reused or the
    routine exits, not until the end of the statement.
  - **(H6) resolved:** `cat([A, B], 0)` and `cat` of temporaries: correct,
    no leak, no double free.
  - **(H8) resolved:** receiver passed as `out` gives `ETorchError`; a
    previous value in an `out` variable is released.
  - New problem found (see open questions): generated parameter names
    collide with `TTensor` members (`dim` vs `Dim`). The stand-ins used
    `dim_` until cloud session 3 (H12).

- **Step 7 done.** `tests/compiler/run.sh` (h*.pas must compile and run,
  xfail_*.pas must be rejected, plus a generated big record). All as
  expected; details recorded in DESIGN.md next to each hypothesis.
  - **(H2)** `[1.0, 2.0]` picks `Single`; `[1, 2]` is ambiguous with both
    `Int32` and `Int64` overloads → drop the `Int32` overload.
  - **(H3)** scalar operators resolve as intended; `**` and unary `-` ok.
  - **(H7)** all escaped names compile and are callable.
  - **(H9)** repeat the default in overrides (else `Derived.Forward(X)`
    does not compile).
  - **(H10)** compiles; `Items` is shared on copy (no copy-on-write).
  - **(H11)** guard local released at routine exit; a discarded
    `NoGradGuard;` is released immediately (pitfall).
  - Record with 2,657 methods via `{$I}` (with a managed field): compiles
    in about 1 s (-O2: 1.6 s); a `default` indexed property gives `T[5][2]`.

- **OCaml toolchain ready (step 8 prerequisite).** `ocaml` 5.4.0 and
  `opam` 2.5.0 from apt (`yaml` is not packaged in Ubuntu, hence opam).
  `opam init --bare --no-setup` (shell rc files untouched), switch
  `tch-fpc` on `ocaml-system`, `opam install dune base stdio yaml`. Use
  `eval $(opam env --switch=tch-fpc)` before building; then
  `cd gen && dune build ./gen.exe` builds warning-free. Run from a tch-rs
  checkout at 4227b89b (it reads
  `third_party/pytorch/Declarations-v2.13.0.yaml` and writes upstream
  paths), the patched gen.exe reproduces `csrc/libtch/torch_api_generated.{h,cpp}`
  byte for byte (2657 functions, 0.5 s).

## Next steps (local implementation)

1. **Toolchain check.** FPC 3.2.2, g++, CMake, libtorch 2.13.0 CPU
   (cxx11 ABI) unpacked into `libtorch/` (already git-ignored).
2. **Vendor libtch.** Copy from tch-rs at the commit above into
   `csrc/libtch/` and `gen/`; add LICENSE-MIT / LICENSE-APACHE. Commit the
   pristine copy first (message names the source files), then a second
   commit with our patch: remove stream/python/image code, add `tchf_free`,
   add `catch (...)` to `PROTECT`, wrap `extern "C"` in
   `torch_api_generated.h` with `#ifdef __cplusplus`. Build `libtch.so` with
   CMake. The YAML (`third_party/pytorch/Declarations-v2.13.0.yaml`) is in
   the tch-rs repo at that commit but may be missing from a shallow/partial
   clone; fetch it only when regenerating.
3. **Smoke test in C** (tiny C program, not Python): create a tensor,
   `atg_add`, print, trigger an error and read `get_and_reset_last_err`.
   Needs the `extern "C"` guard from step 2 to include the generated header.
4. **ABI verification in Pascal** — measure, do not assume:
   - `bool` returns (e.g. `at_autocast_is_enabled`) read correctly as
     `Boolean`.
   - `c10::ScalarType` numeric values match `TKind`, including the Float8
     kinds 23–26 (print `at_scalar_type` for tensors of each kind).
   - `uint8_t x_null` option flags as `Byte`, `int8_t` as `ShortInt`.
   - Linking: `external 'tch'` + rpath so the program finds `libtch.so` and
     libtorch at run time.
   - Passing a `nullptr` tensor to a non-optional argument really is a
     segfault (confirms the need for the nil-handle check).
5. **FPU mask tests.**
   - Without `SetExceptionMask`, does `1.0 / 0` on a CPU tensor crash an FPC
     program? Confirm the fix in `Tch` initialization.
   - **(H4)** After `SetExceptionMask` in the main thread, does a `TThread`
     started afterwards see the mask, or the FPC defaults? Decides whether
     `TchMaskFpuExceptions` must be called in every thread.
   - **(H5)** Raise and catch an `EZeroDivide` in Pascal code, then run a
     tensor op that produces NaN. Does the mask survive the RTL's SIGFPE
     handling?
6. **Ownership tests.** Loop 10^6 times over `(A + B) * C` and check RSS is
   flat; check `B := A` then `A := Default(TTensor)` keeps B alive.
   - **(H1)** Instrument `TTensorHandle.Destroy` and determine when the
     temporaries of `(A + B) * C` are released: end of statement, end of
     loop iteration, or end of routine.
   - **(H6)** `TTensor.cat([A, B], 0)` with an open-array constructor of
     `TTensor`: no leak, no double free.
   - **(H8)** `out` parameter semantics: `T.max_dim(1, False, T, I)` raises
     `ETorchError` (receiver finalised on entry) instead of crashing, and
     `V` passed as `out` with a previous value releases the old handle.
7. **Compiler acceptance tests** (tiny programs before the generator exists):
   - **(H2)** `FromArray([1.0, 2.0])` with overloads on `array of Single`
     and `array of Double` compiles unambiguously; same for `[1, 2]` with
     `Int32`/`Int64`. If not, use `FromSingles`/`FromDoubles`/...
   - **(H3)** Operator overloads `+(TTensor, Double)` and `+(TTensor, Int64)`
     coexist and `T + 1`, `T + 1.5`, `2 * T` all resolve; `**` can be
     overloaded for records; unary `-` too.
   - **(H7)** Record methods named `&div`, `&var`, `&to`, `&set`, `&repeat`,
     and modifiers `absolute`, `alias`, `index` (with and without `&`)
     compile and can be called as `t.&div(u)`.
   - **(H9)** `Forward(const X: TTensor; Train: Boolean = False)` as a
     virtual abstract method with the default repeated in overrides; calling
     `Model.Forward(X)` through the base and derived type.
   - **(H10)** `TIValue = record ... Items: array of TIValue; end;` compiles.
   - **(H11)** An `IInterface` local assigned from `NoGradGuard` is released
     at routine exit (and not earlier).
   - A record with ~2,650 methods pulled in via `{$I}` inside the record
     declaration compiles in reasonable time; `{$modeswitch advancedrecords}`
     properties on records if the `T[i]` sugar is wanted.
   - **(H12, resolved in cloud session 3)** `{$modeswitch duplicatelocals}`
     lets a method parameter share a member's name in `objfpc`; `&out` as
     a parameter name compiles; `result` as a function parameter and a
     local variable named like a member are still rejected.
8. **Generator.** Add `ops.tsv` output to gen.ml (exported name, kind,
   receiver, result shape, args with defaults, return names); write
   `gen/tchgen.pas`; generate `TchSysGenerated.pas` and the `.inc` files;
   compile everything. Include the case-insensitive collision check against
   hand-written names (`data`, `grad`, `detach`, `to_device`, `clone`, ...).
   Parameter and out-names need no collision handling (H12); escape
   reserved words with `&` and rename `result` to `result_` (DESIGN.md 4.4).

## Open questions

- **Native error message format.** libtorch's message is the text plus a
  multi-line C++ origin/backtrace (`Exception raised from ...`). tch-rs
  passes it through unchanged. Option for `ETorchNativeError`: `Message` =
  text before `Exception raised from`, full text in a `Detail` property.
  Decide when writing `TchRaiseNative` (step 4+).
- **`TKind` beyond 26.** libtorch 2.13 has ScalarType 27-45 and they are
  creatable on CPU (step 4). DESIGN's `TKind` stops at 26 and raises on
  unknown kinds. Decide: extend `TKind` to 45 (at least UInt16/32/64 and
  Float8_e8m0fnu look useful), or keep 0-26 and let `Kind` raise.
- **libtorch warnings.** `TORCH_WARN` goes to stderr unconditionally, and
  the upstream `at_autocast_*` functions warn on every call (deprecated
  `at::autocast::is_enabled`/`set_enabled`). Options: patch those two
  functions in the fork to the non-deprecated device-taking API, and/or
  install a `c10::WarningHandler` from libtch that forwards to a Pascal
  callback.
- **Parameter names vs `TTensor` members: decided, then revised**
  (2026-10-10, user). First decision (local session 1): append `_` until
  unique, rename `Dim` → `NumDims`. Revised in cloud session 3 (H12):
  `{$modeswitch duplicatelocals}` in `Tch.pas` removes the collision, names
  stay verbatim, `Dim` is back, reserved words get `&`, `result` →
  `result_`. Rule, measurements and the compiler-source check in DESIGN.md
  4.4.
  Why tch-rs never needed this: in Rust, parameters are plain locals and
  methods are reachable only as `self.dim()` / `Tensor::dim`, so
  `fn dim(&self)` and ~250 ops with `dim: i64` coexist. (`duplicatelocals`
  is the part of `{$mode delphi}` that behaves like Rust here.) What
  tch-rs *does* handle, and step 8 must
  check against our own names: op-name collisions in gen.ml's
  `replace_map` (`where`/`unsafe` for Rust keywords; `view`, `to_device`,
  `t` → `tr` for hand-written methods) and `prefixed_functions` (`g_`
  prefix for `add`, `mul`, `div`, `sub`, ... whose plain names go to
  hand-written versions). These only rename the Rust wrappers; the
  `atg_*` names our generator reads are unaffected. Also in README.
- **Multi-output ops**: `out` parameters (decided) — revisit whether a few
  hand-written result records (`max_dim`, `sort`, `topk`, `lstm`) are worth
  adding after writing real code. Also whether `var` instead of `out` is
  preferable if (H8) shows the aliasing hazard bites in practice.
- **Default arguments.** The generator emits defaults for trailing
  Bool/Int64/Double/String runs. How many ops benefit, and which
  hand-written convenience overloads (e.g. `TTensor.zeros([2, 3])` with
  `kFloat`/CPU) are worth adding in the first milestone.
- **Equality and containers.** `=` is deliberately not overloaded for
  `TTensor`, which rules out FGL (`TFPGList<TTensor>`). Confirm
  `Generics.Collections.TList<TTensor>` and plain `TTensorArray` cover the
  needs of `TchNN`; otherwise reconsider an identity `=`.
- **Train flag.** `Forward(X, Train)` parameter (decided) vs a PyTorch-style
  `Training` property propagated through children. Revisit once
  `TSequential` and a custom-model example exist.
- **`TIValue` representation.** Tagged record: decided, and H10 confirmed
  it compiles (step 7). Copies share `Items`; treat as immutable.
- **VarStore thread safety.** Documented as not thread-safe; add a
  `TCriticalSection` if a data-loader thread ever needs it.
- **Streams / images.** If needed later: replace tch-rs's Rust-implemented
  stream symbols with a struct of C function pointers passed in by Pascal
  (wrapping `TStream`). Images can be done in Pascal (FPImage) instead of stb.
- **Windows / macOS.** Not designed yet beyond `cdecl` and `tchf_free`.
  Windows additionally needs symbol export from `tch.dll`
  (`--export-all-symbols` or a `.def` file).
- **ROCm (deferred; CPU first).** Steps 2-7 use the CPU libtorch. The local
  GPU is an RX 9070 (Navi 48, gfx1201) plus a Raphael iGPU; no system ROCm.
  libtorch 2.13.0 for ROCm exists only as `+rocm7.1` (~5.8 GB, checked
  2026-10-10). Expected, unmeasured: the ROCm build presents HIP as CUDA
  (`cuda` device, `torch::cuda::is_available`), so the C API, `TDevice` and
  `CudaIsAvailable` stay unchanged. To check when added as `libtorch-rocm/`:
  link `torch_hip` with `-Wl,--no-as-needed` (libtch never references it
  directly); gfx1201 kernels present in the zip; iGPU enumeration
  (`HIP_VISIBLE_DEVICES`); whether the bundled deps suffice without system
  ROCm. DESIGN.md section 1 lists only `libtorch_cuda.so`.
- **Release packaging.** Generated files (`ops.tsv`, `torch_api_generated.*`,
  `TchSysGenerated.pas`, `.inc`) are committed, so every tag's source archive
  builds without OCaml. Open: whether releases also attach prebuilt
  `libtch.so`/`tch.dll` per libtorch version and platform, or users always
  build it with CMake. libtorch itself is never bundled.
- **Going public.** The license is decided (below); the repo stays private
  until the user decides to publish it.
