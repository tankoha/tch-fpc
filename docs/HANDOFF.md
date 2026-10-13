# Handoff

Japanese version: [HANDOFF_jp.md](HANDOFF_jp.md)

Read this first when resuming. Decisions and rationale: [DESIGN.md](DESIGN.md).

## Overall review (2026-10-12, after local session 8) — resume here

Three independent Opus reviewers (code vs design, en/jp documents, test
coverage; read-only, no context from the sessions) reviewed the whole
repository. Every finding was checked against the code before acting.
Fixed and committed:
- DESIGN 4.8 body and the section 6 status brought in line with the
  measured "Results" (GpuBackend rule, `at_context_has_cudart` in the L2
  list, which calls raise on a CPU build, GPU test items 1b/2/6/7/8, the
  mock bullet, the fuser flag); 4.6.10's flatten bullet points to 4.8;
  4.6.9's mock counts annotated; a status paragraph for the milestone 3
  layers; the real ROCm archive name; the autocast patch marked compiled.
- HANDOFF: forward pointers on stale older passages (H32 amended, design
  sessions via Fable settled, open questions done, ROCm done), design
  session headings 6–8 point to their local sessions; "(below)" → "(above)".
- UML status table and TchSys box; `CLAUDE.md` workflow; source comments
  in `Tch.pas`, `TchSys.pas`, `tests/jit`, `tests/gpu`.
- Tests: Embedding `PaddingIdx` checked through the gradient (the old check
  could not fail); `GpuBackend`, H33, H34 asserted in the GPU test (variant
  passed by `run.sh`); `DoubleValue` on a GPU scalar must return 6; the
  CPU-build GPU API asserted in `tests/api`; H26's dict case checks a fresh
  `Raw`; mock GPU stubs now fail where real libtorch does, plus a fuser
  flag test in `jit_mock_test`.

**Follow-up (same day): fixed and committed.**
- `TVarStore.AddEntry`: when `<name>__<Count>` is taken too, the next free
  number; all checks before the store changes (test fails on the old code
  with `EListError`). `TBatchNorm` raises unless `Nd` is 1..3, before
  registering.
- Compiler hypotheses asserted: `tests/compiler/h3`, `h10`, `h11` check
  their results and `run.sh` counts a non-zero exit; `tests/ownership`
  asserts the H1 counts; `tests/fpu/run.sh` checks every mode (masked modes
  +Inf twice, nomask-small EZeroDivide, nomask-large and pool-first must
  crash or hang).
- Untested API now tested: tracing a method (incl. a raising one),
  `CreateFromStream(S, Device)`, `TTrainableJitModule` `CreateFromStream`/
  `ForwardIV`/`Method`/`MethodIV`/`SetTrain`/`SetEval`/`Inner`, buffers;
  `NonLinearityGain`, `KaimingNormal`, the `TVarPath` helpers,
  `BackwardStepClipNorm`, transpose and layer-norm configs, `TRNN`
  properties; `SetCudnnUserEnabled`, `TchMaskFpuExceptions`.
- **Bug found by the new tests:** `NonLinearityGain(nlTanh)` was
  1.6666666269 because FPC folds `5.0 / 3.0` in Single; now `Double(5) / 3`
  (DESIGN 4.6.1).
- `tests/common/layers.inc`: the conv padding and RNN hand setups shared by
  `tests/nn` and `tests/gpu`.

**B7, 2026-10-13: handle types decided and implemented.** DESIGN 3 said
`type PTorchTensor = type Pointer;` (distinct pointer types); the code had
plain `Pointer`. *Measured* (FPC 3.2.2): `type Pointer` aliases are not
distinguished (value parameter, `var` parameter and assignment all compile),
but pointers to distinct empty records are ("Incompatible type"; `nil`, an
untyped `Pointer` and casts still compile). `TchSys.pas` now declares
`PTorchTensor`, `PTorchScalar`, `PTorchOptimizer`, `PTorchModule` and
`PTorchIValue` that way; the whole code base including `TchSysGenerated`
(about 4,700 uses) recompiled without error, so no mix-up was hidden, and a
deliberate `at_free(<a PTorchIValue>)` now fails to compile. C side and ABI
unchanged. `{$packrecords c}` is not needed and not added (no struct in
`torch_api.h`, no record crosses the C boundary); add it with the first
function that passes a record. DESIGN 3 updated.

**Still open from the review (ask the user):**
- A layer constructor that raises leaves its variables in the store
  (measured: `TConv2D` with `Groups = 0`). Roll the store back, or check
  preconditions first and document the rest: a design decision.

## Local session 8 (2026-10-12): GPU (4.8) implemented, runs on the RX 9070

**Where things stand.** 4.8 is implemented and committed; the working
tree is clean. Steps 1–10 of design session 8's local list are done. Every
design item handed back at the end of local session 4 is now implemented.

- `csrc/libtch/CMakeLists.txt`: `-DTCH_LIBTORCH_DIR=<dir>` manual-link
  mode (a `CACHE PATH`, not `option()`); default path bit-identical.
  Builds: `build/libtch` (CPU, unchanged), `build/libtch-rocm`
  (`cmake -S csrc/libtch -B build/libtch-rocm
  -DTCH_LIBTORCH_DIR=$PWD/libtorch-rocm -DCMAKE_BUILD_TYPE=Release &&
  cmake --build build/libtch-rocm -j 12`, about a minute),
  `build/libtch-manual` (manual mode on the CPU archive, a check only).
- `TchSys`: the `atc_*`, `at_context_*` (plus `at_context_has_cudart`),
  `atm_fuser_cuda_*` externals. `Tch`: `CudnnIsAvailable`,
  `CudaManualSeed(All)`, `CudaSynchronize`, `CudnnUserEnabled`/`Set…`,
  `SetCudnnBenchmark`, `TGpuBackend`/`GpuBackend`, `GpuRuntimeVersion`,
  `CudnnVersion`, `TDevice.ToString`. `TchJit`:
  `JitFuserCudaEnabled`/`SetJitFuserCudaEnabled`. `TchNN`: the guarded
  cuDNN flatten call in `TRNN`.
- `tests/gpu/gpu_test.pas` + `run.sh [rocm|cuda|cpu]`: all items OK on the
  RX 9070; `cpu` prints `SKIP`. Mock: CPU-build stubs and
  `TestGpuBackend`. CPU jit test: the fuser flag. Regression on CPU (abi,
  api, nn, jit, mock): all OK.
- `libtorch-rocm/lib` needed **eight symlinks** the zip does not carry
  (seven SONAME links and `libnuma.so`); without them the process does not
  start. The command is in DESIGN 4.8 "Results". Redo it after
  re-unpacking.

**Measured, different from the design** (all in DESIGN 4.8 "Results"):
- H34 false: no cuDNN on ROCm (`CudnnIsAvailable` False), so the RNN
  flatten is never called there (H30 moot); RNN numbers reproduced.
- H35: `has_hip` False, `has_cuda` True, `has_cudart` False on ROCm.
  `GpuBackend` uses "HIP, or CUDA without CUDART" for ROCm.
  `GpuRuntimeVersion` raises on ROCm.
- nvfuser is removed in libtorch 2.13: `SetJitFuserCudaEnabled(True)`
  raises on every build.
- `libtorch_rocshmem.so` prints `(null): No such file or directory` four
  times on stderr at start-up (RDMA `dlopen`s); harmless.
- Haiku ran the mock stubs and both regressions; this time its work and
  reports were correct (diffs reviewed).

**Open / next (nothing designed is left to implement):**
- The CUDA path is unmeasured (no NVIDIA GPU here): the `gbCuda` branch,
  the cuDNN flatten call, `GpuRuntimeVersion`/`CudnnVersion` values.
- H29 (`.ot` interchange with tch-rs-written files) still needs a Rust
  toolchain.
- The next milestone has no design yet: pick a topic with the user and run
  a design session (Fable sub-agent, as sessions 6–8).

## Local session 7 (2026-10-12): milestone 3 layers (4.6.10) implemented

**Where things stand.** 4.6.10 is implemented and committed; the working
tree is clean. Steps 1–6, 8 and 9 of design session 7's local list are
done; step 7 (optional mock check) was skipped, since names and order are
checked against libtorch.

- `src/TchNN.pas`: `TConvConfigND`, `DefaultConvConfigND`,
  `ConvConfigToND`, `PadWithMode` (replaces the private `ModeNames`),
  `TConvND` + `TConv1D/2D/3D` (`TConv2D` rebased, API unchanged), the
  conv-transpose family, `TGroupNorm`, `TRNNConfig`/`TRNNState`/`TRNN`/
  `TLSTM`/`TGRU`, `TIdentity`.
- `tests/nn/nn_test.pas` group 11, items 1–11 of 4.6.10 (44 checks): all
  OK against libtorch 2.13.0 CPU, every measured number reproduced
  (conv1d modes, PyTorch-order ND padding `[[7, 10], [5, 8]]`, transpose
  1D/2D/3D, group norm ±0.99998, GRU `0.380797`, LSTM `0.380797 /
  0.181700`, the ten tch-rs RNN configs, `BatchFirst False`). Group 8's
  conv numbers are unchanged through the new base.
- Regression: `tests/abi`, `tests/api`, `tests/nn`, `tests/jit`, mock —
  all OK.
- `docs/UML.md`/`UML_jp.md`: 4.6.10 elements no longer planned.

**Deviations from the design, found while implementing** (recorded at the
end of DESIGN 4.6.10):
- The RNN constructors' parameter is `AHiddenDim`: `HiddenDim` is a
  duplicate identifier next to the `HiddenDim` property.
- The protected `TRNN.Create` draws warning 3018; silenced with
  `{$push}{$warn 3018 off}` as `TJitModule.CreateRaw`.

**Next:** **4.8 GPU** — the only design item left unimplemented. ROCm
prerequisites are met (`libtorch-rocm/`, H32/H33); start at step 1 of
design session 8's local list below. H31, H34, H35 open. The `TRNN`
cuDNN flatten call (H30) now has its `TRNN` to go into.

## Local session 6 (2026-10-12): TorchScript objects (4.6.9) implemented

**Where things stand.** 4.6.9 is implemented and committed; the working
tree is clean. Steps 1–8 of design session 6's local list are done.

- `src/TchSys.pas`: the four externals (`atm_create_class_`,
  `ati_object_method_`, `ati_object_getattr_`, `ati_clone`); "Not
  declared" is down to `atm_fuser_cuda_*`.
- `src/TchJit.pas`: `ivObject = 14`, `IJitObject` (forward-declared
  before `TIValue`), `TJitObjectHandle`, `FromObject`/`:=`/`AsObject`,
  `ToC` clones, `FromC` keeps the pointer (`Keep`),
  `TJitModule.CreateClass`, `JitLiveObjects`.
- `tests/jit/jit_test.pas`: `TestObjects`, items a–g of 4.6.9 (29
  checks), all OK against libtorch 2.13.0 CPU. **H26, H27, H28 hold**
  (results recorded in DESIGN 4.6.9). `[O, 42]` converts element-wise
  through `:=` (was unmeasured).
- Mock: `mock_tch.c` tag-14 objects, `jit_mock_test.pas` `TestObjects`;
  `tests/mock/run.sh` OK.
- Regression: `tests/abi`, `tests/api`, `tests/nn`, `tests/jit`, mock —
  all OK.
- `docs/UML.md`/`UML_jp.md`: 4.6.9 elements no longer planned.

**Deviations from the design, found while implementing:**
- `CreateClass(const ClassName: …)` does not compile: `ClassName` is
  `TObject.ClassName` (duplicate identifier). The parameter is
  `QualifiedName`; DESIGN 4.6.9 updated.
- Mock counts: a mock object is one cell **plus one per attribute**
  (attributes are cloned ivalues), so the design's
  `mock_live_ivalues = 1` is `Cells` (3 for `[1, 2.5]`) in the test. Item
  7 (failing Device sibling) is checked as a delta around the statement:
  temporaries from items 5–6 still hold handles inside the routine (H1).
  The delta is 0 — the converted sibling is released at once.

**Working with Haiku (new this session).** The user wants Claude Haiku
4.5 sub-agents for mechanical work and for running tests and reporting.
This session: Haiku wrote the mock stubs (from the 4.6.9 spec), synced
UML en/jp, ran the regression. Its mock had a leak (`ati_object_method_`
cloned the elements and `ati_tuple` cloned them again), and it reported
the resulting failures as "unrelated"; its UML edit missed one sentence.
Both fixed here. So: give Haiku fully specified tasks, tell it to report
failures verbatim without interpreting them, and review its diff.

**Next (ask which first if the user does not say):**
- **4.6.10 milestone 3 layers** (CPU only; local list in design session 7
  below). 4.8's `TRNN` flatten call depends on it.
- **4.8 GPU** (ROCm prerequisites met; design session 8's local list
  from step 1; H31, H34, H35 open).

## Local session 5 (2026-10-11/12): three designs in, ROCm libtorch obtained

**Where things stand.** Everything below is committed and pushed. Nothing
is half-done; the working tree is clean.

- **Design sessions 6–8 (sections below)** were run as Claude Fable 5.1
  sub-agents from this local session, one item at a time, each reviewed
  and committed here: TorchScript objects (DESIGN 4.6.9), milestone 3
  layers (4.6.10), CUDA/ROCm (4.8). Cost was about $5–6 per design on
  Anthropic credit; a focused prompt (named sections only, a tool-call
  budget, no downloads) kept the later ones cheaper. H36 was checked here
  against tch-rs 4227b89b `src/nn/rnn.rs` (resolved; stride0 2 without
  biases is a stated deviation).
- **`docs/UML.md` / `UML_jp.md`**: eight Mermaid diagrams of what is
  implemented and what is planned (units, class diagrams per unit, the
  ForwardIV ownership and fine-tuning sequences, build variants) plus a
  status table. Update it when a planned element is implemented.
- **ROCm libtorch obtained** (the user-side prerequisite 1 of design
  session 8): `libtorch-shared-with-deps-2.13.0+rocm7.1.zip` from
  `https://download.pytorch.org/libtorch/rocm7.1/`, unpacked as
  `libtorch-rocm/` (12 GB, `build-version` = `2.13.0+rocm7.1`; the zip was
  deleted afterwards). `.gitignore` now has `libtorch-*/`. The archive
  name has no `cxx11-abi` part any more; it is the cxx11 ABI. A
  2.14.1+rocm7.14 archive (1.6 GB) exists too but was not taken: our
  generated ops are from 2.13.0.
  - **(H33) resolved:** `libtorch_hip.so` contains `gfx1201` (RX 9070),
    also gfx1200, gfx11xx, gfx1030, gfx9xx; not `gfx1036` (the iGPU), so
    `HIP_VISIBLE_DEVICES=0` stays required.
  - **(H32) resolved:** `ldd libtorch_hip.so` finds everything;
    `libamdhip64`, `libMIOpen`, `librocblas`, `libhipblaslt`, `librccl`,
    `libhsa-runtime64` are bundled in `lib/`. No AMD ROCm packages needed.
    **Amended in local session 8:** the zip carries no symlinks; eight must
    be added before anything loads (DESIGN 4.8 "Results").
  - **Trap:** `LD_LIBRARY_PATH=libtorch-rocm/lib` makes system tools load
    the bundled `libtinfo.so.6` (bash warns "no version information
    available"). Use the per-build rpath of DESIGN 4.8, never a global
    `LD_LIBRARY_PATH`.
- **Working agreement with the user (also in auto-memory):** after a large
  download finishes, report and stop; unpack/verify only when asked.

**Next (the user starts it after their 18:00 token reset; ask which first
if they do not say):**
- **4.8 GPU** — the ROCm prerequisites are met, so design session 8's
  local list below can start at its step 1 (prerequisite steps 1–3 are
  done: archive present, H32 holds, device access works from the local
  login). H31, H34, H35 are the open hypotheses.
- **4.6.9 TorchScript objects** and **4.6.10 milestone 3 layers** — CPU
  only, local lists in design sessions 6 and 7 below.
- Keep DESIGN/HANDOFF/UML (en + jp) current at the end, as usual.

## Design session 8 (2026-10-11): CUDA/ROCm designed — implemented in local session 8 above

Answers item 3 of "Handing back (end of local session 4)" below — the last
open one. Design only: nothing committed, nothing downloaded or installed;
one throwaway cmake build in the scratchpad. Details and rationale in
DESIGN 4.8.

**Decisions** (DESIGN 4.8):
- One unpacked libtorch per variant (`libtorch/` CPU as today,
  `libtorch-rocm/`, `libtorch-cuda/`) and one build dir per variant
  (`build/libtch`, `build/libtch-rocm`, `build/libtch-cuda`). Pascal units,
  C wrapper and ABI are unchanged: HIP masquerades as CUDA (confirmed in
  `libtorch/include/ATen/Context.h:638-650`), so `TDevice.Cuda(0)`,
  `device_of_int`, `atc_*` address the AMD GPU as is.
- `csrc/libtch/CMakeLists.txt` gains `-DTCH_LIBTORCH_DIR=<dir>`: manual
  include/link lines (`-Wl,--no-as-needed libtorch.so`, then
  `libtorch_cpu.so libc10.so`, `-D_GLIBCXX_USE_CXX11_ABI=1`, `BUILD_RPATH
  <dir>/lib`) instead of `find_package(Torch)`, because a ROCm archive's
  `Caffe2Config.cmake` includes `LoadHIP.cmake`, which wants the ROCm SDK
  at `/opt/rocm`. Default path unchanged for the CPU archive.
- L2: `atc_cudnn_is_available`, `atc_manual_seed(_all)`, `atc_synchronize`,
  `atc_user_enabled_cudnn`, `atc_set_user_enabled_cudnn`,
  `atc_set_benchmark_cudnn`, `at_context_has_cuda/hip/cudnn`,
  `at_context_version_cudart/cudnn`, `atm_fuser_cuda_*`. L3 (4.7 block):
  `CudnnIsAvailable`, `CudaManualSeed(All)`, `CudaSynchronize(Index = -1)`,
  `CudnnUserEnabled`/`SetCudnnUserEnabled`, `SetCudnnBenchmark`,
  `TGpuBackend = (gbNone, gbCuda, gbRocm)` + `GpuBackend`,
  `GpuRuntimeVersion`, `CudnnVersion`, `TDevice.ToString`; `TchJit`:
  `JitFuserCudaEnabled`/`SetJitFuserCudaEnabled`. Only `CudaIsAvailable` /
  `CudaDeviceCount` are safe on a CPU build; the others raise
  `ETorchNativeError` there.
- `TRNN`: tch-rs's flatten call when `Device.IsCuda and CudnnIsAvailable`
  (`_cudnn_rnn_flatten_weight(FFlatWeights, 4 or 2, InputSize, Mode 2/3,
  HiddenSize, 0, NumLayers, BatchFirst, Bidirectional)`), result
  discarded, `ETorchNativeError` swallowed (hint only; MIOpen may refuse).
- `tests/gpu/gpu_test.pas` + `run.sh <variant>`: `SKIP` + exit 0 when
  `CudaIsAvailable` is False; otherwise ten items reusing the CPU numbers
  of 4.6.7/4.6.8/4.6.10 (`3 5 7`, `6`, `[0.9, 0.8]`, `0.380797`,
  `0.1817`), H13's real device move (item 5) and H30 (item 8). Run with
  `HIP_VISIBLE_DEVICES=0` so the Raphael iGPU (`gfx1036`, unsupported) is
  hidden and `CudaDeviceCount = 1`.
- Mock: new `atc_*`/`at_context_*` stubs return what a CPU build returns;
  `mock_test.pas` checks `GpuBackend = gbNone`, the fuser flag round trip.

**Inspected** (nothing changed): `rocm-smi --showproductname` → GPU[0] RX
9070 `gfx1201`, GPU[1] Raphael iGPU `gfx1036`; `/sys/class/kfd` lists both
(kernel 7.0.0 amdgpu/KFD work). Installed ROCm user space is **only**
`rocm-smi 7.1.1` + `librocm_smi64` (Ubuntu); no `/opt/rocm`, no
`libamdhip64`, no MIOpen, no `rocminfo`. User groups: `joerg adm cdrom sudo
dip plugdev users lxd ollama` — not in `render`/`video`, but `/dev/kfd` is
already read-write for the user through the seat ACL. Measured: the
manual-link CMakeLists builds libtch against the CPU archive in 28 s; the
result `NEEDED`s only `libtorch_cpu.so` + `libc10.so` (as-needed dropped
the umbrella — hence the explicit `--no-as-needed` for GPU builds, where
the backend is reached only through `libtorch.so`).

**New hypotheses** (need the ROCm libtorch; none measurable here):
- **H31** the manual-link libtch works unchanged against the ROCm archive
  (no extra defines such as `__HIP_PLATFORM_AMD__` needed by
  `torch_api.cpp`).
- **H32** the `libtorch-cxx11-abi-shared-with-deps-2.13.0+rocmX.Y` archive
  bundles the HIP runtime, MIOpen, rocBLAS, hipBLASLt, RCCL in `lib/`
  (check `ldd libtorch-rocm/lib/libtorch_hip.so | grep 'not found'`). If
  not: AMD's ROCm runtime packages of the same X.Y.
- **H33** that archive contains `gfx1201` code objects (`strings -n 7
  libtorch_hip.so | grep -oE 'gfx[0-9a-f]+' | sort -u`). Not visible
  locally; PyTorch's ROCm 7.x builds are expected to, ROCm 6.x ones are
  not.
- **H34** `atc_cudnn_is_available` is 1 on ROCm (MIOpen stands in for
  cuDNN through the hipified hooks).
- **H35** `at_context_has_hip` is True on ROCm; whether
  `at_context_has_cuda` is also True is open (`GpuBackend` checks HIP
  first, so it is right either way; the test prints both).
- **H36** tch-rs `rnn.rs` flatten call reconstructed from memory (repo not
  local): `is_cuda && cudnn_is_available`, stride0 4, mode 2 (LSTM) /
  3 (GRU), proj_size 0. **Resolved** by the orchestrating session against
  tch-rs 4227b89b `src/nn/rnn.rs` (lines 154–166, 239–251): matches;
  tch-rs hard-codes stride0 4 even without biases, DESIGN 4.8 uses 2
  then (PyTorch), a stated deviation.

**User-side prerequisites** (not done by the design session; described
only):
1. Obtain `libtorch-cxx11-abi-shared-with-deps-2.13.0+rocmX.Y.zip` from
   download.pytorch.org (ROCm variant of the same 2.13.0 used for CPU;
   pick the highest ROCm version offered for 2.13.0 — RDNA4/gfx1201 needs
   ROCm 7.x) and unpack it as `libtorch-rocm/` in the repo root
   (git-ignored). Verify H33 and H32 with the two one-liners above
   before building anything.
2. If H32 fails: install AMD's ROCm runtime packages matching X.Y
   (`rocm-hip-runtime`, `miopen-hip`, `rocblas`, `hipblaslt`, `rccl`,
   `hipfft`, `hipsparse`, `rocsolver`, `hiprand`) from the AMD repo. No SDK
   (`rocm-dev`/cmake files) is needed with the manual-link mode.
3. Device access: `/dev/kfd` and `/dev/dri/renderD*` are `root:render`
   with an ACL (`+`), and `test -r /dev/kfd -a -w /dev/kfd` already
   succeeds from the desktop login (logind grants the seat), so nothing is
   needed for runs from the local session. Only for ssh/cron runs add the
   user to `render` and `video` (`sudo usermod -aG render,video joerg`,
   re-login). Without access HIP reports no devices even though
   `rocm-smi` (which reads sysfs) sees them.
4. For a CUDA box instead: the `+cuXYZ` archive as `libtorch-cuda/` and
   the NVIDIA driver; everything else is identical with `cuda` as the
   variant.

**Local implementation list** (mechanical; order matters):
1. `csrc/libtch/CMakeLists.txt`: the `TCH_LIBTORCH_DIR` branch of DESIGN
   4.8; rebuild `build/libtch` the old way to confirm nothing changed, and
   once with `-DTCH_LIBTORCH_DIR=$PWD/libtorch` into `build/libtch-manual`
   to confirm the manual path (CPU) passes `tests/api`.
2. `src/TchSys.pas`: the L2 externals above (QWord for `uint64_t`, Int64
   for `int64_t`, `Boolean` for C `bool`, `cint` for `int`); trim the unit
   comment's "Not declared" list to nothing.
3. `src/Tch.pas`: the 4.7 block additions, `TGpuBackend`, `GpuBackend`,
   `TDevice.ToString`; API comments stating which calls are CPU-safe.
4. `src/TchJit.pas`: `JitFuserCudaEnabled`/`SetJitFuserCudaEnabled`.
5. `src/TchNN.pas`: the flatten call in `TRNN`'s protected constructor
   (after `FFlatWeights` is complete), guarded and swallowed as above.
6. `tests/mock/mock_tch.c` stubs + `mock_test.pas` checks; `tests/jit`
   adds the fuser-flag round trip on CPU. Run `tests/mock/run.sh`,
   `tests/api`, `tests/nn`, `tests/jit` on the CPU build: unchanged.
7. `tests/gpu/gpu_test.pas` and `tests/gpu/run.sh` (variant argument,
   visibility variable, `-Flbuild/libtch-$v -k-rpath=...`); first run it
   against `build/libtch`: exactly `SKIP: no GPU in this libtorch`, exit 0.
8. After the user-side prerequisites: `cmake -S csrc/libtch -B
   build/libtch-rocm -DTCH_LIBTORCH_DIR=$PWD/libtorch-rocm
   -DCMAKE_BUILD_TYPE=Release && cmake --build build/libtch-rocm -j 8`;
   `ldd build/libtch-rocm/libtch.so | grep 'not found'` must be empty.
9. `tests/gpu/run.sh rocm`: record item 1's capability print, the H13 and
   H30 outcomes, and H31–H35 results next to each hypothesis (DESIGN 4.8
   and here); then `LD_LIBRARY_PATH=build/libtch-rocm` in front of the
   CPU-linked `tests/nn` and `tests/jit` binaries — everything still passes
   on CPU tensors with the GPU libtorch loaded (same ABI).
10. Add a "Local session 7" section here (it became local session 8) (en +
    jp); mark H31–H36 resolved
    or amended; update DESIGN 4.8 numbers if any differ.

## Design session 7 (2026-10-11): milestone 3 layers designed — implemented in local session 7 above

Answers item 2 of "Handing back (end of local session 4)" below; item 3
(CUDA/ROCm) stays open. Design only: nothing committed, no code outside the
scratchpad. Ran on the local machine against the real libtorch 2.13.0 (CPU)
and FPC 3.2.2, so the numbers in DESIGN 4.6.10 are measured, not derived.

**Decisions** (details and rationale in DESIGN 4.6.10):
- tch-rs's generic `ConvConfigND<ND>` becomes one non-generic
  `TConvConfigND` with `TInt64Array` fields (`Nd = Length`), one base class
  `TConvND` (`Forward` switches on `FNd`), and `TConv1D/2D/3D` subclasses
  holding only the scalar-config constructors. `TConv2D`'s API is unchanged;
  its body moves to the base. `ConvConfigToND(Cfg, Nd)`,
  `DefaultConvConfigND(Nd)`, `PadWithMode` (public, `'constant'` for
  `pmZeros`) are added. Same shape for `TConvTransposeND` + `1D/2D/3D`
  with `TConvTransposeConfig[ND]` (weight `[In, Out div Groups, K..]`).
- `TGroupNorm(NumGroups, NumChannels[, Cfg])` over `group_norm`, weight
  then bias `[NumChannels]` when `Affine`.
- `TRNN = class abstract` (not a `TModule`, as tch-rs) with `ZeroState`,
  `Step`, `Seq`, `SeqInit`; one `TRNNState = record H, C: TTensor end`
  for both `TLSTM` and `TGRU` (GRU leaves `C` nil); `TRNNConfig` shared;
  `FFlatWeights` registered as `weight_ih_l{L}[_reverse]`, `weight_hh_…`,
  `bias_ih_…`, `bias_hh_…` per layer per direction. `SeqInit` calls the
  generated `lstm`/`gru` with the `out` results going to the function
  result (output) and `OutState.H`/`.C`.
- `TIdentity` for `nn::Id`.
- Deviations from tch-rs, each stated in 4.6.10: PyTorch's order for
  reversed-padding-repeated-twice (tch-rs's is wrong for asymmetric ND
  padding, measured); `BatchFirst False` handled in `Seq`/`Step`; plain
  `TInit` instead of `Option<Init>` for the bias inits; `ZeroState` reads
  kind/device from the first weight at call time; cuDNN flattening left to
  the CUDA/ROCm milestone.

**Measurements** (probe `scratchpad/fable-layers/probe*.pas`, FPC 3.2.2 +
libtorch 2.13.0 CPU):
- M-A: base constructor with `array of Int64` + two `overload` subclass
  constructors resolve for the scalar calls and for the inherited ND call
  through the subclass. Group-8 conv2d numbers unchanged through the base.
- M-B / M-B2: `SeqInit(X, S, S)` nils `S` on entry (H8) → `ETorchError
  "undefined tensor passed as hx to gru"`; `S := Step(X, S)` is safe
  (result through a temporary).
- M-C: every number in the 4.6.10 test list (conv1d four modes, conv3d,
  asymmetric ND padding, conv_transpose 1D/2D/3D incl. stride 2 and
  `OutputPadding`, group_norm ±0.99998, GRU `0.380797`, LSTM `0.1817 /
  0.380797`, the five tch-rs RNN configs' shapes and variable names,
  `batch_first` False shapes).

**New hypotheses** (not measurable with CPU libtorch here): **H29** `.ot`
interchange with tch-rs-written RNN/conv-transpose/group-norm files (needs
Rust); **H30** the cuDNN flatten call on a CUDA/ROCm store (item 3).

**Local implementation list** (mechanical; all in `src/TchNN.pas` and
`tests/nn/nn_test.pas`, no L2 or C changes):
1. Add `TConvConfigND`, `DefaultConvConfigND`, `ConvConfigToND`,
   `PadWithMode` (replace the private `ModeNames`, `'constant'` for
   `pmZeros`); `TConvND` with the ND constructor (length checks raise
   `ETorchError`), `FPadTwice[2*I] = FPadTwice[2*I+1] = Padding[Nd-1-I]`,
   `Forward` as the 4.6.10 table. Rebase `TConv2D` on it (body →
   `inherited Create(P, In, Out, [K, K], ConvConfigToND(Cfg, 2))`); add
   `TConv1D`, `TConv3D`. Run the existing group 8: unchanged.
2. `TConvTransposeConfig`, `TConvTransposeConfigND`, the two `Default*`,
   `ConvTransposeConfigToND`, `TConvTransposeND`, `TConvTranspose1D/2D/3D`.
3. `TGroupNormConfig`, `DefaultGroupNormConfig`, `TGroupNorm`.
4. `TRNNConfig`, `DefaultRNNConfig`, `TRNNState`, `TRNN` (protected
   constructor = `rnn_weights` with `GateMul` 4 or 3; `Step`, `Seq`,
   `BatchDimIndex`), `TLSTM`, `TGRU` (`ZeroState`, `SeqInit`). Document
   the `InState`/`OutState` distinct-variables rule at the declaration.
5. `TIdentity`.
6. `tests/nn/nn_test.pas` group 11, items 1–11 of the 4.6.10 list; keep
   the final `TchLiveHandles = 0`.
7. Mock (`tests/mock`): nothing numerical to add; optionally construction
   and registration order of `TLSTM(2, 4)` (16 names for 2 layers
   bidirectional) over the shape-only stubs.
8. Regression: `tests/nn`, `tests/api`, `tests/mock/run.sh`, `tests/jit`.
9. Record results next to H29/H30 only if they become testable; add a
   "Local session 6" section here (it became local session 7) (en + jp).
   Item 3 of "Handing back (end
   of local session 4)" remains the open design work.

## Design session 6 (2026-10-11): TorchScript objects designed — implemented in local session 6 above

Answers item 1 of "Handing back (end of local session 4)" below; items 2
(milestone 3 layers) and 3 (CUDA/ROCm) stay open. Design only: nothing
committed, no code outside the scratchpad. The session ran on the local
machine, so the design is measured against the real libtorch 2.13.0 (CPU)
and the built `libtch.so`. Everything is in **DESIGN 4.6.9**; summary:

1. **Handle.** `IJitObject`, an interface refcounted by `TInterfacedObject`
   (as 4.1's `TTensorHandle`; interlocked, deterministic release), with
   the implementing class `TJitObjectHandle` hidden in the implementation
   section. An interface rather than a record because `TIValue` needs the
   field and the object's methods take and return `TIValue`: FPC cannot
   forward-declare a record but can forward-declare an interface
   (measured). `JitLiveObjects` counts live handles like `TchLiveHandles`.
2. **`TIValue`.** `ivObject = 14`, field `Obj: IJitObject`, `FromObject`,
   `:=` from `IJitObject`, `AsObject`, `TagName = 'Object'`. **`FromC`
   keeps the pointer for tag 14** (`Keep` flag; the handle owns it);
   **`ToC` returns `ati_clone(Obj.Raw)`** — the caller frees the clone,
   the handle's pointer is never handed to C for freeing (tch-rs's fix
   for issue 475). Inside containers nothing special: `ati_to_*` hand out
   fresh `IValue`s (measured) and a failing later child releases the
   converted siblings through the compiler's unwinding of
   `IValuesFromC`'s local array.
3. **API.** `IJitObject.MethodIV(Name, Inputs)` and `GetAttr(Name)`
   (shape of `TJitModule.MethodIV`); `TJitModule.CreateClass(ClassName,
   Inputs): IJitObject` (qualified name, e.g. `'__torch__.InputObject'`;
   returns the object rather than tch-rs's `IValue`, stated). An object
   does not keep its module alive and does not need to (measured: method
   calls work after `atm_free`).
4. **L2.** Four externals (`atm_create_class_`, `ati_object_method_`,
   `ati_object_getattr_`, `ati_clone`), declarations in 4.6.9. **Fork
   patch: none** (the unused local in `atm_create_class_` is harmless).
5. **Tests.** Group 14 of `tests/jit/jit_test.pas`: `jit6` (`foo6.pt`,
   `y()` → `[6, 8, 10]`), `jit_double_free` (`foo7.pt`,
   `make_input_object` then `add_them` → `[5, 7, 9]`, twice with the same
   handle), `CreateClass`, the measured error messages, objects through
   tuple/list/dict, sharing, object after `M.Free`, counts back, a `10^4`
   loop. Mock: tag-14 `mock_ivalue` with named attributes `a0, a1, …`,
   `mock_last_class`, `TestObjects` (eight checks).

**Measurements (libtorch 2.13.0 CPU, FPC 3.2.2; probes in the scratchpad
only):**
- M9: raw C entry points on `foo6.pt`/`foo7.pt` — `ati_tag` 14 for both
  object results; `y()` → `[6, 8, 10]`; `getattr x` → `[3, 4, 5]`;
  `ati_clone` works after `ati_free` of the original and after `atm_free`
  of the module; a tuple child comes back as a fresh pointer usable after
  the tuple is freed; `add_them` twice with the same object pointer →
  `[5, 7, 9]` twice; `atm_create_class_('__torch__.InputObject', [T1,
  T2])` → `add_them` `[5, 7, 9]`; the bare class name fails. Nine error
  messages recorded verbatim in 4.6.9 (missing attribute/method, arity,
  `to_tensor`/`length` on an object, unknown class, missing `__init__`
  argument, wrong argument type, non-object pointer).
- M10: FPC accepts `IJitObject = interface;` forward → record field →
  full interface naming the record; copies share the handle, release is
  deterministic, the count returns to 0 after the routine returns (inside
  it a temporary keeps 1, H1); `SizeOf` 16.
- M11: `atm_forward` throws "forward did not return a tensor" for any
  non-tensor result (read in `torch_api.cpp`, not run on an object → H27).

**New hypotheses (need real libtorch):**
- **(H26)** an object round-trips through `GenericList` and as a
  `GenericDict` value as through a tuple (fresh tag-14 `IValue`s that
  outlive the container).
- **(H27)** `CreateClass('__torch__.TorchScriptClass', [x])` on `foo6.pt`
  gives `y() = 2x`; `Forward` (tensor API) on `foo6.pt` raises "forward
  did not return a tensor".
- **(H28)** `10^4` object method calls leave `JitLiveObjects`,
  `TchLiveHandles` and RSS flat.

**Local session, in this order (mechanical):**
1. `src/TchSys.pas`: the four externals of 4.6.9 after `ati_free`; the
   unit comment's "Not declared" list becomes `atm_fuser_cuda_*` only.
2. `src/TchJit.pas`, interface: `ivObject = 14` in `TIValueTag`;
   `IJitObject = interface;` before `TIValue`; `Obj: IJitObject` field,
   `FromObject`, `class operator :=(const O: IJitObject)`, `AsObject`;
   the full `IJitObject` (new GUID; `Raw`, `MethodIV`, `GetAttr`) after
   `TIValue`; `TJitModule.CreateClass`; `function JitLiveObjects:
   Integer`. Header comment: drop "objects not supported", add "an object
   does not keep its module alive".
3. `src/TchJit.pas`, implementation: `TagNames` → `'Object'`;
   `LiveObjects` counter + `JitLiveObjects`; `TJitObjectHandle`
   (constructor order `inherited; Inc; FRaw := P`, destructor `ati_free`
   + `Dec`); `AsObject`/`FromObject`/`:=`; `ToC` branch `ivObject`
   (`ati_clone`, nil `Obj` → `ETorchError('ivalue: nil object')`); `FromC`
   case 14 with the `Keep` local and `if not Keep then ati_free(P)` in the
   `finally`; `TJitObjectHandle.MethodIV`/`GetAttr` and
   `TJitModule.CreateClass` as 4.6.9 step 5 (`IValuesToC`, call,
   `TchCheck`, `finally FreeIValues`, `FromC`; `CreateClass` ends with
   `.AsObject`). Compile `tests/mock/run.sh` (the mock lacks the symbols
   until step 4: the test programs reference only what they use, so the
   existing three still link).
4. Mock: `cls`/`names` in `mock_ivalue`, `iv_clone`/`ati_free` updates,
   `atm_create_class_` + `mock_last_class`, `ati_object_getattr_`,
   `ati_object_method_`, `ati_clone`, `ati_length` failing on 14.
5. `tests/mock/jit_mock_test.pas`: `mock_last_class` external,
   `TestObjects` with the eight checks of 4.6.9 (counts compared from the
   caller), `JitLiveObjects = 0` at the end. `tests/mock/run.sh`.
6. `tests/jit/jit_test.pas`: group 14 "Objects" items a–g of 4.6.9,
   checking H26–H28; `JitLiveObjects = 0` next to the final
   `TchLiveHandles = 0`. Build and run as the file header says.
7. Regression: `tests/jit`, `tests/mock/run.sh`, `tests/abi` (C
   unchanged), `tests/api`, `tests/nn`.
8. Record H26–H28 results next to the hypotheses in DESIGN 4.6.9 (en +
   jp), add a "Local session 5" section here (en + jp); items 2 and 3 of
   "Handing back (end of local session 4)" remain the open design work.

## Local session 4 (2026-10-11): milestone 3 (JIT) implemented

Steps 1–8 of cloud session 5's list (below) are done; every test passes.
Commits `8a912eb` … `b816950`.

1. **`FromArray` revision** (DESIGN 4.3): the `Single` overload is gone, the
   four others take `AKind` (`Double` + `kFloat` rounds into a `Single`
   buffer in Pascal), `FromSingleArray` added. `tests/compiler` has the new
   `h2_fromarray.pas` and `xfail_h2_single.pas`; `tests/api` and `tests/nn`
   updated. Trap found in the test: a real literal in a comparison is
   `Extended` on x86-64, so `T.DoubleValue([0]) = 0.9` is False; compare
   with `Double(0.9)`.
2. **4.6.7 follow-ups:** `{$push}{$warn 3018 off}` around
   `TOptimizer.CreateRaw` (no warning left); the half-precision
   `LoadPartial` check added to `TestSaveLoad` (passes: `u2` `kHalf` mean
   42, `v2` `kFloat` mean 1, missing `['a.b.t2']`).
3. **Fork patch:** `atm_create`, `atm_register_parameter`, `atm_define` in
   `csrc/libtch/torch_api.{h,cpp}`; libtch rebuilt; `tests/abi`, mock pass.
4. **Vendored** `tests/jit/models/foo.pt` … `foo8.pt` with tch-rs's two
   license files and a short README naming the source.
5. **`src/TchSys.pas`:** the JIT types and externals of 4.6.8.
6. **`src/TchJit.pas`:** `TIValue`, `TJitModule`, `TTrainableJitModule`,
   `Jit*` globals. Small deviations, recorded at the end of DESIGN 4.6.8:
   `TagName` is a `case` (FPC rejects an enum with explicit values as an
   array index); `CreateByTracing` cleans up after *any* failure past
   `atm_create_for_tracing`; `Item`/`Key`/`Value` range-check.
7. **Mock:** `atm_*`/`ati_*` in `tests/mock/mock_tch.c`,
   `tests/mock/jit_mock_test.pas` (also: a child failing in `ToC` or
   `FromC` leaves no C ivalue behind), added to `run.sh`.
8. **`tests/jit/jit_test.pas`:** all thirteen groups pass; **H19–H25 all
   hold** (results next to the hypotheses at the end of DESIGN 4.6.8).

- Regression: `tests/api`, `tests/nn`, `tests/generated`,
  `tests/ownership`, `tests/abi`, `tests/compiler/run.sh`,
  `tests/mock/run.sh` pass. `tests/fpu/run.sh` behaves as before (it is a
  measurement: the unmasked modes crash or hang by design).
- Not touched, as asked: milestone 3 layers and CUDA/ROCm.

## Handing back (2026-10-11, end of local session 4) — items 1, 2 and 3 answered in design sessions 6, 7 and 8 above

**Design only** (CLAUDE.md). Nothing is blocked; all of the following are
optional next milestones. Mark what needs real libtorch as H26, ...

1. **TorchScript objects (ati_tag 14)** — *designed: DESIGN 4.6.9, design
   session 6 above.* `ati_object_method_`,
   `ati_object_getattr_`, `ati_clone`, `atm_create_class_`; tch-rs
   `Object`, tests `jit6` and `jit_double_free`, assets `foo6.pt`/`foo7.pt`
   (already vendored, unused). Needs the refcounted handle DESIGN 4.6.8
   mentions (an object keeps its C `IValue` alive), and how `TIValue`
   carries it (copies share; `FromC` currently raises on tag 14).
2. **Milestone 3 layers** (DESIGN 4.6.5 deferred list) — *designed: DESIGN
   4.6.10, design session 7 above.*
3. **CUDA/ROCm** — *designed: DESIGN 4.8, design session 8 above.* Only
   CPU so far (H13's real device move untested;
   `atm_fuser_cuda_*` not declared). The local machine has an AMD Radeon
   RX 9070 (Navi 48) and `rocm-smi`; no ROCm libtorch has been tried yet.
   So the GPU route to design first is ROCm (libtorch's HIP build reports
   the device as CUDA).

**How the next design session runs is still open:** the cloud service
credits are used up, and the user will decide later how to send the
design work to Fable. Until then, nothing here waits on it. (Settled in local session
5: Fable 5.1 sub-agents run from the local session, one design at a time.)

## Cloud session 5 (2026-10-10): JIT designed, `FromArray` revised, local session 3 reviewed — implemented in local session 4 above

The request below ("Handing back to the cloud, end of local session 3") is
answered for items 1–3; item 4 (milestone 3 layers, CUDA/ROCm) is
untouched. Design only: nothing committed, no code outside the scratchpad.
Summary:

1. **JIT (DESIGN 4.6.8).** L2: `PTorchModule`, `PTorchIValue`, the 21
   `atm_*`/`at_set_graph_executor_optimize` and 31 `ati_*` externals listed
   there (`atm_named_parameters` reuses `TTorchLoadCallback`). Not declared:
   TorchScript objects (tag 14: `ati_object_*`, `ati_clone`,
   `atm_create_class_`, tests `jit6`/`jit_double_free`, `foo6.pt`/`foo7.pt`
   — a later milestone, they need a refcounted handle) and `atm_fuser_cuda_*`
   (CUDA). `TIValue`: tag values = `ati_tag`'s 0–13 plus input-only
   `ivStringList`/`ivDevice`; static constructors, implicit `:=` from
   `TTensor`/`Int64`/`Double`/`Boolean`/`String`, `As*`/`Item`/`Key`/`Value`
   accessors; `ToC` (caller frees) / `FromC` (consumes) with nil-initialised
   child slots and `finally` cleanup; `GenericDict` as flat key/value
   `Items`. `TJitModule`: `Create(file[, device])`, `CreateFromStream`,
   `CreateByTracing`, `CreateEmpty` + `RegisterParameter` + `Define` (fork
   addition `atm_create`/`atm_register_parameter`/`atm_define`: three
   one-line functions over `Module::define`, `register_parameter` and a
   `training` attribute), `Forward`/`Method` (tensor), `ForwardIV`/`MethodIV`
   (ivalue), `SetEval`/`SetTrain`, `ToDevice(device, kind)`, `Save`,
   `NamedParameters` (4.6.3's trampoline, duplicated in `TchJit`), the
   `Jit*` globals. `TTrainableJitModule = class(TModule)`:
   `atm_load_on_device(P.Device)`, `P.Add(name with '.'→'_', T,
   T.RequiresGrad)`, shared handles; `Forward(X, Train)` applies `Train`
   through `atm_train`/`atm_eval` when it changes (stated deviation: tch-rs
   ignores it). **Test assets:** vendor tch-rs's `tests/foo*.pt` into
   `tests/jit/models/` (license files next to them) **and** build the
   parameterised / `self.training` modules from TorchScript source in the
   test; tracing alone cannot produce parameters. Thirteen test groups with
   hand-checkable numbers in 4.6.8.
2. **`FromArray` (DESIGN 4.3).** Drop the `Single` overload; the
   `Double`/`Int64`/`Byte`/`Boolean` overloads get a trailing `AKind` with
   the natural default (`kFloat`/`kInt64`/`kUInt8`/`kBool`);
   `FromSingleArray` for `Single` buffers (as `FromInt32Array`). Measured:
   every real literal then resolves to `Double` (float32 tensor by default,
   PyTorch's rule and the same bits as Python); `[1, 2]` stays `Int64`; a
   `Single` variable inside a literal converts, an `array of Single`
   variable needs the named function.
3. **Review of local session 3.** lr 1e-10: `‖g‖² = 80.955² + 8.974² =
   6634.2`, `49 · 1e-10 · 6634.2 = 3.25e-5`, confirmed; the measured
   `-3.24e-5` is within the float32 resolution of a loss near 23 (1.9e-6).
   4.6.7 test 7 corrected to tch-rs's *add* / *add_partial* layouts (`t123`
   once). `TVarPath.Group`, the `FullName` validation (`TryGet` raises on
   `.`/`|`) and the `AddEntry` nil check confirmed. `CreateRaw` stays
   private; warning 3018 silenced with `{$push}{$warn 3018 off} … {$pop}`
   (measured: `strict private` warns too). All extra tests adopted into
   4.6.7; one to add (the half-precision half of
   `save_and_load_partial_var_store_incomplete_file`).

**Measurements in the cloud (FPC 3.2.2; scratchpad only):**
- M4: open-array literals against `FromArray` overload sets (table in 4.3):
  a real literal is `Single` only if exactly representable in binary32 —
  `[1.0, 2.0]`, `[2.5]`, `[1.0, 0.1]` pick `Single`, `[0.9, -1.9]`,
  `[3.14]`, `[0.1]` are ambiguous; without the `Single` overload all resolve
  to `Double`; the `AKind` defaults change nothing in the resolution.
- M5: `TIValue` record with explicit tag values (`TIValueTag(12) =
  ivGenericList`), nested tuples, implicit `:=` from `Int64`/`Double`/
  `Boolean`/`String`/record; **`[T, 42, 1.5]` converts element-wise into an
  `array of TIValue` argument** through those operators; `SizeOf(Boolean) =
  1`.
- M6: FPC warns `(3018) Constructor should be public` for `private` and
  `strict private` constructors; `{$push}{$warn 3018 off} … {$pop}` removes
  it.
- M7: the fork's `atm_*`/`ati_*` region of `torch_api.{h,cpp}` is identical
  to upstream 4227b89b (`diff`); the `foo*.pt` are zip format version 3 and
  their TorchScript (`code/__torch__.py`) is summarised in 4.6.8.
- M8: the whole 4.6.8 interface (`TIValue`, `TJitModule`,
  `TTrainableJitModule`, the `Jit*` functions) compiled as a throwaway unit
  with stub bodies against the real `Tch`/`TchNN` units: no errors, no
  warning from the pragma-wrapped `CreateRaw`, and the test call shapes
  `ForwardIV([X, 42, 1.5])`, `Tuple([2.0, 3.0, 4])`, `GenericDict(['bar', X,
  'foo', X])`, `StringList([...])` resolve.

**New hypotheses (need real libtorch):**
- **(H19)** the vendored `tests/jit/models/foo*.pt` load with libtorch 2.13
  and give `jit_tests.rs`'s numbers (4.6.8 items 1–7).
- **(H20)** fork addition: `Module::define` compiles the two TorchScript
  sources of 4.6.8 (`torch.matmul`, `torch.t`, `self.training`); the
  registered `training` attribute makes `self.training` and
  `atm_train`/`atm_eval` work; `named_parameters` returns the registered
  parameters with their `requires_grad`.
- **(H21)** parameter sharing: after one SGD step on the var-store,
  `TTrainableJitModule.Forward(X) = 3.0`, and `Save` + reload keeps the
  trained values.
- **(H22)** `ati_generic_dict`'s `Dict[str, Tensor]` specialisation is
  accepted by `foo8.pt`'s `generate`.
- **(H23)** `atm_end_tracing(M, name, nil, 0)` after a raising trace
  closure leaves tracing usable for the next `CreateByTracing`.
- **(H24)** a `GenericDict` round trip keeps insertion order (`c10::Dict`
  is order-preserving).
- **(H25)** `JitGetProfilingMode` is True in a fresh process.

**Local session, in this order (mechanical):**
1. `FromArray` revision (DESIGN 4.3): `src/Tch.pas` — drop the
   `array of Single` overload, add `AKind` defaults to the four others
   (`kFloat` fast path: convert to a `Single` buffer in Pascal; otherwise
   `totype`), add `FromSingleArray`. Tests: `tests/api/api_test.pas`
   (`TDoubleArray.Create(0.25, 0.5)` is now `kFloat` — pass `kDouble` or
   change the expectation; add `FromArray([0.9, -1.9])`, `FromArray([1, 2],
   kFloat)`, `FromSingleArray`), `tests/nn/nn_test.pas` (the `Double(…)`
   casts can go), `tests/compiler/h2_fromarray.pas` (the new set; `[0.9,
   -1.9]`, `[3.14]`, `[1.0, 0.1]` → Double) plus a new
   `xfail_h2_single.pas` showing `[0.9, -1.9]` ambiguous with both `Single`
   and `Double` overloads. Run `tests/compiler/run.sh`, `tests/api`,
   `tests/nn`, `tests/mock/run.sh`.
2. 4.6.7 follow-ups: `{$push}{$warn 3018 off}` / `{$pop}` around
   `TOptimizer.CreateRaw`; add the half-precision `LoadPartial` check of
   4.6.7 item 7d to `TestSaveLoad`.
3. Fork patch: `atm_create`, `atm_register_parameter`, `atm_define` in
   `csrc/libtch/torch_api.{h,cpp}` (code in 4.6.8); rebuild libtch;
   `tests/abi`, `tests/mock/run.sh` (the mock gets its stubs in step 7; the
   test programs reference only what they use).
4. Vendor `tests/jit/models/foo.pt` … `foo8.pt` plus `LICENSE-MIT` and
   `LICENSE-APACHE` from tch-rs 4227b89b (`tests/`); commit message names
   the source files and `tests/create_jit_models.py`.
5. `src/TchSys.pas`: the JIT types and externals of 4.6.8.
6. `src/TchJit.pas`, in this order so each part compiles alone: `TIValue`
   (record, constructors, operators, accessors) → `ToC`/`FromC` →
   `TJitModule` (load/free/forward/method/eval/train/to/save) →
   `NamedParameters` trampoline → `CreateByTracing` → `CreateEmpty` /
   `RegisterParameter` / `Define` → `Jit*` globals → `TTrainableJitModule`.
7. Mock: `atm_*`/`ati_*` stubs in `tests/mock/mock_tch.c` (4.6.8 "Mock"),
   `tests/mock/jit_mock_test.pas`, add it to `run.sh`.
8. `tests/jit/jit_test.pas` with the thirteen groups of 4.6.8, checking
   H19–H25 as they come up; record each result next to the (Hn) in
   DESIGN.md.
9. Update DESIGN/HANDOFF (en + jp); then hand TorchScript objects (tag
   14), the milestone 3 layers (4.6.5 deferred list) and CUDA/ROCm back to
   a design session if needed.

## Local session 3 (2026-10-10): milestone 2 implemented

Steps 1–5 of cloud session 4's list (below) are done; every test passes.

1. **Fork patch** `ato_set_momentum_group`: `} if` → `} else if`
   (`csrc/libtch/torch_api.cpp`); libtch rebuilt; `tests/abi` and
   `tests/mock/run.sh` pass. A build of the *unpatched* source is kept in
   `build/libtch-unpatched/` (git-ignored) only to measure H16.
2. **L2/L3 additions:** `src/TchSys.pas` has `TTorchLoadCallback`,
   `at_save_multi`, `at_load_callback_with_device`,
   `at_loadz_callback_with_device`, `PTorchOptimizer` and the 14 `ato_*`
   (13 + `ato_free`); `src/Tch.pas` has `TTensor.CopyFrom` (`at_copy_`).
3. **`src/TchNN.pas`** as sketched in DESIGN 4.6, with these small
   additions: `TVarPath.Group`; `TVarPath.FullName` validates the name, so
   `TryGet` rejects `.`/`|` too; `AddEntry` raises on a nil tensor. FPC
   warns "Constructor should be public" for the private
   `TOptimizer.CreateRaw` (harmless, kept private as designed). tch-rs
   sources used for reference: a sparse clone of `src/nn` and `tests` at
   4227b89b in the session scratchpad (not committed).
4. **`tests/nn/nn_test.pas`** (build line in its header): all ten groups of
   4.6.7 pass, plus AdamW's first step (`[0.899, -1.898]`, decay before the
   update), Load of a missing file (`ETorchNativeError`), `CopyFrom`,
   `TVarPath.SetKind` on a sub-tree, an MLP in `TSequential`, method
   pointers in `AddFn`. **H13–H18 all resolved** (results in DESIGN.md next
   to each). Two corrections to 4.6.7: (a) the `LoadPartial →
   ['a.b.t2']` case needs tch-rs's `add_partial` layout (`t123` once), the
   full layout also misses `a.b.ccc.t123__3`; (b) the lr 1e-10 check is
   signed in tch-rs: the loss really drops by 3.25e-5 (hand) / 3.24e-5
   (measured), so an absolute 1e-5 bound fails. `build/nn_test h16` runs
   only the H16 check (for `LD_LIBRARY_PATH=build/libtch-unpatched`).
5. **Mock:** `tests/mock/mock_tch.c` got the 4.6.7 stubs (optimizer
   counters, save names, load callback with `err_after_first` /
   `nil_tensor` file names, `set_requires_grad`, `set_data`, `fill_`,
   `uniform_`, `copy_`, grad mode); `tests/mock/nn_mock_test.pas`; `run.sh`
   runs both programs. In the mock an in-place op returns a copy, not a
   second handle (plumbing only).

- Regression: `tests/api`, `tests/generated`, `tests/ownership`,
  `tests/compiler/run.sh` pass.
- **New measurement, (H2) addendum** (DESIGN 4.3): an open-array literal
  takes its element type from its first element. `[0.9, -1.9]`, `[3.14]`,
  `[1e-3]` are **ambiguous** between the `Single` and `Double` overloads of
  `FromArray`; `[1.0, 0.1]` silently picks `Single`. Workaround in the
  tests: `[Double(0.9), -1.9]`.

## Handing back to the cloud (2026-10-10, end of local session 3) — answered in cloud session 5 above

**Design only** (CLAUDE.md). Mark what needs real libtorch as H19, ...

1. **JIT milestone (`TchJit`).** DESIGN 4.6 fixes the shape (`TJitModule`
   class, `TIValue` tagged record, H10 resolved, `TTrainableJitModule =
   class(TModule)`). Not yet mechanical: which `atm_*` / `ati_*` entry
   points of `torch_api.h` (lines 201–264) are declared and how
   (`atm_forward` vs `atm_forward_` with `ivalue`, `atm_method[_]`,
   `atm_named_parameters` callback — same trampoline pattern as 4.6.3?);
   `TIValue` ↔ `ivalue` conversion and ownership (`ati_free`, tuples,
   generic lists/dicts, which tags in milestone 3); `atm_to`/`atm_eval`/
   `atm_train`; how `TTrainableJitModule` registers parameters into a
   `TVarPath` (names, trainable flag). **Test assets without Python:**
   tch-rs's tests load `tests/foo*.pt` made by `create_jit_models.py`;
   either vendor those files (license: same MIT/Apache as tch-rs) or create
   modules in Pascal through `atm_create_for_tracing` / `atm_end_tracing` /
   `atm_save`. Hand-checkable numbers as in 4.6.7.
2. **`FromArray` literals (H2 addendum).** Decide: document the first-element
   rule only, add named `FromSingles`/`FromDoubles`, or change the overload
   set (e.g. drop `Single` so real literals pick `Double`, which breaks
   PyTorch's float32 default). Consider also `[1.0, 0.1]` rounding silently.
3. **Review the deviations of local session 3 from the design** (details in
   "Local session 3" above and next to each item in DESIGN.md). Confirm
   or correct each, and make DESIGN 4.6 / 4.6.7 say what the code and
   tests now do:
   - 4.6.7 test 3: the lr 1e-10 bound. DESIGN now says tch-rs's check is
     signed and the loss drops by 3.25e-5 (hand) / 3.24e-5 (measured);
     check the arithmetic.
   - 4.6.7 test 7: `LoadPartial → ['a.b.t2']` needs tch-rs's
     `add_partial` layout (`t123` once); with the full layout
     `a.b.ccc.t123__3` is missing too. Not yet corrected in DESIGN 4.6.7.
   - Additions not in the sketch: `TVarPath.Group`; `TVarPath.FullName`
     validates the name (so `TryGet` rejects `.`/`|` too); `AddEntry`
     raises on a nil tensor.
   - FPC warning "Constructor should be public" for the private
     `TOptimizer.CreateRaw`: keep private, or another construction
     pattern?
   - Test additions beyond 4.6.7 (AdamW first step `[0.899, -1.898]`, Load
     of a missing file, `CopyFrom`, sub-tree `SetKind`, MLP, method
     pointers in `AddFn`): adopt into 4.6.7 or leave as test-only.
4. **Optional, later:** milestone 3 layers (DESIGN 4.6.5 deferred list),
   CUDA/ROCm (H13's real device move is untested; CPU only so far).

Keep DESIGN/HANDOFF (en + jp) current at the end, as usual.

## Cloud session 4 (2026-10-10): DESIGN 4.6 settled — implemented in local session 3 above

The request below ("Handing back to the cloud, end of local session 2") is
answered: DESIGN.md 4.6 now has the interface sketch plus subsections
4.6.1–4.6.7, each grounded in a named tch-rs file at 4227b89b. Nothing was
committed; no code was written outside the scratchpad. Summary of the
decisions:

1. **Init (4.6.1).** `TInit` carries `Kind`, `A`, `B`, and the Kaiming
   payload (`Dist`, `Fan`, `NonLin`, `Gain`). Fan and gain formulas from
   `init.rs` (fan-in = `Dims[1] * prod(Dims[2..])`, ReLU gain `sqrt(2)`,
   uniform bound `sqrt(3) * gain / sqrt(fan)`). `Make` = `f_init` op for op
   (`zeros`/`ones`/`mul_scalar`, `uniform_`, `randn`, `empty.normal_`,
   `t_`, `linalg_qr(…, 'reduced', Q, R)`, `diag(0).sign`, `mul_`,
   `mul_scalar_`, `contiguous`, then `.view(Dims)` — the one stated
   deviation, PyTorch's shape rule). No grad guard in `Make`; `Apply`
   (tch-rs `Init::set`) guards itself. Store kind is used everywhere tch-rs
   hard-codes `Float`.
2. **Store (4.6.2).** Append-only `array of TVarEntry` (name, tensor,
   trainable, group) + `array of Integer` of trainable indices (tch-rs
   `trainable_variables`) + `TDictionary<String, Integer>` index (measured
   on 3.2.2, case-sensitive). Duplicate `Add` renames to `<name>__<Count>`
   as tch-rs; `.`, `|` and empty names raise; `GetOrAdd` returns the
   existing tensor regardless of dims. `Freeze`/`Unfreeze` =
   `set_requires_grad` on the trainable entries; `SetDevice`/`SetKind` =
   `set_data(to/totype)` in place; `SetFloatKind` for the `half()` family;
   `CopyFrom` two-pass, under `NoGradGuard`. Equality/containers question
   closed: no `=` on `TTensor` needed.
3. **Save/Load (4.6.3).** `Save` → `at_save_multi` (`.`→`|`); `Load` /
   `LoadPartial` → `at_load_callback_with_device` (`.bin`/`.pt` →
   `at_loadz_callback_with_device`; `.safetensors` raises). `cdecl`
   trampoline with a `TLoadContext` class instance in `data`, exception
   captured with `AcquireExceptionObject`, re-raised after the C call,
   error slot cleared on every path; measured with a C harness. Precision
   update on load (`set_data(totype(file kind))` then copy) as tch-rs.
   `at_load_multi[_]` unused.
4. **Optimizers (4.6.4).** `PTorchOptimizer = Pointer`, externals as in
   `torch_api.h`; construction adds all trainable entries, `ZeroGrad`/
   `Step`/`BackwardStep*` lazily add new ones (`FVarsInOptimizer`).
   Defaults from `nn/optimizer.rs` (AdamW weight decay 0.01, RmsProp alpha
   0.99). `ClipGradValue` = `grad.clamp_`; `ClipGradNorm` = `norm` per
   grad, `stack(...).norm`, `mul_scalar_(coef)`; both under `NoGradGuard`.
   **Fork bug found:** `ato_set_momentum_group` has `} if` instead of
   `} else if` (`torch_api.cpp:564`), to be patched locally.
5. **Layers (4.6.5).** `TLinear` (`linear`), `TConv2D` (`conv2d`, scalar
   `ConvConfig`, padding modes via `pad`), `TBatchNorm` (Nd 1/2/3,
   `batch_norm`, uses `Train`), `TLayerNorm` (`layer_norm`), `TEmbedding`
   (`embedding`), dropout as a function. Defaults and variable names per
   tch-rs file in the 4.6.5 table. Conv1D/3D, ConvTranspose, GroupNorm,
   LSTM/GRU deferred to milestone 3.
6. **Sequential (4.6.6).** `specialize TObjectList<TModule>`; `AddFn` with
   four overloads (function, function+Train, method, method+Train;
   resolution measured); `Train` passed to every child; `ForwardAll`.
7. **Unit/tests (4.6.7).** Single `TchNN.pas`; `TchSys` gets the
   optimizer/save/load externals, `Tch.pas` gets `TTensor.CopyFrom`;
   `tests/nn/nn_test.pas` with the hand-checkable numbers listed there
   (initial loss 22.99967, first SGD step `w = 0.80955`, `b = 0.08974`,
   Adam first step `lr·sign(g)`, clip numbers from tch-rs, conv padding
   vectors, `LoadPartial → ['a.b.t2']`); the mock covers bookkeeping,
   callback and lazy-add counts only.

**Measurements in the cloud (FPC 3.2.2, gcc 13; scratchpad only):**
- M1: `cdecl` trampoline called three times from a C harness with a class
  instance in `data`; `AcquireExceptionObject` in the handler, re-raise after
  the call: class and message intact, later callbacks skipped, all handles
  owned.
- M2: `Generics.Collections` with `TTensor` against the mock libtch:
  `TDictionary<String, Integer>` case-sensitive, duplicate `Add` →
  `EListError`; `array of record` with a `TTensor` field and
  `TList<TTensor>` release handles correctly (only the H1 expression
  temporary lingers until routine exit); `TObjectList<TModule>` with an
  abstract base and `for .. in` works; `Contnrs.TObjectList` too.
- M3: four `AddFn` overloads on function / method pointers with and without
  `Train` resolve unambiguously in objfpc, chaining works.
- Side effect: `build/mock/libtch.so` was stale (built before local session
  2's stubs) and has been rebuilt with `tests/mock/run.sh` (45 checks pass).
  `build/` is git-ignored; nothing to commit.

**New hypotheses (need real libtorch):**
- **(H13)** `set_data` keeps shared handles in sync: after `SetDevice` /
  `SetKind` a layer's `Weight` and an optimizer created earlier see the new
  storage (tch-rs relies on this).
- **(H14)** Orthogonal init: `Q·Qᵀ = I` for `[3, 5]`, `Qᵀ·Q = I` for `[5, 3]`
  (`allclose` 1e-5), result shape = `Dims`.
- **(H15)** Load precision update: a `kFloat` store loading a `kHalf` file
  ends up `kHalf` with the right values.
- **(H16)** Unpatched `ato_set_momentum_group` raises "unexpected optimizer"
  on Adam; patched (`} else if`) does not.
- **(H17)** `TInit.Apply` on a trainable leaf fails without `NoGradGuard`
  ("a leaf Variable that requires grad is being used in an in-place
  operation") and works with it.
- **(H18)** `Freeze; Step` leaves the values unchanged and raises nothing
  (libtorch skips parameters with an undefined grad).

**Local session, in this order (mechanical):** (steps 1–6 done in local session 3)
1. Fork patch: `csrc/libtch/torch_api.cpp` `ato_set_momentum_group`:
   `} if (auto sgd` → `} else if (auto sgd`; rebuild libtch; `tests/abi`,
   `tests/mock/run.sh`.
2. `src/TchSys.pas`: `PTorchOptimizer`, `TTorchLoadCallback`,
   `at_save_multi`, `at_load_callback_with_device`,
   `at_loadz_callback_with_device`, the 13 `ato_*` (4.6.4 lists the
   signatures). `src/Tch.pas`: `TTensor.CopyFrom`.
3. `src/TchNN.pas`, in this order so each part compiles alone: `TInit`
   (4.6.1) → `TVarPath`/`TVarStore` without save/load (4.6.2) → save/load
   (4.6.3) → `TOptimizer` (4.6.4) → `TModule`, `TFuncModule`, `TSequential`
   (4.6.6) → the five layers (4.6.5).
4. `tests/nn/nn_test.pas` with the ten groups of 4.6.7, checking H13–H18
   as they come up; record each result next to the (Hn) in DESIGN.md.
5. Mock: add the stubs of 4.6.7 to `tests/mock/mock_tch.c` and a
   `tests/mock/nn_mock_test.pas` (or extend `mock_test.pas`) for the
   bookkeeping checks.
6. Update DESIGN/HANDOFF (en + jp); then hand JIT (`TchJit`) and CUDA back
   to a design session if needed.

## Handing back to the cloud (2026-10-10, end of local session 2) — answered in cloud session 4 above

**Design only.** Do not write committed code (units, generator, tests);
small throwaway measurements to settle a question are fine (CLAUDE.md).
Implementation comes back to a local session. Mark anything that needs
real libtorch to confirm as a new hypothesis (H13, ...) for the local
session; the cloud container has FPC but no libtorch (the mock
`libtch.so` in `tests/mock` is available).

Milestone 1 is done (see "Local session 2" below). Next is DESIGN 4.6
(optimizers, VarStore/nn). 4.6 fixes the overall shape, but these points
are not settled enough to implement mechanically:

1. **Initialisation (`TInit.Make`).** Exact formulas, following tch-rs
   `nn/init.rs` at 4227b89b: fan-in/fan-out from `Dims` (which dims, conv
   kernels), `KaimingUniform` gain and bound (tch-rs
   `DEFAULT_KAIMING_UNIFORM`), `KaimingNormal`, `Orthogonal` (which
   generated ops: `linalg_qr`? sign fix?), and whether `Make` runs under
   `NoGradGuard`. Which generated ops each one calls (names as in
   `src/TchTensorGenerated_intf.inc`).
2. **`TVarStore` internals.** Container for named variables (ordering
   matters for `Variables`/`TrainableVariables` and for the optimizer's
   lazy add), the trainable flag and group per variable, name validation
   (tch-rs rejects `.` inside a name), behaviour of `Add` on a duplicate
   name, and what `Freeze`/`Unfreeze`/`SetDevice`/`SetKind`/`CopyFrom` do
   to existing handles (in place, under no-grad?). Equality/containers
   open question: `Generics.Collections` vs a plain array.
3. **Save/Load.** C entry points: `at_save_multi(tensors, names, n,
   filename)`, `at_load_multi(tensors, names, n, filename)` (nil array in,
   new tensors out), `at_load_multi_` (copies into existing), and the
   callback forms `at_load_callback[_with_device]` /
   `at_loadz_callback[_with_device]` with
   `void (*f)(void *data, char *name, tensor t)`. Decide which ones
   `Save`/`Load`/`LoadPartial` use, as tch-rs does. **Constraint from
   step 4:** a Pascal exception must never propagate through C++ frames,
   so a Pascal callback must not raise; it records the error and the
   caller raises after the C call returns. The callback declaration
   (`cdecl`, how `data` carries the Pascal object) belongs in the design.
   `.` ↔ `|` in names as tch-rs.
4. **Optimizers.** C signatures: `ato_sgd(lr, momentum, dampening,
   weight_decay, int nesterov)`, `ato_adam/ato_adamw(lr, beta1, beta2,
   weight_decay, eps, bool amsgrad)`, `ato_rms_prop(lr, alpha, eps,
   weight_decay, momentum, int centered)`, `ato_add_parameters(opt, t,
   size_t group)`, `ato_set_learning_rate[_group]`, `ato_set_momentum[_group]`,
   `ato_set_weight_decay[_group]`, `ato_zero_grad`, `ato_step`, `ato_free`
   (all `PROTECT`ed, so `TchCheck` after each). Decide: the
   `PTorchOptimizer` type, how lazy adding of new trainable variables
   works (tch-rs `variables_in_optimizer`), the gradient clipping
   implementation (`ClipGradValue`/`ClipGradNorm` in Pascal over
   generated ops: which ones, under no-grad), and AdamW/RmsProp default
   values (tch-rs's).
5. **Layers for milestone 2.** Which ones besides `TLinear` (tch-rs order
   of popularity: `Conv2D`, `BatchNorm`, `Embedding`, `LayerNorm`,
   `Dropout` as a function, `LSTM`/`GRU`?), each with its config record,
   default init (tch-rs's), and the generated op its `Forward` calls.
   `TLinear` forward: `matmul` + bias or `linear`?
6. **`TSequential` / `TFuncModule`.** `TObjectList` vs
   `specialize TObjectList<TModule>`, `AddFn` with plain function
   pointers vs `of object`, and whether the sequential needs the train
   flag passed down (it does: `Forward(X, Train)` to every child).
7. **Unit layout and tests.** `TchNN` (and maybe `TchOptim`) units, what
   goes where; a milestone 2 acceptance test with hand-checkable numbers
   (e.g. linear regression with SGD converging to known weights, Adam on
   a tiny problem, save → load round trip, `LoadPartial` missing names),
   plus which parts the mock library can cover.

Keep DESIGN/HANDOFF (en + jp) current at the end, as usual. Local state
the design can rely on: everything in "Local session 2" below.

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
## Cloud session 3, part 2: step 8 implemented in the cloud

The generator was designed *and built* here, because the design could be
measured: FPC 3.2.2 was installed in the container, OCaml came from apt
(`yaml` could not: opam's repository is unreachable through the proxy, so
gen.ml ran against a 150-line stand-in `Yaml` module kept in the session's
scratchpad, not committed). Everything in DESIGN.md section 5 is what the
committed code does.

- **gen.ml** writes `gen/ops.tsv` (`gen.exe <yaml> <csrc/libtch> <ops.tsv>`;
  no arguments = upstream behaviour). The regenerated
  `torch_api_generated.{h,cpp}` were byte-identical to the committed ones.
  `gen/ops.tsv` is committed (2,657 ops).
- **gen/tchgen.pas** (Pascal emitter) generates `src/TchSysGenerated.pas`,
  `src/TchTensorGenerated_intf.inc`, `src/TchTensorGenerated_impl.inc`
  (committed). `src/Tch.pas` now includes them, has the full `TKind`
  (`kNone`, 0..45), `TLayout`, `TDevice`, `TScalar`, `TOptInt64`,
  `TOptDouble`, `TOptIntList`, the nested `TTensorArray`, the helpers the
  generated bodies use, and `ETorchNativeError.Detail`; the stand-in ops
  are gone. `Tch.pas` with all ops compiles in 2.7 s.
- **Decisions** (details in DESIGN.md): error message split at
  `Exception raised from` (4.2); `TKind` extended to 45 with `kNone` for
  optional dtypes, no `TOptKind` (4.3); autocast functions in the fork
  patched to the device-taking API, warning handler deferred because c10's
  handler is thread-local (4.2); default-parameter rule and its numbers
  (4.4); results-that-are-inputs ops have no out parameters (4.4);
  `FromArray` overloads without `Int32` (4.3).
- **tests/mock**: a mock `libtch.so` (C, no libtorch) and `mock_test.pas`
  exercising every generated op shape; 45 checks pass (`tests/mock/run.sh`).
  `tests/compiler/run.sh` still passes. `tests/ownership/ownership_test.pas`
  compiles (not linked here; `ones` now takes `TDevice.Cpu`).

## Local session 2 (2026-10-10): steps 1-6 below done

- **1.** libtch rebuilt with the autocast patch; `tests/abi` passes and no
  `[W` warning line is printed any more (the deprecated-API warning on every
  `at_autocast_*` call is gone). `tests/fpu/run.sh`: same results as local
  session 1.
- **2.** gen.exe built with the real opam `yaml`; run against
  `Declarations-v2.13.0.yaml` from tch-rs 4227b89b (sparse clone of that one
  file, outside the repo): `ops.tsv` and `torch_api_generated.{h,cpp}`
  **byte-identical**, so the cloud's stand-in YAML parser typed every
  default the same way.
- **3.** tchgen: no diff (2657 ops; 479 with defaults, 149 with out
  parameters, 97 fill tensors passed in).
- **4.** `tests/ownership` passes against real libtorch (RSS flat over 10^6
  iterations, 1.4 s); `tests/mock/run.sh` passes.
- **5.** New `tests/generated/generated_test.pas`, all passing, against
  hand-worked values (no Python). Measured:
  - `sum_dim_intlist`: `NoIntList` and `IntList([])` both reduce over all
    dims (0-d, 15) for `sum`, as in PyTorch.
  - `pad(..., 'constant', NoDouble)` fills 0; `9.0` converts to `TOptDouble`.
  - `linalg_norm(2, NoIntList)` on a 2-d tensor is the **matrix** 2-norm
    (largest singular value, sqrt 54 here), not the vector norm; my first
    expectation (sqrt 55) was wrong, libtorch follows PyTorch's
    `torch.linalg.norm`. On a 1-d tensor it is the vector norm.
  - `to_sparse(ltNone, ...)` (nullopt layout) gives sparse COO like
    `ltSparse`.
  - **Error split verified** on a real message: `Message` = "mat1 and mat2
    shapes cannot be multiplied (2x3 and 2x3)", `Detail` = 2,217 chars
    starting with `Message` and containing `Exception raised from`.
  - `TchLiveHandles = 0` at the end.

- **6. Done: hand-written API of milestone 1** (`src/Tch.pas`), test
  `tests/api/api_test.pas` (build line in its header), all passing:
  - `Numel`, `Device`, `ShallowClone`, `Copy`; `FromArray` (Single,
    Double, Int64, Byte, Boolean), `FromInt32Array`, `FromBytes`;
    `To{Single,Double,Int64}Array` (convert first), `CopyData` (own kind).
  - Operators: `+ - * /` on tensors, unary `-`, `** Double`, and
    `+ - * /` with `Double`/`Int64` in both orders. `d - A` is
    `rsub_scalar`; `d / A` is `reciprocal * d` because `pow` with integer
    exponent `-1` fails on Int64 tensors (measured; DESIGN 4.1 table).
  - Autograd: `Backward(KeepGraph, CreateGraph)`, `RequiresGrad`,
    `ZeroGrad`; `grad`/`requires_grad_` are the generated ops. Grad mode
    and globals (4.7): `SetGradEnabled`, `NoGradGuard` (bare call has no
    effect, tested), `ManualSeed`, `Set/GetNumThreads`, `CudaIsAvailable`,
    `CudaDeviceCount`, `TDevice.CudaIfAvailable`.
  - Factory overloads `zeros/ones/empty/rand/randn(ASize; AKind = kFloat)`
    on CPU. Measured: no `overload` directive needed in `objfpc`; tchgen
    now treats `overload;` members as deliberate overloads and reads
    multi-line declarations whole (DESIGN 4.4).
  - `KindElementSize`: table checked against libtorch for all 46 kinds
    (DESIGN 4.3).
  - A local named `T` in a `TTensor` method collides with the generated
    `t` op (compile error, the 4.4 pitfall); hand-written locals avoid
    single-letter op names.
  - `tests/mock/mock_tch.c` got stubs for the new hand-written externals.

Next: done in cloud session 4 (see the top of this file); then the local implementation of milestone 2 as listed there.

Local session, in this order (all mechanical):

1. **Build libtch** again: `csrc/libtch/torch_api.cpp` has the autocast
   patch (`is_autocast_enabled(at::kCUDA)` / `set_autocast_enabled`), not
   compiled in the cloud. Then `tests/abi` and `tests/fpu` as before.
2. **Reproduce `gen/ops.tsv`** with the real toolchain:
   `(cd gen && dune build ./gen.exe)` then
   `gen/_build/default/gen.exe <tch-rs>/third_party/pytorch/Declarations-v2.13.0.yaml csrc/libtch gen/ops.tsv`;
   `git diff` should be empty (ops.tsv and the C files). If a default
   string differs, the stand-in YAML parser typed a scalar differently:
   fix `default_text` in gen.ml, rerun tchgen, commit the real output.
3. **Run tchgen** (`fpc -FUbuild/gen -obuild/gen/tchgen gen/tchgen.pas &&
   build/gen/tchgen gen/ops.tsv src`), expect no diff.
4. **Run the ownership test** against real libtorch (build line in its
   header), then `tests/mock/run.sh` (needs only gcc + fpc).
5. **First real-libtorch test of generated ops** (new, small):
   `split`, `max_dim`, `sort_values`, `cat`, `add_scalar(1.5)`,
   `sum_dim_intlist(IntList([]))` vs `NoIntList`, `pad(..., 'constant',
   NoDouble)`, `linalg_norm` (Scalar + IntListOption + defaults), one
   `LayoutOption` op (`to_sparse`), and an error whose `Message` has no
   backtrace while `Detail` has one. Compare with tch-rs/PyTorch results.
6. Then the hand-written API of milestone 1 (DESIGN section 6):
   `FromArray` overloads, `ToDoubles`/`CopyData`, scalar operators
   (`+ 1`, `* 2.5`), `Zeros/Ones` convenience overloads (check whether
   overloading a generated name needs `overload`), autograd.

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
8. **Generator. Done in cloud session 3** (see the top of this file):
   gen.ml writes `ops.tsv`, `gen/tchgen.pas` writes `TchSysGenerated.pas`
   and the `.inc` files, everything compiles, and `tests/mock` runs the
   generated layer. Local: reproduce `ops.tsv` with the real `yaml`
   library, run the ownership test, then the first real-libtorch test of
   generated ops.

## Open questions

- **Native error message format: decided** (cloud session 3). `Message`
  = text before `#10'Exception raised from '`, `Detail` = full text
  (DESIGN.md 4.2, implemented in `ETorchNativeError.Create`). Verified
  against a real libtorch message in local session 2.
- **`TKind` beyond 26: decided** (cloud session 3). Extended to all 46
  values of 2.13 plus `kNone = -1` for optional dtypes (DESIGN.md 4.3).
  Element sizes resolved in local session 2: `FromBytes`/`CopyData`
  accept every kind, sizes checked against libtorch (DESIGN.md 4.3).
- **libtorch warnings: decided** (cloud session 3). Fork patch of the two
  autocast functions only (compiled and tested in local session 2); c10's warning
  handler is thread-local, so a Pascal callback is deferred until needed
  (DESIGN.md 4.2).
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
- **Multi-output ops**: `out` parameters (decided; 149 ops), and no out
  parameters for the 97 ops whose results are the tensors passed in
  (cloud session 3, DESIGN.md 4.4). Still open: whether a few hand-written
  result records (`max_dim`, `sort`, `topk`, `lstm`) are worth adding
  after writing real code, and `var` instead of `out` if (H8) bites (H8
  resolved since: `out` kept, DESIGN.md 4.4).
- **Default arguments: decided** (cloud session 3, DESIGN.md 4.4): 479 ops
  get defaults; ops with out parameters none (107 would have had some).
  Local session 2: `zeros`, `ones`, `empty`, `rand`, `randn` overloads
  added; no `overload` directive is needed in `objfpc` (measured), but
  tchgen uses `overload;` as the marker for deliberate overloads
  (DESIGN.md 4.4). Still open: `topk(k)`-style defaults for out-param ops.
- **`FromArray` with real literals: decided** (cloud session 5, H2
  addendum in DESIGN 4.3). The first element decides the element type and
  a real literal is `Single` only when exactly representable, so `[0.9, …]`
  was ambiguous and `[1.0, 0.1]` rounded to `Single`. The `Single` overload
  is dropped, the others take `AKind` with the natural default (`kFloat`
  for `Double`), and `FromSingleArray` covers `Single` buffers. Implemented
  in local session 4.
- **Equality and containers: decided** (cloud session 4). `=` stays
  un-overloaded for `TTensor`. `TchNN` needs only an append-only
  `array of TVarEntry`, a `TDictionary<String, Integer>` name index and
  `TObjectList<TModule>`; all measured on 3.2.2 (DESIGN.md 4.6.2, 4.6.6).
  `TList<TTensor>` also works (byte comparer = handle identity) if ever
  needed.
- **Train flag: decided** (cloud session 4). `Forward(X, Train)` stays;
  `TSequential` passes `Train` to every child and a user `TModule` passes it
  on explicitly (DESIGN.md 4.6.6). No `Training` property.
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
- **ROCm (deferred; CPU first) — done in local session 8** (DESIGN.md 4.8
  "Results"; the notes below are the original plan). Steps 2-7 use the CPU libtorch. The local
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
- **Going public.** The license is decided (above, "State" section); the
  repo stays private until the user decides to publish it.
