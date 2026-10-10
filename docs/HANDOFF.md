# Handoff

Japanese version: [HANDOFF_jp.md](HANDOFF_jp.md)

Read this first when resuming. Decisions and rationale: [DESIGN.md](DESIGN.md).

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

## Handing back to the cloud (2026-10-10, end of local session 3)

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
  autocast functions only (not yet compiled locally); c10's warning
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
  after writing real code, and `var` instead of `out` if (H8) bites.
- **Default arguments: decided** (cloud session 3, DESIGN.md 4.4): 479 ops
  get defaults; ops with out parameters none (107 would have had some).
  Local session 2: `zeros`, `ones`, `empty`, `rand`, `randn` overloads
  added; no `overload` directive is needed in `objfpc` (measured), but
  tchgen uses `overload;` as the marker for deliberate overloads
  (DESIGN.md 4.4). Still open: `topk(k)`-style defaults for out-param ops.
- **`FromArray` with real literals** (local session 3, H2 addendum in
  DESIGN 4.3): the first element decides the element type; `[0.9, …]` is
  ambiguous, `[1.0, 0.1]` rounds to `Single`. Handed to the cloud above.
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
