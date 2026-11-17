---
name: tch-fpc
description: Rules, verified examples, pitfalls and traps for writing Free Pascal code against tch-fpc (libtorch bindings, units Tch, TchNN, TchJit). Read before writing or fixing tch-fpc code.
version: 0.1.0
metadata:
  hermes:
    tags: [pascal, fpc, libtorch, pytorch, tch-fpc]
    category: software-development
---

# tch-fpc

tch-fpc 0.1.0: Free Pascal 3.2.2 bindings for libtorch 2.13.0, modelled on
tch-rs. Units: `Tch` (tensors), `TchNN` (store, layers, optimizers), `TchJit`
(TorchScript).

## When to use
- Writing, fixing or reviewing Pascal code that uses `Tch`, `TchNN` or `TchJit`.
- Not a PyTorch tutorial, and not the dialect rules (user code is
  `{$mode objfpc}{$H+}`; see the `pascal-fpc-objfpc` skill).

How to read a file of this skill: Claude Code — Read it relative to this
skill's directory (`reference/ownership.md`); Hermes —
`skill_view(name="tch-fpc", file_path="reference/ownership.md")`.

## Rules
1. **Units and build.** `{$mode objfpc}{$H+}`, `uses Tch, TchNN, TchJit;`.
   Build with `scripts/build-example.sh <file.pas> [<libtch dir>]`; it links
   with an rpath. Never set a global `LD_LIBRARY_PATH`.
2. **Generated ops are PyTorch names in snake_case**: `T.max_pool2d(...)`,
   `T.add_(U)`, `T.sum_dim_intlist(IntList([1]))`. The hand-written API is
   CamelCase: `Shape`, `Kind`, `DoubleValue`, `Backward`.
3. **Reserved words take `&`**: `T.&div(U)`, `T.&to(Dev)`, parameters
   `&end`, `&out`, `&index`. A parameter named `result` is `result_`.
4. **Ops without a self tensor are class functions**:
   `TTensor.arange(6, kFloat, TDevice.Cpu)`, `TTensor.cat([A, B], 0)`.
   Shortcuts with CPU and kFloat defaults: `TTensor.zeros([2, 3])`,
   `ones`, `rand`, `randn`, `empty`.
5. **Optional arguments**: pass a value, or `NoInt64`, `NoDouble`,
   `NoIntList`; a list is `IntList([..])` (`IntList([])` is empty, not none).
   `A.argmax(NoInt64)` vs `A.argmax(1)`.
6. **`TTensor` is a managed record: never `Free` it.** `B := A` shares the
   tensor; `ShallowClone` is a new handle on the same storage; `Copy` is new
   storage. `const` parameters can still be changed in place.
   `IsNil` = no tensor at all; `Defined` = libtorch's defined().
7. **Classes are freed by hand** in `try..finally`: optimizers first, then
   modules, then the `TVarStore` last. A `TVarPath` must not outlive its store.
   `TSequential` owns and frees the modules added to it.
8. **Literals**: an open array is typed by its first element.
   `FromArray([1.0, 2.0])` is kFloat, `FromArray([1, 2])` is kInt64,
   `[1, 2.5]` does not compile. Single or Int32 data: `FromSingleArray`,
   `FromInt32Array`; raw memory: `FromBytes(Buf, Shape, Kind)`.
9. **Data out**: `DoubleValue([i, j])`, `Int64Value([])` for 0-d,
   `ToDoubleArray`, `ToSingleArray`, `CopyData(Buf, Count)` (own kind).
   Print floats with `:0:3`, never a raw `ToString` in tests.
10. **Operators**: `+ - * /` are elementwise with broadcasting; `**` takes a
    Double; scalar operands on either side. Matrix product: `matmul`.
11. **Autograd**: `set_requires_grad(True)`, `Backward`, `grad`, `ZeroGrad`.
    `G := NoGradGuard;` in a local (`G: IInterface`); a bare `NoGradGuard;`
    does nothing. In-place updates of leaf parameters only under the guard.
12. **TchNN**: `VS := TVarStore.Create(TDevice.Cpu)`, `VS.Root / 'name'`,
    layers take a path (`TLinear.Create(VS.Root / 'l1', In, Out)`),
    `Forward(X, Train)` with `Train` default False (pass True while
    training), optimizers are class functions (`TOptimizer.Adam(VS, LR)`),
    `Opt.BackwardStep(Loss)`, `VS.Save` / `VS.Load`.
13. **No anonymous functions in FPC 3.2.2**: `TFuncModule.Create(@Relu)` and
    `TSequential.AddFn(@Relu)` take a named function or method. A `Forward`
    override repeats `Train: Boolean = False`.
14. **TchJit**: `TJitModule.Create(FileName[, Device])`, `Forward([..])` for
    a tensor result, `ForwardIV` / `TIValue` for anything else;
    `TTrainableJitModule.Create(FileName, VS.Root / 'jit')` puts the
    parameters into a store for a normal optimizer.
15. **Devices**: `Dev := TDevice.CudaIfAvailable` and create on it
    (`.&to(Dev)`). On a ROCm build "Cuda" is the AMD GPU; with an iGPU
    present set `HIP_VISIBLE_DEVICES`. `CudnnIsAvailable` is False and
    `GpuRuntimeVersion` raises on ROCm.
16. **Errors**: libtorch failures raise `ETorchNativeError` (`Message` =
    first line, `Detail` = full text); Pascal-side misuse raises
    `ETorchError` (its parent). The units mask FPU exceptions at start-up;
    do not unmask them.

## Examples
Each compiles, runs on the CPU and prints its `.expected`.

| File | Shows |
|---|---|
| `examples/01_tensor_basics.pas` | create, shape, kind, operators, scalar out |
| `examples/02_generated_ops.pas` | snake_case ops, `&` escapes, static factories, optional arguments |
| `examples/03_ownership.pas` | assignment, `ShallowClone`, `Copy`, in-place through `const` |
| `examples/04_data_in_out.pas` | `FromBytes`, `CopyData`, `FromInt32Array`, `totype` |
| `examples/05_autograd.pas` | gradient of a small function, `NoGradGuard` in a local, `ZeroGrad` |
| `examples/06_train_linear.pas` | store, `TLinear`, Adam, loss falls, free order |
| `examples/07_custom_module.pas` | a `TModule` descendant, `TSequential`, `TFuncModule` |
| `examples/08_save_load.pas` | variable names, `Save` / `Load`, `Freeze` |
| `examples/09_device.pas` | device-agnostic code |
| `examples/10_jit_run.pas` | TorchScript `Forward`, `ForwardIV` (needs a checkout) |
| `examples/11_jit_finetune.pas` | `TTrainableJitModule` with an optimizer |
| `examples/12_errors.pas` | `ETorchNativeError` and `ETorchError` |

## Pitfalls (do not compile)

| File | Mistake | Compiler says |
|---|---|---|
| `pitfalls/p01_reserved_div.pas` | `A.div(B)` | `Syntax error, "identifier" expected but "div" found` |
| `pitfalls/p02_mixed_literal.pas` | `FromArray([1, 2.5])` | `Incompatible types: got "Single" expected "Int64"` |
| `pitfalls/p03_int32_fromarray.pas` | Int32 array to `FromArray` | `Incompatible type for arg no. 1: Got "{Dynamic} Array Of LongInt"` |
| `pitfalls/p04_param_result.pas` | a parameter named `result` | `Duplicate identifier "result"` |
| `pitfalls/p05_anonymous_function.pas` | anonymous function for `TFuncModule` | `Illegal expression` |
| `pitfalls/p06_reserved_to.pas` | `A.to(Dev)` | `Syntax error, "identifier" expected but "TO" found` |

## Traps (compile, then go wrong)

| File | Trap |
|---|---|
| `traps/t01_bare_nogradguard.pas` | a bare `NoGradGuard;` has no effect |
| `traps/t02_assignment_shares.pas` | `B := A`, then an in-place op changes both |
| `traps/t03_star_is_elementwise.pas` | `*` where `matmul` was meant |
| `traps/t04_int_literal_mean.pas` | `FromArray([1, 2])` is Int64; `mean` raises |
| `traps/t05_inplace_on_leaf.pas` | in-place op on a leaf that requires grad raises |
| `traps/t06_batchnorm_train_flag.pas` | `Forward(X)` leaves batch-norm statistics alone |

Not runnable in CI, stated in `reference/errors.md` and
`reference/devices.md`: a `TVarPath` used after its store is freed,
unmasked FPU exceptions, two devices in one op, a global `LD_LIBRARY_PATH`.

## Reference (read one when needed)
`reference/`: `naming.md`, `ownership.md`, `autograd.md`, `nn.md`, `jit.md`,
`devices.md`, `build.md`, `errors.md`.

## Where to look
Paths are in a tch-fpc checkout; links are the release. For a signature,
search `src/TchTensorGenerated_intf.inc` for `function <op>(` (4,835 lines).

| Question | In a checkout | Online |
|---|---|---|
| Tensor ownership | `docs/DESIGN.md` "4.1 Tensor ownership" | [DESIGN.md](https://github.com/tankoha/tch-fpc/blob/v0.1.0/docs/DESIGN.md) |
| Errors and the FPU mask | `docs/DESIGN.md` "4.2 Error handling" | [DESIGN.md](https://github.com/tankoha/tch-fpc/blob/v0.1.0/docs/DESIGN.md) |
| Value types, `FromArray` | `docs/DESIGN.md` "4.3 Small value types" | [DESIGN.md](https://github.com/tankoha/tch-fpc/blob/v0.1.0/docs/DESIGN.md) |
| Generated ops and naming | `docs/DESIGN.md` "4.4 Generated op methods and naming" | [DESIGN.md](https://github.com/tankoha/tch-fpc/blob/v0.1.0/docs/DESIGN.md) |
| nn overview | `docs/DESIGN.md` "4.6 nn, VarStore, optimizers and JIT" | [DESIGN.md](https://github.com/tankoha/tch-fpc/blob/v0.1.0/docs/DESIGN.md) |
| Initialisation | `docs/DESIGN.md` "4.6.1 Initialisation" | [DESIGN.md](https://github.com/tankoha/tch-fpc/blob/v0.1.0/docs/DESIGN.md) |
| Store internals | `docs/DESIGN.md` "4.6.2 `TVarStore` internals" | [DESIGN.md](https://github.com/tankoha/tch-fpc/blob/v0.1.0/docs/DESIGN.md) |
| Save and load | `docs/DESIGN.md` "4.6.3 Save and load" | [DESIGN.md](https://github.com/tankoha/tch-fpc/blob/v0.1.0/docs/DESIGN.md) |
| Optimizers | `docs/DESIGN.md` "4.6.4 Optimizers" | [DESIGN.md](https://github.com/tankoha/tch-fpc/blob/v0.1.0/docs/DESIGN.md) |
| Layers | `docs/DESIGN.md` "4.6.5 Layers of milestone 2" | [DESIGN.md](https://github.com/tankoha/tch-fpc/blob/v0.1.0/docs/DESIGN.md) |
| Sequential, function modules | `docs/DESIGN.md` "4.6.6 `TSequential` and `TFuncModule`" | [DESIGN.md](https://github.com/tankoha/tch-fpc/blob/v0.1.0/docs/DESIGN.md) |
| JIT | `docs/DESIGN.md` "4.6.8 JIT (`TchJit`)" | [DESIGN.md](https://github.com/tankoha/tch-fpc/blob/v0.1.0/docs/DESIGN.md) |
| TorchScript objects | `docs/DESIGN.md` "4.6.9 TorchScript objects" | [DESIGN.md](https://github.com/tankoha/tch-fpc/blob/v0.1.0/docs/DESIGN.md) |
| Conv, norm, RNN layers | `docs/DESIGN.md` "4.6.10 Layers of milestone 3" | [DESIGN.md](https://github.com/tankoha/tch-fpc/blob/v0.1.0/docs/DESIGN.md) |
| Grad mode | `docs/DESIGN.md` "4.7 Grad mode and global settings" | [DESIGN.md](https://github.com/tankoha/tch-fpc/blob/v0.1.0/docs/DESIGN.md) |
| GPU builds, ROCm | `docs/DESIGN.md` "4.8 GPU devices" | [DESIGN.md](https://github.com/tankoha/tch-fpc/blob/v0.1.0/docs/DESIGN.md) |
| Class diagrams | `docs/UML.md` "2. Tch: tensor core" | [UML.md](https://github.com/tankoha/tch-fpc/blob/v0.1.0/docs/UML.md) |
| Tensor API | `src/Tch.pas` "interface" | [Tch.pas](https://github.com/tankoha/tch-fpc/blob/v0.1.0/src/Tch.pas) |
| nn API | `src/TchNN.pas` "interface" | [TchNN.pas](https://github.com/tankoha/tch-fpc/blob/v0.1.0/src/TchNN.pas) |
| JIT API | `src/TchJit.pas` "interface" | [TchJit.pas](https://github.com/tankoha/tch-fpc/blob/v0.1.0/src/TchJit.pas) |

## Before you report
Compile and run what you wrote with `scripts/build-example.sh` (set
`TCH_FPC_ROOT` to the checkout when the skill is installed as a plugin) and
quote the program's output. Code that was not run is not done.
