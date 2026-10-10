# tch-fpc design

Japanese version: [DESIGN_jp.md](DESIGN_jp.md)

Decisions and their rationale. Open questions and next steps live in
[HANDOFF.md](HANDOFF.md).

Reference point for everything below: tch-rs commit
`4227b89b72059b0651ff83a38637693574e0a2d6` (2026-08-23, crate 0.26.0),
which targets **libtorch 2.13.0** and generates from
`third_party/pytorch/Declarations-v2.13.0.yaml`.

## 0. Overview

```
 user code (Pascal)
      |
 [L3] Tch           TTensor (managed record), TDevice, TKind, ETorchError,
      |             generated op methods; later TchNN / TchOptim / TchJit
 [L2] TchSys        raw `external` declarations, 1:1 with the C ABI
      |             (hand-written TchSys + generated TchSysGenerated)
 ---- C ABI ---------------------------------------------------------------
 [L1] libtch.so     C wrapper forked from tch-rs torch-sys/libtch,
      |             compiled with g++ against libtorch
 [L0] libtorch      libtorch_cpu.so / libtorch_cuda.so / libc10.so
```

FPC never sees C++. The only contract between Pascal and C++ is the plain C
ABI exported by `libtch`.

## 1. C wrapper: fork tch-rs's libtch (decision)

Options considered:

| Option | Verdict |
|---|---|
| Use tch-rs's `torch-sys/libtch` as-is (e.g. git submodule) | Rejected |
| **Fork it into this repo, keep the ABI and names** | **Chosen** |
| Write our own C wrapper | Rejected |

Why not as-is:
- `torch_api.cpp` calls `tch_write_stream_*` / `tch_read_stream_*`, which are
  *implemented in Rust* (`torch-sys/src/lib.rs`). Linking libtch without the
  Rust side leaves unresolved symbols. We would have to export those exact
  names from Pascal, which ties our build to tch-rs internals.
- It also pulls in `torch_python.*` (pybind) and the stb image headers,
  neither of which we need at first.

Why not our own:
- The generated part is the value: `torch_api_generated.{h,cpp}` is ~19k
  lines covering ~2,650 ops, produced by `gen/gen.ml` from PyTorch's
  `Declarations.yaml`, with years of special cases (excluded functions,
  `no_tensor_options`, optional-scalar handling). Rewriting that buys nothing.
- The ABI is already language-neutral: opaque pointers, `int64_t`, `double`,
  `int`, pointer+length pairs, and errors via a thread-local string.

Fork rules:
- Vendor into `csrc/libtch/`: `torch_api.h`, `torch_api.cpp`,
  `torch_api_generated.h`, `torch_api_generated.cpp`, plus `gen/gen.ml` into
  `gen/`. Copy tch-rs `LICENSE-MIT` and `LICENSE-APACHE` next to them.
  Commit messages name the source files (see CLAUDE.md).
- **Keep tch-rs's C names and calling shapes unchanged** (`at_*`, `atg_*`,
  `ats_*`, `ato_*`, `atm_*`, `ati_*`). This keeps the diff against upstream
  small, so bumping libtorch is "re-sync from tch-rs, re-apply our patch".
- Remove from the fork: stream functions (`*_to_stream`, `*_from_stream*`,
  `tch_*_stream_*`), `torch_python.*`, image functions (`at_load_image*`,
  `at_save_image`, `at_resize_image`) and the stb headers. They can return
  later in a Pascal-friendly form (see HANDOFF).
- Our own additions use a separate prefix `tchf_` (tch-fpc) so they are
  obviously ours. First one: `void tchf_free(void *)` — frees strings and
  arrays that libtch allocates with `strdup`/`malloc`, so Pascal never has to
  assume which C runtime's `free` is right (matters on Windows).
- Build libtch as a **shared library** (`libtch.so` / `tch.dll`). Linking
  C++ statically into an FPC program would drag libstdc++ and libtorch's
  static init into FPC's linker invocation; not worth it.

## 2. C ABI facts the Pascal side relies on

From reading tch-rs (to be **verified by building**, per CLAUDE.md):

- Handles: `tensor`, `scalar`, `optimizer`, `module`, `ivalue` are opaque
  pointers to heap-allocated C++ objects (`new torch::Tensor(...)`). Each must
  be released exactly once with `at_free` / `ats_free` / `ato_free` /
  `atm_free` / `ati_free`.
- Errors: every entry point wraps its body in `PROTECT(...)`, which catches
  `std::exception` (this includes `c10::Error`) and stores
  `strdup(e.what())` in `thread_local char *torch_last_err`.
  `get_and_reset_last_err()` returns and clears it; caller owns the string.
- Generated op shapes (`atg_*`):
  - Fixed number of tensor results: `void atg_X(tensor *out__, ...)`. Caller
    passes an array of N slots, C fills `out__[0..N-1]`.
  - Dynamic number (e.g. `split`): `tensor *atg_X(...)` returns a
    `malloc`'d, `nullptr`-terminated array. Caller frees the array (not the
    tensors) after taking ownership of each element. 33 such ops.
  - Scalar results: `bool` / `int64_t` / `double` returned directly.
- Argument encodings:

  | gen.ml type | C params | meaning |
  |---|---|---|
  | Tensor | `tensor x` | non-null |
  | TensorOption | `tensor x` | `nullptr` = none |
  | Scalar | `scalar x` | from `ats_int` / `ats_float` |
  | Bool | `int x` | 0/1 |
  | Int64 / Double | `int64_t` / `double` | |
  | Int64Option / DoubleOption | `T x_v, uint8_t x_null` | `x_null<>0` = none |
  | IntList | `int64_t *x_data, int x_len` | |
  | IntListOption | same | `x_data == nullptr` = none |
  | DoubleList | `double *x_data, int x_len` | |
  | TensorList / TensorOptList | `tensor *x_data, int x_len` | opt list: null elements = none |
  | String | `char *x_ptr, int x_len` | not NUL-terminated |
  | TensorOptions | `int x_kind, int x_device` | dtype + device |
  | ScalarType | `int` | `c10::ScalarType` value |
  | ScalarTypeOption | `int` | `<0` = none |
  | Device | `int` | `-1` CPU, `-2` MPS, `-3` Vulkan, `>=0` CUDA index |
  | Layout / LayoutOption | `int8_t` | `-1` = none |

- The hand-written `torch_api.h` mixes `int` and C++ `bool` for flags
  (e.g. `at_autocast_is_enabled` returns `bool`). Pascal must declare each
  exactly (`Boolean`/`ByteBool`-sized for `bool`, `cint` for `int`).

Important consequence: **the C header alone is not enough to generate a
good Pascal API.** `Tensor` vs `TensorOption`, `Bool` vs `Int` vs
`ScalarType` vs `Device` all collapse to the same C type. The Pascal
generator needs gen.ml's richer type information (section 5).

## 3. Pascal layer L2: TchSys (raw)

- Units `TchSys` (hand-written part of `torch_api.h`) and `TchSysGenerated`
  (all `atg_*`). Pure `external` declarations, no logic.
- Types: `type PTorchTensor = type Pointer;` etc. (distinct pointer types so
  a scalar handle cannot be passed where a tensor handle is expected).
- Convention: `cdecl` on every declaration (no-op on x86-64 Linux/Windows,
  correct on i386), `{$packrecords c}`, `ctypes` for `cint` etc.
- Library name in one constant: `const LibTch = 'tch';` and
  `external LibTch name 'atg_add'`. Static `external` (link-time) binding,
  not `LoadLibrary` + 2,650 function pointers.
- Names stay exactly as in C (`atg_add`, `at_free`). Reserved words in
  parameter names (`out`, `end`, `to`, ...) are escaped by the generator
  (`&end`) or suffixed (`end_`); parameter names have no ABI meaning.

Sketch:

```pascal
unit TchSys;
{$mode objfpc}{$H+}{$packrecords c}
interface
uses ctypes;
const
  LibTch = 'tch';
type
  PTorchTensor = type Pointer;  PPTorchTensor = ^PTorchTensor;
  PTorchScalar = type Pointer;

function  get_and_reset_last_err: PChar; cdecl; external LibTch;
procedure tchf_free(p: Pointer); cdecl; external LibTch;

function  at_new_tensor: PTorchTensor; cdecl; external LibTch;
function  at_tensor_of_data(vs: Pointer; dims: PInt64; ndims: csize_t;
            element_size_in_bytes: csize_t; type_: cint): PTorchTensor;
            cdecl; external LibTch;
procedure at_copy_data(t: PTorchTensor; vs: Pointer; numel: csize_t;
            element_size_in_bytes: csize_t); cdecl; external LibTch;
function  at_shallow_clone(t: PTorchTensor): PTorchTensor; cdecl; external LibTch;
function  at_dim(t: PTorchTensor): csize_t; cdecl; external LibTch;
procedure at_shape(t: PTorchTensor; dims: PInt64); cdecl; external LibTch;
function  at_scalar_type(t: PTorchTensor): cint; cdecl; external LibTch;
function  at_device(t: PTorchTensor): cint; cdecl; external LibTch;
function  at_to_string(t: PTorchTensor; line_size: cint): PChar; cdecl; external LibTch;
procedure at_free(t: PTorchTensor); cdecl; external LibTch;

function  ats_int(v: Int64): PTorchScalar; cdecl; external LibTch;
function  ats_float(v: Double): PTorchScalar; cdecl; external LibTch;
procedure ats_free(s: PTorchScalar); cdecl; external LibTch;
```

```pascal
unit TchSysGenerated;  // GENERATED, DO NOT EDIT
...
procedure atg_add(out__: PPTorchTensor; self, other: PTorchTensor); cdecl; external LibTch;
procedure atg_max_dim(out__: PPTorchTensor; self: PTorchTensor; dim: Int64;
            keepdim: cint); cdecl; external LibTch;
function  atg_split(self: PTorchTensor; split_size, dim: Int64): PPTorchTensor;
            cdecl; external LibTch;
procedure atg_zeros(out__: PPTorchTensor; size_data: PInt64; size_len: cint;
            options_kind, options_device: cint); cdecl; external LibTch;
```

## 4. Pascal layer L3: Tch (object-oriented)

### 4.1 Tensor ownership: managed record wrapping an interface (decision)

```pascal
type
  ITensorHandle = interface
    function Raw: PTorchTensor;
  end;

  TTensorHandle = class(TInterfacedObject, ITensorHandle)
  private
    FRaw: PTorchTensor;
  public
    constructor Create(ARaw: PTorchTensor);  // takes ownership
    destructor Destroy; override;            // at_free(FRaw)
    function Raw: PTorchTensor;
  end;

  TTensor = record
  private
    FHandle: ITensorHandle;   // nil = undefined tensor
  public
    // hand-written API, Pascal naming
    class function FromRaw(ARaw: PTorchTensor): TTensor; static;
    function Raw: PTorchTensor; inline;      // nil if undefined
    function Defined: Boolean; inline;
    function Dim: Integer;
    function Shape: TInt64Array;
    function Kind: TKind;
    function Device: TDevice;
    function ToString: AnsiString;
    function ShallowClone: TTensor;
    class operator +(const A, B: TTensor): TTensor;
    class operator -(const A, B: TTensor): TTensor;
    class operator *(const A, B: TTensor): TTensor;
    class operator /(const A, B: TTensor): TTensor;
    // generated op methods
    {$I TchTensorGenerated_intf.inc}
  end;
```

Why this shape:
- **Deterministic release.** Interface refcounting frees the C++ tensor (and
  GPU memory) as soon as the last Pascal reference goes away, like tch-rs's
  `Drop`. A plain `class` would force `try..finally Free` around every
  temporary, and `a.Add(b).Mul(c)` would leak the intermediate.
- **Expressions work.** Records support `class operator` overloads in
  `{$mode objfpc}{$modeswitch advancedrecords}`; the intermediate results of
  `(a + b) * c` are compiler-managed temporaries that get released.
- **Robust on FPC 3.2.2.** A record containing an interface field is managed
  by the compiler's long-standing interface refcounting. We avoid FPC's newer
  *management operators* (`class operator Initialize/Finalize/Copy`),
  which are newer and had bugs around temporaries and function results in
  3.2.x. (To be confirmed by tests, not trusted from memory.)
- **Thread-safe refcount.** `TInterfacedObject` uses interlocked
  increment/decrement.
- **Natural "none".** `Default(TTensor)` (nil handle) maps straight to
  `TensorOption = nullptr`.

Semantics:
- `B := A` shares the same C++ `torch::Tensor` object. libtorch tensors are
  themselves refcounted handles to shared storage, so this is the same
  aliasing a C++ or Python user expects. `ShallowClone` (`at_shallow_clone`)
  gives an independent handle to the same storage; `Copy`/`Clone` (generated
  ops) copy data.
- In-place ops keep the PyTorch trailing underscore (`Add_`) and return the
  same tensor for chaining, like tch-rs.
- Releasing early: `A := Default(TTensor);`.

Cost: two heap allocations per tensor (C++ `torch::Tensor` + Pascal
`TTensorHandle`). Acceptable for now; measure before optimizing.

### 4.2 Error handling (decision)

- Single API (no separate fallible/infallible variants as in tch-rs's
  `f_add`/`add`): Pascal has exceptions, so every op raises on failure.
- `ETorchError = class(Exception)`. After **every** libtch call L3 calls

  ```pascal
  procedure CheckTorch; inline;
  var P: PChar;
  begin
    P := get_and_reset_last_err;
    if P <> nil then RaiseTorchError(P);  // copies message, tchf_free(P), raises
  end;
  ```

  `RaiseTorchError` is a separate non-inline procedure so the hot path stays
  small. The error slot is thread-local on the C side and we read it on the
  same thread immediately, so threads do not interfere.
- On error, already-filled `out__` slots are freed before raising (PROTECT can
  fail after partially filling multi-output arrays).
- **Never let a Pascal exception unwind through C++ frames.** Callbacks
  passed into libtch (e.g. `at_load_callback`) are `cdecl` trampolines that
  catch everything, store it, and re-raise after the C call returns.
- **FPU exception mask.** FPC programs on x86 run with floating-point
  exceptions unmasked; libtorch/MKL/oneDNN assume the C default (masked) and
  legitimately produce NaN/Inf. Unit `Tch` calls
  `SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide, exOverflow,
  exUnderflow, exPrecision])` in its `initialization`. Worker threads created
  by libtorch's own pool are unaffected; the issue is the main thread and
  Pascal-created threads. Verify with a test that divides by zero on CPU.

### 4.3 Small value types

```pascal
type
  TKind = (kUInt8 = 0, kInt8 = 1, kInt16 = 2, kInt = 3, kInt64 = 4,
           kHalf = 5, kFloat = 6, kDouble = 7, kComplexHalf = 8,
           kComplexFloat = 9, kComplexDouble = 10, kBool = 11,
           kQInt8 = 12, kQUInt8 = 13, kQInt32 = 14, kBFloat16 = 15);
           // values = c10::ScalarType for libtorch 2.13; verify against
           // c10/core/ScalarType.h before relying on them

  TDeviceType = (dtCpu, dtCuda, dtMps, dtVulkan);
  TDevice = record
    DeviceType: TDeviceType;
    Index: Integer;               // CUDA only
    function ToC: cint;           // -1 / >=0 / -2 / -3
    class function Cpu: TDevice; static;
    class function Cuda(AIndex: Integer = 0): TDevice; static;
    class function CudaIfAvailable: TDevice; static;
  end;

  TScalar = record                // converted to a temporary ats_* handle per call
    IsFloat: Boolean;
    I: Int64; F: Double;
    class operator :=(V: Int64): TScalar;
    class operator :=(V: Double): TScalar;
  end;

  TOptInt64  = record HasValue: Boolean; Value: Int64;  class operator :=(V: Int64): TOptInt64;  end;
  TOptDouble = record HasValue: Boolean; Value: Double; class operator :=(V: Double): TOptDouble; end;
  TOptKind   = record HasValue: Boolean; Value: TKind;  class operator :=(V: TKind): TOptKind;   end;
  // Default(TOptXxx) = none. Hand-written, not generic: fewer 3.2.2 surprises
  // with implicit operators in generic records.

  TInt64Array  = array of Int64;
  TTensorArray = array of TTensor;
```

Parameter mapping used by the generator:

| gen.ml type | Pascal parameter |
|---|---|
| Tensor | `const X: TTensor` (must be defined; checked) |
| TensorOption | `const X: TTensor` (undefined = none) |
| Scalar | `const X: TScalar` |
| Bool | `X: Boolean` |
| Int64 / Double | `X: Int64` / `X: Double` |
| Int64Option / DoubleOption | `const X: TOptInt64` / `TOptDouble` |
| IntList | `const X: array of Int64` (allows `[2, 3]`) |
| IntListOption | `const X: TOptIntList` (open question, see HANDOFF) |
| DoubleList | `const X: array of Double` |
| TensorList / TensorOptList | `const X: array of TTensor` |
| String | `const X: AnsiString` |
| TensorOptions | `AKind: TKind; const ADevice: TDevice` |
| ScalarType / ScalarTypeOption | `TKind` / `TOptKind` |
| Device | `const X: TDevice` |

Results: one tensor → `TTensor`; dynamic → `TTensorArray`; N fixed tensors
→ `out` parameters (see 4.4); `bool`/`int64_t`/`double` → `Boolean`/`Int64`/
`Double`.

### 4.4 Generated op methods and naming (decision)

- Every `atg_*` whose first argument is `self: Tensor` becomes a method of
  `TTensor`; the rest become `class function ... static` on `TTensor`
  (e.g. `TTensor.zeros([2, 3], kFloat, TDevice.Cpu)`), mirroring tch-rs's
  `Tensor::zeros`.
- **Op names are kept verbatim from tch-rs/PyTorch in snake_case**
  (`t.max_pool2d(...)`, `t.add_(u)`, `t.sum_dim_intlist(...)`).
  Hand-written API uses Pascal CamelCase (`Shape`, `Kind`, `ToString`).
  Rationale: snake_case→CamelCase is lossy (`_foo`/`foo`/`foo_` collide),
  verbatim names are searchable in PyTorch/tch-rs docs, and the split makes
  "generated vs hand-written" visible at the call site.
- Checked on tch-rs 4227b89: the 2,657 `atg_*` names contain no uppercase
  letters and no case-insensitive duplicates, so Pascal's case-insensitivity
  is not a problem among themselves. The generator must still check
  generated names against hand-written member names case-insensitively.
- Reserved words: ops `div`, `to`, `set`, `repeat`, `absolute` and parameter
  names `out` (819 uses), `end`, `to` collide with Pascal keywords. Methods
  use FPC's `&` escape (`t.&div(u)`, `t.&to(dev)`); parameters get a `_`
  suffix (`end_`). Friendlier hand-written aliases (`DivBy`, `ToDevice`) can
  be added on top where it matters.
- Fixed multi-tensor results (e.g. `max_dim` → values, indices) become
  `procedure max_dim(Dim: Int64; KeepDim: Boolean; out Values, Indices: TTensor)`.
  Output names come from `Declarations.yaml` `returns[].name` when present,
  else `Result0..N`.
- Default arguments: Pascal supports trailing defaults for simple types only.
  First version: no defaults except where gen.ml already supplies them;
  convenience overloads are hand-written later.

### 4.5 Unit layout

```
src/
  TchSys.pas                 L2 hand-written externals
  TchSysGenerated.pas        L2 generated externals           (generated)
  Tch.pas                    L3 core: TTensor, TDevice, TKind, ETorchError
  TchTensorGenerated_intf.inc  method declarations inside TTensor (generated)
  TchTensorGenerated_impl.inc  method bodies                    (generated)
  later: TchNN.pas (VarStore, Linear, Conv...), TchOptim.pas, TchJit.pas
csrc/libtch/                 forked C wrapper + CMakeLists.txt
gen/                         gen.ml (forked) + Pascal emitter
```

Generated record members must be declared inside the record, hence the two
include files.

## 5. Code generation (decision)

- gen.ml (OCaml) already parses `Declarations-v2.13.0.yaml` and knows the
  real argument kinds. We fork it and add one more output: a line-oriented
  manifest `gen/ops.tsv`, one op per line, e.g.

  ```
  add	method	fixed1	self:Tensor	other:Tensor
  max_dim	method	fixed2	self:Tensor	dim:Int64	keepdim:Bool	->values,indices
  zeros	function	fixed1	size:IntList	options:TensorOptions
  split	method	dynamic	self:Tensor	split_size:Int64	dim:Int64
  ```

- The Pascal emitter is an FPC program (`gen/tchgen.pas`, runnable with
  `instantfpc`) that reads `ops.tsv` and writes `TchSysGenerated.pas` and the
  two `.inc` files. No Python/Node (CLAUDE.md).
- Generated files are committed, so OCaml is only needed when bumping
  libtorch, not to build tch-fpc.
- Rejected: parsing `torch_api_generated.h` directly — it loses the type
  distinctions listed in section 2. Rejected for now: a YAML parser in Pascal
  — doable, but duplicates gen.ml's accumulated special cases.

## 6. Scope of the first milestone

CPU only, Linux x86-64, libtorch 2.13.0:
create tensors from Pascal arrays, read data back, print, arithmetic
operators, a handful of generated ops, error → `ETorchError`, no leaks under
a loop of 10^6 temporaries. Autograd (`at_backward`, `requires_grad_`),
optimizers, VarStore/nn, JIT modules and CUDA follow in that order.
