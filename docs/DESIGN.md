# tch-fpc design

Japanese version: [DESIGN_jp.md](DESIGN_jp.md)

Decisions and their rationale. Open questions and next steps live in
[HANDOFF.md](HANDOFF.md). Statements about FPC 3.2.2 behaviour that have not
been verified by building are marked **(H*n*)** and listed as verification
steps in HANDOFF.md; treat them as hypotheses until then.

Reference point for everything below: tch-rs commit
`4227b89b72059b0651ff83a38637693574e0a2d6` (2026-08-23, crate 0.26.0),
which targets **libtorch 2.13.0** and generates from
`third_party/pytorch/Declarations-v2.13.0.yaml`.

## 0. Overview

```
 user code (Pascal)
      |
 [L3] Tch           TTensor (managed record), TDevice, TKind, TScalar,
      |             ETorchError, generated op methods, grad-mode helpers
      |  TchNN      TVarStore / TVarPath / TInit, TModule and layers,
      |             TSequential, TOptimizer
      |  TchJit     TJitModule, TIValue
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
  lines covering 2,657 ops, produced by `gen/gen.ml` from PyTorch's
  `Declarations.yaml`, with years of special cases (excluded functions,
  `no_tensor_options`, optional-scalar handling). Rewriting that buys nothing.
- The ABI is already language-neutral: opaque pointers, `int64_t`, `double`,
  `int`, pointer+length pairs, and errors via a thread-local string.

Fork rules:
- Vendor into `csrc/libtch/`: `torch_api.h`, `torch_api.cpp`,
  `torch_api_generated.h`, `torch_api_generated.cpp`, plus `gen/gen.ml` into
  `gen/`. Copy tch-rs `LICENSE-MIT` and `LICENSE-APACHE` next to them.
  Commit messages name the source files (see CLAUDE.md).
  `third_party/pytorch/Declarations-v2.13.0.yaml` (needed only to regenerate)
  is in the tch-rs repo at the same commit; it is large and is not vendored.
- **Keep tch-rs's C names and calling shapes unchanged** (`at_*`, `atg_*`,
  `ats_*`, `ato_*`, `atm_*`, `ati_*`). This keeps the diff against upstream
  small, so bumping libtorch is "re-sync from tch-rs, re-apply our patch".
- Remove from the fork: stream functions (`*_to_stream`, `*_from_stream*`,
  `tch_*_stream_*`), `torch_python.*`, image functions (`at_load_image*`,
  `at_save_image`, `at_resize_image`) and the stb headers. They can return
  later in a Pascal-friendly form (see HANDOFF).
- Harden `PROTECT`: upstream catches only `const std::exception&`. Add a
  `catch (...)` branch that stores a fixed message, so that *no* C++ throw
  can ever unwind into Pascal frames (that would abort the process).
- `torch_api_generated.h` opens `extern "C" {` unconditionally (it is only
  ever compiled as C++ in tch-rs). Guard it with `#ifdef __cplusplus` so the
  C smoke test in HANDOFF can include it. Both patches are a few lines and
  easy to re-apply.
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
  `atm_free` / `ati_free`. `at_free` is a bare `delete`: it cannot fail and
  never sets the error slot.
- Errors: every entry point wraps its body in `PROTECT(...)`, which catches
  `std::exception` (this includes `c10::Error`) and stores
  `strdup(e.what())` in `thread_local char *torch_last_err`.
  `get_and_reset_last_err()` returns and clears it; caller owns the string.
  Hand-written functions that return a handle return `nullptr` on error;
  those returning `int` flags return `-1`.
- **Non-optional `tensor` arguments are dereferenced without a null check**
  (`*self` in the generated call). Passing `nullptr` where a `Tensor` (not
  `TensorOption`) is expected is a segfault, not a catchable error. L3 must
  validate before calling (4.2).
- Generated op shapes (`atg_*`, 2,657 in total):
  - Fixed number of tensor results: `void atg_X(tensor *out__, ...)`. Caller
    passes an array of N slots, C fills `out__[0..N-1]` *after* the op has
    succeeded. 246 ops return 2 or more tensors (up to 7).
  - Dynamic number (e.g. `split`): `tensor *atg_X(...)` returns a
    `malloc`'d, `nullptr`-terminated array. Caller frees the array (not the
    tensors) after taking ownership of each element. 33 such ops.
  - Scalar results: `bool` (as `int`) / `int64_t` / `double` returned
    directly. 44 ops. 21 ops return nothing.
- Memory ownership of data:
  - `at_tensor_of_data` **copies** (`torch::zeros` + `memcpy`): the Pascal
    buffer may be released as soon as the call returns.
  - `at_tensor_of_blob` does **not** copy: the buffer must outlive the tensor
    and every view of it. Not exposed in the first milestone.
  - `at_copy_data` copies out, going through CPU and `contiguous()` if
    needed.
  - Strings from `at_to_string`, `ats_to_string`, `ati_to_string` and the
    error slot are `strdup`'d; dynamic result arrays are `malloc`'d. All go
    to `tchf_free`.
- Callback loaders (`at_load_callback*`, `at_loadz_callback*`,
  `atm_named_parameters`) call `f(data, name, new torch::Tensor(...))` on
  the calling thread, inside `PROTECT`. The callee **owns** the tensor it
  receives. The name is borrowed and valid only during the call.
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
  | Layout / LayoutOption | `int8_t` | `-1` = none (7 ops at 2.13: `to_sparse`, `_to_sparse`, `_spdiags`, ... ) |

- The hand-written `torch_api.h` mixes `int` and C++ `bool` for flags
  (e.g. `at_autocast_is_enabled` returns `bool`). Pascal must declare each
  exactly: `Boolean` (1 byte, never `LongBool`) for `bool`, `cint` for
  `int`, `Byte` for `uint8_t`, `ShortInt` for `int8_t`, `csize_t` for
  `size_t`.

Important consequence: **the C header alone is not enough to generate a
good Pascal API.** `Tensor` vs `TensorOption`, `Bool` vs `Int` vs
`ScalarType` vs `Device` all collapse to the same C type. The Pascal
generator needs gen.ml's richer type information (section 5).

## 3. Pascal layer L2: TchSys (raw)

- Units `TchSys` (hand-written part of `torch_api.h`) and `TchSysGenerated`
  (all `atg_*`). Pure `external` declarations, no logic.
- Types: each handle is a pointer to its own empty record
  (`PTorchTensor = ^TTorchTensorOpaque;` with `TTorchTensorOpaque = record
  end;`, likewise `PTorchScalar`, `PTorchOptimizer`, `PTorchModule`,
  `PTorchIValue`), so a scalar handle cannot be passed where a tensor handle
  is expected: a mix-up is a compile error. `nil`, an untyped `Pointer` and
  explicit casts still convert.
  *(changed after the overall review, 2026-10-13.)* The first draft said
  `type PTorchTensor = type Pointer;`. *Measured* on FPC 3.2.2: `type
  Pointer` aliases are not distinguished (a `PIValue` is accepted where a
  `PTensor` is expected, as value parameter, `var` parameter and
  assignment), so that form protects nothing; pointers to distinct empty
  records give "Incompatible type" in all three cases. The whole code base,
  including `TchSysGenerated` (about 4,700 uses of `PTorchTensor`),
  recompiled without error, so no mix-up was hidden. The C side and the ABI
  are unchanged (all handles are `void*` in `torch_api.h`).
- Convention: `cdecl` on every declaration (no-op on x86-64 Linux/Windows,
  correct on i386), `ctypes` for `cint` etc. No `{$packrecords c}`: it is not
  needed, because `torch_api.h` has no struct and no record crosses the C
  boundary (only pointers, integers, doubles and arrays). Add it when a
  function that passes a record is declared.
- Library name in one constant: `const LibTch = 'tch';` and
  `external LibTch name 'atg_add'`. Static `external` (link-time) binding,
  not `LoadLibrary` + 2,657 function pointers.
- Names stay exactly as in C (`atg_add`, `at_free`). Parameter names stay
  too; reserved words among them (`out`, `end`, `to`) are escaped with `&`
  (`&out`), and `result` becomes `result_` (same rule as 4.4). Parameter
  names have no ABI meaning.

Sketch:

```pascal
unit TchSys;
{$mode objfpc}{$H+}
interface
uses ctypes;
const
  LibTch = 'tch';
type
  TTorchTensorOpaque = record end;
  PTorchTensor = ^TTorchTensorOpaque;  PPTorchTensor = ^PTorchTensor;
  TTorchScalarOpaque = record end;
  PTorchScalar = ^TTorchScalarOpaque;
  TTorchLoadCallback = procedure(data: Pointer; name: PChar; t: PTorchTensor); cdecl;

function  get_and_reset_last_err: PChar; cdecl; external LibTch;
procedure tchf_free(p: Pointer); cdecl; external LibTch;

function  at_new_tensor: PTorchTensor; cdecl; external LibTch;
function  at_tensor_of_data(vs: Pointer; dims: PInt64; ndims: csize_t;
            element_size_in_bytes: csize_t; type_: cint): PTorchTensor;
            cdecl; external LibTch;
procedure at_copy_data(t: PTorchTensor; vs: Pointer; numel: csize_t;
            element_size_in_bytes: csize_t); cdecl; external LibTch;
function  at_shallow_clone(t: PTorchTensor): PTorchTensor; cdecl; external LibTch;
function  at_defined(t: PTorchTensor): cint; cdecl; external LibTch;
function  at_dim(t: PTorchTensor): csize_t; cdecl; external LibTch;
procedure at_shape(t: PTorchTensor; dims: PInt64); cdecl; external LibTch;
function  at_scalar_type(t: PTorchTensor): cint; cdecl; external LibTch;
function  at_device(t: PTorchTensor): cint; cdecl; external LibTch;
function  at_to_string(t: PTorchTensor; line_size: cint): PChar; cdecl; external LibTch;
function  at_autocast_is_enabled: Boolean; cdecl; external LibTch;  // C++ bool
procedure at_load_callback(filename: PChar; data: Pointer;
            f: TTorchLoadCallback); cdecl; external LibTch;
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
procedure atg_narrow(out__: PPTorchTensor; self: PTorchTensor; dim, start,
            length: Int64); cdecl; external LibTch;
procedure atg_add_out(out__: PPTorchTensor; &out, self, other: PTorchTensor);
            cdecl; external LibTch;
```

## 4. Pascal layer L3: Tch (object-oriented)

### 4.1 Tensor ownership: managed record wrapping an interface (decision)

Options considered:

| Option | Verdict |
|---|---|
| Plain `class` with explicit `Free` | Rejected: every temporary needs `try..finally`; `(a + b) * c` leaks the intermediate. |
| COM-style `ITensor` interface as the public type | Rejected: operator overloading in FPC is built around records, and whether binary operators on interface types are accepted at all is unclear; 2,657 virtual methods in one interface is heavier than static record methods; and a public interface invites holding the implementing object by class reference, the classic refcount footgun. |
| Record with the raw pointer + management operators (`Initialize/Finalize/Copy`) | Rejected for now: available since 3.2.0 only, with a bug history around temporaries and function results in 3.2.x. Saves one heap allocation per tensor. Revisit only if that allocation is measured to matter. |
| Record helper for the generated ops, record for the core | Rejected (see 4.5). |
| **Record wrapping a refcounted `ITensorHandle`** | **Chosen** |

```pascal
unit Tch;
{$mode objfpc}{$H+}{$modeswitch advancedrecords}
type
  ITensorHandle = interface
    function Raw: PTorchTensor;
  end;

  TTensorHandle = class(TInterfacedObject, ITensorHandle)
  private
    FRaw: PTorchTensor;
  public
    constructor Create(ARaw: PTorchTensor);  // takes ownership
    destructor Destroy; override;            // at_free(FRaw); never raises
    function Raw: PTorchTensor;
  end;

  TTensor = record
  private
    FHandle: ITensorHandle;   // nil = no tensor at all
  public
    // --- hand-written API, Pascal naming ---
    class function FromRaw(ARaw: PTorchTensor): TTensor; static; // takes ownership
    function Raw: PTorchTensor; inline;      // borrowed; nil if IsNil
    function IsNil: Boolean; inline;         // FHandle = nil
    function Defined: Boolean;               // not IsNil and at_defined <> 0
    function Dim: Integer;
    function Shape: TInt64Array;
    function Numel: Int64;
    function Kind: TKind;
    function Device: TDevice;
    function ToString(LineWidth: Integer = 80): AnsiString;
    function ShallowClone: TTensor;          // new handle, same storage
    function Copy: TTensor;                  // new storage (zeros_like + copy_)
    // data in / out (H2)
    class function FromArray(const A: array of Single): TTensor; overload; static;
    class function FromArray(const A: array of Double): TTensor; overload; static;
    class function FromArray(const A: array of Int64): TTensor; overload; static;
    class function FromInt32Array(const A: array of Int32): TTensor; static;  // no Int32 overload (H2)
    class function FromArray(const A: array of Byte): TTensor; overload; static;
    class function FromArray(const A: array of Boolean): TTensor; overload; static;
    class function FromBytes(const Data; const Shape: array of Int64; AKind: TKind): TTensor; static;
    function ToSingleArray: TSingleArray;    // copies out, converts kind
    function ToDoubleArray: TDoubleArray;
    function ToInt64Array: TInt64Array;
    procedure CopyData(var Dest; ACount: Int64);  // own kind, no conversion
    function DoubleValue(const Index: array of Int64): Double;
    function Int64Value(const Index: array of Int64): Int64;
    // autograd
    procedure Backward(KeepGraph: Boolean = False; CreateGraph: Boolean = False);
    function RequiresGrad: Boolean;
    procedure ZeroGrad;                      // grad (generated) .detach_.zero_ if defined
    // operators (4.1 "Operators")
    class operator +(const A, B: TTensor): TTensor;
    class operator -(const A, B: TTensor): TTensor;
    class operator *(const A, B: TTensor): TTensor;    // elementwise (mul)
    class operator /(const A, B: TTensor): TTensor;
    class operator -(const A: TTensor): TTensor;       // neg
    class operator **(const A: TTensor; B: Double): TTensor;
    class operator +(const A: TTensor; B: Double): TTensor;  // and Int64, and reversed
    ...
    // --- generated op methods (4.4) ---
    {$I TchTensorGenerated_intf.inc}
  end;
```

Why this shape:
- **Deterministic release.** Interface refcounting frees the C++ tensor (and
  GPU memory) as soon as the last Pascal reference goes away, like tch-rs's
  `Drop`.
- **Expressions work.** Records support `class operator` overloads in
  `{$mode objfpc}{$modeswitch advancedrecords}`; the intermediate results of
  `(a + b) * c` are compiler-managed temporaries. **(H1) resolved** (step 6,
  FPC 3.2.2): temporaries are *not* released at the end of the statement.
  Each expression site in a routine owns a temporary slot that keeps its
  last value until the slot is reused (next evaluation, e.g. the next loop
  iteration) or the routine exits. A loop therefore does not accumulate
  (10^6 iterations: handle count and RSS flat), but a large intermediate
  stays alive until the routine returns. Mitigation for big tensors: put
  the expression in a small function, or split it into named variables
  and release them with `X := Default(TTensor)`.
- **Robust on FPC 3.2.2.** A record containing an interface field is managed
  by the compiler's long-standing interface refcounting, exercised by every
  Lazarus application. Management operators are avoided (table above).
- **Thread-safe refcount.** `TInterfacedObject` uses interlocked
  increment/decrement.
- **Natural "none".** `Default(TTensor)` (nil handle) maps straight to
  `TensorOption = nullptr`, and a `TTensor` is one pointer in size, so
  copying it around is free.

Semantics:
- `B := A` shares the same C++ `torch::Tensor` object. libtorch tensors are
  themselves refcounted handles to shared storage, so this is the same
  aliasing a C++ or Python user expects. `ShallowClone` (`at_shallow_clone`)
  gives an independent handle to the same storage; `Copy` copies data.
- Two levels of "nothing": `IsNil` (no handle; `Default(TTensor)`; what the
  generator passes as `nullptr` for `TensorOption`) and `Defined` (libtorch's
  own notion; `at_new_tensor` gives a non-nil handle to an undefined tensor,
  which is what `grad` returns when there is no gradient). Passing a nil
  handle for a non-optional `Tensor` argument raises `ETorchError` *before*
  the call, because C would dereference it (section 2).
- `Raw` is a borrowed pointer: valid only while some `TTensor` still holds
  the handle. Never store it.
- `TTensorHandle.Destroy` only calls `at_free`. It never raises and never
  reads the error slot (a destructor running during exception unwinding must
  not throw a second exception).
- In-place ops keep the PyTorch trailing underscore (`add_`) and return the
  same tensor for chaining, like tch-rs. Ignoring the result is fine
  (`{$X+}` is the default).
- Releasing early: `A := Default(TTensor);`.
- **No `=` / `<>` overload.** Equality on tensors is ambiguous (handle
  identity vs elementwise vs `allclose`); spelling it out (`SameHandle`,
  generated `equal`, `allclose`) avoids silent surprises. Consequence: the
  FGL containers (`TFPGList<TTensor>`) will not compile because they require
  `=`; use `TTensorArray` (dynamic array) or `Generics.Collections.TList<TTensor>`,
  which compares by bytes. Open question in HANDOFF.
- Data in: `FromArray` is overloaded on the element type so `TTensor.FromArray([1.0, 2.0])`
  reads naturally. **(H2)** Whether an open-array literal of real constants
  resolves unambiguously between `array of Single` and `array of Double`
  (and integer literals between `Int32` and `Int64`) must be tested; the
  fallback is distinct names (`FromSingles`, `FromDoubles`, ...). Data out
  has to be named by type (`ToDoubleArray`) because Pascal cannot overload on
  return type.
  **(H2) resolved** (step 7): real literals are never ambiguous; with both
  `Single` and `Double` overloads `[1.0, 2.0]` picks **`Single`** (Double
  variables pick `Double`). That matches PyTorch's default float32. Integer
  literals (`[1, 2]`) **are ambiguous when both `Int32` and `Int64`
  overloads exist**; without `Int32` they pick `Int64` (PyTorch's default
  int64), and the set `Single, Double, Int64, Byte, Boolean` resolves every
  literal tried. So: no `array of Int32` overload of `FromArray`; int32
  data goes through a named function (e.g. `FromInt32Array`) or `FromBytes`.
  **(H2) addendum** (local session 3, measured on 3.2.2): the open-array
  constructor takes its element type from the **first** element. A first
  element that is exact in `Single` (`1.0`, `0.5`, an integer) makes the
  whole literal `Single`, so `[1.0, 0.1]` picks `Single` and rounds `0.1`
  silently; a first element that is not (`0.9`, `3.14`, `1e-3`) makes the
  call **ambiguous** (`[0.9, 1.0]`, `[3.14]` do not compile). Workaround:
  type the first element (`[Double(0.9), -1.9]` picks `Double`,
  `[Single(0.9)]` picks `Single`). Open question in HANDOFF.

Operators (hand-written; map to generated ops):

| Pascal | op | note |
|---|---|---|
| `A + B`, `A - B`, `A * B`, `A / B` | `add`, `sub`, `mul`, `div` | `*` is **elementwise**; matrix product is `A.matmul(B)` |
| `A + d`, `A - d`, `A * d`, `A / d` | `add_scalar` ... | `d: Double` and `d: Int64` as explicit overloads **(H3)**, not via `TScalar`'s implicit operator, so that `T + 1` and `T + 1.5` both resolve without going through a user-defined conversion |
| `d + A`, `d - A`, `d * A`, `d / A` | reversed | `d - A` = `A.rsub_scalar(d)` (one op). `d / A` = `A.reciprocal * d` instead of the earlier `A.pow(-1) * d`: measured in local session 2, `pow` with the integer exponent `-1` on an Int64 tensor fails ("Integers to negative integer powers are not allowed."), `-1.0` would work; `reciprocal` promotes to float like true division, and `4 / Int64 tensor` is kFloat |
| `-A` | `neg` | |
| `A ** d` | `pow_tensor_scalar` | FPC allows overloading `**` |

**(H3) resolved** (step 7): with `Double` and `Int64` overloads in both
orders, `T + 1` and `2 * T` pick `Int64`, `T + 1.5` and `2.5 * T` pick
`Double`, an `Integer`/`Byte` variable picks `Int64`, a `Single` variable
picks `Double`; unary `-` and `**` (record operator) work.

Not overloaded: `=`, `<>`, `<`, `>` (comparisons return tensors in
PyTorch; a Boolean result would be a lie), `@`, `in`.

Cost: two heap allocations per tensor (C++ `torch::Tensor` + Pascal
`TTensorHandle`). Acceptable for now; measure before optimizing (the
management-operator variant is the escape hatch).

### 4.2 Error handling (decision)

- Single API (no separate fallible/infallible variants as in tch-rs's
  `f_add`/`add`): Pascal has exceptions, so every op raises on failure.
- Classes: `ETorchError = class(Exception)` is the base and is what
  Pascal-side validation raises (nil handle, shape helpers, unknown kind).
  `ETorchNativeError = class(ETorchError)` carries a message that came from
  libtorch's error slot. Catching `ETorchError` catches both.
- After **every** libtch call L3 calls

  ```pascal
  procedure TchCheck; inline;
  var P: PChar;
  begin
    P := get_and_reset_last_err;
    if P <> nil then TchRaiseNative(P);  // copies message, tchf_free(P), raises
  end;
  ```

  `TchRaiseNative` is a separate non-inline procedure so the hot path stays
  small. The error slot is thread-local on the C side and we read it on the
  same thread immediately, so threads do not interfere. Helpers used by
  generated bodies are prefixed `Tch` so that no op parameter name
  (`self`, `&out`, `data`, `index`, ...) can shadow them.
- **Native message format (decided, cloud session 3).** libtorch's
  `c10::Error::what()` is the message, then the context lines, then
  `"\nException raised from <function> at <file>:<line> (most recent call
  first):\n<backtrace>"` (`Error::compute_what` in `c10/util/Exception.cpp`).
  `ETorchNativeError.Create` splits at the first `#10'Exception raised
  from '`: `Message` is the text before it, the `Detail` property keeps the
  full text. No change on the C side (`what_without_backtrace()` would need
  a `c10::Error` catch clause and would lose the backtrace entirely).
  Checked against the mock in `tests/mock`; the real text was seen in step 3.
- **libtorch warnings (decided, cloud session 3).** `TORCH_WARN` goes to
  stderr through c10's default `WarningHandler`. Two findings from
  `c10/util/Exception.cpp` (2.13): the handler is **thread-local**
  (`ThreadWarningHandler::warning_handler_`), so a handler installed from
  Pascal would cover only the installing thread, not libtorch's worker
  threads or other `TThread`s; and the deprecation warnings seen in step 4
  come from `at::autocast::is_enabled()`/`set_enabled()`, which the fork
  now replaces with `is_autocast_enabled(at::kCUDA)` /
  `set_autocast_enabled(at::kCUDA, b)` in `torch_api.cpp` (same semantics,
  no warning; **not yet compiled** then, the cloud has no libtorch; compiled
  and tested in local session 2). Decision:
  that patch only for the first milestone; libtorch's own warnings stay on
  stderr, as in C++. A per-thread `tch_set_warning_handler(callback, data)`
  in libtch forwarding `Warning::msg()` and `source_location()` to Pascal
  is the follow-up if a program ever needs to capture them.
- Before a call, every non-optional `Tensor` argument (including the
  receiver) is checked for `IsNil`; a nil handle raises
  `ETorchError('undefined tensor passed as <param> to <op>')`. This is the
  only check that costs anything, and it replaces a segfault.
- Fixed-N outputs: slots are initialised to `nil`; on error, non-nil slots are
  freed before raising (in practice C fills them only after success, but
  `new torch::Tensor` could still throw `bad_alloc` part-way).
- **Never let a Pascal exception unwind through C++ frames.** Callbacks
  passed into libtch (`at_load_callback`, `atm_named_parameters`) are
  `cdecl` trampolines whose body is wrapped in `try..except`; the handler
  stores the exception object (`AcquireExceptionObject`) in the context
  record and returns normally. After the C call returns, L3 re-raises it.
  The trampoline takes ownership of the tensor it is handed (section 2) even
  when it will be discarded.
- **FPU exception mask.** FPC programs on x86 run with floating-point
  exceptions (invalid, zero-divide, overflow) *unmasked* in both the x87
  control word and MXCSR, so `0/0` raises `EInvalidOp`. libtorch, MKL and
  oneDNN assume the C default (all masked) and legitimately produce NaN/Inf;
  with the Pascal defaults a CPU kernel that divides by zero raises SIGFPE.
  Measured (step 5, FPC defaults MXCSR=$1900): on the calling thread FPC
  turns it into an `EZeroDivide` raised from inside C++ frames; in
  libtorch's OpenMP workers it is an unhandled exception and the process
  **hangs** in the parallel region. Decision: unit `Tch` calls
  `SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide, exOverflow,
  exUnderflow, exPrecision])` in its `initialization`, before any libtorch
  call can happen, and exports `procedure TchMaskFpuExceptions;` for threads.
  - libtorch's own worker pools are created lazily by the first parallel op,
    and a new pthread inherits the creating thread's FP control state. So
    the pool threads end up masked *because* `Tch` masks before the first
    call. This is why the mask must be set in `initialization`, not lazily.
    Confirmed in step 5: creating the pool unmasked and masking afterwards
    still hangs on the next parallel 1/0.
  - Threads created from Pascal (`TThread`, `BeginThread`) are initialised by
    the RTL. **(H4) resolved** (FPC 3.2.2, Linux x86_64): a `TThread`
    started after `SetExceptionMask` starts with MXCSR=$1F80 / x87CW=$137F
    (all masked), and both small and parallel tensor ops in it yield Inf.
    `TchMaskFpuExceptions` stays exported for threads created outside the
    RTL (e.g. by another C library) but is not required in `TThread`s.
    `BeginThread` was not measured separately.
  - **(H5) resolved**: after catching an integer `EDivByZero` (a real
    SIGFPE through the RTL handler) the mask is still $1F80 and tensor 1/0
    still yields Inf. A float `EZeroDivide` cannot occur in Pascal code
    while everything is masked, so nothing needs re-applying.
  - Trade-off, stated plainly: in a program that uses `Tch`, plain Pascal
    `x / 0.0` yields `Inf` instead of raising. This is the C/C++/Python
    behaviour and is acceptable for a numerical library; it is documented in
    the README rather than hidden.

### 4.3 Small value types

```pascal
type
  { every c10::ScalarType of libtorch 2.13 (torch/headeronly/core/ScalarType.h),
    decided in cloud session 3: 0-26 were verified in step 4, 27-45 exist and
    are creatable on CPU. kNone = -1 is "no dtype" for optional dtype
    parameters only; TTensor.Kind raises ETorchError outside 0..45. }
  TKind = (kNone = -1,
           kUInt8 = 0, kInt8 = 1, kInt16 = 2, kInt = 3, kInt64 = 4,
           kHalf = 5, kFloat = 6, kDouble = 7, kComplexHalf = 8,
           kComplexFloat = 9, kComplexDouble = 10, kBool = 11,
           kQInt8 = 12, kQUInt8 = 13, kQInt32 = 14, kBFloat16 = 15,
           kQUInt4x2 = 16, kQUInt2x4 = 17, kBits1x8 = 18, kBits2x4 = 19,
           kBits4x2 = 20, kBits8 = 21, kBits16 = 22,
           kFloat8e5m2 = 23, kFloat8e4m3fn = 24, kFloat8e5m2fnuz = 25,
           kFloat8e4m3fnuz = 26, kUInt16 = 27, kUInt32 = 28, kUInt64 = 29,
           kUInt1 = 30, ..., kUInt7 = 36, kInt1 = 37, ..., kInt7 = 43,
           kFloat8e8m0fnu = 44, kFloat4e2m1fnx2 = 45);

  { c10::Layout; ltNone = -1 for optional layout parameters only }
  TLayout = (ltNone = -1, ltStrided = 0, ltSparse, ltSparseCsr, ltMkldnn,
             ltSparseCsc, ltSparseBsr, ltSparseBsc, ltJagged);

  TDeviceType = (dtCpu, dtCuda, dtMps, dtVulkan);
  TDevice = record
    DeviceType: TDeviceType;
    Index: Integer;               // CUDA only
    function ToC: cint;           // -1 / >=0 / -2 / -3
    class function FromC(V: cint): TDevice; static;
    class function Cpu: TDevice; static;
    class function Cuda(AIndex: Integer = 0): TDevice; static;
    class function CudaIfAvailable: TDevice; static;
    function IsCuda: Boolean;
  end;

  TInt64Array  = array of Int64;
  TDoubleArray = array of Double;
  TSingleArray = array of Single;
  TTensorPtrArray = array of PTorchTensor;   // generated bodies: tensor-list arguments

  TScalar = record                // converted to a temporary ats_* handle per call
    IsFloat: Boolean;
    I: Int64; F: Double;
    class operator :=(V: Int64): TScalar;
    class operator :=(V: Double): TScalar;
  end;

  TOptInt64  = record HasValue: Boolean; Value: Int64;  class operator :=(V: Int64): TOptInt64;  end;
  TOptDouble = record HasValue: Boolean; Value: Double; class operator :=(V: Double): TOptDouble; end;
  TOptIntList = record HasValue: Boolean; Value: TInt64Array; end;

  TTensor = record
  public type
    TTensorArray = array of TTensor;   // nested: cannot be declared before TTensor
  ...
  end;
  TTensorArray = TTensor.TTensorArray; // alias for user code

const
  NoInt64:   TOptInt64   = (HasValue: False; Value: 0);
  NoDouble:  TOptDouble  = (HasValue: False; Value: 0);
  NoIntList: TOptIntList = (HasValue: False; Value: nil);

function IntList(const A: array of Int64): TOptIntList;  // present, possibly empty
```

- `Default(TOptXxx)` is also "none"; the `NoXxx` constants are just more
  readable at call sites (`t.sum_dim_intlist(IntList([0]))`).
- Hand-written, not generic (`TOpt<T>`): implicit `:=` operators inside
  generic records are an area with known 3.2.x bug reports, and there are
  only three of them. Measured (cloud session 3): `t.argmax(1)`,
  `t.clamp_min_(0.5)` and `t.add_scalar(1)` convert literals and variables
  through the `:=` operators without a cast.
- **No `TOptKind` (revised, cloud session 3).** A record parameter cannot
  have a Pascal default (measured: "It is not possible to define a default
  value for a parameter of type TOptInt64"), and `dtype: ScalarType? =
  None` is the most common trailing optional argument (54 ops: `sum`,
  `mean`, `cumsum`, `softmax`, ...). `ScalarTypeOption` is therefore
  `TKind` with `kNone` meaning "none", exactly the C encoding (`-1`), so
  those parameters get `= kNone` defaults. Same for `LayoutOption` with
  `ltNone`. `Int64Option`/`DoubleOption` keep the records: there is no
  spare value in `Int64`/`Double` to mean "none".
- **`FromArray` overloads (decided, H2 follow-up; revised in cloud session
  5).** First decision (cloud session 3, implemented in local session 2):
  `array of Single`, `array of Double`, `array of Int64`, `array of Byte`,
  `array of Boolean`; `Int32` dropped (ambiguous with `Int64`, H2), `Int32`
  data through `FromInt32Array`, `Int16`/`Int8` through `FromBytes`.
  **H2 addendum, measured** (local session 3, cloud session 5; FPC 3.2.2):
  an open-array constructor takes its element type from its **first**
  element, and a real literal is typed `Single` only when its value is
  exactly representable in binary32. With both `Single` and `Double`
  overloads: `[1.0, 2.0]`, `[2.5]`, `[0.75]`, `[100.0]`, `[1.5e2]` pick
  `Single`, and so does `[1.0, 0.1]` — `0.1` and a `Double` variable in
  `[1.5, D]` are rounded to `Single` silently; `[0.9, -1.9]`, `[3.14]`,
  `[0.1]`, `[1e-3]`, `[0.3]`, `[0.1, 1.0]` are compile errors ("Can't
  determine which overloaded function to call"). Whether a literal
  compiles thus depends on the binary expansion of its first element.
  **Revised decision:** drop the `Single` overload and give the four
  remaining overloads a trailing kind with the natural default:

  ```pascal
  class function FromArray(const A: array of Double;  AKind: TKind = kFloat): TTensor; overload; static;
  class function FromArray(const A: array of Int64;   AKind: TKind = kInt64): TTensor; overload; static;
  class function FromArray(const A: array of Byte;    AKind: TKind = kUInt8): TTensor; overload; static;
  class function FromArray(const A: array of Boolean; AKind: TKind = kBool):  TTensor; overload; static;
  class function FromSingleArray(const A: array of Single; AKind: TKind = kFloat): TTensor; static;
  ```

  Measured with exactly this set: `[1.0, 2.0]`, `[0.9, -1.9]`, `[1.0, 0.1]`,
  `[3.14]`, `[1.5, 2]`, `[D, 1.5]`, `[1.5, D]` and `[S]` (a `Single`
  variable) all resolve to `Double`; `[1, 2]`, `[1, 5000000000]`, `[N, 2]`
  stay `Int64`; `[Byte(200), 1]` `Byte`; `[True, False]` `Boolean`;
  `FromArray([0.9, -1.9], kDouble)` and `FromArray([1, 2], kFloat)` pass the
  kind through. An `array of Single` *variable* no longer matches any
  overload, hence `FromSingleArray` (the same treatment `Int32` got).
  `[1, 2.5]` remains an error (an `Int64` array, then `Single`): write
  `[1.0, 2.5]`. Rule to document for users: a literal array is typed by its
  first element; an integer first element makes an integer array.
  - The `kFloat` default keeps PyTorch's `torch.tensor([1.0, 2.0])` default
    (float32) and produces the same bits as Python does (decimal → binary64
    → binary32); the kind parameter replaces a `.to_dtype` on the result.
  - Implementation: `Double` with `AKind = kFloat` converts into a `Single`
    buffer in Pascal and calls `at_tensor_of_data` once (no second tensor);
    every other combination creates the natural kind and applies
    `totype(AKind)` when it differs. `FromSingleArray` is the old `Single`
    body plus the kind.
  - Existing callers: `tests/api` expects `TDoubleArray.Create(0.25, 0.5)`
    to give `kDouble` (now `kFloat` unless `kDouble` is passed), `tests/nn`
    casts `[Double(0.9), -1.9]` (no longer needed), and
    `tests/compiler/h2_fromarray.pas` documents the old set — all updated
    in local session 4 (HANDOFF).
  - **Done (local session 4).** Implemented exactly so; `tests/compiler`
    has the new `h2_fromarray.pas` (every row above measured again) and
    `xfail_h2_single.pas` (`[0.9, -1.9]` ambiguous with both overloads);
    `tests/api` checks the float32 bits of `[0.9, -1.9]`, the `kDouble`,
    `[1, 2], kFloat` and `FromSingleArray` cases; `tests/nn` uses
    `FromSingleArray` for its `array of Single` variable. One trap met in
    the test, worth telling users: on x86-64 a real literal in a comparison
    is `Extended`, so `T.DoubleValue([0]) = 0.9` is False for an exact
    binary64 `0.9`; compare with `Double(0.9)` (measured).
- **Element sizes (local session 2, measured).** `KindElementSize(AKind)`
  is a Pascal table, checked against libtorch's `element_size()` for all
  46 kinds by `tests/api` (libtch's `at_copy_data` rejects a wrong size):
  every sub-byte and Float8/Float4 kind is 1 byte, `kNone` raises
  `ETorchError`. So `FromBytes` and `CopyData` accept any kind;
  `To{Single,Double,Int64}Array` convert with `to_dtype` first.
- `IntListOption` (103 uses) was an open question: open arrays and dynamic
  arrays cannot distinguish "none" from "empty" (both are nil/zero-length),
  so a record with `HasValue` it is, built with `IntList([...])`.
- `TensorOptList` (13 uses) needs no extra type: `array of TTensor` with
  `IsNil` elements meaning "none".

Parameter mapping used by the generator:

| gen.ml type | Pascal parameter |
|---|---|
| Tensor | `const X: TTensor` (nil handle rejected before the call) |
| TensorOption | `const X: TTensor` (nil handle = none) |
| Scalar | `const X: TScalar` |
| Bool | `X: Boolean` |
| Int64 / Double | `X: Int64` / `X: Double` |
| Int64Option / DoubleOption | `const X: TOptInt64` / `TOptDouble` |
| IntList | `const X: array of Int64` (allows `[2, 3]`) |
| IntListOption | `const X: TOptIntList` (`IntList([..])` / `NoIntList`) |
| DoubleList | `const X: array of Double` |
| TensorList / TensorOptList | `const X: array of TTensor` |
| String | `const X: AnsiString` (passed as pointer + length; `None` reaches C as an empty string, as in tch-rs) |
| TensorOptions | `X_kind: TKind; const X_device: TDevice` (the YAML name is `options` in all 80 ops) |
| ScalarType / ScalarTypeOption | `X: TKind` (`kNone` = none for the option) |
| Device | `const X: TDevice` |
| Layout / LayoutOption | `X: TLayout` (`ltNone` = none for the option) |

Results: one tensor → `TTensor`; dynamic → `TTensorArray`; N fixed tensors
→ `out` parameters, or none when the results are the tensors passed in
(see 4.4); `bool`/`int64_t`/`double` → `Boolean`/`Int64`/`Double`.

**(H6) resolved** (step 6): `TTensor.cat([A, B], 0)` and
`TTensor.cat([A + B, B, A * B], 1)` give the right shapes; 200,000 calls
keep RSS flat and every handle is released at routine exit (no leak, no
double free). The open-array constructor's elements are held like other
temporaries (H1).

### 4.4 Generated op methods and naming (decision)

- Receiver rule, same as tch-rs: an op whose arguments include a `Tensor`
  named `self` becomes a method of `TTensor` with that argument as the
  receiver; otherwise one named `input` (152 ops, e.g. `conv2d`,
  `batch_norm`, so `x.conv2d(w, ...)` reads like PyTorch); otherwise a
  `class function ... static` on `TTensor` (563 ops, e.g.
  `TTensor.zeros([2, 3], kFloat, TDevice.Cpu)`), mirroring tch-rs's
  `Tensor::zeros`. The receiver is `const`; for in-place ops it is still
  `const` (the record is not modified, the C++ tensor behind it is).
- **Op names are kept verbatim from tch-rs/PyTorch in snake_case**
  (`t.max_pool2d(...)`, `t.add_(u)`, `t.sum_dim_intlist(...)`).
  Hand-written API uses Pascal CamelCase (`Shape`, `Kind`, `ToString`).
  Rationale: snake_case→CamelCase is lossy (`_foo`/`foo`/`foo_` collide),
  verbatim names are searchable in PyTorch/tch-rs docs, and the split makes
  "generated vs hand-written" visible at the call site. Leading and double
  underscores (`_cast_byte`, `__and__`, 530 ops) are legal FPC identifiers
  and are kept too; tch-rs's `internal_` rename is not needed.
- Checked on tch-rs 4227b89: the 2,657 `atg_*` names contain no uppercase
  letters and no case-insensitive duplicates, and no single op has two
  parameters that differ only by case. The generator must still check
  generated names against hand-written member names case-insensitively;
  known overlaps that the hand-written API therefore avoids: `data`, `grad`,
  `detach`, `to_device`, `absolute`, `alias`, `index`, and `clone` (which,
  because gen.ml excludes `clone` itself and the overload de-duplication
  gives the bare name to the shortest survivor, is actually `clone_out` —
  hence our hand-written data copy is `Copy`, not `Clone`).
- Reserved words. Op names colliding with FPC reserved words: `div`, `repeat`,
  `set`, `to`, `var`; with modifiers: `absolute`, `alias`, `index`. Methods
  keep their name with FPC's `&` escape at declaration and call site
  (`t.&div(u)`, `t.&var(0)`); a `_` suffix is not an option for ops because
  the trailing underscore means in-place. Modifiers are escaped as well — a
  harmless `&` on a legal identifier avoids guessing which modifiers FPC
  accepts as method names **(H7, resolved in step 7**: `&div`, `&var`,
  `&to`, `&set`, `&repeat`, `&absolute`, `&alias`, `&index` compile and
  are callable as `t.&div(u)`; `absolute`, `alias`, `index` also compile
  without `&`). Parameter names that are reserved words get the same `&`
  escape: `out` (819 uses), `end` (31), `to` (7) become `&out`, `&end`,
  `&to`, so the one escape rule covers op names and parameter names alike
  (verified in H12). The exception is `result` (3 ops:
  `_linalg_det_result`, `_linalg_solve_ex_result`, `linalg_solve_ex_out`):
  `&result` is still the identifier `result`, which is the function result
  symbol, so it becomes `result_`. Friendlier hand-written aliases
  (`DivBy`, `ToDevice`, `Variance`) can be added on top where it matters.
- **Parameter and out-names vs `TTensor` members (decision, revised in
  cloud session 3).** In plain `{$mode objfpc}` a method parameter may not
  share its name (case-insensitive) with a record member declared before
  it (`dim` next to `Dim`, `indices` next to the `indices` op); whether it
  compiles depends on declaration order. The first decision (local session
  1) was a `_` suffix rule (`dim_`, `indices_`, `pow__`) plus renaming
  `Dim` to `NumDims`. Replaced by **(H12, measured)**: `Tch.pas` adds
  `{$modeswitch duplicatelocals}`, the part of Delphi mode that allows
  exactly this, and everything else stays `objfpc`. Measured on FPC 3.2.2
  (`tests/compiler/h12_*`, `xfail_h12_*`):
  - A parameter or `out` name may equal any member, declared before or
    after, including `indices` as an `out` name next to the `indices` op and
    `pow` next to `pow`/`pow_`. Inside the body the bare name is the
    parameter and `Self.Dim` reaches the member.
  - The switch is needed only in the unit that declares the methods
    (`Tch.pas` and its `.inc` files); user units stay in whatever mode they
    like.
  - A *local variable* still may not shadow a member in `objfpc` (that part
    is only lifted by `{$mode delphi}`), so hand-written bodies cannot
    shadow a member by accident through a local.
  - `result` as a *function* parameter is still a duplicate identifier (it
    is the result symbol, not a member); hence `result_` above.
  - In the compiler source the switch (`m_duplicate_names`) is consulted in
    three places of `symtable.pas` only: the parameter-vs-member check and
    the TP/Delphi "local named like the function" case. Nothing else
    changes.
  Consequences: generated signatures keep PyTorch's parameter names verbatim
  (`dim`, `indices`, `pow`), the generator has no suffix logic for members,
  and the hand-written API is free to use `Dim` (back from `NumDims`).
  Pitfall for hand-written code in `Tch.pas`: a parameter named like a
  member silently hides it in that body, so either write `Self.Member` or
  give hand-written parameters names that are not members.
- Fixed multi-tensor results (246 ops, 2 to 7 tensors; e.g. `max_dim` →
  values, indices) become
  `procedure max_dim(dim: Int64; keepdim: Boolean; out values, indices: TTensor)`.
  Output names come from `Declarations.yaml` `returns[].name` when present,
  else `result0..N`. Rationale: Pascal has no destructuring, `out` is the
  idiom (`DecodeDate`, `DivMod`), and a result record per op would add 246
  types. Hand-written result records can be added for the few popular ops
  (`max_dim`, `sort`, `topk`, `lstm`) once real code shows the need.
  Caveat: FPC finalises an `out` argument on entry **(H8, confirmed in
  step 6)**, so passing the receiver or an input as an `out` argument
  (`t.max_dim(1, False, t, i)`) releases it before the body runs; the nil-handle check then raises
  `ETorchError` rather than crashing. Document it; do not try to be clever.
  Also confirmed: a variable holding a previous value passed as `out`
  releases that value (handle count unchanged across the call).
- **Results that are the tensors passed in (decided, cloud session 3).**
  97 of the 246 multi-result ops are "out=" variants (`max_dim_out`,
  `sort_values`, `topk_values`, `qr_q`, `linalg_solve_ex_out`, ...): every
  result name is also an argument name, and libtorch returns handles to
  those very tensors. Measured on ops.tsv: in all 97 ops *every* result
  collides, in none only some. Rule: such an op is a `procedure` **without
  out parameters**; the generated body frees the N result handles after
  the call (`TchFreeSlots`), the caller already holds the tensors
  (`t.sort_values(V, I)` fills `V` and `I`). A partial overlap would be a
  generator error. Single-result "out=" ops (`add_out`, 800+) keep
  returning the `TTensor`, a new handle to `out`, like C++'s `Tensor&`.
- Default arguments (revised, cloud session 3). gen.ml writes each
  argument's `default` to ops.tsv in canonical text (section 5); tchgen
  turns the maximal *trailing* run of defaultable parameters into Pascal
  defaults. Defaultable: `Bool`, `Int64`, `Double`, `String`,
  `ScalarType`, `ScalarTypeOption` (`= kNone`), `Layout`, `LayoutOption`
  (`= ltNone`). Not defaultable, so the run stops there: records
  (`TOptInt64`, `TOptDouble`, `TOptIntList`, `TScalar`, `TDevice`), open
  arrays and `TTensor` (measured: FPC rejects a default on a record
  parameter). `::std::nullopt` on a kind whose Pascal type cannot say
  "none" (`bool?` 23 ops, `str?` 50 ops: C receives a plain bool / an
  empty string) gets **no** default, the caller chooses, as in tch-rs.
  `at::Reduction::Mean` → `1`; `"constant"` → `'constant'` (C escapes
  decoded). Ops with `out` parameters get no defaults (defaults must be
  the last parameters); ops that fill tensors passed in may. Measured on
  2.13: 479 ops get at least one default; 107 of the 149 ops with out
  parameters would have had some (`topk`, `sort`, ...), hand-written
  wrappers can add them later.
- `TensorOptions` arguments stay explicit (`options_kind: TKind; const
  options_device: TDevice`, the C names); a hand-written
  `TTensor.zeros([2, 3])` overload defaulting to `kFloat`/CPU is cheap and
  mirrors tch-rs's `FLOAT_CPU`.
  **Overloading a generated name (local session 2, measured).** In
  `objfpc` a record method overload needs no `overload` directive (with it
  on neither, one or both declarations, `zeros([2, 3])` and
  `zeros([2, 3, 4], kFloat, Dev)` resolve the same). Hand-written overloads
  of generated ops still carry `overload;` as the marker tchgen reads: a
  hand-written member whose declaration says `overload;` is a deliberate
  overload, any other hand-written member named like an op is a collision
  (error). tchgen reads whole multi-line declarations for this (and no
  longer mistakes continuation lines for fields). Milestone 1 has
  `zeros`, `ones`, `empty`, `rand`, `randn` `(ASize; AKind = kFloat)` on CPU.
- **Generated bodies (cloud session 3).** Every local is suffixed `__`
  (`Outs__`, `P__`, `R__`, `<arg>__` for Scalar handles and tensor-list
  pointer arrays); no YAML argument name ends in `__` (checked by tchgen).
  The receiver is always written `Self.Raw`, and intrinsics are qualified
  (`System.Ord`, `System.Length`, `System.PInt64`, `TchSys.PPTorchTensor`)
  because parameters named `ord` and `length` exist and, under
  `duplicatelocals`, would shadow the intrinsic (found when `linalg_norm`
  failed to compile). Scalar handles are created inside `try` and freed in
  `finally`; tensor lists are copied into a `TTensorPtrArray` after the
  nil checks (`TchTensorPtrs`, `TchTensorOptPtrs`); a present-but-empty
  `IntListOption` passes a non-nil pointer with length 0
  (`TchOptIntListPtr`). All of this is exercised by `tests/mock`.

### 4.5 Unit layout

```
src/
  TchSys.pas                 L2 hand-written externals
  TchSysGenerated.pas        L2 generated externals           (generated)
  Tch.pas                    L3 core: TTensor, TDevice, TKind, TScalar,
                             ETorchError, grad-mode helpers
  TchTensorGenerated_intf.inc  method declarations inside TTensor (generated)
  TchTensorGenerated_impl.inc  method bodies                    (generated)
  TchNN.pas                  TInit, TVarStore, TVarPath, TModule, layers,
                             TSequential, TOptimizer (4.6)
  TchJit.pas                 TJitModule, TIValue, TTrainableJitModule (4.6)
gen/
  gen.ml, dune, dune-project  forked tch-rs generator (+ ops.tsv output)
  ops.tsv                    manifest read by tchgen             (generated)
  tchgen.pas                 Pascal emitter (section 5)
csrc/libtch/                 forked C wrapper + CMakeLists.txt
tests/mock/                  mock libtch + test of the generated layer (no libtorch)
tests/jit/                   TchJit test (real libtorch); models/ = tch-rs's tests/foo*.pt
                             with tch-rs's license files next to them (4.6.8)
gen/                         gen.ml (forked) + Pascal emitter
```

- Generated record members must be declared inside the record, hence the two
  include files. Putting the generated ops in a `record helper for TTensor`
  in a separate unit was considered and rejected: FPC resolves only the
  *last* helper in scope for a type, so any user-defined helper for
  `TTensor` would silently hide all 2,657 ops.
- Optimizers live in `TchNN`, not a separate `TchOptim`: `TOptimizer` needs
  the var-store's trainable-variable list and groups (it lazily registers
  variables created after the optimizer, exactly like tch-rs), and that is
  internal state better not exported across units. tch-rs keeps them in
  `nn` for the same reason.

### 4.6 nn, VarStore, optimizers and JIT (decision)

tch-rs's shape and its Pascal equivalent:

| tch-rs | role | Pascal | why |
|---|---|---|---|
| `VarStore` (`Arc<Mutex<Variables>>`, device, kind) | owns all parameters by name | `TVarStore = class` | long-lived identity object with explicit lifetime; `Free` it when the model is done |
| `Path<'a>` (borrowed, `vec<String>` + group) | naming scope, cheap to copy | `TVarPath = record` (non-owning `TVarStore` reference + dotted path + group) | value semantics, no ownership, `P.Sub('layer1')` / `P / 'layer1'` |
| `Entry::or_*` | get-or-create | `TVarPath.GetOrAdd(...)` | no separate type needed |
| `Init` enum with payload | initialisation recipe | `TInit = record` with `Kind` + fields and static constructors | Pascal enums carry no payload |
| `Module` / `ModuleT` traits | forward pass | `TModule = class abstract` with one virtual `Forward(const X; Train: Boolean = False)` | Pascal has no blanket trait impl; a default parameter merges the two traits |
| `Linear`, `Conv2D`, `BatchNorm`, `LSTM`... structs | layers holding `Tensor` fields | classes deriving from `TModule`, parameters as public `TTensor` fields (`Weight`, `Bias`) | same ownership story as tch-rs: the layer and the var-store share the handle |
| `Sequential` (`Vec<Box<dyn Module>>`) | composition | `TSequential = class(TModule)`, owns its children | explicit ownership tree, idiomatic Pascal |
| `Func` (closure layer) | ad-hoc layers | `TFuncModule` taking a plain function pointer or `of object` method | FPC 3.2.2 has no anonymous functions; a named function or a tiny `TModule` subclass replaces `add_fn(|x| ...)` |
| `Optimizer` + `OptimizerConfig` structs | training | `TOptimizer = class` with `class function Adam(VS, LR; Beta1 = 0.9; ...)` constructors | Pascal default parameters replace builder/config structs |
| `no_grad(|| ...)`, `NoGradGuard` | grad mode | `SetGradEnabled(B): Boolean` + `NoGradGuard: IInterface` | see 4.7 |
| `CModule`, `TrainableCModule`, `IValue` | TorchScript | `TJitModule = class`, `TTrainableJitModule = class(TModule)`, `TIValue` tagged record | see below |
| `Iter2` | mini-batches | `TBatchIter` with `GetEnumerator` for `for .. in` | later |

The shape above was decided in cloud session 2. Cloud session 4 settled
the seven points HANDOFF listed as not yet mechanical (initialisation
formulas, store internals, save/load, optimizers, layers, sequential, unit
and tests) against the same tch-rs commit; file names below are tch-rs's.
What needs real libtorch to confirm is marked **(H13)**–**(H18)**; what was
measured on FPC 3.2.2 in the cloud is marked *measured*. Everything lives
in one unit, `TchNN`.

Sketch (interface; bodies are described in 4.6.1–4.6.7):

```pascal
unit TchNN;
{$mode objfpc}{$H+}{$modeswitch advancedrecords}
uses SysUtils, Generics.Collections, TchSys, Tch;
type
  // --- initialisation (tch-rs nn/init.rs) ---
  TFanMode = (fmFanIn, fmFanOut);                              // FanInOut
  TInitDist = (idUniform, idNormal);                           // NormalOrUniform
  TNonLinearity = (nlReLU, nlLinear, nlSigmoid, nlTanh, nlSELU, nlExplicitGain);
  TInitKind = (ikConst, ikUniform, ikRandn, ikKaiming, ikOrthogonal);
  TInit = record
    Kind: TInitKind;
    A, B: Double;                 // Const: A; Uniform: lo, up; Randn: mean, std; Orthogonal: gain in A
    Dist: TInitDist;              // Kaiming only
    Fan: TFanMode;
    NonLin: TNonLinearity;
    Gain: Double;                 // for nlExplicitGain
    class function Constant(V: Double): TInit; static;        // "Const" is reserved
    class function Uniform(Lo, Up: Double): TInit; static;
    class function Randn(Mean, Std: Double): TInit; static;
    class function Kaiming(ADist: TInitDist; AFan: TFanMode; ANonLin: TNonLinearity;
      AGain: Double = 0): TInit; static;
    class function KaimingUniform: TInit; static;   // Kaiming(idUniform, fmFanIn, nlReLU) = DEFAULT_KAIMING_UNIFORM
    class function KaimingNormal: TInit; static;    // Kaiming(idNormal, fmFanIn, nlReLU) = DEFAULT_KAIMING_NORMAL
    class function Orthogonal(AGain: Double = 1): TInit; static;
    function Make(const Dims: array of Int64; const ADevice: TDevice; AKind: TKind): TTensor;  // f_init
    procedure Apply(const T: TTensor);              // Init::set, in place, under NoGradGuard
  end;
  function FanForDims(Fan: TFanMode; const Dims: array of Int64): Int64;   // FanInOut::for_weight_dims
  function NonLinearityGain(NL: TNonLinearity; ExplicitGain: Double = 0): Double;

  // --- var store (tch-rs nn/var_store.rs) ---
  TVarStore = class;

  TVarPath = record
  private
    FStore: TVarStore;            // not owned; must outlive the path
    FPath: String;                // 'encoder.layer1' ('' = root)
    FGroup: Integer;
  public
    function Sub(const Name: String): TVarPath;               // raises on '.' in Name
    class operator /(const P: TVarPath; const Name: String): TVarPath;
    function SetGroup(G: Integer): TVarPath;
    function Group: Integer; inline;                          // local session 3 addition, confirmed
    function Store: TVarStore; inline;
    function Device: TDevice;
    function Kind: TKind;
    function FullName(const Name: String): String;            // Path::path
    function Add(const Name: String; const T: TTensor; Trainable: Boolean): TTensor;
    function Variable(const Name: String; const Dims: array of Int64; const Init: TInit): TTensor; // var
    function Zeros(const Name: String; const Dims: array of Int64): TTensor;
    function Ones(const Name: String; const Dims: array of Int64): TTensor;
    function ZerosNoTrain(const Name: String; const Dims: array of Int64): TTensor;
    function OnesNoTrain(const Name: String; const Dims: array of Int64): TTensor;
    function RandnStandard(const Name: String; const Dims: array of Int64): TTensor;
    function Randn(const Name: String; const Dims: array of Int64; Mean, Std: Double): TTensor;
    function Uniform(const Name: String; const Dims: array of Int64; Lo, Up: Double): TTensor;
    function KaimingUniform(const Name: String; const Dims: array of Int64): TTensor;
    function KaimingNormal(const Name: String; const Dims: array of Int64): TTensor;
    function Orthogonal(const Name: String; const Dims: array of Int64; Gain: Double): TTensor;
    function VarCopy(const Name: String; const T: TTensor): TTensor;
    function TryGet(const Name: String; out T: TTensor): Boolean;   // Path::get
    function GetOrAdd(const Name: String; const Dims: array of Int64; const Init: TInit): TTensor; // entry().or_var
    procedure SetKind(AKind: TKind);        // this sub-tree, all variables
    procedure SetFloatKind(AKind: TKind);   // this sub-tree, floating-point variables only
  end;

  TNamedTensor = record Name: String; Tensor: TTensor; end;
  TNamedTensorArray = array of TNamedTensor;

  TVarEntry = record
    Name: String;                 // full dotted name
    Tensor: TTensor;
    Trainable: Boolean;
    Group: Integer;
  end;
  TVarEntryArray = array of TVarEntry;
  TNameIndex = specialize TDictionary<String, Integer>;

  TVarStore = class
  private
    FVars: TVarEntryArray;        // insertion order, append-only (named_variables + order)
    FTrainable: array of Integer; // indices into FVars, insertion order (trainable_variables)
    FIndex: TNameIndex;           // full name -> index into FVars
    FDevice: TDevice;
    FKind: TKind;
    function AddEntry(const FullName: String; const T: TTensor; Trainable: Boolean;
      Group: Integer): TTensor;   // used by TVarPath.Add
  public
    constructor Create(const ADevice: TDevice; AKind: TKind = kFloat);
    destructor Destroy; override;
    function Root: TVarPath;
    function Count: Integer;
    function TrainableVariables: TTensorArray;
    function Variables: TNamedTensorArray;                    // insertion order
    function TryGet(const FullName: String; out T: TTensor): Boolean;
    procedure RollbackTo(ACount: Integer);                    // undo the newest Adds
    procedure Save(const FileName: String);
    procedure Load(const FileName: String);
    function LoadPartial(const FileName: String): TStringArray;   // missing names
    procedure Freeze;
    procedure Unfreeze;
    procedure SetDevice(const ADevice: TDevice);
    procedure SetKind(AKind: TKind);                          // set_kind: every variable
    procedure SetFloatKind(AKind: TKind);                     // half/bfloat16/float/double
    procedure CopyFrom(Src: TVarStore);                       // VarStore::copy
    property Device: TDevice read FDevice;
    property Kind: TKind read FKind;
  end;

  // --- modules and layers ---
  TModule = class abstract
  public
    function Forward(const X: TTensor; Train: Boolean = False): TTensor; virtual; abstract;
  end;

  TLinearConfig = record WsInit, BsInit: TInit; HasBsInit: Boolean; Bias: Boolean; end;
  TPaddingMode = (pmZeros, pmReflect, pmReplicate, pmCircular);
  TConvConfig = record
    Stride, Padding, Dilation, Groups: Int64; Bias: Boolean;
    WsInit, BsInit: TInit; PaddingMode: TPaddingMode;
  end;
  TBatchNormConfig = record CudnnEnabled: Boolean; Eps, Momentum: Double; Affine: Boolean; WsInit, BsInit: TInit; end;
  TLayerNormConfig = record CudnnEnabled: Boolean; Eps: Double; ElementwiseAffine: Boolean; WsInit, BsInit: TInit; end;
  TEmbeddingConfig = record Sparse, ScaleGradByFreq: Boolean; WsInit: TInit; PaddingIdx: Int64; end;
  function DefaultLinearConfig: TLinearConfig;      // and DefaultConvConfig, DefaultBatchNormConfig, ...

  TLinear = class(TModule)
  public
    Weight, Bias: TTensor;        // Bias.IsNil when the layer has none
    constructor Create(const P: TVarPath; InDim, OutDim: Int64); overload;
    constructor Create(const P: TVarPath; InDim, OutDim: Int64; const Cfg: TLinearConfig); overload;
    function Forward(const X: TTensor; Train: Boolean = False): TTensor; override;
  end;
  TConv2D = class(TModule)       // Conv2D with tch-rs's scalar ConvConfig
  private
    FCfg: TConvConfig;
    FPadTwice: TInt64Array;       // reversed_padding_repeated_twice
  public
    Weight, Bias: TTensor;
    constructor Create(const P: TVarPath; InDim, OutDim, KSize: Int64); overload;
    constructor Create(const P: TVarPath; InDim, OutDim, KSize: Int64; const Cfg: TConvConfig); overload;
    function Forward(const X: TTensor; Train: Boolean = False): TTensor; override;
  end;
  TBatchNorm = class(TModule)    // Nd = 1, 2, 3 for batch_norm1d/2d/3d; any other Nd raises ETorchError before registering
  public
    Weight, Bias, RunningMean, RunningVar: TTensor;
    constructor Create(const P: TVarPath; Nd: Integer; OutDim: Int64); overload;
    constructor Create(const P: TVarPath; Nd: Integer; OutDim: Int64; const Cfg: TBatchNormConfig); overload;
    function Forward(const X: TTensor; Train: Boolean = False): TTensor; override;
  end;
  TLayerNorm = class(TModule)
  public
    Weight, Bias: TTensor;
    constructor Create(const P: TVarPath; const NormalizedShape: array of Int64); overload;
    constructor Create(const P: TVarPath; const NormalizedShape: array of Int64; const Cfg: TLayerNormConfig); overload;
    function Forward(const X: TTensor; Train: Boolean = False): TTensor; override;
  end;
  TEmbedding = class(TModule)
  public
    Weight: TTensor;
    constructor Create(const P: TVarPath; NumEmbeddings, EmbeddingDim: Int64); overload;
    constructor Create(const P: TVarPath; NumEmbeddings, EmbeddingDim: Int64; const Cfg: TEmbeddingConfig); overload;
    function Forward(const X: TTensor; Train: Boolean = False): TTensor; override;
  end;

  TTensorFunc    = function(const X: TTensor): TTensor;
  TTensorFuncT   = function(const X: TTensor; Train: Boolean): TTensor;
  TTensorMethod  = function(const X: TTensor): TTensor of object;
  TTensorMethodT = function(const X: TTensor; Train: Boolean): TTensor of object;

  TFuncModule = class(TModule)   // Func / FuncT
  public
    constructor Create(F: TTensorFunc); overload;
    constructor Create(F: TTensorFuncT); overload;
    constructor Create(F: TTensorMethod); overload;
    constructor Create(F: TTensorMethodT); overload;
    function Forward(const X: TTensor; Train: Boolean = False): TTensor; override;
  end;

  TModuleList = specialize TObjectList<TModule>;

  TSequential = class(TModule)   // SequentialT: Train goes to every child
  private
    FLayers: TModuleList;         // OwnsObjects
  public
    constructor Create;
    destructor Destroy; override;
    function Add(M: TModule): TSequential;                    // takes ownership; returns Self
    function AddFn(F: TTensorFunc): TSequential; overload;
    function AddFn(F: TTensorFuncT): TSequential; overload;
    function AddFn(F: TTensorMethod): TSequential; overload;
    function AddFn(F: TTensorMethodT): TSequential; overload;
    function Forward(const X: TTensor; Train: Boolean = False): TTensor; override;
    function ForwardAll(const X: TTensor; Train: Boolean = False; N: Integer = -1): TTensorArray;
    function Count: Integer;
    property Items[I: Integer]: TModule read GetItem; default;
  end;

  // --- optimizers (tch-rs nn/optimizer.rs, wrappers/optimizer.rs) ---
  TOptimizer = class
  private
    FOpt: PTorchOptimizer;
    FStore: TVarStore;            // not owned; free the optimizer before the store
    FVarsInOptimizer: Integer;    // variables_in_optimizer
    {$push}{$warn 3018 off}       // "Constructor should be public": the factories below are the API (4.6.4)
    constructor CreateRaw(AOpt: PTorchOptimizer; VS: TVarStore);
    {$pop}
    procedure AddMissingVariables;
  public
    class function Sgd(VS: TVarStore; LR: Double; Momentum: Double = 0; Dampening: Double = 0;
      WeightDecay: Double = 0; Nesterov: Boolean = False): TOptimizer; static;
    class function Adam(VS: TVarStore; LR: Double; Beta1: Double = 0.9; Beta2: Double = 0.999;
      WeightDecay: Double = 0; Eps: Double = 1e-8; AmsGrad: Boolean = False): TOptimizer; static;
    class function AdamW(VS: TVarStore; LR: Double; Beta1: Double = 0.9; Beta2: Double = 0.999;
      WeightDecay: Double = 0.01; Eps: Double = 1e-8; AmsGrad: Boolean = False): TOptimizer; static;
    class function RmsProp(VS: TVarStore; LR: Double; Alpha: Double = 0.99; Eps: Double = 1e-8;
      WeightDecay: Double = 0; Momentum: Double = 0; Centered: Boolean = False): TOptimizer; static;
    destructor Destroy; override;                             // ato_free
    procedure ZeroGrad;
    procedure Step;
    procedure BackwardStep(const Loss: TTensor);
    procedure BackwardStepClip(const Loss: TTensor; MaxValue: Double);
    procedure BackwardStepClipNorm(const Loss: TTensor; MaxNorm: Double);
    procedure ClipGradValue(MaxValue: Double);
    procedure ClipGradNorm(MaxNorm: Double);
    procedure SetLearningRate(LR: Double);
    procedure SetLearningRateGroup(Group: Integer; LR: Double);
    procedure SetMomentum(M: Double);                         // beta1 for Adam/AdamW
    procedure SetMomentumGroup(Group: Integer; M: Double);
    procedure SetWeightDecay(WD: Double);
    procedure SetWeightDecayGroup(Group: Integer; WD: Double);
    function TrainableVariables: TTensorArray;
  end;
```

Decisions and rationale (overall shape):
- **`TVarStore` is a class, `TVarPath` a record.** The store has identity,
  mutable state and a clear owner; paths are throwaway values that only
  carry a prefix and a group. A path holds a plain class reference, so the
  rule "the store outlives its paths, layers and optimizers" is documented
  rather than enforced (same as tch-rs's lifetime `'a`, minus the compiler).
- **One `Forward` with `Train: Boolean = False`** instead of `Module` and
  `ModuleT`. Layers that ignore `Train` just ignore it. Overrides must repeat
  the default value for callers through the derived type to see it **(H9,
  resolved in step 7**: with the default repeated, `Forward(X)` works
  through base and derived references; an override without it still works
  through the base type but `Derived.Forward(X)` is a compile error).
  A PyTorch-style `Training` flag on the module was considered and rejected:
  propagating it requires child registration machinery (`__setattr__` magic
  in Python) that Pascal would have to spell out. Closed in cloud session 4:
  `TSequential` passes `Train` to every child (4.6.6) and a user `TModule`
  passes it on explicitly; nothing else is needed.
- **Layers are classes that own nothing but tensors** (refcounted). They are
  created with a `TVarPath` and register their parameters, like tch-rs.
  `TSequential` owns its children. A user model is a `TModule` subclass that
  creates sub-layers in its constructor and frees them in its destructor —
  the Lazarus component pattern.
- **Configs** are records with a `Default*` function (Pascal records cannot
  have default field values); the common case gets an overload without the
  config. Optimizers use default parameters instead, because their options
  are flat.
- **Variable names** use `.` as separator and are written with `|` in
  `.ot` files, exactly as tch-rs does, so files are interchangeable with
  tch-rs and with `Tensor::save_multi` readers.
- **Thread safety:** tch-rs wraps `Variables` in a mutex because Rust forces
  the question. The first Pascal version documents `TVarStore` as not
  thread-safe; a `TCriticalSection` can be added without API change.
- **JIT.** `TJitModule` is a class (few instances, explicit lifetime, like a
  file). `TIValue` is a tagged record (`Tag`, `Tensor`, `I`, `D`, `B`, `S`,
  `Items: array of TIValue`, `Tensors: TTensorArray`, ...) with static
  constructors (`TIValue.Tuple([...])`), implicit `:=` operators from
  `TTensor`, `Int64`, `Double`, `Boolean` and `String`, and `As*` accessors
  that raise on a tag mismatch. **(H10)** A record containing a dynamic
  array of its own type must compile on 3.2.2; the fallback is an
  fpjson-style class hierarchy. **(H10) resolved** (step 7): it compiles and
  nests (tuple of tuples). Semantics to document: `W := V` shares `Items`
  (dynamic arrays are refcounted, not copy-on-write), so writing
  `W.Items[0]` changes `V` too. `TIValue` should be treated as immutable
  after construction. `TTrainableJitModule` registers `NamedParameters`
  into a `TVarPath` and derives from `TModule`, so a scripted model can be
  fine-tuned with `TOptimizer` like any other layer. Not part of milestone
  2; the mechanical design is 4.6.8 (cloud session 5).

#### 4.6.1 Initialisation (`TInit`), from `src/nn/init.rs`

- **Fan** (`FanInOut::for_weight_dims`): `Receptive := product of
  Dims[2..]` (1 when there are fewer than three dims). `fmFanIn` = 1 if
  `Length(Dims) < 2`, else `Dims[1] * Receptive`; `fmFanOut` = 1 if
  `Length(Dims) = 0`, else `Dims[0] * Receptive`. So a linear weight
  `[Out, In]` has fan-in `In`, a conv weight `[Out, In div Groups, K, K]`
  has fan-in `In div Groups * K * K`.
- **Gain** (`NonLinearity::gain`): ReLU `sqrt(2)`, Tanh `5/3`, Linear and
  Sigmoid `1`, SELU `0.75`, ExplicitGain `g`. *Measured* (overall review): FPC folds
  `5.0 / 3.0` in Single (1.6666666269), so the code writes `Double(5) / 3`;
  `tests/nn` checks every gain to 1e-12.
- **`Make(Dims, Device, Kind)`** is tch-rs `f_init`, op for op (generated
  names as in `TchTensorGenerated_intf.inc`):

  | `TInit` | tensor |
  |---|---|
  | `Constant(0)` | `TTensor.zeros(Dims, Kind, Device)` |
  | `Constant(1)` | `TTensor.ones(Dims, Kind, Device)` (tch-rs tests `abs(c - 1) <= f64::EPSILON`; `A = 1.0` gives the same tensor) |
  | `Constant(c)` | `TTensor.ones(...).mul_scalar(c)` |
  | `Uniform(Lo, Up)` | `TTensor.zeros(...).uniform_(Lo, Up)` |
  | `Randn(0, 1)` | `TTensor.randn(Dims, Kind, Device)` |
  | `Randn(M, S)` | `TTensor.randn(...).mul_scalar(S).add_scalar(M)` |
  | `Kaiming(idUniform, Fan, NL)` | `Std := Gain / sqrt(FanForDims(Fan, Dims))`; `Bound := sqrt(3) * Std`; `TTensor.zeros(...).uniform_(-Bound, Bound)` |
  | `Kaiming(idNormal, Fan, NL)` | `TTensor.randn(...).mul_scalar(Std)` |
  | `Orthogonal(G)` | below |

  `KaimingUniform` on a linear weight `[Out, In]` therefore draws from
  `±sqrt(6 / In)` (gain `sqrt(2)`, fan-in `In`): the hand-checkable bound
  for the test.
- **Orthogonal** (`Init::Orthogonal`): `Length(Dims) < 2` raises
  `ETorchError('Only tensors with 2 or more dimensions are supported')`;
  `Rows := Dims[0]`, `Cols := product of Dims[1..]`;
  `F := TTensor.empty([Rows, Cols], Kind, Device).normal_(0, 1)`;
  if `Rows < Cols` then `F := F.t_`; `TTensor.linalg_qr(F, 'reduced', Q, R)`
  (class procedure with `out Q, R`); `Ph := R.diag(0).sign`; `Q.mul_(Ph)`
  (broadcast over the last dim, tch-rs `q *= ph`); if `Rows < Cols` then
  `Q := Q.t_`; `Q.mul_scalar_(G)`; `Result := Q.contiguous.view(Dims)`.
  **Deviation, stated:** tch-rs's `f_init` returns the 2-D `[Rows, Cols]`
  tensor even when `Dims` has more than two entries (its `Init::set` then
  fixes the shape with `view_as`); PyTorch's `orthogonal_` keeps the
  requested shape, so `Make` ends with `.view(Dims)` (a no-op for 2-D
  dims, where the two agree). **(H14)**: `Q·Qᵀ = I` for `[3, 5]` and
  `Qᵀ·Q = I` for `[5, 3]` with `allclose(…, 1e-5)` on real libtorch.
  **(H14) resolved** (local session 3): both hold to 1e-5 (max abs
  difference to `eye`), the shape is `Dims`, and `[2, 3, 2]` with gain 2
  keeps its shape with rows of norm 2.
- **No `NoGradGuard` in `Make`:** every tensor is created inside it with
  `requires_grad = False`, so no history is recorded (tch-rs's `f_init` has
  no `no_grad` either, except a redundant one around the gain multiply).
  `requires_grad` is set afterwards by `TVarPath.Add`.
- **`Apply(T)`** is tch-rs `Init::set` (re-initialise an existing variable
  in place: `fill_`, `uniform_`, `at_copy_` of `randn_like * std + mean`,
  `at_copy_` of `Make(T.Shape, T.Device, T.Kind)` for Orthogonal, Kaiming
  with the fan taken from `T.Shape`). It runs under its own `NoGradGuard`:
  the target is normally a leaf that requires grad, and an in-place op on
  such a leaf raises in libtorch; tch-rs leaves the `no_grad` to the caller
  (its tests wrap `init` in `no_grad`), the Pascal version does it itself.
  **(H17)**: confirm the failure without the guard and success with it.
  **(H17) resolved** (local session 3): `fill_` on a trainable leaf without
  the guard raises `ETorchNativeError` "a leaf Variable that requires grad
  is being used in an in-place operation."; `Apply` succeeds for every
  `TInit` kind and the variable still requires grad afterwards.
  Hand-written `TTensor.CopyFrom(const Src: TTensor)` over `at_copy_` is
  added to `Tch.pas` for this (there is no generated `copy_`; `_copy_from`
  is a different identifier, so tchgen's name check passes).
- **Kind.** `TVarPath.Variable` and friends pass the store's `Kind` (tch-rs
  `f_var` uses `self.kind()`). tch-rs hard-codes `Kind::Float` in
  `zeros_no_train`, `ones_no_train` and `Entry::or_var`; the Pascal version
  uses the store kind everywhere (deviation; identical for the default
  `kFloat`). tch-rs's `Entry::or_kaiming_uniform` uses the *normal* default
  (an upstream slip); `GetOrAdd` takes an explicit `TInit`, so the question
  does not arise.

#### 4.6.2 `TVarStore` internals, from `src/nn/var_store.rs`

- **tch-rs's state:** `named_variables: HashMap<String, Tensor>`
  (unordered) and `trainable_variables: Vec<Var { tensor, group }>` in
  insertion order; the optimizer depends on that order and on the vector's
  length for its lazy add.
- **Container (decided, measured in M2 of cloud session 4):** `FVars:
  array of TVarEntry` in insertion order, append-only (entries are never
  removed), `FTrainable: array of Integer` with the indices of the trainable
  entries in insertion order (= tch-rs `trainable_variables`), and `FIndex:
  specialize TDictionary<String, Integer>` from `Generics.Collections` for
  the name lookup. Measured on FPC 3.2.2: `TDictionary<String, Integer>`
  compiles, keys are case-sensitive (`weight` and `Weight` are two keys),
  a duplicate `Add` raises `EListError`; a dynamic array of a record with a
  `TTensor` field keeps every handle alive across `SetLength` growth and
  releases them all when the array is cleared; `TList<TTensor>` works too
  (`IndexOf` compares the record bytes, i.e. the handle, so it is identity)
  but is not needed. This closes the "equality and containers" question:
  no `=` on `TTensor` is required. `Variables` and `TrainableVariables`
  return insertion order (tch-rs's `variables()` is a HashMap, so
  deterministic order is a harmless improvement).
- **`Add(Name, T, Trainable)`** (`Path::add`): full name = `FPath + '.' +
  Name` (`Name` alone at the root). If the full name already exists it
  becomes `<full>__<Count>` (tch-rs: `format!("{}__{}", path,
  named_variables.len())`); tch-rs's `save_and_load_var_store` test
  registers `t123` twice and expects four variables, so files written by
  tch-rs can contain such names and the rule is kept verbatim. If
  `<full>__<Count>` is taken too (e.g. `x`, `x__2`, then `x` again), the
  next free number is used; every check happens before the store changes
  (added after the overall review, which found the store left
  half-updated by a dictionary error there). If
  `Trainable`: `T.set_requires_grad(True)` (generated; in place, the
  returned handle is discarded) and the index is appended to `FTrainable`.
  The function returns `T` itself: the layer, the store and the caller share
  the handle (tch-rs's `shallow_clone`s only exist for Rust ownership).
- **Name validation:** `Sub` and every variable name raise `ETorchError`
  when the string contains `.` (tch-rs panics: "sub name cannot contain .",
  "variable name cannot contain ."). Stricter than tch-rs, stated: `|` is
  rejected too (it would come back as `.` after a save/load round trip) and
  so is the empty string. The check lives in `TVarPath.FullName`, which
  every path operation goes through, so `TVarPath.TryGet('a.b')` raises
  instead of returning False (tch-rs's `Path::get` only joins the strings;
  a dotted name can never have been added, so raising is the consistent
  choice); the store-level `TVarStore.TryGet(FullName)` takes a full dotted
  name and does not validate. `AddEntry` checks the tensor with
  `TchCheckArg` (4.2: `set_requires_grad` would otherwise be the first C
  call on a nil handle), and `TVarPath.Group` reads the group next to
  `Device`/`Kind`. These three are local session 3 additions, confirmed in
  cloud session 5.
- **`GetOrAdd(Name, Dims, Init)`** (`Entry::or_var`): returns the existing
  tensor by full name whatever `Dims` and `Init` say (tch-rs test
  `var_store_entry`: a second `or_zeros(&[1, 5, 9])` returns the `[3, 1, 4]`
  tensor), otherwise adds a trainable variable.
- **`RollbackTo(ACount)`** (no tch-rs counterpart; added after the overall
  review, 2026-10-13): removes the variables registered after the store held
  `ACount` of them, newest first, from the entries, the name index and
  `FTrainable`. `ACount < 0` means 0; `ACount >= Count` does nothing; it
  never raises. Only variables appended after the mark go, so one that
  existed before (e.g. returned by `GetOrAdd`) stays. A layer constructor
  takes `Mark := Count` and calls it on failure (4.6.5); a custom module can
  use the same pattern.
- **`Freeze` / `Unfreeze`:** `set_requires_grad(False / True)` on every
  `FTrainable` entry; the entries stay trainable and stay in the optimizer
  (libtorch's optimizers skip a parameter whose `grad` is undefined,
  **(H18)**: `Freeze; Step` leaves the values unchanged and raises nothing).
  In place: no handle changes. **(H18) resolved** (local session 3): after
  `ZeroGrad; Freeze`, a backward through `A * E` (E an unfrozen tensor)
  leaves `A.grad` undefined and `Step` leaves `A` unchanged without raising.
- **`SetDevice(D)`** (`set_device`): for every entry
  `T.set_data(T.&to(D))` (generated `set_data` and `&to`), then `FDevice :=
  D`. `set_data` swaps the storage behind the *same* C++ tensor object, so
  every `TTensor` sharing the handle (layers, the optimizer's param groups)
  sees the moved data; no grad guard is needed, `set_data` is not an
  autograd op. Same caveat as tch-rs: an optimizer's state buffers are not
  moved, so create optimizers after `SetDevice`. **(H13)**: a layer's
  `Weight` and the store's entry show the new device after `SetDevice`, and
  an optimizer created before a `SetKind(kDouble)` still steps the variable.
  **(H13) resolved** (local session 3, CPU only): after `SetKind(kDouble)` a
  `TLinear`'s `Weight` (same handle) is `kDouble`, and an SGD optimizer
  created before the conversion steps it to the hand-computed `[0.9, 0.8]`
  and leaves it `kDouble`. `SetDevice(Cpu)` keeps the handles usable; the
  real device move is item 5 of the GPU test in 4.8.
- **`SetKind(K)`** (`set_kind` → `Path::set_kind` on the root): every entry
  `T.set_data(T.totype(K))`, then `FKind := K`. **`SetFloatKind(K)`**
  (tch-rs `half` / `bfloat16` / `float` / `double`): only entries whose
  `is_floating_point` is true. Deviation, stated: tch-rs's `half()` leaves
  the default kind for *new* variables at `Float`; `SetFloatKind` sets
  `FKind` too. `TVarPath.SetKind` / `SetFloatKind` restrict to the sub-tree:
  entries whose name equals the path or starts with the path plus `.`
  (tch-rs uses a plain `starts_with`, which also matches `ab.x` for the path
  `a`; fixed, stated).
- **`CopyFrom(Src)`** (`VarStore::copy`): first pass checks that every name
  of `Self` exists in `Src` and raises `ETorchError('variable <name> not
  found in source var-store')` before anything is copied (tch-rs does the
  same two passes); second pass, under `NoGradGuard`,
  `Dst.CopyFrom(SrcT.&to(FDevice))`.
- **`Destroy`** frees `FIndex`; clearing the arrays releases the store's
  references, tensors shared with layers live on. `TVarPath` values that
  still point at the freed store are dangling (documented, as above).
  `Count = Length(FVars)` (tch-rs `len()`).

#### 4.6.3 Save and load, from `var_store.rs` and `wrappers/tensor.rs`

- **`Save(FileName)`** (`VarStore::save` → `Tensor::save_multi`):
  `at_save_multi(tensors, names, n, filename)` over *all* entries in
  insertion order, each name with `.` replaced by `|`. The Pascal side keeps
  an `array of AnsiString` alive and passes an `array of PChar` into it.
  tch-rs writes safetensors for the `.safetensors` extension in Rust code;
  there is no libtch entry point for it, so that extension raises
  `ETorchError('safetensors files are not supported')`.
- **Reading** (`VarStore::named_tensors`): by extension, exactly as tch-rs:
  `.bin` / `.pt` → `at_loadz_callback_with_device` (pickle zip, as
  exported from Python); `.safetensors` → `ETorchError`; anything else →
  `at_load_callback_with_device(filename, data, @LoadCallback,
  FDevice.ToC)`. Not used, and therefore not declared in `TchSys`:
  `at_load_multi`, `at_load_multi_` (tch-rs does not use them either) and
  the two callback forms without a device (the device forms cover them).
  Tensors arrive on the store's device; extra names in the file are ignored.
- **The callback** (tch-rs `add_callback`; measured in M1 of cloud session
  4 with a C harness calling the trampoline three times):

  ```pascal
  // TchSys
  TTorchLoadCallback = procedure(data: Pointer; name: PChar; t: PTorchTensor); cdecl;
  procedure at_load_callback_with_device(filename: PChar; data: Pointer;
    f: TTorchLoadCallback; device_id: cint); cdecl; external LibTch;
  // TchNN, implementation
  type
    TLoadContext = class
      Items: TNamedTensorArray;
      Error: Exception;                 // acquired exception object, nil = none
    end;
  procedure LoadCallback(data: Pointer; name: PChar; t: PTorchTensor); cdecl;
  var Ctx: TLoadContext; N: String; T: TTensor;
  begin
    Ctx := TLoadContext(data);
    try
      T := TTensor.FromRaw(t);          // own the handle on every path (section 2)
      if Ctx.Error <> nil then Exit;    // an earlier callback failed: keep skipping
      N := StringReplace(StrPas(name), '|', '.', [rfReplaceAll]);
      SetLength(Ctx.Items, Length(Ctx.Items) + 1);
      Ctx.Items[High(Ctx.Items)].Name := N;
      Ctx.Items[High(Ctx.Items)].Tensor := T;
    except
      Ctx.Error := Exception(AcquireExceptionObject);
    end;
  end;
  ```

  After the C call returns: `P := get_and_reset_last_err` unconditionally
  (the slot must be clear on every path); if `Ctx.Error <> nil` then
  `tchf_free(P)`, take the object out of the context and `raise` it;
  otherwise if `P <> nil` then `TchRaiseNative(P)`. Measured: the class
  instance arrives intact through `data`; an exception raised in the
  trampoline is caught there, the acquired object survives the return
  through the C frames and re-raises with its class and message after the
  call; the later callbacks are skipped; every handle handed over is owned.
  The borrowed `name` is copied before the callback returns.
- **Copying into the store** (`load_internal`, `load_partial`,
  `copy_data_with_precision_update`). `Load`: a first pass checks that every
  store name is in the loaded list and raises `ETorchError('variable <name>
  not found in <file>')` before anything is modified (tch-rs stops at the
  first missing name in HashMap order, leaving the earlier ones updated;
  two passes make it deterministic). Then for each entry, under
  `NoGradGuard`: `Dst.set_data(Dst.totype(Src.Kind))` then
  `Dst.CopyFrom(Src)`: the destination takes the *file's* kind, which is why
  a `kFloat` store loading a half-precision file ends up `kHalf` (tch-rs
  test `save_and_load_var_store`, second half). `LoadPartial`: same without
  the first pass; the missing names are returned in insertion order
  (tch-rs's `save_and_load_partial_var_store_incomplete_file` expects
  `['a.b.t2']`). **(H15)**: the precision update on real libtorch.
  **(H15) resolved** (local session 3): after `SetFloatKind(kHalf)`, save,
  and `Load` into a fresh `kFloat` store, both variables are `kHalf` with
  means 42 and 2, as in tch-rs's `save_and_load_var_store`.

#### 4.6.4 Optimizers, from `nn/optimizer.rs`, `wrappers/optimizer.rs` and `torch_api.cpp`

- **L2:** `PTorchOptimizer = ^TTorchOptimizerOpaque;` in `TchSys` (same
  style as the existing `PTorchTensor`, section 3) and the externals exactly
  as `torch_api.h`
  declares them:

  ```pascal
  function ato_sgd(learning_rate, momentum, dampening, weight_decay: Double;
    nesterov: cint): PTorchOptimizer; cdecl; external LibTch;
  function ato_adam(learning_rate, beta1, beta2, weight_decay, eps: Double;
    amsgrad: Boolean): PTorchOptimizer; cdecl; external LibTch;      // C++ bool (section 2)
  function ato_adamw(...same as ato_adam...): PTorchOptimizer; cdecl; external LibTch;
  function ato_rms_prop(learning_rate, alpha, eps, weight_decay, momentum: Double;
    centered: cint): PTorchOptimizer; cdecl; external LibTch;
  procedure ato_add_parameters(opt: PTorchOptimizer; t: PTorchTensor; group: csize_t); cdecl; external LibTch;
  procedure ato_set_learning_rate(opt: PTorchOptimizer; learning_rate: Double); cdecl; external LibTch;
  procedure ato_set_learning_rate_group(opt: PTorchOptimizer; group: csize_t; learning_rate: Double); ...
  procedure ato_set_momentum(opt: PTorchOptimizer; momentum: Double); ...
  procedure ato_set_momentum_group(opt: PTorchOptimizer; group: csize_t; momentum: Double); ...
  procedure ato_set_weight_decay(opt: PTorchOptimizer; weight_decay: Double); ...
  procedure ato_set_weight_decay_group(opt: PTorchOptimizer; group: csize_t; weight_decay: Double); ...
  procedure ato_zero_grad(opt: PTorchOptimizer); ...
  procedure ato_step(opt: PTorchOptimizer); ...
  procedure ato_free(opt: PTorchOptimizer); ...
  ```

  All are `PROTECT`ed except `ato_free` (a bare `delete`), so `TchCheck`
  after each; the constructors return `nullptr` on error, which `TchCheck`
  turns into an exception before the pointer is looked at. A `Boolean`
  argument for a C++ `bool` was measured in step 4 (`at_autocast_set_enabled`).
- **Construction** (`OptimizerConfig::build`): create the C optimizer, then
  `ato_add_parameters(FOpt, Tensor.Raw, Group)` for every `FTrainable`
  entry in order, `FVarsInOptimizer := Length(FStore.FTrainable)`.
  `ato_add_parameters` creates param groups up to `group`, each a clone of
  the defaults, so groups may be sparse (as in tch-rs). `CreateRaw(AOpt,
  VS)` is a private constructor used by the four class functions; FPC 3.2.2
  warns `(3018) Constructor should be public` about it (measured in cloud
  session 5: for `strict private` too). Decision: keep it private — the
  factories are the API and a public raw-pointer constructor is not — and
  silence the warning with `{$push}{$warn 3018 off} … {$pop}` around the
  declaration (measured: no warning left).
- **Lazy add** (`add_missing_variables`): `ZeroGrad`, `Step` and the three
  `BackwardStep*` first add `FStore.FTrainable[FVarsInOptimizer..]` with
  their groups and update the count; `ClipGrad*` and the setters do not.
  This reads `TVarStore`'s private fields, which is why `TOptimizer` lives
  in `TchNN` (4.5).
- **Defaults** (`impl Default` in `nn/optimizer.rs`): SGD momentum 0,
  dampening 0, weight decay 0, nesterov false; Adam β₁ 0.9, β₂ 0.999,
  weight decay 0, eps 1e-8, amsgrad false; **AdamW weight decay 0.01**,
  the rest as Adam; RmsProp alpha 0.99, eps 1e-8, weight decay 0, momentum
  0, centered false. Parameter order in Pascal follows the tch-rs structs,
  `LR` first as in `build(vs, lr)`.
- **`ClipGradValue(Max)`** (`clip_grad_value`): for every trainable entry
  `G := T.grad; if G.Defined then G.clamp_(-Max, Max)`.
  **`ClipGradNorm(Max)`** (`clip_grad_norm`): collect `G.norm` (the
  parameterless generated `norm`, 2-norm of the flattened gradient) of every
  defined gradient; if none, exit (tch-rs would panic in `stack` of an
  empty list); `Total := TTensor.stack(Norms, 0).norm.DoubleValue([])`;
  `Coef := Max / (Total + 1e-6)`; if `Coef < 1` then `G.mul_scalar_(Coef)`
  on each defined gradient. Both run under `NoGradGuard` (tch-rs only the
  norm variant; PyTorch's `clip_grad_value_` is `@torch.no_grad()` too).
  Hand-checkable numbers from tch-rs `tests/nn_tests.rs`
  (`gradient_clip_test`): `v1 = [1, 1]`, `v2 = [1]`, `v3 = [-1, -1]`,
  `loss = sum(cat([v1, 2·v2, 2·v3])²)` → gradients `[2, 2]`, `[8]`,
  `[-8, -8]`; `ClipGradValue(4)` → `[2, 2]`, `[4]`, `[-4, -4]`;
  `ClipGradNorm(1)`: total norm `sqrt(4+4+64+64+64) = sqrt(200) = 14.1421`,
  coefficient `0.070711` → `[0.1414, 0.1414]`, `[0.5657]`,
  `[-0.5657, -0.5657]` (rounded to 4 places).
- **Setters:** `SetLearningRate` (defaults and every group),
  `SetLearningRateGroup`, `SetMomentum` (β₁ for Adam/AdamW, as libtch's
  `ato_set_momentum`), `SetMomentumGroup`, `SetWeightDecay`,
  `SetWeightDecayGroup`; `TchCheck` after each. **Fork patch found while
  reading `torch_api.cpp`:** in `ato_set_momentum_group` the SGD branch is
  `} if (auto sgd = ...)` instead of `} else if`, so for Adam, AdamW and
  RMSprop the value is set and then `"unexpected optimizer"` is thrown
  anyway (upstream tch-rs bug). The fork changes it to `} else if`
  (one token, `torch_api.cpp` line 564); **(H16)**: the unpatched build
  raises `ETorchNativeError('unexpected optimizer')` from
  `SetMomentumGroup(0, …)` on an Adam optimizer, the patched one does not.
  **(H16) resolved** (local session 3): `build/nn_test h16` against a build
  of the unpatched source fails with "unexpected optimizer"; against the
  patched `build/libtch` it passes.
- **`BackwardStep(Loss)`**: lazy add, `ato_zero_grad`, `Loss.Backward`,
  `ato_step`. `BackwardStepClip` / `BackwardStepClipNorm` insert the
  clipping between backward and step. `TrainableVariables` returns the
  store's. `Destroy` calls `ato_free` and never touches the store. The
  param groups hold C++ `Tensor` copies, so libtorch stays memory-safe if
  the store is freed first, but the lazy add would then dereference a freed
  `TVarStore`: free the optimizer before the store (documented).

#### 4.6.5 Layers of milestone 2

Set: `TLinear`, `TConv2D`, `TBatchNorm` (Nd 1/2/3), `TLayerNorm`,
`TEmbedding`, and dropout as a function (`X.dropout(P, Train)` in a user
`Forward`, or `Seq.AddFn(@Drop)` with a `TTensorFuncT`), plus `TFuncModule`
and `TSequential` (4.6.6). Deferred to milestone 3, with the same pattern:
`TConv1D` / `TConv3D` and the per-dimension `ConvConfigND`,
`TConvTranspose`, `TGroupNorm` (identical to `TLayerNorm` over
`group_norm`), `TLSTM` / `TGRU` (`lstm` / `gru` have `out` parameters and
need a state record and `ZeroState`), `Id` — all designed in 4.6.10
(design session 7). Variable names, shapes, default
initialisations and the generated op each `Forward` calls, all from the
tch-rs file named:

| layer (tch-rs file) | variables: name `[shape]`, default init | config defaults | `Forward` |
|---|---|---|---|
| `TLinear` (`linear.rs`) | `weight [Out, In]` KaimingUniform; `bias [Out]` when `Bias`: `BsInit` if `HasBsInit`, else `Uniform(-1/sqrt(In), 1/sqrt(In))` | `WsInit` KaimingUniform, `HasBsInit` False, `Bias` True | `X.linear(Weight, Bias)` — `atg_linear`, as tch-rs; its `matches_pytorch` test shows `linear` at least as close to PyTorch as `matmul(ws.tr()) + bs`. `Bias.IsNil` passes `TensorOption` none |
| `TConv2D` (`conv.rs`, scalar `ConvConfig`) | `bias [Out]` `BsInit` (registered first, as tch-rs); `weight [Out, In div Groups, K, K]` `WsInit` | `Stride` 1, `Padding` 0, `Dilation` 1, `Groups` 1, `Bias` True, `WsInit` KaimingUniform, `BsInit` Constant(0), `PaddingMode` pmZeros | `pmZeros`: `X.conv2d(Weight, Bias, [S, S], [P, P], [D, D], Groups)`; other modes: `X.pad(FPadTwice, Mode, NoDouble).conv2d(Weight, Bias, [S, S], [0, 0], [D, D], Groups)` with `FPadTwice = [P, P, P, P]` (reversed padding repeated twice) and `Mode` `'reflect'` / `'replicate'` / `'circular'` (`PaddingMode::to_string`) |
| `TBatchNorm` (`batch_norm.rs`) | `weight [Out]` `WsInit`, `bias [Out]` `BsInit` when `Affine`; `running_mean [Out]` `ZerosNoTrain`; `running_var [Out]` `OnesNoTrain` (this registration order) | `CudnnEnabled` True, `Eps` 1e-5, `Momentum` 0.1, `Affine` True, `WsInit` Uniform(0, 1), `BsInit` Constant(0) | dim check first (Nd = 1: input must be 2-D or 3-D; else `Nd + 2` dims; `ETorchError` with tch-rs's text), then `X.batch_norm(Weight, Bias, RunningMean, RunningVar, Train, Momentum, Eps, CudnnEnabled)`; the one milestone-2 layer that uses `Train` |
| `TLayerNorm` (`layer_norm.rs`) | `weight [NormalizedShape]` `WsInit`, `bias [NormalizedShape]` `BsInit` when `ElementwiseAffine` | `CudnnEnabled` True, `Eps` 1e-5, `ElementwiseAffine` True, `WsInit` Constant(1), `BsInit` Constant(0) | `X.layer_norm(NormalizedShape, Weight, Bias, Eps, CudnnEnabled)` |
| `TEmbedding` (`sparse.rs`) | `weight [NumEmbeddings, EmbeddingDim]` `WsInit` | `Sparse` False, `ScaleGradByFreq` False, `WsInit` Randn(0, 1), `PaddingIdx` -1 | `TTensor.embedding(Weight, X, PaddingIdx, ScaleGradByFreq, Sparse)` (class function: the receiver rule made `weight` the first argument) |

Every config has a `Default*Config` function and each layer two
constructors (with and without the config). Variable names are tch-rs's,
so `.ot` files interchange.

**Failed construction** (added after the overall review, 2026-10-13). A
constructor that raises after registering variables used to leave them in
the store (*measured*: `TConv2D` with `Groups = 0` left `c.bias` as a
trainable variable, which a later optimizer would take and `Save` would
write). Every constructor that registers variables (milestones 2 and 3
layers, and `TTrainableJitModule.RegisterInto`) now records `Mark :=
P.Store.Count` after `inherited Create` and, on any exception, calls
`P.Store.RollbackTo(Mark)` and re-raises; FPC then runs the destructor of
the half-built object as usual. The user chose rollback over pre-checking
preconditions; it also covers failures no pre-check foresees (an init that
raises after the weight was registered, `TestRollback` item 3). Variables that
existed before the call are never removed, and an optimizer created earlier
is unaffected (it takes variables in only at `ZeroGrad`/`Step`). Test:
`TestRollback` in `tests/nn/nn_test.pas`.

#### 4.6.6 `TSequential` and `TFuncModule`, from `sequential.rs` and `func.rs`

- **Container:** `TModuleList = specialize TObjectList<TModule>`
  (`Generics.Collections`, `OwnsObjects` True by default). Measured (M2):
  it compiles on 3.2.2 with an abstract base whose virtual method has a
  default parameter, `for M in FLayers` works, `Free` frees the children.
  `Contnrs.TObjectList` works as well but needs a cast on every `Items[I]`;
  since `Generics.Collections` is already pulled in for the name index, the
  typed list is used.
- **`AddFn` is one name with four overloads** on `TTensorFunc`,
  `TTensorFuncT`, `TTensorMethod` (`of object`) and `TTensorMethodT`.
  Measured (M3, objfpc): `AddFn(@PlainFunc)`, `AddFn(@PlainFuncT)`,
  `AddFn(@Obj.Method)` and `AddFn(@Obj.MethodT)` each resolve to the
  matching procedural type with no ambiguity, and chaining through the
  returned `Self` works. Each appends a `TFuncModule`, which holds one of
  the four pointers and can also be created standalone. With a method
  pointer the object must outlive the sequential (documented).
- **`Forward(X, Train)`** passes `Train` to every child
  (`SequentialT::forward_t`; one class covers tch-rs's `Sequential` and
  `SequentialT` because `TModule` already merges the two traits). An empty
  sequential returns `X` (a record copy of the same handle, tch-rs's
  `shallow_clone`). `ForwardAll(X, Train, N)` returns the outputs of the
  first `N` layers (`forward_all_t`; `N = -1` means all).
- `Add(M)` takes ownership and returns `Self` for chaining; `Count` and the
  default `Items[I]` property expose the children.

#### 4.6.7 Unit layout, L2 additions and the milestone 2 test

- **One unit `TchNN.pas`** (no `TchOptim`): `TOptimizer` reads
  `TVarStore.FTrainable` / `FVars`, which same-unit private access allows
  (4.5). `TchJit.pas` came with the JIT milestone (4.6.8).
- **`TchSys.pas` additions:** `PTorchOptimizer`, `TTorchLoadCallback`,
  `at_save_multi`, `at_load_callback_with_device`,
  `at_loadz_callback_with_device`, the fourteen `ato_*` of 4.6.4 (thirteen
  plus `ato_free`). **`Tch.pas` addition:** `TTensor.CopyFrom(const Src:
  TTensor)` over `at_copy_`. **Fork patch:** `ato_set_momentum_group`
  `} else if` (4.6.4), then rebuild `libtch.so`.
- **`tests/nn/nn_test.pas`** (real libtorch; build line in its header),
  hand-checkable numbers. This list says what the test does since local
  session 3, as reviewed in cloud session 5 (the additions beyond cloud
  session 4's list are adopted; one addition is still to be made, in 7d):
  1. Init: `KaimingUniform` on `[4, 16]`: every value in
     `±sqrt(6/16) = ±0.6124`, mean within 0.1 of 0; `FanForDims` of
     `[8, 3, 5, 5]` = 75 (in) / 200 (out); `Constant(3)` all 3,
     `Constant(1)` keeps the asked kind; `Orthogonal` `[3, 5]`, `[5, 3]` and
     `[2, 3, 2]` with gain 2 (H14), result shape = `Dims`, 1-d raises.
     `Apply` (H17): `fill_` on a trainable leaf without the guard raises,
     `Apply` of every `TInit` kind succeeds and the variable still requires
     grad.
  2. Store: `a.b.t2` naming through `Sub` and `/`; `Sub('x.y')`,
     `Zeros('a.b', …)`, `Zeros('a|b', …)` and `Sub('')` raise and add
     nothing; adding `t123` twice gives `Count = 4` with the second named
     `a.b.ccc.t123__3` (tch-rs `save_and_load_var_store`); `TryGet` by full
     name, through a path, and of a missing name; `GetOrAdd` of an existing
     name returns the `[3, 1, 4]` tensor whatever dims are asked;
     `ZerosNoTrain` does not require grad; `TrainableVariables` in
     registration order; `Freeze` → `RequiresGrad = False`, `Unfreeze` →
     True; `VarCopy`. `Freeze; Step` changes nothing (H18). `SetKind(kDouble)`
     after creating a `TLinear` and an SGD optimizer: the layer's `Weight`
     is `kDouble`, the step gives the hand value `[0.9, 0.8]` and stays
     `kDouble` (H13); `SetDevice(Cpu)` keeps the handle usable;
     `TVarPath.SetKind(kFloat)` on the `lin` sub-tree converts `weight` and
     `bias` only.
  3. Linear regression (tch-rs `optimizer_test`): `xs = 1..14` as
     `[14, 1]` float, `ys = 0.42·xs + 1.337`, `TLinear(1, 1)` with both
     inits `Constant(0)`, `Sgd(VS, 0)` then `SetLearningRate(1e-2)`. Initial
     `mse_loss` = `mean((0.42x + 1.337)²) = (0.1764·1015 + 1.12308·105 +
     14·1.787569) / 14 = 22.99967`. After **one** step: `w = 0.80955`,
     `b = 0.08974` (`∂/∂w = -2/14·Σx·y = -2/14·566.685 = -80.955`,
     `∂/∂b = -2/14·Σy = -2/14·62.818 = -8.974`), to 1e-3 in float32. After
     49 `BackwardStep`s loss < 0.25; `Apply(Constant(0))` on both resets the
     loss to the initial value; `SetLearningRate(1e-10)` and 49 more steps:
     tch-rs checks `final - initial < 1e-5` (**signed**). An absolute bound
     of 1e-5 does not hold: with the gradient essentially constant, each
     step lowers the loss by `lr·‖g‖² = 1e-10 · (80.955² + 8.974²) =
     1e-10 · 6634.2 = 6.634e-7`, so 49 steps give `3.25e-5`; measured
     `-3.24e-5` (local session 3), within the float32 resolution of a loss
     near 23 (ulp 1.9e-6). The test checks both: the signed tch-rs bound
     and `Near(L - Initial, -3.25e-5, 5e-6)`. Arithmetic confirmed in cloud
     session 5. Also: `TLinear` with `Bias False` registers one variable,
     `Bias.IsNil`, forward `[4, 3] → [4, 2]`.
  4. Adam: parameter `p = [1, -2]`, `loss = sum(p²)`, `Adam(VS, 0.1)`, one
     `BackwardStep` → `p = [0.9, -1.9]` to 1e-6: Adam's first step is
     `-lr·g/(|g| + eps)`, i.e. `lr·sign(g)`. AdamW with the default weight
     decay 0.01 on the same `p`: PyTorch decays first, `p ← p·(1 - lr·wd) =
     [0.999, -1.998]`, then the Adam step `lr·sign(g)` → **`[0.899,
     -1.898]`** to 1e-6 (local session 3 addition, arithmetic confirmed).
     RmsProp step and `SetMomentum`, SGD `SetMomentumGroup`, the Adam setters
     raise nothing; `SetMomentumGroup(0, 0.8)` on Adam raises nothing with
     the patched fork (H16; `build/nn_test h16` runs this check alone).
  5. Gradient clipping: the numbers of 4.6.4; `ClipGradNorm` with no
     gradients exits quietly; `BackwardStepClip(4)` steps with the clipped
     gradient (`v2 = 1 - 4`).
  6. Lazy add: create the optimizer, then add a trainable variable, one
     `BackwardStep` changes it and `TrainableVariables` sees it; a variable
     in group 1 with `SetLearningRateGroup(1, 0)` does not move while group
     0 does.
  7. Save → Load, with the layouts of tch-rs's `tests/var_store.rs`:
     *full* (`save_and_load_var_store`: `a.b.t2 [3]`, `t1 [4]`,
     `a.b.ccc.t123 [3]` **twice** → `t123__3`), *add* (`t1`, `t123` once)
     and *add_partial* (`a.b.t2`, `t1`, `t123` once).
     (a) Two full-layout stores; under `NoGradGuard` `u += 42`, `v *= 2`;
     `Save` to a temp `.ot`; the second store `Load`s it: means 42 and 2,
     nested names round-trip through `|`, the loaded variable still
     requires grad. (b) `Load` into a full-layout store with an extra
     `extra.x` raises `ETorchError` naming it and leaves the store unchanged
     (tch-rs `save_and_load_var_store_incomplete_file`, `should_panic`);
     `LoadPartial` there returns `['extra.x']` and loads the rest.
     (c) `SetFloatKind(kHalf)`, save, `Load` into a fresh float store →
     variables `kHalf` with the same means (H15).
     (d) tch-rs `save_and_load_partial_var_store_incomplete_file`: store 1
     with *add*, store 2 with *add_partial*, `u1 += 42`, `LoadPartial`
     returns **`['a.b.t2']`**, `u2 = 42`, `v2` untouched (1). Cloud session
     4 wrote this case with the full layout, where `a.b.ccc.t123__3` would
     be missing as well; corrected (local session 3 / cloud session 5).
     **Added** (local session 4), the second half of that tch-rs test:
     `SetFloatKind(kHalf)` on store 1, save, `LoadPartial` into a fresh
     float *add_partial* store → `u2` is `kHalf` with mean 42, the missing
     `v2` stays `kFloat` with mean 1, missing = `['a.b.t2']`.
     (e) `Save` to `.safetensors` raises `ETorchError`; `Load` of a missing
     file raises `ETorchNativeError`. (f) `CopyFrom`: copies by name
     (`u2 = 7`); a name missing in the source raises `ETorchError` naming it.
  8. Layers: `TConv2D(1, 1, 3)` with `Padding 1`, `Bias False`, weights
     filled with 1 on `[[1, 2], [3, 4]]` (NCHW `[1, 1, 2, 2]`): zeros
     padding → `[10, 10, 10, 10]`, reflect → `[27, 24, 21, 18]`, circular
     → `[27, 24, 21, 18]`, replicate → `[18, 21, 24, 27]` (tch-rs `conv`
     test); `TConv2D(4, 6, 3)` registers `c.bias` then `c.weight` and maps
     `[2, 4, 8, 8] → [2, 6, 6, 6]`. `TBatchNorm(1, 40)` registers 4
     variables (`running_mean`, `running_var` not trainable; 2 with
     `Affine` False), forward on `[10, 40]` in train mode runs and changes
     the running stats; `[10, 40, 3, 3]` raises; `BatchNorm2d` eval forward
     keeps the shape. `TLayerNorm([2])` fit (tch-rs
     `layer_norm_parameters_test`: `xs = [[42, 42], [42, 24]]`, SGD lr 1.0,
     initial loss > 1, after 49 steps < 0.25, reset restores the initial
     loss). `TEmbedding(10, 4)` on `randint(10, [5, 7])` gives `[5, 7, 4]`
     and the `PaddingIdx` row equals `Weight`'s row. `TLinear` registers
     `weight` and `bias` (`Count` 2, or 1 without bias).
  9. Sequential: an empty sequential returns the same handle and
     `ForwardAll` of it is `[X]`; `AddFn(@Relu)` on `[1, -1, 2]` →
     `[1, 0, 2]`; the four `AddFn` overloads chained (function,
     function+Train, method, method+Train on a `Factor = 3` object):
     `Forward(X, True)` = `[9, 0, 18]` and the `TTensorFuncT` sees `Train`,
     `Forward(X)` = `[3, 0, 6]`; `ForwardAll` returns one tensor per layer,
     `ForwardAll(N = 2)` two; `Items[0] is TFuncModule`. An MLP of two
     `TLinear` and a ReLU: `[4, 3] → [4, 2]`, 4 variables.
  10. `TchLiveHandles` returns to its starting value after freeing the
      optimizer, the layers and the store, in that order (a 5-step Adam
      training loop over an MLP in between), and is 0 at the very end.
- **What the mock can cover** (extend `tests/mock/mock_tch.c`, plumbing
  only): `ato_*` stubs over a counter struct (adds per group, zero_grad and
  step counts, exposed as `mock_opt_*`) → construction and lazy-add counts;
  `at_save_multi` recording the names → `.`→`|` mapping and order;
  `at_load_callback_with_device` calling back with `a|b|t2`, `t1` and mock
  tensors, plus a file name that sets the error slot after the first
  callback → trampoline, `|`→`.` mapping, error ordering;
  `atg_set_requires_grad`, `atg_set_data`, `atg_fill_`, `atg_uniform_`
  (shape-only) → store bookkeeping, `Freeze`, `TInit` plumbing;
  `TSequential` / `TFuncModule` chaining over `add_scalar`. Not coverable
  there: anything numerical (distributions, QR, `linear`, `conv2d`,
  `batch_norm`, backward, `step`), which is the real-libtorch test above.

#### 4.6.8 JIT (`TchJit`), from `wrappers/jit.rs`, `nn/mod.rs`, `tests/jit_tests.rs` and `torch_api.cpp`

Designed in cloud session 5 against the same tch-rs commit. The
`atm_*`/`ati_*` region of the fork's `torch_api.{h,cpp}` is byte-identical
to upstream (checked with `diff`). What needs real libtorch is
**(H19)**–**(H25)**; what was measured on FPC 3.2.2 in the cloud is marked
*measured*. The complete interface below (`TIValue`, `TJitModule`,
`TTrainableJitModule`, the `Jit*` functions) was compiled as a throwaway
unit with stub bodies against the real `Tch`/`TchNN` units (*measured*, M8
in HANDOFF): no errors, and the call shapes used by the tests resolve.

**Scope (milestone 3).** Loading, running, saving and tracing TorchScript
modules; `TIValue` for `ati_tag` values 0–13 (None, Tensor, Double, Int,
Bool, Tuple, IntList, DoubleList, BoolList, String, TensorList,
GenericList, GenericDict) plus the two input-only values StringList and
Device; `TTrainableJitModule`; scripted modules built from TorchScript
source (a three-function fork addition, below); the profiling/fuser
switches. **Deferred**, with their L2 lines: TorchScript objects (tag 14 —
`ati_object_method_`, `ati_object_getattr_`, `ati_clone`,
`atm_create_class_`; tch-rs `Object`, tests `jit6` and `jit_double_free`,
assets `foo6.pt`/`foo7.pt`), because an object keeps its C `IValue` alive
and needs a refcounted handle like `TTensorHandle`, and nothing in the
fine-tuning use case needs them — **designed since in 4.6.9** (design
session 6); and `atm_fuser_cuda_*` (with CUDA) — **designed since in 4.8**
(design session 8) as `JitFuserCudaEnabled`/`SetJitFuserCudaEnabled`.

**L2 (`TchSys.pas`).** `PTorchModule` (`torch::jit::script::Module*`) and
`PTorchIValue` (`torch::jit::IValue*`), each a pointer to its own empty
record (section 3), `PPTorchIValue = ^PTorchIValue`. Externals exactly as
`torch_api.h`, with section 2's types (`cint` for `int`, `Boolean` for C++
`bool`, `csize_t`, `PChar`, `PPChar`, `PInt64`, `PDouble`);
`atm_named_parameters` has the load callback's signature and reuses
`TTorchLoadCallback`:

```pascal
function  atm_load(filename: PChar): PTorchModule; cdecl; external LibTch;
function  atm_load_on_device(filename: PChar; device: cint): PTorchModule; cdecl; external LibTch;
function  atm_load_str(data: PChar; sz: csize_t): PTorchModule; cdecl; external LibTch;
function  atm_load_str_on_device(data: PChar; sz: csize_t; device: cint): PTorchModule; cdecl; external LibTch;
function  atm_forward(m: PTorchModule; tensors: PPTorchTensor; ntensors: cint): PTorchTensor; cdecl; external LibTch;
function  atm_forward_(m: PTorchModule; ivalues: PPTorchIValue; nivalues: cint): PTorchIValue; cdecl; external LibTch;
function  atm_method(m: PTorchModule; method_name: PChar; tensors: PPTorchTensor; ntensors: cint): PTorchTensor; cdecl; external LibTch;
function  atm_method_(m: PTorchModule; method_name: PChar; ivalues: PPTorchIValue; nivalues: cint): PTorchIValue; cdecl; external LibTch;
procedure atm_eval(m: PTorchModule); cdecl; external LibTch;
procedure atm_train(m: PTorchModule); cdecl; external LibTch;
procedure atm_free(m: PTorchModule); cdecl; external LibTch;
procedure atm_to(m: PTorchModule; device, dtype: cint; non_blocking: Boolean); cdecl; external LibTch;
procedure atm_save(m: PTorchModule; filename: PChar); cdecl; external LibTch;
function  atm_get_profiling_mode: cint; cdecl; external LibTch;
procedure atm_set_profiling_mode(b: cint); cdecl; external LibTch;
procedure atm_set_tensor_expr_fuser_enabled(b: cint); cdecl; external LibTch;
function  atm_get_tensor_expr_fuser_enabled: Boolean; cdecl; external LibTch;
procedure at_set_graph_executor_optimize(b: Boolean); cdecl; external LibTch;
procedure atm_named_parameters(m: PTorchModule; data: Pointer; f: TTorchLoadCallback); cdecl; external LibTch;
function  atm_create_for_tracing(modl_name: PChar; inputs: PPTorchTensor; ninputs: cint): PTorchModule; cdecl; external LibTch;
procedure atm_end_tracing(m: PTorchModule; fn_name: PChar; outputs: PPTorchTensor; noutputs: cint); cdecl; external LibTch;
// fork additions (below)
function  atm_create(modl_name: PChar): PTorchModule; cdecl; external LibTch;
procedure atm_register_parameter(m: PTorchModule; name: PChar; t: PTorchTensor; is_buffer: Boolean); cdecl; external LibTch;
procedure atm_define(m: PTorchModule; src: PChar); cdecl; external LibTch;

function  ati_none: PTorchIValue; cdecl; external LibTch;
function  ati_tensor(t: PTorchTensor): PTorchIValue; cdecl; external LibTch;
function  ati_int(v: Int64): PTorchIValue; cdecl; external LibTch;
function  ati_double(v: Double): PTorchIValue; cdecl; external LibTch;
function  ati_bool(v: cint): PTorchIValue; cdecl; external LibTch;
function  ati_string(s: PChar): PTorchIValue; cdecl; external LibTch;
function  ati_tuple(ivalues: PPTorchIValue; n: cint): PTorchIValue; cdecl; external LibTch;
function  ati_generic_list(ivalues: PPTorchIValue; n: cint): PTorchIValue; cdecl; external LibTch;
function  ati_generic_dict(ivalues: PPTorchIValue; npairs: cint): PTorchIValue; cdecl; external LibTch;  // 2*npairs ivalues: k0, v0, k1, v1, ...
function  ati_int_list(v: PInt64; n: cint): PTorchIValue; cdecl; external LibTch;
function  ati_double_list(v: PDouble; n: cint): PTorchIValue; cdecl; external LibTch;
function  ati_bool_list(v: PChar; n: cint): PTorchIValue; cdecl; external LibTch;   // bytes 0/1
function  ati_string_list(v: PPChar; n: cint): PTorchIValue; cdecl; external LibTch;
function  ati_tensor_list(v: PPTorchTensor; n: cint): PTorchIValue; cdecl; external LibTch;
function  ati_device(device: cint): PTorchIValue; cdecl; external LibTch;
function  ati_tag(i: PTorchIValue): cint; cdecl; external LibTch;
function  ati_to_tensor(i: PTorchIValue): PTorchTensor; cdecl; external LibTch;
function  ati_to_int(i: PTorchIValue): Int64; cdecl; external LibTch;
function  ati_to_double(i: PTorchIValue): Double; cdecl; external LibTch;
function  ati_to_string(i: PTorchIValue): PChar; cdecl; external LibTch;   // strdup'd: tchf_free
function  ati_to_bool(i: PTorchIValue): cint; cdecl; external LibTch;
function  ati_length(i: PTorchIValue): cint; cdecl; external LibTch;
function  ati_tuple_length(i: PTorchIValue): cint; cdecl; external LibTch;
procedure ati_to_tuple(i: PTorchIValue; outputs: PPTorchIValue; n: cint); cdecl; external LibTch;
procedure ati_to_generic_list(i: PTorchIValue; outputs: PPTorchIValue; n: cint); cdecl; external LibTch;
procedure ati_to_generic_dict(i: PTorchIValue; outputs: PPTorchIValue; npairs: cint); cdecl; external LibTch;  // writes 2*npairs
procedure ati_to_int_list(i: PTorchIValue; outputs: PInt64; n: cint); cdecl; external LibTch;
procedure ati_to_double_list(i: PTorchIValue; outputs: PDouble; n: cint); cdecl; external LibTch;
procedure ati_to_bool_list(i: PTorchIValue; outputs: PChar; n: cint); cdecl; external LibTch;
procedure ati_to_tensor_list(i: PTorchIValue; outputs: PPTorchTensor; n: cint); cdecl; external LibTch;
procedure ati_free(i: PTorchIValue); cdecl; external LibTch;
```

All are `PROTECT`ed except `atm_free`, `ati_free` and
`at_set_graph_executor_optimize`, so `TchCheck` after each; constructors
return nil on error and `TchCheck` raises before the pointer is looked at
(`ati_to_bool`/`ati_to_int` return -1 on error, also covered). Every
`tensor` handed to C goes through `TchCheckArg` first (4.2). Not declared,
as in 4.6.3's rule for unused entry points: `atm_create_class_`,
`ati_object_method_`, `ati_object_getattr_`, `ati_clone`,
`atm_fuser_cuda_set_enabled`, `atm_fuser_cuda_is_enabled`.

**Fork addition** (three functions, patched like the autocast and
`ato_set_momentum_group` fixes; declared in `torch_api.h` next to
`atm_create_for_tracing`):

```c
module atm_create(char *modl_name) {
  PROTECT(auto m = new torch::jit::script::Module(modl_name);
          m->register_attribute("training", c10::BoolType::get(), true);
          return m;)
  return nullptr;
}
void atm_register_parameter(module m, char *name, tensor t, bool is_buffer) {
  PROTECT(m->register_parameter(name, *t, is_buffer);)
}
void atm_define(module m, char *src) { PROTECT(m->define(src);) }
```

Why: tch-rs's test modules come from `tests/create_jit_models.py`. Without
Python, `torch::jit::Module::define` — libtorch's TorchScript compiler,
public C++ API — is the only way to build, at test time, a module with
parameters, methods with non-tensor signatures or a `training` flag.
`atm_create_for_tracing` cannot: a tensor used inside a traced function
becomes a constant, and a tensor that requires grad cannot be traced as a
constant at all. `training` is registered because `Module::train()`
silently skips modules without that attribute (`module.cpp`, the assert is
commented out upstream) and `self.training` in a method needs it; Python's
`ScriptModule` has it too. **(H20)**: builds and behaves as described in
the tests below. It is also a feature (TorchScript from Pascal), exposed
as `CreateEmpty`/`RegisterParameter`/`Define` and stated as a deviation
from tch-rs.

**`TIValue`.** Tag values equal `ati_tag`'s where one exists (*measured*:
`TIValueTag(12) = ivGenericList`), so `FromC` casts after a range check.
Nested type for the array of itself, as `TTensorArray` (H10):

```pascal
type
  TIValueTag = (ivNone = 0, ivTensor = 1, ivDouble = 2, ivInt = 3, ivBool = 4, ivTuple = 5,
    ivIntList = 6, ivDoubleList = 7, ivBoolList = 8, ivString = 9, ivTensorList = 10,
    ivGenericList = 12, ivGenericDict = 13,
    ivStringList = 20, ivDevice = 21);   // input only: no ati_tag value (see below)
  TBooleanArray = array of Boolean;      // before the record that uses it

  TIValue = record
  public type
    TIValueArray = array of TIValue;
  public
    Tag: TIValueTag;              // Default(TIValue) is None
    Tensor: TTensor;              // ivTensor
    I: Int64;                     // ivInt
    D: Double;                    // ivDouble
    B: Boolean;                   // ivBool
    S: String;                    // ivString
    Ints: TInt64Array;            // ivIntList
    Doubles: TDoubleArray;        // ivDoubleList
    Bools: TBooleanArray;         // ivBoolList
    Strings: TStringArray;        // ivStringList (input only)
    Tensors: TTensorArray;        // ivTensorList
    Items: TIValueArray;          // ivTuple, ivGenericList; ivGenericDict: k0, v0, k1, v1, ... (flat, as the C side)
    Dev: TDevice;                 // ivDevice (input only)
    class function None: TIValue; static;
    class function FromTensor(const T: TTensor): TIValue; static;
    class function FromInt(V: Int64): TIValue; static;
    class function FromDouble(V: Double): TIValue; static;
    class function FromBool(V: Boolean): TIValue; static;
    class function FromString(const V: String): TIValue; static;
    class function FromDevice(const V: TDevice): TIValue; static;
    class function Tuple(const A: array of TIValue): TIValue; static;
    class function IntList(const A: array of Int64): TIValue; static;      // inside TIValue, Tch's IntList needs the unit prefix
    class function DoubleList(const A: array of Double): TIValue; static;
    class function BoolList(const A: array of Boolean): TIValue; static;
    class function StringList(const A: array of String): TIValue; static;
    class function TensorList(const A: array of TTensor): TIValue; static;
    class function GenericList(const A: array of TIValue): TIValue; static;
    class function GenericDict(const KeysAndValues: array of TIValue): TIValue; static;  // even length, else ETorchError
    class operator :=(const V: TTensor): TIValue;
    class operator :=(V: Int64): TIValue;
    class operator :=(V: Double): TIValue;
    class operator :=(V: Boolean): TIValue;
    class operator :=(const V: String): TIValue;
    function IsNone: Boolean;
    function TagName: String;                                 // 'Tensor', 'Tuple', ... (tch-rs type_str)
    function AsTensor: TTensor;        // ETorchError('ivalue: expected Tensor, got Tuple') on a mismatch
    function AsInt: Int64;  function AsDouble: Double;  function AsBool: Boolean;  function AsString: String;
    function AsIntList: TInt64Array;  function AsDoubleList: TDoubleArray;
    function AsBoolList: TBooleanArray;  function AsTensorList: TTensorArray;
    function Count: Integer;           // tuple/list: Length(Items); dict: Length(Items) div 2; typed lists: their length
    function Item(Index: Integer): TIValue;                   // ivTuple or ivGenericList (tch-rs unpacks both)
    function Key(Index: Integer): TIValue;  function Value(Index: Integer): TIValue;   // ivGenericDict
    function ToC: PTorchIValue;                               // new C ivalue, owned by the caller (ati_free)
    class function FromC(P: PTorchIValue): TIValue; static;   // consumes P (ati_free on every path)
  end;
  TIValueArray = TIValue.TIValueArray;
```

- *Measured* (M5): the record compiles with explicit tag values and nests;
  `V := 42` / `2.5` / `0.9` / `True` / `'foo'` / a record give the expected
  tags through the `:=` operators; and **`[T, 42, 1.5]` passed to an
  `array of TIValue` parameter converts element-wise** through them, so
  `M.ForwardIV([X, 42, 1.5])` is legal. There is deliberately no operator
  from arrays (ambiguous); the named constructors cover them.
- Semantics as in 4.6: copies share `Items`, treat as immutable.
- **`ToC`** (tch-rs `to_c`), per tag: `ati_none`; `ati_tensor(T.Raw)` after
  `TchCheckArg`; `ati_int`/`ati_double`/`ati_bool(Ord(B))`;
  `ati_string(PChar(S))`; typed lists over the record's own arrays
  (`ati_bool_list(PChar(@Bools[0]), n)` — `SizeOf(Boolean) = 1`, measured,
  values 0/1; `ati_string_list` over an `array of PChar` built from
  `Strings`; `ati_tensor_list` over `Raw`s, each `TchCheckArg`ed);
  `ati_device(Dev.ToC)`. Containers: convert the children first into a
  nil-initialised `array of PTorchIValue`, build with `ati_tuple` /
  `ati_generic_list` / `ati_generic_dict(…, Length(Items) div 2)`, and free
  every non-nil child in a `finally` — the C container copied the `IValue`s
  (tch-rs frees them too). An odd dict raises `ETorchError`. An empty array
  passes `nil` (never `@A[0]` of an empty array; `ati_tuple(nullptr, 0)` is
  fine). `TchCheck` after each `ati_*`.
- **`FromC`** (tch-rs `from_c`): `try … finally ati_free(P)`; `nil` raises
  `ETorchError`; `Tag := ati_tag(P); TchCheck` — 14 raises
  `ETorchError('ivalue: TorchScript objects are not supported yet')` (until 4.6.9), any
  other value outside the enum raises too (C itself throws
  "unsupported tag" for a Device). Then: Tensor →
  `TTensor.FromRaw(ati_to_tensor(P))` after `TchCheck`; scalars →
  `ati_to_int`/`double`/`bool` + `TchCheck`; String → `ati_to_string`,
  `StrPas`, `tchf_free` (it is `strdup`'d, section 2); typed lists → `N :=
  ati_length`, a buffer of `N`, `ati_to_*_list(P, PtrOrNil(buf), N)`
  (BoolList through a `Byte` buffer, `<> 0`); TensorList → nil-initialised
  `PTorchTensor` slots, `TchCheckSlots` (Tch.pas) on error, else `FromRaw`
  each; Tuple / GenericList → `N := ati_tuple_length` / `ati_length`,
  `ati_to_tuple` / `ati_to_generic_list` into nil-initialised
  `array of PTorchIValue`, then for each child `C := Cs[K]; Cs[K] := nil;
  Items[K] := FromC(C)` so that a `finally` can `ati_free` the children not
  yet consumed when a later conversion raises; GenericDict → `N :=
  ati_length` (pairs), `2·N` slots, `ati_to_generic_dict(P, Cs, N)`, the
  same consumption. The C side checks the size and throws before writing
  any slot, and `new` can fail with `bad_alloc` part-way, hence the
  nil-initialised slots (4.2's fixed-N rule).
- **Input-only values.** `ivStringList` builds a `c10::List<string>`, which
  `ati_tag` reports as 12: it comes back as a GenericList of String items
  (tch-rs's `jit5` expects exactly that). `ivDevice` has no `ati_tag`
  value; a Device output raises `ETorchNativeError` from `ati_tag`.
  GenericDict pairs come back in insertion order: `c10::Dict` is an
  order-preserving hash map, which tch-rs's round-trip test relies on
  **(H24)**.

**`TJitModule`** (tch-rs `CModule`):

```pascal
type
  TTraceFunc   = function(const Inputs: TTensorArray): TTensorArray;
  TTraceMethod = function(const Inputs: TTensorArray): TTensorArray of object;

  TJitModule = class
  private
    FModule: PTorchModule;
    {$push}{$warn 3018 off}
    constructor CreateRaw(M: PTorchModule);          // takes ownership; M <> nil (TchCheck ran)
    {$pop}
  public
    constructor Create(const FileName: String); overload;                           // atm_load
    constructor Create(const FileName: String; const ADevice: TDevice); overload;   // atm_load_on_device
    constructor CreateFromStream(S: TStream); overload;                             // atm_load_str
    constructor CreateFromStream(S: TStream; const ADevice: TDevice); overload;     // atm_load_str_on_device
    constructor CreateEmpty(const ModuleName: String);                              // atm_create (fork)
    class function CreateByTracing(const ModuleName, FnName: String;
      const Inputs: array of TTensor; Fn: TTraceFunc): TJitModule; overload; static;
    class function CreateByTracing(const ModuleName, FnName: String;
      const Inputs: array of TTensor; Fn: TTraceMethod): TJitModule; overload; static;
    destructor Destroy; override;                                                   // atm_free
    function Forward(const Inputs: array of TTensor): TTensor;                      // atm_forward
    function ForwardIV(const Inputs: array of TIValue): TIValue;                    // atm_forward_
    function Method(const Name: String; const Inputs: array of TTensor): TTensor;   // atm_method
    function MethodIV(const Name: String; const Inputs: array of TIValue): TIValue; // atm_method_
    procedure SetEval;                                                              // atm_eval
    procedure SetTrain;                                                             // atm_train
    procedure ToDevice(const ADevice: TDevice; AKind: TKind; NonBlocking: Boolean = False);  // atm_to
    procedure Save(const FileName: String);                                         // atm_save
    function NamedParameters: TNamedTensorArray;                                    // atm_named_parameters
    procedure RegisterParameter(const Name: String; const T: TTensor; IsBuffer: Boolean = False);  // fork
    procedure Define(const Source: String);                                         // fork: TorchScript source
    property Raw: PTorchModule read FModule;
  end;

function  JitGetProfilingMode: Boolean;              procedure JitSetProfilingMode(B: Boolean);
function  JitGetTensorExprFuserEnabled: Boolean;     procedure JitSetTensorExprFuserEnabled(B: Boolean);
procedure JitSetGraphExecutorOptimize(B: Boolean);
```

- `Forward`/`Method` return a tensor, `ForwardIV`/`MethodIV` a `TIValue`:
  that is the C split (`atm_forward` throws "forward did not return a
  tensor" for a tuple result; the test below relies on that message).
  Tensor inputs: `TchCheckArg` each, an `array of PTorchTensor` of `Raw`s,
  `nil` when empty. IValue inputs: `ToC` each into a nil-initialised array,
  call, `TchCheck`, `FromC(result)`; the inputs are freed in a `finally`
  (tch-rs frees them after the call).
- `NamedParameters`: a `cdecl` trampoline exactly as 4.6.3's `LoadCallback`
  with its own `TParamContext` in `TchJit` (fifteen lines duplicated rather
  than exporting `TchNN`'s private helper); names verbatim (`|`→`.` kept,
  harmless); `get_and_reset_last_err` unconditionally after the call, the
  acquired exception re-raised. Buffers are not parameters: `foo.pt`'s
  `value` gives an empty list.
- `CreateByTracing`: `atm_create_for_tracing(Name, inputs)`, `TchCheck`;
  `Outputs := Fn(Inputs as TTensorArray)` (the same handles, as tch-rs
  passes `inputs` to the closure); `atm_end_tracing(M, FnName, outputs,
  n)`, `TchCheck`. If `Fn` raises: libtorch's tracing state is global and
  libtch has no "abandon" entry point, so the handler calls
  `atm_end_tracing(M, PChar(FnName), nil, 0)`, clears the error slot
  (`get_and_reset_last_err` → `tchf_free`), `atm_free(M)` and re-raises —
  best effort, **(H23)**: a later `CreateByTracing` works ("cannot nest
  tracing calls" otherwise).
- `CreateFromStream` reads `S.Size - S.Position` bytes into a `TBytes` and
  calls `atm_load_str(@Buf[0], Length(Buf))` (C copies into a
  `std::string`); an empty stream raises `ETorchError` before the call.
- `ToDevice` keeps tch-rs's `to(device, kind, non_blocking)`: `atm_to` has
  no optional dtype.
- `Define(Source)`: TorchScript text, `def` at column 0, 4-space bodies,
  lines joined with `#10`; several `def`s in one call are fine. A compile
  error arrives as `ETorchNativeError`.
- Globals: `atm_get_profiling_mode <> 0`, `atm_set_profiling_mode(Ord(B))`,
  `atm_set_tensor_expr_fuser_enabled(Ord(B))`,
  `atm_get_tensor_expr_fuser_enabled` (C++ `bool` → `Boolean`),
  `at_set_graph_executor_optimize(B)` (no `PROTECT`, nothing to check).

**`TTrainableJitModule`** (tch-rs `TrainableCModule`, `nn/mod.rs`):

```pascal
type
  TJitMode = (jmUnknown, jmTrain, jmEval);
  TTrainableJitModule = class(TModule)
  private
    FInner: TJitModule;           // owned
    FMode: TJitMode;
    procedure RegisterInto(const P: TVarPath);
  public
    constructor Create(const FileName: String; const P: TVarPath);       // atm_load_on_device(P.Device)
    constructor CreateFromStream(S: TStream; const P: TVarPath);
    destructor Destroy; override;
    function Forward(const X: TTensor; Train: Boolean = False): TTensor; override;
    function ForwardIV(const Inputs: array of TIValue): TIValue;
    function Method(const Name: String; const Inputs: array of TTensor): TTensor;
    function MethodIV(const Name: String; const Inputs: array of TIValue): TIValue;
    procedure Save(const FileName: String);
    procedure SetTrain;  procedure SetEval;                                // also set FMode
    property Inner: TJitModule read FInner;
  end;
```

- Registration = tch-rs `TrainableCModule::load`: for every `(Name, T)` of
  `FInner.NamedParameters`, `P.Add(StringReplace(Name, '.', '_',
  [rfReplaceAll]), T, T.RequiresGrad)` — `.`→`_` because variable names
  reject `.` (`encoder.layer1.weight` → `encoder_layer1_weight` under `P`'s
  prefix); trainable = the loaded tensor's `requires_grad`. `Add` shares
  the handle: the store entry *is* the module's parameter
  (`new torch::Tensor(v)` shares the `TensorImpl`), so a `TOptimizer` on
  the store updates the scripted module in place, `SetDevice`/`SetKind`
  (`set_data`) move it, and `Save` writes the trained values **(H21)**.
  The store's kind is not applied (as tch-rs); the device is `P.Device`
  through `atm_load_on_device`.
- `Forward(X, Train)` = `FInner.Forward([X])`. tch-rs ignores `train`
  (`forward_t(&self, xs, _train)`: `set_train` needs `&mut self`).
  Deviation, stated: the Pascal version applies it — when `FMode` is
  `jmUnknown` or disagrees with `Train`, call `atm_train`/`atm_eval` and
  remember — so a scripted model with dropout or batch-norm behaves like
  the `TchNN` layers under `TSequential`'s `Train`. A loaded module's
  `training` flag is whatever Python saved, hence `jmUnknown` until the
  first `Forward`; `SetTrain`/`SetEval` set `FMode` too.

**Test assets without Python (decision): both of the following**, each for
what the other cannot do.
1. **Vendor tch-rs's `tests/foo.pt` … `foo8.pt`** into `tests/jit/models/`
   (nine files, 15 KB, PyTorch zip format version 3; license MIT OR
   Apache-2.0 like ours; per CLAUDE.md tch-rs's two license files are
   copied next to them and the commit message names `tch-rs 4227b89b
   tests/foo*.pt` and `tests/create_jit_models.py` as the source). They
   are real Python-exported files — the use case — and `jit_tests.rs`
   gives the expectations; `foo6.pt`/`foo7.pt` wait for objects. Contents
   (read from the zips' `code/__torch__.py`): `foo` `2x + y + value` with
   the buffer `value = [42.0]`; `foo1` `2x + y`; `foo2` `(2x + y, x - y)`;
   `foo3` the product of the elements; `foo4` `Tuple[float, float, int]`
   → `x0 + x1·x2`; `foo5` `List[str]` → each string minus its last char;
   `foo8` `generate(Dict[str, Tensor])` → `(d['foo'], d['bar'])`. **(H19)**:
   they load with libtorch 2.13 (tch-rs's CI does so at this commit) and
   give the numbers below.
2. **Build modules in Pascal** with `CreateEmpty` + `RegisterParameter` +
   `Define` (fork, H20) where parameters or `self.training` are needed, and
   `CreateByTracing` for tch-rs's `create_traced`. Tracing alone was
   rejected as the only route: no parameters, no tuples, no non-tensor
   signatures.

**`tests/jit/jit_test.pas`** (real libtorch; run from the repo root like
tch-rs's tests; 0-d inputs as `TTensor.scalar_tensor(42, kInt64,
TDevice.Cpu)`; files under `tests/jit/models/`), hand-checkable numbers:
1. `foo.pt` (`jit`): `x = [3, 1, 4, 1, 5]`, `y = [7]` (kFloat):
   `Forward([x, y]) = 2x + y + 42 = [55, 51, 57, 51, 59]`; the same through
   `Method('forward', …)`; `NamedParameters` is empty (`value` is a
   buffer); `Create(file, TDevice.Cpu)` gives the same result.
2. `foo1.pt` (`jit1`): 0-d kInt64 `42` and `1337` → `2·42 + 1337 = 1421`,
   kind `kInt64`, through `Forward` and `Method('forward')`.
3. `foo2.pt` (`jit2`): `ForwardIV([42-tensor, 1337-tensor])` → `Tag =
   ivTuple`, `Count = 2`, `Item(0).AsTensor` = `1421`, `Item(1)` = `42 -
   1337 = -1295`; the same through `MethodIV`; `AsTensor` on the tuple
   raises `ETorchError` naming both tags; `Forward` (tensor API) on it
   raises `ETorchNativeError` "forward did not return a tensor".
4. `foo3.pt` (`jit3`): `[1, 2, 3, 4, 5]` → `120` (0-d).
5. `foo4.pt` (`jit4`): `ForwardIV([TIValue.Tuple([2.0, 3.0, 4])])` →
   `ivDouble` `2 + 3·4 = 14`; `NamedParameters` empty.
6. `foo5.pt` (`jit5`): `ForwardIV([TIValue.StringList(['foo', 'bar',
   'foobar'])])` → `ivGenericList` of three `ivString`: `'fo'`, `'ba'`,
   `'fooba'`.
7. `foo8.pt` (`specialized_dict`, tch-rs issue 597):
   `MethodIV('generate', [TIValue.GenericDict(['bar', T[1, 7], 'foo',
   T[1, 2, 3]])])` → tuple `([1, 2, 3], [1, 7])`. **(H22)**:
   `ati_generic_dict` specialises to `Dict[str, Tensor]` when every key is
   a string and every value a tensor, which the method's schema requires.
8. Round trips (tch-rs `ivalue_round_trip`, through the public
   `ToC`/`FromC`): `None`; `True`, `False`; `-1`, `42`, `15`; `''`,
   `'foobar'`; `Tuple([42, Pi])`; `IntList([42, 1337])`; `DoubleList([e,
   Pi, 299792458.00001])`; `Tuple([BoolList([T, F, T, T]), DoubleList(…)])`;
   `GenericList([42, 'foobar'])`; `GenericDict([42, 'foobar', 'foo',
   'bar'])` back in the same order (H24); plus a `Tensor` `[1, 2, 3]`
   (values and kind kept), the empty `Tuple([])` (nil-pointer path) and
   `StringList(['a', 'b'])` coming back as `ivGenericList` of `ivString`.
   An odd-length `GenericDict` raises `ETorchError`.
9. Tracing (`create_traced`): `CreateByTracing('MyModule', 'MyFn', [0.0,
   1.0 as 0-d kDouble], @AddFn)` with `AddFn = [Inputs[0] + Inputs[1]]`;
   `Save` to a temp file, `Create` it back, `Method('MyFn', [[1, 2, 3, 4,
   5], [41, 1335, Pi - 3, 4, 5]])` (kDouble) = `[42, 1337, Pi, 8, 10]`
   exactly (`Pi - 3` is exact by Sterbenz's lemma, so `(Pi - 3) + 3 = Pi`);
   `Method('nope', …)` raises `ETorchNativeError`; a closure that raises
   `ETorchError` propagates it, and a second `CreateByTracing` afterwards
   works (H23).
10. Scripted module from source (H20): `M := TJitModule.CreateEmpty('Lin')`;
    `RegisterParameter('weight', zeros [1, 3] with requires_grad)`,
    `('bias', zeros [1] with requires_grad)`, `('scale', ones [1], no
    grad)`; `Define('def forward(self, x):'#10'    return (torch.matmul(x,
    torch.t(self.weight)) + self.bias) * self.scale'#10)`; `Forward([[1, 2,
    3]]) = 0`; `NamedParameters` = `weight`, `bias`, `scale` (three, in
    registration order); `Save` to a temp `.pt`. Then `J :=
    TTrainableJitModule.Create(file, VS.Root / 'jit')`: `VS.Count = 3`
    (`jit.weight`, `jit.bias`, `jit.scale`), `TrainableVariables` two
    (`scale` has no grad); `Sgd(VS, 0.1)`, `X = [[1, 2, 3]]`, `Y = [[1]]`,
    `mse_loss = 1`; one `BackwardStep`: `∂/∂w = -2·x·(1 - 0) = [-2, -4,
    -6]`, `∂/∂b = -2` → `w = [0.2, 0.4, 0.6]`, `b = 0.2` and
    **`J.Forward(X) = 0.2 + 0.8 + 1.8 + 0.2 = 3.0`** (H21: the module sees
    the store's step); `J.Save`, reload as `TJitModule`, `Forward([X]) =
    3.0`.
11. Train flag (H20): `CreateEmpty('Mode')`, `Define('def forward(self,
    x):'#10'    if self.training:'#10'        return x * 2'#10'    return
    x'#10)`: fresh module `Forward([X]) = 2X` (`training` defaults to
    True), `SetEval` → `X`, `SetTrain` → `2X`; saved and reloaded as a
    `TTrainableJitModule`: `Forward(X, True) = 2X`, `Forward(X) = X`,
    `Forward(X, True) = 2X` again.
12. Globals: `JitGetProfilingMode` True in a fresh process **(H25)**, False
    after `JitSetProfilingMode(False)`, True again;
    `JitSetTensorExprFuserEnabled(True/False)` read back;
    `JitSetGraphExecutorOptimize(True)` raises nothing.
13. Errors and handles: `TJitModule.Create('tests/jit/models/missing.pt')`
    → `ETorchNativeError`; `Define('def forward(self, x): return')` →
    `ETorchNativeError`; `TchLiveHandles` back to the start after freeing
    every module, store and optimizer, and 0 at the very end.

**Mock** (`tests/mock/mock_tch.c`, plumbing only; `tests/mock/
jit_mock_test.pas`): `ivalue` as a tagged struct with real payloads and a
`mock_live_ivalues` counter (`ati_*` implemented faithfully for tags 0–13,
`ati_tag` of a string list reporting 12) → every round trip of item 8 on
the mock, `mock_live_ivalues = 0` afterwards, odd dict, nil pointer;
`module` struct with `mock_live_modules`: `atm_load*` fail when the name
contains `missing`, `atm_forward`/`atm_method` return a copy of the first
tensor (error when there is none), `atm_forward_`/`atm_method_` return a
tuple of clones of the inputs and record the method name
(`mock_last_method`), `atm_eval`/`atm_train` → `mock_module_training`,
`atm_to` records device and dtype, `atm_save` the file name,
`atm_named_parameters` calls back with `weight` `[2, 3]` (requires grad)
and `sub.bias` `[2]` (no grad) → `TTrainableJitModule` registers
`jit.weight` trainable and `jit.sub_bias` not; `atm_create_for_tracing` /
`atm_end_tracing` count calls (the raising-closure path calls
`atm_end_tracing` once with 0 outputs); `atm_create` /
`atm_register_parameter` / `atm_define` record counts and the last source;
`atm_free` decrements. Not coverable: TorchScript itself and every number
of items 1–11.

**Hypotheses (need real libtorch):**
- **(H19)** the vendored `foo*.pt` load with libtorch 2.13 and give the
  numbers of items 1–7.
- **(H20)** fork addition: `Module::define` compiles the two sources above
  in libtorch 2.13 (`torch.matmul`, `torch.t`, `self.training`),
  `register_attribute("training", Bool, true)` makes `self.training` and
  `atm_train`/`atm_eval` work, `named_parameters` returns the three
  registered parameters (not buffers) with their `requires_grad`.
- **(H21)** parameter sharing: after one SGD step on the store
  `J.Forward(X) = 3.0`, and the saved module reloads with the trained
  values.
- **(H22)** `ati_generic_dict`'s `Dict[str, Tensor]` specialisation is
  accepted by `foo8.pt`'s `generate`.
- **(H23)** `atm_end_tracing(M, name, nil, 0)` after a raising closure
  leaves tracing usable.
- **(H24)** a `GenericDict` round trip keeps insertion order.
- **(H25)** profiling mode is on by default.

**Results (local session 4, libtorch 2.13.0 CPU): H19–H25 all hold.**
`tests/jit/jit_test.pas` passes every item 1–13 above as written:
- **(H19) resolved:** all seven used models load (also through
  `CreateFromStream` and `Create(file, TDevice.Cpu)`) and give
  `jit_tests.rs`'s numbers.
- **(H20) resolved:** `Module::define` compiles both sources; a fresh
  `CreateEmpty` module runs the `self.training` branch, `SetEval`/
  `SetTrain` switch it; `NamedParameters` returns `weight`, `bias`,
  `scale` in registration order with their `requires_grad`.
- **(H21) resolved:** `w = [0.2, 0.4, 0.6]`, `b = 0.2`,
  `J.Forward(X) = 3.0` after one SGD step on the store, and the module
  saved by `J.Save` reloads with `Forward = 3.0`.
- **(H22) resolved:** `foo8.pt`'s `generate` accepts the specialised dict.
- **(H23) resolved:** after a raising closure, the next `CreateByTracing`
  works.
- **(H24) resolved:** `GenericDict([42, 'foobar', 'foo', 'bar'])` comes back
  in insertion order.
- **(H25) resolved:** `JitGetProfilingMode` is True at program start.

Implementation notes (deviations from the sketch above, all small):
- `TIValueTag` has explicit values, and FPC rejects such an enum as an
  array index ("enums with assignments cannot be used as array index",
  measured), so `TagName` is a `case`.
- `CreateByTracing`'s cleanup (`atm_end_tracing(M, FnName, nil, 0)`, clear
  the slot, `atm_free`) runs for *any* failure after
  `atm_create_for_tracing`, not only a raising `Fn`: a nil output tensor or
  a failing `atm_end_tracing` (before it resets the tracing state) would
  leave tracing on as well.
- `FromC` checks `ati_to_tensor` before wrapping the pointer; `Item`,
  `Key` and `Value` raise `ETorchError` on an index out of range.
- The mock (`tests/mock/jit_mock_test.pas`) additionally checks that a
  child failing in `ToC` (nil tensor) or in `FromC` (Device) leaves no C
  ivalue behind.

#### 4.6.9 TorchScript objects (`ivObject`, tag 14), from `wrappers/jit.rs` (`Object`), `tests/jit_tests.rs` (`jit6`, `jit_double_free`) and `torch_api.cpp`

Designed in design session 6 (2026-10-11) on the local machine, so against
the real libtorch 2.13.0 (CPU) and the built `libtch.so`: everything
marked *measured* below comes from a throwaway probe over the raw C entry
points (M9–M11 in HANDOFF); what the probe could not reach is
**(H26)**–**(H28)**. Scope: the four entry points 4.6.8 left undeclared
(`ati_object_method_`, `ati_object_getattr_`, `ati_clone`,
`atm_create_class_`), tch-rs's `Object`, the tests `jit6` and
`jit_double_free`, the vendored `foo6.pt`/`foo7.pt`. Nothing else in
4.6.8 changes.

**What the C side does** (`torch_api.cpp`, byte-identical to upstream):
- `ati_tag` returns 14 for `isObject()`. *Measured*: `foo6`'s `forward`
  result and `foo7`'s `make_input_object` result are tag 14.
- `ati_object_method_(i, name, ivalues, n)` pushes a *copy* of `*i` as
  `self`, then copies of the arguments, and runs
  `i->toObjectRef().type()->getMethod(name)`; the result is a new
  `IValue`. The pointer `i` is only read. *Measured*: `y()` on `foo6`'s
  object gives `[6, 8, 10]` for `x = [3, 4, 5]`.
- `ati_object_getattr_(i, name)` returns a new `IValue` holding the
  attribute (*measured*: `x` → `[3, 4, 5]`; `foo7`'s `foo`/`bar` →
  `[1, 2, 3]`/`[4, 5, 6]`).
- `ati_clone(i)` is `new IValue(*i)`. For an Object the `IValue` copy
  constructor copies the `c10::intrusive_ptr<ivalue::Object>`
  (`c10/core/ivalue.h`): two C heap cells, one TorchScript object.
  *Measured*: the clone's `y()` still works after `ati_free` of the
  original **and after `atm_free` of the module** — `ivalue::Object` holds
  a `StrongTypePtr`, i.e. a `shared_ptr` to the compilation unit, so an
  object does not need its module kept alive.
- `ati_to_tuple` / `ati_to_generic_list` / `ati_to_generic_dict` write a
  fresh `new IValue(child)` per slot. *Measured*: a tuple built from an
  object, freed, and unpacked gives a child pointer different from the
  original whose `y()` works after the tuple is gone.
- `atm_create_class_(m, clz, ivalues, n)` is
  `m->create_class(QualifiedName(clz), stack)`: looks the class up in the
  module's compilation unit by its **qualified** name and runs `__init__`.
  *Measured*: `'__torch__.InputObject'` with two tensors on `foo7` gives
  an object that `add_them` accepts (`[5, 7, 9]`); the bare
  `'InputObject'` fails. (The unused local `base` in that function is
  harmless and not patched: the region stays identical to upstream.)
- Messages (*measured*, first line of `what()`, libtorch 2.13.0):

  | call | message |
  |---|---|
  | `getattr 'nope'` | `__torch__.TorchScriptClass (of Python compilation unit at: 0x…) does not have an attribute with name 'nope'` |
  | `method 'nope'` | `Couldn't find method: 'nope' on class: '__torch__.TorchScriptClass (…)'` |
  | `method 'y'` with one argument | `Expected at most 1 argument(s) for operator 'y', but received 2 argument(s). Declaration: y(__torch__.TorchScriptClass self) -> Tensor` |
  | `ati_to_tensor` on an object | `Expected Tensor but got Object` |
  | `ati_length` on an object | `unsupported tag for length Object` (returns -1) |
  | `create_class 'InputObject'` / `'__torch__.Nope'` | `Could not find class with name: 'InputObject' in module.` |
  | `create_class` with one argument | `__init__() is missing value for argument 'bar'. Declaration: __init__(__torch__.InputObject self, Tensor foo, Tensor bar) -> NoneType` |
  | `add_them(5)` | `add_them() Expected a value of type '__torch__.InputObject (…)' for argument 'data' but instead found type 'int'.` |
  | `ati_object_getattr_`/`_method_` on a non-object | `isObject() INTERNAL ASSERT FAILED at ".../ivalue_inl.h":1699, please report a bug to PyTorch. Expected Object but got Int` |

  The last row is reachable only with a raw pointer: the Pascal API wraps
  tag-14 pointers only, so no guard is added for it.

**The handle (decision).**

| Option | Verdict |
|---|---|
| Raw `PTorchIValue` in `TIValue`, freed by hand | Rejected: `TIValue` is a record that is copied freely (`Items`, open arrays, results); the copies would double free — tch-rs issue 475 (`jit_double_free`) was exactly a second free of an object's `IValue`. |
| Record with management operators | Rejected as in 4.1. |
| A record `TJitObject` wrapping the handle, like `TTensor` | Rejected on a compiler fact: `TIValue` needs the field and the object's methods take and return `TIValue`, and FPC cannot forward-declare a record. |
| **Interface `IJitObject` refcounted by `TInterfacedObject`, implementing class hidden** | **Chosen.** FPC forward-declares interfaces: `IJitObject = interface;` → `TIValue` with an `Obj: IJitObject` field → the full interface naming `TIValue`. *Measured* (M10, FPC 3.2.2): compiles; `B := A` shares the handle (same `Raw`), releasing one copy keeps the other alive, the live count returns to 0 when the last reference goes; `SizeOf` of a record with a tag and the interface is 16. |

4.1's reasons against a public interface do not apply here: no operator
overloads, three methods, and the implementing class `TJitObjectHandle`
is declared in the implementation section, so nobody can hold it by class
reference. Thread-safety as 4.1: `TInterfacedObject`'s interlocked
refcount; the TorchScript object itself is as unsynchronised as a tensor.
Live counting as `TchLiveHandles`: a `LiveObjects` counter
(`InterlockedIncrement`/`Decrement` in the handle's constructor and
destructor) read through `JitLiveObjects`.

```pascal
unit TchJit;  // additions
type
  TIValueTag = (..., ivGenericDict = 13, ivObject = 14,
    ivStringList = 20, ivDevice = 21);

  IJitObject = interface;                       // forward: TIValue holds one, its methods use TIValue

  TIValue = record
    ...
    Obj: IJitObject;                            // ivObject (nil otherwise)
    class function FromObject(const O: IJitObject): TIValue; static;
    class operator :=(const O: IJitObject): TIValue;
    function AsObject: IJitObject;              // ETorchError('ivalue: expected Object, got Tuple') on a mismatch
    ...
  end;

  { A TorchScript object (tch-rs Object). Copies of a TIValue share it;
    the C IValue is freed when the last reference goes. It does not keep
    its module alive and does not need to. }
  IJitObject = interface
    ['{...new GUID...}']
    function Raw: PTorchIValue;                 // borrowed; always tag 14
    function MethodIV(const Name: String; const Inputs: array of TIValue): TIValue;  // ati_object_method_
    function GetAttr(const Name: String): TIValue;                                   // ati_object_getattr_
  end;

  TJitModule = class
    ...
    { atm_create_class_: QualifiedName is e.g. '__torch__.InputObject' (not ClassName: that is TObject.ClassName, a duplicate identifier — measured in local session 6) }
    function CreateClass(const QualifiedName: String; const Inputs: array of TIValue): IJitObject;
  end;

function JitLiveObjects: Integer;               // live TJitObjectHandle count, as TchLiveHandles

implementation
type
  TJitObjectHandle = class(TInterfacedObject, IJitObject)
  private
    FRaw: PTorchIValue;
  public
    constructor Create(P: PTorchIValue);        // takes ownership; body is two assignments, cannot raise
    destructor Destroy; override;               // ati_free(FRaw); never raises
    function Raw: PTorchIValue;
    function MethodIV(const Name: String; const Inputs: array of TIValue): TIValue;
    function GetAttr(const Name: String): TIValue;
  end;
```

- `TagName` gives `'Object'`; `Count` raises on an object as on a scalar;
  `IsNone` is unchanged. `AsObject` on a non-object raises the usual
  mismatch; an `ivObject` whose `Obj` is nil cannot come out of `FromC`
  and is treated as a nil tensor is: `ToC` raises
  `ETorchError('ivalue: nil object')`.
- `CreateClass` returns the object, not a `TIValue` (deviation from tch-rs's
  `create_class_is`, stated): `Module::create_class` always produces an
  Object, so the result goes through `FromC` and `AsObject` (which would
  raise on the impossible non-object). Class names are qualified, as the
  measured message shows.
- `[O, 42]` as an `array of TIValue` argument should convert element-wise
  through the `:=` operator like `[T, 42, 1.5]` does (M5). *Measured* in
  local session 6: it does (`add_them([O])`, `Tuple([O, 42])`).

**Ownership on every path** (the heart of the design; each line names who
frees what):
1. **`FromC`, tag 14.** `Result.Tag := ivObject; Result.Obj :=
   TJitObjectHandle.Create(P); Keep := True` — the handle now owns `P`.
   The `finally` becomes `FreeIValues(Cs); if not Keep then ati_free(P)`.
   `Keep` is a local initialised to False. The constructor's body is
   `inherited Create; InterlockedIncrement(LiveObjects); FRaw := P` — the
   assignment last, so that if anything before it raised (FPC then runs
   the destructor) `FRaw` is still nil, the destructor frees nothing and
   the `finally` frees `P` (`Keep` not yet set). Every other tag is
   unchanged and still consumes `P`.
2. **`ToC`, `ivObject`.** `Result := ati_clone(Obj.Raw); TchCheck`. The
   caller owns the clone and frees it like every other `ToC` result (the
   `finally FreeIValues(Cs)` in `ForwardIV`/`MethodIV`/`CreateClass` and
   in the container branch); the handle's own pointer is never handed to
   C for freeing. This is tch-rs's fix for issue 475: passing the same
   object twice, or passing it and letting the Pascal value die, is safe
   (*measured* at the C level: `add_them` twice with the same pointer gives
   `[5, 7, 9]` twice; the Pascal path is test item d below).
3. **Object inside a container, `ToC`.** The child is cloned into its
   slot like any child, the C container copies it (another intrusive_ptr
   bump), `FreeIValues(Cs)` frees the clone. A later child failing in
   `ToC` frees the clones already made (`IValuesToC`'s `except`), and the
   handles are untouched.
4. **Object inside a container, `FromC`.** The `ati_to_*` call writes a
   fresh `IValue` per slot (*measured*); the recursive `FromC` wraps it in
   a handle (step 1). If a later child fails: slots not yet consumed are
   freed by `FreeIValues(Cs)`; the children already converted live in
   `IValuesFromC`'s local result array, which the compiler finalises
   while unwinding, releasing their handles, whose destructors `ati_free`
   their pointers. Nothing leaks, nothing is freed twice. The mock test
   checks it (object followed by a Device).
5. **`MethodIV` / `GetAttr` / `CreateClass`.** Same shape as
   `TJitModule.MethodIV`: `Cs := IValuesToC(Inputs)` (objects cloned);
   `P := ati_object_method_(FRaw, PChar(Name), Cs, n)` /
   `ati_object_getattr_(FRaw, PChar(Name))` /
   `atm_create_class_(FModule, PChar(QualifiedName), Cs, n)`; `TchCheck` inside
   a `try … finally FreeIValues(Cs)`; `Result := TIValue.FromC(P)`. C copies
   `self` (`inputs.push_back(*i)`), so `FRaw` stays valid. `GetAttr` keeps
   tch-rs's nil check (`ETorchError('ivalue: getattr returned nil')`,
   unreachable after `TchCheck` under `PROTECT`, like `FromC`'s own).
6. **Destroy.** `if FRaw <> nil then ati_free(FRaw)`;
   `InterlockedDecrement(LiveObjects)`; never raises, never reads the
   error slot (4.1).
7. **Module lifetime.** An object outlives its `TJitModule` (*measured*,
   above); `IJitObject` holds no reference to the module. Say so in the
   unit header.
8. **Temporaries.** As H1 (4.1): an object produced inside an expression
   stays alive in the routine's temporary slot until the routine returns
   (*measured* in M10: the count is 1 inside the routine after every
   named variable was released, 0 after it returns). Tests therefore
   compare `JitLiveObjects` and `TchLiveHandles` from the calling routine.

**L2 (`TchSys.pas`)**, exactly as `torch_api.h`:

```pascal
function  atm_create_class_(m: PTorchModule; clz_name: PChar; ivalues: PPTorchIValue;
  nivalues: cint): PTorchIValue; cdecl; external LibTch;
function  ati_object_method_(i: PTorchIValue; method_name: PChar; ivalues: PPTorchIValue;
  nivalues: cint): PTorchIValue; cdecl; external LibTch;
function  ati_object_getattr_(i: PTorchIValue; attr_name: PChar): PTorchIValue; cdecl; external LibTch;
function  ati_clone(i: PTorchIValue): PTorchIValue; cdecl; external LibTch;
```

All four are `PROTECT`ed and return nil on error (`TchCheck` after each).
The unit comment's "Not declared" list shrinks to `atm_fuser_cuda_*`.
**Fork patch: none.**

**`tests/jit/jit_test.pas`, group 14 "Objects"** (hand-checkable; `x =
[3, 4, 5]`, `T1 = [1, 2, 3]`, `T2 = [4, 5, 6]`, all kFloat; `Before :=
JitLiveObjects` and `TchLiveHandles` taken in the caller):
- a. `jit6`: `foo6.pt`, `R := M.ForwardIV([x])` → `R.Tag = ivObject`,
  `R.TagName = 'Object'`; `O := R.AsObject`; `O.MethodIV('y',
  []).AsTensor = [6, 8, 10]`; `O.GetAttr('x').AsTensor = [3, 4, 5]`;
  `R.AsTensor` raises `ETorchError` `'ivalue: expected Tensor, got
  Object'`; `M.Forward([x])` raises `ETorchNativeError` "forward did not
  return a tensor" (`atm_forward`'s `isTensor` check; **(H27)**).
- b. Errors (`ETorchNativeError`, message contains): `O.GetAttr('nope')`
  → `does not have an attribute with name 'nope'`; `O.MethodIV('nope',
  [])` → `Couldn't find method: 'nope'`; `O.MethodIV('y', [7])` →
  `Expected at most 1 argument(s)`.
- c. `jit_double_free`: `foo7.pt`, `Inp := M.MethodIV('make_input_object',
  [T1, T2])` → `ivObject`; `M.MethodIV('add_them', [Inp]).AsTensor = [5,
  7, 9]`; **the same call again with the same `Inp`** → `[5, 7, 9]` (the
  regression of issue 475: the handle survives being passed);
  `Inp.AsObject.GetAttr('foo') = [1, 2, 3]`, `'bar' = [4, 5, 6]`.
- d. `CreateClass`: `O := M.CreateClass('__torch__.InputObject', [T1,
  T2])`; `M.MethodIV('add_them', [O]) = [5, 7, 9]` (or
  `[TIValue.FromObject(O)]`); `CreateClass('InputObject', [T1, T2])` →
  `Could not find class with name: 'InputObject' in module.`;
  `CreateClass('__torch__.InputObject', [T1])` → `__init__() is missing
  value for argument 'bar'`; `M.MethodIV('add_them', [5])` → `Expected a
  value of type '__torch__.InputObject`; on `foo6.pt`:
  `CreateClass('__torch__.TorchScriptClass', [x]).MethodIV('y', []) = [6,
  8, 10]` (**(H27)**).
- e. Containers: `W := TIValue.FromC(TIValue.Tuple([O, 42]).ToC)` →
  `W.Count = 2`, `W.Item(1).AsInt = 42`, `W.Item(0).AsObject.GetAttr('foo')
  = [1, 2, 3]`, `W.Item(0).AsObject.Raw <> O.Raw` (fresh `IValue`, same
  object); the same object through `GenericList([O])` (`Item(0)`) and
  `GenericDict(['k', O])` (`Value(0)`) (**(H26)**).
- f. Sharing and lifetime: `A := TIValue.FromObject(O); B := A;
  A.AsObject.Raw = B.AsObject.Raw`; `A := Default(TIValue)`;
  `B.AsObject.GetAttr('foo')` still `[1, 2, 3]`; `M.Free` **before**
  `O.GetAttr('bar')` → still `[4, 5, 6]` (the object outlives the
  module). After the group's routine returns: `JitLiveObjects = Before`,
  `TchLiveHandles` back to its start; the program's last line adds
  `JitLiveObjects = 0` next to `TchLiveHandles = 0`.
- g. Leak loop (**(H28)**): `10^4` × `O.MethodIV('y', [])` on `foo6`'s
  object in a small routine; afterwards `JitLiveObjects` and
  `TchLiveHandles` are unchanged (the C++ side has no counter; the Pascal
  counters plus a flat RSS, as in the 4.1 loop, are the evidence).

**Mock** (`tests/mock/mock_tch.c`, plumbing only): `mock_ivalue` gets
`char *cls` and `char **names` for tag 14; the attributes live in `items`
(`n` of them) and are named `a0`, `a1`, … in constructor order.
`atm_create_class_(m, clz, ivs, n)` records `mock_last_class`, fails when
`clz` contains `missing`, else returns an object with `n` cloned
attributes. `ati_object_getattr_(o, name)`: `expect_tag(o, 14)`, a clone
of the named attribute, unknown name → `set_err("mock: no attribute …")`,
nil. `ati_object_method_(o, name, ivs, n)`: `expect_tag(o, 14)`, records
`mock_last_method`; `name == "fail"` → error, nil; else a tuple of clones
of the attributes followed by clones of the arguments. `ati_clone` →
`iv_clone` (a deep copy in the mock, counted as a live ivalue: the mock
checks plumbing, not identity). `iv_clone` and `ati_free` copy/free `cls`
and `names`; `ati_length` on 14 fails like libtorch. The existing
`atm_forward_`/`atm_method_` (tuple of clones) already carry an object.
`tests/mock/jit_mock_test.pas`, new `TestObjects`:
1. `O := M.CreateClass('__torch__.Foo', [1, 2.5])` → `mock_last_class`,
   `mock_live_ivalues = 1`, `JitLiveObjects = 1`.
2. `O.GetAttr('a1').AsDouble = 2.5`; `GetAttr('zz')` → `ETorchNativeError`;
   `mock_live_ivalues` back to 1 after each.
3. `O.MethodIV('m', ['x'])` → tuple `(1, 2.5, 'x')`, `mock_last_method =
   'm'`; `MethodIV('fail', [O])` raises and leaves `mock_live_ivalues = 1`
   (the cloned input was freed).
4. Sharing: `A := TIValue.FromObject(O); B := A; A := Default(TIValue)`;
   `B.AsObject.Raw = O.Raw`; counts unchanged until all three are
   released (checked from the caller).
5. Round trip `FromC(Tuple([O, 42]).ToC)`: `Item(0).Tag = ivObject`,
   `Item(0).AsObject.GetAttr('a0').AsInt = 1`; `mock_live_ivalues = 2`
   while both objects are alive (the mock's clone is a separate cell), 0
   when the caller has released everything.
6. `M.MethodIV('f', [O])` on a mock module: `Item(0).Tag = ivObject`,
   counts back afterwards.
7. Failing sibling: `FromC(Tuple([O, TIValue.FromDevice(Cpu)]).ToC)`
   raises at the Device child; `mock_live_ivalues` is back to the object's
   own cell only and `JitLiveObjects` unchanged (step 4 of the ownership
   list).
8. The final lines of the program add `JitLiveObjects = 0`.
   *As implemented* (local session 6): a mock object is one cell plus one
   per attribute, so the counts "1" and "2" above are `Cells` (3 for
   `[1, 2.5]`) and `2 * Cells`; item 7 is checked as a delta around the
   statement, because the temporaries of items 5–6 still hold handles (H1).

**Hypotheses (need real libtorch; everything else above is measured):**
- **(H26)** an object round-trips through `GenericList` and as a
  `GenericDict` value as it does through a tuple (measured): fresh tag-14
  `IValue`s that outlive the container.
- **(H27)** `atm_create_class_('__torch__.TorchScriptClass', [x])` on
  `foo6.pt` gives an object whose `y()` is `2x`, and `atm_forward` (tensor
  API) on `foo6.pt` raises "forward did not return a tensor" (the
  `isTensor` check in `torch_api.cpp` is read, not run).
- **(H28)** `10^4` object method calls leave `JitLiveObjects`,
  `TchLiveHandles` and RSS flat.
- **Results (local session 6, 2026-10-12, libtorch 2.13.0 CPU):** H26,
  H27 and H28 hold. Tuple, `GenericList` and `GenericDict` value each give
  a fresh tag-14 pointer (`Raw` differs) whose attributes read back;
  `CreateClass('__torch__.TorchScriptClass', [x]).y() = [6, 8, 10]`;
  `Forward` raises "forward did not return a tensor"; `10^4` method calls:
  live handles 2, objects 1 before and after, RSS 127148 KiB → 127148 KiB.
  The six messages tested (items a, b, d) matched the table.
- Not testable with the vendored assets and therefore not claimed: that a
  mutation through one clone is visible through another (no asset has a
  mutating method, and `Define` cannot declare a TorchScript class). It
  follows from `IValue`'s copy constructor; a later asset may check it.

#### 4.6.10 Layers of milestone 3, from `nn/conv.rs`, `conv_transpose.rs`, `group_norm.rs`, `rnn.rs`, `mod.rs` (`Id`) and `tests/nn_tests.rs`

Design session 7 (local machine, real libtorch 2.13.0 CPU, FPC 3.2.2).
The 4.6.5 deferred list, same pattern as 4.6.5: config record +
`Default*Config`, two constructors, tch-rs's variable names and
registration order so `.ot` files interchange. Everything numeric below
was *measured* against the generated ops (probe under the scratchpad, not
committed); the layer classes themselves are the local implementer's.

**Shape of tch-rs's `ConvConfigND<ND>` in FPC (decision).** tch-rs has one
generic config/layer over `ND = [i64; 1|2|3]`, plus `conv1d/2d/3d` taking
the scalar `ConvConfig` and a `Create::make_array` that repeats the scalar.
Pascal gets *one non-generic* `TConvConfigND` whose per-dimension fields are
dynamic arrays (`Nd = Length`), *one* base class `TConvND` that switches on
`FNd` in `Forward`, and `TConv1D/2D/3D` as thin subclasses carrying the
scalar-config constructors. An FPC generic record was rejected without
measuring: a `specialize TConvConfigND<TInt3>` gives three unrelated types,
each still needing its own `Default` function, and the base class would
still have to switch on the dimension (Pascal has no `AsRef<[i64]>` and no
per-type trait impl), so the generic buys nothing. What the chosen shape
relies on was *measured* (M-A): a base constructor
`Create(P, In, Out, const KSizes: array of Int64; const Cfg: TConvConfigND)`
plus two `overload` constructors on the subclass resolve without ambiguity
for `TConv2D.Create(P, 1, 1, 3)`, `TConv2D.Create(P, 1, 1, 3, Cfg)` **and**
the inherited `TConv2D.Create(P, 1, 1, [3, 3], CfgND)`; the subclass
constructor calls `inherited Create(P, In, Out, [K, K], ConvConfigToND(Cfg, 2))`.

```pascal
  TConvConfigND = record                     // ConvConfigND<[i64; N]>, N = Length(Stride)
    Stride, Padding, Dilation: TInt64Array;  // all three of length Nd
    Groups: Int64; Bias: Boolean;
    WsInit, BsInit: TInit; PaddingMode: TPaddingMode;
  end;
  function DefaultConvConfigND(Nd: Integer): TConvConfigND;          // [1..], [0..], [1..], 1, True, KaimingUniform, Constant(0), pmZeros
  function ConvConfigToND(const Cfg: TConvConfig; Nd: Integer): TConvConfigND;  // Create::conv's make_array
  function PadWithMode(const X: TTensor; Mode: TPaddingMode;
    const Pad: array of Int64): TTensor;     // PaddingMode::pad: X.pad(Pad, Name, NoDouble)

  TConvND = class(TModule)                   // Conv<ND>; nn::conv
  private
    FNd: Integer; FCfg: TConvConfigND;
    FPadTwice, FZeroPad: TInt64Array;        // reversed_padding_repeated_twice; [0, ..]
  public
    Weight, Bias: TTensor;
    constructor Create(const P: TVarPath; InDim, OutDim: Int64;
      const KSizes: array of Int64; const Cfg: TConvConfigND); overload;
    function Forward(const X: TTensor; Train: Boolean = False): TTensor; override;
    property Nd: Integer read FNd;
  end;
  TConv1D = class(TConvND)                   // conv1d; TConv2D and TConv3D identical
  public
    constructor Create(const P: TVarPath; InDim, OutDim, KSize: Int64); overload;
    constructor Create(const P: TVarPath; InDim, OutDim, KSize: Int64; const Cfg: TConvConfig); overload;
  end;

  TConvTransposeConfig = record              // ConvTransposeConfig (scalar)
    Stride, Padding, OutputPadding, Dilation, Groups: Int64; Bias: Boolean;
    WsInit, BsInit: TInit;                   // no PaddingMode, as tch-rs
  end;
  TConvTransposeConfigND = record
    Stride, Padding, OutputPadding, Dilation: TInt64Array;
    Groups: Int64; Bias: Boolean; WsInit, BsInit: TInit;
  end;
  function DefaultConvTransposeConfig: TConvTransposeConfig;         // 1, 0, 0, 1, 1, True, KaimingUniform, Constant(0)
  function DefaultConvTransposeConfigND(Nd: Integer): TConvTransposeConfigND;
  function ConvTransposeConfigToND(const Cfg: TConvTransposeConfig; Nd: Integer): TConvTransposeConfigND;
  TConvTransposeND = class(TModule)          // ConvTransposeND<ND>
  private FNd: Integer; FCfg: TConvTransposeConfigND;
  public
    Weight, Bias: TTensor;
    constructor Create(const P: TVarPath; InDim, OutDim: Int64;
      const KSizes: array of Int64; const Cfg: TConvTransposeConfigND); overload;
    function Forward(const X: TTensor; Train: Boolean = False): TTensor; override;
  end;
  TConvTranspose1D = class(TConvTransposeND) { the two scalar constructors } end;  // and 2D, 3D

  TGroupNormConfig = record CudnnEnabled: Boolean; Eps: Double; Affine: Boolean; WsInit, BsInit: TInit; end;
  function DefaultGroupNormConfig: TGroupNormConfig;                 // True, 1e-5, True, Constant(1), Constant(0)
  TGroupNorm = class(TModule)
  private FCfg: TGroupNormConfig; FNumGroups, FNumChannels: Int64;
  public
    Weight, Bias: TTensor;                   // IsNil when Affine is False
    constructor Create(const P: TVarPath; NumGroups, NumChannels: Int64); overload;
    constructor Create(const P: TVarPath; NumGroups, NumChannels: Int64; const Cfg: TGroupNormConfig); overload;
    function Forward(const X: TTensor; Train: Boolean = False): TTensor; override;
    property NumGroups: Int64 read FNumGroups;  property NumChannels: Int64 read FNumChannels;
  end;

  TRNNConfig = record                        // RNNConfig, shared by LSTM and GRU
    HasBiases: Boolean; NumLayers: Int64; Dropout: Double; Train: Boolean;
    Bidirectional, BatchFirst: Boolean;
    WIhInit, WHhInit, BIhInit, BHhInit: TInit;   // b_*_init: Option<Init> in tch-rs, see deviations
  end;
  function DefaultRNNConfig: TRNNConfig;     // True, 1, 0, True, False, True, KaimingUniform x2, Constant(0) x2
  TRNNState = record H, C: TTensor; end;     // LSTMState((h, c)); GRUState(h) leaves C nil
  TRNN = class abstract                      // trait RNN; not a TModule (tch-rs's LSTM/GRU implement no Module)
  private
    FFlatWeights: TTensorArray; FHiddenDim: Int64; FCfg: TRNNConfig;
  protected
    constructor Create(const P: TVarPath; InDim, AHiddenDim, GateMul: Int64; const Cfg: TRNNConfig);  // rnn_weights
    function BatchDimIndex: Integer; inline; // 0 when BatchFirst else 1
  public
    function ZeroState(BatchDim: Int64): TRNNState; virtual; abstract;
    function Step(const Input: TTensor; const State: TRNNState): TRNNState;       // [batch, features]
    function Seq(const Input: TTensor; out State: TRNNState): TTensor;             // [batch, seq, features] when BatchFirst
    function SeqInit(const Input: TTensor; const InState: TRNNState; out OutState: TRNNState): TTensor; virtual; abstract;
    property FlatWeights: TTensorArray read FFlatWeights;
    property HiddenDim: Int64 read FHiddenDim;
    property Config: TRNNConfig read FCfg;
  end;
  TLSTM = class(TRNN)
  public
    constructor Create(const P: TVarPath; InDim, AHiddenDim: Int64); overload;
    constructor Create(const P: TVarPath; InDim, AHiddenDim: Int64; const Cfg: TRNNConfig); overload;
    function ZeroState(BatchDim: Int64): TRNNState; override;
    function SeqInit(const Input: TTensor; const InState: TRNNState; out OutState: TRNNState): TTensor; override;
  end;
  TGRU = class(TRNN)  { same four members } end;

  TIdentity = class(TModule)                 // nn::Id
    function Forward(const X: TTensor; Train: Boolean = False): TTensor; override;  // Result := X
  end;
```

Per layer — variables (name `[shape]`, default init, registration order),
config defaults, the exact op `Forward` calls:

| layer (tch-rs file) | variables, in registration order | config defaults | `Forward` |
|---|---|---|---|
| `TConvND` / `TConv1D` / `TConv3D` (`conv.rs`; `TConv2D` moves onto the same base, API unchanged) | `bias [Out]` `BsInit` when `Bias`; `weight [Out, In div Groups, K1, .., KNd]` `WsInit` | `DefaultConvConfig` as 4.6.5; `DefaultConvConfigND(Nd)` the same values repeated `Nd` times | `pmZeros`: `X.conv{Nd}d(Weight, Bias, Stride, Padding, Dilation, Groups)`; other modes: `PadWithMode(X, Mode, FPadTwice).conv{Nd}d(Weight, Bias, Stride, FZeroPad, Dilation, Groups)`; `case FNd of 1: conv1d; 2: conv2d; 3: conv3d`. The constructor raises `ETorchError` unless `Length(KSizes)` is 1..3 and `Stride`, `Padding`, `Dilation` all have that length |
| `TConvTransposeND` / `1D` / `2D` / `3D` (`conv_transpose.rs`) | `bias [Out]` `BsInit` when `Bias`; `weight [In, Out div Groups, K1, .., KNd]` `WsInit` (note `In` first) | `Stride` 1, `Padding` 0, `OutputPadding` 0, `Dilation` 1, `Groups` 1, `Bias` True, `WsInit` KaimingUniform, `BsInit` Constant(0) | `X.conv_transpose{Nd}d(Weight, Bias, Stride, Padding, OutputPadding, Groups, Dilation)` — the generated argument order (groups before dilation) is the C one; no padding modes |
| `TGroupNorm` (`group_norm.rs`) | `weight [NumChannels]` `WsInit`, `bias [NumChannels]` `BsInit`, both only when `Affine` | `CudnnEnabled` True, `Eps` 1e-5, `Affine` True, `WsInit` Constant(1), `BsInit` Constant(0) | `X.group_norm(NumGroups, Weight, Bias, Eps, CudnnEnabled)` |
| `TLSTM` / `TGRU` (`rnn.rs`, `rnn_weights`) | for `L` in `0..NumLayers-1`, for `D` in `0..NumDirections-1` (`Sfx` = `'_reverse'` when `D = 1`): `weight_ih_l{L}{Sfx} [GateDim, InL]` `WIhInit`, `weight_hh_l{L}{Sfx} [GateDim, Hidden]` `WHhInit`, then when `HasBiases` `bias_ih_l{L}{Sfx} [GateDim]` `BIhInit`, `bias_hh_l{L}{Sfx} [GateDim]` `BHhInit`; `InL = InDim` for `L = 0`, else `Hidden * NumDirections`; `GateDim = 4 * Hidden` (LSTM, gates i, f, g, o) or `3 * Hidden` (GRU, gates r, z, n); `FFlatWeights` is the same list in the same order (what `lstm`/`gru` take as `params`) | `HasBiases` True, `NumLayers` 1, `Dropout` 0, `Train` True, `Bidirectional` False, `BatchFirst` True, `WIhInit`/`WHhInit` KaimingUniform, `BIhInit`/`BHhInit` Constant(0) | `SeqInit`: LSTM `Input.lstm([InState.H, InState.C], FFlatWeights, HasBiases, NumLayers, Dropout, Train, Bidirectional, BatchFirst, Result, OutState.H, OutState.C)`; GRU `Input.gru(InState.H, FFlatWeights, ..., Result, OutState.H)`. `ZeroState(B)`: `TTensor.zeros([NumLayers * NumDirections, B, Hidden], FFlatWeights[0].Kind, FFlatWeights[0].Device)` into `H` (LSTM: and the same handle into `C`, as tch-rs's `shallow_clone`). `Step(X, S)`: `SeqInit(X.unsqueeze(1 - BatchDimIndex), S, Result)`, output discarded. `Seq(X, S)`: `SeqInit(X, ZeroState(X.Shape[BatchDimIndex]), S)` |
| `TIdentity` (`mod.rs` `Id`) | none | — | returns `X` (record copy, same handle) |

Decisions and rationale:
- **One base per family, subclasses only for the scalar constructors.**
  `TConv2D`'s public API (two constructors, `Weight`, `Bias`, `Forward`)
  does not change; its body moves into `TConvND` and `TConv2D.Create(P, In,
  Out, K, Cfg)` becomes `inherited Create(P, In, Out, [K, K],
  ConvConfigToND(Cfg, 2))`. The group-8 test numbers of 4.6.7 were
  re-run through this shape and are unchanged (*measured*). A user who
  wants per-dimension settings builds a `TConvConfigND` and uses either
  `TConvND.Create` or the inherited constructor on `TConv2D`.
- **`PadWithMode` is public** (tch-rs's `PaddingMode::pad`) so tch-rs's
  `pad` test can be ported, and `TConvND.Forward` uses it. `pmZeros` maps
  to `'constant'` (tch-rs `PaddingMode::to_string`; the current private
  `ModeNames` has `'zeros'` in that slot, which `Forward` never reaches —
  replace it).
- **Reversed padding repeated twice follows PyTorch, not tch-rs
  (deviation).** For `Padding = [Ph, Pw]`, `pad` wants `[Pw, Pw, Ph, Ph]`
  (last dimension's left/right first); tch-rs's `conv()` builds
  `[Pw, Ph, Pw, Ph]` (reverse, then reverse again), which is only correct
  when all paddings are equal — i.e. for the scalar config everyone uses.
  *Measured*: `TConv2D` ND with `Padding [1, 0]`, reflect, `KSizes [3, 1]`,
  weights 1 on `[[1, 2], [3, 4]]` gives `[[7, 10], [5, 8]]` with PyTorch's
  order and a `[1, 1, 1, 3]` tensor `[5, 8, 5]` with tch-rs's. So
  `FPadTwice[2*I] = FPadTwice[2*I + 1] = Padding[Nd - 1 - I]`.
- **`TRNNState` is one record for both layers (deviation).** tch-rs has
  `LSTMState((h, c))` and `GRUState(h)` as the `RNN` trait's associated
  type; without generics on associated types, a shared abstract `TRNN`
  needs one state type, so `TGRU` leaves `C` nil (`C.IsNil`). The
  accessors `h()`/`c()`/`value()` are just the fields.
- **`Seq`/`SeqInit` return the output and hand the state back through
  `out`** (tch-rs returns the tuple `(output, state)`); `Step` returns the
  new state as its result, as tch-rs. *Measured* (M-B): FPC finalises an
  `out` record of `TTensor`s on entry (H8 again), so
  `L.SeqInit(X, S, S)` nils `S.H` before the body runs and the op raises
  `ETorchError: undefined tensor passed as hx to gru` — a clean failure,
  documented: `InState` and `OutState` must be distinct variables.
  *Measured* (M-B2): `S := L.Step(X, S)` is safe; the function result goes
  through a temporary, so the body still sees the old `S`.
- **`BatchFirst = False` works (deviation).** tch-rs's `seq` takes
  `size()[0]` as the batch and `step` does `unsqueeze(1)` regardless of
  `batch_first`, so with `batch_first: false` both pass the wrong batch
  size to `zero_state`. Pascal uses `BatchDimIndex` (0 or 1) in both
  places; the cost is two `if`s. *Measured*: `gru` with `batch_first`
  False on `[3, 5, 2]` returns `[3, 5, 4]`, `h [1, 5, 4]`.
- **`BIhInit`/`BHhInit` are plain `TInit`** (tch-rs: `Option<Init>`,
  `unwrap()`ed when `has_biases` — the `None` case only panics). Plain
  fields drop the panic path; `HasBiases` False ignores them.
- **`ZeroState` reads kind and device from `FFlatWeights[0]` at call
  time** (tch-rs stores `vs.device()` at creation and reads the kind at
  call time); so `SetKind`/`SetDevice` on the store after construction are
  honoured. Minor deviation, strictly more permissive.
- **cuDNN weight flattening** was left to the CUDA/ROCm milestone here and
  is implemented by 4.8 (H30): `TRNN`'s constructor calls
  `_cudnn_rnn_flatten_weight` when the store is on CUDA with cuDNN, with
  weight stride 2 without biases (PyTorch; tch-rs passes 4). On CPU and on
  ROCm (no cuDNN) the call is not made.
- **`TRNN` is not a `TModule`**, as in tch-rs; the `Train` the op receives
  is `Config.Train` (dropout between layers only), not a `Forward`
  argument. A user who wants an RNN inside a `TSequential` wraps `Seq` in
  a `TFuncModule` method.
- **`Id` is `TIdentity`** (name only; `Id` reads as "identifier" in
  Pascal). It is the degenerate layer that lets a `TSequential` slot be
  filled conditionally.

Not measurable with CPU libtorch here, kept as hypotheses:
- **(H29)** `.ot` interchange with files written by tch-rs's `lstm`/`gru`/
  `conv_transpose2d`/`group_norm` (names and shapes match by construction;
  needs a Rust toolchain to produce the file).
- **(H30)** the cuDNN flatten call on a CUDA/ROCm store (item 3) — the
  condition, the call and its test are now in 4.8 ("H13 and H30 on the
  GPU", GPU test item 8).

**Implemented in local session 7 (2026-10-12)** in `src/TchNN.pas`;
group 11 of `tests/nn/nn_test.pas` (44 checks) passes against libtorch
2.13.0 CPU with every number above. Two compiler facts the design missed,
both *measured*: a constructor parameter named `HiddenDim` is a duplicate
identifier next to the `HiddenDim` property (FPC checks parameter names
against the class's members), so the RNN constructors take `AHiddenDim`;
and the protected `TRNN.Create` draws warning 3018 ("Constructor should
be public"), silenced with `{$push}{$warn 3018 off}` as
`TJitModule.CreateRaw`. The optional mock check (item 7 of the local
list) was skipped: names and order are checked against libtorch.

Test list, group 11 "Milestone 3 layers" in `tests/nn/nn_test.pas`
(numbers *measured* here through the generated ops unless marked tch-rs):
  1. `PadWithMode` (tch-rs `pad`): `[1, 2, 3]` zeros `[1, 1]` →
     `[0, 1, 2, 3, 0]`; `[1, 2, 3, 4].view([1, 2, 2])` reflect
     `[1, 1, 1, 1]` → `[4, 3, 4, 3, 2, 1, 2, 1, 4, 3, 4, 3, 2, 1, 2, 1]`.
  2. `TConv1D(1, 1, 3)`, `Padding 1`, `Bias False`, weights 1, on
     `[1, 2, 3]` (`[1, 1, 3]`): zeros `[3, 6, 5]`, reflect `[5, 6, 7]`,
     replicate `[4, 6, 8]`, circular `[6, 6, 6]`. `TConv1D(4, 6, 3)`
     registers `bias [6]` then `weight [6, 4, 3]`.
  3. `TConv2D`: group 8's four results unchanged; ND asymmetric padding
     `[1, 0]` reflect, `KSizes [3, 1]`, weights 1 → `[[7, 10], [5, 8]]`
     (shape `[1, 1, 2, 2]`); `DefaultConvConfigND(2)` equals
     `ConvConfigToND(DefaultConvConfig, 2)` field by field; `KSizes` of
     length 4, or `Stride` of the wrong length, raise `ETorchError`.
  4. `TConv3D(1, 1, 2)`, weights 1, no bias, on `1..8` as `[1, 1, 2, 2, 2]`
     → `[36]`, shape `[1, 1, 1, 1, 1]`; `TConvND(2, 3, [3, 3, 3])` with
     `Padding [1, 0, 0]` maps `[1, 2, 4, 4, 4] → [1, 3, 4, 2, 2]`, weight
     `[3, 2, 3, 3, 3]`.
  5. `TConvTranspose2D(1, 1, 2)`, weights 1, `Bias False`, on
     `[[1, 2], [3, 4]]`: stride 1 → `[1, 1, 3, 3]`
     `[1, 3, 2, 4, 10, 6, 3, 7, 4]`; stride 2 → `[1, 1, 4, 4]`
     `[1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4]`.
     `TConvTranspose1D(1, 1, 2)` on `[1, 2, 3]` → `[1, 3, 5, 3]`; stride 2,
     `OutputPadding 1` → `[1, 1, 7]` `[1, 1, 2, 2, 3, 3, 0]`.
     `TConvTranspose3D(2, 3, 3)`: weight `[2, 3, 3, 3, 3]` (registered after
     `bias [3]`), `[1, 2, 4, 4, 4] → [1, 3, 6, 6, 6]`.
  6. `TGroupNorm(2, 4)` on `[1, 2, 3, 4].view([1, 4, 1])` →
     `[-1, 1, -1, 1]` within 1e-4 (exactly ±0.99998 with `Eps` 1e-5);
     registers `weight [4]`, `bias [4]`; `Affine False` registers nothing
     and still normalises. tch-rs `group_norm_test`: `TGroupNorm(2, 10)`
     on `randn [1, 10, 7]` keeps the shape.
  7. `TGRU` and `TLSTM`, tch-rs's five configs (default; `HasBiases`
     False; `Bidirectional`; `NumLayers 2`; both), `InDim 2`,
     `Hidden 4`, batch 5, seq 3: `Step` on `randn [5, 2]` from `ZeroState(5)`
     gives `H` (and `C`) `[LayerDim, 5, 4]`; `Seq` on `randn [5, 3, 2]`
     gives `[5, 3, 4 * NumDirections]`; variable count
     `4 * NumLayers * NumDirections` (`2 *` without biases); for the
     2-layer bidirectional case the names are `weight_ih_l0`,
     `weight_hh_l0`, `bias_ih_l0`, `bias_hh_l0`, `weight_ih_l0_reverse`, …,
     `weight_ih_l1 [12, 8]` (GRU) … `bias_hh_l1_reverse`, in this order.
     `BatchFirst False`: `Seq` on `[3, 5, 2]` → `[3, 5, 4]`.
  8. Hand numbers: `TGRU(1, 1)`, all weights 0, `bias_ih_l0 := [0, 0, 1]`
     (under `NoGradGuard`), `Step` on `zeros [1, 1]` → `H = 0.380797`
     (`= tanh(1) / 2`: r = z = σ(0) = ½, n = tanh(1), h' = (1 − z)·n).
     `TLSTM(1, 1)`, `bias_ih_l0 := [0, 0, 1, 0]` → `C = 0.380797`
     (`= ½·tanh(1)`: i = ½, g = tanh(1)), `H = 0.181700`
     (`= ½·tanh(0.380797)`). Both within 1e-5.
  9. `L.SeqInit(X, S, S)` raises `ETorchError` (message names `hx`);
     `S := L.Step(X, S)` works.
  10. `TIdentity.Forward(X).Raw = X.Raw`; a `TSequential` of
      `TIdentity` + `AddFn(@Relu)` behaves like the ReLU alone.
  11. `TchLiveHandles` back to its starting value after freeing the layers
      and stores.

### 4.7 Grad mode and global settings

```pascal
function  SetGradEnabled(Enabled: Boolean): Boolean;   // returns previous; thread-local in libtorch
function  NoGradGuard: IInterface;                     // restores the previous mode when released
procedure ManualSeed(Seed: Int64);
procedure SetNumThreads(N: Integer); function GetNumThreads: Integer;
function  CudaIsAvailable: Boolean; function CudaDeviceCount: Integer;
```

Usage: `var G: IInterface; ... G := NoGradGuard; Loss := Model.Forward(X);`.
**(H11)** FPC releases interface-typed locals at routine exit (not at block
exit), so the guard covers the rest of the routine; a `try..finally` with
`SetGradEnabled` is the alternative when a narrower scope matters. Autocast
follows the same pattern later.
**(H11) resolved** (step 7): a guard held in a local is released after the
last statement of the routine (not at the end of an inner `begin..end`), and
during unwinding when an exception leaves the routine. **Pitfall:**
calling `NoGradGuard;` without assigning the result releases the guard
immediately, before the next statement, so it has no effect. Document it
in the API comment.

### 4.8 GPU devices: CUDA and ROCm (decision, design session 8)

Scope: run the existing Pascal API on one GPU through a CUDA or ROCm (HIP)
build of libtorch 2.13.0, with the same `TchSys`/`Tch`/`TchNN`/`TchJit`
sources. The first target is ROCm on the local AMD Radeon RX 9070
(`gfx1201`), because that is the hardware at hand; CUDA differs only in
which libtorch archive is unpacked. Everything marked **(H31)**–**(H36)**
needed the ROCm libtorch; all are answered in "Results" at the end of
this section (local session 8), which supersedes the body where they differ.

**Inspected locally (design session 8, 2026-10-11):**
- `rocm-smi --showproductname`: GPU[0] RX 9070, GFX version `gfx1201`;
  GPU[1] the Raphael iGPU, `gfx1036`. `/sys/class/kfd/kfd/topology/nodes/`
  shows both (`gfx_target_version` 120001 and 100306), so the kernel side
  (amdgpu + KFD, kernel 7.0.0) is already working.
- Installed ROCm user space: only `rocm-smi 7.1.1` and `librocm_smi64`
  (Ubuntu packages). No `/opt/rocm`, no `libamdhip64`, no MIOpen, no
  `rocminfo`/`hipconfig`. The user is not in the `render`/`video` groups,
  but `/dev/kfd` and `/dev/dri/renderD*` carry a seat ACL and are already
  read-write for the user from the desktop login.
- `libtorch/include/ATen/Context.h` lines 638–650 (the CPU archive has the
  same headers as every variant): *"HIP masquerades to be CUDA (e.g., when
  you say CUDA, on a HIP build of ATen, this actually means HIP)"*. So on a
  ROCm libtorch `at::kCUDA`, `device_of_int(d >= 0)`, `torch::cuda::*`,
  `atc_*`, `atm_fuser_cuda_*` and `TDevice.Cuda(N)` all address the AMD
  GPU unchanged; "CUDA" in the API below means "the GPU backend".
- `libtorch/share/cmake/Caffe2/Caffe2Config.cmake`: the CPU archive has
  `if (OFF) include(public/LoadHIP.cmake)`. In a ROCm archive that is
  `if (ON)` and `LoadHIP.cmake` looks for the ROCm SDK at `$ROCM_PATH` or
  `/opt/rocm` (`find_package(hip ...)`), i.e. **`find_package(Torch)` on a
  ROCm libtorch needs the ROCm SDK's cmake files**, not just the runtime.
- Measured: libtch builds and links **without `find_package(Torch)`**
  (scratchpad `manual-link/CMakeLists.txt`, 28 s): include dirs
  `<dir>/include` and `<dir>/include/torch/csrc/api/include`,
  `-D_GLIBCXX_USE_CXX11_ABI=1`, link `libtorch.so libtorch_cpu.so
  libc10.so`, `BUILD_RPATH <dir>/lib`. The result `NEEDED`s only
  `libtorch_cpu.so` and `libc10.so`: the linker's `--as-needed` dropped the
  umbrella `libtorch.so`. That matters for GPU builds: the backend lives in
  `libtorch_cuda.so` / `libtorch_hip.so`, pulled in only by `libtorch.so`,
  so the umbrella must be linked with `-Wl,--no-as-needed` (TorchConfig.cmake
  does exactly that for `find_package` users).

#### Build and link

- **Directory layout.** One libtorch per variant next to the CPU one:
  `libtorch/` (CPU, as today), `libtorch-rocm/`, `libtorch-cuda/`
  (unpacked archives, git-ignored like `libtorch/`). One libtch build
  directory per variant: `build/libtch` (CPU), `build/libtch-rocm`,
  `build/libtch-cuda`. Nothing else is duplicated: `libtch.so` has the same
  exported symbols in every variant and the Pascal units never change.
- **`csrc/libtch/CMakeLists.txt` gains a manual mode** (fork change,
  documented in the file header):

  ```cmake
  option(TCH_LIBTORCH_DIR "libtorch root; when set, link it directly instead of find_package(Torch)" "")
  if(TCH_LIBTORCH_DIR)
    add_library(tch SHARED torch_api.cpp torch_api_generated.cpp)
    target_include_directories(tch PRIVATE ${TCH_LIBTORCH_DIR}/include
      ${TCH_LIBTORCH_DIR}/include/torch/csrc/api/include)
    target_compile_definitions(tch PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
    target_link_libraries(tch -Wl,--no-as-needed ${TCH_LIBTORCH_DIR}/lib/libtorch.so
      -Wl,--as-needed ${TCH_LIBTORCH_DIR}/lib/libtorch_cpu.so ${TCH_LIBTORCH_DIR}/lib/libc10.so)
    set_target_properties(tch PROPERTIES BUILD_RPATH ${TCH_LIBTORCH_DIR}/lib)
  else()
    find_package(Torch REQUIRED)         # today's path, CPU archive
    ...
  endif()
  ```

  Default (`find_package`) stays for the CPU archive, so the existing
  commands in HANDOFF keep working. GPU variants use
  `cmake -S csrc/libtch -B build/libtch-rocm -DTCH_LIBTORCH_DIR=$PWD/libtorch-rocm
  -DCMAKE_BUILD_TYPE=Release`, which needs no ROCm SDK on the build
  machine. **(H31)** this manual link produces a working GPU libtch: the
  only compile-time difference between variants is header macros
  (`USE_ROCM`/`USE_CUDA` are not needed by `torch_api.cpp`, which uses only
  `torch::cuda::*` and `at::globalContext()` through hooks resolved at run
  time). If the ROCm archive's headers do need a define for a correct ABI
  (e.g. `__HIP_PLATFORM_AMD__`), the test program of this section is what
  fails, and the define is added under `if(TCH_LIBTORCH_DIR MATCHES rocm)`.
- **Runtime libraries.** The ROCm `libtorch_hip.so` `NEEDED`s
  `libamdhip64.so`, `libMIOpen.so`, `librocblas.so`, `libhipblaslt.so`,
  `librccl.so`, ... **(H32)** the `libtorch-cxx11-abi-shared-with-deps`
  ROCm archive (actual name: `libtorch-shared-with-deps-2.13.0+rocm7.1.zip`,
  no `cxx11-abi` part; it is the cxx11 ABI) bundles them in `lib/` (the ROCm *wheels* do; verify with
  `ls libtorch-rocm/lib | grep -iE 'hip|miopen|rocblas'` and `ldd
  libtorch-rocm/lib/libtorch_hip.so | grep 'not found'`). If it does not,
  the user installs AMD's ROCm runtime packages of the matching version
  (`rocm-hip-runtime`, `miopen-hip`, `rocblas`, `hipblaslt`, `rccl`,
  `hipfft`, `hipsparse`, `rocsolver`, `hiprand`/`rocrand`) — the archive's
  ROCm version is in its name (`+rocmX.Y`), and nothing else about the
  design changes. Either way the kernel driver is already there.
  **(H32) resolved** (local session 5, `2.13.0+rocm7.1`): `ldd` finds
  everything; `libamdhip64`, `libMIOpen`, `librocblas`, `libhipblaslt`,
  `librccl`, `libhsa-runtime64` are bundled in `lib/`. Do not point a
  global `LD_LIBRARY_PATH` at it: system tools then load its bundled
  `libtinfo.so.6`.
  **Amended** in local session 8: the zip carries no symlinks, and eight
  must be added before anything loads (see "Results").
- **Pascal link line is unchanged**: `external 'tch'`,
  `-Fl<build dir> -k-rpath=$PWD/<build dir>`; `libtch.so`'s own RUNPATH
  (set by cmake) finds `libtorch*.so` in the variant's `lib/`. Switching
  variant = recompiling the test with another `-Fl`/`-k-rpath`, or
  `LD_LIBRARY_PATH=build/libtch-rocm` in front of a CPU-linked binary (same
  ABI, same symbols — that is how `build/libtch-unpatched` is swapped in
  today).
- **Device visibility.** The iGPU (`gfx1036`) is not a supported target
  and would appear as `cuda:1`; run every GPU test with
  `HIP_VISIBLE_DEVICES=0` (CUDA: `CUDA_VISIBLE_DEVICES=0`) so
  `CudaDeviceCount = 1` and the numbers below are deterministic. **(H33)**
  the libtorch 2.13 ROCm archive contains `gfx1201` code objects:
  hand-check `strings -n 7 libtorch-rocm/lib/libtorch_hip.so | grep -oE
  'gfx[0-9a-f]+' | sort -u` (the offload bundle ids are plain text). If
  `gfx1201` is absent, this GPU needs a newer archive; no
  `HSA_OVERRIDE_GFX_VERSION` tricks (that is a different ISA).
  **(H33) resolved** (local session 5): the `2.13.0+rocm7.1` archive has
  gfx1201 and gfx1200, gfx11xx, gfx1030, gfx9xx; not gfx1036.
- **C wrapper and Pascal layer: no semantic change.** `device_of_int`,
  `at_device` and all `atc_*` already exist in `torch_api.{h,cpp}`
  (tch-rs's); only L2 declarations and L3 wrappers are added below.

#### Pascal API surface (`Tch.pas`, `TchSys.pas`, `TchJit.pas`)

Already there: `TDevice` (4.3, `ToC` −1/≥0/−2/−3, `Cuda(N)`,
`CudaIfAvailable`, `IsCuda`), `TTensor.Device`, generated `to_device`,
`TVarStore.SetDevice` (4.6.2), load on the store's device (4.6.3),
`TJitModule.ToDevice` (4.6.8), `CudaIsAvailable`, `CudaDeviceCount` (4.7),
L2 `atc_cuda_device_count`, `atc_cuda_is_available`.

Added, following tch-rs's `Cuda` (`src/wrappers/device.rs`) as a Pascal
namespace-free set of functions in `Tch.pas` (prefix `Cuda`, matching the
two that exist):

```pascal
// TchSys.pas (L2), exactly torch_api.h
function  atc_cudnn_is_available: cint; cdecl; external LibTch;      // -1 on error
procedure atc_manual_seed(seed: QWord); cdecl; external LibTch;
procedure atc_manual_seed_all(seed: QWord); cdecl; external LibTch;
procedure atc_synchronize(device_index: Int64); cdecl; external LibTch;
function  atc_user_enabled_cudnn: cint; cdecl; external LibTch;      // -1 on error
procedure atc_set_user_enabled_cudnn(b: cint); cdecl; external LibTch;
procedure atc_set_benchmark_cudnn(b: cint); cdecl; external LibTch;
function  at_context_has_cuda: Boolean; cdecl; external LibTch;      // C bool = 1 byte
function  at_context_has_cudart: Boolean; cdecl; external LibTch;    // added in local session 8 (H35)
function  at_context_has_hip: Boolean; cdecl; external LibTch;
function  at_context_has_cudnn: Boolean; cdecl; external LibTch;
function  at_context_version_cudart: Int64; cdecl; external LibTch;
function  at_context_version_cudnn: Int64; cdecl; external LibTch;
procedure atm_fuser_cuda_set_enabled(b: Boolean); cdecl; external LibTch;
function  atm_fuser_cuda_is_enabled: Boolean; cdecl; external LibTch;

// Tch.pas (L3), 4.7 block grows to:
function  CudaIsAvailable: Boolean; function CudaDeviceCount: Integer;   // existing
function  CudnnIsAvailable: Boolean;                 // atc_cudnn_is_available > 0 (False on ROCm: H34 false)
procedure CudaManualSeed(Seed: QWord);               // current device
procedure CudaManualSeedAll(Seed: QWord);            // every device
procedure CudaSynchronize(DeviceIndex: Int64 = -1); // -1 = current device, as torch::cuda::synchronize
function  CudnnUserEnabled: Boolean; procedure SetCudnnUserEnabled(B: Boolean);
procedure SetCudnnBenchmark(B: Boolean);
type TGpuBackend = (gbNone, gbCuda, gbRocm);
function  GpuBackend: TGpuBackend;                   // see below
function  GpuRuntimeVersion: Int64;                  // at_context_version_cudart (HIP runtime on ROCm)
function  CudnnVersion: Int64;                       // at_context_version_cudnn
```

- Every L3 function calls `TchCheck` after the C call (the `atc_*` getters
  return −1 inside `PROTECT`, which `TchCheck` turns into
  `ETorchNativeError`). On a CPU build (*measured* in local session 8;
  the design expected more to raise) only `CudaSynchronize` ("No CUDA GPUs
  are available"), `GpuRuntimeVersion` and `CudnnVersion` raise; the others
  return what a CPU build has (`CudaIsAvailable` False, `CudaDeviceCount` 0,
  `GpuBackend = gbNone`, `CudnnIsAvailable` False, `CudnnUserEnabled`
  True, seeds and the cuDNN switches accepted). `tests/api` checks exactly
  this; the GPU test still checks `CudaIsAvailable` first.
- **`GpuBackend`** (as implemented after H35 was measured): `gbRocm` when
  `at_context_has_hip`, or when `at_context_has_cuda` and not
  `at_context_has_cudart`; `gbCuda` when CUDA and CUDART are reported;
  else `gbNone`. **(H35)** was "on a ROCm libtorch `at_context_has_hip` is
  True"; *measured* it is False, with `has_cuda` True and `has_cudart`
  False, so the designed rule "gbRocm when has_hip" would have reported
  CUDA. The CUDA side of the rule is not measurable here. `ManualSeed` (4.7,
  `at_manual_seed` = `torch::manual_seed`) already seeds every CUDA device
  too in libtorch; `CudaManualSeedAll` is kept for parity with tch-rs.
- `TDevice`: add `class function CudaIfAvailable(AIndex: Integer): TDevice`
  overload? No — tch-rs has only `cuda_if_available()`; keep as is. Add
  `function TDevice.ToString: String` (`'cpu'`, `'cuda:0'`, `'mps'`,
  `'vulkan'`) for test output; it is not in tch-rs but costs nothing.
- **`TchJit`**: `JitFuserCudaEnabled: Boolean` / `SetJitFuserCudaEnabled`
  next to the profiling/fuser switches of 4.6.8; the unit comment's "Not
  declared" list becomes empty. Designed as plain flag accessors on a CPU
  build; *measured*, enabling raises on every libtorch 2.13 build (nvfuser
  removed), see "Results".
- **Mock** (`tests/mock/mock_tch.c`), as implemented to match what a CPU
  build was measured to do: `atc_cudnn_is_available` → 0, seeds and the
  cuDNN switches no-ops, `atc_user_enabled_cudnn` → 1, `at_context_has_*`
  (incl. `has_cudart`) → 0; `atc_synchronize` and `at_context_version_*`
  set an error; `atm_fuser_cuda_set_enabled(true)` sets an error,
  `(false)` clears the flag. `mock_test.pas` checks `GpuBackend = gbNone`,
  `CudaIsAvailable`, `CudnnIsAvailable` False, `CudnnUserEnabled` True and
  that the three raise; `jit_mock_test.pas` checks the fuser flag (False
  reads back, True raises).

#### H13 and H30 on the GPU

- **H13 (real device move).** `VS.SetDevice(TDevice.Cuda(0))` after
  building `TLinear(VS.Root / 'l', 2, 1)` with the hand-set weights of
  4.6.7: afterwards `VS.Device.IsCuda`, `L.Weight.Device.ToC = 0`, every
  `VS.Variables` entry is on `cuda:0`, and an SGD optimizer **created
  after** the move steps `w` to the same `[0.9, 0.8]` as on CPU (read back
  through `to_device(Cpu)`). Then `SetDevice(Cpu)` brings it back with the
  values intact and `TchLiveHandles` unchanged (set_data swaps storage,
  no new handles). An optimizer created *before* the move is **not** part
  of the test: tch-rs documents it as unsupported and so does 4.6.2.
- **H30 (cuDNN/MIOpen RNN weight flattening).** tch-rs `rnn.rs` (**H36 resolved**,
  checked against tch-rs 4227b89b `src/nn/rnn.rs` lines 154–166 and
  239–251 by the orchestrating session; tch-rs passes `4` even without
  biases, so `2` below is a stated deviation following PyTorch): after creating the weights,
  `if vs.device().is_cuda() && Cuda::cudnn_is_available() {
  Tensor::internal_cudnn_rnn_flatten_weight(&flat_weights, 4, input_dim,
  mode, hidden_dim, proj_size = 0, num_layers, batch_first, bidirectional) }`
  with `mode` 2 = LSTM, 3 = GRU (cuDNN `cudnnRNNMode_t`), `4` = number of
  weights per layer-direction when biased. In `TRNN`'s protected
  constructor add exactly that condition, calling the generated class
  function `TTensor._cudnn_rnn_flatten_weight(FFlatWeights, WeightStride0,
  InputSize, Mode, HiddenSize, 0, NumLayers, BatchFirst, Bidirectional)`
  with `WeightStride0 = 4` when `HasBiases` else `2`, discarding the
  result, inside `try ... except on ETorchNativeError do ; end` — the
  flattening is a performance hint, and on ROCm it may be refused
  (`_cudnn_rnn_flatten_weight` is a cuDNN-dispatch op; MIOpen has its own
  `miopen_rnn`). The H30 measurement records which of the two happens on
  gfx1201 and that the GRU/LSTM numbers of 4.6.10 (`0.380797`, `0.1817`)
  are reproduced on the GPU either way.

#### GPU test: `tests/gpu/gpu_test.pas` (+ `run.sh`)

Built like `tests/nn` but linked against `build/libtch-rocm` (or `-cuda`;
`run.sh` takes the variant as `$1`, default `rocm`, and sets
`HIP_VISIBLE_DEVICES=0` / `CUDA_VISIBLE_DEVICES=0`). First statement:
`if not CudaIsAvailable then begin WriteLn('SKIP: no GPU in this libtorch');
Halt(0); end;` — so the same program exits 0 under `build/libtch` (CPU)
and the regression list can include it unconditionally. Numbers are the
CPU ones of earlier groups, so they are hand-checkable:

1. Capability print (not checked, recorded in HANDOFF): `GpuBackend`,
   `CudaDeviceCount` (expect 1 with the visibility variable),
   `CudnnIsAvailable`, `GpuRuntimeVersion`, `CudnnVersion`,
   `at_context_has_cuda`/`has_hip`. Added in local session 8 as item 1b,
   checked: `CudaDeviceCount = 1` (H33), and for the variant named on the
   command line `GpuBackend = gbRocm` and `CudnnIsAvailable` False (H35,
   H34), or `gbCuda`.
2. `T := TTensor.FromArray([1.0, 2.0, 3.0]).to_device(Cuda(0), kFloat)`:
   `T.Device.ToC = 0`, `T.Device.IsCuda`; `(T * 2 + 1).to_device(Cpu)` →
   `3 5 7`; `T.sum.to_device(Cpu)` → `6`; `T.sum.DoubleValue([])` directly
   on the CUDA tensor was expected to raise; *measured*, it returns 6
   (libtorch copies the scalar to the host), and the test checks that.
3. `CudaSynchronize` after a `matmul` of `[64, 64]` ones → `to_device(Cpu)`
   all `64`; `CudaManualSeedAll(7); A := randn([4], Cuda(0)); 
   CudaManualSeedAll(7); B := randn([4], Cuda(0))` → `A.equal(B)` True
   (same-seed reproducibility, values not hand-checked).
4. Autograd on GPU: `x := [3.0]` on `cuda:0` with `requires_grad_`;
   `(x * x).sum.Backward`; `x.grad.to_device(Cpu)` → `6`.
5. H13 as above (`TLinear`, `SetDevice`, SGD → `[0.9, 0.8]`, back to CPU).
6. `TVarStore.Load` of the 4.6.7 `.ot` file into a store created on
   `Cuda(0)`: variables arrive on `cuda:0` (`at_load_callback_with_device`
   with `device_id = 0`), values equal to the CPU load. Implemented with a
   one-variable file written by a CPU store (nested names are not
   exercised on the GPU).
7. JIT: `TJitModule.Create('tests/jit/models/foo1.pt')`, `ToDevice(Cuda(0),
   kFloat)`, forward with inputs moved to `cuda:0` → the same result as
   group 1 of 4.6.8, read back through `to_device(Cpu)`.
   `SetJitFuserCudaEnabled(True)` was to be set and read back True;
   *measured*, it raises on every libtorch 2.13 build (nvfuser removed), so
   the test checks the exception. Implemented with `foo.pt` (two float
   inputs) instead of `foo1.pt` (0-d Int64), compared with the CPU result.
8. H30: `TGRU(VS on Cuda(0), 1, 1)` and `TLSTM(…, 1, 1)` with the 4.6.10
   hand weights → `0.380797` / `0.380797, 0.181700` read back on CPU; the
   flatten condition is printed (False on ROCm). `TConv2D` group-8 numbers
   on the GPU (MIOpen conv path); `TBatchNorm` in training mode compared
   with the CPU on the same input (output and `running_var` within 1e-5).
9. Mixed-device error: `TTensor.FromArray([1.0]) + T` (CPU + CUDA) raises
   `ETorchNativeError` with "Expected all tensors to be on the same
   device" in the message; the process continues.
10. `TchLiveHandles = 0` at the end.

Expected CPU-only run: the single `SKIP` line. `tests/mock` is unaffected
(the GPU test is never linked against the mock).

#### Results (local session 8, 2026-10-12, RX 9070, `2.13.0+rocm7.1`)

Implemented as above; `tests/gpu/run.sh rocm` passes every item, `run.sh
cpu` prints the single `SKIP`. Measured, against the hypotheses and the
design:

- **(H31) holds.** The manual-link libtch builds against `libtorch-rocm/`
  with no extra define (same four `-Wc++20-extensions` warnings from the
  libtorch headers as the CPU build) and works.
- **(H32) amended.** `ldd libtorch_hip.so` was clean, but the archive (a
  zip) has no symlinks: seven libraries exist only under their unversioned
  file name while their SONAMEs and dependents use the versioned one, and
  `libtorch_rocshmem.so`'s initialiser `dlopen`s an unversioned
  `libnuma.so` and exits the process when it is missing. Fix after
  unpacking (git-ignored directory, reversible):
  `cd libtorch-rocm/lib && for f in libamd_comgr.so libamdhip64.so
  libhipblas.so libhipsparse.so libhsa-runtime64.so librocblas.so
  libroctx64.so; do ln -s $f $(readelf -d $f | sed -n
  's/.*SONAME.*\[\(.*\)\]/\1/p'); done && ln -s libnuma.so.1 libnuma.so`.
  Its other `dlopen`s (`libibverbs.so`, `libmlx5.so`, `libbnxt_re.so`,
  `libionic.so`, RDMA) fail harmlessly but print `(null): No such file or
  directory` on stderr at start-up, four times.
- **(H33)** confirmed at run time: `CudaDeviceCount = 1` with
  `HIP_VISIBLE_DEVICES=0`, every kernel used runs on gfx1201.
- **(H34) false.** `CudnnIsAvailable` is False and `at_context_has_cudnn`
  False on ROCm; MIOpen does not stand in through these hooks.
  `CudnnVersion` raises "Cannot query CuDNN version if ATen_cuda is not
  built with CuDNN". So the `TRNN` flatten guard is False on ROCm and the
  call is never made (H30: nothing to refuse); GRU `0.380797` and LSTM
  `0.380797 / 0.181700` are reproduced on the GPU.
- **(H35) answered, design changed.** On ROCm `at_context_has_hip` is
  **False** and `at_context_has_cuda` True, so "ROCm when has_hip" would
  report CUDA. `at_context_has_cudart` is False there, and
  `at_context_version_cudart` raises "Cannot query CUDART version because
  CUDART is not available". `GpuBackend` therefore is `gbRocm` when HIP is
  reported or CUDA without CUDART, `gbCuda` when CUDA with CUDART
  (`at_context_has_cudart` added to the L2 list); the CUDA side is not
  measurable here. `GpuRuntimeVersion` raises on ROCm and CPU builds.
- **nvfuser.** `atm_fuser_cuda_set_enabled(true)` raises on every build of
  libtorch 2.13, CPU included ("nvfuser support in torchscript is removed
  and cannot be enabled!"), and both accessors print a deprecation
  warning. `SetJitFuserCudaEnabled` is kept for parity: False is a no-op,
  True raises; the CPU jit test and GPU item 7 check exactly that.
- **CMake.** `option()` takes only booleans, so `TCH_LIBTORCH_DIR` is a
  `CACHE PATH` variable. The default path is unchanged (the CPU
  `libtch.so` is bit-identical after reconfiguring).
- Item 2: `DoubleValue` on a GPU scalar returns the value (6), it does not
  raise; libtorch copies to the host. H13 on the GPU: no new handles, SGD
  gives `[0.9, 0.8]`, values intact after moving back. BatchNorm2d in
  training mode differs from the CPU by 2.4e-7. The CPU-linked `tests/nn`
  and `tests/jit` pass with `LD_LIBRARY_PATH=build/libtch-rocm`;
  `tests/api` fails its three CPU-build assertions there, as it should.
- HIP start-up costs about 4 s of system time per process.

## 5. Code generation (decision)

Implemented in cloud session 3; everything below is what the code does
(`gen/gen.ml`, `gen/tchgen.pas`), measured on libtorch 2.13.0's
`Declarations.yaml` (2,657 ops).

- gen.ml (OCaml) already parses `Declarations-v2.13.0.yaml` and knows the
  real argument kinds. The fork adds one output, the manifest `gen/ops.tsv`,
  written when run as `gen.exe <Declarations.yaml> <csrc/libtch dir>
  <ops.tsv>` (no Rust files in that mode; without arguments it behaves as
  upstream, for the byte-for-byte check). One op per line, TAB separated,
  `#` lines are comments:

  ```
  name  receiver  result  arg:Type[=default]...  [->ret,ret,...]
  ```

  | column | values |
  |---|---|
  | name | exported name; the C function is `atg_<name>` |
  | receiver | `self`, `input` (the Tensor argument that becomes the Pascal receiver, tch-rs's rule) or `-` (static class function) |
  | result | `nothing`, `fixed<N>` (N = 1..7), `dynamic`, `bool`, `int64`, `double` |
  | args | every argument in C order, receiver included; `Type` is gen.ml's constructor name (`Tensor`, `TensorOption`, `Int64Option`, `IntList`, ..., `LayoutOption`); `=default` only when the YAML has one |
  | `->` | only for `fixed<N>` with N ≥ 2: the N result names from `returns[].name` (`result0..` when the YAML has none) |

  Default text is canonical (`default_text` in gen.ml), because the yaml
  library types plain scalars (`1` becomes `Float 1.`): `true`/`false`,
  integral numbers without a point (`-1`, `0`), other numbers as `%.16g`
  (`1e-05`), strings verbatim including a C++ literal's quotes
  (`"constant"`, `::std::nullopt`, `at::Reduction::Mean`, `{}`). Names
  contain no `:`, `=` or TAB, defaults no TAB or newline (asserted).
  Examples:

  ```
  add	self	fixed1	self:Tensor	other:Tensor
  max_dim	self	fixed2	self:Tensor	dim:Int64	keepdim:Bool=false	->values,indices
  conv2d	input	fixed1	input:Tensor	weight:Tensor	bias:TensorOption={}	stride:IntList=1	...	groups:Int64=1
  zeros	-	fixed1	size:IntList	options:TensorOptions={}
  split	self	dynamic	self:Tensor	split_size:Int64	dim:Int64=0
  sort_values	self	fixed2	values:Tensor	indices:Tensor	self:Tensor	dim:Int64=-1	descending:Bool=false	->values,indices
  ```

- The Pascal emitter `gen/tchgen.pas` (FPC program; `fpc gen/tchgen.pas`
  or `instantfpc`) reads `gen/ops.tsv` and `src/Tch.pas` and writes
  `src/TchSysGenerated.pas`, `src/TchTensorGenerated_intf.inc` and
  `src/TchTensorGenerated_impl.inc`. It owns all Pascal-specific policy
  (4.3, 4.4): the parameter type per argument kind, `&` escaping of
  reserved words for op and parameter names (a table of FPC keywords and
  modifiers; `result` → `result_`; `self` is only ever the receiver),
  default-parameter runs, out parameters or none, nil-handle checks,
  Scalar and tensor-list temporaries. It scans the `TTensor = record`
  declaration in `Tch.pas` for the hand-written member names and stops
  with an error on: an op name equal to a member (case-insensitive), a
  generated local (`<arg>__`, `Outs__`, `P__`, `R__`) equal to a member, a
  parameter named `system`/`tch*`/`ttensor`/`self`, a partial overlap of
  result and argument names, a default it cannot translate, an unknown
  argument kind. Nothing is skipped silently. No Python/Node (CLAUDE.md).
- Regeneration (only when bumping libtorch; OCaml is a maintainer tool):

  ```
  eval $(opam env --switch=tch-fpc)
  (cd gen && dune build ./gen.exe)
  gen/_build/default/gen.exe path/to/Declarations-v2.XX.yaml csrc/libtch gen/ops.tsv
  fpc -FUbuild/gen -obuild/gen/tchgen gen/tchgen.pas && build/gen/tchgen gen/ops.tsv src
  ```

  The cloud session had no `yaml` opam package (proxy); it ran the same
  gen.ml against a 150-line stand-in `Yaml` module that parses this file's
  block subset with ocaml-yaml's plain-scalar typing, and the resulting
  `torch_api_generated.{h,cpp}` were byte-identical to the committed ones.
  The committed `ops.tsv` comes from that run; the local session should
  reproduce it with the real library once (expected identical).
- Generated files (`ops.tsv`, `torch_api_generated.*`,
  `TchSysGenerated.pas`, the `.inc` files) are committed, so OCaml is only
  needed when bumping libtorch, not to build tch-fpc. `Tch.pas` with all
  2,657 ops compiles in 2.7 s (48,691 lines).
- `tests/mock` (cloud session 3): a mock `libtch.so` in C (every `atg_*`
  stubbed via `gen_stubs.sh` to report "mock: not implemented", a dozen
  implemented on a tiny double-array tensor) plus `mock_test.pas`, which
  runs the generated layer without libtorch: fixed-1, out parameters,
  results-are-inputs, dynamic, scalar results, Scalar temporaries freed,
  tensor lists (nil element rejected in Pascal, empty list rejected in C),
  `IntListOption` nil-vs-empty, String + `DoubleOption`, defaults, the
  error split, leak counters. 45 checks, `tests/mock/run.sh`. It is a
  plumbing test, not a numerical one.
- Rejected: parsing `torch_api_generated.h` directly — it loses the type
  distinctions listed in section 2. Rejected for now: a YAML parser in Pascal
  — doable, but duplicates gen.ml's accumulated special cases.

## 6. Scope of the first milestone

CPU only, Linux x86-64, libtorch 2.13.0:
create tensors from Pascal arrays, read data back, print, arithmetic
operators, a handful of generated ops, error → `ETorchError`, no leaks under
a loop of 10^6 temporaries, FPU mask verified. Autograd (`at_backward`,
`requires_grad_`), optimizers, VarStore/nn, JIT modules and CUDA follow in
that order.

**Status (local session 2, 2026-10-10): milestone 1 done.** Hand-written
API in `src/Tch.pas` as sketched in 4.1/4.3/4.7 (with `FromInt32Array`,
`CopyData`, `ZeroGrad`, `Backward(KeepGraph, CreateGraph)` and the factory
overloads of 4.4); tests: `tests/generated` (generated ops on real
libtorch), `tests/api` (hand-written API, element sizes), plus the
earlier `tests/ownership`, `tests/fpu`, `tests/abi`, `tests/mock`,
`tests/compiler`. Autograd basics (`requires_grad_`, `Backward`, `grad`,
`NoGradGuard`) are in; optimizers, VarStore/nn (4.6), JIT and CUDA are
next, in that order.

**Milestone 2 (designed in cloud session 4, 2026-10-10):** everything in
4.6 except JIT — `TInit`, `TVarStore`/`TVarPath`, save/load, `TOptimizer`,
`TLinear`, `TConv2D`, `TBatchNorm`, `TLayerNorm`, `TEmbedding`,
`TFuncModule`, `TSequential` — with the acceptance test of 4.6.7. Order
of implementation and the new hypotheses H13–H18 are in HANDOFF.md.

**Status (local session 3, 2026-10-10): milestone 2 done.** `src/TchNN.pas`
implements 4.6 except JIT; `tests/nn/nn_test.pas` (real libtorch) passes
with every number of 4.6.7 and H13–H18 resolved (results next to each
hypothesis above); `tests/mock/nn_mock_test.pas` covers the plumbing.
Next: JIT (`TchJit`), then CUDA.

**Milestone 3 (designed in cloud session 5, 2026-10-10): JIT.**
`src/TchJit.pas` as in 4.6.8 — `TIValue` for tags 0–13, `TJitModule`,
`TTrainableJitModule`, tracing, scripted modules from TorchScript source
through a three-function fork addition, the profiling/fuser switches —
with tch-rs's `tests/foo*.pt` vendored into `tests/jit/models/`,
`tests/jit/jit_test.pas` and the mock stubs. TorchScript objects (tag 14)
and `atm_fuser_cuda_*` are deferred. The same local session also applies
the `FromArray` revision of 4.3 and the 4.6.7 follow-ups. Order and the
hypotheses H19–H25 are in HANDOFF.md.

**Status (local session 4, 2026-10-11): milestone 3 done.** `src/TchJit.pas`
implements 4.6.8 with the fork addition in `csrc/libtch`;
`tests/jit/jit_test.pas` passes items 1–13 and H19–H25 hold (results at
the end of 4.6.8); `tests/mock/jit_mock_test.pas` covers the plumbing. The
`FromArray` revision (4.3) and the 4.6.7 follow-ups are in. Still open:
TorchScript objects (tag 14), the milestone 3 layers (4.6.5 deferred
list), CUDA/ROCm. (All three are implemented since: local sessions
6, 7 and 8, below.)

**TorchScript objects (designed in design session 6, 2026-10-11; implemented in local session 6, 2026-10-12):** 4.6.9
— `ivObject = 14`, the refcounted `IJitObject` handle inside `TIValue`
(`MethodIV`, `GetAttr`), `TJitModule.CreateClass`, `JitLiveObjects`, the
four L2 externals, no fork patch; tests `jit6`/`jit_double_free` on the
vendored `foo6.pt`/`foo7.pt` as group 14 of `tests/jit/jit_test.pas`, plus
the mock's `TestObjects`. Most of it is measured against libtorch 2.13.0;
H26–H28 hold (results in 4.6.9). Still open after it: the milestone 3
layers (implemented since, local session 7) and CUDA/ROCm.

**Milestone 3 layers (designed in design session 7, 2026-10-11; implemented
in local session 7, 2026-10-12):** 4.6.10 — `TConvND` with
`TConv1D/2D/3D` (`TConv2D` rebased, API unchanged), the conv-transpose
family, `TGroupNorm`, `TRNN`/`TLSTM`/`TGRU`, `TIdentity`; group 11 of
`tests/nn/nn_test.pas` reproduces every measured number. H29 (`.ot`
interchange with tch-rs) stays open: it needs a Rust toolchain.

**CUDA/ROCm (designed in design session 8, 2026-10-11; implemented and run on the RX 9070 in local session 8, 2026-10-12, results at the end of 4.8):** 4.8 — one
libtorch directory and one `build/libtch-<variant>` per variant, a
`TCH_LIBTORCH_DIR` manual-link mode in the fork's CMakeLists (measured to
build against the CPU archive; avoids needing the ROCm SDK's cmake files),
no C or ABI change (HIP masquerades as CUDA, confirmed in the local
headers), the remaining `atc_*` / `at_context_*` / `atm_fuser_cuda_*`
externals with their `Cuda*`/`Cudnn*`/`GpuBackend` wrappers, the cuDNN
RNN flatten call in `TRNN`, and `tests/gpu/gpu_test.pas`, which prints
`SKIP` and exits 0 on a CPU build. Measured on the RX 9070: H31 holds,
H32 amended (eight symlinks after unpacking), H33 confirmed, H34 false, H35
answered (detection rule changed), H36 resolved, H30 moot on ROCm; the
CUDA path is unmeasured. With this, every design item from "Handing back
(end of local session 4)" is implemented.

## 7. LLM-facing knowledge package (decision, design session 9)

A skill that ships in this repository so that users and their coding agents
(Claude Code, Hermes Agent, others) write correct tch-fpc code. It is the
retrieval and few-shot material of the project; it is also what the local
Qwen reads in section 8. Model: the `pascal-dialects` plugin of
`tankoha/polyglot-scripting` (knowledge in `skills/`, examples and pitfalls
verified by a script). Nothing below is implemented yet; statements that
need a run are marked **(H37)**–**(H39)**.

### 7.1 Location and format

```
.claude-plugin/marketplace.json     repo root: one plugin, "source": "./plugin"
plugin/
  .claude-plugin/plugin.json        name tch-fpc, license "MIT OR Apache-2.0"
  skills/tch-fpc/
    SKILL.md                        the only file an agent must read
    reference/                      one topic per file, read on demand
      naming.md  ownership.md  autograd.md  nn.md  jit.md
      devices.md  build.md  errors.md
    examples/NN_name.pas + NN_name.expected
    pitfalls/pNN_name.pas           must fail to compile ({ expect-error: ... })
    traps/tNN_name.pas + tNN_name.expected   compiles; shows wrong and right
    scripts/build-example.sh        the one build line, parameterised
tests/skill/check-examples.sh       verification (7.4)
tests/skill/check-index.sh          SKILL.md tables and doc index vs files
```

- **One skill, not several.** The API is one library; splitting by unit
  would make an agent guess which skill to load. Progressive disclosure is
  inside the skill: `SKILL.md` carries the rules that prevent most errors
  and three tables (examples, pitfalls and traps, doc index); `reference/`
  and the example files are read one at a time.
- **Size budget (for a 64k-context model):** `SKILL.md` at most 160 lines;
  each `reference/*.md` at most 120 lines; each example at most 80 lines
  with one topic. `check-index.sh` enforces the line limits. Not measured in
  tokens; the limits are a starting point to revise after H39.
- **A plugin subdirectory, not the repo root.** `"source": "./plugin"`
  keeps the 53,000 generated lines and the docs out of the plugin cache.
  The price: an installed plugin cannot see `docs/` or `src/`. So the doc
  index names each target twice, as a path inside a checkout
  (`docs/DESIGN.md`, heading text) and as a URL pinned to the release tag.
  A user of tch-fpc has a checkout anyway (the units and the C wrapper are
  built from it).
- **Both loaders, one tree.** Claude Code: `/plugin install tch-fpc
  --marketplace tankoha/tch-fpc` reads `plugin/skills/`. Hermes:
  `skills.external_dirs` gets `<checkout>/plugin/skills`. `SKILL.md` front
  matter carries `name`, `description`, `version` and `metadata.hermes`
  (tags, category) as in `pascal-dialects`; how to read a file of the skill
  is given in both spellings (Read relative to the skill directory;
  `skill_view(name="tch-fpc", file_path=...)`). **(H37)** both loaders
  accept this tree as laid out; the layout is copied from a working one,
  but not run for this repository.
- **No hooks in the first version.** `pascal-dialects` forces a read and
  compiles after every write. Here a compile needs a built `libtch` and its
  path, which a hook cannot know. Instead the skill gives the agent one
  command, `scripts/build-example.sh <file.pas> [<libtch build dir>]`
  (`TCH_FPC_ROOT`, default the checkout the script lives in), and tells it
  to compile and run before reporting. The same script is what
  `check-examples.sh` calls, so the documented build line cannot drift from
  the tested one. Dialect rules stay in `pascal-dialects`; the skill says
  that user code is `{$mode objfpc}{$H+}` and points there.
- **Language.** English only, an exception to the `_jp` rule: a second
  `SKILL.md` would be a second skill with the same triggers. `README.md` /
  `README_jp.md` get a short section on installing it.
- **Licence.** Every file under the repository's `MIT OR Apache-2.0`.
  Examples are written for this repository; one ported from tch-rs's
  `examples/` names its source file in a header comment and in the commit
  message. The JIT examples read `tests/jit/models/foo*.pt` by path (their
  tch-rs licence files are already next to them); no model file is copied
  into the plugin, and an example that needs one says so and is skipped by
  an installed plugin without a checkout.
- **Version.** `plugin.json` and the `SKILL.md` front matter carry the
  tch-fpc release they describe; `SKILL.md` names the libtorch version
  (2.13.0) and FPC 3.2.2.

### 7.2 Contents of `SKILL.md`

1. *When to use*, and what it is not (not a PyTorch tutorial; not the
   dialect rules).
2. *Rules* — numbered, each one line plus a two-line code pair:
   - Units and mode: `uses Tch, TchNN, TchJit`; `{$mode objfpc}{$H+}`; the
     build line; never a global `LD_LIBRARY_PATH` (rpath per build).
   - Naming: generated ops are PyTorch names in snake_case
     (`t.max_pool2d`, `t.add_(u)`), the hand-written API is CamelCase;
     reserved words take `&` (`t.&div(u)`, `t.&to(dev)`, `&out`, `&end`);
     the parameter `result` is `result_`; ops without a `self`/`input`
     tensor are `TTensor.zeros(...)`; optional arguments are `NoInt64`,
     `NoDouble`, `NoIntList`, `IntList([...])`.
   - Ownership: `TTensor` is a managed record, never `Free` it; `B := A`
     shares the tensor; `ShallowClone` is a new handle on the same storage;
     `Copy` is new storage; `IsNil` and `Defined` differ. Classes
     (`TVarStore`, `TModule` descendants, `TOptimizer`, `TJitModule`) are
     freed by hand in `try..finally`, the store last; a `TVarPath` must not
     outlive its store.
   - Data in and out: an array literal is typed by its first element
     (`[1.0, 2.0]` Double to `kFloat`, `[1, 2]` Int64); `FromSingleArray`,
     `FromInt32Array`, `FromBytes`; `DoubleValue`, `ToSingleArray`,
     `CopyData`.
   - Operators: `*` is elementwise, `matmul` is the matrix product; scalar
     overloads on both sides.
   - Autograd: `set_requires_grad`, `Backward`, `grad`, `ZeroGrad`;
     `G := NoGradGuard;` must be kept in a local.
   - `TchNN`: `VS := TVarStore.Create(Device)`, `VS.Root / 'name'`, layer
     constructors take a path, `Forward(X, Train)`, optimizers are class
     functions (`TOptimizer.Adam(VS, LR)`), `BackwardStep`, `Save`/`Load`;
     no anonymous functions in FPC 3.2.2, so `TFuncModule` takes a named
     function or method.
   - `TchJit`: `TJitModule.Create(FileName[, Device])`, `Forward`,
     `ForwardIV` with `TIValue`, `TTrainableJitModule`.
   - Devices: `TDevice.CudaIfAvailable`; "Cuda" means the AMD GPU on a ROCm
     build; `HIP_VISIBLE_DEVICES=0` when an iGPU is present;
     `CudnnIsAvailable` is False and `GpuRuntimeVersion` raises on ROCm;
     the ROCm archive needs eight symlinks.
   - Errors: `ETorchError`, `ETorchNativeError` (`Message`, `Detail`);
     libtorch needs FPU exceptions masked, the units do it at start-up, do
     not unmask.
3. *Examples*, *Pitfalls*, *Traps* — one table each: file, what it shows,
   and for pitfalls the compiler message.
4. *Where to look* — the doc index (7.3).
5. *Before you report*: compile and run with `build-example.sh`; quote the
   output.

`reference/*.md` hold what does not fit a line: the full reserved-word
list, the receiver rule, the ownership table for classes, the variable
naming of the store, the `TIValue` tags, the build variants. They are
written from DESIGN, short, and each ends with the DESIGN section it
summarises. They restate rules; they do not copy rationale.

### 7.3 Index instead of copies

`SKILL.md` has a table *question → where*: a topic, the file and heading
in a checkout, and the pinned URL. Rows planned: tensor ownership (4.1),
errors and FPU mask (4.2), value types and `FromArray` (4.3), generated ops
and naming (4.4), nn overview and each of 4.6.1–4.6.10, grad mode (4.7),
GPU builds and the ROCm results (4.8), the UML diagrams, and for exact
signatures `src/Tch.pas`, `src/TchNN.pas`, `src/TchJit.pas` (interface
part) and `src/TchTensorGenerated_intf.inc` with the instruction to search
it for `function <op>` rather than read it (4,835 lines). `check-index.sh`
greps every indexed heading in the named file and fails when one is gone,
so the index cannot rot silently.

### 7.4 Examples, pitfalls, traps and their verification

Three kinds, because most tch-fpc mistakes compile:

| Kind | Files | Check |
|---|---|---|
| example | `examples/NN_name.pas`, `.expected` | compiles, runs on CPU, output equals `.expected` |
| pitfall | `pitfalls/pNN_name.pas` | fails to compile, log contains the `expect-error:` text |
| trap | `traps/tNN_name.pas`, `.expected` | compiles and runs; prints the wrong result or the exception, then the right way; output equals `.expected` |

Planned set (each is one topic; names may change in implementation):

- Examples: `01_tensor_basics` (create, shape, kind, operators, scalar
  out); `02_generated_ops` (snake_case, `&`-escapes, static factories,
  optional arguments); `03_ownership` (assignment, `ShallowClone`, `Copy`,
  in-place through a `const`); `04_data_in_out` (`FromBytes`, `CopyData`,
  `FromInt32Array`, `totype`); `05_autograd` (gradient of a small
  function, `NoGradGuard` in a local, `ZeroGrad`); `06_train_linear`
  (store, `TLinear`, Adam, loss falls, free order); `07_custom_module`
  (a `TModule` descendant, `TSequential`, `TFuncModule` with a named
  function); `08_save_load` (variable names, `Save`/`Load`, `Freeze`);
  `09_device` (device-agnostic code, prints `cpu` in CI);
  `10_jit_run` (`foo*.pt`, `Forward`, `ForwardIV`); `11_jit_finetune`
  (`TTrainableJitModule`); `12_errors` (catching `ETorchNativeError`).
- Pitfalls (compile errors): `p01` a reserved-word op without `&`
  (`t.div(u)`); `p02` a mixed literal `[1, 2.5]`; `p03` an Int32 array
  passed to `FromArray`; `p04` a parameter written `&result`; `p05` an
  anonymous function for `TFuncModule`; `p06` `t.to(dev)` without `&`.
  `tests/compiler/xfail_*` already measures p02–p04; the skill reuses
  those facts, with its own short sources.
- Traps (run and show): `t01` a bare `NoGradGuard;` has no effect; `t02`
  `B := A` then an in-place op changes both; `t03` `*` where `matmul` was
  meant; `t04` `FromArray([1, 2])` is Int64 and `mean` raises; `t05` an
  in-place op on a leaf that requires grad raises without a guard; `t06`
  batch norm with `Train` left False does not update running statistics.
- Documented only, not runnable in CI (stated as such in `SKILL.md`): a
  `TVarPath` used after its store is freed; unmasking FPU exceptions; a
  device mismatch; the `LD_LIBRARY_PATH` trap; everything ROCm.

Expected output must be stable across libtorch CPU builds: examples print
shapes, kinds, integers, Booleans and floats rounded to three decimals, and
compare with a tolerance inside the program where the value comes from an
optimiser. No example prints a raw `ToString` of floats.

`tests/skill/check-examples.sh [name...]` (bash, same shape as
`pascal-dialects/tests/check-examples.sh`): for each file it calls
`scripts/build-example.sh` into a temporary directory against
`build/libtch`, runs the program from the repo root, and applies the check
of the table. It prints `OK`/`NG` per file and exits non-zero on any `NG`.
CI: one more step in the existing `libtorch-cpu` job, after "Run Pascal
tests", plus `check-index.sh` in the `no-libtorch` job (it needs no
libtorch). Estimated cost: 21 small programs at about 2–4 s compile each
(`Tch.pas` is compiled once into a shared unit directory); not measured.
**(H38)** every planned pitfall fails with a stable, greppable message on
FPC 3.2.2 (p02–p04 are measured in `tests/compiler`, the others not).

### 7.5 Does it work: the acceptance measure

A skill that is only verified to compile says nothing about whether a model
writes better code with it. **(H39)** is therefore a measurement, done once
the skill exists: ten small tasks ("write a program that …", each with a
checker script that compiles and runs the answer), given to the local Qwen
through Hermes with and without the skill, pass rate reported for both.
The tasks live in `tests/skill/tasks/` and are not part of the plugin. The
same task set is the readiness gate of section 8 (condition C4).

The skill's examples are retrieval material. They must never enter a
fine-tuning data set (8.7): they are written by Claude.

**H39 result (local session 10, 2026-11-17, *measured*).** Local Qwen
(`qwen3.8-27b-64k`) as the main model of Hermes, toolsets terminal, file
and skills, 40 turns, one run per task and mode:

| | with the skill (preloaded) | without |
|---|---|---|
| passed | 10 / 10 | 10 / 10 |
| total time | 1,759 s (29 min) | 3,305 s (55 min) |
| slowest task | 375 s (06) | 717 s (05) |

The pass rate cannot separate the two (both at the ceiling); the skill
cuts the time to about half; every task but 02 is faster with it (02:
352 s against 173 s). The tasks
are small; a harder set would be needed to show a difference in pass
rate. A first trial run without links to the units lost most of its 15
minutes to the Pascal verify hook (plain `fpc` cannot find `Tch`);
`run-h39.sh` now links the tch-fpc units into each task directory.

**H39 with other local models (local session 12, 2026-11-22,
*measured*).** Through Hermes, magistral small 24B passed 0/10 with and
without the skill and devstral-small-2 24B 2/10 with, 0/10 without; both
failed mostly at tool calling. A harness without an agent
(`tests/skill/run-direct.bb`: the model answers with the whole program,
compiler messages or output differences go back, up to 6 answers;
"without" gives no API information at all) gave:

| Model | with the skill | without |
|---|---|---|
| magistral small 24B | 0/10 | 2/10 |
| devstral-small-2 24B | 2/10 | 1/10 |
| Qwen3.8 27B | 9/10 | 1/10 |

Without the skill every model invents the API; with it only Qwen writes
working programs. The skill is what makes Qwen able to write tch-fpc
code when it cannot read the sources; the 24B models are not fallbacks
for the coding tasks (R4). Details in HANDOFF, local session 12.

## 8. Fine-tuning Qwen3.8 27B with tch-fpc: feasibility gate (design session 9)

**Verdict: GO with conditions. The design stops at this gate.** Training
the 27B model on two 16 GB GPUs is not ruled out by any number below, but
four things are unmeasured or need a decision by the user, and one of them
(how a trained adapter gets back into the running model) changes what has
to be built. The conditions and the options are in HANDOFF, "Design session
9". No module structure, LoRA classes, data specification or milestones are
designed yet; that is the next session, after the gate. **(Since done:
the gate passed and the detailed design is 8.9–8.19, design session 10.)**

Goal as given: improve the local Qwen's code in five Pascal dialects by
LoRA/QLoRA; the training program is a tch-fpc sample in Pascal, generated
by Qwen3.8 27B itself through the Hermes SDLC process.

How to read the numbers: *measured* = read from this machine today
(2026-11-16); *computed* = arithmetic on measured values; *estimate* = not
measured; *from memory* = not checked against a source in this session.
No libtorch is installed on the host today and the GPUs were not used, so
nothing about libtorch on the GPUs is measured.

### 8.1 What the model is

*Measured* from `ollama show -v` (866 tensors): architecture `qwen35`,
embedding 5120, FFN 17408, vocabulary 248320, 64 transformer blocks
(`blk.0`–`blk.63`) plus `blk.64`, the next-token-prediction (MTP) head that
Ollama uses as a draft model. Blocks 3, 7, …, 63 (16) are gated full
attention; the other 48 are linear attention (Gated DeltaNet). Token
embedding and output matrix are separate tensors. The vision projector is a
separate blob and is not needed for text training.

| Group | Parameters | Packed size (Q4_K 4.5, Q6_K 6.5625 bit/weight) |
|---|---|---|
| 48 linear-attention blocks | 18.40 B (383.3 M each) | 11.23 GB (up to 252.3 MB each) |
| 16 full-attention blocks | 5.96 B (372.3 M each) | 3.55 GB (up to 233.8 MB each) |
| `token_embd` (Q4_K) | 1.27 B | 0.72 GB |
| `output` (Q6_K) | 1.27 B | 1.04 GB |
| `blk.64` MTP head | 0.43 B | 0.26 GB (not loaded for training) |
| all | 27.32 B | 16.80 GB |

(*computed* from the tensor list. Q4_K_M mixes types by layer: `attn_qkv`, `attn_v` and `ffn_down` are Q6_K in half of the blocks, so a block is 209–252 MB.)

Block structure, read from upstream llama.cpp `src/models/qwen35.cpp`
(master, fetched 2026-11-16 and summarised by a tool; the commit Ollama
bundles, 39febd2ad, was not read) and consistent with the measured shapes:

- Linear block: RMS norm; `attn_qkv` 5120→10240 = q 2048 + k 2048 +
  v 6144 (16 key heads and 48 value heads of 128); `attn_gate` 5120→6144
  (`z`); `ssm_beta`, `ssm_alpha` 5120→48; β = sigmoid; decay gate =
  softplus(α + `ssm_dt`) · `ssm_a`; a causal depthwise conv of kernel 4
  over all 10240 channels, then SiLU; q and k L2-normalised; the gated
  delta rule with a state of 128×128 per value head; RMS norm with
  `ssm_norm` times SiLU(z); `ssm_out` 6144→5120. Then RMS norm and a SwiGLU
  FFN.
- Full block: `attn_q` 5120→12288 = 24 heads × (256 query + 256 gate);
  `attn_k`, `attn_v` 5120→1024 (4 KV heads, GQA); RMS norms on q and k per
  head; RoPE (64 dimensions, base 1e7, sections for multimodal positions);
  attention; output times sigmoid(gate); `attn_output` 6144→5120.
- Not read: the delta-rule kernel (`build_recurrent_attn`), RoPE details,
  the chunk size. *From memory*: the reference implementation uses a
  chunked form with chunk 64 and float32 inside the rule.

### 8.2 Memory budget per GPU

Layout: GPU 0 holds `token_embd` and blocks 0–31, GPU 1 blocks 32–63 and
`output`. The weights stay **packed** on the GPU as the GGUF's own bytes
and are dequantised per matrix when a block runs (8.3). One sequence of
T = 2048 tokens, activations in bf16, LoRA rank 16 on the attention-side
projections only (39.6 M trainable parameters, *computed*).

Backward without `torch.utils.checkpoint` (libtorch has none): block-wise
recomputation with ops tch-fpc already has. Forward all blocks under
`NoGradGuard` keeping each block's input; backward from the last block:
`x := saved.detach.set_requires_grad(True)`, `y := Block(x)`,
`(y * g).sum.Backward`, then `g := x.grad`. Only one block's graph is alive
at a time, the two devices never share a graph, and `g` crosses the split
with `&to`. Cost: two forwards and one backward per step.

| Item | GPU 0 | GPU 1 | Basis |
|---|---|---|---|
| packed weights | 8.10 GB | 8.43 GB | computed |
| dequantised weights of the block in flight (bf16) | 0.77 | 0.77 | computed; held by the graph |
| dequantisation temporaries | 0.5 | 0.5 | estimate |
| graph of one block (FFN 6×71 MB; attention scores 24×2048² ×3; delta-rule chunks) | 1.0–1.5 | 1.0–1.5 | estimate |
| saved block inputs (32 × 21 MB) | 0.67 | 0.67 | computed; can live in host RAM |
| LoRA parameters, gradients, AdamW state (fp32, ×4) | 0.32 | 0.32 | computed |
| output matrix dequantised (bf16), head phase only | – | 2.54 | computed |
| logits and loss for a 256-token slice (fp32, ×3), head phase only | – | 0.76 | computed |
| HIP context, BLAS/MIOpen handles, allocator slack | 1.0–1.5 | 1.0–1.5 | estimate, not measured |
| **peak** | **≈ 13.4 GB** | **≈ 13.7 GB (block), ≈ 14.2 GB (head)** | of 16 GB |

- The loss must be **chunked over the sequence**: full logits at T = 2048
  are 2.03 GB in fp32 before the softmax and its gradient.
- T = 4096 does not fit this way (attention scores grow fourfold);
  T ≤ 2048 is the design limit, 1024 the comfortable one. Code samples are
  expected to be 300–1500 tokens.
- LoRA on the FFN matrices as well: 108.8 M parameters, 0.87 GB per GPU
  instead of 0.32. Fits on paper, with under 1.5 GB margin.
- Host RAM (30 GB, 19 GB free, *measured*) is not a constraint: the file
  is read tensor by tensor (largest packed tensor 1.04 GB) or mapped with
  the generated `from_file`.
- While training runs, Ollama cannot hold the model (it uses about 21 GB
  of the 32 GB for weights and a 64k KV cache, *measured* from its log).
  Training and the local LLM exclude each other.

### 8.3 Base weights and dequantisation

Options:

| | GGUF blocks, dequantised in tensor ops | HF safetensors (bf16) + own 4-bit |
|---|---|---|
| Download | none; the 16.8 GB blob is on disk but readable only by the `ollama` user | about 54 GB |
| Adapter matches the deployed weights | exactly | no (trained on a different quantisation) |
| Code | GGUF reader; Q4_K and Q6_K dequantisation with byte ops | safetensors reader; a quantiser and its dequantiser |
| Memory | 16.8 GB packed | about 13.7 GB at 4 bit + scales |
| Deployment (8.4) | all three paths | needs upstream conversion tools (Python) for a new GGUF |

Leaning: **GGUF**. It needs no download, the adapter is trained against
the weights that will run, and every deployment path of 8.4 starts from
the GGUF anyway. Needed from the user: read access to the blob (a copy, or
a group/ACL change). The ops for dequantising exist in the generated API
(`view_dtype`, `bitwise_and`, `bitwise_right_shift`, `&index`/`gather`,
`from_file`; *measured* by grep). Cost per step: all 24.3 B block weights
are dequantised twice (two forwards); the rate on the GPU is not measured.
The token embedding needs only the rows of the batch (a Q4_K row of 5120
is 20 whole super-blocks, *computed*).

### 8.4 Getting the adapter back into the running model

*Measured* on the installed Ollama 0.40.13:
- The binary contains "LoRA adapters are deprecated; the adapter layer is
  carried over but new adapters cannot be created" and "LoRA adapters are
  no longer supported"; the current `docs/modelfile.mdx` no longer lists
  `ADAPTER` (the parser still accepts the word). Behaviour was not
  exercised, but **a Modelfile `ADAPTER` is not a path to build on.**
- Ollama serves GGUF models through a bundled upstream `llama-server`
  (0.8.0-dev, 39febd2ad, ROCm backend; the log shows `ROCm0`/`ROCm1` and
  an MTP draft context). `/usr/local/lib/ollama/llama-server --help` lists
  `--lora` and `--lora-scaled`; `llama-quantize` is bundled too and has
  `--tensor-type name=type` and `--allow-requantize`.
- Upstream `qwen35.cpp` sends every projection of both block types through
  `build_lora_mm` (the LoRA-aware product), including `ssm_out`,
  `attn_qkv`, `attn_gate`.

| Path | What is built | Trade-off |
|---|---|---|
| D2: GGUF LoRA adapter + `llama-server --lora`, outside Ollama | a small GGUF writer (F32 `*.lora_a`/`*.lora_b`) | exactly the trained configuration; Hermes must point at a second endpoint and something must start that server; whether the bundled build applies it to `qwen35` is unverified **(H44)** |
| D3: merged GGUF, adapted tensors stored as Q8_0, the rest copied byte for byte, `ollama create` from the file | GGUF writer, merge, Q8_0 quantiser (or F16 + `llama-quantize --tensor-type`) | stays in Ollama; about +3.3 GB for attention-side targets (*computed*: 7.21 B weights, 4.39 GB today, 7.67 GB at 8.5 bit); whether Ollama imports and runs it is unverified **(H45)** |
| D4: merge into the original bf16 weights, convert and quantise with upstream tools | 54 GB download, Python conversion script | the textbook route; heaviest; breaks the "no Python" habit for a vendor tool |

Not an option: merging into the dequantised weights and re-quantising to
Q4_K. A LoRA update is typically smaller than a 4-bit step, so most of it
would be rounded away (*estimate*, not measured; the standard reason merges
are done at full precision).

With every path the MTP head (`blk.64`) is no longer matched to the main
model; expect a lower draft acceptance (today 0.42–0.89, *measured* from
the log) and so slower generation, not wrong output (*estimate*).

Tokenizer: the GGUF carries tokens, merges and types (`gpt2`, pre-tokenizer
`qwen35`). Ollama 0.40.13 has no public tokenize endpoint (*measured*:
`POST /api/tokenize` and `/v1/tokenize` return 404; the bundled
`llama-server` has `/tokenize`). Leaning: a BPE tokenizer in Pascal from the
GGUF metadata, checked sample by sample against the bundled `llama-server`
started on the CPU for `/tokenize` **(H46)**. The chat template is rendered
inside Ollama (`renderer qwen3.8`), with thinking on by default; training
text must reproduce it exactly, which needs one captured rendered prompt.

### 8.5 The architecture in tch-fpc ops

*Measured* by grep in `src/TchTensorGenerated_intf.inc`: present are
`linear`, `matmul`, `conv1d`, `rms_norm`, `silu`, `sigmoid`, `softplus`,
`scaled_dot_product_attention`, `linalg_solve_triangular`, `einsum`,
`cumsum`, `tril`, `cross_entropy_loss`, `log_softmax`, `embedding`,
`view_dtype`, the bit ops, `from_file`, `totype`, `&to`, `detach`,
`set_requires_grad`, `grad`; `kBFloat16` and `kHalf` exist. The binding is
not the bottleneck.

Missing, all small:
- GPU memory statistics and an "empty cache" call: not in the C wrapper
  (a two-function fork addition); until then VRAM is read from
  `/sys/class/drm/card*/device/mem_info_vram_used`.
- `at_run_backward`, `at_autocast_*`, `at_tensor_of_blob`, `at_data_ptr`
  are in the C wrapper but not declared in `TchSys`; none is required.
- In the sample itself: GGUF reader and writer, K-quant dequantisation,
  tokenizer, the two block types, LoRA wrapper, the block-wise backward.

Risks:
- **No reference in this environment.** Correctness can only be shown
  end to end: next-token log-probabilities of the Pascal forward against
  the running model's (`top_logprobs` exists in the Ollama binary;
  *measured* as a string, not exercised) **(H43)**. Unit level: the chunked
  delta rule against a naive token-by-token recurrence written first; the
  tensor-op dequantiser against a scalar Pascal one.
- **Precision on ROCm.** Whether bf16 matrix products run on gfx1201 at
  all, and at what speed, is not known (H40). fp16 is the fallback, with an
  overflow risk in the residual stream; fp32 doubles the transient memory
  and does not fit T = 2048. The delta rule should run in fp32.
- **ROCm maturity.** `linalg_solve_triangular`, `conv1d` without cuDNN
  (H34: no cuDNN on ROCm) and SDPA are untested on this build; each has a
  plain-op fallback.

### 8.6 Speed

*Measured* (Ollama log, 2026-11-16, this model, both GPUs, llama.cpp ROCm
backend): prompt processing 740–1000 tokens/s in batches of 512,
generation 31–48 tokens/s. Prompt processing is one forward pass, so it is
the reference for what these GPUs do with this model.

*Estimate* for training: two forwards and one backward of about one to two
forwards are 3–4 forward equivalents; libtorch's dense half-precision
products instead of llama.cpp's kernels, plus dequantising twice per step,
cost a further factor of 1–3. Range **60–250 tokens/s, planning value
100**. Only one GPU works at a time (the layers are in series).

| Data set | Tokens per epoch | At 100 tok/s | Range |
|---|---|---|---|
| 5 dialects × 1000 samples × 700 tokens | 3.5 M | 9.7 h | 3.9–16 h |
| three epochs | 10.5 M | 29 h | 12–49 h |

An evaluation run (250 tasks × about 600 generated tokens at the measured
31–48 tokens/s) is about 1–1.4 h, before and after.

What only a measurement settles (the gate, about an hour of GPU time with
Ollama unloaded, needs the ROCm libtorch back: 12 GB unpacked,
`libtorch-shared-with-deps-2.13.0+rocm7.1.zip`, plus the eight symlinks of
4.8): **(H40)** bf16 and fp16 `matmul` of 2048×5120 by 5120×17408 on
gfx1201, in TFLOP/s, forward and backward; **(H41)** Q4_K/Q6_K
dequantisation rate of an 89 M-weight matrix; **(H42)** peak VRAM of one
synthetic block (random packed weights, T = 1024 and 2048) through the
block-wise backward, and whether two devices work from one process.

### 8.7 Can Qwen3.8 27B write this through Hermes

*Measured*: the SDLC harness ran one small project end to end
(2026-11-16, `hermes-agent-setup/sdlc/README.md`): Qwen produced detailed
design, test cases, tests and code with 100 % coverage. Not measured: any
tch-fpc code by Qwen, any numerical code of this size.

*Estimate*: not in one piece. The sample is about ten modules of which
four (dequantiser, delta rule, RoPE and gated attention, block-wise
backward) must match a reference to several digits, on an API the model
has never seen. It becomes plausible only if:
- the skill of section 7 exists and H39 shows it helps;
- each module has a numeric oracle the harness can run (golden vectors
  from a naive version, a scalar version, or the running model), written
  into the IT cases by the external design;
- the external design carries the formulas and tensor shapes (from the
  upstream sources, both permissively licensed), so Qwen translates rather
  than invents;
- there is a rehearsal on a small model of the same architecture on the
  CPU first (Qwen3.5 4B and 9B exist, seen in search results; smaller
  ones *from memory*): same code, no GPU, no conflict with Ollama;
- while a 27B run is on the GPUs, Qwen itself is unavailable; the
  27B stage is run by the user or by Mistral alone.

Data rules, recorded now because they bind every later design:
- Claude writes a data specification only, never samples (Anthropic's
  terms on training models with outputs). The skill's examples (section 7)
  and `pascal-dialects`' examples are Claude-written and stay out of the
  data set.
- Mistral generates the samples; **Mistral's terms on using outputs as
  training data must be checked by the user before generation starts**
  (not checked in this session). The base model's licence is Apache-2.0
  (*measured* from the GGUF metadata).
- A sample is kept only if it compiles and runs with its dialect's
  compiler. Oxygene cannot be verified here; leaning: exclude it from
  training and from the measured evaluation, and keep it retrieval-only.
  Delphi is approximated with FPC as in `pascal-dialects`, and reported
  separately from the three exactly verified dialects.

### 8.8 Verdict and conditions

**GO with conditions.** Memory fits on paper with about 2 GB margin per
GPU at T = 2048 (computed and estimated); the speed estimate gives a
day or two of training, not weeks; the binding lacks nothing essential.
What stops the design here:

- **C1 (decision):** the deployment path, D2 or D3 (or D4), since the
  Ollama `ADAPTER` route is closed. It decides whether a GGUF writer, a
  merge and a second server are in scope.
- **C2 (user action + measurement):** restore the ROCm libtorch and run
  the gate H40–H42 with Ollama unloaded. Abort criteria: no working
  half-precision product, under 30 tokens/s projected, or a peak above
  15 GB at T = 1024.
- **C3 (user action):** read access to the model blob; confirm the GGUF
  base (8.3); check Mistral's output terms.
- **C4 (measurement):** section 7 implemented and H39 run: Qwen writes
  tch-fpc code with the skill at a pass rate the user accepts.

If C2 fails, the fallbacks are a smaller model of the same family as the
training target, or renting a larger GPU for the training run only; both
are for the user to weigh and are listed in HANDOFF.

#### Gate results (2026-11-16, local, after design session 9)

Run with the ROCm libtorch restored (2.13.0+rocm7.1), Ollama unloaded,
`HIP_VISIBLE_DEVICES=0,1`. Device 0 is the RX 9070 XT (PCI 03:00.0, it
also drives the desktop: about 0.7 GiB in use before any run), device 1
the RX 9070 (PCI 09:00.0). VRAM is the whole card from
`mem_info_vram_used`, sampled after every block. The throwaway programs
are kept, git-ignored, in `build/gate-2026-11-16/` (`h40.pas`, `h41.pas`,
`h42.pas`, `kquant.pas`, `gatecommon.pas`).

- **(H40) holds.** 2048×5120 by 5120×17408, 50 iterations, *measured*:
  bf16 forward 100 TFLOP/s (XT) / 81 (9070), backward (two products)
  103 / 86; fp16 103 / 87 forward, 99 / 82 backward. Max relative error
  against fp32: bf16 4.1e-3, fp16 5.1e-4. Autograd through bf16 and fp16
  products works. **fp32 products are slow on this build: 1.6 / 1.3
  TFLOP/s**, so fp32 stays limited to the small delta-rule tensors.
- **(H41) holds.** Q4_K and Q6_K dequantisation written in tensor ops
  (`kquant.pas`) equal a scalar Pascal reference exactly (max abs diff 0
  over 512 random blocks each). Rate for an 89 M-weight matrix to bf16:
  Q4_K 3.2 ms, Q6_K 4.7 ms on either card (28 / 19 G weights/s). A whole
  model is dequantised in about 1 s per forward.
- **Chunked delta rule.** The chunked gated delta rule (chunk 64,
  `linalg_solve_triangular` for the in-chunk inverse) matches a naive
  token-by-token recurrence to 5.4e-7 (H = 2, T = 256, D = 16, fp32, CPU)
  and runs forward and backward on gfx1201. **Implemented (T09, local
  session 12):** `DeltaRuleChunked`, within 7.2e-7 of T08's token form at
  the 27B sizes for T = 1–2048 (4.4e-6 with strong decay, gradients
  2.1e-6); 13–17 ms at T = 2048 on the RX 9070. G must be finite (g = −inf
  gives NaN in the chunked form).
- **(H42), one training step of a synthetic stack**: 32 + 32 blocks with
  the real tensor types and shapes (every fourth block gated full
  attention with SDPA and GQA, the others Gated-DeltaNet), packed random
  weights, LoRA rank 16 on the attention-side projections (39 M
  parameters, plus two fp32 Adam moments), forward under NoGrad keeping
  block inputs, head loss over 256-token slices, block-wise backward with
  recomputation, both devices from one process. Not modelled: RoPE, MTP,
  vision, real data. Weights + LoRA + moments: 8.99 / 9.28 GiB.

  | T | step (2nd) | tokens/s | peak, whole card, XT / 9070 (of 15.92 GiB) |
  |---|---|---|---|
  | 1024 | 6.17 s (fwd 2.06, head 0.16, bwd 3.95) | **166** | 12.79 / **14.89** |
  | 2048 | 14.08 s (1st step) | 145 | 14.29 / 14.61, then **OOM on the 9070** in step 2 (head phase) |

  At T = 2048 the 9070 ran out in the head phase of the second step both
  with the default allocator (1.1 GiB reserved but unallocated, i.e.
  fragmentation) and with `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`
  (tried to map 2.37 GiB). The head phase alone adds about 3.5 GiB (the
  2.54 GB bf16 output matrix plus slice logits); dequantising it in one
  piece, as in a first try, added about 10 GiB of temporaries, so it is
  dequantised 8192 rows at a time.
- **New risk, measured: the desktop.** During the second T = 2048 run the
  kernel logged `ring gfx_0.0.0 timeout` on 0000:03:00.0 for labwc's
  command submission, reset the ring ("device wedged, but recovered
  through reset") and the Wayland session restarted. Long compute on the
  card that drives the display can take the desktop down.

**Reading against the abort criteria.** Half precision works (H40);
the projected speed, 145–166 tokens/s on the synthetic stack, is far
above 30 and at the top of 8.6's 60–250 range (real data, RoPE and the
optimizer step will cost some of it). The memory criterion, "peak above
15 GB at T = 1024", is met only just: 14.89 GiB on the 9070, which holds
the head. **T = 2048 does not fit with the 32/32 split.** So the gate
passes for T = 1024 with three new conditions:

1. Move the desktop off the two RX 9070s (monitor on the Raphael iGPU, or
   no session while training), which removes both the hang risk and the
   0.7 GiB the XT gives to the display.
2. Re-balance the split by memory: the 9070 carries the head, so it gets
   fewer blocks (e.g. 36/28); the XT is also about 20 % faster (H40).
3. Keep T ≤ 1024 for training samples, or move the saved block inputs to
   host memory and re-measure T = 2048 after 1 and 2.

#### Gate re-run (2026-11-17, display on the iGPU)

Both monitors moved to the Raphael iGPU and the BIOS primary display set to
it: `boot_vga` is now on 10:00.0, both RX 9070 cards idle at 0.08 GiB, and
the DRM card numbers changed (XT = `card0`, 9070 = `card1`, iGPU =
`card2`; HIP order unchanged). `h42` takes the split and the host offload
as arguments (`h42 step <T> <blocks on GPU 0> <blocks on GPU 1> [host]`).

| T | split | saved inputs | step (2nd) | tokens/s | peak, whole card, XT / 9070 (of 15.92 GiB) |
|---|---|---|---|---|---|
| 1024 | 36/28 | GPU | 6.12 s | **167** | 13.06 / 13.88 |
| 2048 | 36/28 | host | 11.90 s | **172** | 14.03 / 14.74 |

No GPU ring timeouts in the kernel log. **H42 now holds for T = 2048 as
well**, with the 36/28 split and the saved block inputs in host memory
(0.67 GiB per GPU moved out; copies cost little next to the step). The
margin is about 1.2 GiB on the 9070, which holds the head. Conditions 1–3
of the gate are met; the remaining Theme B conditions are C1, C3 and C4.

#### Decisions (2026-11-17)

- **C1: D3.** The deployment path is a merged GGUF: the adapted tensors
  stored as Q8_0, everything else copied byte for byte, imported with
  `ollama create`. The sample therefore needs a GGUF writer, the merge and
  a Q8_0 quantiser; H45 (Ollama imports and runs it) becomes a milestone
  check. D2 is not a deliverable; whether `llama-server --lora` (H44) is
  still worth running as an intermediate check, since D2's writer is a
  subset of D3's, is left to the next design session. D4 is dropped.
- **C3, Mistral's terms: settled by the user.** Mistral Large 3 is
  licensed under Apache-2.0, so its outputs may be used as training data.
  (Recorded as the user's answer; not re-checked in this session.)
- **C3, GGUF base: confirmed for the first run.** Train on the Q4_K_M GGUF
  through to a deployed D3 model; only after that run completes is the
  54 GB original considered (for a second run or a D4-style merge).
- **C3, Oxygene: in scope.** The user is asking RemObjects for an Oxygene
  licence. Oxygene stays in the training data and the evaluation; until a
  compiler is available here, its samples cannot pass the
  "compiles and runs" rule, so the next design session must say how
  Oxygene samples are gated and reported meanwhile (held back, or a
  separate unverified class). Delphi via FPC stays a separately reported
  class. *(2026-11-22: the compiler arrived before the data set was
  frozen; Oxygene is compiled and run like the other dialects and is in
  the first run — 8.16, 8.17.)*
- **C3, blob read access: done** (the user set read-only ACLs: traverse
  on the Ollama directories, list on `blobs/`, read on the two blobs of
  `qwen3.8:27b`). Both start with `GGUF`: the text model (16,810,714,464
  bytes, Q4_K_M) and a vision projector (931,146,016 bytes,
  `general.type` `mmproj`). Training uses the text model only; the D3
  Modelfile must keep the projector as its second `FROM` line, as the
  current model's does.

### 8.9 Detailed design: scope and conventions (design session 10, 2026-11-17)

Sections 8.9–8.19 continue from the passed gate and the decisions of
2026-11-17 (C1 = D3, C3). They fix the milestones, the units of the sample,
the LoRA layer, the block-wise backward, the D3 writer, the data pipeline,
the evaluation, the cut into Hermes tasks, and the stop criteria. Nothing
here is implemented.

Marks, as in 8.1–8.8: *measured* = read from this machine in this session
(GGUF header, tensor directory and a few values of the two blobs;
`ollama show --modelfile`; the bundled `llama-server --version`); *read in
source* = read in upstream llama.cpp at commit `39febd2ad`, the commit the
bundled `llama-server` reports (*measured*); *computed* = arithmetic on
those; *estimate*; *from memory* = not checked in this session. Throwaway
tools of this session are in the git-ignored `build/design10/`
(`ggufdump.pas`, the fetched upstream files).

Roles, binding for everything below:

| Who | Does | Never |
|---|---|---|
| Design session (Claude) | this design; reviews | committed code, training samples |
| Opus / Sonnet | the data specification (instructions, 8.16), the held-out evaluation tasks (8.17) | training samples |
| Mistral Large 3 | generates the training samples (external); was the main agent of the Hermes SDLC process for T06, T01 and T02, replaced for its false reports (`samples/qwen-lora/docs/mistral-large-3-issues.md`) | changes to the formulas and decisions of this section |
| GLM 5 (`zai.glm-5`, Bedrock; Hermes main agent from T03) | the external design and the IT cases of every task of 8.18 | changes to the formulas and decisions of this section |
| Hermes sub-agent: the local Qwen3.8 27B (128k) until T02 and again from T14; GLM 5 on Bedrock T05b3–T13, other models in between | detailed design, unit tests and code of the sample | — |
| User | downloads, every GPU run, `ollama create`, decisions | — |

New hypotheses of this session:

- **(H47)** a merged D3 GGUF of the rehearsal model, served by the bundled
  `llama-server` on the CPU, reproduces the log-probabilities of the Pascal
  forward with the adapter attached (8.15).
- **(H48)** at least 40 % of Mistral's samples pass the compile/run filter,
  and under 20 % of the accepted ones are near-duplicates (8.16).
- **(H49)** after the adapter is merged, generation speed with the MTP
  draft stays at 70 % or more of today's (8.17).
- **(H50)** a coverage tool produces an lcov report for an FPC test binary
  that the SDLC gate accepts (8.18). Tool: kcov (decided 2026-11-17).
  *Measured* 2026-11-17 with kcov 43 on a two-unit FPC program built
  with `-gl -gw -O-`: per-line hits are right (the one untaken `else`
  line has 0 hits); kcov writes Cobertura, SonarQube and JSON but no
  lcov, and a 15-line Babashka converter (Cobertura to `SF`/`DA`/`LF`/`LH`)
  produces lcov. Branch counts are not available (`:coverage-kind :c0`).
  Still open for M0: the converted report through the SDLC gate itself.

H44 (`llama-server --lora`) is **taken off the plan**. A merge only needs
the adapter convention to be consistent inside the sample (W' = W + s·B·A
in the sample's own layout), so an upstream LoRA loader proves nothing the
merge depends on. The intermediate check that does carry weight is H47:
upstream llama.cpp reads the merged file. H44 stays as a diagnostic if H45
fails and the user wants D2 as a fallback.

### 8.10 Architecture facts, verified against the source

Upstream files read at `39febd2ad`: `src/models/qwen35.cpp`,
`src/models/delta-net-base.cpp`, `src/models/models.h`
(`build_gdn_l2_norm`), parts of `src/llama-graph.cpp` (`build_lora_mm`,
`build_norm`, `build_qkv`, `build_attn_mha`, `build_ffn`,
`llm_graph_input_pos::set_input`), `src/llama-model.cpp` (RoPE type,
`n_rot`), `ggml/src/ggml-cpu/ops.cpp` (RoPE, RMS norm, `ssm_conv`,
`repeat`, flash attention head mapping), `ggml/src/ggml-cpu/ggml-cpu.c`
(`mul_mat` broadcast), `ggml/src/ggml-quants.c` (`quantize_row_q8_0_ref`),
`ggml/src/gguf.cpp` (alignment), `src/llama-vocab.cpp` (pre-tokenizer).
`master` differs from `39febd2ad` in `qwen35.cpp` only by optional fused
tensors (`attn_qkv_gate`, a concatenated beta/alpha) that this file does
not have; the design is pinned to `39febd2ad`.

**File** (*measured*): GGUF version 3, 42 metadata keys, 866 tensors, no
`general.alignment` (so 32), data section at byte 10,995,040; the end of
the last tensor equals the file size (*computed* = `stat`). Tensor order in
the file: `output.weight`, `output_norm.weight`, `token_embd.weight`, then
`blk.0` … `blk.64`. Types: 360 F32 (type 0), 439 Q4_K (12), 67 Q6_K (14).
866 = 3 + 48 × 14 + 16 × 11 + 15 (*computed*). There are no per-tensor
scale tensors and no Hadamard tensors, so those branches of
`build_lora_mm` are inactive. `general.file_type` 15.

Hyperparameters (*measured* from the metadata): `block_count` 65,
`nextn_predict_layers` 1 (so 64 trunk blocks + 1 MTP block),
`embedding_length` 5120, `feed_forward_length` 17408, `head_count` 24,
`head_count_kv` 4, `key_length` = `value_length` 256,
`layer_norm_rms_epsilon` 1e-6, `rope.dimension_count` 64,
`rope.freq_base` 1e7, `rope.dimension_sections` [11, 11, 10, 0],
`full_attention_interval` 4, `ssm.conv_kernel` 4, `ssm.state_size` 128,
`ssm.group_count` 16, `ssm.time_step_rank` 48, `ssm.inner_size` 6144,
`context_length` 262144.

Shapes below are in libtorch order `[out, in]`; a GGUF dimension list is
the reverse (`attn_qkv.weight` is listed as `5120 10240`). K-quant blocks
run along the flattened row-major data, so a row of 5120 is 20 whole
super-blocks.

**Block skeleton** (*read in source*), for trunk block `i`, hidden `h`:

```
a  = Attn_i(rms(h, attn_norm))          linear if (i+1) mod 4 <> 0, else full
h1 = h + a
h2 = h1 + Wdown · (silu(Wgate · n) ⊙ (Wup · n)),   n = rms(h1, post_attention_norm)
logits = Woutput · rms(h_64, output_norm)
rms(x, w) = x / sqrt(mean(x²) + 1e-6) ⊙ w          (mean over the last dimension)
```

No bias anywhere. Activations are F32 throughout upstream.

**Linear block (Gated DeltaNet)** (*read in source*), `x = rms(h)`,
T tokens, one sequence:

1. `qkv = Wqkv·x` [T, 10240]; `z = Wgate·x` [T, 6144];
   `β = sigmoid(Wbeta·x)` [T, 48];
   `g = softplus(Walpha·x + ssm_dt.bias) ⊙ ssm_a` [T, 48]. `ssm_a` is
   negative (*measured*: block 0 starts −0.0406, −0.0095, …), so `g ≤ 0`
   and `exp(g)` is a decay.
2. Causal depthwise convolution over time, no bias, three zeros in front:
   `u[t, c] = Σ_{i=0..3} w[c, i] · qkv[t − 3 + i, c]`, then SiLU.
   `ssm_conv1d.weight` is `[10240, 4]` in libtorch order; index 3
   multiplies the current token (*measured*: it dominates in block 0).
   This is `conv1d` with `groups = 10240`, left padding 3.
3. Split `u`: `q = u[:, 0:2048]` as [T, 16, 128], `k = u[:, 2048:4096]`
   as [T, 16, 128], `v = u[:, 4096:10240]` as [T, 48, 128].
4. `q`, `k`: L2 norm per head, `x / sqrt(Σx² + 1e-6)`
   (`build_gdn_l2_norm` is an RMS norm with ε/n, scaled by 1/√n).
5. **Key heads are tiled to the 48 value heads: value head `j` uses key
   head `j mod 16`** (`ggml_repeat_4d`, whose kernel copies whole source
   blocks in turn). This differs from the full-attention blocks below.
6. `q := q / sqrt(128)`.
7. Gated delta rule per value head, state `M` 128 × 128 (value × key),
   zero at sequence start; per token (the autoregressive form in
   `build_delta_net_autoregressive`, which is the oracle):
   ```
   M := exp(g_t) · M
   d := β_t · (v_t − M · k_t)
   M := M + d · k_tᵀ
   o_t := M · q_t
   ```
   The multi-token form is chunked with chunk size 64, the sequence padded
   to a multiple of 64, a unit-lower-triangular solve inside the chunk and
   one state update per chunk; no clamp on the cumulative gate. (8.1's
   "chunk 64" is now *read in source*; "float32 inside the rule" is
   simply upstream's F32 activations.)
8. `o` [T, 48, 128] → `rms(o, ssm_norm)` over the 128 (weight of 128,
   shared by the heads) `⊙ silu(z as [T, 48, 128])` → [T, 6144] →
   `Wssm_out` → [T, 5120].

**Full block (gated attention)** (*read in source*):

1. `qg = Wq·x` as [T, 24, 512]; per head the first 256 are the query, the
   last 256 the output gate. `k = Wk·x`, `v = Wv·x` as [T, 4, 256].
2. `q := rms(q, attn_q_norm)`, `k := rms(k, attn_k_norm)` over the 256
   (weights of 256, shared by the heads). `v` is not normalised.
3. RoPE on `q` and `k`, type IMROPE (interleaved multimodal). For text
   tokens the position is `[p, p, p, 0]`, and with sections [11, 11, 10, 0]
   every one of the 32 rotated pairs takes `p` (*computed* from the read
   kernel), so for text it is: **only the first 64 of the 256 dimensions
   of a head rotate**, pair `(i, i + 32)` for `i = 0..31`, angle
   `p · 1e7^(−i/32)`; `x0' = x0·cos − x1·sin`, `x1' = x0·sin + x1·cos`;
   dimensions 64–255 pass unchanged. Positions start at 0.
4. Causal attention, scale `1/sqrt(256)`, softmax in F32. GQA: **query
   head `h` uses key/value head `h div 6`** (`iq2 / rk2` in the kernel).
5. Output [T, 6144] `⊙ sigmoid(gate as [T, 6144])` → `Wattn_output`.

**MTP block `blk.64`** (*read in source*): input
`eh_proj · concat(rms(emb(token), enorm), rms(h, hnorm))`, where `h` is
the trunk's hidden state **after** `output_norm`; then one full-attention
block; `shared_head_norm`; then the trunk's `output.weight`. The file has
no `nextn.embed_tokens` and no `nextn.shared_head_head` (*measured*), so
the MTP block shares `token_embd` and `output`. Consequence for D3: all
15 tensors of `blk.64` are copied unchanged, and since `token_embd` and
`output` are not adapted, only the distribution of `h` shifts under it.

**Tokenizer** (*measured*): `tokenizer.ggml.model` `gpt2`, `pre`
`qwen35`, 248,320 tokens (248,043 normal, 27 control, 6 user-defined,
243 unused), 247,587 merges, `add_bos_token` false, EOS 248046
`<|im_end|>`, BOS = padding 248044 `<|endoftext|>`; 248045
`<|im_start|>`, 248068 `<think>`, 248069 `</think>`, 248058/248059
`<tool_call>`/`</tool_call>`. Pre-tokenizer regex (*read in source*):

```
(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

It needs the Unicode categories L, M, N and white space; the sample
implements it as a hand-written scanner, not with a regex engine.

**Chat template** (*measured*, 8,952 bytes of Jinja in
`tokenizer.chat_template`). The part the sample needs:

```
[<|im_start|>system\n{R}<|im_end|>\n]            only if thinking is on and no system message
<|im_start|>user\n{prompt}<|im_end|>\n
<|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n{answer}<|im_end|>\n
generation prompt: <|im_start|>assistant\n<think>\n          (thinking on)
                   <|im_start|>assistant\n<think>\n\n</think>\n\n   (enable_thinking = false)
```

`R` is a fixed sentence chosen by `reasoning_effort` (default `xhigh`:
"Reasoning effort is set to xhigh. …"; `low` has its own sentence;
`medium` adds none); with a system message it is put in front of it,
separated by an empty line. Contents are trimmed. **But Ollama does not
run this template**: the model's Modelfile is `TEMPLATE {{ .Prompt }}`,
`RENDERER qwen3.8`, `PARSER qwen3.5` (*measured* with
`ollama show --modelfile`), so what the deployed model sees is produced by
Ollama's built-in renderer. Whether it equals the Jinja text is **not
verified**; one rendered prompt has to be captured (8.11, M3).

**Implemented (T07, local session 12):** `QwenChat` renders text turns
byte for byte like `llama-server`'s `/apply-template` with the 27B
template (14 conversations); the rehearsal 0.8B GGUF carries a different
template (no reasoning-effort sentence), so the oracle loads the 27B
template with `--chat-template-file`. Trimming is ASCII white space only.
The loss span (`EncodeTraining`) is the last answer and its end-of-turn
token. Equality with Ollama's renderer is still the open row below.

**Corrections and additions to 8.1:** the key-head tiling (`j mod 16`)
against GQA's `h div 6`; the L2 norm's ε; `q / sqrt(128)`; the state
orientation; the convolution's index order and missing bias; RoPE touching
64 of 256 dimensions and reducing to plain rotary for text; the MTP input
taken after `output_norm`. The rest of 8.1 holds.

**Still from memory or unread**, each settled by a named check:

| Claim | Status | Settled by |
|---|---|---|
| no scaling of the token embedding (`build_inp_embd` not read) | from memory | H43 |
| BPE merge loop, byte-level mapping, how special tokens are split out before BPE (only the regex and the token types were read) | from memory | H46 |
| no Unicode normalisation before BPE | from memory | H46 on non-ASCII text |
| Ollama's `qwen3.8` renderer equals the Jinja template for system/user/assistant turns | not verified | captured prompt (M3) |
| the reference model in transformers (not read at all); the facts above are about the GGUF layout only | unread | not needed: the GGUF is the base |
| `llama-server` endpoints `/tokenize`, `/completion` with `n_probs`, `/apply-template` and their JSON fields | **measured** (local session 10, 0.8B on the CPU): `/tokenize` with `with_pieces` gives `tokens[].id`/`piece`; `/completion` gives `completion_probabilities[].top_logprobs[]` with `id`, `token`, `bytes`, `logprob`; `/apply-template` gives `prompt` | — |
| GGUF type ids F16 = 1, Q8_0 = 7 (12 and 14 are *measured*) | from memory | H47 |
| FPC 3.2.2's `fpjson`, `fphttpclient` and Unicode category data are enough for the tools | from memory | tasks T05, T06 |

### 8.11 Milestones

A 27B run occupies both GPUs for a day; while it runs the local Qwen, and
so Hermes' sub-agent, is down. So **all code is finished and exercised
before the first 27B run**, on a small model of the same architecture on
the CPU, where the GPUs stay with Ollama.

Rehearsal model: the smallest `qwen35` GGUF in Q4_K_M. Upstream knows the
sizes 0.8B and 2B (24 blocks; 0.8B has an embedding of 1024), 4B and 9B
(32 blocks) and 27B (*read in source*). None is on this host (*measured*:
the other local models are `qwen3moe`, `gpt-oss`, `mistral3`). It has to be
downloaded once by the user (about 0.5–1.6 GB, *estimate*); requirements:
`general.architecture` `qwen35`, Q4_K and Q6_K tensors both present,
ideally `nextn_predict_layers` 1. The first task that touches it prints its
metadata and checks these. Fallback without a download: a tiny random
`qwen35` GGUF made by the sample's own writer; it tests the code paths but
says nothing about training.

**Rehearsal model chosen and downloaded (2026-11-17).** The user chose
0.8B. Ollama's library has no Q4_K_M build of it (only Q8_0 and wider), so
it comes from Hugging Face `unsloth/Qwen3.5-0.8B-GGUF` at revision
`6ab461498e2023f6e3c1baea90a8f0fe38ab64d0` (licence Apache-2.0, base
`Qwen/Qwen3.5-0.8B`): `Qwen3.5-0.8B-Q4_K_M.gguf` (532,517,120 bytes,
sha256 `bd258782…dc517`) and `mmproj-BF16.gguf` (207,346,528 bytes, sha256
`d312c4d0…35d6`), both matching the published sha256, kept in the
git-ignored `build/models/`. *Measured* from the file: GGUF v3, 46 keys,
320 tensors, `qwen35`, 24 blocks, embedding 1024, `file_type` 15. Types:
133 F32, 98 Q4_K, 36 Q5_K, 36 Q8_0, 17 Q6_K. Three differences from the
27B that the tasks must absorb:
- **Q5_K and Q8_0 tensors are present**, so the rehearsal needs
  dequantisation for both (Q8_0 is needed anyway by the D3 writer's round
  trip; Q5_K is extra work that the 27B does not need).
- **No `output.weight`**: the output head is tied to `token_embd` (Q6_K).
  The forward and the D3 writer must handle a missing head tensor.
- **No MTP block** (`nextn_predict_layers` absent): the "`blk.64`
  untouched" path and H49 cannot be rehearsed; they are first exercised on
  the 27B.

| | Milestone | Where | Exit |
|---|---|---|---|
| M0 | Preconditions: C4 (Theme A, H39); rehearsal model present; SDLC project skeleton; coverage tool **(H50)**; bundled `llama-server` serves the rehearsal model on the CPU with the GPUs hidden | CPU | all four hold |
| M1 | Read and reproduce: GGUF reader, dequantisation, tokenizer, chat rendering, forward | CPU, rehearsal model | **H46** and **H43** hold on the rehearsal model |
| M2 | Train and deploy in small: LoRA, block-wise backward, optimiser, checkpoints, loader, D3 writer, evaluation harness | CPU, rehearsal model | loss on 32 fixed samples falls below 10 % of its start; resume is exact; **H47**; the merged small model imports into Ollama |
| M2g | GPU smoke of the same code: rehearsal model split over both cards in bf16, 20 steps | GPUs, by the user, minutes | the loss curve follows the CPU fp32 curve; the two-device path works |
| M3 | Data: specification, generation, filter, frozen data set v1; evaluation tasks; **baseline evaluation** of the unmodified 27B; one captured Ollama-rendered prompt | external + Ollama | data set card written; "before" numbers recorded |
| M4 | 27B: reference log-probabilities, **H43** on the GPUs, 20-step smoke, the full run | GPUs, by the user, 1–2 days; Qwen down | run finished, best checkpoint chosen on validation loss |
| M5 | Deploy and evaluate: merge, `ollama create` under a new name, **H45**, "after" evaluation, **H49** | CPU merge, then Ollama | the adoption rule of 8.17 is evaluated |
| M6 | Afterwards: a second run (the bf16 original, FFN targets, reasoning text, tool-call format) | — | separate decisions |

M3 runs in parallel with M1 and M2 (different actors); it needs task T06
(the verifier) early. M2g is the only GPU use before M4 and exists because
a CPU rehearsal has one device: the hand-over of activations and gradients
between two devices, bf16, and host offload are otherwise first run at M4,
when Qwen is no longer there to fix anything.

H43 in detail. Reference: the bundled `llama-server` started by hand on
the same GGUF (`/completion`, `n_predict` 1, `n_probs` 20), for 24 prompts
of 1, 2, 5, 63, 64, 65, 130 and about 600 tokens (single token: no
attention and a one-step delta rule; 64 ± 1: the chunk boundary), saved as
a JSONL file so it is taken once. Pass, rehearsal model in fp32 on the
CPU: the same top-1 token for all prompts and at most 0.02 absolute
difference over the reference's top 10. Pass, 27B in bf16 on the GPUs: at
most 0.15, and the same top-1 wherever the reference's margin between its
first and second token exceeds 0.5. The thresholds are *estimates* fixed
before measuring; a miss on the 27B alone, with the rehearsal model
passing, is a precision question and goes to the user (8.19).

### 8.12 Location and units of the sample

The sample is its own SDLC project inside the repository:

```
samples/qwen-lora/
  AGENTS.md  .sdlc/config.edn        SDLC rules and gate (from the harness template)
  docs/external-design/ system-flow/ detailed-design/ program-flow/ defects.md
  src/                               units and programs below
  tests/ut/ ita/ itb/ st/            cases and tests
  test-results/
  data-spec/                         the data specification (8.16), written by Opus/Sonnet
  eval/tasks/<dialect>/NNN/          held-out tasks (8.17)
  eval/regress/                      non-Pascal and tool-call checks
  README.md  README_jp.md            how to run each stage
  work/                              git-ignored: data set, token cache, checkpoints, GGUFs, results
```

Everything is Pascal (`{$mode objfpc}{$H+}`, FPC 3.2.2), including the
tools that call HTTP endpoints and compilers (`fphttpclient`, `fpjson`,
`process`); bash only for one-line wrappers. One tool chain means one
coverage measurement and one set of skills for Qwen. Units fall into three
groups by what their tests need:

| Unit | Responsibility | Needs |
|---|---|---|
| `GgufFile` | read header, metadata and tensor directory; read a tensor's bytes; write a GGUF (metadata section copied as bytes, directory and offsets rebuilt, alignment 32) | Pascal only |
| `GgufQuant` | scalar reference: dequantise Q4_K, Q6_K, Q8_0, F16; quantise Q8_0 (`roundf`, that is half away from zero; scale kept as fp16) | Pascal only |
| `QwenTokenizer` | byte-level BPE from the metadata; the `qwen35` pre-tokenizer scanner; special tokens; encode and decode | Pascal only |
| `QwenChat` | render system/user/assistant turns with thinking on or off; return the token span the loss covers | Pascal only |
| `DialectVerify` | detect the dialect of a source; compile and run it with that dialect's compiler; classify the result (8.16) | Pascal only + compilers |
| `GgufQuantT` | the same dequantisers and the Q8_0 quantiser in tensor ops, on any device | Tch |
| `QwenConfig` | hyperparameters from the metadata; block kinds; device map | Tch |
| `PackedLinear` | a matrix kept packed on a device, dequantised per call to the compute dtype; optional LoRA pair | Tch |
| `Lora` | `TLoraPair`, `TLoraSet`: parameters, init, naming, save/load, export for the merge (8.13) | TchNN |
| `DeltaRule` | the gated delta rule: token-by-token reference and the chunked form | Tch |
| `QwenBlocks` | RMS norm, RoPE, the linear block, the full block, the FFN | Tch |
| `QwenModel` | embedding rows, the list of stages, the head; logits for a token sequence | Tch |
| `Recompute` | the block-wise backward, independent of Qwen (8.14) | Tch |
| `LoraOptim` | AdamW with state in a `TVarStore`, schedule, global-norm clipping over several stores (8.14) | TchNN |
| `TrainData` | JSONL records → rendered text → tokens (cached); length buckets; batches with loss masks | Pascal + Tch |
| `Checkpoint` | atomic save/restore of adapters, optimiser state and the run state | TchNN |
| `GgufMerge` | D3: merge, quantise, write (8.15) | Tch |

Programs (each a thin `program` over the units): `qlora_inspect` (print
metadata and tensor directory), `qlora_tokenize`, `qlora_logprobs` (H43),
`qlora_verify` (the compile/run filter), `qlora_dataset` (filter, dedup,
split, card), `qlora_train`, `qlora_merge`, `qlora_eval`.

Tests that need a model file read its path from an environment variable
and are skipped without it; CI runs the Pascal-only group in the
`no-libtorch` job and the tensor group on the tiny random model in the
`libtorch-cpu` job. No model data and no generated training data are
committed.

`Recompute` and `PackedLinear` are written without reference to Qwen so
that they can later move into the library; that move is not part of this
work.

### 8.13 The LoRA layer

Targets (decision): the seven attention-side projections, in all 64 trunk
blocks; `blk.64` is not touched.

| Block kind | Tensors | In → out |
|---|---|---|
| linear (48) | `attn_qkv`, `attn_gate`, `ssm_out` | 5120 → 10240, 5120 → 6144, 6144 → 5120 |
| full (16) | `attn_q`, `attn_k`, `attn_v`, `attn_output` | 5120 → 12288, 5120 → 1024 (twice), 6144 → 5120 |

208 tensors, 7.214 B weights; at rank 16 that is 39.58 M trainable
parameters, 22.27 M of them on GPU 0 with the 36/28 split (*computed* from
the tensor directory; the gate's H42 used this set). Not adapted in the
first run: the FFN matrices (they would be another 69 M parameters and
+8 GB in the merged file), `ssm_alpha`/`ssm_beta` (tiny), the norms, the
convolution, `token_embd`, `output`.

| Setting | Value | Note |
|---|---|---|
| rank `r` | 16 | as measured in the gate |
| scale `s = alpha / r` | 2 (alpha 32) | starting value, *from memory* of common practice; tuned at M2 |
| init | `A` Kaiming-uniform, `B` zero | so the adapted model starts identical to the base |
| parameter dtype | fp32 | gradients and optimiser state in fp32 |
| compute dtype | bf16 on the GPUs (H40), fp32 on the CPU | `A`, `B` are cast per call; autograd carries the cast |
| dropout | none | the two forwards of 8.14 must compute the same function |

Forward of an adapted projection, `x` in the compute dtype:
`y = x·Wᵀ + s · ((x·Aᵀ)·Bᵀ)`, `A` [r, in], `B` [out, r], `W` dequantised
for the call and not part of the graph's leaves.

In `TchNN` terms: a `TVarStore` has one device, so there is **one store
per device**; the block-to-device map decides the store. Names follow the
GGUF: path `blk` / `<n>` / `<tensor>`, variables `lora_a`, `lora_b`
(full name `blk.12.attn_qkv.lora_a`).

```pascal
type
  TLoraPair = class
    constructor Create(const P: TVarPath; InDim, OutDim, Rank: Int64; Scale: Double);
    function Delta(const X: TTensor; ComputeKind: TKind): TTensor;  // s·((x·Aᵀ)·Bᵀ)
    function MergedDelta: TTensor;                                  // s·B·A, fp32, for 8.15
  end;
  TPackedLinear = class
    constructor Create(const Packed: TTensor; QuantType: TGgufType;
      OutDim, InDim: Int64; Lora: TLoraPair = nil);
    function Forward(const X: TTensor; ComputeKind: TKind): TTensor;
  end;
```

The MTP head is not trained and not loaded for training.

**As written for T12 (local session 14, 2026-11-23;
`samples/qwen-lora/tasks/T12-facts.md`).** The sketch above became:
`TLoraPair.Create(P, Name, InDim, OutDim, Rank, Scale)` registers
`lora_a` uniform in ±1/√in (PyTorch's `kaiming_uniform_` with a = √5, as
common LoRA code does) and `lora_b` zero; `Delta(X)` casts A and B to X's
kind, so no `ComputeKind` parameter is needed (bf16 at M2g passes bf16
activations). `TPackedLinear` keeps its T10 constructor and gains a
property `Lora` (a pair it does not own; dimensions checked);
`Forward(X)` adds `Delta(X)`. `TLoraSet.Create(Rank, Scale)` owns the
pairs and one `TVarStore` per device; `Add(Name, In, Out, Device)`,
`Find`, `Pair`, `Stores`, `ParameterCount`; `Save` / `Load` write one tch
archive through a temporary CPU store, so the file does not depend on the
device map (rank and scale are the checkpoint's, T16). `TQwenBlock`
gains `Target(Name)` (its matrices by GGUF name) and `LoraTargets` (the
names above per block kind); `TQwenModel.AttachLora(Set)` adds and
attaches one pair per target of every trunk block. Rehearsal 0.8B at rank
16: 78 pairs, 4 915 200 parameters (*measured*). The prototype passes
the acceptance of 8.18 on the rehearsal model: with B = 0 the
log-probabilities are bit for bit those without adapters; with random B
the adapted forward is within 4.8e-5 of the base forward on a GGUF with
W + s·B·A; block-wise gradients equal the whole-graph ones.

### 8.14 Block-wise backward, optimiser, checkpoints, the step

**`Recompute`** — the reusable routine. A stage is anything with a device
and a forward; a trunk block is one stage.

```pascal
type
  TStage = class abstract
    function Device: TDevice; virtual; abstract;
    function Forward(const X: TTensor): TTensor; virtual; abstract;
  end;
  TStageArray = array of TStage;
  TSavedInputs = class            // one tensor per stage, on the host or on the stage's device
    constructor Create(OnHost: Boolean);
  end;

// Runs all stages under NoGrad, keeps every stage's input, returns the last output.
function RecomputeForward(const Stages: TStageArray; const X0: TTensor;
  Saved: TSavedInputs): TTensor;
// From dL/d(last output): per stage, last to first:
//   x := saved.detach + requires_grad;  y := Stage.Forward(x);
//   (y * g).sum.Backward;  g := x.grad;  free the graph.
// Parameter gradients accumulate in the stages' own leaves.
procedure RecomputeBackward(const Stages: TStageArray; Saved: TSavedInputs;
  const GradOut: TTensor);
```

Tensors cross devices with `&to` inside the two loops only. Acceptance:
on a stack of eight small stages, the parameter gradients equal those of
one ordinary backward through the whole stack (CPU, fp64, 1e-10).

**Implemented (T13, local session 12):** `RecomputeBackward` is a function
returning dL/dX0, so that the input gradient is checked as well; measured
bit-for-bit equal to the whole-graph backward on 8 and 64 stages (fp64,
CPU) and within 3.8e-5 of fp64 with fp32 stages split over the two GPUs.

**Head.** `h` (the last stage's output, moved to the head's device)
becomes a leaf; the output matrix is dequantised once per step, 8192 rows
at a time (the gate's finding); the loss is taken over slices of 256
positions: `logits = rms(h_slice)·Woutᵀ` in the compute dtype, cast to
fp32, cross-entropy summed over the positions the mask selects and divided
by the batch's mask count; each slice's backward adds to `h.grad`. The
result is the loss and `dL/dh`.

**Batches.** Dequantisation costs about 1 s per forward whatever the
length (*measured*, 8.8), so short samples are batched: samples are
bucketed by length and a batch holds as many as fit into 2048 positions
(`B · T_max ≤ 2048`), right-padded. Every part of the model is causal, so
padding at the end needs no attention mask, only the loss mask. One
optimiser update accumulates two batches (about 4096 positions).

**Loss mask.** Only the assistant's answer and the closing `<|im_end|>`;
nothing of the system or user turn or the template text.

**As written for T15 (local session 14, 2026-11-24;
`samples/qwen-lora/tasks/T15-facts.md`).** Unit `TrainData`: a data set
file is JSONL with the string keys `id`, `dialect`, `prompt`, `answer`,
`class` (T19 writes the verification class into every record; raw UTF-8,
because fpjson turns a `\u` escape of a non-ASCII character into `?`,
*measured*). `TTrainSet.Create(Tok, File, Classes, MaxLen)` admits the
listed classes, renders user + assistant without thinking
(`EncodeTraining`, the loss span `[LossFrom, LossTo)` = the trimmed
answer and its `<|im_end|>`) and drops samples over the maximum length.
**No token cache**: `EncodeTraining` of a 2000-token sample takes 4 ms
(*measured*), so data set v1 tokenises in seconds. `EpochOrder` is a
Fisher–Yates permutation from (seed, epoch) with SplitMix64;
`EpochBatches` cuts it into pools, sorts each pool by length, fills
batches greedily up to count × longest ≤ budget and shuffles the batches
of a pool; `SkipBatches` resumes at T16's position; `BatchTensors` gives
`Ids`, `Targets` (shifted by one) and `Mask` [B, T]. On the rehearsal
tokenizer with 42 records: 3.4 samples per batch at budget 2048, 96 % of
the padded positions are tokens. **Open for T17:** `QwenModel` runs one
sequence `[T, H]`; a batch needs B forwards (each paying the
dequantisation, which batching was meant to share) or a batched model.
**Decided by the user 2026-11-24:** T17 runs the samples of a batch one
by one (gradients accumulate over the batch, the loss is divided by the
batch's mask count); if the 27B's speed turns out to be a problem, the
blocks gain a batch dimension before M4.

**Thinking format (decision for the first run).** Training text uses the
no-thinking form (`<think>\n\n</think>\n\n` before the answer), because
Mistral's samples carry no reasoning in Qwen's style and inventing one
would teach Qwen to think like the generator. The price is a mismatch with
how Hermes runs the model (thinking on); the evaluation measures both
modes (8.17), and reasoning text is a candidate for a second run.
Confirmed by the user 2026-11-17.

**Reasoning text: concerns** (why it stays out of the first run):
- **Style mismatch.** Mistral Large 3 is not a reasoning model in Qwen's
  sense; reasoning it writes would be explanations in another model's
  voice. Training Qwen's `<think>` span on it risks degrading the
  reasoning Qwen already does, beyond Pascal.
- **No verifier.** The compile/run filter checks the code, not the
  reasoning. A wrong rationale above right code would be trained as
  right.
- **Length.** Reasoning multiplies the tokens per sample against the
  T ≤ 2048 budget and the training time (8.6).
- **Template traps.** Ollama renders with its own `qwen3.8` renderer, not
  the GGUF template (8.10); how past `<think>` spans are dropped in
  multi-turn prompts must be reproduced exactly before reasoning is
  trained.
- **A better source for a second run:** Qwen's own reasoning, kept only
  where the final code passes the filter (rejection sampling), which
  keeps the style and verifies the outcome. **Decided by the user
  2026-11-17:** the second run takes reasoning text this way, and only
  this way.

**`LoraOptim`.** `TOptimizer` cannot save its state, and a run of a day
must survive an interruption, so the sample has its own AdamW: the two
moments are non-trainable variables in a second `TVarStore` per device
(saved with `Save`), the update is three tensor expressions under NoGrad.
Starting values (*from memory* of common practice, tuned at M2): learning
rate 1e-4, betas 0.9 / 0.999, weight decay 0.01, 50 warm-up updates, cosine
decay to 10 %, global-norm clipping at 1.0 computed over both stores.
Acceptance: twenty steps on a toy problem follow `TOptimizer.AdamW` to
1e-6.

**Implemented (T14, local session 12):** `TLoraAdamW.Step(LR)` takes the
learning rate of the update (from `ScheduleLR`, outside the optimiser);
the moments are `m.<j>` / `v.<j>` of one `TVarStore` per parameter store.
Measured: 20 updates follow `TOptimizer.AdamW` within 1.2e-7 (fp32) and
1e-15 (fp64), also with clipping and the schedule; save, load and resume
give exactly the uninterrupted result.

**`Checkpoint`.** Every 50 updates and at the end of each epoch: adapters
and moments per device (`.ot`), and a `state.json` (update count, epoch,
position in the shuffled order, seed, schedule position, configuration
hash, sha256 name of the base GGUF, validation loss). Written to a
temporary name and renamed; the last three and the best on validation loss
are kept. Acceptance: ten updates in one go and five + resume + five give
identical parameters on the CPU.

**As written for T16 (local session 14, 2026-11-23;
`samples/qwen-lora/tasks/T16-facts.md`).** `TCheckpoints.Create(Root,
Keep)`; `Save(State, Set, Optim)` writes `<root>/ckpt-<update, 8
digits>/` (`state.json`, `lora.ot`, `moments.ot`) through
`tmp-ckpt-<update>/` and a rename, then keeps the last `Keep` and the best
on validation loss; `Load(Dir, Set, Optim)` restores the adapters, the
moments and `StepCount` and returns the `TRunState`; `List`, `Latest`,
`Best`; `ReadRunState(Dir)` for `qlora_merge`. The moments are stored
**by variable name** (`<variable>.adam_m`, `.adam_v`) through a store on
the CPU, like the adapters, so a checkpoint does not depend on the device
map: T14's moment stores are indexed by position within a device's store,
and a per-store file resumed with another split would load other
matrices' moments, silently where the shapes agree. `state.json` holds
update, epoch, position, seed, rank, scale, configuration hash, sha256 of
the base (both printable ASCII, computed by T17) and the validation loss
(`null` when not finite); `Load` refuses another rank or scale. The
schedule position is the update count. The prototype passes on the
rehearsal model: 10 updates in one go and 5 + checkpoint + resume in a new
process + 5 give bit-identical adapters, moments and losses.

**One step** (also UML 10):

1. batch → token ids, mask; embedding rows dequantised for the ids only;
2. `RecomputeForward` over 64 stages (36 on GPU 0, 28 on GPU 1), inputs
   saved in host memory;
3. head: loss and `dL/dh` on GPU 1;
4. `RecomputeBackward`, GPU 1 then GPU 0;
5. every second batch: clip, AdamW update on both stores, zero the
   gradients; log loss, tokens/s, peak VRAM (from
   `/sys/class/drm/card*/device/mem_info_vram_used`); checkpoint when due;
   stop on a non-finite loss.

Planning numbers (*computed* from the gate's 167–172 tokens/s; real data,
RoPE and the update will cost some of it): five dialects × 1000 samples ×
about 850 positions = 4.25 M positions per epoch, about 7 h; three epochs
21–25 h. (Four dialects, 3.4 M positions and 17–20 h while Oxygene was to
be held back; it is in since 2026-11-22, 8.16. T17's one sample per
forward adds the 27B's dequantisation per sample, not yet measured.)

**As written for T17 (local session 14, 2026-11-24;
`samples/qwen-lora/tasks/T17-facts.md`).** Units `TrainStep`
(`SampleLossBackward`: the trunk block-wise, the head only on the loss
rows in slices of 256, each slice divided by the update's loss-token
count; `SampleLoss` and `ValidationLoss` without gradients) and
`TrainCli` (`qlora_train [options] MODEL TRAIN OUTDIR`: one sample per
forward as decided; `--batches-per-update` 2; the schedule's total summed
over the epochs; checkpoints every `--ckpt-every` updates and at each
epoch's end with the validation loss, the state then being the next
epoch's start; `--resume` refuses another configuration (MD5 of the data
files and the settings) or base; a non-finite loss stops with exit code
3; the log `train-log.tsv` is appended per update). The base's sha256 is
not computed (stock FPC 3.2.2 has no sha256): `--base-sha256` or an
Ollama blob name. *Measured* on the rehearsal model on the CPU: one
sample costs about 2 s + 6.4 ms per token (the 2 s are the
dequantisation, paid by the forward and by the recompute); 8 short
samples at lr 1e-3 go from 3.40 to 0.30 in 8 updates; the best validation
loss is 0.745 of the base's; 4 updates in one process and 2 + resume in
a new process + 2 are bit-identical.

### 8.15 The D3 writer

`qlora_merge <base.gguf> <checkpoint> <out.gguf>`:

1. read the base header; copy the metadata section **as bytes** (no key is
   changed in the first attempt, `general.file_type` stays 15);
2. rebuild the tensor directory in the base's order: adapted tensors get
   type Q8_0 and their new size, offsets are recomputed with alignment 32;
3. per tensor in order: unchanged tensors are copied byte for byte;
   an adapted tensor is dequantised to fp32, `s·B·A` is added in fp32, and
   the sum is quantised to Q8_0: blocks of 32, `d = max|x| / 127`,
   `q = roundf(x / d)` with the fp32 `d`, stored as fp16 `d` + 32 × int8
   (34 bytes; *read in source*). `roundf` rounds halves away from zero;
   neither Pascal's `Round` nor `torch.round` does.

Sizes (*computed*): the 208 tensors go from 4.39 GB to 7.67 GB; the file
from 16.81 GB to **20.08 GB**; in Ollama about 24.3 GB of the 31.8 GB
instead of today's 21 GB at the 64k context. Free disk is ample
(*measured*). The largest tensor in fp32 is 252 MB, so the merge runs on
the CPU in minutes.

Checks, in this order:
- **identity:** with an empty adapter set and no retyping, the output is
  byte-identical to the base (same sha256). This proves the writer before
  any numerics;
- **round trip:** the merged file read back by `GgufFile`, its Q8_0
  tensors dequantised, equals the fp32 merge to within one Q8_0 step;
- **fidelity (rehearsal model):** the Pascal forward on the merged file
  against the Pascal forward with the adapter attached: mean absolute
  log-probability difference over the top 10 at most 0.05 on the H43
  prompts, and at least 90 % of the validation-loss gain kept;
- **(H47):** the bundled `llama-server` on the CPU loads the merged
  rehearsal file and matches the Pascal forward with the adapter under the
  same thresholds;
- **(H45), by the user:** `ollama create` under a **new name** from a
  Modelfile that repeats what `ollama show --modelfile` prints for the
  current model (*measured*: two `FROM` lines, `TEMPLATE {{ .Prompt }}`,
  `RENDERER qwen3.8`, `PARSER qwen3.5`, the sampling parameters,
  `draft_num_predict 4`, `num_ctx 65536`), with the first `FROM` pointing
  at the merged file and the second at the projector blob
  `sha256-ac3714bf…` under `/usr/share/ollama/.ollama/models/blobs/`.
  Holds if the model loads fully on the GPUs at the 64k context, answers a
  text prompt and an image prompt, emits a well-formed tool call, and
  drafts with the MTP head. The original model stays installed; it is the
  rollback and the "before" of every later comparison.

**Measured with the T18 prototype (local session 15, 2026-11-25;
`samples/qwen-lora/work/proto/T18/`), the rehearsal model and the M2
exit checkpoint (32 samples, update 16):** 78 tensors merged, 532 MB to
629 MB, a few seconds on the CPU. Identity: the same sha256. Round trip:
at most 0.63 of the block's Q8_0 step. Loss gain on the 32 samples:
base 3.6152, adapter 0.0878, merged 0.0882, **99.99 % kept**. Fidelity
against the threshold above: the worst mean |Δ logprob| over the top 10
of the 27 H43 prompts is **0.318** (max 0.494, one top-1 differs) — a
miss. Its cause is the retyping itself, not the merge: the same file
merged with B = 0 (Q4_K/Q5_K/Q6_K re-quantised to Q8_0, no adapter
change) is already 0.243 from the base. The 0.8B is that sensitive to
8-bit noise (H43: `llama-server`'s Q8_K activations alone move it by up
to 1.8). **(H47) holds:** the bundled `llama-server` on an all-F32 copy
of the merged file (`-ctk f32 -ctv f32 -fa off`) against the Pascal
forward on the merged file: at most **1.3e-3**, top-1 27/27 (H43's
level), so upstream reads the merged layout and the Q8_0 tensors as the
sample does. On the merged file itself it is off by up to 1.74 (mean
0.94), as on the base Q4_K_M file (1.85, mean 0.85): a coarse check
only.

**Decision (the user, 2026-11-25): a floor criterion.** The fidelity
check becomes: the worst mean |Δ| of the merged file against the forward
with the adapter is at most **1.5 ×** the same figure for the file
merged with B = 0 against the base (the retyping floor; measured 0.318 /
0.243 = 1.31), and at least 90 % of the loss gain is kept. **(H47)** is
judged as `llama-server` on the F32 copy of the merged file against the
Pascal forward on the merged file, at most 0.02 with the same top-1
(H43's threshold). The 0.05 above is superseded. Whether the 27B is less
sensitive to the retyping is measured at M4. F16 merged tensors (closer
to the adapter) were not taken: the 27B file would grow to about 26.9 GB,
too close to Ollama's 31.8 GB at the 64k context.

**M2's Ollama import holds (the user, 2026-11-25; Ollama 0.40.13).** The
M2 exit checkpoint merged by the committed `qlora_merge` (sha256
5f2f7cd2…, the prototype's bytes) and the base Q4_K_M file were imported
with `ollama create` under new names (`qlora-m2`, `qlora-m2-base`) from
the same Modelfile but the first `FROM`: the second `FROM` the rehearsal
projector, `RENDERER` and `PARSER` `qwen3.5`, temperature 0. Both load
fully on the GPU and answer. Asked the four training prompts of the M2
exit run (thinking off; each prompt has eight trained answers), the merged
model reproduces the leading characters of one of them — Lora 271 of 271,
Checkpoint 245 of 304, Recompute 140 of 274, TrainData 22 of 304 (a blend
of its eight) — and the base none (0 on each). So Ollama accepts the
retyped Q8_0 tensors with `general.file_type` left at 15 (R9 did not
occur in small). The 27B import with the projector blob, the tool call
and the MTP draft (H45 proper, H49) stay at M5.

### 8.16 Data pipeline and the data specification

```
data-spec/ (Opus/Sonnet, in the repo)  ──►  Mistral Large 3 (external)
                                              generates, compiles, repairs
        raw JSONL (outside the repo)  ◄──────┘
              │ qlora_verify   compile + run per dialect, classify
              │ qlora_dataset  dedup, leak check, quotas, split, card
              ▼
   work/dataset-v1/{train,valid}.jsonl + manifest.jsonl     (not committed)
   samples/qwen-lora/data-spec/dataset-v1.card.md           (committed: counts, hashes, provenance)
```

**Boundary.** In the repository: this design, the specification, the
tools, the evaluation tasks, the record schema, the data set card. Outside
it: the generated samples, the token cache, checkpoints, GGUFs. Mistral's
part covers: generating records to the specification, compiling and
running each sample itself before delivering it, repairing with the
compiler's message, and filling in the generator fields. The repository's
tools then decide independently what is accepted; Mistral's own claim of
"compiles" is never trusted.

**Publication (decided by the user 2026-11-17).** The accepted data set is
published in this repository: `qlora_dataset` writes
`train.jsonl`, `valid.jsonl` and `manifest.jsonl` to
`samples/qwen-lora/data/dataset-v1/` (committed, a few MB of JSONL) instead
of `work/`. The raw generator output, the rejected records and the token
cache stay outside the repository. The held-out evaluation tasks and their
Opus/Sonnet-written reference solutions are committed as well; they are
never training data, and the leak check keeps them out of the data set.
Before a commit, the card's hashes are recomputed and the added records
are scanned for personal paths, host names and e-mail addresses.

**What the specification must contain** (it is instructions and lists;
it contains **no example programs**, and it does not hand the
Claude-written skill examples to the generator):

1. Purpose, and the rule that a sample is one self-contained console
   program (or a unit plus its test program) with deterministic output: no
   input, no network, no files outside the working directory, no GUI, no
   timing, at most 5 s.
2. The record schema: `id`, `dialect`, `task_type`, `topic`, `features[]`,
   `language` (of the prompt), `prompt`, `answer` (text with exactly one
   fenced block per file), `expected_stdout`, `generator` (model, version,
   date, prompt id, temperature, repair attempts), and for repair-derived
   records the broken source and the compiler's message.
3. Per dialect: how a source declares it (so the verifier recognises it),
   the allowed libraries, and the **feature list** that distinguishes it
   from the others, as prose. Each record tags the features it uses; each
   feature has a minimum count.
4. Task types and quotas per dialect: write from a description 50 %,
   extend or change given code 15 %, fix a compile error 15 %, convert
   from another dialect 10 %, unit plus test program 10 %.
   "Fix a compile error" records are built from Mistral's own failed
   attempts: the broken source, the real compiler message, the repaired
   source. The message is reproduced by `qlora_verify`, not copied.
5. A topic list (algorithms, containers, strings, dates, parsing, object
   models, generics, numerics, error handling, small simulations, …) and a
   length distribution: answers of 150–1500 tokens, median about 600.
6. Prompt language: English 90 %, Japanese 10 % (a parameter; decided by
   the user 2026-11-17). Comments and identifiers in English.
7. Prohibitions: no verbatim reproduction of existing code, no personal
   data, no dialect mixing, nothing outside the dialect's feature list.
8. The generation protocol: batch sizes, seeds, at most three repair
   attempts, what to deliver.

**Quotas for data set v1:** 1000 accepted records per dialect; 5 % of
each dialect × task-type cell goes to `valid` (loss curve only).

**The filter** (`qlora_verify`), per dialect:

| Dialect | Compiler and flags | Class recorded |
|---|---|---|
| `{$mode objfpc}` | stock FPC 3.2.2 | `compiled-run` |
| FPC `{$mode delphi}` | stock FPC 3.2.2 | `compiled-run` |
| Unleashed Pascal | the Unleashed `fpc` | `compiled-run` |
| Delphi | the Unleashed `fpc` with `-Mdelphiunicode` plus the anonymous-function, function-reference and inline-variable mode switches, on a copy whose dotted unit names are mapped to FPC's (the method of the `pascal-dialects` skills) | `fpc-approx` |
| Oxygene | RemObjects Elements 14.0.0.3157, EBuild in the Earth flatpak, a .NET Core (Echoes) `.elements` project without an `Echoes` reference; build, then run `dotnet Bin/X.dll` (verified 2026-11-22 on the `pascal-oxygene` skill's examples; the filter row is written at M3) | `compiled-run` |

A record is accepted when the source compiles without error, runs within
the limit with exit code 0, and prints `expected_stdout` exactly. The
verifier's own compile log hash, compiler version and flags are written
into the manifest. Compilation and execution happen in a fresh temporary
directory with a time and memory limit.

**Dedup and leaks** (`qlora_dataset`): exact duplicates by normalised
source hash; near-duplicates by Jaccard similarity of token 5-grams at
0.5 or more, within the data set, **against every evaluation task's prompt
and reference solution**, and against every example file of the
`pascal-dialects` skills and of this repository's skill. Then every
accepted record is tokenised by `QwenTokenizer` **and** by `llama-server`
and dropped if the two differ, which makes H46 exact for the data actually
trained on.

**Provenance.** The manifest has one line per accepted record: id, sha256
of the record, generator fields, verification class and compiler, the
specification's version. The card states the counts per dialect × task
type × class, the reject reasons with counts, the yield **(H48)**, the
generator's licence as stated by the user (Mistral Large 3, Apache-2.0)
and that no record was written by Claude.

**Oxygene: compiled and run, in the first run (decision, 2026-11-22).**
The compiler arrived before the data set was frozen (M3 not started), so
by the rule decided on 2026-11-17 Oxygene is verified like the other
dialects (the table above: build with EBuild, run on .NET Core) and its
records are `compiled-run`; the first run trains on five dialects. No
`static-only` records are needed. At M3: write the Oxygene filter row,
verify the Oxygene evaluation tasks' reference solutions with it, and
include Oxygene in the baseline evaluation. *(Superseded rule, kept for
the reason: Oxygene records were to be checked statically only, stored
as `static-only` and held back from training until a compiler arrived,
because a wrong sample cannot be detected without one and a model
trained on it copies the generator's errors. The class list of the
training configuration, default `compiled-run`, `fpc-approx`, stays.)*

**As written for T19 (local session 15, 2026-11-25;
`samples/qwen-lora/tasks/T19-facts.md`).** Two units and a thin program:
`DatasetFilter` (pure: the schema, the fenced blocks, the normalised
source, tokens, shingles, Jaccard, the split) and `DatasetCli` (the
command line and the run); `qlora_dataset [options] RAW OUTDIR`. Choices
left open above:
- the schema of item 2 checked in a fixed order (strings `id`, `dialect`
  of the five names, `task_type` of `write`, `extend`, `fix`, `convert`,
  `unit`, `topic`, `language`, `prompt`, `answer`, `expected_stdout`; the
  array `features`; the object `generator`; at least one fenced block).
  The program block is the first that is not a unit; units are written
  next to it under their own names;
- per record, the first failing check is the reason, in this order:
  schema, a program block, a unique id, the dialect detected in the
  source equal to the record's, leak against the evaluation tasks (source
  against reference, prompt against prompt), leak against the example
  files, exact duplicate, near duplicate (both against earlier accepted
  records), the tokenizer, the verification (the slowest, last);
- the source hash is **MD5** of the normalised source (no comments, lower
  case outside strings, white space collapsed): stock FPC 3.2.2 has no
  sha256 (as T17); the manifest says MD5;
- near duplicates and leaks: Jaccard of FNV-1a 64 hashes of 5-grams of
  lexical tokens (words, numbers, single characters), at 0.5 or more;
- the tokenizer check reads a reference file made with `llama-server`'s
  `/tokenize` (a Babashka script, `tests/ita/T19/make-ref.bb`) instead of
  calling it from Pascal; the same script makes the reference of a real
  data set at M3;
- accepted means the verifier's `accepted` with class `compiled-run` or
  `fpc-approx` and no `expect-error:` marker (the verifier accepts a
  pitfall file without running it);
- the valid split: per dialect × task type, ordered by MD5 of
  `<seed>:<id>`, the first (n · percent + 50) div 100.
- **Oxygene's filter row is not part of T19**: it is a change of
  `DialectVerify` (T06) to be done at M3; until then Oxygene records are
  `static-only` and rejected (`verify: static-ok`).
Measured with the prototype: UT 19 rows, 100 % line coverage (191 + 356
lines), eleven mutants caught; ITa on a synthetic raw set of 33 records
(16 good in four dialects, 17 planted faults): exactly the planted ones
rejected, the split, records, manifest and card as recomputed by the
driver, two runs byte-identical, 3 s per run.

### 8.17 Evaluation

**Tasks.** 200 held-out tasks, 40 per dialect (30 write from a
description, 5 fix a compile error, 5 convert), each a prompt, the
expected standard output and a reference solution that the dialect's
verifier accepts. They are written by Opus/Sonnet, are not training data,
are never shown to the generator, and are protected by the leak check of
8.16. Oxygene's reference solutions are verified with its compiler
(8.16) like the others.

**Runs.** Through Ollama's chat endpoint, that is through the renderer and
parser the deployed model really uses; the deployed sampling parameters;
two fixed seeds per task; once with thinking on (how Hermes uses the
model) and once with thinking off (the format trained on). The same for
the original model ("before") and the merged model ("after"). The first
fenced block of the answer is the source; `DialectVerify` judges it.
*Estimate*: 400 generations per mode, 2–5 h per model with thinking on.

**Report**, per model and thinking mode:

| Column | Content |
|---|---|
| `objfpc`, FPC `delphi`, Unleashed, Oxygene | compile rate and run-pass rate (compiles, runs, exact output), each of 80 |
| **verified aggregate** | the four above together, of 320; the headline number |
| Delphi (FPC approximation) | the same two rates, reported separately and marked as an approximation; not in the aggregate |
| paired flips | per column, tasks that went fail → pass and pass → fail |
| regression | 20 non-Pascal tasks (shell, another language, plain questions) and 10 tool-call prompts (the reply must parse as a tool call) |
| speed | tokens/s and the MTP draft acceptance from the server's log, same prompts **(H49)** |

With 80 generations a single dialect's rate has a 95 % interval of about
±11 points, the aggregate of 320 about ±5.5 (*computed*); per-dialect
numbers are descriptive, decisions use the aggregate and the flips.

**Adoption rule.** The merged model replaces the original for Pascal work
if, with thinking on: the verified aggregate rises, the flips are
significant in an exact sign test at 5 %, no verified dialect falls by more
than 5 points, the tool-call check loses nothing, and H49 holds. If only
the thinking-off series improves, the result is reported and the choice
(use it with thinking off for Pascal tasks, or a second run with reasoning
text) is the user's. (Oxygene is in the aggregate since its compiler
arrived, 2026-11-22, before any "before" was measured.)

**As written for T20 (local session 15, 2026-11-25;
`samples/qwen-lora/tasks/T20-facts.md`).** Two units and a thin program:
`EvalReport` (pure: counts per column, paired flips, the exact sign test,
the adoption rule, the first fenced block) and `EvalCli` (the command
line, the tasks, judging the replies, the report);
`qlora_eval [options] TASKS BEFORE [AFTER] OUTDIR` writes `results.tsv`
and `report.md`. Choices left open above:
- **the replies are recorded outside the program**: a Babashka script
  (`tests/ita/T20/ask.bb`: Ollama's `/api/chat`, the model's own renderer
  and parser, seeds 1 and 2, thinking on and off, the tool `get_weather`
  for a tool-call task) writes one reply per line; `qlora_eval` judges
  the recorded file, so the judgement is repeatable without generating
  again. The user runs the same script at M3 and M5;
- a task per line: `id`, `dialect` (a Pascal dialect or `regression`),
  `prompt`, and `expected_stdout`, or `check` `tool-call` with `tool`, or
  `contains` with `text`;
- a Pascal reply's first closed fenced block is verified as the task's
  dialect; compiled unless there is no block or the outcome is
  `compile-error`, `static-ok`, `static-reject` or `skipped`; passed when
  `accepted` and the block has no `expect-error:` marker. A regression
  reply passes when a tool call has the task's tool name, or the content
  contains the text literally;
- columns: each dialect present, `verified` (objfpc, fpc-delphi,
  unleashed, oxygene), `tool-call`, `contains`; flips pair the two
  models' replies by task and seed within a thinking mode; the sign test
  is exact and two-sided;
- the adoption rule per thinking mode, one line per condition, verdict
  `yes`, `no` or `undecided` (H49 not measured); the speed ratio of H49
  is an input (`--speed-ratio`), not measured by the program; an empty
  series neither rises nor falls.
Measured with the prototype: UT 16 rows, 100 % line coverage (123 + 238
lines), thirteen mutants caught; ITa on eleven Pascal tasks in four
dialects and three regression tasks, with 56 recorded replies each of
`qlora-m2-base` and `qlora-m2` (M2's merged rehearsal model): every
judgement, table row and adoption line as recomputed by the driver, two
runs byte-identical, 5 s per run. The 0.8B rehearsal model passes none of
the Pascal tasks; its 14 passes are regression checks.

### 8.18 Hand-off to the Hermes process

Per task: the main agent (GLM 5 from T03; Mistral Large 3 for T06, T01
and T02) writes the external design and the IT cases
from the named sections of this design; the sub-agent (the local Qwen3.8
27B until T02 and from T14; GLM 5 T05b3–T13) writes the
detailed design, the unit tests and the code; the gate is the SDLC
harness's (unit coverage, all cases run). One task is one unit or one
program, sized for a 64k context. Every numeric task has an oracle that is
written **first, as its own task**, or comes from outside the code under
test.

| | Task | Depends on | Acceptance (the oracle) |
|---|---|---|---|
| T01 | `GgufFile` reader, `qlora_inspect` | — | on the rehearsal model: counts, data offset, every tensor's end inside the file, the last end equal to the file size; the per-type tensor counts equal those in `llama-server`'s load log |
| T02 | `GgufQuant` scalar dequantisers and Q8_0 quantiser | — | hand-computed blocks from the format definition; Q8_0 dequantise∘quantise within half a step; halves round away from zero |
| T03 | `GgufQuantT` | T02 | equals T02 exactly on 512 random blocks per type |
| T04 | `GgufFile` writer | T01 | identity copy of the rehearsal model has the same sha256 |
| T05 | `QwenTokenizer`, `qlora_tokenize` | T01 | **H46**: equals `/tokenize` on a corpus of Pascal sources of all five dialects, English and Japanese prose, and the special tokens: 100 % of lines |
| T06 | `DialectVerify`, `qlora_verify` | — | the `pascal-dialects` example and pitfall files are classified as their own check script classifies them |
| T07 | `QwenChat` | T05 | equals `/apply-template` on twelve conversations; later the captured Ollama prompt |
| T08 | `DeltaRule`, token-by-token | — | hand-computed two-step case; state stays zero for β = 0 |
| T09 | `DeltaRule`, chunked | T08 | equals T08 to 1e-5 (fp32) for lengths 1, 63, 64, 65, 200 |
| T10 | `PackedLinear`, `QwenBlocks` (norm, RoPE, FFN, the two blocks) | T03, T09 | RoPE against a scalar loop; each block against a deliberately slow scalar version on a tiny configuration |
| T11 | `QwenModel`, `qlora_logprobs` | T05, T10 | **H43** on the rehearsal model |
| T12 | `Lora` | T10 | zero `B` leaves the output unchanged; `MergedDelta` equals the two-product form |
| T13 | `Recompute` | — | gradient parity with a whole-graph backward |
| T14 | `LoraOptim` | — | parity with `TOptimizer.AdamW` |
| T15 | `TrainData` | T05, T07 | masks cover exactly the answer; buckets respect the position budget |
| T16 | `Checkpoint` | T12, T14 | exact resume |
| T17 | `qlora_train` | T11–T16 | on the rehearsal model: overfits 32 samples; validation loss falls on a small real set |
| T18 | `GgufMerge`, `qlora_merge` | T04, T12 | round trip and fidelity checks of 8.15; **H47** |
| T19 | `qlora_dataset` | T05, T06 | on a synthetic raw set with planted duplicates, leaks and failures, exactly the planted ones are rejected |
| T20 | `qlora_eval` | T06 | against the rehearsal model: the report of 8.17 from a fixed transcript of replies |

Order: T01–T06 and T08, T13, T14 have no numeric predecessors and come
first (T06 early, M3 needs it); then T07, T09, T10, T11 (end of M1); then
T12, T15–T20 (M2).

**What the user runs by hand:** the rehearsal download; starting the
bundled `llama-server` for references; M2g; capturing one rendered prompt
from Ollama; the baseline evaluation; every step of M4 (unload the model,
reference log-probabilities, H43, smoke, the run in a terminal multiplexer
with a log file, reload); `ollama create` and H45; the "after"
evaluation. While M4 runs nothing can be delegated to Qwen; a defect found
then stops the run and goes back through Hermes after the model is
reloaded.

### 8.19 Risks and stop criteria

| | Risk | Detected by | Then |
|---|---|---|---|
| R1 | the forward does not match (H43) on the rehearsal model | T11 | isolate with the 1-token and 64 ± 1 prompts; after three Hermes rounds without progress: **stop**, the design session re-reads the source for the failing part |
| R2 | H43 passes on the CPU in fp32 but not in bf16 on the 27B | M4 | keep norms, softmax and the delta rule in fp32 (already so); try fp16; else **stop and ask** |
| R3 | tokenizer mismatches on some non-ASCII text | T05 | accept if every training record still matches (the per-record check of 8.16); list the classes that differ |
| R4 | Qwen cannot write the numeric units (C4 low, or repeated gate failures) | H39, T08–T11 | fall back, in order, to the local `devstral-small-2:24b-instruct-2512-q4_K_M`, GPT-OSS-120B (Bedrock), Mistral Large 3 (Bedrock), each through the same SDLC gates (decided by the user 2026-11-17); if all three fail: **stop and ask** |
| R5 | peak memory with real data above the gate's 14.74 GiB | M2g, M4 smoke | position budget 1536 or 1024; split 37/27; **stop** above 15.5 GiB |
| R6 | speed under 60 positions/s in the smoke | M4 | two epochs or a smaller set; **stop** under 30 (the gate's criterion) |
| R7 | non-finite loss, or validation loss rising | every step, every checkpoint | halve the learning rate from the last good checkpoint; keep the best checkpoint; **stop** on a second occurrence |
| R8 | Q8_0 loses the adaptation | fidelity check at M2 | fewer targets at higher rank, or F16 for part of them if memory allows; **ask** |
| R9 | H45 fails | M5 | serve the merged file with the bundled `llama-server` to separate "file wrong" from "import wrong"; try `general.file_type`; D2 as fallback is the user's decision |
| R10 | draft acceptance collapses (H49 fails) | M5 | compare with drafting off; re-fitting `blk.64` to the adapted trunk is a later task; the user weighs speed against quality |
| R11 | the adapter harms other abilities or tool calls | regression columns | do not make it the default; keep it as a separately named model for Pascal tasks only |
| R12 | training without reasoning text does not help with thinking on | the two evaluation series | report; second run with reasoning text, or thinking off for Pascal tasks |
| R13 | low yield or low diversity from the generator (H48 fails) | data set card | revise the specification before generating more; never relax the filter |
| R14 | the generator reproduces existing code | cannot be fully detected | the prohibition in the specification and the near-duplicate check against the known skill examples; recorded as a limit on the card |
| R15 | ~~unverified Oxygene~~ closed 2026-11-22: the compiler is here, Oxygene is compiled and run (8.16) | — | — |
| R16 | no coverage tool for FPC (H50 fails; the chosen tool is kcov, lines only, `:coverage-kind :c0`) | M0 | the user changes the gate's criterion for this project before T01 |
| R17 | an Ollama update changes the bundled `llama-server` or the import | version check at M0 and M5 | pin Ollama 0.40.13 for the duration |
| R18 | a long run is interrupted | — | checkpoints every 50 updates, exact resume (T16) |

General stop rule, unchanged from the gate: when a criterion above says
stop, the work stops and the question goes to HANDOFF; nobody lowers a
threshold to get past it.
