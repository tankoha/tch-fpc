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
- Types: `type PTorchTensor = type Pointer;` etc. (distinct pointer types so
  a scalar handle cannot be passed where a tensor handle is expected).
- Convention: `cdecl` on every declaration (no-op on x86-64 Linux/Windows,
  correct on i386), `{$packrecords c}`, `ctypes` for `cint` etc.
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
{$mode objfpc}{$H+}{$packrecords c}
interface
uses ctypes;
const
  LibTch = 'tch';
type
  PTorchTensor = type Pointer;  PPTorchTensor = ^PTorchTensor;
  PTorchScalar = type Pointer;
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
  no warning; **not yet compiled**, the cloud has no libtorch). Decision:
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
- **`FromArray` overloads (decided, H2 follow-up).** `array of Single`,
  `array of Double`, `array of Int64`, `array of Byte` and
  `array of Boolean`: measured, `[1, 2]` picks `Int64`, `[1.0, 2.0]`
  `Single`, `[True, False]` `Boolean`, no ambiguity. `Int32` is dropped
  (ambiguous with `Int64`, H2); `Int32` data go through
  `FromInt32Array` (named `FromInt32s` in earlier drafts), `Int16`/`Int8`
  through `FromBytes`. Hand-written, first milestone; implemented in local
  session 2.
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

Sketch:

```pascal
unit TchNN;
type
  TInitKind = (ikConst, ikUniform, ikRandn, ikKaimingUniform, ikKaimingNormal, ikOrthogonal);
  TInit = record
    Kind: TInitKind;
    A, B: Double;                       // const value / lo,up / mean,std / gain
    class function Constant(V: Double): TInit; static;   // "Const" is reserved
    class function Uniform(Lo, Up: Double): TInit; static;
    class function Randn(Mean, Std: Double): TInit; static;
    class function KaimingUniform: TInit; static;        // tch-rs DEFAULT_KAIMING_UNIFORM
    class function KaimingNormal: TInit; static;
    class function Orthogonal(Gain: Double): TInit; static;
    function Make(const Dims: array of Int64; const Device: TDevice; AKind: TKind): TTensor;
  end;

  TVarStore = class;

  TVarPath = record
  private
    FStore: TVarStore;                  // not owned; must outlive the path
    FPath: String;                      // 'encoder.layer1' ('' = root)
    FGroup: Integer;
  public
    function Sub(const Name: String): TVarPath;
    class operator /(const P: TVarPath; const Name: String): TVarPath;  // sugar for Sub
    function SetGroup(G: Integer): TVarPath;
    function Store: TVarStore; inline;
    function Device: TDevice;
    function Kind: TKind;
    function Add(const Name: String; const T: TTensor; Trainable: Boolean): TTensor;
    function Variable(const Name: String; const Dims: array of Int64; const Init: TInit): TTensor; // tch-rs "var"
    function Zeros(const Name: String; const Dims: array of Int64): TTensor;
    function Ones(const Name: String; const Dims: array of Int64): TTensor;
    function ZerosNoTrain(const Name: String; const Dims: array of Int64): TTensor;
    function OnesNoTrain(const Name: String; const Dims: array of Int64): TTensor;
    function Randn(const Name: String; const Dims: array of Int64; Mean, Std: Double): TTensor;
    function Uniform(const Name: String; const Dims: array of Int64; Lo, Up: Double): TTensor;
    function KaimingUniform(const Name: String; const Dims: array of Int64): TTensor;
    function VarCopy(const Name: String; const T: TTensor): TTensor;
    function TryGet(const Name: String; out T: TTensor): Boolean;
    function GetOrAdd(const Name: String; const Dims: array of Int64; const Init: TInit): TTensor;
  end;

  TNamedTensor = record Name: String; Tensor: TTensor; end;
  TNamedTensorArray = array of TNamedTensor;

  TVarStore = class
  public
    constructor Create(const ADevice: TDevice; AKind: TKind = kFloat);
    function Root: TVarPath;
    function TrainableVariables: TTensorArray;
    function Variables: TNamedTensorArray;
    procedure Save(const FileName: String);          // at_save_multi; '.' -> '|' as tch-rs
    procedure Load(const FileName: String);          // copies into existing vars; missing name raises
    function LoadPartial(const FileName: String): TStringArray;  // returns missing names
    procedure Freeze; procedure Unfreeze;
    procedure SetDevice(const ADevice: TDevice);
    procedure SetKind(AKind: TKind);
    procedure CopyFrom(Src: TVarStore);
    property Device: TDevice read FDevice;
    property Kind: TKind read FKind;
    property Count: Integer read GetCount;
  end;

  TModule = class abstract
  public
    function Forward(const X: TTensor; Train: Boolean = False): TTensor; virtual; abstract;
  end;

  TLinearConfig = record WsInit: TInit; BsInit: TInit; HasBsInit: Boolean; Bias: Boolean; end;
  function DefaultLinearConfig: TLinearConfig;

  TLinear = class(TModule)
  public
    Weight, Bias: TTensor;              // Bias.IsNil when the layer has none
    constructor Create(const P: TVarPath; InDim, OutDim: Int64); overload;
    constructor Create(const P: TVarPath; InDim, OutDim: Int64; const Cfg: TLinearConfig); overload;
    function Forward(const X: TTensor; Train: Boolean = False): TTensor; override;
  end;

  TTensorFunc = function(const X: TTensor): TTensor;
  TTensorFuncT = function(const X: TTensor; Train: Boolean): TTensor;

  TSequential = class(TModule)
  private
    FLayers: TObjectList;               // OwnsObjects = True
  public
    function Add(M: TModule): TSequential;            // returns Self for chaining
    function AddFn(F: TTensorFunc): TSequential;
    function AddFnT(F: TTensorFuncT): TSequential;
    function Forward(const X: TTensor; Train: Boolean = False): TTensor; override;
  end;

  TOptimizer = class
  public
    class function Sgd(VS: TVarStore; LR: Double; Momentum: Double = 0;
      Dampening: Double = 0; WeightDecay: Double = 0; Nesterov: Boolean = False): TOptimizer; static;
    class function Adam(VS: TVarStore; LR: Double; Beta1: Double = 0.9;
      Beta2: Double = 0.999; WeightDecay: Double = 0; Eps: Double = 1e-8;
      AmsGrad: Boolean = False): TOptimizer; static;
    class function AdamW(...): TOptimizer; static;
    class function RmsProp(...): TOptimizer; static;
    destructor Destroy; override;                      // ato_free
    procedure ZeroGrad;
    procedure Step;
    procedure BackwardStep(const Loss: TTensor);       // zero_grad; backward; step
    procedure BackwardStepClip(const Loss: TTensor; MaxValue: Double);
    procedure BackwardStepClipNorm(const Loss: TTensor; MaxNorm: Double);
    procedure ClipGradValue(MaxValue: Double);
    procedure ClipGradNorm(MaxNorm: Double);
    procedure SetLearningRate(LR: Double);             // + group variants, momentum, weight decay
  private
    FOpt: PTorchOptimizer;
    FStore: TVarStore;                                 // not owned
    FVarsInOptimizer: Integer;                         // lazily adds new trainable vars, as tch-rs
  end;
```

Decisions and rationale:
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
  A PyTorch-style `Training` flag on the module was considered and rejected
  for now: propagating it requires child registration machinery
  (`__setattr__` magic in Python) that Pascal would have to spell out.
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
  constructors (`TIValue.Tuple([...])`) and `As*` accessors that raise on a
  tag mismatch. **(H10)** A record containing a dynamic array of its own type
  must compile on 3.2.2; the fallback is an fpjson-style class hierarchy.
  **(H10) resolved** (step 7): it compiles and nests (tuple of tuples).
  Semantics to document: `W := V` shares `Items` (dynamic arrays are
  refcounted, not copy-on-write), so writing `W.Items[0]` changes `V` too.
  `TIValue` should be treated as immutable after construction.
  `TTrainableJitModule` registers `NamedParameters` into a `TVarPath` and
  derives from `TModule`, so a scripted model can be fine-tuned with
  `TOptimizer` like any other layer.

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
