# T01 detailed design — `GgufFile` unit and `qlora_inspect` program

Based on `tasks/T01-facts.md` (authoritative; these documents only restate and
organise it), `docs/external-design/T01-external.md` and
`docs/system-flow/T01-system.md`. Where a T01 document differs, the facts win.
DESIGN.md is not used (the facts were taken from it).

## 1. Module composition

One unit per file in `src/`, named as in the facts. The unit `GgufFile` is
built in two stages (T01b1: types, header and metadata through `KVEnd`,
`GgufValueTypeName`, `FindKey`, the properties and `Destroy`; T01b2: the
tensor directory, `GgufTypeName`, `GgufTensorBytes`, `ReadTensorBytes`); this
document describes the final complete unit. The UT case tables (T01a step 4;
`tests/ut/cases/GgufFile.cases.md`, `tests/ut/cases/InspectCli.cases.md`)
are written from this document.

| File | Kind | Responsibility |
|---|---|---|
| `src/GgufFile.pas` | unit | Read-only GGUF reader: `TGgufFile` (`Open`, `FindKey`, `FindTensor`, `ReadTensorBytes`, `Destroy`, the read-only properties); `EGgufError`; `TGgufValueType`, `TGgufKV`, `TGgufTensorInfo`; `GgufTypeName`, `GgufTensorBytes`, `GgufValueTypeName` |
| `src/InspectCli.pas` | unit | `RunInspectCLI` — argument parsing, the output lines, the `data_end` check and the exit code (all of the program's logic; the UT calls it directly) |
| `src/qlora_inspect.pas` | program | Thin entry: copies the arguments, calls `RunInspectCLI`, writes the two lists, halts with the returned code |

Everything is Free Pascal 3.2.2, `{$mode objfpc}{$H+}`. No GPU and no
environment variable are used by these units (tests read `QLORA_MODEL` for
the rehearsal file; the units themselves take their path as an argument).

## 2. Routine specifications

### 2.1 Fingerprint (facts, final; implementation and tests must match)

```pascal
type
  EGgufError = class(Exception);

  TGgufValueType = (gvUInt8, gvInt8, gvUInt16, gvInt16, gvUInt32, gvInt32,
    gvFloat32, gvBool, gvString, gvArray, gvUInt64, gvInt64, gvFloat64);

  TGgufKV = record
    Key: RawByteString;
    VType: TGgufValueType;
    ArrType: TGgufValueType;   { element type; only when VType = gvArray }
    Count: QWord;              { array element count; 1 for a scalar }
    IntVal: Int64;             { all integer types and bool (0/1); uint64 above High(Int64) is stored as is (typecast) }
    FloatVal: Double;          { float32 and float64 }
    StrVal: RawByteString;     { string }
    Strs: array of RawByteString;  { array of string }
    Ints: array of Int64;          { array of any integer type or bool }
    Floats: array of Double;       { array of float32 or float64 }
  end;

  TGgufTensorInfo = record
    Name: RawByteString;
    NDims: Integer;
    Dims: array[0..3] of QWord;  { GGUF order: Dims[0] is the row length; unused = 1 }
    TType: Cardinal;
    Offset: QWord;               { relative to the data section }
    NBytes: QWord;
  end;

  TGgufFile = class
  public
    constructor Open(const APath: string);   { reads and checks header, metadata and tensor directory; raises EGgufError }
    destructor Destroy; override;
    function FindKey(const AKey: RawByteString): Integer;     { -1 if absent; exact, case-sensitive }
    function FindTensor(const AName: RawByteString): Integer; { -1 if absent; exact, case-sensitive }
    procedure ReadTensorBytes(AIndex: Integer; out ABytes: TBytes);
    property Path: string;          { as given to Open }
    property FileSize: QWord;
    property Version: Cardinal;
    property Alignment: QWord;
    property KVStart: QWord;        { byte offset of the first key: always 24 }
    property KVEnd: QWord;          { byte offset just after the last value (T04 copies [KVStart, KVEnd) as bytes) }
    property DataOffset: QWord;     { absolute start of the data section }
    property KVCount: Integer;
    property KV[I: Integer]: TGgufKV;
    property TensorCount: Integer;
    property Tensor[I: Integer]: TGgufTensorInfo;
  end;

function GgufTypeName(ATType: Cardinal): string;      { 'f32', 'q4_K', ...; 'type<N>' for unknown }
function GgufTensorBytes(ATType: Cardinal; const ADims: array of QWord): QWord;  { raises EGgufError for an unknown type or a bad row length }
function GgufValueTypeName(AVType: TGgufValueType): string;  { 'uint8' ... 'float64' as in the value-type list }
```

Properties are read-only: each has a private field and a `read` specifier
(`property FileSize: QWord read FFileSize;`), the indexed ones through a
getter. All text is `RawByteString`: UTF-8 bytes, kept as is, never
converted, compared byte-for-byte (exact, case-sensitive).

### 2.2 `function GgufValueTypeName(AVType: TGgufValueType): string`

- Purpose: the value-type name for the `--kv` output lines (`kv-types.tsv`
  oracle).
- Interface: as in the facts (2.1); no raise.
- Checks and errors: none.
- Internal logic: table lookup of `TGgufValueType` → name, the facts'
  value-type list: `0 uint8, 1 int8, 2 uint16, 3 int16, 4 uint32, 5 int32,
  6 float32, 7 bool (1 byte, non-zero = true), 8 string, 9 array,
  10 uint64, 11 int64, 12 float64.`

### 2.3 `function GgufTypeName(ATType: Cardinal): string`

- Purpose: the tensor-type name as `llama-server` prints it, for the `type`
  and `tensor` output lines.
- Interface: as in the facts (2.1); no raise.
- Checks and errors: none.
- Internal logic: table lookup of the facts' tensor-type table —
  `0 f32, 1 f16, 8 q8_0, 12 q4_K, 13 q5_K, 14 q6_K, 30 bf16`; any other id
  returns `'type<N>'`.

### 2.4 `function GgufTensorBytes(ATType: Cardinal; const ADims: array of QWord): QWord`

- Purpose: byte size of one tensor, computed from its type and dimensions;
  used by `Open` for every tensor and for the offset checks.
- Interface: as in the facts (2.1).
- Checks and errors (in this order):
  - type — `ATType` must be a supported tensor type; `Any other id is an
    error "unsupported tensor type N"` → `EGgufError`.
  - row length — `The first dimension (ne[0], the row length) must be a
    multiple of the block size` → otherwise `EGgufError` (message per
    2.6/11).
- Internal logic, in order:
  1. Look up `(block size, block bytes)` for `ATType` in the table below.
  2. `Elements = product of the dimensions.` (in code:
     `Elements := ADims[0] * ADims[1] * ... * ADims[High(ADims)]`).
  3. Check the row-length condition on `ADims[0]` and the `block size` of
     step 1 (`Open` applies it to every tensor directory entry via this
     call, 2.6/8).
  4. `Bytes = elements / block × block bytes.` (in code:
     `Result := (Elements div block) * blockBytes`; the check in step 3
     makes the division exact).
  Supported types, with block size (elements) and block bytes (facts table):

| id | name | block | bytes |
|---|---|---|---|
| 0 | `f32` | 1 | 4 |
| 1 | `f16` | 1 | 2 |
| 8 | `q8_0` | 32 | 34 |
| 12 | `q4_K` | 256 | 144 |
| 13 | `q5_K` | 256 | 176 |
| 14 | `q6_K` | 256 | 210 |
| 30 | `bf16` | 1 | 2 |

### 2.5 `constructor TGgufFile.Open(const APath: string)`

- Purpose: open `APath`, read and check the header, the metadata (all keys
  and values) and the tensor directory. The file is read with `TFileStream`
  (`fmOpenRead or fmShareDenyNone`) and kept open for `ReadTensorBytes`;
  nothing else is held in memory but the metadata and the directory.

- Interface: as in the facts (2.1) — `constructor Open(const APath: string);`
  reads and checks header, metadata and tensor directory; raises `EGgufError`.

### 2.6 Open — checks and errors, in the order the facts give them

1. magic — `magic 4 bytes 'G' 'G' 'U' 'F' (as a UInt32: $46554747)`;
   otherwise error (`EGgufError`).
2. version — `version UInt32 2 or 3 accepted; anything else is an error`.
3. header — `n_tensors UInt64`, `n_kv UInt64` (a missing read falls under
   10. below).
4. each of the `n_kv` entries — `n_kv times: key (string), value type
   (UInt32), value`; a string is read as the count then the bytes (9.).
5. value type — `any other number is an error` (0 uint8 … 12 float64).
6. array — `An array is: element type (UInt32), count (UInt64), the
   elements. Arrays of arrays are an error (upstream does not allow
   them).`
7. alignment — `general.alignment` if present must be `uint32, non-zero and
   a power of two, else error`; `otherwise 32` (used in 12.).
8. tensor directory — for each of the `n_tensors` entries (its fields, in
   file order: name, `n_dims`, dimensions, type, offset):
   - `n_dims (UInt32, 1..4)` — outside 1..4 is an error;
   - tensor type — must be in the supported table (2.4); `Any other id is
     an error "unsupported tensor type N"`;
   - row length — `The first dimension (ne[0], the row length) must be a
     multiple of the block size`;
   - offset — `tensor i's offset must equal the sum over the tensors before
     it of their bytes, each rounded up to the alignment (upstream checks
     exactly this). So the first offset is 0.`;
   - extent — `The end of a tensor (data start + offset + bytes) must not
     exceed the file size. On a well-formed file the last tensor's end
     equals the file size.`
9. strings — `A string is a UInt64 byte count followed by that many bytes
   (no terminator; UTF-8, kept as RawByteString, never converted).`
10. end of file — `A file that ends early (any read past the end) is an
    error.`
11. messages — `Error messages name what was wrong and where: the key or
    tensor name and the byte position when known.`

Any failed check raises `EGgufError`; on failure the object is not
published (Open completes only after every check above has passed).

### 2.7 Open — internal logic

- Open the file with `TFileStream` (`fmOpenRead or fmShareDenyNone`); keep
  the stream for the lifetime of the object (freed by `Destroy`;
  `ReadTensorBytes` reuses it).
- Header, read in order at file start: magic `UInt32 $46554747` (2.6/1);
  `Version` `UInt32` checked 2 or 3 (2.6/2); `n_tensors` and `n_kv`
  (`UInt64`).
- Metadata, `n_kv` times, starting at `KVStart = 24` (24 = header;
  `KVStart` property is always 24). Per entry, in file order:
  - `Key` = the key string (2.6/9);
  - `VType` = the value-type `UInt32` mapped to `TGgufValueType` (2.6/5);
  - the value, decoded by `VType` into `TGgufKV`:
    - integer types (uint8/int8/uint16/int16/uint32/int32/uint64/int64) and
      bool → `IntVal` (bool as 0/1; `uint64 above High(Int64) is stored as
      is (typecast)`);
    - float32 / float64 → `FloatVal` (as Double);
    - string → `StrVal` (RawByteString, never converted);
    - array → `ArrType`, `Count`, then the elements into `Strs` / `Ints` /
      `Floats` by `ArrType` (arrays of arrays are rejected, 2.6/6);
    - `Count = 1` for a scalar, the array element count for an array.
  - when the key is `general.alignment`, remember its value as the
    alignment (2.6/7; checked there: `uint32, non-zero and a power of two,
    else error`);
  - after the last entry set `KVEnd` — `byte offset just after the last
    value (T04 copies [KVStart, KVEnd) as bytes)`.
- Alignment: the remembered `general.alignment`, `otherwise 32` (2.6/7).
- Tensor directory, `n_tensors` times, in file order, per entry:
  - `Name` (2.6/9), `NDims` (2.6/8), `NDims` `Dim` values (`UInt64`) into
    `Dims` (GGUF order: `Dims[0]` is the row length; unused entries = 1),
    `TType`, `Offset` (`relative to the data section`).
  - `NBytes := GgufTensorBytes(TType, Dims)` (2.4 — raises `"unsupported
    tensor type N"` / bad row length).
  - Offset check (2.6/8): keep `run` over the earlier tensors
    (`run := run + round-up(earlier NBytes, Alignment)` before each tensor);
    if the entry's `Offset` differs from `run` → `EGgufError`; `run` is
    0 for the first tensor (`the first offset is 0`).
  - Extent check (2.6/8): `DataOffset + Offset + NBytes > FileSize` is an
    error.
- After the whole directory, `DataOffset` — `the position after the last
  tensor info, rounded up to the alignment`:
  `DataOffset := (PosAfterDirectory + Alignment - 1) div Alignment * Alignment`.
  The bytes between `PosAfterDirectory` and `DataOffset` are the padding and
  are skipped without being read.
- Publish the read-only fields (2.1): `Path` (as given), `FileSize`,
  `Version`, `Alignment`, `KVStart`, `KVEnd`, `DataOffset`, `KVCount`,
  `KV[0..KVCount-1]`, `TensorCount`, `Tensor[0..TensorCount-1]`.

### 2.8 `function TGgufFile.FindKey(const AKey: RawByteString): Integer`

- Purpose: index of a key in the metadata; used by callers (e.g. the
  tokenizer (T05) resolves `tokenizer.ggml.*` through `KV[]`).
- Interface: as in the facts (2.1) — returns the index, `-1 if absent`.
- Checks and errors: none (an absent key is `-1`, not an error).
- Internal logic: linear scan `i = 0..KVCount-1`; `KV[i].Key = AKey` is the
  byte-for-byte, case-sensitive equality of `RawByteString`s (keys are kept
  as read, never converted; lookups are exact). First matching `i` or `-1`.

### 2.9 `function TGgufFile.FindTensor(const AName: RawByteString): Integer`

- Purpose: index of a tensor in the directory; same contract as `FindKey`
  for tensor names.
- Interface: as in the facts (2.1) — returns the index, `-1 if absent`.
- Checks and errors: none (an absent name is `-1`, not an error).
- Internal logic: linear scan `i = 0..TensorCount-1`;
  `Tensor[i].Name = AName` byte-for-byte, case-sensitive (names kept as
  `RawByteString`, never converted; lookups are exact). First matching `i`
  or `-1`.

### 2.10 `procedure TGgufFile.ReadTensorBytes(AIndex: Integer; out ABytes: TBytes)`

- Purpose: read one tensor's data section, so a caller (e.g. `Tch` in
  later tasks) gets exactly the file's bytes without the tensor being
  loaded by the unit.
- Interface: as in the facts (2.1).
- Checks and errors:
  - extent: covered by Open (2.6/8) — a tensor that ends past
    `FileSize` has already been rejected, so
    `DataOffset + Tensor[AIndex].Offset + Tensor[AIndex].NBytes <= FileSize`;
  - end of file — `A file that ends early (any read past the end) is an
    error.` (any short read while filling `ABytes` → `EGgufError`).
- Internal logic: `AIndex` must be a valid index `0..TensorCount-1` (a
  precondition; an out-of-range index is caller error). `ABytes` is sized
  `Tensor[AIndex].NBytes`. Position the stream at
  `DataOffset + Tensor[AIndex].Offset` and read `Tensor[AIndex].NBytes`
  bytes — exactly the bytes written for that tensor (UT: `ReadTensorBytes
  returns exactly the bytes written for that tensor`).

### 2.11 `destructor TGgufFile.Destroy`

- Purpose: release the open file handle (the stream is kept open since
  `Open` for `ReadTensorBytes`; 2.5).
- Interface: as in the facts (2.1) — `destructor Destroy; override;`.
- Checks and errors: none.
- Internal logic: free the `TFileStream` (if open), clear the held
  metadata and directory; call `inherited Destroy`.

### 2.12 `function RunInspectCLI(const AArgs: array of string; AOut, AErr: TStrings): Integer`

- Purpose: the whole CLI of `qlora_inspect` — argument parsing, the
  summary / `type` / `kv` / `tensor` output lines and the exit code. The
  unit `InspectCli` holds the program's logic; `src/qlora_inspect.pas`
  only copies the arguments, calls it and writes the two lists. The UT
  calls `RunInspectCLI` directly.
- Interface (facts): `function RunInspectCLI(const AArgs: array of string;
  AOut, AErr: TStrings): Integer;` — returns the exit code and fills the
  stdout and stderr lines. `AArgs` is the command line
  `qlora_inspect [--kv] [--tensors] FILE`.
- Checks and errors — the facts' exact sentence:

  > Exit code 0 when the file reads and `data_end = file_size`; 1 when the
  > file is not valid GGUF (message `error: ...` on stderr, nothing on
  > stdout) or `data_end <> file_size` (all lines printed, then
  > `error: data ends at X, file size Y` on stderr); 2 for a usage error.

  ...and the checks in the order the CLI actually applies them:
  1. usage — `2 for a usage error`: no `FILE`, or an unrecognized option;
     the usage line on stderr, nothing on stdout; return 2.
  2. not valid GGUF — `TGgufFile.Open` (2.6) raises `EGgufError` →
     message `error: <the EGgufError message>` on stderr, nothing on
     stdout; return 1.
  3. `data_end <> file_size` — all lines on stdout printed first, then
     `error: data ends at X, file size Y` on stderr (X = `data_end`,
     Y = `file_size`); return 1.
  4. success — the file reads and `data_end = file_size`; return 0.
- Internal logic, in order:
  1. Parse `AArgs`: the `--kv` and `--tensors` flags and the `FILE`
     argument (the `Path`, absolute or relative, as given — never modified).
     A usage error → exit 2 with the usage line on stderr
     (`Usage: qlora_inspect [--kv] [--tensors] FILE`, external design) and
     nothing on stdout.
  2. Open the file: `F := TGgufFile.Open(FPath)`; if it raises `EGgufError`
     (2.6) → `AErr` gets `error: <EGgufError message>`, return `1` (the
     error is caught before any line is written — nothing on stdout).
  3. `data_end` := `data_offset when there are no tensors`; else the end of
     the last tensor = `DataOffset + Tensor[TensorCount-1].Offset +
     Tensor[TensorCount-1].NBytes` (the offsets were already checked in
     Open, so this is the sum of the earlier tensors' bytes, each rounded
     up to the alignment, then the last tensor's bytes).
  4. Summary lines to `AOut`, in this order (output on stdout,
     tab-separated; the values below are the rehearsal model's — each is
     replaced by `this file's own value`):
     ```
     version<TAB>3
     alignment<TAB>32
     kv<TAB>46
     tensors<TAB>320
     data_offset<TAB>10961920
     data_end<TAB>532517120
     file_size<TAB>532517120
     ```
     `data_end` = `end of the last tensor; data_offset when there are no
     tensors`.
  5. Then the `type` lines to `AOut`: `one line per tensor type present,
     ordered by type id`, one line per type — e.g.
     `type<TAB>f32<TAB>133` (name from 2.3, count = the number of tensors
     of that type).
  6. If `--kv`: `one line per key, in file order` —
     `kv<TAB><key><TAB><type>`, `where type is the value-type name, or
     array<elem>[count] for arrays (for example array<string>[248320]). No
     values (floats and long strings make the output unstable).` (`<type>`
     from 2.2.)
  7. If `--tensors`: `one line per tensor, in file order` —
     `tensor<TAB><name><TAB><type name><TAB><dims><TAB><offset><TAB><bytes>`,
     `dims in GGUF order separated by single spaces (1024 248320).`
     (`<type name>` from 2.3.)
  8. Result: if `data_end = file_size` → return `0` (all lines on stdout,
     nothing on stderr). Otherwise the lines are already printed and
     `AErr` gets `error: data ends at X, file size Y` with X = `data_end`,
     Y = `file_size`; return `1`.
  9. Free the `TGgufFile` (stream closed).
- `src/qlora_inspect.pas` (program): copies `ParamStr` into `AArgs`, calls
  `RunInspectCLI` with fresh `AOut` / `AErr`, writes `AOut` to stdout and
  `AErr` to stderr, halts with the returned code (0, 1 or 2).

## 3. Data structures and constants

- GGUF file layout, little-endian throughout (facts):

| part | size / shape |
|---|---|
| `magic` | 4 bytes `'G' 'G' 'U' 'F'` (as a `UInt32: $46554747`) |
| `version` | `UInt32` — 2 or 3 accepted |
| `n_tensors` | `UInt64` |
| `n_kv` | `UInt64` |
| KV | `n_kv` times: key (string), value type (`UInt32`), value |
| tensor directory | `n_tensors` times: name (string), `n_dims` (`UInt32`, 1..4), `n_dims` dimensions (`UInt64` each), tensor type (`UInt32`), offset (`UInt64`, relative to the data section) |
| padding | to the alignment (skipped without being read) |
| data section | `DataOffset` onward; `A file that ends early (any read past the end) is an error.` |

- A string: `a UInt64 byte count followed by that many bytes (no
  terminator; UTF-8, kept as RawByteString, never converted).`
- Value types (id → name; `GgufValueTypeName`, 2.2): `0 uint8, 1 int8,
  2 uint16, 3 int16, 4 uint32, 5 int32, 6 float32, 7 bool, 8 string,
  9 array, 10 uint64, 11 int64, 12 float64`.
- Tensor types (id → name / block / block bytes; `GgufTypeName`, 2.3, and
  `GgufTensorBytes`, 2.4): the table in 2.4.
- Alignment: the value of `general.alignment` if present (`it must be
  uint32, non-zero and a power of two, else error`), otherwise 32.
- `DataOffset` (data section start) = `the position after the last tensor
  info, rounded up to the alignment`. `KVStart` is always 24 (the header
  size: `4 + 4 + 8 + 8`); `KVEnd` = the byte offset just after the last
  value.
- Tensor bytes: `Elements = product of the dimensions.` `The first
  dimension (ne[0], the row length) must be a multiple of the block size.`
  `Bytes = elements / block × block bytes.`
- Held in memory: the metadata (`TGgufKV` array), the tensor directory
  (`TGgufTensorInfo` array), the header fields and the open `TFileStream`
  (2.5). Nothing from the data section is read by `Open`.

## 4. Business-rule → implementation map

Every rule of the external design / facts for T01 is realized here:

| # | Rule (facts / external design) | Where |
|---|---|---|
| R1 | magic `$46554747`; version 2 or 3 accepted, anything else an error | Open, 2.6/1–2 |
| R2 | `general.alignment` if present: `uint32, non-zero and a power of two, else error`; otherwise 32 | Open KV loop + 2.6/7 |
| R3 | value types 0..12; `any other number is an error`; `Arrays of arrays are an error` | Open KV loop, 2.6/5–6 |
| R4 | strings: UInt64 count then bytes, no terminator, `RawByteString, never converted` | Open KV/directory reads, 2.6/9 |
| R5 | tensor directory: `n_dims 1..4`, supported types only (`unsupported tensor type N`), `ne[0]` a multiple of the block size | Open directory loop + `GgufTensorBytes` (2.4), 2.6/8 |
| R6 | `tensor i's offset must equal the sum over the tensors before it of their bytes, each rounded up to the alignment … the first offset is 0` | Open directory loop, 2.7 |
| R7 | the end of a tensor must not exceed the file size; `On a well-formed file the last tensor's end equals the file size` | Open (2.6/8); `data_end` check |
| R8 | `DataOffset` = position after the last tensor info, rounded up to the alignment; padding skipped (not read) | Open, 2.7 |
| R9 | `KVStart` always 24; `KVEnd` just after the last value (T04 copies `[KVStart, KVEnd)`) | Open, 2.7 |
| R10 | `A file that ends early (any read past the end) is an error` | Open / `ReadTensorBytes`, 2.6/10, 2.10 |
| R11 | `FindKey` / `FindTensor`: `-1 if absent; exact, case-sensitive` | 2.8 / 2.9 |
| R12 | `ReadTensorBytes` — `exactly the bytes written for that tensor` (kept-open `TFileStream`, `fmOpenRead or fmShareDenyNone`) | 2.10, 2.5 |
| R13 | summary lines `version / alignment / kv / tensors / data_offset / data_end / file_size`, tab-separated, in this order | `RunInspectCLI` step 4, 2.12 |
| R14 | `type<TAB><name><TAB><count>` — `one line per tensor type present, ordered by type id` | `RunInspectCLI` step 5 |
| R15 | `--kv`: `kv<TAB><key><TAB><type>`, file order, `array<elem>[count] for arrays`, `No values` | `RunInspectCLI` step 6 |
| R16 | `--tensors`: `tensor<TAB><name><TAB><type name><TAB><dims><TAB><offset><TAB><bytes>`, `dims in GGUF order separated by single spaces` | `RunInspectCLI` step 7 |
| R17 | `data_end` = `end of the last tensor; data_offset when there are no tensors` | `RunInspectCLI` step 3 |
| R18 | exit 0 when `data_end = file_size`; 1 for not valid GGUF (`error: ...` on stderr, nothing on stdout) or `data_end <> file_size` (all lines printed, then `error: data ends at X, file size Y`); 2 for a usage error | `RunInspectCLI` steps 1, 2, 8; checks 2.12 |
| R19 | `Error messages name what was wrong and where: the key or tensor name and the byte position when known` | `EGgufError` messages, 2.6/11 |
| R20 | the program is a thin wrapper: `src/qlora_inspect.pas` only copies the arguments, calls `RunInspectCLI` and writes the two lists | 2.12, step 9 / program |

## 5. Edge cases and error handling

| Situation | Behavior |
|---|---|
| File cut in the middle of a string / any read past the end | `EGgufError` on that read (`A file that ends early (any read past the end) is an error.`); `qlora_inspect` → exit 1, `error: ...` on stderr, nothing on stdout (2.12) |
| `n_tensors = 0` | valid; `DataOffset` rounded up after the (absent) directory; `data_end = data_offset`; exit 0 still possible (2.12 step 3) |
| zero tensors vs. a tensor past the end | `zero tensors (data_end = data_offset)` is fine; `a tensor past the end of the file` → `EGgufError` in Open (2.6/8) |
| `general.alignment` absent / present / not uint32 / zero / not a power of two | 32 / that value / all three invalid shapes → `EGgufError` (2.6/7) |
| a key with upper-case letters | stored as read, never converted; lookups are exact, case-sensitive (2.8, 2.6/9) |
| an empty array, an empty string | read and stored without error (count 0 / empty `RawByteString`; 2.7 KV decode) |
| a tensor type outside the table | `EGgufError "unsupported tensor type N"` (`GgufTensorBytes`, 2.4; Open 2.6/8) |
| `n_dims` = 5 (or 0) | `n_dims` outside 1..4 → `EGgufError` (2.6/8) |
| `data_end <> file_size` (bytes after the last tensor, all checks of Open passing; a file cut short fails the extent check in Open instead) | `RunInspectCLI` prints all lines, then `error: data ends at X, file size Y`; exit 1 (2.12) |
| the same key or tensor name twice | first index reported by `FindKey` / `FindTensor` (linear scan) — the facts do not say keys/names are unique; not an Open error |

## 6. Interpretation notes (no external-design change requested)

- **I-1** The facts fix the exact message only for two situations:
  `"unsupported tensor type N"` (`GgufTensorBytes`, 2.4) and
  `error: data ends at X, file size Y` (2.12). All other `EGgufError`
  messages follow only the rule in 2.6/11 (`Error messages name what was
  wrong and where: the key or tensor name and the byte position when
  known.`); the ITa cases (e.g. ITa-T01-007 with `error: bad magic`)
  expect `error: ` + the `EGgufError` message, so the message text is
  settled by the UT (T01a step 4) from that rule — this design invents no
  message of its own.
- **I-2** `Open` reads the metadata and the tensor directory only; the
  data section is read on demand by `ReadTensorBytes` via the kept-open
  `TFileStream` (2.5, 2.12). `KVStart` / `KVEnd` exist for T04 (byte copy
  of the metadata) and are not used by `qlora_inspect`.
- **I-3** `RunInspectCLI` never touches the environment; the model path
  comes from `AArgs`. The UT builds its own small GGUF files (a writer in
  the test under `{proj}/work/ut/gguf/`) and calls `RunInspectCLI`
  directly (facts "Unit tests").
- **I-4** The tokenizer (T05) is a later consumer: it reads
  `tokenizer.ggml.tokens`, `tokenizer.ggml.merges` and
  `tokenizer.ggml.token_type` through `FindKey` + `KV[]` (facts).
- **I-5** The oracle for `qlora_inspect --kv --tensors $QLORA_MODEL` is the
  rehearsal model (the model is `build/models/Qwen3.5-0.8B-Q4_K_M.gguf` in
  the repository root (git-ignored); tests read its path from `QLORA_MODEL`
  and are skipped without it): `{proj}/work/oracle/T01/` — `summary.txt`
  (46 keys, 320 tensors, data section at 10961920, end 532517120 = the file
  size; f32 133, q8_0 36, q4_K 98, q5_K 36, q6_K 17), `kv-types.tsv`
  (`<key><TAB><type>` for the 46 keys, in file order), `tensors.tsv`
  (`<name><TAB><type><TAB><dims><TAB><offset><TAB><bytes>` for the 320
  tensors, in file order), `llama-server-load.log` (its
  `llama_model_loader: - type ...` lines give the same per-type counts) —
  facts "Oracle": the `kv` lines without the leading `kv<TAB>` equal
  `kv-types.tsv`, the `tensor` lines without `tensor<TAB>` equal
  `tensors.tsv`, and the summary lines equal `summary.txt`.
- **I-6** The facts "Unit tests" list — every value type, scalar and
  array, including an empty array and an empty string; a key with upper-case
  letters; `general.alignment` absent (32), present (for example 64), not
  uint32, zero, not a power of two; each supported tensor type, and an
  unknown one; a row length that is not a multiple of the block size; 1 to
  4 dimensions and 5 (error); offsets: correct with padding, one tensor out
  of place, a tensor past the end of the file; zero tensors
  (`data_end = data_offset`); bad magic, versions 1, 2, 3 and 4, a file cut
  in the middle of a string, an empty file; `Open` with a relative path as
  well as an absolute one; `ReadTensorBytes` returns exactly the bytes
  written for that tensor — is the input for the T01a step 4 case tables
  (`tests/ut/cases/GgufFile.cases.md`, `InspectCli.cases.md`); §5's edge
  cases and the 2.x checks/errors are its basis.

## 7. Change history

| Date | Change | Reason |
|---|---|---|
| 2026-11-18 | Initial detailed design | T01a step 3; from `tasks/T01-facts.md` (checked by the harness owner) |

