# T01 facts (checked by the harness owner)

T01 is the unit `GgufFile` (reading only; the writer is T04) and the program
`qlora_inspect`. These facts were checked against upstream llama.cpp
`ggml/src/gguf.cpp` at `39febd2ad` and against the rehearsal model
(Qwen3.5 0.8B Q4_K_M) with an independent dumper and `llama-server`'s load
log. They are final; where a T01 document differs, these facts win. Do not
investigate the format further.

## The GGUF format (little-endian throughout)

```
magic      4 bytes  'G' 'G' 'U' 'F'  (as a UInt32: $46554747)
version    UInt32   2 or 3 accepted; anything else is an error
n_tensors  UInt64
n_kv       UInt64
n_kv times:      key (string), value type (UInt32), value
n_tensors times: name (string), n_dims (UInt32, 1..4),
                 n_dims dimensions (UInt64 each), tensor type (UInt32),
                 offset (UInt64, relative to the data section)
padding to the alignment, then the data section
```

- A string is a UInt64 byte count followed by that many bytes (no
  terminator; UTF-8, kept as `RawByteString`, never converted).
- Value types: 0 uint8, 1 int8, 2 uint16, 3 int16, 4 uint32, 5 int32,
  6 float32, 7 bool (1 byte, non-zero = true), 8 string, 9 array,
  10 uint64, 11 int64, 12 float64. Any other number is an error.
- An array is: element type (UInt32), count (UInt64), the elements. Arrays
  of arrays are an error (upstream does not allow them).
- Alignment: the value of `general.alignment` if present (it must be
  uint32, non-zero and a power of two, else error), otherwise 32.
- Data section start = the position after the last tensor info, rounded up
  to the alignment.
- Tensor types supported, with block size (elements) and block bytes:

| id | name (as `llama-server` prints it) | block | bytes |
|---|---|---|---|
| 0 | `f32` | 1 | 4 |
| 1 | `f16` | 1 | 2 |
| 8 | `q8_0` | 32 | 34 |
| 12 | `q4_K` | 256 | 144 |
| 13 | `q5_K` | 256 | 176 |
| 14 | `q6_K` | 256 | 210 |
| 30 | `bf16` | 1 | 2 |

  Any other id is an error "unsupported tensor type N" (the rehearsal model
  and the 27B use only these).
- Elements = product of the dimensions. The first dimension (`ne[0]`, the
  row length) must be a multiple of the block size. Bytes = elements /
  block × block bytes.
- Offsets: tensor *i*'s offset must equal the sum over the tensors before
  it of their bytes, each rounded up to the alignment (upstream checks
  exactly this). So the first offset is 0.
- The end of a tensor (data start + offset + bytes) must not exceed the
  file size. On a well-formed file the last tensor's end equals the file
  size.
- A file that ends early (any read past the end) is an error.
- So a file cut short (even by one byte) fails `Open` (the last tensor's end
  exceeds the file size): exit 1, nothing on stdout. `data_end <> file_size`
  with all lines printed happens only when the file is longer than its last
  tensor (trailing bytes).

## Unit `GgufFile` (interface, final)

The properties are read-only; give each a private field and a `read` specifier
(`property FileSize: QWord read FFileSize;`, indexed ones through a getter).

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

- The file is read with `TFileStream` (`fmOpenRead or fmShareDenyNone`) and
  kept open for `ReadTensorBytes`; nothing else is held in memory but the
  metadata and the directory.
- The tokenizer (T05) reads `tokenizer.ggml.tokens` (array of string,
  248320 entries in the rehearsal model), `tokenizer.ggml.merges` (array of
  string) and `tokenizer.ggml.token_type` (array of int32) through `KV[]`.
- Error messages name what was wrong and where: the key or tensor name and
  the byte position when known.

## Program `qlora_inspect`

```
qlora_inspect [--kv] [--tensors] FILE
```

Output on stdout, tab-separated, in this order:

```
version<TAB>3
alignment<TAB>32
kv<TAB>46
tensors<TAB>320
data_offset<TAB>10961920
data_end<TAB>532517120          (end of the last tensor; data_offset when there are no tensors)
file_size<TAB>532517120
type<TAB>f32<TAB>133            (one line per tensor type present, ordered by type id)
```

- `--kv` adds after that one line per key, in file order:
  `kv<TAB><key><TAB><type>` where type is the value-type name, or
  `array<elem>[count]` for arrays (for example `array<string>[248320]`).
  No values (floats and long strings make the output unstable).
- `--tensors` adds one line per tensor, in file order:
  `tensor<TAB><name><TAB><type name><TAB><dims><TAB><offset><TAB><bytes>`,
  dims in GGUF order separated by single spaces (`1024 248320`).
- Exit code 0 when the file reads and `data_end = file_size`; 1 when the
  file is not valid GGUF (message `error: ...` on stderr, nothing on
  stdout) or `data_end <> file_size` (all lines printed, then
  `error: data ends at X, file size Y` on stderr); 2 for a usage error.

## Oracle (the rehearsal model)

The model is `build/models/Qwen3.5-0.8B-Q4_K_M.gguf` in the repository root
(git-ignored); tests read its path from `QLORA_MODEL` and are skipped
without it. The expected values were produced outside the code under test
and are in `{proj}/work/oracle/T01/`:

- `summary.txt`: the first lines of the output above for this model
  (46 keys, 320 tensors, data section at 10961920, end 532517120 = the
  file size; f32 133, q8_0 36, q4_K 98, q5_K 36, q6_K 17).
- `llama-server-load.log`: `llama-server`'s load log; its
  `llama_model_loader: - type ...` lines give the same per-type counts.
- `kv-types.tsv`: `<key><TAB><type>` for the 46 keys, in file order.
- `tensors.tsv`: `<name><TAB><type><TAB><dims><TAB><offset><TAB><bytes>`
  for the 320 tensors, in file order.

`qlora_inspect --kv --tensors $QLORA_MODEL` must reproduce all of them:
the `kv` lines without the leading `kv<TAB>` equal `kv-types.tsv`, the
`tensor` lines without `tensor<TAB>` equal `tensors.tsv`, and the summary
lines equal `summary.txt`.

## Unit tests

UT never need the model. The test program builds small GGUF files itself
(a tiny writer in the test, under `{proj}/work/ut/gguf/`, made with
`mktemp`-style unique names) and covers at least:

- every value type, scalar and array, including an empty array and an
  empty string; a key with upper-case letters (lookups are exact);
- `general.alignment` absent (32), present (for example 64), not uint32,
  zero, not a power of two;
- each supported tensor type, and an unknown one; a row length that is not
  a multiple of the block size; 1 to 4 dimensions and 5 (error);
- offsets: correct with padding, one tensor out of place, a tensor past the
  end of the file; zero tensors (data_end = data_offset);
- bad magic, versions 1, 2, 3 and 4, a file cut in the middle of a string,
  an empty file;
- `Open` with a relative path as well as an absolute one;
- `ReadTensorBytes` returns exactly the bytes written for that tensor.

The program's logic lives in the unit `InspectCli`:
`function RunInspectCLI(const AArgs: array of string; AOut, AErr: TStrings): Integer;`
returns the exit code and fills the stdout and stderr lines;
`src/qlora_inspect.pas` only copies the arguments, calls it and writes the
two lists. The UT calls `RunInspectCLI` directly.
