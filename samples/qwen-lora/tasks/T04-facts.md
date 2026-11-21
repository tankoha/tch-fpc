# T04 facts (checked by the harness owner)

T04 is the unit `GgufWriter`: it writes a GGUF file planned from a source
file read by `GgufFile` (T01). The metadata is copied **as bytes**, the
tensor directory is rebuilt in the source's order, a tensor may be given a
new type (T18 stores merged tensors as Q8_0), offsets are recomputed, and
the tensor data is written in order. With nothing retyped the output is
**byte-identical** to the source (DESIGN 8.15, "identity").

Measured by the harness owner with a prototype written exactly as below:
the identity copy of the rehearsal model (532 517 120 bytes) and of the
projector `mmproj-BF16.gguf` (207 346 528 bytes) have the source's sha256;
the rehearsal model with 78 attention-side tensors retyped to Q8_0 equals,
byte for byte, the file assembled from upstream ggml's own GGUF code
(`gguf_set_tensor_type`, `gguf_get_meta_data`) and the C reference Q8_0;
the small files of the UT give the numbers in "Small source file". These
facts are final; do not investigate further or write probes.

## File layout (GGUF version 3, little-endian)

1. Header, 24 bytes: magic `GGUF` (4 bytes), version (u32), tensor count
   (u64), metadata key count (u64). The metadata starts at byte 24
   (`TGgufFile.KVStart`).
2. Metadata: the source's bytes `KVStart` up to `KVEnd`, copied unchanged;
   the key count is the source's `KVCount`.
3. Tensor directory, one entry per tensor in the source's order: name
   length (u64), name bytes (no terminator), number of dimensions (u32,
   `NDims`), `NDims` dimensions (u64 each; **only** `NDims` of them, so a
   dimension 1 inside `NDims` is written, the unused `Dims` entries are
   not), type (u32), offset (u64). One entry is
   `8 + Length(Name) + 4 + 8 * NDims + 4 + 8` bytes.
4. Zero bytes up to the next multiple of the alignment: that position is
   the data offset. It is padded even when there are no tensors.
5. Tensor data in directory order. Each tensor's bytes are followed by
   zero bytes up to the next multiple of the alignment **counted from the
   data offset** (also after the last tensor). So tensor 0 has offset 0,
   tensor i+1 has offset `Offset(i) + AlignUp(NBytes(i))`, and the file
   size is `DataOffset + Offset(last) + AlignUp(NBytes(last))`
   (`DataOffset` alone without tensors).

`AlignUp(x) = ((x + A - 1) div A) * A` with A = the source's `Alignment`
(T01: `general.alignment` if present and valid, else 32). The metadata is
copied, so the output keeps the same alignment key. A tensor's byte count
is `GgufTensorBytes(TType, Dims[0..NDims-1])` (T01; it raises EGgufError
for an unknown type and for a row length that is not a multiple of the
type's block size).

## Unit `GgufWriter` (interface, final)

```pascal
unit GgufWriter;
{$mode objfpc}{$H+}
interface
uses Classes, SysUtils, GgufFile;
type
  TGgufWriter = class
  public
    { Plans the output from ASrc (which must stay open until Create returns):
      version, alignment, KVCount, the metadata bytes KVStart..KVEnd
      (read from ASrc.Path with its own TFileStream, fmOpenRead or
      fmShareDenyNone), and a copy of every tensor entry; the layout
      (Offset, NBytes of every tensor, DataOffset) is computed at once. }
    constructor Create(ASrc: TGgufFile);
    { Closes the output; an output that was begun and not finished is
      deleted (no half-written file is left). }
    destructor Destroy; override;
    { Gives the tensor named AName (byte-exact, case-sensitive) the type
      ATType and recomputes the layout. Errors (EGgufError): no tensor of
      that name ('no tensor named "x"'); called after BeginWrite;
      GgufTensorBytes' own errors (unknown type, row length). }
    procedure SetTensorType(const AName: RawByteString; ATType: Cardinal);
    { Creates APath (replacing a file) and writes header, metadata,
      directory and the zero padding up to DataOffset. Errors: called
      twice; the file cannot be created (EGgufError naming the path; the
      RTL's EStreamError is not let through). }
    procedure BeginWrite(const APath: string);
    { Writes the next tensor's data (directory order) and its padding.
      Errors: before BeginWrite; every tensor already written;
      Length(ABytes) <> NBytes of that tensor (the message names the
      tensor and both lengths). }
    procedure WriteTensor(const ABytes: TBytes);
    { Checks that every tensor was written and closes the file. Errors:
      before BeginWrite; tensors missing ("N of M tensors written"). }
    procedure Finish;
    property TensorCount: Integer;                       { = source's }
    property Tensor[I: Integer]: TGgufTensorInfo;       { planned Name, NDims, Dims, TType, Offset, NBytes }
    property DataOffset: QWord;
    property FileSize: QWord;                           { planned size of the output }
  end;

{ The identity copy: Open ASrcPath, Create, BeginWrite(ADstPath), every
  tensor's ReadTensorBytes written in order, Finish; frees everything,
  also on an error. }
procedure GgufCopy(const ASrcPath, ADstPath: string);
```

Implementation notes (from the prototype):
- Keep the plan in an `array of TGgufTensorInfo` and one `Layout` routine
  that recomputes every `NBytes`, every `Offset` and `DataOffset`; call it
  from `Create` and after each `SetTensorType`. `Slice(T.Dims, T.NDims)`
  passes the used dimensions to `GgufTensorBytes`.
- In `SetTensorType` call `GgufTensorBytes` with the new type **before**
  changing the entry, so a failed call leaves the plan unchanged.
- Write with `TFileStream.WriteBuffer`; the padding from a small zeroed
  buffer. The output position after the directory, before padding, is
  `24 + (KVEnd - KVStart) + Σ entry sizes`.
- `TFileStream.Create(APath, fmCreate)` inside `try … except on E:
  EStreamError do raise EGgufError.CreateFmt(…)`.
- The destructor deletes the output only when `BeginWrite` succeeded and
  `Finish` did not run.

## Small source file (the UT's `WriteSmallGguf`, in the test skeleton)

`WriteSmallGguf(Path, AAlign64, ANoTensors)`: key `general.architecture`
= "test" (and `general.alignment` = 64 when AAlign64); tensors
`a` f32 [3] (NDims 1), `w` q4_K [256, 2] (NDims 2), `b` f32 [5, 1]
(NDims 2) unless ANoTensors. Data byte k of the data section is
`(k * 7 + 1) mod 256`, padding is 0. Measured with the prototype:

| Source | File size | KVEnd | DataOffset | a (off, bytes) | w | b |
|---|---|---|---|---|---|---|
| align 32 | 544 | 68 | 192 | 0, 12 | 32, 288 | 320, 20 |
| align 64 | 704 | 101 | 256 | 0, 12 | 64, 288 | 384, 20 |
| no tensors | 96 | 68 | 96 | — | — | — |

Plans on the align-32 source:
- `w` → Q8_0 (8): w 32, 544 bytes; b offset **576**; FileSize **800**.
- then `w` → q4_K (12) again: back to the table (FileSize 544).
- `a` → f16 (1): a 6 bytes; offsets unchanged (w 32, b 320); FileSize 544.
- errors (EGgufError): `x` → 'gguf writer: no tensor named "x"'; `W`
  (upper case) → the same; `b` → Q8_0: "the row length 5 is not a multiple
  of the block size 32"; `w` → 99: "unsupported tensor type 99".

Written with `w` → Q8_0 (a and b their source bytes, w 544 bytes of any
pattern) and read back with `TGgufFile.Open`: size 800, 3 tensors, KVCount
1, DataOffset 192, w type 8 offset 32, b offset 576, `ReadTensorBytes`
gives the bytes written; bytes DataOffset+12 .. DataOffset+31 are 0.
`GgufCopy` of each of the three sources equals the source byte for byte.
`BeginWrite('/nonexistent-dir/x.gguf')` raises EGgufError; freeing a
writer after `BeginWrite` without `Finish` leaves no file.

## Unit tests

Every row creates its files in the test's `Dir` (fresh per test, removed
by `TearDown`); the expected numbers are the table above. Stage 1 (plan,
rows from 001): Create on each of the three sources (TensorCount, names in
order, NDims/Dims copied, TType, Offset, NBytes, DataOffset, FileSize);
each plan of "Plans on the align-32 source"; each SetTensorType error,
including the upper-case name. Stage 2 (writing, rows from 101): GgufCopy
of each source equals the source bytes; a copy to a **bare relative name**
after `SetCurrentDir(Dir)` (restore the previous directory in a `finally`);
the `w` → Q8_0 file read back (all the numbers above, padding zero,
metadata key `general.architecture` still "test"); every error of
BeginWrite, WriteTensor, Finish and SetTensorType-after-BeginWrite; Free
after BeginWrite without Finish leaves no file. Compare files as bytes
(`FileBytes`, `CompareMem`), never by size alone.

The rows (every row has one test method `TestUT_GgufWriter_NNN`, every
test method belongs to a row):

| Row | Title | Expected |
|---|---|---|
| 001 | Create, align-32 source | 3 tensors a, w, b; NDims 1, 2, 2; Dims [3], [256,2], [5,1]; types 0, 12, 0; offsets 0, 32, 320; bytes 12, 288, 20; DataOffset 192; FileSize 544 |
| 002 | Create, align-64 source | offsets 0, 64, 384; DataOffset 256; FileSize 704 |
| 003 | Create, source without tensors | TensorCount 0; DataOffset 96; FileSize 96 |
| 004 | w → Q8_0 | w type 8, offset 32, 544 bytes; b offset 576; a unchanged; FileSize 800 |
| 005 | w → Q8_0, then w → q4_K | the plan of row 001 again |
| 006 | a → f16 | a 6 bytes; w 32, b 320; FileSize 544 |
| 007 | unknown name `x` | EGgufError, message contains `x` |
| 008 | name `W` (upper case) | EGgufError |
| 009 | b → Q8_0 | EGgufError (row length 5); the plan is that of row 001 afterwards |
| 010 | w → type 99 | EGgufError (unsupported type 99); the plan is that of row 001 afterwards |
| 101 | GgufCopy, align-32 source | output bytes = source bytes (544) |
| 102 | GgufCopy, align-64 source | output bytes = source bytes (704) |
| 103 | GgufCopy, source without tensors | output bytes = source bytes (96) |
| 104 | GgufCopy to a bare relative name | after `SetCurrentDir(Dir)`: `GgufCopy('src.gguf', 'copy.gguf')`, bytes equal; previous directory restored |
| 105 | w → Q8_0 written and read back | size 800; 3 tensors; KVCount 1; key `general.architecture` = "test"; DataOffset 192; w type 8 offset 32; b offset 576; ReadTensorBytes of a, w, b = the bytes written; bytes DataOffset+12 .. +31 are 0 |
| 106 | BeginWrite twice | EGgufError |
| 107 | BeginWrite to `/nonexistent-dir/x.gguf` | EGgufError, message contains the path |
| 108 | WriteTensor before BeginWrite | EGgufError |
| 109 | WriteTensor with 11 bytes for `a` | EGgufError, message contains `a`, 12 and 11 |
| 110 | WriteTensor after all three | EGgufError |
| 111 | Finish before BeginWrite | EGgufError |
| 112 | Finish after one tensor | EGgufError, message contains "1 of 3" |
| 113 | SetTensorType after BeginWrite | EGgufError |
| 114 | Free after BeginWrite without Finish | the output file does not exist |
| 115 | GgufCopy of a missing source | the exception of `TGgufFile.Open` (measured: EFOpenError) passes through; no destination file |
| 116 | Finish twice; WriteTensor after Finish (added in T04b3) | EGgufError for each |

## ITa (prepared by the harness owner)

`QLORA_MODEL=<absolute path of the rehearsal GGUF> {proj}/tests/ita/T04-run.sh`
builds `tests/ita/T04/writer_check.pas`, runs it from a fresh directory
outside the project and writes `{proj}/work/ita/T04/summary.tsv` (case,
OK/NG, diff lines, lines ours, lines oracle, note; then the drivers' exit
codes and the stderr line count). About 30 s. Cases:

| Case | Compared | Lines |
|---|---|---|
| ITa-T04-001 | identity copy of the model: sha256 and size | 1 |
| ITa-T04-002 | planned layout of the copy equals upstream | 322 |
| ITa-T04-003 | identity copy of the projector: sha256 and size | 1 |
| ITa-T04-004 | planned layout after retyping 78 tensors equals upstream | 322 |
| ITa-T04-005 | retyped file: sha256 and size | 1 |
| ITa-T04-006 | upstream reads the retyped file back with that layout | 322 |

The oracle (`work/oracle/T04`, not in git): `gguf_meta.c` linked against
Ollama's `libggml-base.so` (upstream gguf.cpp), `expected.tsv`,
`retype-names.txt`, `retype-layout.tsv`, made by `make-oracle.sh`.

## Stages and case IDs

- Stage 1 (task T04b1): `Create`, `Layout`, `SetTensorType`, the
  properties; UT rows `UT-GgufWriter-001` to `099`.
- Stage 2 (task T04b2): `BeginWrite`, `WriteTensor`, `Finish`, `Destroy`,
  `GgufCopy`; UT rows from `UT-GgufWriter-101`.
