# T04: GgufWriter (external design)

## Purpose

Unit `GgufWriter` writes a GGUF file planned from a source file read by `GgufFile` (T01). The metadata is copied as bytes, the tensor directory is rebuilt in the source's order, a tensor may be given a new type (T18 stores merged tensors as Q8_0), offsets are recomputed, and the tensor data is written in order. With nothing retyped the output is byte-identical to the source (DESIGN 8.15, "identity").

## Interface

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

## Errors

All routines raise `EGgufError` with descriptive messages:

- `SetTensorType`:
  - No tensor of that name: `'gguf writer: no tensor named "x"'`
  - Called after `BeginWrite`
  - GgufTensorBytes' own errors (unknown type, row length not a multiple of block size)

- `BeginWrite`:
  - Called twice
  - File cannot be created: EGgufError naming the path (RTL's EStreamError is not let through)

- `WriteTensor`:
  - Before `BeginWrite`
  - Every tensor already written
  - `Length(ABytes) <> NBytes` of that tensor (message names the tensor and both lengths)

- `Finish`:
  - Before `BeginWrite`
  - Tensors missing: `"N of M tensors written"`

`GgufFile`'s `EGgufError` is reused.
