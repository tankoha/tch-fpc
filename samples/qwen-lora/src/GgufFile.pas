{ GgufFile — read-only GGUF reader (T01), per DESIGN.md 8.12 and
  tasks/T01-facts.md (authoritative).
  Stage 1 of T01b (task T01b1): the types of the facts' interface,
  GgufValueTypeName, TGgufFile.Open up to and including the metadata
  (magic, version, counts, every key and value, KVStart, KVEnd, the
  alignment rule), FindKey, the read-only properties, Destroy.
  Stage 2 (task T01b2): the tensor directory (Open), GgufTypeName,
  GgufTensorBytes, FindTensor, ReadTensorBytes and the tensor properties
  (T01-detailed §2.3, 2.4, 2.6/8, 2.7, 2.9, 2.10).
  Contract: docs/detailed-design/T01-detailed.md §2.1–2.2, 2.5–2.8, 2.11,
  §3 (header + KV), §4 (R1–R4, R9–R11, R19), §5, §6 (I-1, I-3). }
unit GgufFile;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils;

type
  { The only error class (T01-facts "errors"): every failure during Open
    raises EGgufError with a message that says what, where; the
    constructor then raises, so a failed Open never publishes a half-open
    object. }
  EGgufError = class(Exception);

  { The 13 GGUF metadata value types, in wire order (0..12), per T01-facts
    "interface" and T01-detailed §2.6/5. }
  TGgufValueType = (gvUInt8, gvInt8, gvUInt16, gvInt16, gvUInt32, gvInt32,
    gvFloat32, gvBool, gvString, gvArray, gvUInt64, gvInt64, gvFloat64);

  { One parsed metadata entry (T01-detailed §3). Which members are valid
    depends on VType/ArrType, as the facts' comments say. Texts are
    RawByteString: UTF-8 bytes as read, never converted. }
  TGgufKV = record
    Key: RawByteString;
    VType: TGgufValueType;
    ArrType: TGgufValueType; { element type; valid when VType = gvArray }
    Count: QWord;            { array: number of elements; scalar: 1 }
    IntVal: Int64;           { every integer type and bool (0/1); uint64 above High(Int64) is stored as-is (typecast) }
    FloatVal: Double;        { float32 and float64 }
    StrVal: RawByteString;   { string }
    Strs: array of RawByteString; { array of string }
    Ints: array of Int64;         { array of any integer type or bool; uint64 above High(Int64) is stored as-is (typecast) }
    Floats: array of Double;      { array of float32 or float64 }
  end;

  { One tensor directory entry (T01-detailed §2.7, the facts'
    TGgufTensorInfo). Dims is in GGUF order (Dims[0] is the row length,
    ne[0]); unused Dims entries are 1. Offset is relative to the data
    section; NBytes is computed by GgufTensorBytes (§2.4) and stored for
    the read. }
  TGgufTensorInfo = record
    Name: RawByteString;
    NDims: Integer;
    Dims: array[0..3] of QWord;
    TType: Cardinal;
    Offset: QWord;
    NBytes: QWord;
  end;

  { Read-only GGUF reader (T01-facts "interface" stage 1: Path, FileSize,
    Version, Alignment, KVStart, KVEnd, KVCount, KV[], FindKey + Open and
    Destroy; the tensor members are stage 2). The file is read with
    TFileStream (fmOpenRead or fmShareDenyNone) and kept open until
    Destroy (T01-facts "open"); on the same disk the file is assumed not
    to change in the meantime. All members are filled by Open. }
  TGgufFile = class
  private
    FPath: string;
    FStream: TFileStream;
    FFileSize: QWord;
    FVersion: Cardinal;
    FAlignment: QWord; { general.alignment if valid and a power of two, else 32 }
    FKVStart: QWord;
    FKVEnd: QWord;
    FDataOffset: QWord; { data section start (2.7): after the last tensor info, rounded up }
    FKVs: array of TGgufKV;
    FTensors: array of TGgufTensorInfo;
    procedure ReadAt(APos: QWord; var ABuf; ASize: QWord);
    function ReadU32(APos: QWord): Cardinal;
    function ReadU64(APos: QWord): QWord;
    function ReadGgufString(APos: QWord): RawByteString;
    function ReadKVEntry(AStart: QWord; out AEnd: QWord): TGgufKV;
    function GetKV(AI: Integer): TGgufKV;
    function GetKVCount: Integer;
    function GetTensor(AI: Integer): TGgufTensorInfo;
    function GetTensorCount: Integer;
  public
    { Opens APath and reads through the metadata (T01-detailed §2.5–2.7):
      all checks in wire order; on a failed check the object is freed and
      EGgufError raised, so no half-open object is published. Stage 1 stops
      after the metadata (§2.11; the tensor directory is T01b2). }
    constructor Open(const APath: string);
    destructor Destroy; override;
    { Index of the first key that matches AKey byte for byte (case
      sensitive), or -1 (T01-detailed §2.8). }
    function FindKey(const AKey: RawByteString): Integer;
    { Index of the first tensor that matches AName byte for byte (case
      sensitive), or -1 (T01-detailed §2.9). }
    function FindTensor(const AName: RawByteString): Integer;
    { ABytes gets the NBytes bytes of tensor AIndex, read from the stream
      still open at file offset DataOffset + Offset (§2.10). }
    procedure ReadTensorBytes(AIndex: Integer; out ABytes: TBytes);
    property Path: string read FPath;
    property FileSize: QWord read FFileSize;
    property Version: Cardinal read FVersion;
    property Alignment: QWord read FAlignment;
    property KVStart: QWord read FKVStart;
    property KVEnd: QWord read FKVEnd;
    property KVCount: Integer read GetKVCount;
    property KV[I: Integer]: TGgufKV read GetKV;
    property DataOffset: QWord read FDataOffset;
    property TensorCount: Integer read GetTensorCount;
    property Tensor[I: Integer]: TGgufTensorInfo read GetTensor;
  end;

{ The wire-order name of a metadata value type (T01-detailed §2.2, the
  facts' GgufValueTypeName). }
function GgufValueTypeName(AVType: TGgufValueType): string;

{ The tensor-type name as llama-server prints it (T01-detailed §2.3, the
  facts' GgufTypeName); an unknown id is 'type<N>'. }
function GgufTypeName(AType: Cardinal): string;

{ Byte size of one tensor from its type and GGUF-ordered dimensions
  (T01-detailed §2.4, the facts' GgufTensorBytes). }
function GgufTensorBytes(AType: Cardinal; const ADims: array of QWord): QWord;

implementation

{ 2.11 — release the stream, the KV array and the tensor array;
  a failed Open never leaves them behind either. }
destructor TGgufFile.Destroy;
begin
  if FStream <> nil then
    FStream.Free;
  FStream := nil;
  SetLength(FKVs, 0);
  SetLength(FTensors, 0);
  FFileSize := 0;
  FDataOffset := 0;
  inherited;
end;

{ 2.9 — linear scan, exact case-sensitive comparison of the whole name,
  the first matching index or -1 (same contract as FindKey). }
function TGgufFile.FindTensor(const AName: RawByteString): Integer;
var
  I: Integer;
begin
  Result := -1;
  for I := 0 to GetTensorCount - 1 do
    if FTensors[I].Name = AName then
    begin
      Result := I;
      Break;
    end;
end;

{ 2.7 — indexed read of one tensor directory entry (FPC bounds-checks). }
function TGgufFile.GetTensor(AI: Integer): TGgufTensorInfo;
begin
  Result := FTensors[AI];
end;

{ 2.7 — the number of tensor directory entries read in Open. }
function TGgufFile.GetTensorCount: Integer;
begin
  Result := Length(FTensors);
end;

{ 2.8 — linear scan, exact case-sensitive byte-for-byte comparison, first
  matching index. }
function TGgufFile.FindKey(const AKey: RawByteString): Integer;
var
  I: Integer;
begin
  Result := -1;
  for I := 0 to GetKVCount - 1 do
    if FKVs[I].Key = AKey then
    begin
      Result := I;
      Break;
    end;
end;

{ 2.9 — indexed access, stage 1 has no bound check of its own (FPC does
  its own on the array). }
function TGgufFile.GetKV(AI: Integer): TGgufKV;
begin
  Result := FKVs[AI];
end;

{ 2.9 — how many metadata entries were read. }
function TGgufFile.GetKVCount: Integer;
begin
  Result := Length(FKVs);
end;

{ 2.2 — the facts' GgufValueTypeName: the wire-order name of each value
  type, exact as in T01-facts "interface". }
function GgufValueTypeName(AVType: TGgufValueType): string;
begin
  case AVType of
    gvUInt8:   Result := 'uint8';
    gvInt8:    Result := 'int8';
    gvUInt16:  Result := 'uint16';
    gvInt16:   Result := 'int16';
    gvUInt32:  Result := 'uint32';
    gvInt32:   Result := 'int32';
    gvFloat32: Result := 'float32';
    gvBool:    Result := 'bool';
    gvString:  Result := 'string';
    gvArray:   Result := 'array';
    gvUInt64:  Result := 'uint64';
    gvInt64:   Result := 'int64';
    gvFloat64: Result := 'float64';
  end;
end;

{ 2.3 — the facts' GgufTypeName table, exact as in T01-facts "interface";
  an unknown id is 'type<N>'. }
function GgufTypeName(AType: Cardinal): string;
begin
  case AType of
    0:  Result := 'f32';
    1:  Result := 'f16';
    8:  Result := 'q8_0';
    12: Result := 'q4_K';
    13: Result := 'q5_K';
    14: Result := 'q6_K';
    30: Result := 'bf16';
  else
    Result := 'type' + IntToStr(Int64(AType));
  end;
end;

{ 2.4 — the facts' GgufTensorBytes: look up the (block size, block bytes)
  pair, Elements = product of the dimensions, the row length (ADims[0],
  ne[0]) must be a multiple of the block size, Bytes =
  Elements div block * block bytes. An unknown type is "unsupported
  tensor type N" (facts); a bad row length says what is wrong (2.6/11). }
function GgufTensorBytes(AType: Cardinal; const ADims: array of QWord): QWord;
var
  Block, BlockBytes, Elements: QWord;
  I: Integer;
begin
  case AType of
    0:  begin Block := 1;   BlockBytes := 4;   end; { f32  }
    1:  begin Block := 1;   BlockBytes := 2;   end; { f16  }
    8:  begin Block := 32;  BlockBytes := 34;  end; { q8_0 }
    12: begin Block := 256; BlockBytes := 144; end; { q4_K }
    13: begin Block := 256; BlockBytes := 176; end; { q5_K }
    14: begin Block := 256; BlockBytes := 210; end; { q6_K }
    30: begin Block := 1;   BlockBytes := 2;   end; { bf16 }
  else
    raise EGgufError.CreateFmt('gguf: unsupported tensor type %d', [AType]);
  end;
  Elements := 1;
  for I := Low(ADims) to High(ADims) do
    Elements := Elements * ADims[I];
  if ADims[Low(ADims)] mod Block <> 0 then
    raise EGgufError.CreateFmt(
      'gguf: tensor type %s: the row length %d is not a multiple of the block size %d',
      [GgufTypeName(AType), ADims[Low(ADims)], Block]);
  Result := (Elements div Block) * BlockBytes;
end;

{ 2.6/10 — read ASize bytes at APos from the stream; any read that would
  run past the file is EGgufError (a cut file, T01-detailed 5.9). The
  position is an explicit argument, so Open stays stepwise and the
  stream can be reused later (T01b2 ReadTensorBytes). }
procedure TGgufFile.ReadAt(APos: QWord; var ABuf; ASize: QWord);
begin
  if APos + ASize > FFileSize then
    raise EGgufError.CreateFmt(
      'gguf: file is %d bytes; cannot read %d bytes at position %d',
      [FFileSize, ASize, APos]);
  FStream.Position := APos;
  FStream.Read(ABuf, LongInt(ASize));
end;

{ 2.10 — read one tensor's data section from the stream, still open
  (R12): ABytes gets exactly NBytes bytes at file offset
  DataOffset + Offset. AIndex is in 0..TensorCount-1 (the caller's
  contract); the extent check already happened in Open (2.6/8), so the
  bytes are inside the file — a short read is an EGgufError by ReadAt. }
procedure TGgufFile.ReadTensorBytes(AIndex: Integer; out ABytes: TBytes);
var
  N: QWord;
begin
  N := FTensors[AIndex].NBytes;
  SetLength(ABytes, LongInt(N));
  if N > 0 then
    ReadAt(FDataOffset + FTensors[AIndex].Offset, PByte(@ABytes[0])^, N);
end;

{ 2.6/5 — little-endian UInt32 at APos (T01-detailed 5.7/8). }
function TGgufFile.ReadU32(APos: QWord): Cardinal;
var
  V: Cardinal;
begin
  ReadAt(APos, V, 4);
  Result := V;
end;

{ 2.6/5 — little-endian UInt64 at APos (T01-detailed 5.7/8). }
function TGgufFile.ReadU64(APos: QWord): QWord;
var
  V: QWord;
begin
  ReadAt(APos, V, 8);
  Result := V;
end;

{ 2.6/9 — a GGUF string: UInt64 number of bytes, then that many bytes,
  no terminator. Kept as RawByteString, never converted (UTF-8 bytes as
  in the file; T01-detailed 5.9/5.10, case 005/008). ReadAt does the
  "not past the end of the file" check, so a cut string is EGgufError. }
function TGgufFile.ReadGgufString(APos: QWord): RawByteString;
var
  C: QWord;
  Buf: TBytes;
begin
  C := ReadU64(APos);
  Result := ''; { explicit: C = 0 must yield a definite empty string }
  SetLength(Buf, LongInt(C));
  if C > 0 then
  begin
    ReadAt(APos + 8, PByte(@Buf[0])^, C);
    SetString(Result, PAnsiChar(@Buf[0]), LongInt(C));
  end;
end;

{ 2.7 — one metadata entry at AStart: key (2.6/9), value type (2.6/5),
  the value decoded by type (2.6/4–6, 9, 10; 2.7). AEnd is set to the
  offset just past the value. Unknown value types, array-of-arrays and
  truncated values raise EGgufError naming the key (2.6/5, 2.7,
  T01-detailed 5.9). }
function TGgufFile.ReadKVEntry(AStart: QWord; out AEnd: QWord): TGgufKV;
var
  Pos: QWord;
  VT: Cardinal;
  ArrVT: Cardinal;
  N: Int64; { signed: `0 to N-1` must skip when N = 0 (QWord underflows) }
  I: Int64;
  B: Byte;
  W: Word;
  L: LongWord;
  U: QWord;
  Sng: Single;
  Dbl: Double;
  S: RawByteString;
begin
  Result.Key := '';
  Result.VType := gvUInt8;
  Result.ArrType := gvUInt8;
  Result.Count := 1;
  Result.IntVal := 0;
  Result.FloatVal := 0;
  Result.StrVal := '';
  SetLength(Result.Strs, 0);
  SetLength(Result.Ints, 0);
  SetLength(Result.Floats, 0);

  Pos := AStart;
  Result.Key := ReadGgufString(Pos);
  Inc(Pos, 8 + QWord(Length(Result.Key)));
  VT := ReadU32(Pos);
  Inc(Pos, 4);
  if VT > 12 then
    raise EGgufError.CreateFmt(
      'gguf: unknown value type %d for key "%s" at position %d',
      [VT, Result.Key, AStart]);
  Result.VType := TGgufValueType(VT);
  case TGgufValueType(VT) of
    gvUInt8:
      begin
        ReadAt(Pos, B, 1);
        Result.IntVal := Int64(B);
        Inc(Pos, 1);
      end;
    gvInt8:
      begin
        ReadAt(Pos, B, 1);
        Result.IntVal := Int64(Int8(B));
        Inc(Pos, 1);
      end;
    gvUInt16:
      begin
        ReadAt(Pos, W, 2);
        Result.IntVal := Int64(W);
        Inc(Pos, 2);
      end;
    gvInt16:
      begin
        ReadAt(Pos, W, 2);
        Result.IntVal := Int64(Int16(W));
        Inc(Pos, 2);
      end;
    gvUInt32:
      begin
        ReadAt(Pos, L, 4);
        Result.IntVal := Int64(L);
        Inc(Pos, 4);
      end;
    gvInt32:
      begin
        ReadAt(Pos, L, 4);
        Result.IntVal := Int64(LongInt(L));
        Inc(Pos, 4);
      end;
    gvUInt64, gvInt64:
      begin
        U := ReadU64(Pos);
        Result.IntVal := Int64(U); { uint64 above High(Int64) as-is (typecast) }
        Inc(Pos, 8);
      end;
    gvFloat32:
      begin
        ReadAt(Pos, Sng, 4);
        Result.FloatVal := Double(Sng);
        Inc(Pos, 4);
      end;
    gvFloat64:
      begin
        ReadAt(Pos, Dbl, 8);
        Result.FloatVal := Dbl;
        Inc(Pos, 8);
      end;
    gvBool:
      begin
        ReadAt(Pos, B, 1);
        Result.IntVal := Ord(B <> 0);
        Inc(Pos, 1);
      end;
    gvString:
      begin
        Result.StrVal := ReadGgufString(Pos);
        Inc(Pos, 8 + QWord(Length(Result.StrVal)));
      end;
    gvArray:
      begin
        ArrVT := ReadU32(Pos);
        Inc(Pos, 4);
        if (ArrVT = 9) or (ArrVT > 12) then { array of arrays (2.6/6); bad type }
          raise EGgufError.CreateFmt(
            'gguf: unsupported array element type %d for key "%s" at position %d',
            [ArrVT, Result.Key, AStart]);
        N := ReadU64(Pos);
        Inc(Pos, 8);
        Result.ArrType := TGgufValueType(ArrVT);
        Result.Count := N;
        case TGgufValueType(ArrVT) of
          gvString:
            begin
              SetLength(Result.Strs, LongInt(N));
              for I := 0 to N - 1 do
              begin
                S := ReadGgufString(Pos);
                Result.Strs[LongInt(I)] := S;
                Inc(Pos, 8 + QWord(Length(S)));
              end;
            end;
          gvFloat32:
            begin
              SetLength(Result.Floats, LongInt(N));
              for I := 0 to N - 1 do
              begin
                ReadAt(Pos, Sng, 4);
                Result.Floats[LongInt(I)] := Double(Sng);
                Inc(Pos, 4);
              end;
            end;
          gvFloat64:
            begin
              SetLength(Result.Floats, LongInt(N));
              for I := 0 to N - 1 do
              begin
                ReadAt(Pos, Dbl, 8);
                Result.Floats[LongInt(I)] := Dbl;
                Inc(Pos, 8);
              end;
            end;
        else { int8..int64 and bool: one value per element (2.7) }
            begin
              SetLength(Result.Ints, LongInt(N));
              for I := 0 to N - 1 do
                case TGgufValueType(ArrVT) of
                  gvUInt8:
                    begin
                      ReadAt(Pos, B, 1);
                      Result.Ints[LongInt(I)] := Int64(B);
                      Inc(Pos, 1);
                    end;
                  gvInt8:
                    begin
                      ReadAt(Pos, B, 1);
                      Result.Ints[LongInt(I)] := Int64(Int8(B));
                      Inc(Pos, 1);
                    end;
                  gvUInt16:
                    begin
                      ReadAt(Pos, W, 2);
                      Result.Ints[LongInt(I)] := Int64(W);
                      Inc(Pos, 2);
                    end;
                  gvInt16:
                    begin
                      ReadAt(Pos, W, 2);
                      Result.Ints[LongInt(I)] := Int64(Int16(W));
                      Inc(Pos, 2);
                    end;
                  gvUInt32:
                    begin
                      ReadAt(Pos, L, 4);
                      Result.Ints[LongInt(I)] := Int64(L);
                      Inc(Pos, 4);
                    end;
                  gvInt32:
                    begin
                      ReadAt(Pos, L, 4);
                      Result.Ints[LongInt(I)] := Int64(LongInt(L));
                      Inc(Pos, 4);
                    end;
                  gvUInt64, gvInt64:
                    begin
                      U := ReadU64(Pos);
                      Result.Ints[LongInt(I)] := Int64(U);
                      Inc(Pos, 8);
                    end;
                  gvBool:
                    begin
                      ReadAt(Pos, B, 1);
                      Result.Ints[LongInt(I)] := Ord(B <> 0);
                      Inc(Pos, 1);
                    end;
                end;
            end;
        end;
      end;
  end;
  AEnd := Pos;
end;

{ 2.5, 2.6, 2.7 — header (magic, version, n_tensors, n_kv), then the n_kv
  metadata entries in wire order; stage 1 stops after the metadata
  (T01-detailed 2.11 — the tensor directory is T01b2, so the test files
  of this stage have n_tensors = 0). Checks in wire order (2.6); on a
  failed check the object is freed (Destroy) and then the EGgufError is
  raised, so no half-open object is published. }
constructor TGgufFile.Open(const APath: string);
var
  Magic: Cardinal;
  NTensors: QWord;
  NKV: QWord;
  I: Int64; { signed bound: N = 0 must skip the loop (QWord 0-1 underflows) }
  J: Int64;
  P: QWord;
  EntryStart: QWord;
  EntryEnd: QWord;
  Entry: TGgufKV;
  NDims: Cardinal; { a tensor's n_dims; must be 1..4 (2.6/8) }
  TName: RawByteString; { the tensor name for the error messages (2.6/11) }
  TMsg: string; { the message of the wrapped GgufTensorBytes error }
  TOff: QWord; { the offset as written in the bad entry }
  TEnd: QWord; { DataOffset + offset + size: the byte just past the tensor }
  Run: QWord; { padded byte sum of the preceding tensors (2.7) }
  EntryStarts: array of QWord; { the file position of each tensor entry }
begin
  FPath := APath;
  FStream := TFileStream.Create(APath, fmOpenRead or fmShareDenyNone);
  FFileSize := FStream.Size;
  FAlignment := 32;
  FKVStart := 24;
  FKVEnd := 24;
  SetLength(FKVs, 0);
  if FFileSize < 24 then { 2.6/10: the header needs 24 bytes }
  begin
    raise EGgufError.CreateFmt(
      'gguf: file is %d bytes; the header (magic, version, n_tensors, n_kv) needs 24',
      [FFileSize]);
  end;
  Magic := ReadU32(0);
  if Magic <> $46554747 then { 2.6/1: "GGUF" little-endian }
  begin
    raise EGgufError.Create('gguf: bad magic at position 0 (expected "GGUF")');
  end;
  FVersion := ReadU32(4);
  if (FVersion <> 2) and (FVersion <> 3) then { 2.6/2 }
  begin
    raise EGgufError.CreateFmt(
      'gguf: unsupported version %d at position 4 (expected 2 or 3)',
      [FVersion]);
  end;
  NTensors := ReadU64(8); { T01b2: its n_tensors directory entries follow the metadata }
  NKV := ReadU64(16);
  SetLength(FKVs, LongInt(NKV));
  P := FKVStart;
  for I := 0 to Int64(NKV) - 1 do
  begin
    EntryStart := P;
    Entry := ReadKVEntry(EntryStart, EntryEnd);
    FKVs[I] := Entry;
    P := EntryEnd;
    if FKVs[I].Key = 'general.alignment' then { 2.6/7, R2 }
      begin
        if (FKVs[I].VType <> gvUInt32)
           or (FKVs[I].IntVal = 0)
           or (FKVs[I].IntVal and (FKVs[I].IntVal - 1) <> 0) then
        begin
          raise EGgufError.CreateFmt(
            'gguf: key "general.alignment" at position %d: the value must be a uint32, non-zero and a power of two',
            [EntryStart]);
        end;
        FAlignment := QWord(FKVs[I].IntVal);
      end;
  end;
  FKVEnd := P; { 2.7/R9: the offset just past the last value }

  { --- tensor directory (2.6/8, 2.7). First pass: read every entry in
    wire order (name, n_dims with the 1..4 check, the dimensions, type,
    offset). Second pass: check every entry in entry order — tensor type
    and row length (GgufTensorBytes, 2.4), then the offset against the
    padded sum of the preceding entries (2.6/8), then the extent against
    DataOffset and the file size (2.6/8). DataOffset is only known after
    the last entry's info, hence the two passes. On a failed check the
    object is destroyed and EGgufError raised, the tensor named and the
    position shown (2.6/11). }
  SetLength(FTensors, LongInt(NTensors));
  SetLength(EntryStarts, LongInt(NTensors));
  for I := 0 to Int64(NTensors) - 1 do
  begin
    EntryStarts[I] := P;
    FTensors[I].Name := ReadGgufString(P); { 2.6/9 }
    Inc(P, 8 + QWord(Length(FTensors[I].Name)));
    NDims := ReadU32(P); { 2.6/8: n_dims (UInt32, 1..4) }
    Inc(P, 4);
    if (NDims < 1) or (NDims > 4) then
    begin
      TName := FTensors[I].Name;
      raise EGgufError.CreateFmt(
        'gguf: tensor "%s" at position %d: n_dims must be 1..4 (got %d)',
        [TName, EntryStarts[I], NDims]);
    end;
    FTensors[I].NDims := Integer(NDims);
    for J := 1 to Integer(NDims) do
    begin
      FTensors[I].Dims[J - 1] := ReadU64(P); { 2.6/8: the dimensions }
      Inc(P, 8);
    end;
    for J := Integer(NDims) to 3 do
      FTensors[I].Dims[J] := 1; { unused Dims entries = 1 (2.7) }
    FTensors[I].TType := ReadU32(P); { 2.6/8: tensor type (UInt32) }
    Inc(P, 4);
    FTensors[I].Offset := ReadU64(P); { 2.6/8: offset (UInt64) }
    Inc(P, 8);
  end;
  { 2.7 — DataOffset: the position after the last tensor info, rounded up
    to the alignment. }
  FDataOffset := (P + FAlignment - 1) div FAlignment * FAlignment;

  Run := 0;
  for I := 0 to Int64(NTensors) - 1 do
  begin
    EntryStart := EntryStarts[I];
    TName := FTensors[I].Name;
    TOff := FTensors[I].Offset;
    try
      FTensors[I].NBytes := GgufTensorBytes(
        FTensors[I].TType, FTensors[I].Dims);
    except
      on E: EGgufError do { bad type or row length (2.4); name the tensor (2.6/11) }
        begin
          TMsg := E.Message;
          raise EGgufError.CreateFmt(
            'gguf: tensor "%s" at position %d: %s',
            [TName, EntryStart, TMsg]);
        end;
    end;
    if TOff <> Run then { 2.6/8: the offset is the padded sum before it }
    begin
      raise EGgufError.CreateFmt(
        'gguf: tensor "%s" at position %d: offset %d, expected %d (alignment %d)',
        [TName, EntryStart, TOff, Run, FAlignment]);
    end;
    TEnd := FDataOffset + TOff + FTensors[I].NBytes;
    if TEnd > FFileSize then { 2.6/8: extent }
    begin
      raise EGgufError.CreateFmt(
        'gguf: tensor "%s" at position %d ends at %d, but the file is %d bytes',
        [TName, EntryStart, TEnd, FFileSize]);
    end;
    Run := Run + ((FTensors[I].NBytes + FAlignment - 1) div FAlignment) * FAlignment;
  end;
end;

end.
