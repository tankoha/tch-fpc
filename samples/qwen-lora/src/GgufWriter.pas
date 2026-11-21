{ GgufWriter — writes a GGUF file planned from a source read by GgufFile
  (T04), per DESIGN.md 8.15 and tasks/T04-facts.md (authoritative): the
  metadata copied as bytes, the tensor directory rebuilt in the source's
  order with optional new types, offsets recomputed, the data written in
  order with zero padding to the alignment.
  Skeleton by the harness owner (the facts' interface); stage 1 (T04b1)
  writes Create, Layout, SetTensorType and the properties, stage 2
  (T04b2) BeginWrite, WriteTensor, Finish, Destroy and GgufCopy. }
unit GgufWriter;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, GgufFile;

type
  TGgufWriter = class
  private
    FPath: string;
    FVersion: Cardinal;
    FAlignment: QWord;
    FKVCount: QWord;
    FMeta: TBytes;                       { source bytes KVStart..KVEnd }
    FTensors: array of TGgufTensorInfo;  { the plan }
    FDataOffset: QWord;
    FStream: TFileStream;                { the output, after BeginWrite }
    FNext: Integer;                      { next tensor to write }
    FFinished: Boolean;
    procedure Layout;
    function GetTensor(AI: Integer): TGgufTensorInfo;
    function GetTensorCount: Integer;
    function GetFileSize: QWord;
  public
    constructor Create(ASrc: TGgufFile);
    destructor Destroy; override;
    procedure SetTensorType(const AName: RawByteString; ATType: Cardinal);
    procedure BeginWrite(const APath: string);
    procedure WriteTensor(const ABytes: TBytes);
    procedure Finish;
    property TensorCount: Integer read GetTensorCount;
    property Tensor[I: Integer]: TGgufTensorInfo read GetTensor;
    property DataOffset: QWord read FDataOffset;
    property FileSize: QWord read GetFileSize;
  end;

procedure GgufCopy(const ASrcPath, ADstPath: string);

implementation

constructor TGgufWriter.Create(ASrc: TGgufFile);
var
  FS: TFileStream;
  MetaSize: QWord;
  I: Integer;
begin
  inherited Create;
  // copy basic properties from source
  FVersion := ASrc.Version;
  FAlignment := ASrc.Alignment;
  FKVCount := ASrc.KVCount;
  // read metadata bytes
  MetaSize := ASrc.KVEnd - ASrc.KVStart;
  SetLength(FMeta, LongInt(MetaSize));
  if MetaSize > 0 then
  begin
    FS := TFileStream.Create(ASrc.Path, fmOpenRead or fmShareDenyNone);
    try
      FS.Position := ASrc.KVStart;
      FS.ReadBuffer(FMeta[0], LongInt(MetaSize));
    finally
      FS.Free;
    end;
  end;
  // copy tensor directory entries (plan)
  SetLength(FTensors, ASrc.TensorCount);
  for I := 0 to ASrc.TensorCount - 1 do
    FTensors[I] := ASrc.Tensor[I];
  // compute layout based on copied data
  Layout;
end;

destructor TGgufWriter.Destroy;
begin
  if Assigned(FStream) and not FFinished then
  begin
    FStream.Free;
    DeleteFile(FPath);
  end;
  inherited Destroy;
end;

procedure TGgufWriter.Layout;
var
  I: Integer;
  DirSize, NameLen, Run: QWord;
begin
  // compute NBytes for each tensor
  for I := 0 to High(FTensors) do
    FTensors[I].NBytes := GgufTensorBytes(FTensors[I].TType, FTensors[I].Dims);
  // compute offsets for tensor data (padded)
  Run := 0;
  for I := 0 to High(FTensors) do
  begin
    FTensors[I].Offset := Run;
    Run := Run + ((FTensors[I].NBytes + FAlignment - 1) div FAlignment) * FAlignment;
  end;
  // compute directory size
  DirSize := 0;
  for I := 0 to High(FTensors) do
  begin
    NameLen := Length(FTensors[I].Name);
    DirSize := DirSize + 8 + NameLen + 4 + 8 * FTensors[I].NDims + 4 + 8;
  end;
  // data offset after header, metadata, and directory, aligned
  FDataOffset := ((24 + Length(FMeta) + DirSize + FAlignment - 1) div FAlignment) * FAlignment;
end;

function TGgufWriter.GetTensor(AI: Integer): TGgufTensorInfo;
begin
  Result := FTensors[AI];
end;

function TGgufWriter.GetTensorCount: Integer;
begin
  Result := Length(FTensors);
end;

function TGgufWriter.GetFileSize: QWord;
var
  I: Integer;
  TotalData: QWord;
begin
  TotalData := 0;
  for I := 0 to High(FTensors) do
    TotalData := TotalData + ((FTensors[I].NBytes + FAlignment - 1) div FAlignment) * FAlignment;
  Result := FDataOffset + TotalData;
end;

procedure TGgufWriter.SetTensorType(const AName: RawByteString; ATType: Cardinal);
var
  Idx: Integer;
  NewBytes: QWord;
  I: Integer;
begin
  if Assigned(FStream) or FFinished then
    raise EGgufError.Create('gguf writer: SetTensorType after BeginWrite or after Finish');
  // find tensor by name (case-sensitive)
  Idx := -1;
  for I := 0 to High(FTensors) do
    if FTensors[I].Name = AName then
    begin
      Idx := I;
      Break;
    end;
  if Idx = -1 then
    raise EGgufError.CreateFmt('gguf writer: no tensor named "%s"', [AName]);
  // compute new byte size before mutating (to keep plan unchanged on error)
  NewBytes := GgufTensorBytes(ATType, FTensors[Idx].Dims);
  // apply change
  FTensors[Idx].TType := ATType;
  FTensors[Idx].NBytes := NewBytes;
  // recompute layout (offsets, DataOffset, etc.)
  Layout;
end;

procedure TGgufWriter.BeginWrite(const APath: string);
var
  FS: TFileStream;
  I: Integer;
  NameLen: QWord;
  Zero: Byte = 0;
  PadSize: QWord;
  J: Integer;
  TensorCountVar: QWord;
begin
  if Assigned(FStream) then
    raise EGgufError.Create('TGgufWriter.BeginWrite: already called');
  FPath := APath;
  try
    FS := TFileStream.Create(APath, fmCreate);
  except
    on E: EStreamError do
      raise EGgufError.CreateFmt('TGgufWriter.BeginWrite: cannot create file %s', [APath]);
  end;
  FStream := FS;
  // header: magic, version, tensor count, KV count
  FS.WriteBuffer(PAnsiChar('GGUF')^, 4);
  FS.WriteBuffer(FVersion, SizeOf(FVersion));
  // tensor count (u64)
  TensorCountVar := Length(FTensors);
  FS.WriteBuffer(TensorCountVar, SizeOf(QWord));
  // KV count (u64)
  FS.WriteBuffer(FKVCount, SizeOf(FKVCount));
  // metadata key/value bytes
  if Length(FMeta) > 0 then
    FS.WriteBuffer(FMeta[0], Length(FMeta));
  // tensor directory entries in source order
  for I := 0 to High(FTensors) do
  begin
    NameLen := Length(FTensors[I].Name);
    FS.WriteBuffer(NameLen, SizeOf(QWord));
    if NameLen > 0 then
      FS.WriteBuffer(FTensors[I].Name[1], NameLen);
    FS.WriteBuffer(FTensors[I].NDims, SizeOf(Cardinal));
    // write each used dimension
    for J := 0 to FTensors[I].NDims - 1 do
      FS.WriteBuffer(FTensors[I].Dims[J], SizeOf(QWord));
    FS.WriteBuffer(FTensors[I].TType, SizeOf(Cardinal));
    FS.WriteBuffer(FTensors[I].Offset, SizeOf(QWord));
  end;
  // pad up to DataOffset
  PadSize := FDataOffset - FS.Position;
  while PadSize > 0 do
  begin
    FS.WriteBuffer(Zero, 1);
    Dec(PadSize);
  end;
end;

procedure TGgufWriter.WriteTensor(const ABytes: TBytes);
var
  Cur: TGgufTensorInfo;
  PadSize: QWord;
  Zero: Byte = 0;
begin
  if not Assigned(FStream) then
    raise EGgufError.Create('TGgufWriter.WriteTensor: BeginWrite not called');
  if FFinished then begin raise EGgufError.Create('TGgufWriter.WriteTensor: Finish already called'); end;
  if FNext >= Length(FTensors) then
    raise EGgufError.Create('TGgufWriter.WriteTensor: all tensors already written');
  Cur := FTensors[FNext];
  if Length(ABytes) <> Cur.NBytes then
    raise EGgufError.CreateFmt('TGgufWriter.WriteTensor: %s length %d <> expected %d', [Cur.Name, Length(ABytes), Cur.NBytes]);
  // write tensor data
  FStream.WriteBuffer(ABytes[0], Cur.NBytes);
  // pad to next alignment (relative to DataOffset)
  PadSize := ((Cur.NBytes + FAlignment - 1) div FAlignment) * FAlignment - Cur.NBytes;
  while PadSize > 0 do
  begin
    FStream.WriteBuffer(Zero, 1);
    Dec(PadSize);
  end;
  Inc(FNext);
end;

procedure TGgufWriter.Finish;
begin
  if not Assigned(FStream) then
    raise EGgufError.Create('TGgufWriter.Finish: BeginWrite not called');
  if FFinished then begin raise EGgufError.Create('TGgufWriter.Finish: already finished'); end;
  if FNext <> Length(FTensors) then
    raise EGgufError.CreateFmt('TGgufWriter.Finish: %d of %d tensors written', [FNext, Length(FTensors)]);
  // close file
  FStream.Free;
  FStream := nil;
  FFinished := True;
end;

procedure GgufCopy(const ASrcPath, ADstPath: string);
var
  Src: TGgufFile;
  Writer: TGgufWriter;
  I: Integer;
  Data: TBytes;
begin
  // source may raise its own exception if not found
  Src := TGgufFile.Open(ASrcPath);
  try
    Writer := TGgufWriter.Create(Src);
    try
      Writer.BeginWrite(ADstPath);
      for I := 0 to Writer.TensorCount - 1 do
      begin
        Src.ReadTensorBytes(I, Data);
        Writer.WriteTensor(Data);
      end;
      Writer.Finish;
    finally
      Writer.Free;
    end;
  finally
    Src.Free;
  end;
end;

end.
