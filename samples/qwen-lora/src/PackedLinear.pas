{ T10: a matrix kept packed on a device (the GGUF's own bytes), dequantised
  per call to fp32 by GgufQuantT.DequantizeT (DESIGN 8.12, "PackedLinear").
  Shapes in libtorch order [Out, In]; a GGUF matrix lists its dims as
  (In, Out). LoadPacked / LoadDense read a tensor of a TGgufFile. With a
  LoRA pair (T12; not owned) Forward adds its Delta. Written without
  reference to Qwen.

  Interface by the harness owner (tasks/T10-facts.md, T12-facts.md); the
  bodies are written in T10b and T12b. }
unit PackedLinear;

{$mode objfpc}{$H+}

interface

uses
  SysUtils, Tch, GgufFile, GgufQuantT, Lora;

type
  EPackedLinear = class(Exception);

  TPackedLinear = class
  private
    FRaw: TTensor;
    FTType: Cardinal;
    FOutDim, FInDim: Int64;
    FLora: TLoraPair;
    procedure SetLora(AValue: TLoraPair);
  public
    constructor Create(const ARaw: TTensor; ATType: Cardinal; AOutDim, AInDim: Int64);
    function Weight: TTensor;
    function Forward(const X: TTensor): TTensor;
    function Device: TDevice;
    property TType: Cardinal read FTType;
    property OutDim: Int64 read FOutDim;
    property InDim: Int64 read FInDim;
    { Not owned; nil (the default) for none. }
    property Lora: TLoraPair read FLora write SetLora;
  end;

function TensorRaw(G: TGgufFile; const AName: RawByteString; const ADevice: TDevice;
  out ATType: Cardinal; out ADims: TInt64Array): TTensor;
function LoadPacked(G: TGgufFile; const AName: RawByteString; const ADevice: TDevice): TPackedLinear;
function LoadDense(G: TGgufFile; const AName: RawByteString; const ADevice: TDevice): TTensor;

implementation

function ShapeStr(const T: TTensor): string;
var
  S: TInt64Array;
  I: Integer;
begin
  S := T.Shape;
  Result := '[';
  for I := 0 to High(S) do
  begin
    if I > 0 then
      Result := Result + ', ';
    Result := Result + IntToStr(S[I]);
  end;
  Result := Result + ']';
end;

constructor TPackedLinear.Create(const ARaw: TTensor; ATType: Cardinal; AOutDim, AInDim: Int64);
var
  Want: QWord;
begin
  inherited Create;
  if (ARaw.Kind <> kUInt8) or (ARaw.Dim <> 1) then
    raise EPackedLinear.Create('the packed bytes must be a 1-D kUInt8 tensor, got ' + ShapeStr(ARaw));
  if (AOutDim < 1) or (AInDim < 1) then
    raise EPackedLinear.CreateFmt('dimensions must be positive, got %d x %d', [AOutDim, AInDim]);
  Want := GgufTensorBytes(ATType, [QWord(AOutDim * AInDim)]);
  if QWord(ARaw.Shape[0]) <> Want then
    raise EPackedLinear.CreateFmt('%s holds %d bytes, %d x %d of type %s need %d',
      [ShapeStr(ARaw), ARaw.Shape[0], AOutDim, AInDim, GgufTypeName(ATType), Want]);
  FRaw := ARaw;
  FTType := ATType;
  FOutDim := AOutDim;
  FInDim := AInDim;
end;

function TPackedLinear.Weight: TTensor;
begin
  Result := DequantizeT(FRaw, FTType, FOutDim * FInDim).view([FOutDim, FInDim]);
end;

function TPackedLinear.Forward(const X: TTensor): TTensor;
var
  S: TInt64Array;
begin
  S := X.Shape;
  if X.Kind <> kFloat then
    raise EPackedLinear.Create('X must be kFloat (fp32)');
  if (Length(S) = 0) or (S[High(S)] <> FInDim) then
    raise EPackedLinear.CreateFmt('X must end in %d, got %s', [FInDim, ShapeStr(X)]);
  Result := X.matmul(Weight.transpose(0, 1));
  if FLora <> nil then
    Result := Result + FLora.Delta(X);
end;

procedure TPackedLinear.SetLora(AValue: TLoraPair);
begin
  if (AValue <> nil) and ((AValue.InDim <> FInDim) or (AValue.OutDim <> FOutDim)) then
    raise EPackedLinear.CreateFmt('LoRA pair %s is %d -> %d, the matrix %d -> %d',
      [AValue.Name, AValue.InDim, AValue.OutDim, FInDim, FOutDim]);
  FLora := AValue;
end;

function TPackedLinear.Device: TDevice;
begin
  Result := FRaw.Device;
end;

function TensorRaw(G: TGgufFile; const AName: RawByteString; const ADevice: TDevice;
  out ATType: Cardinal; out ADims: TInt64Array): TTensor;
var
  I, J: Integer;
  Info: TGgufTensorInfo;
  Bytes: TBytes;
begin
  I := G.FindTensor(AName);
  if I < 0 then
    raise EPackedLinear.Create('tensor not in the file: ' + AName);
  Info := G.Tensor[I];
  G.ReadTensorBytes(I, Bytes);
  ATType := Info.TType;
  ADims := nil;
  SetLength(ADims, Info.NDims);
  for J := 0 to Info.NDims - 1 do
    ADims[J] := Info.Dims[Info.NDims - 1 - J];
  Result := TTensor.FromBytes(Bytes[0], [Length(Bytes)], kUInt8).&to(ADevice);
end;

function LoadPacked(G: TGgufFile; const AName: RawByteString; const ADevice: TDevice): TPackedLinear;
var
  T: Cardinal;
  D: TInt64Array;
  Raw: TTensor;
begin
  Raw := TensorRaw(G, AName, ADevice, T, D);
  if Length(D) <> 2 then
    raise EPackedLinear.CreateFmt('%s is not a matrix (%d dimensions)', [AName, Length(D)]);
  Result := TPackedLinear.Create(Raw, T, D[0], D[1]);
end;

function LoadDense(G: TGgufFile; const AName: RawByteString; const ADevice: TDevice): TTensor;
var
  T: Cardinal;
  D: TInt64Array;
  Raw: TTensor;
  N: Int64;
  J: Integer;
begin
  Raw := TensorRaw(G, AName, ADevice, T, D);
  N := 1;
  for J := 0 to High(D) do
    N := N * D[J];
  Result := DequantizeT(Raw, T, N).view(D);
end;

end.
