{ T10: a matrix kept packed on a device (the GGUF's own bytes), dequantised
  per call to fp32 by GgufQuantT.DequantizeT (DESIGN 8.12, "PackedLinear").
  Shapes in libtorch order [Out, In]; a GGUF matrix lists its dims as
  (In, Out). LoadPacked / LoadDense read a tensor of a TGgufFile. LoRA is
  added in T12. Written without reference to Qwen.

  Interface by the harness owner (tasks/T10-facts.md); the bodies are
  written in T10b. }
unit PackedLinear;

{$mode objfpc}{$H+}

interface

uses
  SysUtils, Tch, GgufFile, GgufQuantT;

type
  EPackedLinear = class(Exception);

  TPackedLinear = class
  private
    FRaw: TTensor;
    FTType: Cardinal;
    FOutDim, FInDim: Int64;
  public
    constructor Create(const ARaw: TTensor; ATType: Cardinal; AOutDim, AInDim: Int64);
    function Weight: TTensor;
    function Forward(const X: TTensor): TTensor;
    function Device: TDevice;
    property TType: Cardinal read FTType;
    property OutDim: Int64 read FOutDim;
    property InDim: Int64 read FInDim;
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
begin
  inherited Create;
  raise EPackedLinear.Create('TPackedLinear.Create: not implemented yet');
end;

function TPackedLinear.Weight: TTensor;
begin
  Result := Default(TTensor);
  raise EPackedLinear.Create('TPackedLinear.Weight: not implemented yet');
end;

function TPackedLinear.Forward(const X: TTensor): TTensor;
begin
  Result := Default(TTensor);
  raise EPackedLinear.Create('TPackedLinear.Forward: not implemented yet');
end;

function TPackedLinear.Device: TDevice;
begin
  Result := Default(TDevice);
  raise EPackedLinear.Create('TPackedLinear.Device: not implemented yet');
end;

function TensorRaw(G: TGgufFile; const AName: RawByteString; const ADevice: TDevice;
  out ATType: Cardinal; out ADims: TInt64Array): TTensor;
begin
  ATType := 0;
  ADims := nil;
  Result := Default(TTensor);
  raise EPackedLinear.Create('TensorRaw: not implemented yet');
end;

function LoadPacked(G: TGgufFile; const AName: RawByteString; const ADevice: TDevice): TPackedLinear;
begin
  Result := nil;
  raise EPackedLinear.Create('LoadPacked: not implemented yet');
end;

function LoadDense(G: TGgufFile; const AName: RawByteString; const ADevice: TDevice): TTensor;
begin
  Result := Default(TTensor);
  raise EPackedLinear.Create('LoadDense: not implemented yet');
end;

end.
