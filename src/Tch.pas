{ L3 core: TTensor as a record wrapping a refcounted handle (DESIGN.md 4.1),
  errors (4.2) and the FPU mask (4.2).

  Minimal first version for the ownership tests (HANDOFF step 6). The op
  methods marked "generated" are hand-written stand-ins in the shape the
  generator will emit; they move to TchTensorGenerated_*.inc in step 8. }
unit Tch;

{$mode objfpc}{$H+}
{$modeswitch advancedrecords}

interface

uses
  SysUtils, Math, TchSys;

type
  ETorchError = class(Exception);
  ETorchNativeError = class(ETorchError);

  TKind = (kUInt8 = 0, kInt8 = 1, kInt16 = 2, kInt = 3, kInt64 = 4,
           kHalf = 5, kFloat = 6, kDouble = 7, kComplexHalf = 8,
           kComplexFloat = 9, kComplexDouble = 10, kBool = 11,
           kQInt8 = 12, kQUInt8 = 13, kQInt32 = 14, kBFloat16 = 15,
           kFloat8e5m2 = 23, kFloat8e4m3fn = 24, kFloat8e5m2fnuz = 25,
           kFloat8e4m3fnuz = 26);

  TInt64Array = array of Int64;

  ITensorHandle = interface
    ['{6F0C2B7E-3D7A-4C1E-9E0B-2C8D5A41F7A3}']
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
    class function FromRaw(ARaw: PTorchTensor): TTensor; static;
    function Raw: PTorchTensor; inline;
    function IsNil: Boolean; inline;
    function Defined: Boolean;
    function Dim: Integer;
    function Shape: TInt64Array;
    function ToString(LineWidth: Integer = 80): AnsiString;
    function DoubleValue(const Index: array of Int64): Double;
    function Int64Value(const Index: array of Int64): Int64;
    function SameHandle(const Other: TTensor): Boolean; inline;

    class operator +(const A, B: TTensor): TTensor;
    class operator *(const A, B: TTensor): TTensor;

    // generated (stand-ins until step 8)
    function add(const other: TTensor): TTensor;
    function mul(const other: TTensor): TTensor;
    class function cat(const tensors: array of TTensor; dim_: Int64): TTensor;
      static;
    procedure max_dim(dim_: Int64; keepdim: Boolean;
      out values, indices: TTensor);
    class function ones(const size: array of Int64; options_kind: TKind;
      options_device: Integer): TTensor; static;
  end;

{ Reads and clears libtch's error slot; raises ETorchNativeError if set. }
procedure TchCheck; inline;
procedure TchRaiseNative(P: PChar);
{ Raises ETorchError if T has no handle (C would dereference nullptr). }
procedure TchCheckArg(const T: TTensor; const ParamName, OpName: String);
  inline;
procedure TchRaiseNilArg(const ParamName, OpName: String);
{ Masks all FPU exceptions on the calling thread (done for the main thread
  in initialization; TThreads inherit it, DESIGN.md 4.2 H4). }
procedure TchMaskFpuExceptions;
{ Number of live TTensorHandle objects, for leak and lifetime tests. }
function TchLiveHandles: Integer;

implementation

const
  CpuDevice = -1;

var
  LiveHandles: Integer = 0;

function TchLiveHandles: Integer;
begin
  Result := LiveHandles;
end;

procedure TchMaskFpuExceptions;
begin
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide, exOverflow,
    exUnderflow, exPrecision]);
end;

procedure TchRaiseNative(P: PChar);
var
  Msg: String;
begin
  Msg := StrPas(P);
  tchf_free(P);
  raise ETorchNativeError.Create(Msg);
end;

procedure TchCheck;
var
  P: PChar;
begin
  P := get_and_reset_last_err;
  if P <> nil then
    TchRaiseNative(P);
end;

procedure TchRaiseNilArg(const ParamName, OpName: String);
begin
  raise ETorchError.CreateFmt('undefined tensor passed as %s to %s',
    [ParamName, OpName]);
end;

procedure TchCheckArg(const T: TTensor; const ParamName, OpName: String);
begin
  if T.FHandle = nil then
    TchRaiseNilArg(ParamName, OpName);
end;

{ Frees the non-nil slots of a fixed-N output array after a failed call. }
procedure TchFreeSlots(Slots: PPTorchTensor; N: Integer);
var
  I: Integer;
begin
  for I := 0 to N - 1 do
    if Slots[I] <> nil then
      at_free(Slots[I]);
end;

procedure TchCheckSlots(Slots: PPTorchTensor; N: Integer);
var
  P: PChar;
begin
  P := get_and_reset_last_err;
  if P <> nil then
  begin
    TchFreeSlots(Slots, N);
    TchRaiseNative(P);
  end;
end;

{ TTensorHandle }

constructor TTensorHandle.Create(ARaw: PTorchTensor);
begin
  inherited Create;
  FRaw := ARaw;
  InterLockedIncrement(LiveHandles);
end;

destructor TTensorHandle.Destroy;
begin
  if FRaw <> nil then
    at_free(FRaw);
  InterLockedDecrement(LiveHandles);
  inherited Destroy;
end;

function TTensorHandle.Raw: PTorchTensor;
begin
  Result := FRaw;
end;

{ TTensor }

class function TTensor.FromRaw(ARaw: PTorchTensor): TTensor;
begin
  if ARaw = nil then
    raise ETorchError.Create('TTensor.FromRaw: nil pointer');
  Result.FHandle := TTensorHandle.Create(ARaw);
end;

function TTensor.Raw: PTorchTensor;
begin
  if FHandle = nil then
    Result := nil
  else
    Result := FHandle.Raw;
end;

function TTensor.IsNil: Boolean;
begin
  Result := FHandle = nil;
end;

function TTensor.SameHandle(const Other: TTensor): Boolean;
begin
  Result := FHandle = Other.FHandle;
end;

function TTensor.Defined: Boolean;
var
  R: Integer;
begin
  if FHandle = nil then
    Exit(False);
  R := at_defined(Raw);
  TchCheck;
  Result := R <> 0;
end;

function TTensor.Dim: Integer;
begin
  TchCheckArg(Self, 'self', 'Dim');
  Result := at_dim(Raw);
  TchCheck;
end;

function TTensor.Shape: TInt64Array;
begin
  Result := nil;
  SetLength(Result, Dim);
  if Length(Result) > 0 then
  begin
    at_shape(Raw, @Result[0]);
    TchCheck;
  end;
end;

function TTensor.ToString(LineWidth: Integer): AnsiString;
var
  P: PChar;
begin
  TchCheckArg(Self, 'self', 'ToString');
  P := at_to_string(Raw, LineWidth);
  TchCheck;
  Result := StrPas(P);
  tchf_free(P);
end;

function TTensor.DoubleValue(const Index: array of Int64): Double;
begin
  TchCheckArg(Self, 'self', 'DoubleValue');
  Result := at_double_value_at_indexes(Raw, PInt64(Index), Length(Index));
  TchCheck;
end;

function TTensor.Int64Value(const Index: array of Int64): Int64;
begin
  TchCheckArg(Self, 'self', 'Int64Value');
  Result := at_int64_value_at_indexes(Raw, PInt64(Index), Length(Index));
  TchCheck;
end;

class operator TTensor.+(const A, B: TTensor): TTensor;
begin
  Result := A.add(B);
end;

class operator TTensor.*(const A, B: TTensor): TTensor;
begin
  Result := A.mul(B);
end;

{ generated stand-ins }

function TTensor.add(const other: TTensor): TTensor;
var
  Outs: array[0..0] of PTorchTensor;
begin
  TchCheckArg(Self, 'self', 'add');
  TchCheckArg(other, 'other', 'add');
  Outs[0] := nil;
  atg_add(@Outs[0], Raw, other.Raw);
  TchCheckSlots(@Outs[0], 1);
  Result := TTensor.FromRaw(Outs[0]);
end;

function TTensor.mul(const other: TTensor): TTensor;
var
  Outs: array[0..0] of PTorchTensor;
begin
  TchCheckArg(Self, 'self', 'mul');
  TchCheckArg(other, 'other', 'mul');
  Outs[0] := nil;
  atg_mul(@Outs[0], Raw, other.Raw);
  TchCheckSlots(@Outs[0], 1);
  Result := TTensor.FromRaw(Outs[0]);
end;

class function TTensor.cat(const tensors: array of TTensor;
  dim_: Int64): TTensor;
var
  Outs: array[0..0] of PTorchTensor;
  Raws: array of PTorchTensor;
  I: Integer;
begin
  Raws := nil;
  SetLength(Raws, Length(tensors));
  for I := 0 to High(tensors) do
  begin
    TchCheckArg(tensors[I], 'tensors[' + IntToStr(I) + ']', 'cat');
    Raws[I] := tensors[I].Raw;
  end;
  Outs[0] := nil;
  atg_cat(@Outs[0], PPTorchTensor(Raws), Length(Raws), dim_);
  TchCheckSlots(@Outs[0], 1);
  Result := TTensor.FromRaw(Outs[0]);
end;

procedure TTensor.max_dim(dim_: Int64; keepdim: Boolean;
  out values, indices: TTensor);
var
  Outs: array[0..1] of PTorchTensor;
begin
  TchCheckArg(Self, 'self', 'max_dim');
  Outs[0] := nil;
  Outs[1] := nil;
  atg_max_dim(@Outs[0], Raw, dim_, Ord(keepdim));
  TchCheckSlots(@Outs[0], 2);
  values := TTensor.FromRaw(Outs[0]);
  indices := TTensor.FromRaw(Outs[1]);
end;

class function TTensor.ones(const size: array of Int64; options_kind: TKind;
  options_device: Integer): TTensor;
var
  Outs: array[0..0] of PTorchTensor;
begin
  Outs[0] := nil;
  atg_ones(@Outs[0], PInt64(size), Length(size), Ord(options_kind),
    options_device);
  TchCheckSlots(@Outs[0], 1);
  Result := TTensor.FromRaw(Outs[0]);
end;

initialization
  TchMaskFpuExceptions;
end.
