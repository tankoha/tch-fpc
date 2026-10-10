{ L3 core: TTensor as a record wrapping a refcounted handle (DESIGN.md 4.1),
  errors (4.2), the FPU mask (4.2), the small value types (4.3) and the
  generated op methods (4.4, pulled in from TchTensorGenerated_*.inc). }
unit Tch;

{$mode objfpc}{$H+}
{$modeswitch advancedrecords}
{ duplicatelocals: a method parameter may have the same name as a TTensor
  member (dim, indices, pow, ...), as in Delphi mode. Generated ops keep the
  PyTorch parameter names verbatim because of this (DESIGN.md 4.4, H12).
  Inside a method body the bare name is the parameter; use Self.X to reach
  the member. Local variables still may not shadow members. }
{$modeswitch duplicatelocals}

interface

uses
  SysUtils, Math, ctypes, TchSys, TchSysGenerated;

type
  ETorchError = class(Exception);
  { An error reported by libtorch. Message is the text before the C++
    backtrace ("Exception raised from ..."); Detail is the full text. }
  ETorchNativeError = class(ETorchError)
  private
    FDetail: String;
  public
    constructor Create(const AFull: String);
    property Detail: String read FDetail;
  end;

  { c10::ScalarType values of libtorch 2.13 (torch/headeronly/core/ScalarType.h).
    kNone (-1) is "no dtype" for optional dtype parameters only; TTensor.Kind
    never returns it. Values 0..26 verified against at_scalar_type in step 4. }
  TKind = (kNone = -1,
           kUInt8 = 0, kInt8 = 1, kInt16 = 2, kInt = 3, kInt64 = 4,
           kHalf = 5, kFloat = 6, kDouble = 7, kComplexHalf = 8,
           kComplexFloat = 9, kComplexDouble = 10, kBool = 11,
           kQInt8 = 12, kQUInt8 = 13, kQInt32 = 14, kBFloat16 = 15,
           kQUInt4x2 = 16, kQUInt2x4 = 17, kBits1x8 = 18, kBits2x4 = 19,
           kBits4x2 = 20, kBits8 = 21, kBits16 = 22,
           kFloat8e5m2 = 23, kFloat8e4m3fn = 24, kFloat8e5m2fnuz = 25,
           kFloat8e4m3fnuz = 26, kUInt16 = 27, kUInt32 = 28, kUInt64 = 29,
           kUInt1 = 30, kUInt2 = 31, kUInt3 = 32, kUInt4 = 33, kUInt5 = 34,
           kUInt6 = 35, kUInt7 = 36, kInt1 = 37, kInt2 = 38, kInt3 = 39,
           kInt4 = 40, kInt5 = 41, kInt6 = 42, kInt7 = 43,
           kFloat8e8m0fnu = 44, kFloat4e2m1fnx2 = 45);

  { c10::Layout (torch/headeronly/core/Layout.h); ltNone (-1) = no layout,
    for optional layout parameters only. }
  TLayout = (ltNone = -1, ltStrided = 0, ltSparse = 1, ltSparseCsr = 2,
             ltMkldnn = 3, ltSparseCsc = 4, ltSparseBsr = 5, ltSparseBsc = 6,
             ltJagged = 7);

  TDeviceType = (dtCpu, dtCuda, dtMps, dtVulkan);

  TDevice = record
    DeviceType: TDeviceType;
    Index: Integer;               // CUDA only
    function ToC: cint;           // -1 CPU, >= 0 CUDA index, -2 MPS, -3 Vulkan
    class function FromC(V: cint): TDevice; static;
    class function Cpu: TDevice; static;
    class function Cuda(AIndex: Integer = 0): TDevice; static;
    function IsCuda: Boolean;
  end;

  TInt64Array  = array of Int64;
  TDoubleArray = array of Double;
  TSingleArray = array of Single;

  { A number passed to an op that takes at::Scalar; converted to a temporary
    ats_* handle for the call. `T + 1` and `T + 1.5` both convert. }
  TScalar = record
    IsFloat: Boolean;
    I: Int64;
    F: Double;
    class operator :=(V: Int64): TScalar;
    class operator :=(V: Double): TScalar;
  end;

  TOptInt64 = record
    HasValue: Boolean;
    Value: Int64;
    class operator :=(V: Int64): TOptInt64;
  end;

  TOptDouble = record
    HasValue: Boolean;
    Value: Double;
    class operator :=(V: Double): TOptDouble;
  end;

  { An optional list of integers; "none" and "empty" are different
    (DESIGN.md 4.3). Build with IntList([..]) or use NoIntList. }
  TOptIntList = record
    HasValue: Boolean;
    Value: TInt64Array;
  end;

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

  TTensorPtrArray = array of PTorchTensor;

  TTensor = record
  public type
    { Result type of ops returning a variable number of tensors (split,
      unbind, ...); nested because an array of TTensor cannot be declared
      before TTensor. TTensorArray below is an alias. }
    TTensorArray = array of TTensor;
  private
    FHandle: ITensorHandle;   // nil = no tensor at all
  public
    class function FromRaw(ARaw: PTorchTensor): TTensor; static;
    function Raw: PTorchTensor; inline;
    function IsNil: Boolean; inline;
    function Defined: Boolean;
    function Dim: Integer;
    function Shape: TInt64Array;
    function Kind: TKind;
    function ToString(LineWidth: Integer = 80): AnsiString;
    function DoubleValue(const Index: array of Int64): Double;
    function Int64Value(const Index: array of Int64): Int64;
    function SameHandle(const Other: TTensor): Boolean; inline;

    class operator +(const A, B: TTensor): TTensor;
    class operator *(const A, B: TTensor): TTensor;

    {$I TchTensorGenerated_intf.inc}
  end;

  TTensorArray = TTensor.TTensorArray;

const
  NoInt64:   TOptInt64   = (HasValue: False; Value: 0);
  NoDouble:  TOptDouble  = (HasValue: False; Value: 0);
  NoIntList: TOptIntList = (HasValue: False; Value: nil);

function IntList(const A: array of Int64): TOptIntList;  // present, possibly empty

{ Reads and clears libtch's error slot; raises ETorchNativeError if set. }
procedure TchCheck; inline;
procedure TchRaiseNative(P: PChar);
{ Raises ETorchError if T has no handle (C would dereference nullptr). }
procedure TchCheckArg(const T: TTensor; const ParamName, OpName: String);
  inline;
procedure TchRaiseNilArg(const ParamName, OpName: String);
{ Helpers for generated bodies (prefixed Tch so no parameter name shadows them). }
procedure TchCheckSlots(Slots: PPTorchTensor; N: Integer);
procedure TchFreeSlots(Slots: PPTorchTensor; N: Integer);
function TchScalarToC(const S: TScalar): PTorchScalar;
function TchTensorPtrs(const L: array of TTensor; const ParamName, OpName: String): TTensorPtrArray;
function TchTensorOptPtrs(const L: array of TTensor): TTensorPtrArray;
function TchOptIntListPtr(const L: TOptIntList): PInt64;
function TchTakeTensorArray(P: PPTorchTensor): TTensorArray;
{ Masks all FPU exceptions on the calling thread (done for the main thread
  in initialization; TThreads inherit it, DESIGN.md 4.2 H4). }
procedure TchMaskFpuExceptions;
{ Number of live TTensorHandle objects, for leak and lifetime tests. }
function TchLiveHandles: Integer;

implementation

var
  LiveHandles: Integer = 0;
  EmptyInt64: Int64 = 0;   // non-nil pointer for a present-but-empty IntListOption

function TchLiveHandles: Integer;
begin
  Result := LiveHandles;
end;

procedure TchMaskFpuExceptions;
begin
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide, exOverflow,
    exUnderflow, exPrecision]);
end;

{ ETorchNativeError }

constructor ETorchNativeError.Create(const AFull: String);
const
  Marker = #10'Exception raised from ';
var
  P: Integer;
begin
  FDetail := AFull;
  P := Pos(Marker, AFull);
  if P > 0 then
    inherited Create(Copy(AFull, 1, P - 1))
  else
    inherited Create(AFull);
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

{ Frees the non-nil slots of a fixed-N output array. }
procedure TchFreeSlots(Slots: PPTorchTensor; N: Integer);
var
  I: Integer;
begin
  for I := 0 to N - 1 do
    if Slots[I] <> nil then
      at_free(Slots[I]);
end;

{ After a fixed-N call: on error, frees whatever was filled and raises. }
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

function TchScalarToC(const S: TScalar): PTorchScalar;
begin
  if S.IsFloat then
    Result := ats_float(S.F)
  else
    Result := ats_int(S.I);
  TchCheck;
end;

function TchTensorPtrs(const L: array of TTensor; const ParamName, OpName: String): TTensorPtrArray;
var
  I: Integer;
begin
  Result := nil;
  SetLength(Result, Length(L));
  for I := 0 to High(L) do
  begin
    if L[I].FHandle = nil then
      TchRaiseNilArg(ParamName + '[' + IntToStr(I) + ']', OpName);
    Result[I] := L[I].Raw;
  end;
end;

function TchTensorOptPtrs(const L: array of TTensor): TTensorPtrArray;
var
  I: Integer;
begin
  Result := nil;
  SetLength(Result, Length(L));
  for I := 0 to High(L) do
    Result[I] := L[I].Raw;   // nil = none
end;

function TchOptIntListPtr(const L: TOptIntList): PInt64;
begin
  if not L.HasValue then
    Result := nil
  else if Length(L.Value) = 0 then
    Result := @EmptyInt64
  else
    Result := @L.Value[0];
end;

{ Takes ownership of a malloc'd, nil-terminated array of new tensor handles
  (dynamic results) and of the array itself. }
function TchTakeTensorArray(P: PPTorchTensor): TTensorArray;
var
  N, I: Integer;
begin
  Result := nil;
  if P = nil then
    Exit;
  N := 0;
  while P[N] <> nil do
    Inc(N);
  SetLength(Result, N);
  for I := 0 to N - 1 do
    Result[I] := TTensor.FromRaw(P[I]);
  tchf_free(P);
end;

function IntList(const A: array of Int64): TOptIntList;
var
  I: Integer;
begin
  Result.HasValue := True;
  Result.Value := nil;
  SetLength(Result.Value, Length(A));
  for I := 0 to High(A) do
    Result.Value[I] := A[I];
end;

{ TDevice }

function TDevice.ToC: cint;
begin
  case DeviceType of
    dtCpu: Result := -1;
    dtCuda: Result := Index;
    dtMps: Result := -2;
    dtVulkan: Result := -3;
  else
    Result := -1;
  end;
end;

class function TDevice.FromC(V: cint): TDevice;
begin
  Result.Index := 0;
  case V of
    -1: Result.DeviceType := dtCpu;
    -2: Result.DeviceType := dtMps;
    -3: Result.DeviceType := dtVulkan;
  else
    if V >= 0 then
    begin
      Result.DeviceType := dtCuda;
      Result.Index := V;
    end
    else
      raise ETorchError.CreateFmt('unknown device code %d', [V]);
  end;
end;

class function TDevice.Cpu: TDevice;
begin
  Result.DeviceType := dtCpu;
  Result.Index := 0;
end;

class function TDevice.Cuda(AIndex: Integer): TDevice;
begin
  Result.DeviceType := dtCuda;
  Result.Index := AIndex;
end;

function TDevice.IsCuda: Boolean;
begin
  Result := DeviceType = dtCuda;
end;

{ TScalar, TOpt* }

class operator TScalar.:=(V: Int64): TScalar;
begin
  Result.IsFloat := False;
  Result.I := V;
  Result.F := 0;
end;

class operator TScalar.:=(V: Double): TScalar;
begin
  Result.IsFloat := True;
  Result.I := 0;
  Result.F := V;
end;

class operator TOptInt64.:=(V: Int64): TOptInt64;
begin
  Result.HasValue := True;
  Result.Value := V;
end;

class operator TOptDouble.:=(V: Double): TOptDouble;
begin
  Result.HasValue := True;
  Result.Value := V;
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

function TTensor.Kind: TKind;
var
  K: cint;
begin
  TchCheckArg(Self, 'self', 'Kind');
  K := at_scalar_type(Raw);
  TchCheck;
  if (K < Ord(kUInt8)) or (K > Ord(kFloat4e2m1fnx2)) then
    raise ETorchError.CreateFmt('unknown scalar type %d', [K]);
  Result := TKind(K);
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

{$I TchTensorGenerated_impl.inc}

initialization
  TchMaskFpuExceptions;
end.
