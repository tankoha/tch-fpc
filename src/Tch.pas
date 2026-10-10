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
    class function CudaIfAvailable: TDevice; static;
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
    function Numel: Int64;
    function Device: TDevice;
    { New handle to the same storage (B := A shares the handle itself). }
    function ShallowClone: TTensor;
    { New storage with a copy of the data (zeros_like + copy_). }
    function Copy: TTensor;

    { Data in: a 1-d tensor of the array's length. [1.0, 2.0] picks Single
      (kFloat), [1, 2] Int64 (kInt64), as PyTorch's defaults (H2). Int32
      data: FromInt32Array (an Int32 overload would make [1, 2] ambiguous). }
    class function FromArray(const A: array of Single): TTensor; overload; static;
    class function FromArray(const A: array of Double): TTensor; overload; static;
    class function FromArray(const A: array of Int64): TTensor; overload; static;
    class function FromArray(const A: array of Byte): TTensor; overload; static;
    class function FromArray(const A: array of Boolean): TTensor; overload; static;
    class function FromInt32Array(const A: array of Int32): TTensor; static;
    { Count = product of AShape elements of AKind's size, read from Data. }
    class function FromBytes(const Data; const AShape: array of Int64;
      AKind: TKind): TTensor; static;
    { Data out: converts to the element type first if Kind differs. }
    function ToSingleArray: TSingleArray;
    function ToDoubleArray: TDoubleArray;
    function ToInt64Array: TInt64Array;
    { Copies the first ACount elements, in this tensor's own Kind (no
      conversion; element size from KindElementSize), to Dest. }
    procedure CopyData(var Dest; ACount: Int64);

    { Autograd. Grad, requires_grad_ and set_requires_grad are generated. }
    procedure Backward(KeepGraph: Boolean = False; CreateGraph: Boolean = False);
    function RequiresGrad: Boolean;
    { Zeroes the gradient if there is one (grad.detach_.zero_). }
    procedure ZeroGrad;

    { Convenience overloads of generated factories (DESIGN.md 4.4): CPU,
      kFloat unless given, like tch-rs's FLOAT_CPU. }
    class function zeros(const ASize: array of Int64; AKind: TKind = kFloat): TTensor; overload; static;
    class function ones(const ASize: array of Int64; AKind: TKind = kFloat): TTensor; overload; static;
    class function empty(const ASize: array of Int64; AKind: TKind = kFloat): TTensor; overload; static;
    class function rand(const ASize: array of Int64; AKind: TKind = kFloat): TTensor; overload; static;
    class function randn(const ASize: array of Int64; AKind: TKind = kFloat): TTensor; overload; static;

    { Operators (DESIGN.md 4.1). * is elementwise; use matmul for the
      matrix product. Double and Int64 overloads in both orders (H3). }
    class operator +(const A, B: TTensor): TTensor;
    class operator -(const A, B: TTensor): TTensor;
    class operator *(const A, B: TTensor): TTensor;
    class operator /(const A, B: TTensor): TTensor;
    class operator -(const A: TTensor): TTensor;
    class operator **(const A: TTensor; B: Double): TTensor;
    class operator +(const A: TTensor; B: Double): TTensor;
    class operator +(const A: TTensor; B: Int64): TTensor;
    class operator +(A: Double; const B: TTensor): TTensor;
    class operator +(A: Int64; const B: TTensor): TTensor;
    class operator -(const A: TTensor; B: Double): TTensor;
    class operator -(const A: TTensor; B: Int64): TTensor;
    class operator -(A: Double; const B: TTensor): TTensor;
    class operator -(A: Int64; const B: TTensor): TTensor;
    class operator *(const A: TTensor; B: Double): TTensor;
    class operator *(const A: TTensor; B: Int64): TTensor;
    class operator *(A: Double; const B: TTensor): TTensor;
    class operator *(A: Int64; const B: TTensor): TTensor;
    class operator /(const A: TTensor; B: Double): TTensor;
    class operator /(const A: TTensor; B: Int64): TTensor;
    class operator /(A: Double; const B: TTensor): TTensor;
    class operator /(A: Int64; const B: TTensor): TTensor;

    {$I TchTensorGenerated_intf.inc}
  end;

  TTensorArray = TTensor.TTensorArray;

const
  NoInt64:   TOptInt64   = (HasValue: False; Value: 0);
  NoDouble:  TOptDouble  = (HasValue: False; Value: 0);
  NoIntList: TOptIntList = (HasValue: False; Value: nil);

function IntList(const A: array of Int64): TOptIntList;  // present, possibly empty

{ Bytes per element of AKind, as libtorch's element_size(); ETorchError for
  kNone. }
function KindElementSize(AKind: TKind): Integer;

{ Grad mode (DESIGN.md 4.7); thread-local in libtorch. Returns the
  previous mode. }
function SetGradEnabled(Enabled: Boolean): Boolean;
{ Disables grad mode until the result is released, then restores the
  previous mode. Keep it in a local: `G := NoGradGuard;` covers the rest of
  the routine (H11). A bare `NoGradGuard;` is released at once and has no
  effect. }
function NoGradGuard: IInterface;
procedure ManualSeed(Seed: Int64);
procedure SetNumThreads(N: Integer);
function GetNumThreads: Integer;
function CudaIsAvailable: Boolean;
function CudaDeviceCount: Integer;

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

class function TDevice.CudaIfAvailable: TDevice;
begin
  if CudaIsAvailable then
    Result := Cuda(0)
  else
    Result := Cpu;
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

function TTensor.Numel: Int64;
var
  S: TInt64Array;
  K: Integer;
begin
  S := Shape;
  Result := 1;
  for K := 0 to High(S) do
    Result := Result * S[K];
end;

function TTensor.Device: TDevice;
var
  D: cint;
begin
  TchCheckArg(Self, 'self', 'Device');
  D := at_device(Raw);
  TchCheck;
  Result := TDevice.FromC(D);
end;

function TTensor.ShallowClone: TTensor;
var
  P: PTorchTensor;
begin
  TchCheckArg(Self, 'self', 'ShallowClone');
  P := at_shallow_clone(Raw);
  TchCheck;
  Result := FromRaw(P);
end;

function TTensor.Copy: TTensor;
begin
  Result := zeros_like;
  at_copy_(Result.Raw, Raw);
  TchCheck;
end;

{ Data in / out }

function TensorOfData(Data: Pointer; N: Int64; ElementSize: Integer;
  AKind: TKind): TTensor;
var
  P: PTorchTensor;
begin
  P := at_tensor_of_data(Data, @N, 1, ElementSize, Ord(AKind));
  TchCheck;
  Result := TTensor.FromRaw(P);
end;

class function TTensor.FromArray(const A: array of Single): TTensor;
begin
  Result := TensorOfData(@A, System.Length(A), SizeOf(Single), kFloat);
end;

class function TTensor.FromArray(const A: array of Double): TTensor;
begin
  Result := TensorOfData(@A, System.Length(A), SizeOf(Double), kDouble);
end;

class function TTensor.FromArray(const A: array of Int64): TTensor;
begin
  Result := TensorOfData(@A, System.Length(A), SizeOf(Int64), kInt64);
end;

class function TTensor.FromArray(const A: array of Byte): TTensor;
begin
  Result := TensorOfData(@A, System.Length(A), SizeOf(Byte), kUInt8);
end;

{ Pascal Boolean is one byte holding 0 or 1, the layout of c10's bool. }
class function TTensor.FromArray(const A: array of Boolean): TTensor;
begin
  Result := TensorOfData(@A, System.Length(A), SizeOf(Boolean), kBool);
end;

class function TTensor.FromInt32Array(const A: array of Int32): TTensor;
begin
  Result := TensorOfData(@A, System.Length(A), SizeOf(Int32), kInt);
end;

class function TTensor.FromBytes(const Data; const AShape: array of Int64;
  AKind: TKind): TTensor;
var
  P: PTorchTensor;
begin
  P := at_tensor_of_data(@Data, PInt64(AShape), System.Length(AShape),
    KindElementSize(AKind), Ord(AKind));
  TchCheck;
  Result := FromRaw(P);
end;

procedure TTensor.CopyData(var Dest; ACount: Int64);
begin
  TchCheckArg(Self, 'self', 'CopyData');
  if ACount < 0 then
    raise ETorchError.CreateFmt('CopyData: negative count %d', [ACount]);
  at_copy_data(Raw, @Dest, ACount, KindElementSize(Kind));
  TchCheck;
end;

{ Self converted to AKind, or Self when it already is. }
function AsKind(const T: TTensor; AKind: TKind): TTensor;
begin
  if T.Kind = AKind then
    Result := T
  else
    Result := T.to_dtype(AKind);
end;

function TTensor.ToSingleArray: TSingleArray;
var
  Src: TTensor;
begin
  Src := AsKind(Self, kFloat);
  Result := nil;
  SetLength(Result, Src.Numel);
  if System.Length(Result) > 0 then
    Src.CopyData(Result[0], System.Length(Result));
end;

function TTensor.ToDoubleArray: TDoubleArray;
var
  Src: TTensor;
begin
  Src := AsKind(Self, kDouble);
  Result := nil;
  SetLength(Result, Src.Numel);
  if System.Length(Result) > 0 then
    Src.CopyData(Result[0], System.Length(Result));
end;

function TTensor.ToInt64Array: TInt64Array;
var
  Src: TTensor;
begin
  Src := AsKind(Self, kInt64);
  Result := nil;
  SetLength(Result, Src.Numel);
  if System.Length(Result) > 0 then
    Src.CopyData(Result[0], System.Length(Result));
end;

{ Autograd }

procedure TTensor.Backward(KeepGraph: Boolean; CreateGraph: Boolean);
begin
  TchCheckArg(Self, 'self', 'Backward');
  at_backward(Raw, System.Ord(KeepGraph), System.Ord(CreateGraph));
  TchCheck;
end;

function TTensor.RequiresGrad: Boolean;
var
  R: cint;
begin
  TchCheckArg(Self, 'self', 'RequiresGrad');
  R := at_requires_grad(Raw);
  TchCheck;
  Result := R <> 0;
end;

procedure TTensor.ZeroGrad;
var
  G: TTensor;
begin
  G := grad;
  if G.Defined then
    G.detach_.zero_;
end;

{ Factory overloads }

class function TTensor.zeros(const ASize: array of Int64; AKind: TKind): TTensor;
begin
  Result := zeros(ASize, AKind, TDevice.Cpu);
end;

class function TTensor.ones(const ASize: array of Int64; AKind: TKind): TTensor;
begin
  Result := ones(ASize, AKind, TDevice.Cpu);
end;

class function TTensor.empty(const ASize: array of Int64; AKind: TKind): TTensor;
begin
  Result := empty(ASize, AKind, TDevice.Cpu);
end;

class function TTensor.rand(const ASize: array of Int64; AKind: TKind): TTensor;
begin
  Result := rand(ASize, AKind, TDevice.Cpu);
end;

class function TTensor.randn(const ASize: array of Int64; AKind: TKind): TTensor;
begin
  Result := randn(ASize, AKind, TDevice.Cpu);
end;

{ Operators }

class operator TTensor.+(const A, B: TTensor): TTensor;
begin
  Result := A.add(B);
end;

class operator TTensor.-(const A, B: TTensor): TTensor;
begin
  Result := A.sub(B);
end;

class operator TTensor.*(const A, B: TTensor): TTensor;
begin
  Result := A.mul(B);
end;

class operator TTensor./(const A, B: TTensor): TTensor;
begin
  Result := A.&div(B);
end;

class operator TTensor.-(const A: TTensor): TTensor;
begin
  Result := A.neg;
end;

class operator TTensor.**(const A: TTensor; B: Double): TTensor;
begin
  Result := A.pow_tensor_scalar(B);
end;

class operator TTensor.+(const A: TTensor; B: Double): TTensor;
begin
  Result := A.add_scalar(B);
end;

class operator TTensor.+(const A: TTensor; B: Int64): TTensor;
begin
  Result := A.add_scalar(B);
end;

class operator TTensor.+(A: Double; const B: TTensor): TTensor;
begin
  Result := B.add_scalar(A);
end;

class operator TTensor.+(A: Int64; const B: TTensor): TTensor;
begin
  Result := B.add_scalar(A);
end;

class operator TTensor.-(const A: TTensor; B: Double): TTensor;
begin
  Result := A.sub_scalar(B);
end;

class operator TTensor.-(const A: TTensor; B: Int64): TTensor;
begin
  Result := A.sub_scalar(B);
end;

{ d - A is rsub (other - self), one op instead of tch-rs's (-A) + d. }
class operator TTensor.-(A: Double; const B: TTensor): TTensor;
begin
  Result := B.rsub_scalar(A);
end;

class operator TTensor.-(A: Int64; const B: TTensor): TTensor;
begin
  Result := B.rsub_scalar(A);
end;

class operator TTensor.*(const A: TTensor; B: Double): TTensor;
begin
  Result := A.mul_scalar(B);
end;

class operator TTensor.*(const A: TTensor; B: Int64): TTensor;
begin
  Result := A.mul_scalar(B);
end;

class operator TTensor.*(A: Double; const B: TTensor): TTensor;
begin
  Result := B.mul_scalar(A);
end;

class operator TTensor.*(A: Int64; const B: TTensor): TTensor;
begin
  Result := B.mul_scalar(A);
end;

class operator TTensor./(const A: TTensor; B: Double): TTensor;
begin
  Result := A.div_scalar(B);
end;

class operator TTensor./(const A: TTensor; B: Int64): TTensor;
begin
  Result := A.div_scalar(B);
end;

{ d / A as reciprocal * d: reciprocal promotes integer tensors to float, as
  true division does (pow(-1), tch-rs's choice, rejects integer tensors). }
class operator TTensor./(A: Double; const B: TTensor): TTensor;
begin
  Result := B.reciprocal.mul_scalar(A);
end;

class operator TTensor./(A: Int64; const B: TTensor): TTensor;
begin
  Result := B.reciprocal.mul_scalar(A);
end;

{ Global settings (DESIGN.md 4.7) }

function KindElementSize(AKind: TKind): Integer;
begin
  case AKind of
    kUInt8, kInt8, kBool, kQInt8, kQUInt8, kQUInt4x2, kQUInt2x4,
    kBits1x8, kBits2x4, kBits4x2, kBits8,
    kFloat8e5m2, kFloat8e4m3fn, kFloat8e5m2fnuz, kFloat8e4m3fnuz,
    kUInt1..kUInt7, kInt1..kInt7, kFloat8e8m0fnu, kFloat4e2m1fnx2:
      Result := 1;
    kInt16, kHalf, kBFloat16, kBits16, kUInt16:
      Result := 2;
    kInt, kFloat, kComplexHalf, kQInt32, kUInt32:
      Result := 4;
    kInt64, kDouble, kComplexFloat, kUInt64:
      Result := 8;
    kComplexDouble:
      Result := 16;
  else
    raise ETorchError.CreateFmt('no element size for kind %d', [Ord(AKind)]);
  end;
end;

function SetGradEnabled(Enabled: Boolean): Boolean;
var
  R: cint;
begin
  R := at_grad_set_enabled(System.Ord(Enabled));
  TchCheck;
  Result := R <> 0;
end;

type
  TGradModeGuard = class(TInterfacedObject)
  private
    FPrevious: Boolean;
  public
    constructor Create;
    destructor Destroy; override;
  end;

constructor TGradModeGuard.Create;
begin
  inherited Create;
  FPrevious := SetGradEnabled(False);
end;

{ Never raises (may run during unwinding); a failure stays in the error
  slot and surfaces at the next checked call. }
destructor TGradModeGuard.Destroy;
begin
  at_grad_set_enabled(System.Ord(FPrevious));
  inherited Destroy;
end;

function NoGradGuard: IInterface;
begin
  Result := TGradModeGuard.Create;
end;

procedure ManualSeed(Seed: Int64);
begin
  at_manual_seed(Seed);
end;

procedure SetNumThreads(N: Integer);
begin
  at_set_num_threads(N);
  TchCheck;
end;

function GetNumThreads: Integer;
begin
  Result := at_get_num_threads;
  TchCheck;
end;

function CudaIsAvailable: Boolean;
var
  R: cint;
begin
  R := atc_cuda_is_available;
  TchCheck;
  Result := R > 0;
end;

function CudaDeviceCount: Integer;
begin
  Result := atc_cuda_device_count;
  TchCheck;
end;

{$I TchTensorGenerated_impl.inc}

initialization
  TchMaskFpuExceptions;
end.
