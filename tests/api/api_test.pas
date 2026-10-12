{ Hand-written API of milestone 1 (DESIGN.md 4.1, 4.3, 4.7, section 6)
  against the real libtorch: data in/out, Numel/Device/ShallowClone/Copy,
  operators, factory overloads, autograd and grad mode, global settings,
  and KindElementSize checked against libtorch's element_size for every
  kind. Expected values are worked out by hand in the comments.

  Build (from the repo root, after building build/libtch):
    fpc -Mobjfpc -Sh -Fusrc -FUbuild/api -Flbuild/libtch \
      -k-rpath=$PWD/build/libtch tests/api/api_test.pas -obuild/api_test
    build/api_test }
program api_test;

uses
  SysUtils, TypInfo, Tch;

var
  Failures: Integer = 0;

procedure Check(Cond: Boolean; const What: String);
begin
  if Cond then
    WriteLn('  ok    ', What)
  else
  begin
    WriteLn('  FAIL  ', What);
    Inc(Failures);
  end;
end;

function Near(A, B: Double): Boolean;
begin
  Result := Abs(A - B) < 1e-5;
end;

function ShapeStr(const T: TTensor): String;
var
  S: TInt64Array;
  K: Integer;
begin
  S := T.Shape;
  Result := '[';
  for K := 0 to High(S) do
  begin
    if K > 0 then
      Result := Result + ',';
    Result := Result + IntToStr(S[K]);
  end;
  Result := Result + ']';
end;

function Values(const T: TTensor): String;
var
  D: TDoubleArray;
  K: Integer;
begin
  D := T.ToDoubleArray;
  Result := '';
  for K := 0 to High(D) do
  begin
    if K > 0 then
      Result := Result + ' ';
    Result := Result + FloatToStr(D[K]);
  end;
end;

{ X = [[0, 1, 2], [3, 4, 5]] as Float }
function MakeX: TTensor;
begin
  Result := TTensor.FromArray([0.0, 1, 2, 3, 4, 5]).reshape([2, 3]);
end;

procedure TestDataIn;
var
  T: TTensor;
  Empty: TDoubleArray;
  I32: array[0..2] of Int32 = (7, -8, 9);
  Raw: array[0..3] of Double = (1.5, 2.5, 3.5, 4.5);
begin
  WriteLn('data in: FromArray overloads, FromSingleArray, FromInt32Array, FromBytes');
  T := TTensor.FromArray([1.0, 2.0]);
  Check((T.Kind = kFloat) and (ShapeStr(T) = '[2]'), '[1.0, 2.0]: kFloat, [2]');
  T := TTensor.FromArray([1, 2, 3]);
  Check((T.Kind = kInt64) and (T.Int64Value([2]) = 3), '[1, 2, 3]: kInt64');
  T := TTensor.FromArray(TDoubleArray.Create(0.25, 0.5));
  Check((T.Kind = kFloat) and Near(T.DoubleValue([1]), 0.5), 'TDoubleArray: kFloat by default');
  T := TTensor.FromArray(TDoubleArray.Create(0.25, 0.5), kDouble);
  Check((T.Kind = kDouble) and Near(T.DoubleValue([1]), 0.5), 'TDoubleArray, kDouble');
  { 0.9 has no exact binary32 form: the literal used to be ambiguous between
    the Single and Double overloads; now Double, rounded once to float32 }
  T := TTensor.FromArray([0.9, -1.9]);
  Check((T.Kind = kFloat) and (T.DoubleValue([0]) = Double(Single(0.9)))
    and (T.DoubleValue([1]) = Double(Single(-1.9))), '[0.9, -1.9]: kFloat, float32 bits');
  T := TTensor.FromArray([0.9, -1.9], kDouble);
  Check((T.Kind = kDouble) and (T.DoubleValue([0]) = Double(0.9)), '[0.9, -1.9], kDouble: exact 0.9');
  T := TTensor.FromArray([1, 2], kFloat);
  Check((T.Kind = kFloat) and Near(T.DoubleValue([1]), 2), '[1, 2], kFloat');
  T := TTensor.FromSingleArray(TSingleArray.Create(1.5, -2.5));
  Check((T.Kind = kFloat) and Near(T.DoubleValue([1]), -2.5), 'FromSingleArray: kFloat');
  T := TTensor.FromSingleArray(TSingleArray.Create(1.5, -2.5), kDouble);
  Check((T.Kind = kDouble) and Near(T.DoubleValue([0]), 1.5), 'FromSingleArray, kDouble');
  T := TTensor.FromArray([Byte(3)], kInt64);
  Check((T.Kind = kInt64) and (T.Int64Value([0]) = 3), 'Byte, kInt64');
  T := TTensor.FromArray([Byte(200), Byte(1)]);
  Check((T.Kind = kUInt8) and (T.Int64Value([0]) = 200), 'Byte: kUInt8, 200');
  T := TTensor.FromArray([True, False, True]);
  Check((T.Kind = kBool) and (T.Int64Value([0]) = 1) and (T.Int64Value([1]) = 0),
    'Boolean: kBool, [1, 0, 1]');
  T := TTensor.FromInt32Array(I32);
  Check((T.Kind = kInt) and (T.Int64Value([1]) = -8), 'FromInt32Array: kInt, -8');
  Empty := nil;
  T := TTensor.FromArray(Empty);
  Check((ShapeStr(T) = '[0]') and (T.Numel = 0), 'empty array: shape [0]');
  T := TTensor.FromBytes(Raw, [2, 2], kDouble);
  Check((ShapeStr(T) = '[2,2]') and Near(T.DoubleValue([1, 0]), 3.5),
    'FromBytes [2,2] kDouble: [1,0] = 3.5');
  try
    TTensor.FromBytes(Raw, [2], kNone);
    Check(False, 'FromBytes kNone raises');
  except
    on E: ETorchError do
      Check(not (E is ETorchNativeError), 'FromBytes kNone: ETorchError before the call');
  end;
end;

procedure TestDataOut;
var
  X: TTensor;
  S: TSingleArray;
  I: TInt64Array;
  Buf: array[0..5] of Single;
begin
  WriteLn('data out: To*Array, CopyData');
  X := MakeX;
  S := X.ToSingleArray;
  Check((Length(S) = 6) and (S[5] = 5), 'ToSingleArray: 6 values, last 5');
  Check(Values(X.t) = '0 3 1 4 2 5', 'transposed (non-contiguous): ' + Values(X.t));
  I := (X * 0.75).ToInt64Array;              // [0, .75, 1.5, 2.25, 3, 3.75] truncated
  Check((I[1] = 0) and (I[2] = 1) and (I[5] = 3), 'ToInt64Array converts (truncates)');
  Check(Values(TTensor.FromArray([1, 2])) = '1 2', 'ToDoubleArray from Int64');
  FillChar(Buf, SizeOf(Buf), 0);
  X.CopyData(Buf, 4);
  Check((Buf[3] = 3) and (Buf[4] = 0), 'CopyData 4 of 6: [3] = 3, [4] untouched');
  try
    X.CopyData(Buf, 7);
    Check(False, 'CopyData beyond Numel raises');
  except
    on E: ETorchNativeError do
      Check(True, 'CopyData beyond Numel: ' + E.Message);
  end;
end;

procedure TestBasics;
var
  X, A, C: TTensor;
begin
  WriteLn('Numel, Device, ShallowClone, Copy');
  X := MakeX;
  Check(X.Numel = 6, 'Numel = 6');
  Check(X.Device.DeviceType = dtCpu, 'Device = CPU');
  A := X.ShallowClone;
  Check(not A.SameHandle(X), 'ShallowClone: new handle');
  A.add_(TTensor.ones([2, 3]));
  Check(Near(X.DoubleValue([0, 0]), 1), 'ShallowClone: shares storage');
  C := X.Copy;
  C.add_(TTensor.ones([2, 3]));
  Check(Near(X.DoubleValue([0, 0]), 1) and Near(C.DoubleValue([0, 0]), 2),
    'Copy: own storage');
end;

procedure TestOperators;
var
  X, N: TTensor;
begin
  WriteLn('operators');
  X := TTensor.FromArray([1.0, 2, 4]);
  Check(Values(X + X) = '2 4 8', 'A + B');
  Check(Values(X - X * X) = '0 -2 -12', 'A - B, A * B');
  Check(Values(X / (X + X)) = '0.5 0.5 0.5', 'A / B');
  Check(Values(-X) = '-1 -2 -4', '-A');
  Check(Values(X ** 2) = '1 4 16', 'A ** 2');
  Check(Values(X + 1) = '2 3 5', 'A + 1');
  Check(Values(X + 0.5) = '1.5 2.5 4.5', 'A + 0.5');
  Check(Values(1 + X) = '2 3 5', '1 + A');
  Check(Values(0.5 + X) = '1.5 2.5 4.5', '0.5 + A');
  Check(Values(X - 1) = '0 1 3', 'A - 1');
  Check(Values(X - 0.5) = '0.5 1.5 3.5', 'A - 0.5');
  Check(Values(10 - X) = '9 8 6', '10 - A');
  Check(Values(0.5 - X) = '-0.5 -1.5 -3.5', '0.5 - A');
  Check(Values(X * 3) = '3 6 12', 'A * 3');
  Check(Values(X * 0.5) = '0.5 1 2', 'A * 0.5');
  Check(Values(3 * X) = '3 6 12', '3 * A');
  Check(Values(2.5 * X) = '2.5 5 10', '2.5 * A');
  Check(Values(X / 2) = '0.5 1 2', 'A / 2');
  Check(Values(X / 0.5) = '2 4 8', 'A / 0.5');
  Check(Values(4 / X) = '4 2 1', '4 / A');
  Check(Values(1.0 / X) = '1 0.5 0.25', '1.0 / A');
  N := TTensor.FromArray([1, 2, 4]);         // Int64 tensor
  Check((N + 1).Kind = kInt64, 'Int64 tensor + 1 stays kInt64');
  Check(((N / 2).Kind = kFloat) and (Values(N / 2) = '0.5 1 2'),
    'Int64 tensor / 2: true division, kFloat');
  Check(((4 / N).Kind = kFloat) and (Values(4 / N) = '4 2 1'),
    '4 / Int64 tensor: kFloat');
  Check((N * 2.5).Kind = kFloat, 'Int64 tensor * 2.5: kFloat');
end;

procedure TestFactories;
var
  A, B: TTensor;
  D: TDoubleArray;
  K: Integer;
  InRange: Boolean;
begin
  WriteLn('factory overloads, ManualSeed');
  A := TTensor.zeros([2, 3]);
  Check((A.Kind = kFloat) and (ShapeStr(A) = '[2,3]') and (A.Device.DeviceType = dtCpu),
    'zeros([2, 3]): kFloat, CPU');
  A := TTensor.ones([2], kInt64);
  Check((A.Kind = kInt64) and (A.Int64Value([1]) = 1), 'ones([2], kInt64)');
  A := TTensor.zeros([1], kDouble, TDevice.Cpu);
  Check(A.Kind = kDouble, 'generated zeros(size, kind, device) still callable');
  A := TTensor.empty([4]);
  Check(ShapeStr(A) = '[4]', 'empty([4])');
  D := TTensor.rand([1000]).ToDoubleArray;
  InRange := True;
  for K := 0 to High(D) do
    InRange := InRange and (D[K] >= 0) and (D[K] < 1);
  Check(InRange, 'rand: 1000 values in [0, 1)');
  ManualSeed(42);
  A := TTensor.randn([5]);
  ManualSeed(42);
  B := TTensor.randn([5]);
  Check(Values(A) = Values(B), 'ManualSeed(42) twice: same randn');
  Check(Values(A) <> Values(TTensor.randn([5])), 'next randn differs');
end;

procedure NoGradBody(const X: TTensor);
var
  G: IInterface;
begin
  G := NoGradGuard;
  Check(not (X * 2).RequiresGrad, 'inside NoGradGuard: result does not require grad');
end;

procedure BareGuardBody(const X: TTensor);
begin
  NoGradGuard;   // pitfall: released at once
  Check((X * 2).RequiresGrad, 'bare NoGradGuard; has no effect (H11 pitfall)');
end;

procedure TestAutograd;
var
  X, Y: TTensor;
begin
  WriteLn('autograd and grad mode');
  X := TTensor.FromArray([1.0, 2, 3]).requires_grad_;
  Check(X.RequiresGrad, 'requires_grad_ sets RequiresGrad');
  Check(not X.grad.Defined, 'no grad before Backward');
  Y := (X * X).sum;
  Y.Backward;
  Check(Values(X.grad) = '2 4 6', 'd(sum x^2)/dx = 2x: ' + Values(X.grad));
  Y := (X * X).sum;
  Y.Backward;
  Check(Values(X.grad) = '4 8 12', 'second Backward accumulates');
  X.ZeroGrad;
  Check(Values(X.grad) = '0 0 0', 'ZeroGrad');
  Y := (X ** 3).sum;
  Y.Backward(True);
  Y.Backward;                                // KeepGraph allowed the second pass
  Check(Values(X.grad) = '6 24 54', 'Backward(KeepGraph) twice: 2 * 3x^2');
  try
    Y.Backward;
    Check(False, 'third Backward on a freed graph raises');
  except
    on E: ETorchNativeError do
      Check(True, 'Backward on a freed graph: ETorchNativeError');
  end;
  NoGradBody(X);
  Check((X * 2).RequiresGrad, 'after the routine: grad mode restored');
  BareGuardBody(X);
  Check(SetGradEnabled(False), 'SetGradEnabled returns previous (True)');
  Check(not SetGradEnabled(True), 'SetGradEnabled returns previous (False)');
  X.ZeroGrad;
  Check(not TTensor.zeros([2]).grad.Defined, 'grad of a leaf without grad: undefined');
  TTensor.zeros([2]).ZeroGrad;               // no grad: no-op, no error
end;

function RaisesNative(P: TProcedure): Boolean;
begin
  Result := False;
  try
    P;
  except
    on ETorchNativeError do Result := True;
  end;
end;

procedure CallSync;
begin
  CudaSynchronize;
end;

procedure CallRuntimeVersion;
begin
  GpuRuntimeVersion;
end;

procedure CallCudnnVersion;
begin
  CudnnVersion;
end;

procedure TestSettings;
var
  N0: Integer;
begin
  WriteLn('global settings');
  N0 := GetNumThreads;
  SetNumThreads(2);
  Check(GetNumThreads = 2, 'SetNumThreads(2)');
  SetNumThreads(N0);
  Check(GetNumThreads = N0, 'restored to ' + IntToStr(N0));
  Check(not CudaIsAvailable, 'CudaIsAvailable = False (CPU build)');
  Check(CudaDeviceCount = 0, 'CudaDeviceCount = 0');
  Check(TDevice.CudaIfAvailable.DeviceType = dtCpu, 'CudaIfAvailable = CPU');
  { DESIGN 4.8: on a CPU build these work, the next three raise }
  Check(GpuBackend = gbNone, 'GpuBackend = gbNone (CPU build)');
  Check(not CudnnIsAvailable and CudnnUserEnabled, 'CudnnIsAvailable False, CudnnUserEnabled True');
  CudaManualSeed(1);
  SetCudnnBenchmark(False);
  Check(True, 'CudaManualSeed, SetCudnnBenchmark raise nothing');
  Check(RaisesNative(@CallSync), 'CudaSynchronize raises ETorchNativeError');
  Check(RaisesNative(@CallRuntimeVersion), 'GpuRuntimeVersion raises ETorchNativeError');
  Check(RaisesNative(@CallCudnnVersion), 'CudnnVersion raises ETorchNativeError');
end;

procedure TestElementSizes;
var
  K: TKind;
  T: TTensor;
  Buf: array[0..15] of Byte;
  Bad: String;
begin
  WriteLn('KindElementSize vs libtorch element_size (all kinds)');
  Bad := '';
  for K := kUInt8 to kFloat4e2m1fnx2 do
  begin
    try
      T := TTensor.empty([1], K);
      T.CopyData(Buf, 0);                    // libtch checks the element size
    except
      on E: Exception do
        Bad := Bad + ' ' + GetEnumName(TypeInfo(TKind), Ord(K)) + '(' + E.Message + ')';
    end;
  end;
  Check(Bad = '', 'all 46 kinds agree' + Bad);
end;

begin
  TestDataIn;
  TestDataOut;
  TestBasics;
  TestOperators;
  TestFactories;
  TestAutograd;
  TestSettings;
  TestElementSizes;
  Check(TchLiveHandles = 0, 'no live handles at the end');
  if Failures = 0 then
    WriteLn('OK')
  else
  begin
    WriteLn(Failures, ' FAILED');
    Halt(1);
  end;
end.
