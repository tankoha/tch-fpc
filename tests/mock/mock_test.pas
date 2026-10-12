{ Exercises the generated op shapes against the mock libtch (tests/mock/run.sh):
  fixed-1, fixed-N with out parameters, fixed-N filling tensors passed in,
  dynamic, scalar results, Scalar temporaries, tensor lists, IntListOption,
  String + DoubleOption, defaults, nil checks and the native error split. }
program mock_test;

{$mode objfpc}{$H+}

uses
  SysUtils, ctypes, Tch;

{ mock-only introspection, exported by tests/mock/mock_tch.c }
function mock_live_tensors: cint; cdecl; external 'tch';
function mock_live_scalars: cint; cdecl; external 'tch';
function mock_last_dim_null: cint; cdecl; external 'tch';
function mock_last_dim_len: cint; cdecl; external 'tch';
function mock_last_mode: PChar; cdecl; external 'tch';
function mock_last_value_null: cint; cdecl; external 'tch';
function mock_last_value: Double; cdecl; external 'tch';
function mock_last_scalar_is_float: cint; cdecl; external 'tch';
function mock_last_scalar_i: Int64; cdecl; external 'tch';
function mock_last_scalar_f: Double; cdecl; external 'tch';

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

function Shape(const T: TTensor): String;
var
  S: TInt64Array;
  I: Integer;
begin
  S := T.Shape;
  Result := '[';
  for I := 0 to High(S) do
  begin
    if I > 0 then Result := Result + ',';
    Result := Result + IntToStr(S[I]);
  end;
  Result := Result + ']';
end;

procedure TestFixed1;
var
  A, B, C: TTensor;
begin
  WriteLn('fixed1: ones, add, operators');
  A := TTensor.ones([2, 3], kFloat, TDevice.Cpu);
  Check(A.Dim = 2, 'ones([2,3]).Dim = 2');
  Check(Shape(A) = '[2,3]', 'Shape = [2,3]');
  Check(A.Kind = kFloat, 'Kind = kFloat');
  B := A.add(A);
  Check(B.DoubleValue([1, 2]) = 2.0, 'add: 1 + 1 = 2');
  C := (A + B) * B;
  Check(C.DoubleValue([0, 0]) = 6.0, '(A + B) * B = 6');
  C := A.&to(TDevice.Cpu);
  Check(Shape(C) = '[2,3]', '&to(TDevice.Cpu) is callable');
  C := A.totype(kDouble);
  Check(C.Kind = kDouble, 'totype(kDouble)');
end;

procedure TestScalarArgs;
var
  A, B: TTensor;
  Base: Integer;
begin
  WriteLn('Scalar arguments: temporaries created and freed');
  A := TTensor.ones([2], kFloat, TDevice.Cpu);
  Base := mock_live_scalars;
  B := A.add_scalar(1);
  Check((mock_last_scalar_is_float = 0) and (mock_last_scalar_i = 1), 'add_scalar(1) passes an int scalar');
  Check(B.DoubleValue([0]) = 2.0, 'result 1 + 1 = 2');
  B := A.add_scalar(1.5);
  Check((mock_last_scalar_is_float = 1) and (mock_last_scalar_f = 1.5), 'add_scalar(1.5) passes a float scalar');
  Check(mock_live_scalars = Base, 'ats_* handles freed after the calls');
end;

procedure TestOutParams;
var
  A, V, I: TTensor;
begin
  WriteLn('fixedN with out parameters: max_dim');
  A := TTensor.ones([2, 3], kFloat, TDevice.Cpu);
  A := A.add_scalar(0);     // fresh copy we can look at
  A.max_dim(1, False, V, I);
  Check(Shape(V) = '[2]', 'values shape [2]');
  Check(Shape(I) = '[2]', 'indices shape [2]');
  Check(I.Kind = kInt64, 'indices are kInt64');
  A.max_dim(1, True, V, I);
  Check(Shape(V) = '[2,1]', 'keepdim: values shape [2,1]');
end;

procedure TestOutLess;
var
  A, V, I: TTensor;
  Base: Integer;
begin
  WriteLn('fixedN filling tensors passed in: sort_values (no out parameters, defaults allowed)');
  A := TTensor.zeros([3], kFloat, TDevice.Cpu);
  A := A.add_scalar(5);                 // [5,5,5]
  V := TTensor.zeros([1], kFloat, TDevice.Cpu);
  I := TTensor.zeros([1], kInt64, TDevice.Cpu);
  Base := TchLiveHandles;
  A.sort_values(V, I);                  // dim = -1, descending = False by default
  Check(TchLiveHandles = Base, 'no new handles kept');
  Check(Shape(V) = '[3]', 'values resized in place to [3]');
  Check(I.DoubleValue([2]) = 2, 'indices filled in place');
  Check(mock_live_tensors = TchLiveHandles, 'mock tensors = live handles (result handles freed)');
end;

procedure TestDynamic;
var
  A: TTensor;
  Parts: TTensorArray;
  Base: Integer;
begin
  WriteLn('dynamic: split');
  A := TTensor.ones([5, 2], kFloat, TDevice.Cpu);
  Base := TchLiveHandles;
  Parts := A.split(2);                  // dim = 0 by default
  Check(Length(Parts) = 3, 'split(2) of [5,2] gives 3 parts');
  Check(TchLiveHandles = Base + 3, 'three new handles');
  Check(Shape(Parts[2]) = '[1,2]', 'last part is [1,2]');
  Parts := nil;
  Check(TchLiveHandles = Base, 'released when the array is cleared');
end;

procedure TestScalarResults;
var
  A: TTensor;
begin
  WriteLn('scalar results: is_nonzero');
  A := TTensor.ones([1], kFloat, TDevice.Cpu);
  Check(A.is_nonzero, 'ones([1]).is_nonzero = True');
  Check(not TTensor.zeros([1], kFloat, TDevice.Cpu).is_nonzero, 'zeros([1]).is_nonzero = False');
  try
    TTensor.ones([2], kFloat, TDevice.Cpu).is_nonzero;
    Check(False, 'is_nonzero on 2 elements raises');
  except
    on E: ETorchNativeError do
      Check(Pos('ambiguous', E.Message) > 0, 'is_nonzero on 2 elements raises: ' + E.Message);
  end;
end;

procedure TestTensorList;
var
  A, B, C: TTensor;
begin
  WriteLn('TensorList: cat');
  A := TTensor.ones([2, 2], kFloat, TDevice.Cpu);
  B := TTensor.zeros([1, 2], kFloat, TDevice.Cpu);
  C := TTensor.cat([A, B]);             // dim = 0 by default
  Check(Shape(C) = '[3,2]', 'cat([A, B]) is [3,2]');
  C := TTensor.cat([A, B, A + A], 0);
  Check(Shape(C) = '[5,2]', 'cat with a temporary element');
  try
    C := TTensor.cat([A, TTensor(Default(TTensor))], 0);
    Check(False, 'nil element rejected');
  except
    on E: ETorchError do
      Check(Pos('tensors[1]', E.Message) > 0, 'nil element rejected: ' + E.Message);
  end;
  try
    C := TTensor.cat([], 0);
    Check(False, 'empty list reaches C and is rejected there');
  except
    on E: ETorchNativeError do
      Check(True, 'empty list reaches C and is rejected there: ' + E.Message);
  end;
end;

procedure TestIntListOption;
var
  A, S: TTensor;
begin
  WriteLn('IntListOption and ScalarTypeOption: sum_dim_intlist');
  A := TTensor.ones([2, 3], kFloat, TDevice.Cpu);
  S := A.sum_dim_intlist(NoIntList);
  Check((mock_last_dim_null = 1) and (S.Dim = 0) and (S.DoubleValue([]) = 6.0), 'NoIntList -> nullptr, sums everything');
  S := A.sum_dim_intlist(IntList([]));
  Check((mock_last_dim_null = 0) and (mock_last_dim_len = 0), 'IntList([]) -> non-nil pointer, length 0');
  S := A.sum_dim_intlist(IntList([0]), False, kDouble);
  Check((Shape(S) = '[3]') and (S.Kind = kDouble), 'IntList([0]) + dtype');
  S := A.sum_dim_intlist(IntList([0]));
  Check(S.Kind = kFloat, 'dtype defaults to kNone (-1): kind kept');
end;

procedure TestStringAndDoubleOption;
var
  A, B: TTensor;
begin
  WriteLn('String and DoubleOption: pad');
  A := TTensor.ones([2], kFloat, TDevice.Cpu);
  B := A.pad([1, 1], 'reflect', NoDouble);
  Check((StrPas(mock_last_mode) = 'reflect') and (mock_last_value_null = 1), 'mode string by pointer+length, value None');
  B := A.pad([1, 1], 'constant', 0.5);
  Check((StrPas(mock_last_mode) = 'constant') and (mock_last_value_null = 0) and (mock_last_value = 0.5), 'value 0.5 via implicit TOptDouble');
end;

procedure TestErrors;
var
  A, B, C, N: TTensor;
begin
  WriteLn('errors');
  A := TTensor.ones([2, 3], kFloat, TDevice.Cpu);
  B := TTensor.ones([3, 2], kFloat, TDevice.Cpu);
  try
    C := A + B;
    Check(False, 'shape mismatch raises');
  except
    on E: ETorchNativeError do
    begin
      Check(Pos('Exception raised from', E.Message) = 0, 'Message has no backtrace: ' + E.Message);
      Check(Pos('Exception raised from', E.Detail) > 0, 'Detail keeps the backtrace');
    end;
  end;
  try
    C := N.add(A);
    Check(False, 'nil receiver raises');
  except
    on E: ETorchError do
      Check((E.ClassType = ETorchError) and (Pos('self', E.Message) > 0), 'nil receiver: ' + E.Message);
  end;
  try
    C := A.add(N);
    Check(False, 'nil argument raises');
  except
    on E: ETorchError do
      Check(Pos('other to add', E.Message) > 0, 'nil argument: ' + E.Message);
  end;
  try
    C := A.abs;
    Check(False, 'unimplemented mock op raises');
  except
    on E: ETorchNativeError do
      Check(Pos('not implemented: atg_abs', E.Message) > 0, 'unimplemented mock op raises: ' + E.Message);
  end;
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

procedure TestGpuBackend;
begin
  WriteLn('GPU backend: CPU-only mock');
  Check(GpuBackend = gbNone, 'GpuBackend = gbNone');
  Check(not CudaIsAvailable, 'CudaIsAvailable = False');
  Check(not CudnnIsAvailable, 'CudnnIsAvailable = False');
  Check(CudnnUserEnabled, 'CudnnUserEnabled = True');
  { these raise on a CPU build of libtorch 2.13 (measured, DESIGN 4.8) }
  Check(RaisesNative(@CallSync), 'CudaSynchronize raises ETorchNativeError');
  Check(RaisesNative(@CallRuntimeVersion), 'GpuRuntimeVersion raises ETorchNativeError');
  Check(RaisesNative(@CallCudnnVersion), 'CudnnVersion raises ETorchNativeError');
end;

var
  Base: Integer;
begin
  Base := TchLiveHandles;
  TestFixed1;
  TestScalarArgs;
  TestOutParams;
  TestOutLess;
  TestDynamic;
  TestScalarResults;
  TestTensorList;
  TestIntListOption;
  TestStringAndDoubleOption;
  TestErrors;
  TestGpuBackend;
  WriteLn('leaks');
  Check(TchLiveHandles = Base, 'all handles released');
  Check(mock_live_tensors = 0, 'mock: no C tensors left');
  Check(mock_live_scalars = 0, 'mock: no C scalars left');
  WriteLn;
  if Failures = 0 then
    WriteLn('OK')
  else
  begin
    WriteLn('FAILED (', Failures, ')');
    Halt(1);
  end;
end.
