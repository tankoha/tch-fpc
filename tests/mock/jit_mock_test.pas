{ TchJit plumbing against the mock libtch (tests/mock/run.sh), no libtorch:
  TIValue ToC/FromC round trips and their error paths (no C ivalue left
  behind), module construction and argument passing, the tracing error
  path, TTrainableJitModule registration, TorchScript object plumbing
  (DESIGN.md 4.6.9). TorchScript itself and every number of jit_test are
  not coverable here (DESIGN.md 4.6.8, "Mock"). }
program jit_mock_test;

{$mode objfpc}{$H+}

uses
  SysUtils, Classes, ctypes, Tch, TchNN, TchJit;

{ mock-only introspection, exported by tests/mock/mock_tch.c }
function mock_live_tensors: cint; cdecl; external 'tch';
function mock_live_ivalues: cint; cdecl; external 'tch';
function mock_live_modules: cint; cdecl; external 'tch';
function mock_module_training: cint; cdecl; external 'tch';
function mock_last_to_device: cint; cdecl; external 'tch';
function mock_last_to_dtype: cint; cdecl; external 'tch';
function mock_tracing_begins: cint; cdecl; external 'tch';
function mock_tracing_ends: cint; cdecl; external 'tch';
function mock_last_end_noutputs: cint; cdecl; external 'tch';
function mock_creates: cint; cdecl; external 'tch';
function mock_registered: cint; cdecl; external 'tch';
function mock_defines: cint; cdecl; external 'tch';
function mock_last_load_str_size: cint; cdecl; external 'tch';
function mock_last_method: PChar; cdecl; external 'tch';
function mock_last_class: PChar; cdecl; external 'tch';
function mock_last_module_file: PChar; cdecl; external 'tch';
function mock_last_source: PChar; cdecl; external 'tch';

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

function RoundTrip(const V: TIValue): TIValue;
begin
  Result := TIValue.FromC(V.ToC);
end;

procedure TestIValues;
var
  R: TIValue;
  NilT: TTensor;
  Raised: Boolean;
begin
  WriteLn('ivalue: round trips and error paths');
  Check(RoundTrip(TIValue.None).IsNone, 'None');
  Check(RoundTrip(True).AsBool and not RoundTrip(False).AsBool, 'Bool');
  Check(RoundTrip(-1).AsInt = -1, 'Int');
  Check(RoundTrip(2.5).AsDouble = 2.5, 'Double');
  Check((RoundTrip('').AsString = '') and (RoundTrip('foobar').AsString = 'foobar'), 'String');
  R := RoundTrip(TIValue.Tuple([42, 1.5]));
  Check((R.Tag = ivTuple) and (R.Item(0).AsInt = 42) and (R.Item(1).AsDouble = 1.5), 'Tuple');
  R := RoundTrip(TIValue.IntList([42, 1337]));
  Check((R.Count = 2) and (R.AsIntList[1] = 1337), 'IntList');
  R := RoundTrip(TIValue.DoubleList([0.5, 2.5]));
  Check((R.Count = 2) and (R.AsDoubleList[1] = 2.5), 'DoubleList');
  R := RoundTrip(TIValue.BoolList([True, False, True]));
  Check((R.Count = 3) and R.AsBoolList[2] and not R.AsBoolList[1], 'BoolList');
  R := RoundTrip(TIValue.GenericList([42, 'foobar']));
  Check((R.Tag = ivGenericList) and (R.Item(1).AsString = 'foobar'), 'GenericList');
  R := RoundTrip(TIValue.GenericDict([42, 'foobar', 'foo', 'bar']));
  Check((R.Count = 2) and (R.Key(0).AsInt = 42) and (R.Value(1).AsString = 'bar'), 'GenericDict');
  R := RoundTrip(TIValue.StringList(['a', 'b']));
  Check((R.Tag = ivGenericList) and (R.Item(0).AsString = 'a'), 'StringList comes back as GenericList');
  R := RoundTrip(TTensor.zeros([2, 3], kFloat, TDevice.Cpu));
  Check((R.Tag = ivTensor) and (R.AsTensor.Numel = 6), 'Tensor');
  R := RoundTrip(TIValue.TensorList([TTensor.zeros([2], kFloat, TDevice.Cpu),
    TTensor.ones([3], kFloat, TDevice.Cpu)]));
  Check((R.Count = 2) and (R.AsTensorList[1].Numel = 3), 'TensorList');
  Check(RoundTrip(TIValue.Tuple([])).Count = 0, 'empty Tuple (nil pointer)');
  Check(RoundTrip(TIValue.IntList([])).Count = 0, 'empty IntList');
  R := Default(TIValue);
  Check(mock_live_ivalues = 0, 'no C ivalue left after the round trips');

  Raised := False;
  try
    TIValue.FromC(nil);
  except
    on ETorchError do Raised := True;
  end;
  Check(Raised, 'FromC(nil) raises ETorchError');
  Raised := False;
  try
    RoundTrip(TIValue.FromDevice(TDevice.Cpu));
  except
    on ETorchNativeError do Raised := True;
  end;
  Check(Raised and (mock_live_ivalues = 0), 'Device output raises, freed');
  { the second child fails in FromC: the third is freed by the finally }
  Raised := False;
  try
    RoundTrip(TIValue.Tuple([1, TIValue.FromDevice(TDevice.Cpu), 2]));
  except
    on ETorchNativeError do Raised := True;
  end;
  Check(Raised and (mock_live_ivalues = 0), 'failing child in FromC: the rest freed');
  { the second child fails in ToC: the first is freed }
  NilT := Default(TTensor);
  Raised := False;
  try
    TIValue.Tuple([1, NilT]).ToC;
  except
    on ETorchError do Raised := True;
  end;
  Check(Raised and (mock_live_ivalues = 0), 'nil tensor child in ToC: the rest freed');
  Raised := False;
  try
    TIValue.GenericDict(['odd']);
  except
    on ETorchError do Raised := True;
  end;
  Check(Raised, 'odd GenericDict raises');
  Raised := False;
  try
    TIValue.FromInt(1).AsString;
  except
    on E: ETorchError do Raised := E.Message = 'ivalue: expected String, got Int';
  end;
  Check(Raised, 'AsString on an Int: "ivalue: expected String, got Int"');
end;

procedure TestModule;
var
  M: TJitModule;
  X: TTensor;
  V: TIValue;
  NP: TNamedTensorArray;
  S: TStringStream;
  Raised: Boolean;
begin
  WriteLn('module: load, forward, method, eval/train, to, save, parameters');
  X := TTensor.ones([2], kFloat, TDevice.Cpu);
  M := TJitModule.Create('model.pt');
  try
    Check(M.Forward([X]).Numel = 2, 'Forward returns the mock copy');
    Check(StrPas(mock_last_method) = 'forward', 'forward');
    M.Method('encode', [X]);
    Check(StrPas(mock_last_method) = 'encode', 'Method name passed');
    V := M.ForwardIV([X, 42, 1.5]);
    Check((V.Count = 3) and (V.Item(1).AsInt = 42) and (V.Item(2).AsDouble = 1.5),
      'ForwardIV([X, 42, 1.5]) converts element-wise');
    V := M.MethodIV('gen', ['a']);
    Check((StrPas(mock_last_method) = 'gen') and (V.Item(0).AsString = 'a'), 'MethodIV');
    M.SetEval;
    Check(mock_module_training = 0, 'SetEval');
    M.SetTrain;
    Check(mock_module_training = 1, 'SetTrain');
    M.ToDevice(TDevice.Cpu, kDouble);
    Check((mock_last_to_device = -1) and (mock_last_to_dtype = Ord(kDouble)), 'ToDevice(Cpu, kDouble)');
    M.Save('out.pt');
    Check(StrPas(mock_last_module_file) = 'out.pt', 'Save file name');
    NP := M.NamedParameters;
    Check((Length(NP) = 2) and (NP[0].Name = 'weight') and (NP[1].Name = 'sub.bias'),
      'NamedParameters through the trampoline');
    NP := nil;
    Raised := False;
    try
      M.Forward([]);
    except
      on ETorchNativeError do Raised := True;
    end;
    Check(Raised, 'mock error reaches Pascal');
  finally
    M.Free;
  end;
  Check(mock_live_ivalues = 0, 'inputs and results freed');
  Raised := False;
  try
    TJitModule.Create('missing.pt').Free;
  except
    on ETorchNativeError do Raised := True;
  end;
  Check(Raised, 'missing file raises');
  S := TStringStream.Create('0123456789');
  try
    S.Position := 4;
    M := TJitModule.CreateFromStream(S);
    M.Free;
    Check(mock_last_load_str_size = 6, 'CreateFromStream passes the rest (6 bytes)');
    Raised := False;
    try
      TJitModule.CreateFromStream(S).Free;
    except
      on ETorchError do Raised := True;
    end;
    Check(Raised, 'empty stream raises before the call');
  finally
    S.Free;
  end;
  M := TJitModule.CreateEmpty('Lin');
  try
    M.RegisterParameter('weight', X);
    M.Define('def forward(self, x):'#10'    return x'#10);
    Check((mock_creates = 1) and (mock_registered = 1) and (mock_defines = 1) and
      (StrPas(mock_last_source) = 'def forward(self, x):'#10'    return x'#10),
      'CreateEmpty, RegisterParameter, Define');
  finally
    M.Free;
  end;
  Check(mock_live_modules = 0, 'every module freed');
end;

function AddFn(const Inputs: TTensorArray): TTensorArray;
begin
  Result := [Inputs[0] + Inputs[1]];
end;

function FailingFn(const Inputs: TTensorArray): TTensorArray;
begin
  Result := nil;
  raise ETorchError.Create('closure failed');
end;

procedure TestTracing;
var
  M: TJitModule;
  X: TTensor;
  Raised: Boolean;
begin
  WriteLn('tracing: outputs passed, raising closure ends tracing once');
  X := TTensor.ones([2], kFloat, TDevice.Cpu);
  M := TJitModule.CreateByTracing('M', 'Fn', [X, X], @AddFn);
  M.Free;
  Check((mock_tracing_begins = 1) and (mock_tracing_ends = 1) and
    (mock_last_end_noutputs = 1), 'begin, end with one output');
  Raised := False;
  try
    TJitModule.CreateByTracing('M', 'Fn', [X], @FailingFn).Free;
  except
    on E: ETorchError do Raised := E.Message = 'closure failed';
  end;
  Check(Raised and (mock_tracing_ends = 2) and (mock_last_end_noutputs = 0),
    'raising closure: re-raised, atm_end_tracing once with 0 outputs');
  Check(mock_live_modules = 0, 'traced modules freed');
end;

{ Counts are compared in TestObjects: temporaries keep handles alive
  until the routine returns (H1). }
procedure ObjectsInner(M: TJitModule);
var
  O: IJitObject;
  V, A, B: TIValue;
  Raised: Boolean;
  Cells, Before, BeforeObj: Integer;
begin
  O := M.CreateClass('__torch__.Foo', [1, 2.5]);
  { the mock object is one cell plus one per attribute }
  Cells := mock_live_ivalues;
  Check((StrPas(mock_last_class) = '__torch__.Foo') and (Cells = 3) and
    (JitLiveObjects = 1), '1. CreateClass: class name passed, object + 2 attributes, one handle');
  Check(O.GetAttr('a1').AsDouble = 2.5, '2. GetAttr a1 = 2.5');
  Raised := False;
  try
    O.GetAttr('zz');
  except
    on ETorchNativeError do Raised := True;
  end;
  Check(Raised and (mock_live_ivalues = Cells), '2. GetAttr zz raises, ivalues back to the object''s');
  V := O.MethodIV('m', ['x']);
  Check((StrPas(mock_last_method) = 'm') and (V.Count = 3) and (V.Item(0).AsInt = 1) and
    (V.Item(1).AsDouble = 2.5) and (V.Item(2).AsString = 'x'), '3. MethodIV m(x) = (1, 2.5, x)');
  Raised := False;
  try
    O.MethodIV('fail', [O]);
  except
    on ETorchNativeError do Raised := True;
  end;
  Check(Raised and (mock_live_ivalues = Cells), '3. MethodIV fail raises, the cloned input is freed');
  A := TIValue.FromObject(O);
  B := A;
  A := Default(TIValue);
  Check((B.AsObject.Raw = O.Raw) and (mock_live_ivalues = Cells) and (JitLiveObjects = 1),
    '4. copies share the handle');
  V := RoundTrip(TIValue.Tuple([O, 42]));
  Check((V.Item(0).Tag = ivObject) and (V.Item(0).AsObject.GetAttr('a0').AsInt = 1) and
    (V.Item(1).AsInt = 42), '5. Tuple([O, 42]) round trip');
  Check((mock_live_ivalues = 2 * Cells) and (JitLiveObjects = 2), '5. two objects alive, each its own cells');
  V := M.MethodIV('f', [O]);
  Check((StrPas(mock_last_method) = 'f') and (V.Item(0).Tag = ivObject), '6. module MethodIV(f, [O])');
  V := Default(TIValue);
  { a delta: the temporaries of items 5 and 6 still hold handles (H1) }
  Before := mock_live_ivalues;
  BeforeObj := JitLiveObjects;
  Raised := False;
  try
    RoundTrip(TIValue.Tuple([O, TIValue.FromDevice(TDevice.Cpu)]));
  except
    on ETorchError do Raised := True;
  end;
  Check(Raised and (mock_live_ivalues = Before) and (JitLiveObjects = BeforeObj),
    '7. failing Device sibling: the converted object is released');
end;

procedure TestObjects;
var
  M: TJitModule;
  Before: Integer;
begin
  WriteLn('objects: create, getattr, method, sharing, containers');
  Before := TchLiveHandles;
  M := TJitModule.Create('model.pt');
  try
    ObjectsInner(M);
  finally
    M.Free;
  end;
  Check((JitLiveObjects = 0) and (mock_live_ivalues = 0) and (TchLiveHandles = Before),
    '4.-7. everything released after the caller returns');
end;

procedure TestTrainable;
var
  VS: TVarStore;
  J: TTrainableJitModule;
  NT: TNamedTensorArray;
  X: TTensor;
begin
  WriteLn('trainable: registration, Train flag');
  VS := TVarStore.Create(TDevice.Cpu);
  J := nil;
  try
    J := TTrainableJitModule.Create('model.pt', VS.Root / 'jit');
    NT := VS.Variables;
    Check((Length(NT) = 2) and (NT[0].Name = 'jit.weight') and (NT[1].Name = 'jit.sub_bias'),
      'jit.weight, jit.sub_bias');
    Check(Length(VS.TrainableVariables) = 1, 'only weight trainable');
    Check(mock_last_to_device = -1, 'loaded on the store device');
    X := TTensor.ones([2], kFloat, TDevice.Cpu);
    J.Forward(X, True);
    Check(mock_module_training = 1, 'Forward(X, True) trains');
    J.Forward(X);
    Check(mock_module_training = 0, 'Forward(X) evals');
  finally
    J.Free;
    VS.Free;
  end;
  Check(mock_live_modules = 0, 'module freed');
end;

begin
  TestIValues;
  TestModule;
  TestTracing;
  TestTrainable;
  TestObjects;
  Check(JitGetProfilingMode, 'profiling getter');
  Check(TchLiveHandles = 0, 'no live handles');
  Check(mock_live_tensors = 0, 'mock: no tensors left');
  Check(mock_live_ivalues = 0, 'mock: no ivalues left');
  Check(JitLiveObjects = 0, 'no live objects');
  if Failures = 0 then
    WriteLn('OK')
  else
  begin
    WriteLn(Failures, ' FAILED');
    Halt(1);
  end;
end.
