{ Milestone 3 JIT (DESIGN.md 4.6.8) against the real libtorch: the
  vendored tch-rs models (tests/jit/models, tch-rs 4227b89b
  tests/jit_tests.rs), ivalue round trips, tracing, scripted modules from
  TorchScript source, TTrainableJitModule, globals, and hypotheses
  H19-H25. Expected values are worked out by hand in DESIGN.md 4.6.8.

  Build and run (from the repo root, after building build/libtch):
    fpc -Mobjfpc -Sh -Fusrc -FUbuild/jit -Flbuild/libtch \
      -k-rpath=$PWD/build/libtch tests/jit/jit_test.pas -obuild/jit_test
    build/jit_test }
program jit_test;

uses
  SysUtils, Classes, Tch, TchNN, TchJit;

const
  Models = 'tests/jit/models/';

var
  Failures: Integer = 0;
  InitialProfiling: Boolean;

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

function Near(A, B, Tol: Double): Boolean;
begin
  Result := Abs(A - B) <= Tol;
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
    Result := Result + FormatFloat('0.####', D[K]);
  end;
end;

function TempFile(const Tag, Ext: String): String;
begin
  Result := GetTempDir + Format('tch-fpc-jit-%s-%d%s', [Tag, GetProcessID, Ext]);
end;

function I64(V: Int64): TTensor;
begin
  Result := TTensor.scalar_tensor(V, kInt64, TDevice.Cpu);
end;

{ Structural equality; tensors by kind, shape and values. }
function SameIValue(const A, B: TIValue): Boolean;
var
  K: Integer;
begin
  Result := A.Tag = B.Tag;
  if not Result then
    Exit;
  case A.Tag of
    ivNone: ;
    ivTensor:
      Result := (A.Tensor.Kind = B.Tensor.Kind) and
        (Length(A.Tensor.Shape) = Length(B.Tensor.Shape)) and
        (Values(A.Tensor) = Values(B.Tensor));
    ivDouble: Result := A.D = B.D;
    ivInt: Result := A.I = B.I;
    ivBool: Result := A.B = B.B;
    ivString: Result := A.S = B.S;
    ivIntList:
      begin
        Result := Length(A.Ints) = Length(B.Ints);
        for K := 0 to High(A.Ints) do
          Result := Result and (A.Ints[K] = B.Ints[K]);
      end;
    ivDoubleList:
      begin
        Result := Length(A.Doubles) = Length(B.Doubles);
        for K := 0 to High(A.Doubles) do
          Result := Result and (A.Doubles[K] = B.Doubles[K]);
      end;
    ivBoolList:
      begin
        Result := Length(A.Bools) = Length(B.Bools);
        for K := 0 to High(A.Bools) do
          Result := Result and (A.Bools[K] = B.Bools[K]);
      end;
    ivTuple, ivGenericList, ivGenericDict:
      begin
        Result := Length(A.Items) = Length(B.Items);
        for K := 0 to High(A.Items) do
          Result := Result and SameIValue(A.Items[K], B.Items[K]);
      end;
  else
    Result := False;
  end;
end;

function RoundTrip(const V: TIValue): TIValue;
begin
  Result := TIValue.FromC(V.ToC);
end;

{ 1. foo.pt (tch-rs jit, jit_data) }
procedure TestFoo;
var
  M: TJitModule;
  X, Y: TTensor;
  S: TFileStream;
begin
  WriteLn('foo.pt: 2x + y + value (H19)');
  X := TTensor.FromArray([3.0, 1, 4, 1, 5]);
  Y := TTensor.FromArray([7.0]);
  M := TJitModule.Create(Models + 'foo.pt');
  try
    Check(Values(M.Forward([X, Y])) = '55 51 57 51 59', 'Forward = [55, 51, 57, 51, 59]');
    Check(Values(M.Method('forward', [X, Y])) = '55 51 57 51 59', 'Method(forward) the same');
    Check(Length(M.NamedParameters) = 0, 'NamedParameters empty (value is a buffer)');
  finally
    M.Free;
  end;
  M := TJitModule.Create(Models + 'foo.pt', TDevice.Cpu);
  try
    Check(Values(M.Forward([X, Y])) = '55 51 57 51 59', 'Create(file, Cpu) the same');
  finally
    M.Free;
  end;
  S := TFileStream.Create(Models + 'foo.pt', fmOpenRead);
  try
    M := TJitModule.CreateFromStream(S);
    try
      Check(Values(M.Forward([X, Y])) = '55 51 57 51 59', 'CreateFromStream the same');
    finally
      M.Free;
    end;
  finally
    S.Free;
  end;
end;

{ 2. foo1.pt (jit1) }
procedure TestFoo1;
var
  M: TJitModule;
  R: TTensor;
begin
  WriteLn('foo1.pt: 2x + y on 0-d Int64');
  M := TJitModule.Create(Models + 'foo1.pt');
  try
    R := M.Forward([I64(42), I64(1337)]);
    Check((R.Kind = kInt64) and (R.Int64Value([]) = 1421), 'Forward = 1421, kInt64');
    R := M.Method('forward', [I64(42), I64(1337)]);
    Check(R.Int64Value([]) = 1421, 'Method(forward) = 1421');
  finally
    M.Free;
  end;
end;

{ 3. foo2.pt (jit2) }
procedure TestFoo2;
var
  M: TJitModule;
  V: TIValue;
  Raised: Boolean;
begin
  WriteLn('foo2.pt: tuple result');
  M := TJitModule.Create(Models + 'foo2.pt');
  try
    V := M.ForwardIV([I64(42), I64(1337)]);
    Check((V.Tag = ivTuple) and (V.Count = 2), 'ForwardIV: tuple of 2');
    Check(V.Item(0).AsTensor.Int64Value([]) = 1421, 'item 0 = 1421');
    Check(V.Item(1).AsTensor.Int64Value([]) = -1295, 'item 1 = -1295');
    V := M.MethodIV('forward', [I64(42), I64(1337)]);
    Check((V.Count = 2) and (V.Item(1).AsTensor.Int64Value([]) = -1295),
      'MethodIV(forward) the same');
    Raised := False;
    try
      V.AsTensor;
    except
      on E: ETorchError do
        Raised := (Pos('Tensor', E.Message) > 0) and (Pos('Tuple', E.Message) > 0);
    end;
    Check(Raised, 'AsTensor on a tuple raises naming both tags');
    Raised := False;
    try
      M.Forward([I64(42), I64(1337)]);
    except
      on E: ETorchNativeError do
        Raised := Pos('forward did not return a tensor', E.Message) > 0;
    end;
    Check(Raised, 'Forward on a tuple result: "forward did not return a tensor"');
  finally
    M.Free;
  end;
end;

{ 4.-7. foo3, foo4, foo5, foo8 (jit3, jit4, jit5, specialized_dict) }
procedure TestFoo3To8;
var
  M: TJitModule;
  V: TIValue;
begin
  WriteLn('foo3/4/5/8: product, Tuple[float, float, int], List[str], Dict[str, Tensor] (H22)');
  M := TJitModule.Create(Models + 'foo3.pt');
  try
    Check(M.Forward([TTensor.FromArray([1.0, 2, 3, 4, 5])]).DoubleValue([]) = 120,
      'foo3: product of [1..5] = 120');
  finally
    M.Free;
  end;
  M := TJitModule.Create(Models + 'foo4.pt');
  try
    V := M.ForwardIV([TIValue.Tuple([2.0, 3.0, 4])]);
    Check((V.Tag = ivDouble) and (V.AsDouble = 14), 'foo4: 2 + 3 * 4 = 14.0');
    Check(Length(M.NamedParameters) = 0, 'foo4: NamedParameters empty');
  finally
    M.Free;
  end;
  M := TJitModule.Create(Models + 'foo5.pt');
  try
    V := M.ForwardIV([TIValue.StringList(['foo', 'bar', 'foobar'])]);
    Check((V.Tag = ivGenericList) and (V.Count = 3) and
      (V.Item(0).AsString = 'fo') and (V.Item(1).AsString = 'ba') and
      (V.Item(2).AsString = 'fooba'), 'foo5: [fo, ba, fooba]');
  finally
    M.Free;
  end;
  M := TJitModule.Create(Models + 'foo8.pt');
  try
    V := M.MethodIV('generate', [TIValue.GenericDict(['bar',
      TTensor.FromArray([1.0, 7]), 'foo', TTensor.FromArray([1.0, 2, 3])])]);
    Check((V.Tag = ivTuple) and (Values(V.Item(0).AsTensor) = '1 2 3') and
      (Values(V.Item(1).AsTensor) = '1 7'), 'foo8: generate = ([1, 2, 3], [1, 7]) (H22)');
  finally
    M.Free;
  end;
end;

{ 8. ivalue round trips (tch-rs ivalue_round_trip) }
procedure TestRoundTrip;
var
  PiD, E: Double;
  V, R: TIValue;
  Raised: Boolean;

  procedure Same(const A: TIValue; const What: String);
  begin
    Check(SameIValue(RoundTrip(A), A), 'round trip ' + What);
  end;

begin
  WriteLn('ivalue round trips (H24)');
  PiD := Pi;
  E := Exp(1);
  Same(TIValue.None, 'None');
  Same(True, 'True');
  Same(False, 'False');
  Same(-1, '-1');
  Same(42, '42');
  Same(15, '15');
  Same('', 'empty string');
  Same('foobar', 'foobar');
  Same(TIValue.Tuple([42, PiD]), 'Tuple(42, Pi)');
  Same(TIValue.IntList([42, 1337]), 'IntList');
  Same(TIValue.DoubleList([E, PiD, 299792458.00001]), 'DoubleList');
  Same(TIValue.Tuple([TIValue.BoolList([True, False, True, True]),
    TIValue.DoubleList([E, PiD, 299792458.00001])]), 'Tuple(BoolList, DoubleList)');
  Same(TIValue.GenericList([42, 'foobar']), 'GenericList(42, foobar)');
  V := TIValue.GenericDict([42, 'foobar', 'foo', 'bar']);
  R := RoundTrip(V);
  Check(SameIValue(R, V) and (R.Key(0).AsInt = 42) and (R.Value(1).AsString = 'bar'),
    'GenericDict keeps insertion order (H24)');
  Same(TTensor.FromArray([1, 2, 3]), 'Tensor [1, 2, 3] kInt64');
  Same(TTensor.FromArray([1.5, 2.5]), 'Tensor [1.5, 2.5] kFloat');
  Same(TIValue.Tuple([]), 'empty Tuple');
  Same(TIValue.IntList([]), 'empty IntList');
  R := RoundTrip(TIValue.TensorList([TTensor.FromArray([1, 2]), TTensor.FromArray([3.0])]));
  Check((R.Tag = ivTensorList) and (R.Count = 2) and
    (Values(R.AsTensorList[1]) = '3'), 'TensorList round trip');
  R := RoundTrip(TIValue.StringList(['a', 'b']));
  Check((R.Tag = ivGenericList) and (R.Count = 2) and (R.Item(1).AsString = 'b'),
    'StringList comes back as GenericList of String');
  Raised := False;
  try
    TIValue.GenericDict([42, 'foobar', 'foo']);
  except
    on ETorchError do Raised := True;
  end;
  Check(Raised, 'odd GenericDict raises ETorchError');
  Raised := False;
  try
    RoundTrip(TIValue.FromDevice(TDevice.Cpu));
  except
    on ETorchNativeError do Raised := True;
  end;
  Check(Raised, 'a Device has no ati_tag: ETorchNativeError');
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

{ 9. tracing (create_traced) }
procedure TestTracing;
var
  M, M2: TJitModule;
  PiD: Double;
  R: TTensor;
  FileName: String;
  Raised: Boolean;
  D: TDoubleArray;
begin
  WriteLn('tracing: create_traced, raising closure (H23)');
  FileName := TempFile('traced', '.pt');
  M := TJitModule.CreateByTracing('MyModule', 'MyFn',
    [TTensor.scalar_tensor(0.0, kDouble, TDevice.Cpu),
     TTensor.scalar_tensor(1.0, kDouble, TDevice.Cpu)], @AddFn);
  try
    M.Save(FileName);
  finally
    M.Free;
  end;
  PiD := Pi;
  M := TJitModule.Create(FileName);
  try
    R := M.Method('MyFn', [TTensor.FromArray([1.0, 2, 3, 4, 5], kDouble),
      TTensor.FromArray([41.0, 1335, PiD - 3, 4, 5], kDouble)]);
    D := R.ToDoubleArray;
    Check((Length(D) = 5) and (D[0] = 42) and (D[1] = 1337) and (D[2] = PiD) and
      (D[3] = 8) and (D[4] = 10), 'MyFn = [42, 1337, Pi, 8, 10] exactly');
    Raised := False;
    try
      M.Method('nope', [R]);
    except
      on ETorchNativeError do Raised := True;
    end;
    Check(Raised, 'unknown method raises ETorchNativeError');
  finally
    M.Free;
  end;
  DeleteFile(FileName);

  Raised := False;
  try
    TJitModule.CreateByTracing('Bad', 'Fn', [TTensor.FromArray([1.0])], @FailingFn).Free;
  except
    on E: ETorchError do Raised := E.Message = 'closure failed';
  end;
  Check(Raised, 'a raising closure propagates its exception');
  M2 := nil;
  try
    M2 := TJitModule.CreateByTracing('Again', 'Fn', [TTensor.FromArray([1.0]),
      TTensor.FromArray([2.0])], @AddFn);
    Check(Values(M2.Method('Fn', [TTensor.FromArray([3.0]), TTensor.FromArray([4.0])])) = '7',
      'tracing works again afterwards (H23)');
  except
    on E: Exception do
      Check(False, 'tracing again after a failure: ' + E.Message);
  end;
  M2.Free;
end;

const
  LinSource =
    'def forward(self, x):'#10 +
    '    return (torch.matmul(x, torch.t(self.weight)) + self.bias) * self.scale'#10;
  ModeSource =
    'def forward(self, x):'#10 +
    '    if self.training:'#10 +
    '        return x * 2'#10 +
    '    return x'#10;

{ 10. scripted module from source and TTrainableJitModule (H20, H21) }
procedure TestScripted;
var
  M: TJitModule;
  VS: TVarStore;
  J: TTrainableJitModule;
  Opt: TOptimizer;
  X, Y: TTensor;
  NP: TNamedTensorArray;
  Vars: TNamedTensorArray;
  FileName, FileName2: String;
begin
  WriteLn('scripted module from source, TTrainableJitModule (H20, H21)');
  FileName := TempFile('lin', '.pt');
  FileName2 := TempFile('lin-trained', '.pt');
  X := TTensor.FromArray([1.0, 2, 3]).view([1, 3]);
  Y := TTensor.FromArray([1.0]).view([1, 1]);
  M := TJitModule.CreateEmpty('Lin');
  try
    M.RegisterParameter('weight', TTensor.zeros([1, 3], kFloat, TDevice.Cpu).requires_grad_);
    M.RegisterParameter('bias', TTensor.zeros([1], kFloat, TDevice.Cpu).requires_grad_);
    M.RegisterParameter('scale', TTensor.ones([1], kFloat, TDevice.Cpu));
    M.Define(LinSource);
    Check(Values(M.Forward([X])) = '0', 'Forward([[1, 2, 3]]) = 0');
    NP := M.NamedParameters;
    Check((Length(NP) = 3) and (NP[0].Name = 'weight') and (NP[1].Name = 'bias') and
      (NP[2].Name = 'scale'), 'NamedParameters: weight, bias, scale');
    Check((Length(NP) = 3) and NP[0].Tensor.RequiresGrad and not NP[2].Tensor.RequiresGrad,
      'requires_grad kept (H20)');
    M.Save(FileName);
  finally
    M.Free;
  end;

  VS := TVarStore.Create(TDevice.Cpu);
  J := nil;
  Opt := nil;
  try
    J := TTrainableJitModule.Create(FileName, VS.Root / 'jit');
    Vars := VS.Variables;
    Check((VS.Count = 3) and (Vars[0].Name = 'jit.weight') and
      (Vars[1].Name = 'jit.bias') and (Vars[2].Name = 'jit.scale'),
      'store: jit.weight, jit.bias, jit.scale');
    Check(Length(VS.TrainableVariables) = 2, 'two trainable (scale has no grad)');
    Opt := TOptimizer.Sgd(VS, 0.1);
    Check(J.Forward(X).mse_loss(Y).DoubleValue([]) = 1, 'loss before = 1');
    Opt.BackwardStep(J.Forward(X).mse_loss(Y));
    Check(Values(Vars[0].Tensor) = '0.2 0.4 0.6', 'w = [0.2, 0.4, 0.6]');
    Check(Values(Vars[1].Tensor) = '0.2', 'b = 0.2');
    Check(Near(J.Forward(X).DoubleValue([0, 0]), 3.0, 1e-5),
      'J.Forward(X) = 3.0 after one SGD step on the store (H21)');
    J.Save(FileName2);
  finally
    Opt.Free;
    J.Free;
    VS.Free;
  end;
  M := TJitModule.Create(FileName2);
  try
    Check(Near(M.Forward([X]).DoubleValue([0, 0]), 3.0, 1e-5),
      'saved trained module reloads: Forward = 3.0 (H21)');
  finally
    M.Free;
  end;
  DeleteFile(FileName);
  DeleteFile(FileName2);
end;

{ 11. train flag (H20) }
procedure TestTrainFlag;
var
  M: TJitModule;
  VS: TVarStore;
  J: TTrainableJitModule;
  X: TTensor;
  FileName: String;
begin
  WriteLn('train flag: self.training, SetEval/SetTrain, Forward(X, Train) (H20)');
  FileName := TempFile('mode', '.pt');
  X := TTensor.FromArray([1.0, 2]);
  M := TJitModule.CreateEmpty('Mode');
  try
    M.Define(ModeSource);
    Check(Values(M.Forward([X])) = '2 4', 'fresh module trains: 2x');
    M.SetEval;
    Check(Values(M.Forward([X])) = '1 2', 'SetEval: x');
    M.SetTrain;
    Check(Values(M.Forward([X])) = '2 4', 'SetTrain: 2x');
    M.Save(FileName);
  finally
    M.Free;
  end;
  VS := TVarStore.Create(TDevice.Cpu);
  J := nil;
  try
    J := TTrainableJitModule.Create(FileName, VS.Root / 'm');
    Check(Values(J.Forward(X, True)) = '2 4', 'Forward(X, True) = 2x');
    Check(Values(J.Forward(X)) = '1 2', 'Forward(X) = x');
    Check(Values(J.Forward(X, True)) = '2 4', 'Forward(X, True) = 2x again');
  finally
    J.Free;
    VS.Free;
  end;
  DeleteFile(FileName);
end;

{ 12. globals (profiling_mode, tensor_expr_fuser) }
procedure TestGlobals;
var
  Raised: Boolean;
begin
  WriteLn('globals (H25)');
  Check(InitialProfiling, 'profiling mode on in a fresh process (H25)');
  JitSetProfilingMode(False);
  Check(not JitGetProfilingMode, 'SetProfilingMode(False)');
  JitSetProfilingMode(True);
  Check(JitGetProfilingMode, 'SetProfilingMode(True)');
  JitSetTensorExprFuserEnabled(True);
  Check(JitGetTensorExprFuserEnabled, 'tensor expr fuser on');
  JitSetTensorExprFuserEnabled(False);
  Check(not JitGetTensorExprFuserEnabled, 'tensor expr fuser off');
  JitSetGraphExecutorOptimize(True);
  Check(True, 'SetGraphExecutorOptimize(True) raises nothing');
  { libtorch 2.13 removed nvfuser: enabling raises (DESIGN 4.8, measured) }
  SetJitFuserCudaEnabled(False);
  Check(not JitFuserCudaEnabled, 'CUDA fuser flag off');
  Raised := False;
  try
    SetJitFuserCudaEnabled(True);
  except
    on E: ETorchNativeError do Raised := Pos('nvfuser support in torchscript is removed', E.Message) > 0;
  end;
  Check(Raised and not JitFuserCudaEnabled, 'SetJitFuserCudaEnabled(True) raises: nvfuser removed');
end;

{ 13. errors and handles }
{ 14. TorchScript objects (tch-rs jit6, jit_double_free; DESIGN 4.6.9) }

function RssKiB: Int64;
var
  F: TextFile;
  Size, Resident: Int64;
begin
  AssignFile(F, '/proc/self/statm');
  Reset(F);
  Read(F, Size, Resident);
  CloseFile(F);
  Result := Resident * 4;
end;

function FX: TTensor;
begin
  Result := TTensor.FromArray([3.0, 4, 5]);
end;

function FT1: TTensor;
begin
  Result := TTensor.FromArray([1.0, 2, 3]);
end;

function FT2: TTensor;
begin
  Result := TTensor.FromArray([4.0, 5, 6]);
end;

{ Message of the ETorchNativeError the last Call* raised, '' if none. }
var
  NativeMsg: String;

procedure NoteNative(E: Exception);
begin
  if E is ETorchNativeError then
    NativeMsg := E.Message
  else
    NativeMsg := 'other ' + E.ClassName + ': ' + E.Message;
end;

function Has(const Sub: String): Boolean;
begin
  Result := Pos(Sub, NativeMsg) > 0;
end;

procedure TestObjectsFoo6;
var
  M: TJitModule;
  R: TIValue;
  O: IJitObject;
  Raised: Boolean;
begin
  M := TJitModule.Create(Models + 'foo6.pt');
  try
    R := M.ForwardIV([FX]);
    Check((R.Tag = ivObject) and (R.TagName = 'Object'), 'a. foo6: forward is an Object');
    O := R.AsObject;
    Check(Values(O.MethodIV('y', []).AsTensor) = '6 8 10', 'a. foo6: y() = [6, 8, 10] (jit6)');
    Check(Values(O.GetAttr('x').AsTensor) = '3 4 5', 'a. foo6: getattr x = [3, 4, 5]');
    Raised := False;
    try
      R.AsTensor;
    except
      on E: ETorchError do
        Raised := (not (E is ETorchNativeError)) and
          (E.Message = 'ivalue: expected Tensor, got Object');
    end;
    Check(Raised, 'a. AsTensor on an object: ETorchError expected Tensor, got Object');
    NativeMsg := '';
    try
      M.Forward([FX]);
    except
      on E: Exception do NoteNative(E);
    end;
    Check(Has('forward did not return a tensor'), 'a. Forward (tensor API): ' + NativeMsg + ' (H27)');

    NativeMsg := '';
    try
      O.GetAttr('nope');
    except
      on E: Exception do NoteNative(E);
    end;
    Check(Has('does not have an attribute with name ''nope'''), 'b. GetAttr nope: ' + NativeMsg);
    NativeMsg := '';
    try
      O.MethodIV('nope', []);
    except
      on E: Exception do NoteNative(E);
    end;
    Check(Has('Couldn''t find method: ''nope'''), 'b. MethodIV nope: ' + NativeMsg);
    NativeMsg := '';
    try
      O.MethodIV('y', [7]);
    except
      on E: Exception do NoteNative(E);
    end;
    Check(Has('Expected at most 1 argument(s)'), 'b. MethodIV y(7): ' + NativeMsg);

    O := M.CreateClass('__torch__.TorchScriptClass', [FX]);
    Check(Values(O.MethodIV('y', []).AsTensor) = '6 8 10',
      'd. foo6: CreateClass TorchScriptClass(x).y() = [6, 8, 10] (H27)');
  finally
    M.Free;
  end;
end;

procedure TestObjectsFoo7;
var
  M: TJitModule;
  Inp, W, A, B: TIValue;
  O: IJitObject;
begin
  M := TJitModule.Create(Models + 'foo7.pt');
  try
    Inp := M.MethodIV('make_input_object', [FT1, FT2]);
    Check(Inp.Tag = ivObject, 'c. foo7: make_input_object is an Object');
    Check(Values(M.MethodIV('add_them', [Inp]).AsTensor) = '5 7 9', 'c. add_them = [5, 7, 9]');
    Check(Values(M.MethodIV('add_them', [Inp]).AsTensor) = '5 7 9',
      'c. add_them again with the same handle = [5, 7, 9] (jit_double_free)');
    Check((Values(Inp.AsObject.GetAttr('foo').AsTensor) = '1 2 3') and
      (Values(Inp.AsObject.GetAttr('bar').AsTensor) = '4 5 6'), 'c. getattr foo, bar');

    O := M.CreateClass('__torch__.InputObject', [FT1, FT2]);
    Check(Values(M.MethodIV('add_them', [O]).AsTensor) = '5 7 9',
      'd. CreateClass InputObject; add_them([O]) = [5, 7, 9]');
    Check(Values(M.MethodIV('add_them', [TIValue.FromObject(O)]).AsTensor) = '5 7 9',
      'd. add_them([FromObject(O)]) = [5, 7, 9]');
    NativeMsg := '';
    try
      M.CreateClass('InputObject', [FT1, FT2]);
    except
      on E: Exception do NoteNative(E);
    end;
    Check(Has('Could not find class with name: ''InputObject'' in module.'),
      'd. CreateClass unqualified: ' + NativeMsg);
    NativeMsg := '';
    try
      M.CreateClass('__torch__.InputObject', [FT1]);
    except
      on E: Exception do NoteNative(E);
    end;
    Check(Has('__init__() is missing value for argument ''bar'''),
      'd. CreateClass one argument: ' + NativeMsg);
    NativeMsg := '';
    try
      M.MethodIV('add_them', [5]);
    except
      on E: Exception do NoteNative(E);
    end;
    Check(Has('Expected a value of type ''__torch__.InputObject'),
      'd. add_them(5): ' + NativeMsg);

    W := RoundTrip(TIValue.Tuple([O, 42]));
    Check((W.Count = 2) and (W.Item(1).AsInt = 42) and
      (Values(W.Item(0).AsObject.GetAttr('foo').AsTensor) = '1 2 3') and
      (W.Item(0).AsObject.Raw <> O.Raw), 'e. object through a Tuple: fresh IValue, same object');
    W := RoundTrip(TIValue.GenericList([O]));
    Check((W.Tag = ivGenericList) and (W.Count = 1) and
      (Values(W.Item(0).AsObject.GetAttr('bar').AsTensor) = '4 5 6') and
      (W.Item(0).AsObject.Raw <> O.Raw), 'e. object through a GenericList (H26)');
    W := RoundTrip(TIValue.GenericDict(['k', O]));
    Check((W.Tag = ivGenericDict) and (W.Count = 1) and (W.Key(0).AsString = 'k') and
      (Values(W.Value(0).AsObject.GetAttr('foo').AsTensor) = '1 2 3'),
      'e. object as a GenericDict value (H26)');

    A := TIValue.FromObject(O);
    B := A;
    Check(A.AsObject.Raw = B.AsObject.Raw, 'f. copies share the handle');
    A := Default(TIValue);
    Check(Values(B.AsObject.GetAttr('foo').AsTensor) = '1 2 3', 'f. the copy survives the original');
  finally
    M.Free;
  end;
  Check(Values(O.GetAttr('bar').AsTensor) = '4 5 6', 'f. the object outlives its module');
end;

procedure ObjectLoop(N: Integer);
var
  M: TJitModule;
  O: IJitObject;
  V: TIValue;
  I, Live1, Live2, Obj1, Obj2: Integer;
  Rss1, Rss2: Int64;
begin
  M := TJitModule.Create(Models + 'foo6.pt');
  try
    O := M.ForwardIV([FX]).AsObject;
  finally
    M.Free;
  end;
  for I := 1 to N div 10 do
    V := O.MethodIV('y', []);
  Rss1 := RssKiB;
  Live1 := TchLiveHandles;
  Obj1 := JitLiveObjects;
  for I := 1 to N do
    V := O.MethodIV('y', []);
  Rss2 := RssKiB;
  Live2 := TchLiveHandles;
  Obj2 := JitLiveObjects;
  WriteLn(Format('  RSS KiB: %d -> %d over %d object method calls', [Rss1, Rss2, N]));
  Check((Live1 = Live2) and (Obj1 = Obj2), Format('g. live handles %d, objects %d constant (H28)',
    [Live2, Obj2]));
  Check(Rss2 - Rss1 < 1024, 'g. RSS flat (< 1 MiB) (H28)');
  Check(Values(V.AsTensor) = '6 8 10', 'g. last y() = [6, 8, 10]');
end;

procedure TestObjects;
var
  Before, BeforeObj: Integer;
begin
  WriteLn('foo6/foo7: TorchScript objects (H26-H28)');
  Before := TchLiveHandles;
  BeforeObj := JitLiveObjects;
  TestObjectsFoo6;
  TestObjectsFoo7;
  Check((TchLiveHandles = Before) and (JitLiveObjects = BeforeObj),
    Format('f. live handles %d -> %d, objects %d -> %d', [Before, TchLiveHandles,
    BeforeObj, JitLiveObjects]));
  ObjectLoop(10000);
end;

procedure TestErrors;
var
  M: TJitModule;
  Raised: Boolean;
  Before: Integer;
begin
  WriteLn('errors and handles');
  Raised := False;
  try
    TJitModule.Create(Models + 'missing.pt').Free;
  except
    on ETorchNativeError do Raised := True;
  end;
  Check(Raised, 'missing file: ETorchNativeError');
  M := TJitModule.CreateEmpty('Bad');
  try
    Raised := False;
    try
      M.Define('def forward(self, x): return');
    except
      on ETorchNativeError do Raised := True;
    end;
    Check(Raised, 'TorchScript compile error: ETorchNativeError');
  finally
    M.Free;
  end;
  Before := TchLiveHandles;
  TestFoo;
  TestScripted;
  Check(TchLiveHandles = Before, Format('live handles %d -> %d after modules, store, optimizer',
    [Before, TchLiveHandles]));
end;

begin
  InitialProfiling := JitGetProfilingMode;
  TestFoo;
  TestFoo1;
  TestFoo2;
  TestFoo3To8;
  TestRoundTrip;
  TestTracing;
  TestScripted;
  TestTrainFlag;
  TestGlobals;
  TestErrors;
  TestObjects;
  Check(TchLiveHandles = 0, 'no live handles at the end');
  Check(JitLiveObjects = 0, 'no live objects at the end');
  if Failures = 0 then
    WriteLn('OK')
  else
  begin
    WriteLn(Failures, ' FAILED');
    Halt(1);
  end;
end.
