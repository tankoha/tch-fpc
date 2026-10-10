{ TchNN plumbing against the mock libtch (tests/mock/run.sh), no libtorch:
  store bookkeeping, TInit argument passing, save names, the load callback
  trampoline and its error paths, optimizer construction and lazy-add
  counts, TSequential chaining. Numbers are the mock's, not libtorch's
  (DESIGN.md 4.6.7, "What the mock can cover"). }
program nn_mock_test;

{$mode objfpc}{$H+}

uses
  SysUtils, ctypes, Tch, TchNN;

{ mock-only introspection, exported by tests/mock/mock_tch.c }
function mock_live_tensors: cint; cdecl; external 'tch';
function mock_grad_enabled: cint; cdecl; external 'tch';
function mock_last_uniform_from: Double; cdecl; external 'tch';
function mock_last_uniform_to: Double; cdecl; external 'tch';
function mock_saved_count: cint; cdecl; external 'tch';
function mock_saved_name(I: cint): PChar; cdecl; external 'tch';
function mock_saved_file: PChar; cdecl; external 'tch';
function mock_last_load: cint; cdecl; external 'tch';
function mock_last_load_device: cint; cdecl; external 'tch';
function mock_live_opts: cint; cdecl; external 'tch';
function mock_opt_kind: cint; cdecl; external 'tch';
function mock_opt_adds(Group: cint): cint; cdecl; external 'tch';
function mock_opt_groups: cint; cdecl; external 'tch';
function mock_opt_zero_grads: cint; cdecl; external 'tch';
function mock_opt_steps: cint; cdecl; external 'tch';
function mock_opt_lr: Double; cdecl; external 'tch';

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

procedure TestStore;
var
  VS: TVarStore;
  R: TVarPath;
  T, U: TTensor;
  NT: TNamedTensorArray;
begin
  WriteLn('store: names, duplicates, requires_grad, Freeze');
  VS := TVarStore.Create(TDevice.Cpu);
  try
    R := VS.Root;
    T := (R / 'a' / 'b').Zeros('t2', [3]);
    R.Ones('t1', [4]);
    R.Ones('t1', [4]);
    U := R.ZerosNoTrain('buf', [2]);
    NT := VS.Variables;
    Check((VS.Count = 4) and (NT[0].Name = 'a.b.t2') and (NT[1].Name = 't1') and
      (NT[2].Name = 't1__2') and (NT[3].Name = 'buf'), 'a.b.t2 t1 t1__2 buf');
    Check(T.RequiresGrad and not U.RequiresGrad, 'trainable flag set through set_requires_grad');
    Check(Length(VS.TrainableVariables) = 3, 'three trainable');
    VS.Freeze;
    Check(not T.RequiresGrad, 'Freeze');
    VS.Unfreeze;
    Check(T.RequiresGrad and not U.RequiresGrad, 'Unfreeze leaves the buffer alone');
    Check(T.SameHandle(NT[0].Tensor), 'store and caller share the handle');
  finally
    VS.Free;
  end;
end;

procedure TestInitPlumbing;
var
  VS: TVarStore;
  T: TTensor;
begin
  WriteLn('init: Uniform / Kaiming bounds, Apply under no-grad');
  VS := TVarStore.Create(TDevice.Cpu);
  try
    T := VS.Root.KaimingUniform('w', [4, 16]);
    Check(Abs(mock_last_uniform_to - Sqrt(6 / 16)) < 1e-12, 'KaimingUniform [4, 16]: bound sqrt(6/16)');
    Check(mock_last_uniform_from = -mock_last_uniform_to, 'symmetric');
    VS.Root.Uniform('u', [2], -0.5, 0.25);
    Check((mock_last_uniform_from = -0.5) and (mock_last_uniform_to = 0.25), 'Uniform(-0.5, 0.25)');
    TInit.Constant(3).Apply(T);
    Check(T.DoubleValue([1, 2]) = 3, 'Apply(Constant(3)) fills');
    Check(mock_grad_enabled = 1, 'grad mode restored after Apply');
    TInit.Kaiming(idUniform, fmFanOut, nlLinear).Apply(T);
    Check(Abs(mock_last_uniform_to - Sqrt(3) / 2) < 1e-12, 'Kaiming fan-out, linear gain: sqrt(3)/sqrt(4)');
  finally
    VS.Free;
  end;
end;

procedure TestSave;
var
  VS: TVarStore;
begin
  WriteLn('save: "." -> "|", insertion order');
  VS := TVarStore.Create(TDevice.Cpu);
  try
    (VS.Root / 'enc' / 'l1').Zeros('weight', [2]);
    VS.Root.Zeros('t1', [1]);
    VS.Save('model.ot');
    Check(mock_saved_count = 2, 'two tensors');
    Check((StrPas(mock_saved_name(0)) = 'enc|l1|weight') and
      (StrPas(mock_saved_name(1)) = 't1'), 'names enc|l1|weight, t1');
    Check(StrPas(mock_saved_file) = 'model.ot', 'file name passed');
  finally
    VS.Free;
  end;
end;

procedure TestLoad;
var
  VS: TVarStore;
  T2, T1: TTensor;
  Live: Integer;
  Missing: TStringArray;
  Msg: String;
begin
  WriteLn('load: callback, "|" -> ".", error paths');
  VS := TVarStore.Create(TDevice.Cpu);
  try
    T2 := (VS.Root / 'a' / 'b').Zeros('t2', [3]);
    T1 := VS.Root.Zeros('t1', [4]);
    VS.Load('m.ot');
    Check((mock_last_load = 1) and (mock_last_load_device = -1), '.ot: at_load_callback_with_device, CPU');
    Check((T2.DoubleValue([0]) = 7) and (T1.DoubleValue([3]) = 42), 'a|b|t2 -> a.b.t2 = 7, t1 = 42');
    Check(mock_grad_enabled = 1, 'grad mode restored after Load');
    VS.Load('m.pt');
    Check(mock_last_load = 2, '.pt: at_loadz_callback_with_device');

    Live := mock_live_tensors;
    Msg := '';
    try
      VS.Load('err_after_first.ot');
    except
      on E: ETorchNativeError do Msg := E.Message;
    end;
    Check(Msg = 'mock: load failed', 'C error after a callback: ETorchNativeError');
    Check(mock_live_tensors = Live, 'tensor handed over before the error is freed');

    Msg := '';
    try
      VS.Load('nil_tensor.ot');
    except
      on E: ETorchNativeError do Msg := 'native';
      on E: ETorchError do Msg := E.Message;
    end;
    Check(Msg = 'TTensor.FromRaw: nil pointer', 'exception in the callback re-raised after the call: ' + Msg);
    Check(mock_live_tensors = Live, 'later callback skipped, its tensor still freed');
    Check(T1.Dim = 1, 'error slot clear afterwards');

    VS.Root.Zeros('extra', [1]);
    Msg := '';
    try
      VS.Load('m.ot');
    except
      on E: ETorchError do Msg := E.Message;
    end;
    Check(Msg = 'variable extra not found in m.ot', 'missing name: ' + Msg);
    Missing := VS.LoadPartial('m.ot');
    Check((Length(Missing) = 1) and (Missing[0] = 'extra'), 'LoadPartial returns [extra]');
  finally
    VS.Free;
  end;
end;

procedure TestOptimizer;
var
  VS: TVarStore;
  Opt: TOptimizer;
  Raised: Boolean;
begin
  WriteLn('optimizer: construction, lazy add, groups, free');
  VS := TVarStore.Create(TDevice.Cpu);
  Opt := nil;
  try
    VS.Root.Zeros('a', [1]);
    VS.Root.ZerosNoTrain('buf', [1]);
    VS.Root.SetGroup(2).Zeros('b', [1]);
    Opt := TOptimizer.Adam(VS, 0.1);
    Check(mock_live_opts = 1, 'one optimizer alive');
    Check(mock_opt_kind = 1, 'ato_adam');
    Check((mock_opt_adds(0) = 1) and (mock_opt_adds(2) = 1) and (mock_opt_groups = 3),
      'construction adds a to group 0, b to group 2, not the buffer');
    VS.Root.Zeros('c', [1]);
    VS.Root.SetGroup(1).Zeros('d', [1]);
    Opt.SetLearningRate(0.5);
    Check(mock_opt_adds(0) = 1, 'setters do not add');
    Check(mock_opt_lr = 0.5, 'SetLearningRate');
    Opt.ZeroGrad;
    Check((mock_opt_adds(0) = 2) and (mock_opt_adds(1) = 1) and (mock_opt_zero_grads = 1),
      'ZeroGrad adds c and d first');
    Opt.Step;
    Check((mock_opt_adds(0) = 2) and (mock_opt_steps = 1), 'Step adds nothing new');
    Opt.SetLearningRateGroup(1, 0);
    Raised := False;
    try Opt.SetLearningRateGroup(5, 0);
    except on ETorchNativeError do Raised := True; end;
    Check(Raised, 'SetLearningRateGroup on a missing group raises');
    FreeAndNil(Opt);
    Check(mock_live_opts = 0, 'ato_free on Destroy');

    Raised := False;
    try
      Opt := TOptimizer.Sgd(VS, -1);
    except
      on ETorchNativeError do Raised := True;
    end;
    Check(Raised and (Opt = nil) and (mock_live_opts = 0), 'constructor error raises before CreateRaw');
    Opt := TOptimizer.RmsProp(VS, 0.01);
    Check((mock_opt_kind = 3) and (mock_opt_adds(0) = 2) and (mock_opt_adds(1) = 1) and
      (mock_opt_adds(2) = 1), 'RmsProp adds all four trainable at construction');
    FreeAndNil(Opt);
    Opt := TOptimizer.AdamW(VS, 0.01);
    Check(mock_opt_kind = 2, 'ato_adamw');
  finally
    Opt.Free;
    VS.Free;
  end;
end;

function AddOne(const X: TTensor): TTensor;
begin
  Result := X.add_scalar(1);
end;

function AddTrain(const X: TTensor; Train: Boolean): TTensor;
begin
  if Train then
    Result := X.add_scalar(10)
  else
    Result := X;
end;

procedure TestSequential;
var
  Seq: TSequential;
  X: TTensor;
  All: TTensorArray;
begin
  WriteLn('sequential: chaining over add_scalar');
  Seq := TSequential.Create;
  try
    Seq.AddFn(@AddOne).AddFn(@AddTrain).AddFn(@AddOne);
    X := TTensor.zeros([2], kFloat, TDevice.Cpu);
    Check(Seq.Forward(X).DoubleValue([0]) = 2, 'Forward(X) = X + 2');
    Check(Seq.Forward(X, True).DoubleValue([1]) = 12, 'Forward(X, True) = X + 12');
    All := Seq.ForwardAll(X, True);
    Check((Length(All) = 3) and (All[1].DoubleValue([0]) = 11), 'ForwardAll');
  finally
    Seq.Free;
  end;
end;

begin
  TestStore;
  TestInitPlumbing;
  TestSave;
  TestLoad;
  TestOptimizer;
  TestSequential;
  Check(TchLiveHandles = 0, 'no live handles');
  Check(mock_live_tensors = 0, 'mock: no tensors left');
  if Failures = 0 then
    WriteLn('OK')
  else
  begin
    WriteLn(Failures, ' FAILED');
    Halt(1);
  end;
end.
