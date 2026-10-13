{ Milestones 2 and 3 (DESIGN.md 4.6.7, 4.6.10) against the real libtorch:
  TInit, TVarStore, save/load, optimizers, layers, TSequential, hypotheses
  H13-H18, and the milestone 3 layers (group 11). Expected values are
  worked out by hand in DESIGN.md 4.6.7 and 4.6.10 or follow the named
  tch-rs 4227b89b test.

  Build (from the repo root, after building build/libtch):
    fpc -Mobjfpc -Sh -Fusrc -FUbuild/nn -Flbuild/libtch \
      -k-rpath=$PWD/build/libtch tests/nn/nn_test.pas -obuild/nn_test
    build/nn_test           # all checks
    build/nn_test h16       # SetMomentumGroup on Adam only; run once more
                            # with LD_LIBRARY_PATH=build/libtch-unpatched }
program nn_test;

uses
  SysUtils, Tch, TchNN;

{$I ../common/layers.inc}

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

function Near(A, B, Tol: Double): Boolean;
begin
  Result := Abs(A - B) <= Tol;
end;

function MaxAbsDiff(const A, B: TTensor): Double;
begin
  Result := (A - B).abs.max.DoubleValue([]);
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

function ShapeIs(const T: TTensor; const S: array of Int64): Boolean;
var
  A: TInt64Array;
  K: Integer;
begin
  A := T.Shape;
  Result := Length(A) = Length(S);
  if Result then
    for K := 0 to High(S) do
      if A[K] <> S[K] then
        Exit(False);
end;

function MeanOf(const T: TTensor): Double;
begin
  Result := T.mean(kFloat).DoubleValue([]);
end;

function TempFile(const Tag: String): String;
begin
  Result := GetTempDir + Format('tch-fpc-nn-%s-%d.ot', [Tag, GetProcessID]);
end;

{ 1. Init }
procedure TestInit;
var
  T, Q, I3: TTensor;
  D: TDoubleArray;
  K: Integer;
  InRange: Boolean;
  Bound: Double;
begin
  WriteLn('init: KaimingUniform bound, Constant, Orthogonal (H14)');
  ManualSeed(42);
  T := TInit.KaimingUniform.Make([4, 16], TDevice.Cpu, kFloat);
  Check(ShapeIs(T, [4, 16]), 'KaimingUniform shape [4, 16]');
  Bound := Sqrt(6 / 16);
  D := T.ToDoubleArray;
  InRange := True;
  for K := 0 to High(D) do
    if Abs(D[K]) > Bound then
      InRange := False;
  Check(InRange, Format('KaimingUniform values within +-%.4f', [Bound]));
  Check(Abs(MeanOf(T)) < 0.1, 'KaimingUniform mean near 0');
  Check(FanForDims(fmFanIn, [8, 3, 5, 5]) = 75, 'fan-in of conv weight [8, 3, 5, 5] = 75');
  Check(FanForDims(fmFanOut, [8, 3, 5, 5]) = 200, 'fan-out of conv weight = 200');

  T := TInit.Constant(3).Make([2, 3], TDevice.Cpu, kFloat);
  Check(Values(T) = '3 3 3 3 3 3', 'Constant(3) all 3');
  T := TInit.Constant(1).Make([2], TDevice.Cpu, kDouble);
  Check((T.Kind = kDouble) and (Values(T) = '1 1'), 'Constant(1) kDouble');

  Q := TInit.Orthogonal.Make([3, 5], TDevice.Cpu, kFloat);
  I3 := TTensor.eye(3, kFloat, TDevice.Cpu);
  Check(ShapeIs(Q, [3, 5]), 'Orthogonal [3, 5] shape');
  Check(MaxAbsDiff(Q.matmul(Q.t), I3) < 1e-5, 'Orthogonal [3, 5]: Q Qt = I (H14)');
  Q := TInit.Orthogonal.Make([5, 3], TDevice.Cpu, kFloat);
  Check(ShapeIs(Q, [5, 3]), 'Orthogonal [5, 3] shape');
  Check(MaxAbsDiff(Q.t.matmul(Q), I3) < 1e-5, 'Orthogonal [5, 3]: Qt Q = I (H14)');
  Q := TInit.Orthogonal(2).Make([2, 3, 2], TDevice.Cpu, kFloat);
  Check(ShapeIs(Q, [2, 3, 2]), 'Orthogonal [2, 3, 2] keeps the requested shape');
  Check(MaxAbsDiff(Q.view([2, 6]).matmul(Q.view([2, 6]).t),
    TTensor.eye(2, kFloat, TDevice.Cpu).mul_scalar(4)) < 1e-4,
    'Orthogonal gain 2: rows have norm 2');
  try
    TInit.Orthogonal.Make([4], TDevice.Cpu, kFloat);
    Check(False, 'Orthogonal of 1-d raises');
  except
    on E: ETorchError do
      Check(Pos('2 or more dimensions', E.Message) > 0, 'Orthogonal of 1-d raises');
  end;
end;

{ H17: in-place init of a trainable leaf needs no-grad }
procedure TestApplyGuard;
var
  VS: TVarStore;
  W: TTensor;
begin
  WriteLn('init: Apply on a trainable leaf (H17)');
  VS := TVarStore.Create(TDevice.Cpu);
  try
    W := VS.Root.Zeros('w', [3]);
    try
      W.fill_(5);
      Check(False, 'fill_ without NoGradGuard raises');
    except
      on E: ETorchNativeError do
        Check(Pos('leaf Variable that requires grad', E.Message) > 0,
          'fill_ without NoGradGuard raises: ' + E.Message);
    end;
    TInit.Constant(5).Apply(W);
    Check(Values(W) = '5 5 5', 'Apply(Constant(5)) works');
    Check(W.RequiresGrad, 'still requires grad after Apply');
    TInit.Uniform(2, 3).Apply(W);
    Check((W.min.DoubleValue([]) >= 2) and (W.max.DoubleValue([]) <= 3), 'Apply(Uniform(2, 3))');
    TInit.Randn(10, 0).Apply(W);
    Check(Values(W) = '10 10 10', 'Apply(Randn(10, 0))');
    TInit.KaimingUniform.Apply(W);
    Check(W.abs.max.DoubleValue([]) <= Sqrt(6 / 1), 'Apply(KaimingUniform) on 1-d: fan 1');
    W := VS.Root.Zeros('o', [3, 5]);
    TInit.Orthogonal.Apply(W);
    Check(MaxAbsDiff(W.matmul(W.t), TTensor.eye(3, kFloat, TDevice.Cpu)) < 1e-5,
      'Apply(Orthogonal)');
  finally
    VS.Free;
  end;
end;

{ 2. Store }
procedure TestStore;
var
  VS: TVarStore;
  R: TVarPath;
  T1, T2, U: TTensor;
  NT: TNamedTensorArray;
  TV: TTensorArray;
  Raised: Boolean;
begin
  WriteLn('store: naming, duplicates, GetOrAdd, order, Freeze');
  VS := TVarStore.Create(TDevice.Cpu);
  try
    R := VS.Root;
    R.Sub('a').Sub('b').Ones('t2', [3]);
    R.Zeros('t1', [4]);
    (R / 'a' / 'b' / 'ccc').Ones('t123', [3]);
    R.Sub('a').Sub('b').Sub('ccc').Ones('t123', [3]);
    Check(VS.Count = 4, 'Count = 4');
    NT := VS.Variables;
    Check((NT[0].Name = 'a.b.t2') and (NT[1].Name = 't1') and
      (NT[2].Name = 'a.b.ccc.t123') and (NT[3].Name = 'a.b.ccc.t123__3'),
      'names in order: ' + NT[0].Name + ' ' + NT[1].Name + ' ' + NT[2].Name + ' ' + NT[3].Name);
    Check(VS.TryGet('a.b.t2', U) and ShapeIs(U, [3]), 'TryGet by full name');
    Check(not VS.TryGet('a.b.zz', U) and U.IsNil, 'TryGet of a missing name');
    Check(R.Sub('a').Sub('b').TryGet('t2', U), 'TVarPath.TryGet');

    Raised := False;
    try R.Sub('x.y'); except on ETorchError do Raised := True; end;
    Check(Raised, 'Sub(''x.y'') raises');
    Raised := False;
    try R.Zeros('a.b', [1]); except on ETorchError do Raised := True; end;
    Check(Raised, 'Zeros(''a.b'') raises');
    Raised := False;
    try R.Zeros('a|b', [1]); except on ETorchError do Raised := True; end;
    Check(Raised, 'Zeros(''a|b'') raises');
    Raised := False;
    try R.Sub(''); except on ETorchError do Raised := True; end;
    Check(Raised, 'Sub('''') raises');
    Check(VS.Count = 4, 'nothing added by the failed calls');

    T1 := R.GetOrAdd('key', [3, 1, 4], TInit.Constant(0));
    T2 := R.GetOrAdd('key', [1, 5, 9], TInit.Constant(0));
    Check(ShapeIs(T1, [3, 1, 4]) and ShapeIs(T2, [3, 1, 4]) and T1.SameHandle(T2),
      'GetOrAdd returns the existing [3, 1, 4] tensor');

    U := R.ZerosNoTrain('buf', [2]);
    Check(not U.RequiresGrad, 'ZerosNoTrain does not require grad');
    TV := VS.TrainableVariables;
    Check(Length(TV) = 5, 'five trainable variables');
    Check(TV[1].SameHandle(NT[1].Tensor) and TV[4].SameHandle(T1),
      'TrainableVariables in registration order');

    VS.Freeze;
    Check(not TV[0].RequiresGrad and not TV[4].RequiresGrad, 'Freeze: RequiresGrad = False');
    VS.Unfreeze;
    Check(TV[0].RequiresGrad and TV[4].RequiresGrad and not U.RequiresGrad,
      'Unfreeze: trainable ones True, buffer still False');

    U := R.VarCopy('copy', TTensor.FromArray([1.0, 2.0, 3.0]));
    Check((Values(U) = '1 2 3') and U.RequiresGrad, 'VarCopy');
  finally
    VS.Free;
  end;
end;

{ Review fixes: a renamed duplicate that is itself taken, TBatchNorm's Nd }
procedure TestStoreEdges;
var
  VS: TVarStore;
  NT: TNamedTensorArray;
  T: TTensor;
  Raised: Boolean;
begin
  WriteLn('store edges: rename clash, TBatchNorm Nd');
  VS := TVarStore.Create(TDevice.Cpu);
  try
    VS.Root.Zeros('x', [1]);
    VS.Root.Zeros('x__2', [1]);
    VS.Root.Zeros('x', [1]);       // x__2 is taken: the next free number
    NT := VS.Variables;
    Check((VS.Count = 3) and (NT[2].Name = 'x__3') and VS.TryGet('x__3', T) and
      (Length(VS.TrainableVariables) = 3), 'duplicate whose x__<Count> is taken becomes x__3');
    Raised := False;
    try
      TBatchNorm.Create(VS.Root / 'bn', 0, 3).Free;
    except
      on E: ETorchError do Raised := Pos('Nd = 0', E.Message) > 0;
    end;
    Check(Raised and (VS.Count = 3), 'TBatchNorm with Nd = 0 raises before registering');
  finally
    VS.Free;
  end;
end;

{ TVarStore.RollbackTo and the layer constructors that call it on failure }
procedure TestRollback;
var
  VS: TVarStore;
  T: TTensor;
  NT: TNamedTensorArray;
  Raised: Boolean;
  Msg: String;
  Cnt, TrCnt: Integer;
  Conv: TConv2D;
  Lin: TLinear;
  G: TGRU;
  CCfg: TConvConfig;
  LCfg: TLinearConfig;
  RCfg: TRNNConfig;
  Opt: TOptimizer;
  W: TTensor;
begin
  WriteLn('rollback: RollbackTo and failing layer constructors');
  VS := TVarStore.Create(TDevice.Cpu);
  try
    VS.Root.Zeros('a', [1]);
    VS.Root.ZerosNoTrain('b', [1]);
    VS.Root.Ones('c', [1]);
    VS.RollbackTo(1);
    NT := VS.Variables;
    Check((VS.Count = 1) and not VS.TryGet('b', T) and not VS.TryGet('c', T) and
      (Length(VS.TrainableVariables) = 1) and (NT[0].Name = 'a'),
      '1. RollbackTo(1) keeps only a, drops b and c from the index and trainable list');
    VS.Root.Zeros('b', [1]);
    NT := VS.Variables;
    Check((VS.Count = 2) and (NT[1].Name = 'b'), '1. b is registered again as plain b, not b__1');
    VS.RollbackTo(1);
    VS.RollbackTo(99);
    Check((VS.Count = 1) and (Length(VS.TrainableVariables) = 1), '1. RollbackTo(99) changes nothing');
    VS.RollbackTo(-1);
    Check((VS.Count = 0) and (Length(VS.TrainableVariables) = 0), '1. RollbackTo(-1) empties the store');
  finally
    VS.Free;
  end;

  VS := TVarStore.Create(TDevice.Cpu);
  Conv := nil;
  try
    VS.Root.Zeros('pre', [1]);
    Cnt := VS.Count;
    TrCnt := Length(VS.TrainableVariables);
    CCfg := DefaultConvConfig;
    CCfg.Groups := 0;
    Raised := False;
    Msg := '';
    try
      TConv2D.Create(VS.Root / 'cv', 1, 2, 3, CCfg).Free;
    except
      on E: Exception do
      begin
        Raised := True;
        Msg := E.ClassName;
      end;
    end;
    Check(Raised and (VS.Count = Cnt) and (Length(VS.TrainableVariables) = TrCnt),
      '2. TConv2D with Groups = 0 raises (' + Msg + '), store unchanged');
    CCfg.Groups := 1;
    Conv := TConv2D.Create(VS.Root / 'cv', 1, 2, 3, CCfg);
    NT := VS.Variables;
    Check((VS.Count = Cnt + 2) and (NT[Cnt].Name = 'cv.bias') and (NT[Cnt + 1].Name = 'cv.weight'),
      '2. a valid TConv2D at the same path registers cv.bias and cv.weight, no suffix');
  finally
    Conv.Free;
    VS.Free;
  end;

  VS := TVarStore.Create(TDevice.Cpu);
  try
    VS.Root.Zeros('pre', [1]);
    Cnt := VS.Count;
    TrCnt := Length(VS.TrainableVariables);
    LCfg := DefaultLinearConfig;
    LCfg.HasBsInit := True;
    LCfg.BsInit := TInit.Orthogonal;
    Raised := False;
    Msg := '';
    try
      TLinear.Create(VS.Root / 'l', 3, 2, LCfg).Free;
    except
      on E: Exception do
      begin
        Raised := True;
        Msg := E.ClassName;
      end;
    end;
    Check(Raised and (VS.Count = Cnt) and (Length(VS.TrainableVariables) = TrCnt) and
      not VS.TryGet('l.weight', T) and not VS.TryGet('l.bias', T),
      '3. TLinear with Orthogonal bias raises (' + Msg + ') after the weight, store rolled back');
  finally
    VS.Free;
  end;

  VS := TVarStore.Create(TDevice.Cpu);
  try
    VS.Root.Zeros('pre', [1]);
    Cnt := VS.Count;
    TrCnt := Length(VS.TrainableVariables);
    RCfg := DefaultRNNConfig;
    RCfg.BIhInit := TInit.Orthogonal;
    Raised := False;
    Msg := '';
    try
      TGRU.Create(VS.Root / 'g', 2, 4, RCfg).Free;
    except
      on E: Exception do
      begin
        Raised := True;
        Msg := E.ClassName;
      end;
    end;
    Check(Raised and (VS.Count = Cnt) and (Length(VS.TrainableVariables) = TrCnt) and
      not VS.TryGet('g.weight_ih_l0', T) and not VS.TryGet('g.weight_hh_l0', T) and
      not VS.TryGet('g.bias_ih_l0', T),
      '4. TGRU with Orthogonal BIhInit raises (' + Msg + ') after the weights, store rolled back');
  finally
    VS.Free;
  end;

  VS := TVarStore.Create(TDevice.Cpu);
  Opt := nil;
  try
    W := VS.Root.Zeros('w', [2]);
    Opt := TOptimizer.Sgd(VS, 0.1);
    LCfg := DefaultLinearConfig;
    LCfg.HasBsInit := True;
    LCfg.BsInit := TInit.Orthogonal;
    Raised := False;
    try
      Lin := TLinear.Create(VS.Root / 'l', 3, 2, LCfg);
      Lin.Free;
    except
      on E: Exception do Raised := True;
    end;
    Opt.BackwardStep((W * TTensor.FromArray([1.0, 2.0])).sum);
    Check(Raised and (Values(W) = '-0.1 -0.2') and (Length(Opt.TrainableVariables) = 1) and
      (VS.Count = 1), '5. an earlier optimizer keeps working after a failed layer');
  finally
    Opt.Free;
    VS.Free;
  end;
end;

procedure TestRollbackHandles;
var
  Before: Integer;
begin
  Before := TchLiveHandles;
  TestRollback;
  Check(TchLiveHandles = Before, Format('6. live handles %d -> %d after the rollback tests',
    [Before, TchLiveHandles]));
end;

{ H18 + H13: Freeze; Step and set_data with an earlier optimizer }
procedure TestFreezeAndSetKind;
var
  VS: TVarStore;
  Lin: TLinear;
  Opt: TOptimizer;
  A, E, Before: TTensor;
  Cfg: TLinearConfig;
begin
  WriteLn('store: Freeze; Step (H18), SetKind after optimizer (H13)');
  VS := TVarStore.Create(TDevice.Cpu);
  Opt := nil;
  try
    A := VS.Root.Ones('a', [2]);
    Opt := TOptimizer.Sgd(VS, 0.5);
    Opt.ZeroGrad;
    VS.Freeze;
    E := TTensor.ones([1]).set_requires_grad(True);
    (A * E).sum.Backward;
    Check(not A.grad.Defined, 'frozen variable gets no grad');
    Check(E.grad.Defined, 'unfrozen tensor gets a grad');
    Opt.Step;
    Check(Values(A) = '1 1', 'Freeze; Step leaves the values unchanged (H18)');
    VS.Unfreeze;
  finally
    Opt.Free;
    VS.Free;
  end;

  VS := TVarStore.Create(TDevice.Cpu);
  Lin := nil;
  Opt := nil;
  try
    Cfg := DefaultLinearConfig;
    Cfg.WsInit := TInit.Constant(1);
    Cfg.HasBsInit := True;
    Cfg.BsInit := TInit.Constant(0);
    Lin := TLinear.Create(VS.Root / 'lin', 2, 1, Cfg);
    Opt := TOptimizer.Sgd(VS, 0.1);
    VS.SetKind(kDouble);
    Check(Lin.Weight.Kind = kDouble, 'layer Weight sees kDouble after SetKind (H13)');
    Check(VS.Kind = kDouble, 'store kind is kDouble');
    Before := Lin.Weight.Copy;
    Opt.BackwardStep(Lin.Forward(TTensor.FromArray([1.0, 2.0]).to_dtype(kDouble).view([1, 2])).sum);
    { d/dW sum(W x + b) = x = [1, 2]; W := 1 - 0.1 x }
    Check(Values(Lin.Weight) = '0.9 0.8',
      'earlier optimizer steps the converted variable (H13): ' + Values(Lin.Weight));
    Check(Lin.Weight.Kind = kDouble, 'still kDouble after the step');
    VS.SetDevice(TDevice.Cpu);
    Check(Lin.Weight.Device.DeviceType = dtCpu, 'SetDevice(Cpu) keeps the handle usable');
    (VS.Root / 'lin').SetKind(kFloat);
    Check((Lin.Weight.Kind = kFloat) and (Lin.Bias.Kind = kFloat),
      'TVarPath.SetKind on the sub-tree');
  finally
    Opt.Free;
    Lin.Free;
    VS.Free;
  end;
end;

function LinearXs: TTensor;
var
  A: array[0..13] of Single;
  K: Integer;
begin
  for K := 0 to 13 do
    A[K] := K + 1;
  Result := TTensor.FromSingleArray(A).view([14, 1]);
end;

{ 3. Linear regression (tch-rs optimizer_test) }
procedure TestLinearRegression;
var
  VS: TVarStore;
  Lin: TLinear;
  Opt: TOptimizer;
  Cfg: TLinearConfig;
  Xs, Ys: TTensor;
  Initial, L: Double;
  K: Integer;
begin
  WriteLn('optimizer: linear regression with SGD');
  VS := TVarStore.Create(TDevice.Cpu);
  Lin := nil;
  Opt := nil;
  try
    Xs := LinearXs;
    Ys := Xs * 0.42 + 1.337;
    Opt := TOptimizer.Sgd(VS, 0);
    Cfg := DefaultLinearConfig;
    Cfg.WsInit := TInit.Constant(0);
    Cfg.BsInit := TInit.Constant(0);
    Cfg.HasBsInit := True;
    Lin := TLinear.Create(VS.Root, 1, 1, Cfg);
    Check(VS.Count = 2, 'TLinear registers weight and bias');
    Initial := Lin.Forward(Xs).mse_loss(Ys).DoubleValue([]);
    Check(Near(Initial, 22.99967, 1e-3), Format('initial loss %.5f = 22.99967', [Initial]));
    Opt.SetLearningRate(1e-2);
    Opt.BackwardStep(Lin.Forward(Xs).mse_loss(Ys));
    Check(Near(Lin.Weight.DoubleValue([0, 0]), 0.80955, 1e-3) and
      Near(Lin.Bias.DoubleValue([0]), 0.08974, 1e-3),
      Format('one step: w = %.5f, b = %.5f', [Lin.Weight.DoubleValue([0, 0]),
        Lin.Bias.DoubleValue([0])]));
    for K := 2 to 49 do
      Opt.BackwardStep(Lin.Forward(Xs).mse_loss(Ys));
    L := Lin.Forward(Xs).mse_loss(Ys).DoubleValue([]);
    Check(L < 0.25, Format('loss after 49 steps %.4f < 0.25', [L]));
    TInit.Constant(0).Apply(Lin.Weight);
    TInit.Constant(0).Apply(Lin.Bias);
    L := Lin.Forward(Xs).mse_loss(Ys).DoubleValue([]);
    Check(L = Initial, 'Apply(Constant(0)) restores the initial loss');
    Opt.SetLearningRate(1e-10);
    for K := 1 to 49 do
      Opt.BackwardStep(Lin.Forward(Xs).mse_loss(Ys));
    L := Lin.Forward(Xs).mse_loss(Ys).DoubleValue([]);
    { tch-rs checks final - initial < 1e-5 (signed). By hand the loss drops
      by 49 * 1e-10 * (80.955^2 + 8.974^2) = 3.25e-5 (gradient barely moves). }
    Check(L - Initial < 1e-5, 'lr 1e-10: loss does not grow');
    Check(Near(L - Initial, -3.25e-5, 5e-6),
      Format('lr 1e-10: loss drops by %.3g (hand: -3.25e-5)', [L - Initial]));
  finally
    Opt.Free;
    Lin.Free;
    VS.Free;
  end;

  VS := TVarStore.Create(TDevice.Cpu);
  Lin := nil;
  try
    Cfg := DefaultLinearConfig;
    Cfg.Bias := False;
    Lin := TLinear.Create(VS.Root, 3, 2, Cfg);
    Check((VS.Count = 1) and Lin.Bias.IsNil, 'TLinear without bias: one variable');
    Check(ShapeIs(Lin.Forward(TTensor.ones([4, 3])), [4, 2]), 'forward without bias');
  finally
    Lin.Free;
    VS.Free;
  end;
end;

{ 4. Adam's first step is lr * sign(g); other constructors and setters }
procedure TestAdam;
var
  VS: TVarStore;
  Opt: TOptimizer;
  P: TTensor;
begin
  WriteLn('optimizer: Adam, AdamW, RmsProp, setters');
  VS := TVarStore.Create(TDevice.Cpu);
  Opt := nil;
  try
    P := VS.Root.VarCopy('p', TTensor.FromArray([1.0, -2.0]));
    Opt := TOptimizer.Adam(VS, 0.1);
    Opt.BackwardStep((P * P).sum);
    Check(MaxAbsDiff(P, TTensor.FromArray([0.9, -1.9])) < 1e-6,
      'Adam one step: [0.9, -1.9], got ' + Values(P));
    Opt.SetMomentum(0.8);
    Opt.SetWeightDecay(0.0);
    Opt.SetWeightDecayGroup(0, 0.0);
    Opt.SetLearningRateGroup(0, 0.05);
    Check(True, 'Adam setters raise nothing');
  finally
    FreeAndNil(Opt);
    VS.Free;
  end;

  VS := TVarStore.Create(TDevice.Cpu);
  try
    P := VS.Root.VarCopy('p', TTensor.FromArray([1.0, -2.0]));
    Opt := TOptimizer.AdamW(VS, 0.1);
    Opt.BackwardStep((P * P).sum);
    { AdamW decays first: p * (1 - lr * 0.01), then lr * sign(g) }
    Check(MaxAbsDiff(P, TTensor.FromArray([0.999 - 0.1, -1.998 + 0.1])) < 1e-6,
      'AdamW one step with default weight decay 0.01: ' + Values(P));
    FreeAndNil(Opt);
    Opt := TOptimizer.RmsProp(VS, 0.01);
    Opt.BackwardStep((P * P).sum);
    Opt.SetMomentum(0.5);
    Check(True, 'RmsProp step and SetMomentum');
    FreeAndNil(Opt);
    Opt := TOptimizer.Sgd(VS, 0.1, 0.9);
    Opt.SetMomentumGroup(0, 0.5);
    Check(True, 'SGD SetMomentumGroup');
  finally
    Opt.Free;
    VS.Free;
  end;
end;

{ H16: unpatched libtch raises "unexpected optimizer" here }
procedure TestMomentumGroup;
var
  VS: TVarStore;
  Opt: TOptimizer;
begin
  WriteLn('optimizer: SetMomentumGroup on Adam (H16)');
  VS := TVarStore.Create(TDevice.Cpu);
  Opt := nil;
  try
    VS.Root.Zeros('p', [2]);
    Opt := TOptimizer.Adam(VS, 0.1);
    try
      Opt.SetMomentumGroup(0, 0.8);
      Check(True, 'SetMomentumGroup(0, 0.8) on Adam raises nothing');
    except
      on E: ETorchNativeError do
        Check(False, 'SetMomentumGroup on Adam raised: ' + E.Message);
    end;
  finally
    Opt.Free;
    VS.Free;
  end;
end;

{ 5. Gradient clipping (tch-rs gradient_clip_test) }
procedure TestClip;
var
  VS: TVarStore;
  Opt: TOptimizer;
  V1, V2, V3: TTensor;

  function Loss: TTensor;
  begin
    Result := TTensor.cat([V1, V2 * 2.0, V3 * 2.0], 0);
    Result := (Result * Result).sum;
  end;

begin
  WriteLn('optimizer: ClipGradValue, ClipGradNorm');
  VS := TVarStore.Create(TDevice.Cpu);
  Opt := nil;
  try
    V1 := VS.Root.VarCopy('v1', TTensor.FromArray([1.0, 1.0]));
    V2 := VS.Root.VarCopy('v2', TTensor.FromArray([1.0]));
    V3 := VS.Root.VarCopy('v3', TTensor.FromArray([-1.0, -1.0]));
    Opt := TOptimizer.Sgd(VS, 1);
    Opt.ZeroGrad;
    Loss.Backward;
    Check((Values(V1.grad) = '2 2') and (Values(V2.grad) = '8') and
      (Values(V3.grad) = '-8 -8'), 'gradients [2, 2], [8], [-8, -8]');
    Opt.ClipGradValue(4);
    Check((Values(V1.grad) = '2 2') and (Values(V2.grad) = '4') and
      (Values(V3.grad) = '-4 -4'), 'ClipGradValue(4): [2, 2], [4], [-4, -4]');
    Opt.ZeroGrad;
    Loss.Backward;
    Opt.ClipGradNorm(1);
    Check((Values(V1.grad) = '0.1414 0.1414') and (Values(V2.grad) = '0.5657') and
      (Values(V3.grad) = '-0.5657 -0.5657'),
      'ClipGradNorm(1): ' + Values(V1.grad) + ' / ' + Values(V2.grad) + ' / ' + Values(V3.grad));
    Opt.ZeroGrad;
    Opt.ClipGradNorm(1);
    Check(True, 'ClipGradNorm with no gradients exits quietly');
    { BackwardStepClip: v2 moves by lr * 4 instead of 8 }
    Opt.BackwardStepClip(Loss, 4);
    Check(Near(V2.DoubleValue([0]), 1 - 4, 1e-6), 'BackwardStepClip(4) steps the clipped grad');
  finally
    Opt.Free;
    VS.Free;
  end;
end;

{ 6. Lazy add and groups }
procedure TestLazyAdd;
var
  VS: TVarStore;
  Opt: TOptimizer;
  A, B, C: TTensor;
begin
  WriteLn('optimizer: lazy add, groups');
  VS := TVarStore.Create(TDevice.Cpu);
  Opt := nil;
  try
    A := VS.Root.Ones('a', [1]);
    Opt := TOptimizer.Sgd(VS, 0.5);
    B := VS.Root.Ones('b', [1]);
    C := VS.Root.SetGroup(1).Ones('c', [1]);
    Check(Length(Opt.TrainableVariables) = 3, 'TrainableVariables sees later variables');
    Opt.ZeroGrad;                         // adds b and c (group 1)
    Opt.SetLearningRateGroup(1, 0);
    Opt.BackwardStep((A + B + C).sum);
    Check(Values(A) = '0.5', 'a (at construction) moves');
    Check(Values(B) = '0.5', 'b (added after construction) moves');
    Check(Values(C) = '1', 'c in group 1 with lr 0 stays');
  finally
    Opt.Free;
    VS.Free;
  end;
end;

{ Store layout of tch-rs save_and_load_var_store; U = t1, V = a.b.t2 }
procedure AddLayout(VS: TVarStore; out U, V: TTensor);
var
  R: TVarPath;
begin
  R := VS.Root;
  V := R.Sub('a').Sub('b').Ones('t2', [3]);
  U := R.Zeros('t1', [4]);
  R.Sub('a').Sub('b').Sub('ccc').Ones('t123', [3]);
  R.Sub('a').Sub('b').Sub('ccc').Ones('t123', [3]);
end;

{ tch-rs add_partial: t123 only once }
procedure AddPartialLayout(VS: TVarStore; out U, V: TTensor);
var
  R: TVarPath;
begin
  R := VS.Root;
  V := R.Sub('a').Sub('b').Ones('t2', [3]);
  U := R.Zeros('t1', [4]);
  R.Sub('a').Sub('b').Sub('ccc').Ones('t123', [3]);
end;

{ 7. Save / Load (tch-rs save_and_load_var_store and ..._incomplete_file) }
procedure TestSaveLoad;
var
  VS1, VS2, VS3: TVarStore;
  U1, V1, U2, V2, Dummy: TTensor;
  G: IInterface;
  FileName: String;
  Missing: TStringArray;
  Raised: Boolean;
begin
  WriteLn('save/load: round trip, missing names, LoadPartial, half (H15)');
  FileName := TempFile('vs');
  VS1 := TVarStore.Create(TDevice.Cpu);
  VS2 := TVarStore.Create(TDevice.Cpu);
  VS3 := nil;
  try
    AddLayout(VS1, U1, V1);
    AddLayout(VS2, U2, V2);
    G := NoGradGuard;
    U1.add_scalar_(42);
    V1.mul_scalar_(2);
    G := nil;
    Check((MeanOf(U1) = 42) and (MeanOf(V1) = 2) and (MeanOf(U2) = 0) and
      (MeanOf(V2) = 1), 'before: 42 2 0 1');
    VS1.Save(FileName);
    VS2.Load(FileName);
    Check((MeanOf(U1) = 42) and (MeanOf(U2) = 42) and (MeanOf(V2) = 2),
      'after Load: u2 = 42, v2 = 2');
    Check(U2.RequiresGrad, 'loaded variable still requires grad');

    VS3 := TVarStore.Create(TDevice.Cpu);
    AddLayout(VS3, Dummy, Dummy);
    VS3.Root.Sub('extra').Zeros('x', [1]);
    Raised := False;
    try
      VS3.Load(FileName);
    except
      on E: ETorchError do
        Raised := Pos('extra.x', E.Message) > 0;
    end;
    Check(Raised, 'Load into a store with an extra variable names it');
    Check(MeanOf(VS3.Variables[1].Tensor) = 0, 'and leaves the store unchanged');
    Missing := VS3.LoadPartial(FileName);
    Check((Length(Missing) = 1) and (Missing[0] = 'extra.x'), 'LoadPartial returns [extra.x]');
    Check(MeanOf(VS3.Variables[1].Tensor) = 42, 'LoadPartial loads the rest');
    FreeAndNil(VS3);
    DeleteFile(FileName);

    { half precision: the variables take the file's kind (H15) }
    VS1.SetFloatKind(kHalf);
    Check(U1.Kind = kHalf, 'SetFloatKind(kHalf) converts in place');
    VS1.Save(FileName);
    VS3 := TVarStore.Create(TDevice.Cpu);
    AddLayout(VS3, U2, V2);
    VS3.Load(FileName);
    Check((U2.Kind = kHalf) and (V2.Kind = kHalf), 'float store loading a half file is kHalf (H15)');
    Check((MeanOf(U2) = 42) and (MeanOf(V2) = 2), 'with the same means');
    FreeAndNil(VS3);
    DeleteFile(FileName);
  finally
    VS3.Free;
    VS2.Free;
    VS1.Free;
  end;

  { incomplete file }
  VS1 := TVarStore.Create(TDevice.Cpu);
  VS2 := TVarStore.Create(TDevice.Cpu);
  try
    U1 := VS1.Root.Zeros('t1', [4]);
    VS1.Root.Sub('a').Sub('b').Sub('ccc').Ones('t123', [3]);
    AddPartialLayout(VS2, U2, V2);
    G := NoGradGuard;
    U1.add_scalar_(42);
    G := nil;
    VS1.Save(FileName);
    Missing := VS2.LoadPartial(FileName);
    Check((Length(Missing) = 1) and (Missing[0] = 'a.b.t2'), 'LoadPartial returns [a.b.t2]');
    Check((MeanOf(U2) = 42) and (MeanOf(V2) = 1), 'u2 = 42, v2 untouched');
    DeleteFile(FileName);

    { second half of the tch-rs test: a half file into a fresh float store;
      the loaded u2 takes the file's kind, the missing v2 keeps kFloat }
    VS1.SetFloatKind(kHalf);
    VS1.Save(FileName);
    FreeAndNil(VS2);
    VS2 := TVarStore.Create(TDevice.Cpu);
    AddPartialLayout(VS2, U2, V2);
    Missing := VS2.LoadPartial(FileName);
    Check((Length(Missing) = 1) and (Missing[0] = 'a.b.t2'), 'half: LoadPartial returns [a.b.t2]');
    Check((U2.Kind = kHalf) and (MeanOf(U2) = 42), 'half: u2 is kHalf, mean 42');
    Check((V2.Kind = kFloat) and (MeanOf(V2) = 1), 'half: missing v2 stays kFloat, mean 1');
    DeleteFile(FileName);

    Raised := False;
    try VS1.Save(ChangeFileExt(FileName, '.safetensors'));
    except on ETorchError do Raised := True; end;
    Check(Raised, 'Save to .safetensors raises');
    Raised := False;
    try VS1.Load(FileName);
    except on E: ETorchNativeError do Raised := True; end;
    Check(Raised, 'Load of a missing file raises ETorchNativeError');
  finally
    VS2.Free;
    VS1.Free;
  end;

  { CopyFrom }
  VS1 := TVarStore.Create(TDevice.Cpu);
  VS2 := TVarStore.Create(TDevice.Cpu);
  try
    AddLayout(VS1, U1, V1);
    AddLayout(VS2, U2, V2);
    G := NoGradGuard;
    U1.add_scalar_(7);
    G := nil;
    VS2.CopyFrom(VS1);
    Check(MeanOf(U2) = 7, 'CopyFrom copies by name');
    VS2.Root.Zeros('only2', [1]);
    Raised := False;
    try VS2.CopyFrom(VS1);
    except on E: ETorchError do Raised := Pos('only2', E.Message) > 0; end;
    Check(Raised, 'CopyFrom with a name missing in the source raises');
  finally
    VS2.Free;
    VS1.Free;
  end;
end;

{ 8. Layers }
procedure TestLayers;
var
  Mode: TPaddingMode;
  VS: TVarStore;
  BN: TBatchNorm;
  LN: TLayerNorm;
  Emb: TEmbedding;
  Conv: TConv2D;
  Opt: TOptimizer;
  BNCfg: TBatchNormConfig;
  EmbCfg: TEmbeddingConfig;
  Xs, Ys, Y, Idx: TTensor;
  Initial, L: Double;
  K: Integer;
  Raised: Boolean;
  NT: TNamedTensorArray;
begin
  WriteLn('layers: Conv2D padding modes, BatchNorm, LayerNorm, Embedding');
  for Mode := Low(TPaddingMode) to High(TPaddingMode) do
    Check(Values(HandConvPadded(TDevice.Cpu, Mode)) = ConvPadExpected[Mode],
      'conv ' + ConvPadModeNames[Mode] + ': ' + ConvPadExpected[Mode]);

  VS := TVarStore.Create(TDevice.Cpu);
  Conv := nil;
  try
    Conv := TConv2D.Create(VS.Root / 'c', 4, 6, 3);
    NT := VS.Variables;
    Check((NT[0].Name = 'c.bias') and (NT[1].Name = 'c.weight') and
      ShapeIs(Conv.Weight, [6, 4, 3, 3]), 'Conv2D registers bias, then weight [6, 4, 3, 3]');
    Check(ShapeIs(Conv.Forward(TTensor.ones([2, 4, 8, 8])), [2, 6, 6, 6]), 'Conv2D forward shape');
  finally
    Conv.Free;
    VS.Free;
  end;

  VS := TVarStore.Create(TDevice.Cpu);
  BN := nil;
  try
    BN := TBatchNorm.Create(VS.Root / 'bn', 1, 40);
    NT := VS.Variables;
    Check((VS.Count = 4) and (NT[2].Name = 'bn.running_mean') and
      (NT[3].Name = 'bn.running_var'), 'BatchNorm registers 4 variables');
    Check(Length(VS.TrainableVariables) = 2, 'two of them trainable');
    Y := BN.Forward(TTensor.randn([10, 40]), True);
    Check(ShapeIs(Y, [10, 40]), 'BatchNorm forward [10, 40] in train mode');
    Check(MeanOf(BN.RunningVar) <> 1, 'train mode updates the running stats');
    Raised := False;
    try BN.Forward(TTensor.randn([10, 40, 3, 3]), True);
    except on E: ETorchError do Raised := Pos('2 or 3 dims', E.Message) > 0; end;
    Check(Raised, 'BatchNorm1d on 4-d input raises');
    FreeAndNil(BN);
    BNCfg := DefaultBatchNormConfig;
    BNCfg.Affine := False;
    BN := TBatchNorm.Create(VS.Root / 'bn2', 2, 3, BNCfg);
    Check(VS.Count = 6, 'BatchNorm without affine: 2 variables');
    Check(ShapeIs(BN.Forward(TTensor.randn([2, 3, 4, 4])), [2, 3, 4, 4]), 'BatchNorm2d eval forward');
  finally
    BN.Free;
    VS.Free;
  end;

  { tch-rs layer_norm_parameters_test }
  VS := TVarStore.Create(TDevice.Cpu);
  LN := nil;
  Opt := nil;
  try
    Xs := TTensor.FromArray([42.0, 42.0, 42.0, 24.0]).view([-1, 2]);
    Ys := Xs * 0.42 + 1.337;
    Opt := TOptimizer.Sgd(VS, 1.0);
    LN := TLayerNorm.Create(VS.Root, [2]);
    Initial := LN.Forward(Xs).mse_loss(Ys).DoubleValue([]);
    Check(Initial > 1, Format('LayerNorm initial loss %.4f > 1', [Initial]));
    for K := 1 to 49 do
      Opt.BackwardStep(LN.Forward(Xs).mse_loss(Ys));
    L := LN.Forward(Xs).mse_loss(Ys).DoubleValue([]);
    Check(L < 0.25, Format('LayerNorm loss after 49 steps %.4f < 0.25', [L]));
    TInit.Constant(1).Apply(LN.Weight);
    TInit.Constant(0).Apply(LN.Bias);
    Check(LN.Forward(Xs).mse_loss(Ys).DoubleValue([]) = Initial, 'reset restores the initial loss');
  finally
    Opt.Free;
    LN.Free;
    VS.Free;
  end;

  VS := TVarStore.Create(TDevice.Cpu);
  Emb := nil;
  try
    EmbCfg := DefaultEmbeddingConfig;
    EmbCfg.PaddingIdx := 3;
    Emb := TEmbedding.Create(VS.Root, 10, 4, EmbCfg);
    Idx := TTensor.randint(10, [5, 7], kInt64, TDevice.Cpu);
    Check(ShapeIs(Emb.Forward(Idx), [5, 7, 4]), 'Embedding(10, 4) on [5, 7] gives [5, 7, 4]');
    Y := Emb.Forward(TTensor.FromArray([Int64(3)]));
    Check(MaxAbsDiff(Y.view([4]), Emb.Weight.select(0, 3)) = 0, 'PaddingIdx row equals Weight''s row');
    { padding_idx acts in the backward pass only: its row gets no gradient }
    Emb.Forward(TTensor.FromArray([Int64(3), 5])).sum.Backward;
    Check((Values(Emb.Weight.grad.select(0, 3)) = '0 0 0 0') and
      (Values(Emb.Weight.grad.select(0, 5)) = '1 1 1 1'),
      'PaddingIdx 3: its row gets no gradient, row 5 gets ones');
  finally
    Emb.Free;
    VS.Free;
  end;
end;

{ 9. Sequential }
var
  SeenTrain: Boolean;

function ReluFn(const X: TTensor): TTensor;
begin
  Result := X.relu;
end;

function RecordTrain(const X: TTensor; Train: Boolean): TTensor;
begin
  SeenTrain := Train;
  Result := X;
end;

type
  TScaler = class
    Factor: Double;
    function Scale(const X: TTensor): TTensor;
    function ScaleT(const X: TTensor; Train: Boolean): TTensor;
  end;

function TScaler.Scale(const X: TTensor): TTensor;
begin
  Result := X * Factor;
end;

function TScaler.ScaleT(const X: TTensor; Train: Boolean): TTensor;
begin
  if Train then
    Result := X * Factor
  else
    Result := X;
end;

procedure TestSequential;
var
  Seq: TSequential;
  Sc: TScaler;
  X: TTensor;
  All: TTensorArray;
  VS: TVarStore;
begin
  WriteLn('sequential: AddFn overloads, Train flag, ForwardAll');
  X := TTensor.FromArray([1.0, -1.0, 2.0]);
  Sc := TScaler.Create;
  Sc.Factor := 3;
  Seq := TSequential.Create;
  try
    Check(Seq.Forward(X).SameHandle(X), 'empty sequential returns X');
    Check(Length(Seq.ForwardAll(X)) = 1, 'empty ForwardAll returns [X]');
    Seq.AddFn(@ReluFn);
    Check(Values(Seq.Forward(X)) = '1 0 2', 'AddFn(@Relu): [1, 0, 2]');
    Seq.AddFn(@RecordTrain).AddFn(@Sc.Scale).AddFn(@Sc.ScaleT);
    Check(Seq.Count = 4, 'chained AddFn: 4 layers');
    SeenTrain := False;
    Check(Values(Seq.Forward(X, True)) = '9 0 18', 'Forward(X, True) through all four');
    Check(SeenTrain, 'TTensorFuncT sees Train = True');
    Check(Values(Seq.Forward(X)) = '3 0 6', 'Forward(X): ScaleT skipped');
    Check(not SeenTrain, 'TTensorFuncT sees Train = False');
    All := Seq.ForwardAll(X, True);
    Check((Length(All) = 4) and (Values(All[0]) = '1 0 2') and (Values(All[3]) = '9 0 18'),
      'ForwardAll returns one tensor per layer');
    Check(Length(Seq.ForwardAll(X, False, 2)) = 2, 'ForwardAll(N = 2)');
    Check(Seq[0] is TFuncModule, 'Items[0] is a TFuncModule');
  finally
    Seq.Free;
    Sc.Free;
  end;

  VS := TVarStore.Create(TDevice.Cpu);
  Seq := TSequential.Create;
  try
    Seq.Add(TLinear.Create(VS.Root / 'l1', 3, 5)).AddFn(@ReluFn)
       .Add(TLinear.Create(VS.Root / 'l2', 5, 2));
    Check(ShapeIs(Seq.Forward(TTensor.ones([4, 3])), [4, 2]), 'MLP of two TLinear: [4, 2]');
    Check(VS.Count = 4, 'MLP registers 4 variables');
  finally
    Seq.Free;
    VS.Free;
  end;
end;

{ 11. Milestone 3 layers (DESIGN.md 4.6.10) }

function SameArr(const A, B: TInt64Array): Boolean;
var
  K: Integer;
begin
  Result := Length(A) = Length(B);
  if Result then
    for K := 0 to High(A) do
      if A[K] <> B[K] then
        Exit(False);
end;

function SameInit(const A, B: TInit): Boolean;
begin
  Result := (A.Kind = B.Kind) and (A.A = B.A) and (A.B = B.B) and
    (A.Dist = B.Dist) and (A.Fan = B.Fan) and (A.NonLin = B.NonLin);
end;

function Names(VS: TVarStore): String;
var
  NT: TNamedTensorArray;
  K: Integer;
begin
  NT := VS.Variables;
  Result := '';
  for K := 0 to High(NT) do
  begin
    if K > 0 then
      Result := Result + ' ';
    Result := Result + NT[K].Name;
  end;
end;

function ConvConfigNoBias: TConvConfig;
begin
  Result := DefaultConvConfig;
  Result.Bias := False;
end;

function ApplyConv1D(Mode: TPaddingMode): String;
var
  VS: TVarStore;
  Conv: TConv1D;
  Cfg: TConvConfig;
begin
  VS := TVarStore.Create(TDevice.Cpu);
  Conv := nil;
  try
    Cfg := DefaultConvConfig;
    Cfg.Padding := 1;
    Cfg.Bias := False;
    Cfg.PaddingMode := Mode;
    Conv := TConv1D.Create(VS.Root, 1, 1, 3, Cfg);
    TInit.Constant(1).Apply(Conv.Weight);
    Result := Values(Conv.Forward(TTensor.FromArray([1.0, 2, 3]).view([1, 1, 3])));
  finally
    Conv.Free;
    VS.Free;
  end;
end;

procedure TestPadAndConvND;
var
  VS: TVarStore;
  Conv: TConvND;
  C1: TConv1D;
  C3: TConv3D;
  Cfg: TConvConfigND;
  A, B: TConvConfigND;
  Y: TTensor;
  Raised: Boolean;
begin
  WriteLn('milestone 3: PadWithMode, TConv1D, TConv2D ND, TConv3D');
  Check(Values(PadWithMode(TTensor.FromArray([1.0, 2, 3]), pmZeros, [1, 1])) = '0 1 2 3 0',
    '1. pad zeros [1, 1] (tch-rs pad)');
  Check(Values(PadWithMode(TTensor.FromArray([1.0, 2, 3, 4]).view([1, 2, 2]), pmReflect,
    [1, 1, 1, 1])) = '4 3 4 3 2 1 2 1 4 3 4 3 2 1 2 1', '1. pad reflect [1, 1, 1, 1] (tch-rs pad)');

  Check(ApplyConv1D(pmZeros) = '3 6 5', '2. conv1d zeros: 3 6 5');
  Check(ApplyConv1D(pmReflect) = '5 6 7', '2. conv1d reflect: 5 6 7');
  Check(ApplyConv1D(pmReplicate) = '4 6 8', '2. conv1d replicate: 4 6 8');
  Check(ApplyConv1D(pmCircular) = '6 6 6', '2. conv1d circular: 6 6 6');
  VS := TVarStore.Create(TDevice.Cpu);
  C1 := nil;
  try
    C1 := TConv1D.Create(VS.Root, 4, 6, 3);
    Check((Names(VS) = 'bias weight') and ShapeIs(C1.Bias, [6]) and ShapeIs(C1.Weight, [6, 4, 3]),
      '2. TConv1D(4, 6, 3) registers bias [6], then weight [6, 4, 3]');
  finally
    C1.Free;
    VS.Free;
  end;

  VS := TVarStore.Create(TDevice.Cpu);
  Conv := nil;
  try
    Cfg := DefaultConvConfigND(2);
    Cfg.Padding := [1, 0];
    Cfg.Bias := False;
    Cfg.PaddingMode := pmReflect;
    Conv := TConv2D.Create(VS.Root, 1, 1, [3, 1], Cfg);
    TInit.Constant(1).Apply(Conv.Weight);
    Y := Conv.Forward(TTensor.FromArray([1.0, 2, 3, 4]).view([1, 1, 2, 2]));
    Check(ShapeIs(Y, [1, 1, 2, 2]) and (Values(Y) = '7 10 5 8'),
      '3. ND padding [1, 0] reflect, KSizes [3, 1]: [[7, 10], [5, 8]] (PyTorch order)');
  finally
    Conv.Free;
    VS.Free;
  end;
  A := DefaultConvConfigND(2);
  B := ConvConfigToND(DefaultConvConfig, 2);
  Check(SameArr(A.Stride, B.Stride) and SameArr(A.Padding, B.Padding) and
    SameArr(A.Dilation, B.Dilation) and SameArr(A.Stride, [1, 1]) and
    SameArr(A.Padding, [0, 0]) and (A.Groups = B.Groups) and (A.Bias = B.Bias) and
    SameInit(A.WsInit, B.WsInit) and SameInit(A.BsInit, B.BsInit) and
    (A.PaddingMode = B.PaddingMode), '3. DefaultConvConfigND(2) = ConvConfigToND(DefaultConvConfig, 2)');
  VS := TVarStore.Create(TDevice.Cpu);
  try
    Raised := False;
    try
      TConvND.Create(VS.Root, 1, 1, [3, 3, 3, 3], DefaultConvConfigND(4)).Free;
    except
      on ETorchError do Raised := True;
    end;
    Check(Raised, '3. four kernel sizes raise ETorchError');
    Raised := False;
    Cfg := DefaultConvConfigND(2);
    Cfg.Stride := [1];
    try
      TConvND.Create(VS.Root, 1, 1, [3, 3], Cfg).Free;
    except
      on ETorchError do Raised := True;
    end;
    Check(Raised and (VS.Count = 0), '3. Stride of the wrong length raises before registering');
  finally
    VS.Free;
  end;

  VS := TVarStore.Create(TDevice.Cpu);
  C3 := nil;
  Conv := nil;
  try
    Cfg := DefaultConvConfigND(3);
    C3 := TConv3D.Create(VS.Root / 'a', 1, 1, 2, ConvConfigNoBias);
    TInit.Constant(1).Apply(C3.Weight);
    Y := C3.Forward(TTensor.FromArray([1.0, 2, 3, 4, 5, 6, 7, 8]).view([1, 1, 2, 2, 2]));
    Check(ShapeIs(Y, [1, 1, 1, 1, 1]) and (Values(Y) = '36'), '4. TConv3D(1, 1, 2) on 1..8: [36]');
    Cfg.Padding := [1, 0, 0];
    Conv := TConvND.Create(VS.Root / 'b', 2, 3, [3, 3, 3], Cfg);
    Check(ShapeIs(Conv.Weight, [3, 2, 3, 3, 3]) and (Conv.Nd = 3) and
      ShapeIs(Conv.Forward(TTensor.randn([1, 2, 4, 4, 4])), [1, 3, 4, 2, 2]),
      '4. TConvND(2, 3, [3, 3, 3]), Padding [1, 0, 0]: [1, 3, 4, 2, 2]');
  finally
    Conv.Free;
    C3.Free;
    VS.Free;
  end;
end;

function TransposeValues(Nd, Stride, OutputPadding: Int64; const X: TTensor): TTensor;
var
  VS: TVarStore;
  L: TConvTransposeND;
  Cfg: TConvTransposeConfig;
begin
  VS := TVarStore.Create(TDevice.Cpu);
  L := nil;
  try
    Cfg := DefaultConvTransposeConfig;
    Cfg.Bias := False;
    Cfg.Stride := Stride;
    Cfg.OutputPadding := OutputPadding;
    if Nd = 1 then
      L := TConvTranspose1D.Create(VS.Root, 1, 1, 2, Cfg)
    else
      L := TConvTranspose2D.Create(VS.Root, 1, 1, 2, Cfg);
    TInit.Constant(1).Apply(L.Weight);
    Result := L.Forward(X);
  finally
    L.Free;
    VS.Free;
  end;
end;

procedure TestConvTransposeAndGroupNorm;
var
  VS: TVarStore;
  T3: TConvTranspose3D;
  GN: TGroupNorm;
  GCfg: TGroupNormConfig;
  X2, X1, Y: TTensor;
  D: TDoubleArray;
begin
  WriteLn('milestone 3: TConvTranspose1D/2D/3D, TGroupNorm');
  X2 := TTensor.FromArray([1.0, 2, 3, 4]).view([1, 1, 2, 2]);
  Y := TransposeValues(2, 1, 0, X2);
  Check(ShapeIs(Y, [1, 1, 3, 3]) and (Values(Y) = '1 3 2 4 10 6 3 7 4'),
    '5. TConvTranspose2D stride 1: [1 3 2 4 10 6 3 7 4]');
  Y := TransposeValues(2, 2, 0, X2);
  Check(ShapeIs(Y, [1, 1, 4, 4]) and (Values(Y) = '1 1 2 2 1 1 2 2 3 3 4 4 3 3 4 4'),
    '5. TConvTranspose2D stride 2: [1 1 2 2 1 1 2 2 3 3 4 4 3 3 4 4]');
  X1 := TTensor.FromArray([1.0, 2, 3]).view([1, 1, 3]);
  Check(Values(TransposeValues(1, 1, 0, X1)) = '1 3 5 3', '5. TConvTranspose1D: [1 3 5 3]');
  Y := TransposeValues(1, 2, 1, X1);
  Check(ShapeIs(Y, [1, 1, 7]) and (Values(Y) = '1 1 2 2 3 3 0'),
    '5. TConvTranspose1D stride 2, OutputPadding 1: [1 1 2 2 3 3 0]');
  VS := TVarStore.Create(TDevice.Cpu);
  T3 := nil;
  try
    T3 := TConvTranspose3D.Create(VS.Root, 2, 3, 3);
    Check((Names(VS) = 'bias weight') and ShapeIs(T3.Bias, [3]) and
      ShapeIs(T3.Weight, [2, 3, 3, 3, 3]), '5. TConvTranspose3D(2, 3, 3): bias [3], then weight [2, 3, 3, 3, 3]');
    Check(ShapeIs(T3.Forward(TTensor.randn([1, 2, 4, 4, 4])), [1, 3, 6, 6, 6]),
      '5. TConvTranspose3D: [1, 2, 4, 4, 4] -> [1, 3, 6, 6, 6]');
  finally
    T3.Free;
    VS.Free;
  end;

  VS := TVarStore.Create(TDevice.Cpu);
  GN := nil;
  try
    GN := TGroupNorm.Create(VS.Root, 2, 4);
    Check((Names(VS) = 'weight bias') and ShapeIs(GN.Weight, [4]) and ShapeIs(GN.Bias, [4]),
      '6. TGroupNorm(2, 4) registers weight [4], bias [4]');
    D := GN.Forward(TTensor.FromArray([1.0, 2, 3, 4]).view([1, 4, 1])).ToDoubleArray;
    Check(Near(D[0], -1, 1e-4) and Near(D[1], 1, 1e-4) and Near(D[2], -1, 1e-4) and
      Near(D[3], 1, 1e-4), Format('6. GroupNorm(2, 4) on [1, 2, 3, 4]: %.5f %.5f %.5f %.5f',
      [D[0], D[1], D[2], D[3]]));
    FreeAndNil(GN);
    GCfg := DefaultGroupNormConfig;
    GCfg.Affine := False;
    GN := TGroupNorm.Create(VS.Root / 'na', 2, 4, GCfg);
    D := GN.Forward(TTensor.FromArray([1.0, 2, 3, 4]).view([1, 4, 1])).ToDoubleArray;
    Check((VS.Count = 2) and GN.Weight.IsNil and Near(D[0], -1, 1e-4) and Near(D[3], 1, 1e-4),
      '6. Affine False registers nothing and still normalises');
    FreeAndNil(GN);
    GN := TGroupNorm.Create(VS.Root / 't', 2, 10);
    Check(ShapeIs(GN.Forward(TTensor.randn([1, 10, 7])), [1, 10, 7]),
      '6. tch-rs group_norm_test: GroupNorm(2, 10) keeps [1, 10, 7]');
  finally
    GN.Free;
    VS.Free;
  end;
end;

{ tch-rs gru_test / lstm_test configs: one layer type, one config }
procedure CheckRNN(IsLSTM: Boolean; const Cfg: TRNNConfig; const What: String);
var
  VS: TVarStore;
  R: TRNN;
  S: TRNNState;
  Y: TTensor;
  Dirs, LayerDim, NVars: Int64;
begin
  Dirs := 1 + Ord(Cfg.Bidirectional);
  LayerDim := Cfg.NumLayers * Dirs;
  if Cfg.HasBiases then
    NVars := 4 * LayerDim
  else
    NVars := 2 * LayerDim;
  VS := TVarStore.Create(TDevice.Cpu);
  R := nil;
  try
    if IsLSTM then
      R := TLSTM.Create(VS.Root, 2, 4, Cfg)
    else
      R := TGRU.Create(VS.Root, 2, 4, Cfg);
    S := R.Step(TTensor.randn([5, 2]), R.ZeroState(5));
    Y := R.Seq(TTensor.randn([5, 3, 2]), S);
    Check((VS.Count = NVars) and (Length(R.FlatWeights) = NVars) and
      ShapeIs(R.Step(TTensor.randn([5, 2]), R.ZeroState(5)).H, [LayerDim, 5, 4]) and
      (IsLSTM = not R.ZeroState(5).C.IsNil) and ShapeIs(S.H, [LayerDim, 5, 4]) and
      (not IsLSTM or ShapeIs(S.C, [LayerDim, 5, 4])) and (IsLSTM or S.C.IsNil) and
      ShapeIs(Y, [5, 3, 4 * Dirs]), Format('7. %s: %d variables, Step [%d, 5, 4], Seq [5, 3, %d]',
      [What, NVars, LayerDim, 4 * Dirs]));
  finally
    R.Free;
    VS.Free;
  end;
end;

procedure TestRNN;
const
  Kinds: array[Boolean] of String = ('GRU', 'LSTM');
var
  IsLSTM: Boolean;
  Cfg: TRNNConfig;
  VS: TVarStore;
  R: TRNN;
  G: TGRU;
  L: TLSTM;
  S, S2: TRNNState;
  Y: TTensor;
  Raised: Boolean;
  Msg, Expected: String;
begin
  WriteLn('milestone 3: TGRU, TLSTM (tch-rs gru_test, lstm_test), hand numbers');
  for IsLSTM := False to True do
  begin
    Cfg := DefaultRNNConfig;
    CheckRNN(IsLSTM, Cfg, Kinds[IsLSTM] + ' default');
    Cfg.HasBiases := False;
    CheckRNN(IsLSTM, Cfg, Kinds[IsLSTM] + ' no biases');
    Cfg := DefaultRNNConfig;
    Cfg.Bidirectional := True;
    CheckRNN(IsLSTM, Cfg, Kinds[IsLSTM] + ' bidirectional');
    Cfg := DefaultRNNConfig;
    Cfg.NumLayers := 2;
    CheckRNN(IsLSTM, Cfg, Kinds[IsLSTM] + ' 2 layers');
    Cfg.Bidirectional := True;
    CheckRNN(IsLSTM, Cfg, Kinds[IsLSTM] + ' 2 layers bidirectional');
  end;

  Expected := 'weight_ih_l0 weight_hh_l0 bias_ih_l0 bias_hh_l0 weight_ih_l0_reverse ' +
    'weight_hh_l0_reverse bias_ih_l0_reverse bias_hh_l0_reverse weight_ih_l1 ' +
    'weight_hh_l1 bias_ih_l1 bias_hh_l1 weight_ih_l1_reverse weight_hh_l1_reverse ' +
    'bias_ih_l1_reverse bias_hh_l1_reverse';
  VS := TVarStore.Create(TDevice.Cpu);
  R := nil;
  try
    R := TGRU.Create(VS.Root, 2, 4, Cfg);
    Check((Names(VS) = Expected) and ShapeIs(R.FlatWeights[8], [12, 8]) and
      ShapeIs(R.FlatWeights[0], [12, 2]) and ShapeIs(R.FlatWeights[1], [12, 4]),
      '7. GRU 2 layers bidirectional: names in order, weight_ih_l1 [12, 8]');
    FreeAndNil(R);
    VS.Free;
    VS := TVarStore.Create(TDevice.Cpu);
    R := TLSTM.Create(VS.Root, 2, 4, Cfg);
    Check((Names(VS) = Expected) and ShapeIs(R.FlatWeights[8], [16, 8]),
      '7. LSTM 2 layers bidirectional: same names, weight_ih_l1 [16, 8]');
    FreeAndNil(R);
    Cfg := DefaultRNNConfig;
    Cfg.BatchFirst := False;
    R := TGRU.Create(VS.Root / 'bf', 2, 4, Cfg);
    Y := R.Seq(TTensor.randn([3, 5, 2]), S);
    Check(ShapeIs(Y, [3, 5, 4]) and ShapeIs(S.H, [1, 5, 4]) and
      ShapeIs(R.Step(TTensor.randn([5, 2]), R.ZeroState(5)).H, [1, 5, 4]),
      '7. BatchFirst False: Seq on [3, 5, 2] -> [3, 5, 4], h [1, 5, 4]; Step batch 5');
  finally
    R.Free;
    VS.Free;
  end;

  VS := TVarStore.Create(TDevice.Cpu);
  G := nil;
  L := nil;
  try
    G := TGRU.Create(VS.Root / 'g', 1, 1);
    L := TLSTM.Create(VS.Root / 'l', 1, 1);
    SetHandRNNWeights(G, HandGruBias);
    SetHandRNNWeights(L, HandLstmBias);
    S := G.Step(TTensor.zeros([1, 1]), G.ZeroState(1));
    Check(Near(S.H.DoubleValue([0, 0, 0]), HandGruH, 1e-5),
      Format('8. GRU hand number: H = %.6f (tanh(1) / 2 = 0.380797)', [S.H.DoubleValue([0, 0, 0])]));
    S := L.Step(TTensor.zeros([1, 1]), L.ZeroState(1));
    Check(Near(S.C.DoubleValue([0, 0, 0]), HandLstmC, 1e-5) and
      Near(S.H.DoubleValue([0, 0, 0]), HandLstmH, 1e-5),
      Format('8. LSTM hand numbers: C = %.6f (0.380797), H = %.6f (0.181700)',
      [S.C.DoubleValue([0, 0, 0]), S.H.DoubleValue([0, 0, 0])]));

    S := G.ZeroState(1);
    Raised := False;
    Msg := '';
    try
      G.SeqInit(TTensor.zeros([1, 1, 1]), S, S);
    except
      on E: ETorchError do
      begin
        Raised := True;
        Msg := E.Message;
      end;
    end;
    Check(Raised and (Pos('hx', Msg) > 0), '9. SeqInit(X, S, S) raises: ' + Msg);
    S := G.ZeroState(1);
    S := G.Step(TTensor.zeros([1, 1]), S);
    S2 := S;
    S := G.Step(TTensor.zeros([1, 1]), S);
    Check(Near(S2.H.DoubleValue([0, 0, 0]), 0.380797, 1e-5) and
      (S.H.DoubleValue([0, 0, 0]) > S2.H.DoubleValue([0, 0, 0])),
      '9. S := Step(X, S) works (second step builds on the first)');
  finally
    L.Free;
    G.Free;
    VS.Free;
  end;
end;

procedure TestIdentity;
var
  Id: TIdentity;
  Seq: TSequential;
  X: TTensor;
begin
  WriteLn('milestone 3: TIdentity');
  X := TTensor.FromArray([1.0, -1.0, 2.0]);
  Id := TIdentity.Create;
  try
    Check(Id.Forward(X).Raw = X.Raw, '10. TIdentity.Forward(X) is X (same handle)');
  finally
    Id.Free;
  end;
  Seq := TSequential.Create;
  try
    Seq.Add(TIdentity.Create).AddFn(@ReluFn);
    Check(Values(Seq.Forward(X)) = '1 0 2', '10. TIdentity + ReLU behaves like ReLU');
  finally
    Seq.Free;
  end;
end;

procedure TestMilestone3;
var
  Before: Integer;
begin
  Before := TchLiveHandles;
  TestPadAndConvND;
  TestConvTransposeAndGroupNorm;
  TestRNN;
  TestIdentity;
  Check(TchLiveHandles = Before, Format('11. live handles %d -> %d after the milestone 3 layers',
    [Before, TchLiveHandles]));
end;

{ 12. API not covered above (overall review, C17) }
procedure TestMoreApi;
var
  VS: TVarStore;
  P: TVarPath;
  W, T, Q: TTensor;
  Opt: TOptimizer;
  G: TGRU;
  RCfg: TRNNConfig;
  A, B: TConvTransposeConfigND;
  LCfg: TLayerNormConfig;
  Std, M: Double;
  D: TDoubleArray;
begin
  WriteLn('more API: TVarPath helpers, inits, ClipNorm, configs, TRNN properties');
  Check(Near(NonLinearityGain(nlReLU), Sqrt(2), 1e-12) and Near(NonLinearityGain(nlTanh), Double(5) / 3, 1e-12) and
    (NonLinearityGain(nlSELU) = 0.75) and (NonLinearityGain(nlLinear) = 1) and
    (NonLinearityGain(nlSigmoid) = 1) and (NonLinearityGain(nlExplicitGain, 2.5) = 2.5),
    'NonLinearityGain: ReLU sqrt 2, Tanh 5/3, SELU 3/4, Linear and Sigmoid 1, explicit');
  ManualSeed(11);
  T := TInit.KaimingNormal.Make([256, 1024], TDevice.Cpu, kFloat);
  Std := T.std(True).DoubleValue([]);
  Check(Abs(Std / Sqrt(2 / 1024) - 1) < 0.05,
    Format('TInit.KaimingNormal [256, 1024]: std %.4f ~ sqrt(2 / fan_in) = %.4f', [Std, Sqrt(2 / 1024)]));

  VS := TVarStore.Create(TDevice.Cpu);
  Opt := nil;
  G := nil;
  try
    P := VS.Root / 'b';
    Check((P.FullName('a') = 'b.a') and (VS.Root.FullName('a') = 'a') and
      (P.Device.DeviceType = dtCpu) and (P.Kind = kFloat) and (P.Group = 0) and
      (P.SetGroup(2).Group = 2), 'TVarPath: FullName, Device, Kind, Group, SetGroup');
    W := P.OnesNoTrain('o', [2]);
    Check((Values(W) = '1 1') and not W.RequiresGrad and (Length(VS.TrainableVariables) = 0),
      'OnesNoTrain: ones, not trainable');
    T := TTensor.FromArray([5.0, 6.0]);
    W := P.Add('added', T, False);
    Check(W.SameHandle(T) and VS.TryGet('b.added', W) and not W.RequiresGrad,
      'Add(T, Trainable = False): same handle, registered, no grad');
    W := P.Variable('v', [3], TInit.Constant(7));
    Check((Values(W) = '7 7 7') and W.RequiresGrad, 'Variable(Constant(7)): trainable sevens');
    W := P.RandnStandard('rs', [4, 5]);
    Check(ShapeIs(W, [4, 5]) and W.RequiresGrad, 'RandnStandard: shape [4, 5], trainable');
    W := P.Randn('rn', [20000], 3, 0.5);
    M := MeanOf(W);
    Std := W.std(True).DoubleValue([]);
    Check(Near(M, 3, 0.02) and Near(Std, 0.5, 0.02), Format('Randn(3, 0.5): mean %.3f, std %.3f', [M, Std]));
    W := P.KaimingNormal('kn', [256, 1024]);
    Check(W.RequiresGrad and (Abs(W.std(True).DoubleValue([]) / Sqrt(2 / 1024) - 1) < 0.05),
      'TVarPath.KaimingNormal: trainable, std ~ sqrt(2 / fan_in)');
    W := P.Orthogonal('or', [4, 4], 2);
    Q := W.matmul(W.t);
    D := (Q - TTensor.eye(4, kFloat, TDevice.Cpu) * 4).abs.max.ToDoubleArray;
    Check(D[0] < 1e-4, 'Orthogonal(gain 2) [4, 4]: W Wt = 4 I');

    { clip_grad_norm: grad [3, 4] has norm 5, clipped to norm 1, SGD lr 1 }
    W := (VS.Root / 'clip').Zeros('w', [2]);
    Opt := TOptimizer.Sgd(VS, 1.0);
    Opt.ZeroGrad;
    Opt.BackwardStepClipNorm((W * TTensor.FromArray([3.0, 4.0])).sum, 1.0);
    D := W.ToDoubleArray;
    Check(Near(D[0], -0.6, 1e-5) and Near(D[1], -0.8, 1e-5),
      Format('BackwardStepClipNorm(1): w = [%.4f, %.4f] (-0.6, -0.8)', [D[0], D[1]]));

    RCfg := DefaultRNNConfig;
    RCfg.NumLayers := 2;
    G := TGRU.Create(VS.Root / 'g', 2, 5, RCfg);
    Check((G.HiddenDim = 5) and (G.Config.NumLayers = 2) and G.Config.BatchFirst,
      'TRNN.HiddenDim and Config');
  finally
    G.Free;
    Opt.Free;
    VS.Free;
  end;

  A := DefaultConvTransposeConfigND(3);
  B := ConvTransposeConfigToND(DefaultConvTransposeConfig, 3);
  Check(SameArr(A.Stride, [1, 1, 1]) and SameArr(A.Padding, [0, 0, 0]) and
    SameArr(A.OutputPadding, [0, 0, 0]) and SameArr(A.Dilation, [1, 1, 1]) and
    SameArr(A.Stride, B.Stride) and SameArr(A.OutputPadding, B.OutputPadding) and
    (A.Groups = 1) and A.Bias and SameInit(A.WsInit, B.WsInit) and SameInit(A.BsInit, B.BsInit),
    'DefaultConvTransposeConfigND(3) = ConvTransposeConfigToND(default, 3)');
  LCfg := DefaultLayerNormConfig;
  Check(LCfg.CudnnEnabled and (LCfg.Eps = Double(1e-5)) and LCfg.ElementwiseAffine and
    SameInit(LCfg.WsInit, TInit.Constant(1)) and SameInit(LCfg.BsInit, TInit.Constant(0)),
    'DefaultLayerNormConfig: cuDNN, 1e-5, affine, Constant(1), Constant(0)');
end;

{ 10. Handles: optimizer, layers, store freed in that order }
procedure Scenario;
var
  VS: TVarStore;
  Seq: TSequential;
  Opt: TOptimizer;
  X: TTensor;
  K: Integer;
begin
  VS := TVarStore.Create(TDevice.Cpu);
  Seq := TSequential.Create;
  Seq.Add(TLinear.Create(VS.Root / 'a', 4, 8)).AddFn(@ReluFn)
     .Add(TLinear.Create(VS.Root / 'b', 8, 1));
  Opt := TOptimizer.Adam(VS, 1e-3);
  X := TTensor.randn([16, 4]);
  for K := 1 to 5 do
    Opt.BackwardStep(Seq.Forward(X, True).mean);
  Opt.Free;
  Seq.Free;
  VS.Free;
end;

procedure TestHandles;
var
  Before: Integer;
begin
  WriteLn('handles: back to the start after freeing everything');
  Before := TchLiveHandles;
  Scenario;
  Check(TchLiveHandles = Before, Format('live handles %d -> %d', [Before, TchLiveHandles]));
end;

begin
  if (ParamCount > 0) and (ParamStr(1) = 'h16') then
    TestMomentumGroup
  else
  begin
    TestInit;
    TestApplyGuard;
    TestStore;
    TestStoreEdges;
    TestRollbackHandles;
    TestFreezeAndSetKind;
    TestLinearRegression;
    TestAdam;
    TestMomentumGroup;
    TestClip;
    TestLazyAdd;
    TestSaveLoad;
    TestLayers;
    TestSequential;
    TestMilestone3;
    TestMoreApi;
    TestHandles;
  end;
  Check(TchLiveHandles = 0, 'no live handles at the end');
  if Failures = 0 then
    WriteLn('OK')
  else
  begin
    WriteLn(Failures, ' FAILED');
    Halt(1);
  end;
end.
