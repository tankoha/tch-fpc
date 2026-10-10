{ Milestone 2 (DESIGN.md 4.6.7) against the real libtorch: TInit, TVarStore,
  save/load, optimizers, layers, TSequential, and hypotheses H13-H18.
  Expected values are worked out by hand in DESIGN.md 4.6.7 or follow the
  named tch-rs 4227b89b test.

  Build (from the repo root, after building build/libtch):
    fpc -Mobjfpc -Sh -Fusrc -FUbuild/nn -Flbuild/libtch \
      -k-rpath=$PWD/build/libtch tests/nn/nn_test.pas -obuild/nn_test
    build/nn_test           # all checks
    build/nn_test h16       # SetMomentumGroup on Adam only; run once more
                            # with LD_LIBRARY_PATH=build/libtch-unpatched }
program nn_test;

uses
  SysUtils, Tch, TchNN;

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
  Result := TTensor.FromArray(A).view([14, 1]);
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
    Check(MaxAbsDiff(P, TTensor.FromArray([Double(0.9), -1.9])) < 1e-6,
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
    Check(MaxAbsDiff(P, TTensor.FromArray([Double(0.999 - 0.1), -1.998 + 0.1])) < 1e-6,
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

function ApplyConv(const Xs: TTensor; Mode: TPaddingMode): String;
var
  VS: TVarStore;
  Conv: TConv2D;
  Cfg: TConvConfig;
begin
  VS := TVarStore.Create(TDevice.Cpu);
  Conv := nil;
  try
    Cfg := DefaultConvConfig;
    Cfg.Padding := 1;
    Cfg.Bias := False;
    Cfg.PaddingMode := Mode;
    Conv := TConv2D.Create(VS.Root, 1, 1, 3, Cfg);
    TInit.Constant(1).Apply(Conv.Weight);
    Result := Values(Conv.Forward(Xs).reshape([-1]));
  finally
    Conv.Free;
    VS.Free;
  end;
end;

{ 8. Layers }
procedure TestLayers;
var
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
  Xs := TTensor.FromArray([1.0, 2.0, 3.0, 4.0]).view([1, 1, 2, 2]);
  Check(ApplyConv(Xs, pmZeros) = '10 10 10 10', 'conv zeros: 10 10 10 10');
  Check(ApplyConv(Xs, pmReflect) = '27 24 21 18', 'conv reflect: 27 24 21 18');
  Check(ApplyConv(Xs, pmCircular) = '27 24 21 18', 'conv circular: 27 24 21 18');
  Check(ApplyConv(Xs, pmReplicate) = '18 21 24 27', 'conv replicate: 18 21 24 27');

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
    TestFreezeAndSetKind;
    TestLinearRegression;
    TestAdam;
    TestMomentumGroup;
    TestClip;
    TestLazyAdd;
    TestSaveLoad;
    TestLayers;
    TestSequential;
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
