{ ITa driver of T14 (harness owner's): TLoraAdamW against tch-fpc's
  TOptimizer.AdamW (libtorch) on the toy problem of
  tests/ut/LoraOptimFixture.pas. Writes one line per case to stdout: id,
  value, limit, note (tab separated; OK when value <= limit). With the
  argument "gpu" it runs the GPU cases instead (harness owner only;
  agents run with the GPUs hidden). }
program loraoptim_check;

{$mode objfpc}{$H+}

uses
  SysUtils, Math, Tch, TchNN, LoraOptim, LoraOptimFixture;

procedure Emit(const Id: string; Value, Limit: Double; const Note: string);
begin
  WriteLn(Id, #9, Format('%.2E', [Value]), #9, Format('%.2E', [Limit]), #9, Note);
end;

{ Ours on MakeToyStore(StoreSeed) moved to Dev; returns max diff of w and
  b against ReferenceRun on the CPU. }
function Parity(StoreSeed, DataSeed: Int64; AKind: TKind; Steps: Integer;
  LR, Clip: Double; const Dev: TDevice): Double;
var
  S: TVarStore;
  O: TLoraAdamW;
  W, B, X, Y, RW, RB: TTensor;
  K: Integer;
begin
  S := MakeToyStore(StoreSeed, AKind, W, B);
  S.SetDevice(Dev);
  MakeToyData(DataSeed, AKind, X, Y);
  X := X.&to(Dev);
  Y := Y.&to(Dev);
  O := TLoraAdamW.Create([S], DefaultAdamWConfig);
  try
    for K := 1 to Steps do
    begin
      O.ZeroGrad;
      ToyLoss(W, B, X, Y).Backward;
      if Clip > 0 then
        O.ClipGlobalNorm(Clip);
      O.Step(LR);
    end;
    ReferenceRun(StoreSeed, DataSeed, AKind, Steps, LR, 0.01, Clip, RW, RB);
    Result := Max(MaxAbsDiff(W, RW), MaxAbsDiff(B, RB));
  finally
    O.Free;
    S.Free;
  end;
end;

{ One store (w, b) on the CPU against two stores (w on Dev1, b on Dev2),
  clipping at 0.5, 50 updates. }
function TwoStores(AKind: TKind; const Dev1, Dev2: TDevice): Double;
var
  SU, S1, S2: TVarStore;
  OU, O2: TLoraAdamW;
  W, B, W2, B2, X, Y: TTensor;
  K: Integer;
begin
  SU := MakeToyStore(21, AKind, W, B);
  ManualSeed(21);
  S1 := TVarStore.Create(TDevice.Cpu, AKind);
  W2 := S1.Root.RandnStandard('w', [4, 3]);
  S2 := TVarStore.Create(TDevice.Cpu, AKind);
  B2 := S2.Root.Zeros('b', [3]);
  S1.SetDevice(Dev1);
  S2.SetDevice(Dev2);
  MakeToyData(22, AKind, X, Y);
  OU := TLoraAdamW.Create([SU], DefaultAdamWConfig);
  O2 := TLoraAdamW.Create([S1, S2], DefaultAdamWConfig);
  try
    for K := 1 to 50 do
    begin
      OU.ZeroGrad;
      ToyLoss(W, B, X, Y).Backward;
      OU.ClipGlobalNorm(0.5);
      OU.Step(0.05);
      O2.ZeroGrad;
      ToyLoss(W2, B2.&to(Dev1), X.&to(Dev1), Y.&to(Dev1)).Backward;
      O2.ClipGlobalNorm(0.5);
      O2.Step(0.05);
    end;
    Result := Max(MaxAbsDiff(W, W2), MaxAbsDiff(B, B2));
  finally
    OU.Free;
    O2.Free;
    SU.Free;
    S1.Free;
    S2.Free;
  end;
end;

{ 100 updates with ScheduleLR(0.05, u, 10, 100, 0.1) on both sides
  (the reference sets it with SetLearningRate). }
function Scheduled: Double;
var
  S, SR: TVarStore;
  O: TLoraAdamW;
  Ref: TOptimizer;
  W, B, RW, RB, X, Y: TTensor;
  K: Integer;
begin
  S := MakeToyStore(31, kFloat, W, B);
  SR := MakeToyStore(31, kFloat, RW, RB);
  MakeToyData(32, kFloat, X, Y);
  O := TLoraAdamW.Create([S], DefaultAdamWConfig);
  Ref := TOptimizer.AdamW(SR, 0.05, 0.9, 0.999, 0.01, 1e-8);
  try
    for K := 1 to 100 do
    begin
      O.ZeroGrad;
      ToyLoss(W, B, X, Y).Backward;
      O.Step(ScheduleLR(0.05, K, 10, 100, 0.1));
      Ref.ZeroGrad;
      ToyLoss(RW, RB, X, Y).Backward;
      Ref.SetLearningRate(ScheduleLR(0.05, K, 10, 100, 0.1));
      Ref.Step;
    end;
    Result := Max(MaxAbsDiff(W, RW), MaxAbsDiff(B, RB));
  finally
    O.Free;
    Ref.Free;
    S.Free;
    SR.Free;
  end;
end;

{ 10 updates in one go against 5, save (store and moments), a new store
  and optimizer loaded from the files, StepCount := 5, 5 more. }
function Resume(const Dir: string): Double;
var
  SA, SB, SC: TVarStore;
  OA, OB, OC: TLoraAdamW;
  WA, BA, WB, BB, WC, BC, X, Y: TTensor;
  K: Integer;
begin
  SA := MakeToyStore(41, kFloat, WA, BA);
  SB := MakeToyStore(41, kFloat, WB, BB);
  SC := MakeToyStore(99, kFloat, WC, BC);
  MakeToyData(42, kFloat, X, Y);
  OA := TLoraAdamW.Create([SA], DefaultAdamWConfig);
  OB := TLoraAdamW.Create([SB], DefaultAdamWConfig);
  OC := nil;
  try
    for K := 1 to 10 do
    begin
      OA.ZeroGrad;
      ToyLoss(WA, BA, X, Y).Backward;
      OA.Step(0.05);
    end;
    for K := 1 to 5 do
    begin
      OB.ZeroGrad;
      ToyLoss(WB, BB, X, Y).Backward;
      OB.Step(0.05);
    end;
    SB.Save(Dir + '/params.ot');
    OB.Moments(0).Save(Dir + '/moments.ot');
    SC.Load(Dir + '/params.ot');
    OC := TLoraAdamW.Create([SC], DefaultAdamWConfig);
    OC.Moments(0).Load(Dir + '/moments.ot');
    OC.StepCount := 5;
    for K := 1 to 5 do
    begin
      OC.ZeroGrad;
      ToyLoss(WC, BC, X, Y).Backward;
      OC.Step(0.05);
    end;
    Result := Max(MaxAbsDiff(WA, WC), MaxAbsDiff(BA, BC));
  finally
    OA.Free;
    OB.Free;
    OC.Free;
    SA.Free;
    SB.Free;
    SC.Free;
  end;
end;

{ One backward on two equal toy stores; ours clips with
  ClipGlobalNorm(0.1), the reference with ClipGradNorm(0.1). Max of the
  clipped gradients' difference and |our norm - the norm of all
  gradients before clipping|. }
function ClipCheck: Double;
var
  S, SR: TVarStore;
  O: TLoraAdamW;
  Ref: TOptimizer;
  W, B, RW, RB, X, Y: TTensor;
  N, NRef: Double;
begin
  S := MakeToyStore(61, kDouble, W, B);
  SR := MakeToyStore(61, kDouble, RW, RB);
  MakeToyData(62, kDouble, X, Y);
  O := TLoraAdamW.Create([S], DefaultAdamWConfig);
  Ref := TOptimizer.AdamW(SR, 0.05, 0.9, 0.999, 0.01, 1e-8);
  try
    O.ZeroGrad;
    ToyLoss(W, B, X, Y).Backward;
    Ref.ZeroGrad;
    ToyLoss(RW, RB, X, Y).Backward;
    NRef := Sqrt(Sqr(RW.grad.norm.DoubleValue([])) + Sqr(RB.grad.norm.DoubleValue([])));
    N := O.ClipGlobalNorm(0.1);
    Ref.ClipGradNorm(0.1);
    Result := Max(Max(MaxAbsDiff(W.grad, RW.grad), MaxAbsDiff(B.grad, RB.grad)), Abs(N - NRef));
  finally
    O.Free;
    Ref.Free;
    S.Free;
    SR.Free;
  end;
end;

procedure CpuCases;
begin
  Emit('ITa-T14-001', Parity(1, 2, kFloat, 20, 0.05, 0, TDevice.Cpu), 1e-6,
    'fp32, 20 updates, lr 0.05, wd 0.01 against TOptimizer.AdamW');
  Emit('ITa-T14-002', Parity(3, 4, kDouble, 200, 0.05, 0, TDevice.Cpu), 1e-12,
    'fp64, 200 updates against TOptimizer.AdamW');
  Emit('ITa-T14-003', Parity(5, 6, kFloat, 20, 0.05, 0.5, TDevice.Cpu), 1e-6,
    'fp32, 20 updates, global-norm clipping at 0.5 against ClipGradNorm + AdamW');
  Emit('ITa-T14-004', TwoStores(kDouble, TDevice.Cpu, TDevice.Cpu), 0,
    'fp64, 50 updates with clipping: two stores (w; b) against one store (w, b)');
  Emit('ITa-T14-005', Scheduled, 1e-6,
    'fp32, 100 updates, ScheduleLR(0.05, u, 10, 100, 0.1) on both sides');
  Emit('ITa-T14-006', Resume(GetCurrentDir), 0,
    'fp32: 10 updates against 5 + save + load + StepCount 5 + 5');
  Emit('ITa-T14-007', ClipCheck, 1e-12,
    'fp64: gradients after ClipGlobalNorm(0.1) against ClipGradNorm(0.1), and the returned norm');
end;

procedure GpuCases;
begin
  if CudaDeviceCount < 2 then
  begin
    Emit('ITa-T14-101', 1, 0, Format('needs two GPUs, found %d', [CudaDeviceCount]));
    Exit;
  end;
  Emit('ITa-T14-101', Parity(51, 52, kFloat, 20, 0.05, 0.5, TDevice.Cuda(0)), 1e-5,
    'fp32 on cuda:0 with clipping against TOptimizer.AdamW on the CPU');
  Emit('ITa-T14-102', TwoStores(kFloat, TDevice.Cuda(0), TDevice.Cuda(1)), 1e-5,
    'fp32: w on cuda:0, b on cuda:1, clipping over both, against one CPU store');
end;

begin
  if (ParamCount >= 1) and (ParamStr(1) = 'gpu') then
    GpuCases
  else
    CpuCases;
end.
