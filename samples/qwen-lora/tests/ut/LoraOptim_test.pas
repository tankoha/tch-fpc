{ UT program for src/LoraOptim.pas: the rows UT-LoraOptim-NNN of
  tests/ut/cases/LoraOptim.cases.md (stage 1 from 001: DefaultAdamWConfig
  and ScheduleLR; stage 2 from 011: Create, the moment stores, ZeroGrad;
  stage 3 from 101: Step and ClipGlobalNorm). The toy problem and the
  reference run with TOptimizer.AdamW come from the harness owner's
  tests/ut/LoraOptimFixture.pas.
  Runs on the CPU (.sdlc/ut-coverage.sh hides the GPUs).
  Invoked with `--all --format=plain` (see .sdlc/ut-coverage.sh).

  Skeleton by the harness owner; the test methods are written in T14b. }
program LoraOptim_test;

{$mode objfpc}{$H+}

uses
  SysUtils, Math, Tch, TchNN, LoraOptim, LoraOptimFixture,
  fpcunit, testregistry, consoletestrunner;

type
  TLoraOptimTest = class(TTestCase)
  public
    { A CPU store of kDouble with one trainable variable 'p' of ones or
      zeros (AOnes) of length N; P receives it. }
    class function OneVarStore(N: Int64; AOnes: Boolean; out P: TTensor): TVarStore; static;
  published
    procedure TestUT_LoraOptim_001;
    procedure TestUT_LoraOptim_002;
    procedure TestUT_LoraOptim_003;
    procedure TestUT_LoraOptim_004;
    procedure TestUT_LoraOptim_005;
    procedure TestUT_LoraOptim_006;
    procedure TestUT_LoraOptim_007;
    procedure TestUT_LoraOptim_008;
    procedure TestUT_LoraOptim_009;
    procedure TestUT_LoraOptim_010;
    procedure TestUT_LoraOptim_011;
    procedure TestUT_LoraOptim_012;
    procedure TestUT_LoraOptim_013;
    procedure TestUT_LoraOptim_014;
    procedure TestUT_LoraOptim_015;
    procedure TestUT_LoraOptim_016;
    procedure TestUT_LoraOptim_101;
    procedure TestUT_LoraOptim_102;
    procedure TestUT_LoraOptim_103;
    procedure TestUT_LoraOptim_104;
    procedure TestUT_LoraOptim_105;
    procedure TestUT_LoraOptim_106;
    procedure TestUT_LoraOptim_107;
    procedure TestUT_LoraOptim_108;
    procedure TestUT_LoraOptim_109;
    procedure TestUT_LoraOptim_110;
    procedure TestUT_LoraOptim_111;
    procedure TestUT_LoraOptim_112;
  end;

class function TLoraOptimTest.OneVarStore(N: Int64; AOnes: Boolean; out P: TTensor): TVarStore;
begin
  Result := TVarStore.Create(TDevice.Cpu, kDouble);
  if AOnes then
    P := Result.Root.Ones('p', [N])
  else
    P := Result.Root.Zeros('p', [N]);
end;

{ UT-LoraOptim-001: default config, exact values. }
procedure TLoraOptimTest.TestUT_LoraOptim_001;
var
  C: TLoraAdamWConfig;
begin
  C := DefaultAdamWConfig;
  CheckEquals(0.9, C.Beta1);
  CheckEquals(0.999, C.Beta2);
  CheckEquals(1e-8, C.Eps);
  CheckEquals(0.01, C.WeightDecay);
end;

{ UT-LoraOptim-002: warm-up, first update. }
procedure TLoraOptimTest.TestUT_LoraOptim_002;
begin
  AssertEquals(2e-6, ScheduleLR(1e-4, 1, 50, 1000, 0.1), 1e-15);
end;

{ UT-LoraOptim-003: warm-up, last update. }
procedure TLoraOptimTest.TestUT_LoraOptim_003;
begin
  AssertEquals(1e-4, ScheduleLR(1e-4, 50, 50, 1000, 0.1), 1e-15);
end;

{ UT-LoraOptim-004: cosine decay, middle. }
procedure TLoraOptimTest.TestUT_LoraOptim_004;
begin
  AssertEquals(5.5e-5, ScheduleLR(1e-4, 525, 50, 1000, 0.1), 1e-15);
end;

{ UT-LoraOptim-005: decay, total update. }
procedure TLoraOptimTest.TestUT_LoraOptim_005;
begin
  AssertEquals(1e-5, ScheduleLR(1e-4, 1000, 50, 1000, 0.1), 1e-15);
end;

{ UT-LoraOptim-006: no warm-up. }
procedure TLoraOptimTest.TestUT_LoraOptim_006;
begin
  AssertEquals(7.75e-5, ScheduleLR(1e-4, 1, 0, 3, 0.1), 1e-15);
end;

{ UT-LoraOptim-007: update below range. }
procedure TLoraOptimTest.TestUT_LoraOptim_007;
begin
  try
    ScheduleLR(1e-4, 0, 50, 1000, 0.1);
    Fail('no ELoraOptim');
  except
    on E: ELoraOptim do
      AssertEquals('update 0 outside 1..1000', E.Message);
  end;
end;

{ UT-LoraOptim-008: update above range. }
procedure TLoraOptimTest.TestUT_LoraOptim_008;
begin
  try
    ScheduleLR(1e-4, 1001, 50, 1000, 0.1);
    Fail('no ELoraOptim');
  except
    on E: ELoraOptim do
      AssertEquals('update 1001 outside 1..1000', E.Message);
  end;
end;

{ UT-LoraOptim-009: warm-up above range. }
procedure TLoraOptimTest.TestUT_LoraOptim_009;
begin
  try
    ScheduleLR(1e-4, 10, 1000, 1000, 0.1);
    Fail('no ELoraOptim');
  except
    on E: ELoraOptim do
      AssertEquals('warm-up 1000 outside 0..999', E.Message);
  end;
end;

{ UT-LoraOptim-010: warm-up negative. }
procedure TLoraOptimTest.TestUT_LoraOptim_010;
begin
  try
    ScheduleLR(1e-4, 10, -1, 1000, 0.1);
    Fail('no ELoraOptim');
  except
    on E: ELoraOptim do
      AssertEquals('warm-up -1 outside 0..999', E.Message);
  end;
end;

{ UT-LoraOptim-011: Create, two stores, the moment layout. }
procedure TLoraOptimTest.TestUT_LoraOptim_011;
var
  S1, S2: TVarStore;
  W, B: TTensor;
  Opt: TLoraAdamW;
begin
  S1 := MakeToyStore(3, kDouble, W, B);
  S2 := TVarStore.Create(TDevice.Cpu, kFloat);
  try
    S2.Root.Zeros('c', [2]);
    Opt := TLoraAdamW.Create([S1, S2], DefaultAdamWConfig);
    CheckEquals(2, Opt.StoreCount);
    CheckEquals(4, Opt.Moments(0).Count);
    CheckEquals(2, Opt.Moments(1).Count);
    CheckEquals('m.0', Opt.Moments(0).Variables[0].Name);
    CheckEquals('v.0', Opt.Moments(0).Variables[1].Name);
    CheckEquals('m.1', Opt.Moments(0).Variables[2].Name);
    CheckEquals('v.1', Opt.Moments(0).Variables[3].Name);
    CheckEquals(0, Length(Opt.Moments(0).TrainableVariables));
    CheckEquals(Ord(kDouble), Ord(Opt.Moments(0).Kind));
    CheckEquals(Ord(kFloat), Ord(Opt.Moments(1).Kind));
    CheckEquals(0.0, Opt.Moments(0).Variables[0].Tensor.abs.sum(kNone).DoubleValue([]));
    CheckEquals(0, Opt.StepCount);
  finally
    Opt.Free;
    S2.Free;
    S1.Free;
  end;
end;

{ UT-LoraOptim-012: Create, no stores. }
procedure TLoraOptimTest.TestUT_LoraOptim_012;
begin
  try
    TLoraAdamW.Create([], DefaultAdamWConfig).Free;
    Fail('no ELoraOptim');
  except
    on E: ELoraOptim do
      AssertEquals('no stores', E.Message);
  end;
end;

{ UT-LoraOptim-013: Create, a store without trainable variables. }
procedure TLoraOptimTest.TestUT_LoraOptim_013;
var
  S1, S2: TVarStore;
  W, B: TTensor;
  Opt: TLoraAdamW;
begin
  S1 := MakeToyStore(3, kDouble, W, B);
  S2 := TVarStore.Create(TDevice.Cpu, kDouble);
  try
    S2.Root.ZerosNoTrain('z', [2]);
    try
      Opt := TLoraAdamW.Create([S1, S2], DefaultAdamWConfig);
      try
        Fail('no ELoraOptim');
      finally
        Opt.Free;
      end;
    except
      on E: ELoraOptim do
        AssertEquals('store 1 has no trainable variables', E.Message);
    end;
  finally
    S2.Free;
    S1.Free;
  end;
end;

{ UT-LoraOptim-014: ZeroGrad clears the gradients of the parameters. }
procedure TLoraOptimTest.TestUT_LoraOptim_014;
var
  S1: TVarStore;
  W, B: TTensor;
  X, Y: TTensor;
  Opt: TLoraAdamW;
begin
  S1 := MakeToyStore(3, kDouble, W, B);
  MakeToyData(4, kDouble, X, Y);
  try
    Opt := TLoraAdamW.Create([S1], DefaultAdamWConfig);
    ToyLoss(W, B, X, Y).Backward;
    CheckTrue(W.grad.abs.sum(kNone).DoubleValue([]) > 0);
    Opt.ZeroGrad;
    CheckEquals('0', ValuesText(W.grad.abs.sum(kNone)));
    CheckEquals('0', ValuesText(B.grad.abs.sum(kNone)));
  finally
    Opt.Free;
    S1.Free;
  end;
end;

{ UT-LoraOptim-015: Moments, index above the range. }
procedure TLoraOptimTest.TestUT_LoraOptim_015;
var
  S1, S2: TVarStore;
  W, B: TTensor;
  Opt: TLoraAdamW;
begin
  S1 := MakeToyStore(3, kDouble, W, B);
  S2 := TVarStore.Create(TDevice.Cpu, kFloat);
  try
    S2.Root.Zeros('c', [2]);
    Opt := TLoraAdamW.Create([S1, S2], DefaultAdamWConfig);
    try
      Opt.Moments(2);
      Fail('no ELoraOptim');
    except
      on E: ELoraOptim do
        AssertEquals('store 2 out of range (count 2)', E.Message);
    end;
  finally
    Opt.Free;
    S2.Free;
    S1.Free;
  end;
end;

{ UT-LoraOptim-016: Moments, negative index. }
procedure TLoraOptimTest.TestUT_LoraOptim_016;
var
  S1, S2: TVarStore;
  W, B: TTensor;
  Opt: TLoraAdamW;
begin
  S1 := MakeToyStore(3, kDouble, W, B);
  S2 := TVarStore.Create(TDevice.Cpu, kFloat);
  try
    S2.Root.Zeros('c', [2]);
    Opt := TLoraAdamW.Create([S1, S2], DefaultAdamWConfig);
    try
      Opt.Moments(-1);
      Fail('no ELoraOptim');
    except
      on E: ELoraOptim do
        AssertEquals('store -1 out of range (count 2)', E.Message);
    end;
  finally
    Opt.Free;
    S2.Free;
    S1.Free;
  end;
end;

{ UT-LoraOptim-101: Step, one step, hand values (Eps := 0). }
procedure TLoraOptimTest.TestUT_LoraOptim_101;
var
  S: TVarStore;
  P: TTensor;
  Cfg: TLoraAdamWConfig;
  Opt: TLoraAdamW;
begin
  S := OneVarStore(1, True, P);
  try
    Cfg := DefaultAdamWConfig;
    Cfg.Eps := 0;
    Opt := TLoraAdamW.Create([S], Cfg);
    Opt.ZeroGrad;
    (P * 0.5).sum(kNone).Backward;
    Opt.Step(0.1);
    AssertEquals(0.899, P.DoubleValue([0]), 1e-12);
    CheckEquals('0.05', ValuesText(Opt.Moments(0).Variables[0].Tensor));
    CheckEquals('0.00025', ValuesText(Opt.Moments(0).Variables[1].Tensor));
    CheckEquals(1, Opt.StepCount);
  finally
    Opt.Free;
    S.Free;
  end;
end;

{ UT-LoraOptim-102: Step, second step, hand values (Eps := 0). }
procedure TLoraOptimTest.TestUT_LoraOptim_102;
var
  S: TVarStore;
  P: TTensor;
  Cfg: TLoraAdamWConfig;
  Opt: TLoraAdamW;
begin
  S := OneVarStore(1, True, P);
  try
    Cfg := DefaultAdamWConfig;
    Cfg.Eps := 0;
    Opt := TLoraAdamW.Create([S], Cfg);
    Opt.ZeroGrad;
    (P * 0.5).sum(kNone).Backward;
    Opt.Step(0.1);
    Opt.ZeroGrad;
    (P * 0.5).sum(kNone).Backward;
    Opt.Step(0.1);
    AssertEquals(0.798101, P.DoubleValue([0]), 1e-12);
    CheckEquals('0.095', ValuesText(Opt.Moments(0).Variables[0].Tensor));
    CheckEquals('0.00049975', ValuesText(Opt.Moments(0).Variables[1].Tensor));
    CheckEquals(2, Opt.StepCount);
  finally
    Opt.Free;
    S.Free;
  end;
end;

{ UT-LoraOptim-103: Step, parity with TOptimizer.AdamW, fp32. }
procedure TLoraOptimTest.TestUT_LoraOptim_103;
var
  S: TVarStore;
  W, B, X, Y, RW, RB: TTensor;
  Opt: TLoraAdamW;
  I: Integer;
begin
  S := MakeToyStore(5, kFloat, W, B);
  MakeToyData(6, kFloat, X, Y);
  try
    Opt := TLoraAdamW.Create([S], DefaultAdamWConfig);
    for I := 1 to 20 do
    begin
      Opt.ZeroGrad;
      ToyLoss(W, B, X, Y).Backward;
      Opt.Step(0.05);
    end;
    ReferenceRun(5, 6, kFloat, 20, 0.05, 0.01, 0, RW, RB);
    CheckTrue(MaxAbsDiff(W, RW) <= 1e-6);
    CheckTrue(MaxAbsDiff(B, RB) <= 1e-6);
    CheckEquals(20, Opt.StepCount);
  finally
    Opt.Free;
    S.Free;
  end;
end;

{ UT-LoraOptim-104: Step, parity with TOptimizer.AdamW, fp64. }
procedure TLoraOptimTest.TestUT_LoraOptim_104;
var
  S: TVarStore;
  W, B, X, Y, RW, RB: TTensor;
  Opt: TLoraAdamW;
  I: Integer;
begin
  S := MakeToyStore(5, kDouble, W, B);
  MakeToyData(6, kDouble, X, Y);
  try
    Opt := TLoraAdamW.Create([S], DefaultAdamWConfig);
    for I := 1 to 20 do
    begin
      Opt.ZeroGrad;
      ToyLoss(W, B, X, Y).Backward;
      Opt.Step(0.05);
    end;
    ReferenceRun(5, 6, kDouble, 20, 0.05, 0.01, 0, RW, RB);
    CheckTrue(MaxAbsDiff(W, RW) <= 1e-12);
    CheckTrue(MaxAbsDiff(B, RB) <= 1e-12);
    CheckEquals(20, Opt.StepCount);
  finally
    Opt.Free;
    S.Free;
  end;
end;

{ UT-LoraOptim-105: Step, a trainable variable without a grad is skipped
  (no update, no weight decay, its moments stay zero). }
procedure TLoraOptimTest.TestUT_LoraOptim_105;
var
  S: TVarStore;
  W, B, U, X, Y: TTensor;
  Opt: TLoraAdamW;
begin
  S := MakeToyStore(7, kDouble, W, B);
  U := S.Root.Ones('u', [2]);
  MakeToyData(8, kDouble, X, Y);
  try
    Opt := TLoraAdamW.Create([S], DefaultAdamWConfig);
    ToyLoss(W, B, X, Y).Backward;
    Opt.Step(0.1);
    CheckEquals('1 1', ValuesText(U));
    CheckEquals('m.2', Opt.Moments(0).Variables[4].Name);
    CheckEquals('0 0', ValuesText(Opt.Moments(0).Variables[4].Tensor));
    CheckEquals('v.2', Opt.Moments(0).Variables[5].Name);
    CheckEquals('0 0', ValuesText(Opt.Moments(0).Variables[5].Tensor));
  finally
    Opt.Free;
    S.Free;
  end;
end;

{ UT-LoraOptim-106: Step, zero learning rate: no change in P, moments
  updated, StepCount 1. }
procedure TLoraOptimTest.TestUT_LoraOptim_106;
var
  S: TVarStore;
  P: TTensor;
  Opt: TLoraAdamW;
begin
  S := OneVarStore(1, True, P);
  try
    Opt := TLoraAdamW.Create([S], DefaultAdamWConfig);
    (P * 0.5).sum(kNone).Backward;
    Opt.Step(0);
    CheckEquals('1', ValuesText(P));
    CheckEquals('0.05', ValuesText(Opt.Moments(0).Variables[0].Tensor));
    CheckEquals(1, Opt.StepCount);
  finally
    Opt.Free;
    S.Free;
  end;
end;

{ UT-LoraOptim-107: ClipGlobalNorm, clips (grad [3,4] -> ~[0.6,0.8]). }
procedure TLoraOptimTest.TestUT_LoraOptim_107;
var
  S: TVarStore;
  P: TTensor;
  Opt: TLoraAdamW;
  R: Double;
begin
  S := OneVarStore(2, False, P);
  try
    Opt := TLoraAdamW.Create([S], DefaultAdamWConfig);
    (P * TTensor.FromArray([3.0, 4], kDouble)).sum(kNone).Backward;
    R := Opt.ClipGlobalNorm(1);
    AssertEquals(5, R, 1e-12);
    CheckTrue(MaxAbsDiff(P.grad, TTensor.FromArray([0.6, 0.8], kDouble)) <= 1e-6);
  finally
    Opt.Free;
    S.Free;
  end;
end;

{ UT-LoraOptim-108: ClipGlobalNorm, no clipping (MaxNorm 10 > norm 5). }
procedure TLoraOptimTest.TestUT_LoraOptim_108;
var
  S: TVarStore;
  P: TTensor;
  Opt: TLoraAdamW;
  R: Double;
begin
  S := OneVarStore(2, False, P);
  try
    Opt := TLoraAdamW.Create([S], DefaultAdamWConfig);
    (P * TTensor.FromArray([3.0, 4], kDouble)).sum(kNone).Backward;
    R := Opt.ClipGlobalNorm(10);
    AssertEquals(5, R, 1e-12);
    CheckEquals('3 4', ValuesText(P.grad));
  finally
    Opt.Free;
    S.Free;
  end;
end;

{ UT-LoraOptim-109: ClipGlobalNorm, two stores (grads [3] and [4],
  global norm 5, both scaled by 1/(5+1e-6)). }
procedure TLoraOptimTest.TestUT_LoraOptim_109;
var
  S1, S2: TVarStore;
  P, C: TTensor;
  Opt: TLoraAdamW;
  R: Double;
begin
  S1 := TVarStore.Create(TDevice.Cpu, kDouble);
  S2 := TVarStore.Create(TDevice.Cpu, kDouble);
  try
    P := S1.Root.Zeros('p', [1]);
    C := S2.Root.Zeros('c', [1]);
    Opt := TLoraAdamW.Create([S1, S2], DefaultAdamWConfig);
    (P * 3.0 + C * 4.0).sum(kNone).Backward;
    R := Opt.ClipGlobalNorm(1);
    AssertEquals(5, R, 1e-12);
    CheckTrue(MaxAbsDiff(P.grad, TTensor.FromArray([0.6], kDouble)) <= 1e-6);
    CheckTrue(MaxAbsDiff(C.grad, TTensor.FromArray([0.8], kDouble)) <= 1e-6);
  finally
    Opt.Free;
    S2.Free;
    S1.Free;
  end;
end;

{ UT-LoraOptim-110: parity with ClipGradNorm + TOptimizer.AdamW, fp32. }
procedure TLoraOptimTest.TestUT_LoraOptim_110;
var
  S: TVarStore;
  W, B, X, Y, RW, RB: TTensor;
  Opt: TLoraAdamW;
  I: Integer;
begin
  S := MakeToyStore(9, kFloat, W, B);
  MakeToyData(10, kFloat, X, Y);
  try
    Opt := TLoraAdamW.Create([S], DefaultAdamWConfig);
    for I := 1 to 20 do
    begin
      Opt.ZeroGrad;
      ToyLoss(W, B, X, Y).Backward;
      Opt.ClipGlobalNorm(0.5);
      Opt.Step(0.05);
    end;
    ReferenceRun(9, 10, kFloat, 20, 0.05, 0.01, 0.5, RW, RB);
    CheckTrue(MaxAbsDiff(W, RW) <= 1e-6);
    CheckTrue(MaxAbsDiff(B, RB) <= 1e-6);
  finally
    Opt.Free;
    S.Free;
  end;
end;

{ UT-LoraOptim-111: two stores equal one store (exact, fp64): SU holds
  w+b, [S2,S3] hold w and b; same clip+step loop -> identical results. }
procedure TLoraOptimTest.TestUT_LoraOptim_111;
var
  SU, S2, S3: TVarStore;
  W, B, W2, B2, X, Y: TTensor;
  Opt1, Opt2: TLoraAdamW;
  I: Integer;
begin
  SU := MakeToyStore(11, kDouble, W, B);
  S2 := TVarStore.Create(TDevice.Cpu, kDouble);
  S3 := TVarStore.Create(TDevice.Cpu, kDouble);
  ManualSeed(11);
  W2 := S2.Root.RandnStandard('w', [4, 3]);
  B2 := S3.Root.Zeros('b', [3]);
  MakeToyData(12, kDouble, X, Y);
  try
    Opt1 := TLoraAdamW.Create([SU], DefaultAdamWConfig);
    Opt2 := TLoraAdamW.Create([S2, S3], DefaultAdamWConfig);
    for I := 1 to 20 do
    begin
      Opt1.ZeroGrad;
      ToyLoss(W, B, X, Y).Backward;
      Opt1.ClipGlobalNorm(0.5);
      Opt1.Step(0.05);
      Opt2.ZeroGrad;
      ToyLoss(W2, B2, X, Y).Backward;
      Opt2.ClipGlobalNorm(0.5);
      Opt2.Step(0.05);
    end;
    CheckEquals(0, MaxAbsDiff(W, W2));
    CheckEquals(0, MaxAbsDiff(B, B2));
  finally
    Opt1.Free;
    Opt2.Free;
    SU.Free;
    S3.Free;
    S2.Free;
  end;
end;

{ UT-LoraOptim-112: no gradients: ClipGlobalNorm returns 0, Step is a
  no-op (P unchanged), StepCount 1. }
procedure TLoraOptimTest.TestUT_LoraOptim_112;
var
  S: TVarStore;
  P: TTensor;
  Opt: TLoraAdamW;
  R: Double;
begin
  S := OneVarStore(2, True, P);
  try
    Opt := TLoraAdamW.Create([S], DefaultAdamWConfig);
    R := Opt.ClipGlobalNorm(1);
    Opt.Step(0.1);
    AssertEquals(0, R, 1e-12);
    CheckEquals('1 1', ValuesText(P));
    CheckEquals(1, Opt.StepCount);
  finally
    Opt.Free;
    S.Free;
  end;
end;

var
  App: TTestRunner;
begin
  RegisterTest(TLoraOptimTest);
  App := TTestRunner.Create(nil);
  App.Initialize;
  App.Title := 'LoraOptim UT';
  App.Run;
  App.Free;
end.
