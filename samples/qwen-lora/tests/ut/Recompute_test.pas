{ UT program for src/Recompute.pas: the rows UT-Recompute-NNN of
  tests/ut/cases/Recompute.cases.md (stage 1 from 001: TSavedInputs and
  RecomputeForward; stage 2 from 101: RecomputeBackward). Test stages and
  the whole-graph reference come from the harness owner's
  tests/ut/RecomputeFixture.pas.
  Runs on the CPU (.sdlc/ut-coverage.sh hides the GPUs).
  Invoked with `--all --format=plain` (see .sdlc/ut-coverage.sh).

  Skeleton by the harness owner; the test methods are written in T13b. }
program Recompute_test;

{$mode objfpc}{$H+}

uses
  SysUtils, Tch, Recompute, RecomputeFixture,
  fpcunit, testregistry, consoletestrunner;

type
  TRecomputeTest = class(TTestCase)
  public
    { Two scale stages, w = 2 (S0, first) and w = 3 (S1); free them with
      FreeStages. }
    class function Hand(out S0, S1: TScaleStage): TStageArray; static;
  published
    procedure TestUT_Recompute_001;
    procedure TestUT_Recompute_002;
    procedure TestUT_Recompute_003;
    procedure TestUT_Recompute_004;
    procedure TestUT_Recompute_005;
    procedure TestUT_Recompute_006;
    procedure TestUT_Recompute_007;
    procedure TestUT_Recompute_008;
    procedure TestUT_Recompute_009;
    procedure TestUT_Recompute_010;
    procedure TestUT_Recompute_011;
    procedure TestUT_Recompute_012;
    procedure TestUT_Recompute_013;
    procedure TestUT_Recompute_101;
    procedure TestUT_Recompute_102;
    procedure TestUT_Recompute_103;
    procedure TestUT_Recompute_104;
    procedure TestUT_Recompute_105;
    procedure TestUT_Recompute_106;
    procedure TestUT_Recompute_107;
    procedure TestUT_Recompute_108;
  end;

class function TRecomputeTest.Hand(out S0, S1: TScaleStage): TStageArray;
begin
  Result := nil;
  S0 := TScaleStage.Create(2);
  S1 := TScaleStage.Create(3);
  SetLength(Result, 2);
  Result[0] := S0;
  Result[1] := S1;
end;

procedure TRecomputeTest.TestUT_Recompute_001;
var
  Saved: TSavedInputs;
begin
  Saved := TSavedInputs.Create(False);
  try
    AssertEquals(False, Saved.OnHost);
    AssertEquals(0, Saved.Count);
  finally
    Saved.Free;
  end;
end;

procedure TRecomputeTest.TestUT_Recompute_002;
var
  Saved: TSavedInputs;
begin
  Saved := TSavedInputs.Create(True);
  try
    AssertEquals(True, Saved.OnHost);
    AssertEquals(0, Saved.Count);
  finally
    Saved.Free;
  end;
end;

procedure TRecomputeTest.TestUT_Recompute_003;
var
  Saved: TSavedInputs;
  X: TTensor;
begin
  Saved := TSavedInputs.Create(False);
  try
    X := TTensor.FromArray([1.0, 2]).requires_grad_(True);
    Saved.Add(X);
    Saved.Add(TTensor.FromArray([3.0]));
    AssertEquals(2, Saved.Count);
    AssertEquals('1 2', ValuesText(Saved.Item(0)));
    AssertEquals(False, Saved.Item(0).RequiresGrad);
    AssertEquals(False, Saved.Item(0).SameHandle(X));
    AssertEquals('3', ValuesText(Saved.Item(1)));
  finally
    Saved.Free;
  end;
end;

procedure TRecomputeTest.TestUT_Recompute_004;
var
  Saved: TSavedInputs;
begin
  Saved := TSavedInputs.Create(True);
  try
    Saved.Add(TTensor.FromArray([1.0, 2]));
    AssertEquals('1 2', ValuesText(Saved.Item(0)));
    AssertEquals('cpu', Saved.Item(0).Device.ToString);
  finally
    Saved.Free;
  end;
end;

procedure TRecomputeTest.TestUT_Recompute_005;
var
  Saved: TSavedInputs;
begin
  Saved := TSavedInputs.Create(True);
  try
    Saved.Add(TTensor.FromArray([1.0]));
    Saved.Add(TTensor.FromArray([2.0]));
    Saved.Clear;
    AssertEquals(0, Saved.Count);
  finally
    Saved.Free;
  end;
end;

procedure TRecomputeTest.TestUT_Recompute_006;
var
  Saved: TSavedInputs;
begin
  Saved := TSavedInputs.Create(False);
  try
    Saved.Add(TTensor.FromArray([1.0]));
    Saved.Add(TTensor.FromArray([2.0]));
    try
      Saved.Item(2);
      Fail('no ERecompute');
    except
      on E: ERecompute do
        AssertEquals('saved input 2 out of range (count 2)', E.Message);
    end;
  finally
    Saved.Free;
  end;
end;

procedure TRecomputeTest.TestUT_Recompute_007;
var
  Saved: TSavedInputs;
begin
  Saved := TSavedInputs.Create(False);
  try
    Saved.Add(TTensor.FromArray([1.0]));
    Saved.Add(TTensor.FromArray([2.0]));
    try
      Saved.Item(-1);
      Fail('no ERecompute');
    except
      on E: ERecompute do
        AssertEquals('saved input -1 out of range (count 2)', E.Message);
    end;
  finally
    Saved.Free;
  end;
end;

procedure TRecomputeTest.TestUT_Recompute_008;
var
  St: TStageArray;
  S0, S1: TScaleStage;
  Saved: TSavedInputs;
  Y: TTensor;
begin
  St := Hand(S0, S1);
  Saved := TSavedInputs.Create(False);
  try
    Y := RecomputeForward(St, TTensor.FromArray([1.0, 2]), Saved);
    AssertEquals('6 12', ValuesText(Y));
    AssertEquals(False, Y.RequiresGrad);
    AssertEquals(2, Saved.Count);
    AssertEquals('1 2', ValuesText(Saved.Item(0)));
    AssertEquals('2 4', ValuesText(Saved.Item(1)));
    AssertEquals(1, S0.Calls);
    AssertEquals(1, S1.Calls);
    AssertEquals(False, S0.LastRequiresGrad);
    AssertEquals(False, S1.LastRequiresGrad);
  finally
    Saved.Free;
    FreeStages(St);
  end;
end;

procedure TRecomputeTest.TestUT_Recompute_009;
var
  St: TStageArray;
  S0, S1: TScaleStage;
  Saved: TSavedInputs;
  Y: TTensor;
begin
  St := Hand(S0, S1);
  Saved := TSavedInputs.Create(False);
  try
    Saved.Add(TTensor.FromArray([1.0]));
    Saved.Add(TTensor.FromArray([1.0]));
    Saved.Add(TTensor.FromArray([1.0]));
    Y := RecomputeForward(St, TTensor.FromArray([1.0, 2]), Saved);
    AssertEquals(2, Saved.Count);
  finally
    Saved.Free;
    FreeStages(St);
  end;
end;

procedure TRecomputeTest.TestUT_Recompute_010;
var
  A: TStageArray;
  Saved: TSavedInputs;
  X, Y: TTensor;
  I: Integer;
begin
  A := MakeStages(51, 4, 8, kDouble, [TDevice.Cpu]);
  Saved := TSavedInputs.Create(False);
  try
    ManualSeed(52);
    X := TTensor.randn([3, 8], kDouble);
    Y := RecomputeForward(A, X, Saved);
    for I := 0 to High(A) do
    begin
      AssertEquals(1, (A[I] as TAffineStage).Calls);
      AssertEquals(False, (A[I] as TAffineStage).LastRequiresGrad);
    end;
  finally
    Saved.Free;
    FreeStages(A);
  end;
end;

procedure TRecomputeTest.TestUT_Recompute_011;
var
  A, B: TStageArray;
  Saved: TSavedInputs;
  X, YA, YB: TTensor;
  I: Integer;
begin
  A := MakeStages(51, 4, 8, kDouble, [TDevice.Cpu]);
  B := CloneStages(A, kDouble, [TDevice.Cpu]);
  Saved := TSavedInputs.Create(False);
  try
    ManualSeed(52);
    X := TTensor.randn([3, 8], kDouble);
    YA := RecomputeForward(A, X, Saved);
    YB := X;
    for I := 0 to High(B) do
      YB := B[I].Forward(YB.&to(B[I].Device));
    AssertEquals('[3, 8]', ShapeText(YA));
    AssertEquals(0.0, MaxAbsDiff(YA, YB));
  finally
    Saved.Free;
    FreeStages(A);
    FreeStages(B);
  end;
end;

procedure TRecomputeTest.TestUT_Recompute_012;
var
  Saved: TSavedInputs;
begin
  Saved := TSavedInputs.Create(False);
  try
    try
      RecomputeForward(nil, TTensor.FromArray([1.0]), Saved);
      Fail('no ERecompute');
    except
      on E: ERecompute do
        AssertEquals('no stages', E.Message);
    end;
  finally
    Saved.Free;
  end;
end;

procedure TRecomputeTest.TestUT_Recompute_013;
var
  St: TStageArray;
  S0, S1: TScaleStage;
  Saved: TSavedInputs;
begin
  St := Hand(S0, S1);
  Saved := TSavedInputs.Create(False);
  try
    try
      RecomputeForward(St, Default(TTensor), Saved);
      Fail('no ERecompute');
    except
      on E: ERecompute do
        AssertEquals('X0 is undefined', E.Message);
    end;
  finally
    Saved.Free;
    FreeStages(St);
  end;
end;

procedure TRecomputeTest.TestUT_Recompute_101;
var
  St: TStageArray;
  S0, S1: TScaleStage;
  Saved: TSavedInputs;
  Y, GIn: TTensor;
begin
  St := Hand(S0, S1);
  Saved := TSavedInputs.Create(False);
  try
    Y := RecomputeForward(St, TTensor.FromArray([1.0, 2]), Saved);
    GIn := RecomputeBackward(St, Saved, TTensor.FromArray([1.0, 1]));
    AssertEquals('6 6', ValuesText(GIn));
    AssertEquals('9', ValuesText(S0.W.grad));
    AssertEquals('6', ValuesText(S1.W.grad));
    AssertEquals(2, S0.Calls);
    AssertEquals(2, S1.Calls);
    AssertEquals(True, S0.LastRequiresGrad);
    AssertEquals(True, S1.LastRequiresGrad);
  finally
    Saved.Free;
    FreeStages(St);
  end;
end;

procedure TRecomputeTest.TestUT_Recompute_102;
var
  A, B: TStageArray;
  Saved: TSavedInputs;
  X, GOut, GA, GB: TTensor;
begin
  A := MakeStages(53, 8, 16, kDouble, [TDevice.Cpu]);
  B := CloneStages(A, kDouble, [TDevice.Cpu]);
  Saved := TSavedInputs.Create(False);
  try
    ManualSeed(54);
    X := TTensor.randn([4, 16], kDouble);
    GOut := TTensor.randn([4, 16], kDouble);
    RecomputeForward(A, X, Saved);
    GA := RecomputeBackward(A, Saved, GOut);
    GB := WholeGraphBackward(B, X, GOut);
    AssertEquals(0.0, MaxAbsDiff(GA, GB));
    AssertEquals(0.0, MaxParamGradDiff(A, B));
  finally
    Saved.Free;
    FreeStages(A);
    FreeStages(B);
  end;
end;

procedure TRecomputeTest.TestUT_Recompute_103;
var
  A: TStageArray;
  Saved: TSavedInputs;
  X, Y: TTensor;
  I: Integer;
begin
  A := MakeStages(51, 4, 8, kDouble, [TDevice.Cpu]);
  Saved := TSavedInputs.Create(False);
  try
    ManualSeed(52);
    X := TTensor.randn([3, 8], kDouble);
    Y := RecomputeForward(A, X, Saved);
    RecomputeBackward(A, Saved, TTensor.ones([3, 8], kDouble));
    for I := 0 to High(A) do
    begin
      AssertEquals(2, (A[I] as TAffineStage).Calls);
      AssertEquals(True, (A[I] as TAffineStage).LastRequiresGrad);
    end;
  finally
    Saved.Free;
    FreeStages(A);
  end;
end;

procedure TRecomputeTest.TestUT_Recompute_104;
var
  St: TStageArray;
  S0, S1: TScaleStage;
  Saved: TSavedInputs;
  Y, GIn: TTensor;
begin
  St := Hand(S0, S1);
  Saved := TSavedInputs.Create(False);
  try
    Y := RecomputeForward(St, TTensor.FromArray([1.0, 2]), Saved);
    RecomputeBackward(St, Saved, TTensor.FromArray([1.0, 1]));
    GIn := RecomputeBackward(St, Saved, TTensor.FromArray([1.0, 1]));
    AssertEquals('6 6', ValuesText(GIn));
    AssertEquals('18', ValuesText(S0.W.grad));
    AssertEquals('12', ValuesText(S1.W.grad));
  finally
    Saved.Free;
    FreeStages(St);
  end;
end;

procedure TRecomputeTest.TestUT_Recompute_105;
var
  A, C: TStageArray;
  SavedA, SavedC: TSavedInputs;
  X, GOut, GA, GC: TTensor;
begin
  A := MakeStages(53, 8, 16, kDouble, [TDevice.Cpu]);
  C := CloneStages(A, kDouble, [TDevice.Cpu]);
  SavedA := TSavedInputs.Create(False);
  SavedC := TSavedInputs.Create(True);
  try
    ManualSeed(54);
    X := TTensor.randn([4, 16], kDouble);
    GOut := TTensor.randn([4, 16], kDouble);
    RecomputeForward(A, X, SavedA);
    GA := RecomputeBackward(A, SavedA, GOut);
    RecomputeForward(C, X, SavedC);
    GC := RecomputeBackward(C, SavedC, GOut);
    AssertEquals(0.0, MaxAbsDiff(GA, GC));
    AssertEquals(0.0, MaxParamGradDiff(A, C));
  finally
    SavedA.Free;
    SavedC.Free;
    FreeStages(A);
    FreeStages(C);
  end;
end;

procedure TRecomputeTest.TestUT_Recompute_106;
var
  St: TStageArray;
  S0, S1: TScaleStage;
  Saved: TSavedInputs;
begin
  St := Hand(S0, S1);
  Saved := TSavedInputs.Create(False);
  try
    Saved.Add(TTensor.FromArray([1.0]));
    try
      RecomputeBackward(St, Saved, TTensor.FromArray([1.0, 1]));
      Fail('no ERecompute');
    except
      on E: ERecompute do
        AssertEquals('saved inputs: 1, stages: 2', E.Message);
    end;
  finally
    Saved.Free;
    FreeStages(St);
  end;
end;

procedure TRecomputeTest.TestUT_Recompute_107;
var
  St: TStageArray;
  S0, S1: TScaleStage;
  Saved: TSavedInputs;
  Y: TTensor;
begin
  St := Hand(S0, S1);
  Saved := TSavedInputs.Create(False);
  try
    Y := RecomputeForward(St, TTensor.FromArray([1.0, 2]), Saved);
    try
      RecomputeBackward(St, Saved, Default(TTensor));
      Fail('no ERecompute');
    except
      on E: ERecompute do
        AssertEquals('GradOut is undefined', E.Message);
    end;
  finally
    Saved.Free;
    FreeStages(St);
  end;
end;

procedure TRecomputeTest.TestUT_Recompute_108;
var
  St: TStageArray;
  S0, S1: TScaleStage;
  Saved: TSavedInputs;
  Y: TTensor;
begin
  St := Hand(S0, S1);
  Saved := TSavedInputs.Create(False);
  try
    Y := RecomputeForward(St, TTensor.FromArray([1.0, 2]), Saved);
    try
      RecomputeBackward(St, Saved, TTensor.FromArray([1.0, 1, 1]));
      Fail('no ERecompute');
    except
      on E: ERecompute do
        AssertEquals('stage 1: output [2], gradient [3]', E.Message);
    end;
  finally
    Saved.Free;
    FreeStages(St);
  end;
end;

var
  App: TTestRunner;
begin
  RegisterTest(TRecomputeTest);
  App := TTestRunner.Create(nil);
  App.Initialize;
  App.Title := 'Recompute UT';
  App.Run;
  App.Free;
end.
