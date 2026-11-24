{ UT program for src/TrainStep.pas: the rows UT-TrainStep-NNN of
  tests/ut/cases/TrainStep.cases.md. Runs on the CPU (.sdlc/ut-coverage.sh
  hides the GPUs) on the tiny model of tests/ut/QwenBlocksFixture.pas.
  Invoked with `--all --format=plain` (see .sdlc/ut-coverage.sh).

  Skeleton by the harness owner; the test methods are written in T17b1. }
program TrainStep_test;

{$mode objfpc}{$H+}

uses
  SysUtils, Classes, Math, Tch, GgufFile, QwenTokenizer, QwenModel, Lora, TrainData, TrainStep,
  QwenBlocksFixture, fpcunit, testregistry, consoletestrunner;

type
  TTrainStepTest = class(TTestCase)
  public
    { The tiny model (TinyPath, tied head) on the CPU; free M, then G. }
    class procedure LoadTiny(out G: TGgufFile; out M: TQwenModel); static;
    { A set of rank 4, scale 2 attached to M, every B := 0.1 N(0, 1) after
      ManualSeed(ASeed) (ASeed < 0: B stays zero). Free M before the set. }
    class function NewSet(M: TQwenModel; ASeed: Integer): TLoraSet; static;
    { Tokens (7 i + 3) mod 270, i = 0..19; loss span 12..19; id x. }
    class function FixSample: TTrainSample; static;
    { The summed cross-entropy of the span through the whole graph (one
      forward, one log_softmax over all positions), with a graph. }
    class function RefLoss(M: TQwenModel; const S: TTrainSample): TTensor; static;
    { Copies of every A grad, then every B grad (zeros where undefined). }
    class function Grads(S: TLoraSet): TTensorArray; static;
    { Zeroes every grad of the set. }
    class procedure ZeroGrads(S: TLoraSet); static;
    { Max over the tensors of max |a - b| / max(max |b|, 1e-30). }
    class function RelDiff(const A, B: TTensorArray): Double; static;
  published
    procedure TestUT_TrainStep_001;
    procedure TestUT_TrainStep_002;
    procedure TestUT_TrainStep_003;
    procedure TestUT_TrainStep_004;
    procedure TestUT_TrainStep_005;
    procedure TestUT_TrainStep_006;
    procedure TestUT_TrainStep_007;
    procedure TestUT_TrainStep_008;
    procedure TestUT_TrainStep_009;
  end;

var
  TinyPath, TmpDir: string;

class procedure TTrainStepTest.LoadTiny(out G: TGgufFile; out M: TQwenModel);
begin
  G := TGgufFile.Open(TinyPath);
  M := TQwenModel.Load(G, TDevice.Cpu, TDevice.Cpu, 0);
end;

procedure RandomB(S: TLoraSet; ASeed: Integer);
var
  NG: IInterface;
  J: Integer;
begin
  NG := NoGradGuard;
  ManualSeed(ASeed);
  for J := 0 to S.Count - 1 do
    S.Pair(J).B.CopyFrom(TTensor.randn([S.Pair(J).OutDim, S.Pair(J).Rank]) * 0.1);
end;

class function TTrainStepTest.NewSet(M: TQwenModel; ASeed: Integer): TLoraSet;
begin
  ManualSeed(3);
  Result := TLoraSet.Create(4, 2.0);
  M.AttachLora(Result);
  if ASeed >= 0 then
    RandomB(Result, ASeed);
end;

class function TTrainStepTest.FixSample: TTrainSample;
var
  I: Integer;
begin
  Result := Default(TTrainSample);
  Result.Id := 'x';
  SetLength(Result.Tokens, 20);
  for I := 0 to 19 do
    Result.Tokens[I] := (7 * I + 3) mod 270;
  Result.LossFrom := 12;
  Result.LossTo := 19;
end;

class function TTrainStepTest.RefLoss(M: TQwenModel; const S: TTrainSample): TTensor;
var
  L: TTensor;
  I: Integer;
begin
  L := M.Logits(M.Trunk(M.Embed(S.Tokens))).log_softmax(-1);
  Result := TTensor.zeros([]);
  for I := S.LossFrom to S.LossTo - 1 do
    Result := Result - L.select(0, I - 1).select(0, S.Tokens[I]);
end;

class function TTrainStepTest.Grads(S: TLoraSet): TTensorArray;
var
  I: Integer;
  G: TTensor;
begin
  Result := nil;
  SetLength(Result, 2 * S.Count);
  for I := 0 to S.Count - 1 do
  begin
    G := S.Pair(I).A.grad;
    if G.Defined then Result[I] := G.Copy else Result[I] := S.Pair(I).A.zeros_like;
    G := S.Pair(I).B.grad;
    if G.Defined then Result[S.Count + I] := G.Copy else Result[S.Count + I] := S.Pair(I).B.zeros_like;
  end;
end;

class procedure TTrainStepTest.ZeroGrads(S: TLoraSet);
var
  I: Integer;
begin
  for I := 0 to S.Count - 1 do
  begin
    S.Pair(I).A.ZeroGrad;
    S.Pair(I).B.ZeroGrad;
  end;
end;

class function TTrainStepTest.RelDiff(const A, B: TTensorArray): Double;
var
  I: Integer;
begin
  Result := 0;
  for I := 0 to High(A) do
    Result := Max(Result, (A[I] - B[I]).abs.max.DoubleValue([]) / Max(B[I].abs.max.DoubleValue([]), 1e-30));
end;

{ ETrainStep from SampleLossBackward (ABackward) or SampleLoss of AS. }
procedure CheckStep(T: TTestCase; M: TQwenModel; const AS0: TTrainSample; ADenom: Double;
  ASlice: Integer; ABackward: Boolean; const AMessage: string);
begin
  try
    if ABackward then
      SampleLossBackward(M, AS0, ADenom, ASlice)
    else
      SampleLoss(M, AS0, ASlice);
    T.Fail('no ETrainStep');
  except
    on E: ETrainStep do
      T.AssertEquals(AMessage, E.Message);
  end;
end;

{ The reference loss's value and, after (Ref / ADenom).Backward, the grads. }
function RefBackward(M: TQwenModel; S: TLoraSet; const ASample: TTrainSample; ADenom: Double;
  out AValue: Double): TTensorArray;
var
  R: TTensor;
begin
  TTrainStepTest.ZeroGrads(S);
  R := TTrainStepTest.RefLoss(M, ASample);
  AValue := R.DoubleValue([]);
  (R / ADenom).Backward;
  Result := TTrainStepTest.Grads(S);
  TTrainStepTest.ZeroGrads(S);
end;

{ UT-TrainStep-001: sample loss without gradients. }
procedure TTrainStepTest.TestUT_TrainStep_001;
var
  G: TGgufFile;
  M: TQwenModel;
  S: TLoraSet;
  R: TSampleLoss;
  Ref: Double;
begin
  LoadTiny(G, M);
  S := NewSet(M, 6);
  try
    Ref := RefLoss(M, FixSample).DoubleValue([]);
    R := SampleLoss(M, FixSample, 3);
    AssertEquals('count 7', 7, R.Count);
    AssertTrue(Format('sum = RefLoss rel 1e-5, got %g vs %g', [R.Sum, Ref]),
      Abs(R.Sum - Ref) / Max(Abs(Ref), 1e-30) <= 1e-5);
  finally
    M.Free;
    S.Free;
    G.Free;
  end;
end;

{ UT-TrainStep-002: backward: loss and grads against the reference. }
procedure TTrainStepTest.TestUT_TrainStep_002;
var
  G: TGgufFile;
  M: TQwenModel;
  S: TLoraSet;
  R: TSampleLoss;
  Ref: Double;
  I: Integer;
  BMax: Double;
  Want: TTensorArray;
begin
  LoadTiny(G, M);
  S := NewSet(M, 6);
  try
    Want := RefBackward(M, S, FixSample, 3, Ref);
    R := SampleLossBackward(M, FixSample, 3, 3);
    AssertEquals('count 7', 7, R.Count);
    AssertTrue(Format('sum = Ref rel 1e-5, got %g vs %g', [R.Sum, Ref]),
      Abs(R.Sum - Ref) / Max(Abs(Ref), 1e-30) <= 1e-5);
    AssertTrue(Format('grads within 1e-5 of the reference, got %g', [RelDiff(Grads(S), Want)]),
      RelDiff(Grads(S), Want) <= 1e-5);
    BMax := 0;
    for I := 0 to S.Count - 1 do
      BMax := Max(BMax, Want[S.Count + I].abs.max.DoubleValue([]));
    AssertTrue(Format('the B grads of Want not zero (max > 0.1), got %g', [BMax]), BMax > 0.1);
  finally
    M.Free;
    S.Free;
    G.Free;
  end;
end;

{ UT-TrainStep-003: backward: slice size 1, 3, 7, 100. }
procedure TTrainStepTest.TestUT_TrainStep_003;
var
  Slices: array of Integer;
  G: TGgufFile;
  M: TQwenModel;
  S: TLoraSet;
  R: TSampleLoss;
  Ref: Double;
  I: Integer;
  Want: TTensorArray;
begin
  LoadTiny(G, M);
  S := NewSet(M, 6);
  try
    Want := RefBackward(M, S, FixSample, 1, Ref);
    SetLength(Slices, 4);
    Slices[0] := 1;
    Slices[1] := 3;
    Slices[2] := 7;
    Slices[3] := 100;
    for I := 0 to High(Slices) do
    begin
      ZeroGrads(S);
      R := SampleLossBackward(M, FixSample, 1, Slices[I]);
      AssertEquals(Format('slice %d: count 7', [Slices[I]]), 7, R.Count);
      AssertTrue(Format('slice %d: sum = Ref rel 1e-5, got %g vs %g', [Slices[I], R.Sum, Ref]),
        Abs(R.Sum - Ref) / Max(Abs(Ref), 1e-30) <= 1e-5);
      AssertTrue(Format('slice %d: grads within 1e-5 of the reference, got %g',
        [Slices[I], RelDiff(Grads(S), Want)]),
        RelDiff(Grads(S), Want) <= 1e-5);
    end;
  finally
    M.Free;
    S.Free;
    G.Free;
  end;
end;

{ UT-TrainStep-004: denominator scaling and accumulation. }
procedure TTrainStepTest.TestUT_TrainStep_004;
var
  G: TGgufFile;
  M: TQwenModel;
  S: TLoraSet;
  R: TSampleLoss;
  I: Integer;
  G3, G6, G62, Acc: TTensorArray;
begin
  LoadTiny(G, M);
  S := NewSet(M, 6);
  try
    ZeroGrads(S);
    R := SampleLossBackward(M, FixSample, 3, 3);
    G3 := Grads(S);
    ZeroGrads(S);
    R := SampleLossBackward(M, FixSample, 6, 3);
    G6 := Grads(S);
    R := SampleLossBackward(M, FixSample, 6, 3);  { no zeroing }
    Acc := Grads(S);
    SetLength(G62, Length(G6));
    for I := 0 to High(G6) do
      G62[I] := G6[I] * 2.0;
    AssertTrue(Format('grads at 6 x 2 within 1e-6 of those at 3, got %g', [RelDiff(G62, G3)]),
      RelDiff(G62, G3) <= 1e-6);
    AssertTrue(Format('after the second call at 6 the grads equal those at 3 (rel 1e-6), got %g',
      [RelDiff(Acc, G3)]),
      RelDiff(Acc, G3) <= 1e-6);
  finally
    M.Free;
    S.Free;
    G.Free;
  end;
end;

{ UT-TrainStep-005: adapters with B zero: same loss as without adapters. }
procedure TTrainStepTest.TestUT_TrainStep_005;
var
  G, G0: TGgufFile;
  M, M0: TQwenModel;
  S: TLoraSet;
  R, R0: TSampleLoss;
begin
  LoadTiny(G, M);
  LoadTiny(G0, M0);
  S := NewSet(M, -1);
  try
    R := SampleLoss(M, FixSample, 3);
    R0 := SampleLoss(M0, FixSample, 3);
    AssertTrue(Format('same loss as without adapters exact, got %g vs %g', [R.Sum, R0.Sum]),
      R.Sum = R0.Sum);
  finally
    M.Free;
    S.Free;
    M0.Free;
    G0.Free;
    G.Free;
  end;
end;

{ UT-TrainStep-006: SampleLoss defines no grads. }
procedure TTrainStepTest.TestUT_TrainStep_006;
var
  G: TGgufFile;
  M: TQwenModel;
  S: TLoraSet;
  R: TSampleLoss;
  I: Integer;
begin
  LoadTiny(G, M);
  S := NewSet(M, 6);
  try
    R := SampleLoss(M, FixSample, 3);
    AssertEquals('count 7', 7, R.Count);
    for I := 0 to S.Count - 1 do
    begin
      AssertTrue(Format('pair %d: no A grad defined', [I]), not S.Pair(I).A.grad.Defined);
      AssertTrue(Format('pair %d: no B grad defined', [I]), not S.Pair(I).B.grad.Defined);
    end;
  finally
    M.Free;
    S.Free;
    G.Free;
  end;
end;

{ UT-TrainStep-007: ETrainStep messages: slice, denominator, loss span. }
procedure TTrainStepTest.TestUT_TrainStep_007;
var
  G: TGgufFile;
  M: TQwenModel;
  S0: TTrainSample;
begin
  LoadTiny(G, M);
  try
    CheckStep(Self, M, FixSample, 3, 0, True, 'slice must be positive, got 0');
    CheckStep(Self, M, FixSample, 3, 0, False, 'slice must be positive, got 0');
    CheckStep(Self, M, FixSample, 0, 3, True, 'denominator must be positive, got 0');
    CheckStep(Self, M, FixSample, -2, 3, True, 'denominator must be positive, got -2');
    S0 := FixSample;
    S0.LossFrom := 0;
    CheckStep(Self, M, S0, 3, 3, False, 'x: loss span 0..19 outside 1..20');
    S0 := FixSample;
    S0.LossTo := 12;
    CheckStep(Self, M, S0, 3, 3, False, 'x: loss span 12..12 outside 1..20');
    S0 := FixSample;
    S0.LossTo := 21;
    CheckStep(Self, M, S0, 3, 3, False, 'x: loss span 12..21 outside 1..20');
  finally
    M.Free;
    G.Free;
  end;
end;

{ UT-TrainStep-008: ValidationLoss over a set; empty set NaN. }
procedure TTrainStepTest.TestUT_TrainStep_008;
procedure WriteFile(const AName: string; const ALines: array of string);
var
  F: TextFile;
  J: Integer;
begin
  AssignFile(F, TmpDir + AName);
  Rewrite(F);
  for J := 0 to High(ALines) do
    Writeln(F, ALines[J]);
  CloseFile(F);
end;
var
  G, G2: TGgufFile;
  M: TQwenModel;
  Tok: TQwenTokenizer;
  V, E: TTrainSet;
  R: TSampleLoss;
  VL: Double;
  Sum: Double;
  N: Integer;
  I: Integer;
begin
  LoadTiny(G, M);
  try
    WriteFile('valid.jsonl', [
      '{"id":"r0","dialect":"objfpc","prompt":"hello","answer":"world","class":"c"}',
      '{"id":"r1","dialect":"objfpc","prompt":"ab","answer":"abc abc","class":"c"}',
      '{"id":"r2","dialect":"objfpc","prompt":"x","answer":"y","class":"c"}']);
    WriteFile('empty.jsonl', []);
    G2 := TGgufFile.Open(TinyPath);
    Tok := TQwenTokenizer.Create(G2);
    try
      V := TTrainSet.Create(Tok, TmpDir + 'valid.jsonl', ['c'], 100000);
      try
        AssertEquals('count 3', 3, V.Count);
        Sum := 0;
        N := 0;
        for I := 0 to V.Count - 1 do
        begin
          R := SampleLoss(M, V[I], 256);
          Sum := Sum + R.Sum;
          Inc(N, R.Count);
        end;
        VL := ValidationLoss(M, V, 2);
        AssertTrue(Format('validation loss = sum/count rel 1e-5, got %g vs %g', [VL, Sum / N]),
          Abs(VL - Sum / N) / Max(Abs(Sum / N), 1e-30) <= 1e-5);
        E := TTrainSet.Create(Tok, TmpDir + 'empty.jsonl', ['c'], 100000);
        try
          AssertTrue('empty set: NaN', IsNaN(ValidationLoss(M, E, 2)));
        finally
          E.Free;
        end;
      finally
        V.Free;
      end;
    finally
      Tok.Free;
      G2.Free;
    end;
  finally
    M.Free;
    G.Free;
  end;
end;

{ UT-TrainStep-009: backward with the loss span at the last token. }
procedure TTrainStepTest.TestUT_TrainStep_009;
var
  G: TGgufFile;
  M: TQwenModel;
  S: TLoraSet;
  F: TTrainSample;
  R: TSampleLoss;
  Ref: Double;
  Want: TTensorArray;
begin
  LoadTiny(G, M);
  S := NewSet(M, 6);
  try
    F := FixSample;
    F.LossFrom := 19;
    F.LossTo := 20;
    Want := RefBackward(M, S, F, 1, Ref);
    R := SampleLossBackward(M, F, 1, 256);
    AssertEquals('count 1', 1, R.Count);
    AssertTrue(Format('sum = Ref rel 1e-5, got %g vs %g', [R.Sum, Ref]),
      Abs(R.Sum - Ref) / Max(Abs(Ref), 1e-30) <= 1e-5);
    AssertTrue(Format('grads within 1e-5 of the reference, got %g', [RelDiff(Grads(S), Want)]),
      RelDiff(Grads(S), Want) <= 1e-5);
  finally
    M.Free;
    S.Free;
    G.Free;
  end;
end;

var
  App: TTestRunner;
begin
  TmpDir := GetTempDir + 'trainstep-ut-' + IntToStr(GetProcessID) + PathDelim;
  ForceDirectories(TmpDir);
  TinyPath := TmpDir + 'tiny.gguf';
  WriteTinyQwenModel(TinyPath, 5, thTied, False);
  RegisterTest(TTrainStepTest);
  App := TTestRunner.Create(nil);
  App.Initialize;
  App.Title := 'TrainStep UT';
  App.Run;
  App.Free;
  DeleteFile(TmpDir + 'valid.jsonl');
  DeleteFile(TmpDir + 'empty.jsonl');
  DeleteFile(TinyPath);
  RemoveDir(TmpDir);
end.
