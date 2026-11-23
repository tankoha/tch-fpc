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
