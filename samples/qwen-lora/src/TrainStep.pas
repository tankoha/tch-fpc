{$mode objfpc}{$H+}
{ One training sample through the model (DESIGN 8.14 "Head", "One step"; T17).
  Purpose:
    The loss of a sample is the cross-entropy of the targets t + 1 in
    [LossFrom, LossTo) (TrainData), summed. With gradients it runs the trunk
    block-wise (Recompute: forward under NoGrad keeping the inputs, then
    block by block backward) and the head on slices of the loss rows only,
    each slice's sum divided by the caller's denominator and sent backward
    on its own; the parameter gradients accumulate in the LoRA leaves.
  Batches:
    The model runs one sequence per forward (decided 2026-11-24), so a
    batch is its samples one after the other, each divided by the mask
    count of the whole update.
  Interface by the harness owner (tasks/T17-facts.md); the bodies are
  written in T17b. }
unit TrainStep;

interface

uses
  SysUtils, Math, Tch, QwenModel, Recompute, TrainData;

type
  ETrainStep = class(Exception);

  TSampleLoss = record
    Sum: Double;       // the summed cross-entropy of the loss tokens
    Count: Integer;    // the loss tokens (LossTo - LossFrom)
  end;

{ Forward and backward of sample ASample: the loss rows (positions LossFrom - 1
  .. LossTo - 2) in slices of ASlice rows; every slice's sum divided by
  ADenom goes backward; then the trunk block-wise. Gradients accumulate.
  Raises ETrainStep when ADenom <= 0 or ASlice < 1. }
function SampleLossBackward(M: TQwenModel; const ASample: TTrainSample; ADenom: Double;
  ASlice: Integer): TSampleLoss;
{ The same loss without gradients (whole trunk under NoGrad). }
function SampleLoss(M: TQwenModel; const ASample: TTrainSample; ASlice: Integer): TSampleLoss;
{ The mean cross-entropy per loss token over every sample of ASet; NaN for
  an empty set. }
function ValidationLoss(M: TQwenModel; ASet: TTrainSet; ASlice: Integer): Double;

implementation

{ The targets of rows AFrom .. AFrom + ACount - 1: tokens AFrom + 1 ... }
function Targets(const ASample: TTrainSample; AFrom, ACount: Integer): TTensor;
var
  T: array of Int64;
  I: Integer;
begin
  T := nil;
  SetLength(T, ACount);
  for I := 0 to ACount - 1 do
    T[I] := ASample.Tokens[AFrom + I + 1];
  Result := TTensor.FromArray(T).view([-1, 1]);
end;

{ The summed cross-entropy of rows [AFrom, AFrom + ACount) of H. }
function SliceLoss(M: TQwenModel; const H: TTensor; const ASample: TTrainSample;
  AFrom, ACount: Integer): TTensor;
var
  Lg: TTensor;
begin
  Lg := M.Logits(H.narrow(0, AFrom, ACount));
  Result := -Lg.log_softmax(-1).gather(1, Targets(ASample, AFrom, ACount).&to(Lg.Device)).sum(kNone);
end;

procedure CheckSample(const ASample: TTrainSample; ASlice: Integer);
begin
  if ASlice < 1 then
    raise ETrainStep.CreateFmt('slice must be positive, got %d', [ASlice]);
  if (ASample.LossFrom < 1) or (ASample.LossTo <= ASample.LossFrom) or (ASample.LossTo > Length(ASample.Tokens)) then
    raise ETrainStep.CreateFmt('%s: loss span %d..%d outside 1..%d', [ASample.Id, ASample.LossFrom,
      ASample.LossTo, Length(ASample.Tokens)]);
end;

function SampleLossBackward(M: TQwenModel; const ASample: TTrainSample; ADenom: Double;
  ASlice: Integer): TSampleLoss;
var
  Saved: TSavedInputs;
  H, Hl, L: TTensor;
  P, N: Integer;
begin
  CheckSample(ASample, ASlice);
  if not (ADenom > 0) then
    raise ETrainStep.CreateFmt('denominator must be positive, got %g', [ADenom]);
  Result.Sum := 0;
  Result.Count := ASample.LossTo - ASample.LossFrom;
  Saved := TSavedInputs.Create(True);
  try
    H := RecomputeForward(M.Stages, M.Embed(ASample.Tokens), Saved);
    Hl := H.detach.requires_grad_(True);
    P := ASample.LossFrom - 1;
    while P < ASample.LossTo - 1 do
    begin
      N := Min(ASlice, ASample.LossTo - 1 - P);
      L := SliceLoss(M, Hl, ASample, P, N);
      Result.Sum := Result.Sum + L.DoubleValue([]);
      (L / ADenom).Backward;
      Inc(P, N);
    end;
    RecomputeBackward(M.Stages, Saved, Hl.grad);
  finally
    Saved.Free;
  end;
end;

function SampleLoss(M: TQwenModel; const ASample: TTrainSample; ASlice: Integer): TSampleLoss;
var
  G: IInterface;
  H: TTensor;
  P, N: Integer;
begin
  CheckSample(ASample, ASlice);
  G := NoGradGuard;
  Result.Sum := 0;
  Result.Count := ASample.LossTo - ASample.LossFrom;
  H := M.Trunk(M.Embed(ASample.Tokens));
  P := ASample.LossFrom - 1;
  while P < ASample.LossTo - 1 do
  begin
    N := Min(ASlice, ASample.LossTo - 1 - P);
    Result.Sum := Result.Sum + SliceLoss(M, H, ASample, P, N).DoubleValue([]);
    Inc(P, N);
  end;
end;

function ValidationLoss(M: TQwenModel; ASet: TTrainSet; ASlice: Integer): Double;
var
  I: Integer;
  R: TSampleLoss;
  Sum: Double;
  N: Int64;
begin
  if ASet.Count = 0 then
    Exit(NaN);
  Sum := 0;
  N := 0;
  for I := 0 to ASet.Count - 1 do
  begin
    R := SampleLoss(M, ASet[I], ASlice);
    Sum := Sum + R.Sum;
    Inc(N, R.Count);
  end;
  Result := Sum / N;
end;

end.
