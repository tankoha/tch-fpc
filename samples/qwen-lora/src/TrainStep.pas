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

function SampleLossBackward(M: TQwenModel; const ASample: TTrainSample; ADenom: Double;
  ASlice: Integer): TSampleLoss;
begin
  Result := Default(TSampleLoss);
  raise ETrainStep.Create('SampleLossBackward: not implemented yet');
end;

function SampleLoss(M: TQwenModel; const ASample: TTrainSample; ASlice: Integer): TSampleLoss;
begin
  Result := Default(TSampleLoss);
  raise ETrainStep.Create('SampleLoss: not implemented yet');
end;

function ValidationLoss(M: TQwenModel; ASet: TTrainSet; ASlice: Integer): Double;
begin
  Result := 0;
  raise ETrainStep.Create('ValidationLoss: not implemented yet');
end;

end.
