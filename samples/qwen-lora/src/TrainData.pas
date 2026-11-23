{$mode objfpc}{$H+}
{ Training data of a LoRA run (DESIGN 8.14 "Batches", "Loss mask", 8.16; T15).
  Records:
    A data set file is JSONL, one record per line with the string keys id,
    dialect, prompt, answer and class (the verification class of 8.16);
    other keys are ignored. Text is raw UTF-8: fpjson turns a \u escape of a
    non-ASCII character into '?', so such an escape is refused.
  Samples:
    A record becomes the conversation user = prompt, assistant = answer,
    rendered without thinking (QwenChat, EnableThinking False: the empty
    think block before the answer) and tokenised; the loss covers the
    answer and its closing im_end (EncodeTraining's [LossFrom, LossTo)).
  Batches:
    An epoch's order is a permutation of the samples from (seed, epoch); it
    is cut into pools, each pool sorted by length and cut into batches of
    at most Budget positions (count times the longest), and the batches of a
    pool are shuffled. Everything is a function of (seed, epoch), so a run
    resumes by skipping the batches of the samples already consumed.
  Interface by the harness owner (tasks/T15-facts.md); the bodies are
  written in T15b. }
unit TrainData;

interface

uses
  SysUtils, Classes, Math, fpjson, jsonparser, Tch, QwenTokenizer, QwenChat;

type
  ETrainData = class(Exception);

  TIntArray = array of Integer;

  TTrainSample = record
    Id: string;
    Dialect: string;
    Tokens: TTokenArray;
    LossFrom, LossTo: Integer;   // [LossFrom, LossTo) of Tokens: the answer and its im_end
  end;

  TBatch = TIntArray;            // indices into the set's samples
  TBatches = array of TBatch;

  TTrainSet = class
  private
    FSamples: array of TTrainSample;
    FSkippedClass, FSkippedLong: Integer;
    function GetSample(I: Integer): TTrainSample;
  public
    { Reads AFile; admits the records whose class is in AClasses (compared
      literally); drops the samples longer than AMaxLen tokens (counted).
      Raises ETrainData. }
    constructor Create(Tok: TQwenTokenizer; const AFile: string;
      const AClasses: array of string; AMaxLen: Integer);
    function Count: Integer;
    { The length of the longest sample (0 for none). }
    function MaxLength: Integer;
    property Samples[I: Integer]: TTrainSample read GetSample; default;
    property SkippedClass: Integer read FSkippedClass;
    property SkippedLong: Integer read FSkippedLong;
  end;

{ The tokens of the conversation user APrompt, assistant AAnswer, rendered
  without thinking; [LossFrom, LossTo) is the answer and its im_end. }
function EncodeSample(Tok: TQwenTokenizer; const APrompt, AAnswer: RawByteString;
  out LossFrom, LossTo: Integer): TTokenArray;
{ A permutation of 0 .. N - 1 from (ASeed, AEpoch) (Fisher-Yates, SplitMix64). }
function EpochOrder(N: Integer; ASeed, AEpoch: Int64): TIntArray;
{ The batches of epoch AEpoch. Raises ETrainData when APool < 1 or ABudget
  is below the longest sample. }
function EpochBatches(ASet: TTrainSet; ASeed, AEpoch: Int64; APool, ABudget: Integer): TBatches;
{ The batches after the first APosition samples of AB (APosition must end a
  batch; 0 gives AB). Raises ETrainData. }
function SkipBatches(const AB: TBatches; APosition: Int64): TBatches;
{ The samples of a batch: AIds [B, T] (int64, right-padded with APad), ATargets
  [B, T] (int64; AIds shifted left by one, APad at the end), AMask [B, T]
  (float; 1 where the target is a loss token: t + 1 in [LossFrom, LossTo)).
  T is the longest sample. Raises ETrainData for an empty batch. }
procedure BatchTensors(ASet: TTrainSet; const ABatch: TBatch; APad: Integer;
  out AIds, ATargets, AMask: TTensor);

implementation

function TTrainSet.GetSample(I: Integer): TTrainSample;
begin
  Result := Default(TTrainSample);
  raise ETrainData.Create('TTrainSet.GetSample: not implemented yet');
end;

constructor TTrainSet.Create(Tok: TQwenTokenizer; const AFile: string;
  const AClasses: array of string; AMaxLen: Integer);
begin
  inherited Create;
  raise ETrainData.Create('TTrainSet.Create: not implemented yet');
end;

function TTrainSet.Count: Integer;
begin
  Result := 0;
  raise ETrainData.Create('TTrainSet.Count: not implemented yet');
end;

function TTrainSet.MaxLength: Integer;
begin
  Result := 0;
  raise ETrainData.Create('TTrainSet.MaxLength: not implemented yet');
end;

function EncodeSample(Tok: TQwenTokenizer; const APrompt, AAnswer: RawByteString;
  out LossFrom, LossTo: Integer): TTokenArray;
begin
  Result := nil;
  raise ETrainData.Create('EncodeSample: not implemented yet');
end;

function EpochOrder(N: Integer; ASeed, AEpoch: Int64): TIntArray;
begin
  Result := nil;
  raise ETrainData.Create('EpochOrder: not implemented yet');
end;

function EpochBatches(ASet: TTrainSet; ASeed, AEpoch: Int64; APool, ABudget: Integer): TBatches;
begin
  Result := nil;
  raise ETrainData.Create('EpochBatches: not implemented yet');
end;

function SkipBatches(const AB: TBatches; APosition: Int64): TBatches;
begin
  Result := nil;
  raise ETrainData.Create('SkipBatches: not implemented yet');
end;

procedure BatchTensors(ASet: TTrainSet; const ABatch: TBatch; APad: Integer;
  out AIds, ATargets, AMask: TTensor);
begin
  raise ETrainData.Create('BatchTensors: not implemented yet');
end;

end.
