{$mode objfpc}{$H+}
{ qlora_train: LoRA training on the GGUF's packed weights (DESIGN 8.14; T17).
  Command line:
    qlora_train [options] MODEL TRAIN OUTDIR
    --valid FILE     validation records (TrainData format); the loss at
                     every checkpoint
    --rank N (16)  --alpha A (32: scale A / N)  --lr X (1e-4)
    --warmup N (50)  --min-lr-ratio X (0.1)  --epochs N (3)  --seed N (1)
    --pool N (64)  --budget N (2048)  --batches-per-update N (2)
    --clip X (1.0)  --ckpt-every N (50)  --keep N (3)  --slice N (256)
    --classes A,B (compiled-run,fpc-approx)
    --split S        blocks 0..S-1 on cuda:0, the rest and the head on
                     cuda:1 (default: all on the CPU)
    --base-sha256 H  the base GGUF's sha256 (default: from a file name
                     sha256-<64 hex>, else empty)
    --max-updates N  stop after update N (0: run to the end)
    --resume         continue from OUTDIR/ckpt's latest checkpoint
  Run:
    One update is --batches-per-update consecutive batches of an epoch
    (TrainData), their samples one by one (TrainStep), every sample's loss
    divided by the update's loss-token count; then ClipGlobalNorm, AdamW
    at ScheduleLR(lr, update, warmup, total, min ratio), ZeroGrad. A
    checkpoint (Checkpoint, OUTDIR/ckpt) after every --ckpt-every updates
    and at the end of each epoch, with the validation loss. A non-finite
    loss stops the run before its update.
  Output:
    OUTDIR/train-log.tsv, appended, one line per update: update, epoch,
    position, loss, lr, grad norm, tokens, seconds; one line per
    checkpoint: 'ckpt', update, validation loss, directory.
  Exit codes: 0 done (or --max-updates reached), 1 error, 2 usage,
    3 non-finite loss.
  Interface by the harness owner (tasks/T17-facts.md); the bodies are
  written in T17b. }
unit TrainCli;

interface

uses
  Classes, SysUtils, Math, DateUtils, md5, Tch, GgufFile, QwenTokenizer, QwenModel, Lora,
  LoraOptim, Checkpoint, TrainData, TrainStep;

type
  TTrainConfig = record
    Model, Train, OutDir, Valid: string;
    Rank: Integer;
    Alpha, Lr, MinLrRatio, Clip: Double;
    Warmup, Epochs, Seed, Pool, Budget, BatchesPerUpdate, CkptEvery, Keep, Slice: Integer;
    Classes: TStringArray;
    Split: Integer;            // -1: CPU
    BaseSha256: string;
    MaxUpdates: Int64;
    Resume: Boolean;
  end;

function DefaultTrainConfig: TTrainConfig;
{ '' when AArgs are valid, else the usage error. }
function ParseTrainArgs(const AArgs: array of string; out C: TTrainConfig): string;
{ MD5 (hex) of the settings that must not change across a resume: the
  contents of TRAIN and VALID, rank, alpha, lr, warm-up, min ratio, clip,
  epochs, seed, pool, budget, batches per update, slice, classes. }
function ConfigHash(const C: TTrainConfig): string;
{ The updates of all epochs: per epoch ceil(batches / batches per update). }
function TotalUpdates(ASet: TTrainSet; const C: TTrainConfig): Int64;
{ The base's sha256 from a file name sha256-<64 hex>, else ''. }
function ShaFromName(const AFile: string): string;
{ AOut / AErr are filled (caller-created, caller-owned); the result is the
  exit code. }
function RunTrainCLI(const AArgs: array of string; AOut, AErr: TStrings): Integer;

implementation

function DefaultTrainConfig: TTrainConfig;
begin
  Result := Default(TTrainConfig);
  raise Exception.Create('DefaultTrainConfig: not implemented yet');
end;

function ParseTrainArgs(const AArgs: array of string; out C: TTrainConfig): string;
begin
  Result := '';
  raise Exception.Create('ParseTrainArgs: not implemented yet');
end;

function ShaFromName(const AFile: string): string;
begin
  Result := '';
  raise Exception.Create('ShaFromName: not implemented yet');
end;

function ConfigHash(const C: TTrainConfig): string;
begin
  Result := '';
  raise Exception.Create('ConfigHash: not implemented yet');
end;

function TotalUpdates(ASet: TTrainSet; const C: TTrainConfig): Int64;
begin
  Result := 0;
  raise Exception.Create('TotalUpdates: not implemented yet');
end;

function RunTrainCLI(const AArgs: array of string; AOut, AErr: TStrings): Integer;
begin
  Result := 0;
  raise Exception.Create('RunTrainCLI: not implemented yet');
end;

end.
