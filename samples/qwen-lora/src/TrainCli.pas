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

const
  Usage = 'usage: qlora_train [options] MODEL TRAIN OUTDIR (see the unit header)';

function DefaultTrainConfig: TTrainConfig;
begin
  Result := Default(TTrainConfig);
  Result.Rank := 16;
  Result.Alpha := 32;
  Result.Lr := 1e-4;
  Result.MinLrRatio := 0.1;
  Result.Clip := 1.0;
  Result.Warmup := 50;
  Result.Epochs := 3;
  Result.Seed := 1;
  Result.Pool := 64;
  Result.Budget := 2048;
  Result.BatchesPerUpdate := 2;
  Result.CkptEvery := 50;
  Result.Keep := 3;
  Result.Slice := 256;
  Result.Classes := TStringArray.Create('compiled-run', 'fpc-approx');
  Result.Split := -1;
  Result.MaxUpdates := 0;
end;

function ParseTrainArgs(const AArgs: array of string; out C: TTrainConfig): string;
var
  I, N: Integer;
  Pos: array of string;
  A, V: string;
  X: Double;
  K: Int64;
  FS: TFormatSettings;

  function Int(const AName: string; ALo, AHi: Int64; out R: Int64): Boolean;
  begin
    Result := TryStrToInt64(V, R) and (R >= ALo) and (R <= AHi);
    if not Result then
      ParseTrainArgs := Format('%s: an integer in %d..%d, got "%s"', [AName, ALo, AHi, V]);
  end;

  function Num(const AName: string; ALo: Double; out R: Double): Boolean;
  begin
    Result := TryStrToFloat(V, R, FS) and not IsNan(R) and not IsInfinite(R) and (R >= ALo);
    if not Result then
      ParseTrainArgs := Format('%s: a number >= %g, got "%s"', [AName, ALo, V], FS);
  end;

begin
  FS := DefaultFormatSettings;
  FS.DecimalSeparator := '.';
  C := DefaultTrainConfig;
  Result := '';
  Pos := nil;
  I := 0;
  N := Length(AArgs);
  while I < N do
  begin
    A := AArgs[I];
    if A = '--resume' then
    begin
      C.Resume := True;
      Inc(I);
      Continue;
    end;
    if Copy(A, 1, 2) <> '--' then
    begin
      SetLength(Pos, Length(Pos) + 1);
      Pos[High(Pos)] := A;
      Inc(I);
      Continue;
    end;
    if I + 1 >= N then
      Exit(A + ': value missing');
    V := AArgs[I + 1];
    Inc(I, 2);
    if A = '--valid' then C.Valid := V
    else if A = '--base-sha256' then C.BaseSha256 := V
    else if A = '--classes' then C.Classes := V.Split([','])
    else if A = '--rank' then begin if not Int(A, 1, 4096, K) then Exit; C.Rank := K; end
    else if A = '--warmup' then begin if not Int(A, 0, MaxInt, K) then Exit; C.Warmup := K; end
    else if A = '--epochs' then begin if not Int(A, 1, 1000, K) then Exit; C.Epochs := K; end
    else if A = '--seed' then begin if not Int(A, -MaxInt, MaxInt, K) then Exit; C.Seed := K; end
    else if A = '--pool' then begin if not Int(A, 1, MaxInt, K) then Exit; C.Pool := K; end
    else if A = '--budget' then begin if not Int(A, 1, MaxInt, K) then Exit; C.Budget := K; end
    else if A = '--batches-per-update' then begin if not Int(A, 1, 1000, K) then Exit; C.BatchesPerUpdate := K; end
    else if A = '--ckpt-every' then begin if not Int(A, 1, MaxInt, K) then Exit; C.CkptEvery := K; end
    else if A = '--keep' then begin if not Int(A, 1, 1000, K) then Exit; C.Keep := K; end
    else if A = '--slice' then begin if not Int(A, 1, MaxInt, K) then Exit; C.Slice := K; end
    else if A = '--split' then begin if not Int(A, 0, 1000, K) then Exit; C.Split := K; end
    else if A = '--max-updates' then begin if not Int(A, 0, High(Int64), K) then Exit; C.MaxUpdates := K; end
    else if A = '--alpha' then begin if not Num(A, 0, X) then Exit; C.Alpha := X; end
    else if A = '--lr' then begin if not Num(A, 0, X) then Exit; C.Lr := X; end
    else if A = '--min-lr-ratio' then begin if not Num(A, 0, X) then Exit; C.MinLrRatio := X; end
    else if A = '--clip' then begin if not Num(A, 0, X) then Exit; C.Clip := X; end
    else
      Exit('unknown option ' + A);
  end;
  if Length(Pos) <> 3 then
    Exit(Format('expected MODEL TRAIN OUTDIR, got %d arguments', [Length(Pos)]));
  C.Model := Pos[0];
  C.Train := Pos[1];
  C.OutDir := IncludeTrailingPathDelimiter(Pos[2]);
  if C.BaseSha256 = '' then
    C.BaseSha256 := ShaFromName(C.Model);
end;

function ShaFromName(const AFile: string): string;
var
  N: string;
  I: Integer;
begin
  Result := '';
  N := ExtractFileName(AFile);
  if (Length(N) <> 71) or (Copy(N, 1, 7) <> 'sha256-') then
    Exit;
  for I := 8 to 71 do
    if not (N[I] in ['0'..'9', 'a'..'f']) then
      Exit;
  Result := Copy(N, 8, 64);
end;

function ConfigHash(const C: TTrainConfig): string;
var
  T: string;
  FS: TFormatSettings;
begin
  FS := DefaultFormatSettings;
  FS.DecimalSeparator := '.';
  T := 'train=' + MD5Print(MD5File(C.Train));
  if C.Valid <> '' then
    T := T + ' valid=' + MD5Print(MD5File(C.Valid));
  T := T + Format(' rank=%d alpha=%g lr=%g warmup=%d min=%g clip=%g epochs=%d seed=%d pool=%d budget=%d bpu=%d slice=%d classes=%s',
    [C.Rank, C.Alpha, C.Lr, C.Warmup, C.MinLrRatio, C.Clip, C.Epochs, C.Seed, C.Pool,
     C.Budget, C.BatchesPerUpdate, C.Slice, string.Join(',', C.Classes)], FS);
  Result := MD5Print(MD5String(T));
end;

function TotalUpdates(ASet: TTrainSet; const C: TTrainConfig): Int64;
var
  E: Integer;
begin
  Result := 0;
  for E := 0 to C.Epochs - 1 do
    Inc(Result, (Length(EpochBatches(ASet, C.Seed, E, C.Pool, C.Budget)) + C.BatchesPerUpdate - 1)
      div C.BatchesPerUpdate);
end;

procedure AppendLine(const AFile, ALine: string);
var
  F: TextFile;
begin
  AssignFile(F, AFile);
  if FileExists(AFile) then
    Append(F)
  else
    Rewrite(F);
  WriteLn(F, ALine);
  CloseFile(F);
end;

function G17(X: Double): string;
var
  FS: TFormatSettings;
begin
  FS := DefaultFormatSettings;
  FS.DecimalSeparator := '.';
  if IsNan(X) then
    Result := 'nan'
  else
    Result := FloatToStrF(X, ffGeneral, 17, 0, FS);
end;

function RunTrainCLI(const AArgs: array of string; AOut, AErr: TStrings): Integer;
begin
  Result := 0;
  raise Exception.Create('RunTrainCLI: not implemented yet');
end;

end.
