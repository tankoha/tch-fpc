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
var
  C: TTrainConfig;
  Msg, Hash, LogFile, Dir: string;
  G: TGgufFile;
  Tok: TQwenTokenizer;
  M: TQwenModel;
  TrainSet, ValidSet: TTrainSet;
  S: TLoraSet;
  O: TLoraAdamW;
  Ck: TCheckpoints;
  St: TRunState;
  B: TBatches;
  Total, U: Int64;
  Epoch, I, J, K: Integer;
  Pos, Tokens: Int64;
  Denom, Sum, Loss, Norm, Lr, Val: Double;
  R: TSampleLoss;
  T0: TDateTime;
  EndOfEpoch: Boolean;
begin
  Msg := ParseTrainArgs(AArgs, C);
  if Msg <> '' then
  begin
    AErr.Add(Msg);
    AErr.Add(Usage);
    Exit(2);
  end;
  G := nil;
  Tok := nil;
  M := nil;
  TrainSet := nil;
  ValidSet := nil;
  S := nil;
  O := nil;
  Ck := nil;
  try
    try
      if not FileExists(C.Model) then
        raise Exception.Create('no model file ' + C.Model);
      G := TGgufFile.Open(C.Model);
      Tok := TQwenTokenizer.Create(G);
      TrainSet := TTrainSet.Create(Tok, C.Train, C.Classes, C.Budget);
      if TrainSet.Count = 0 then
        raise Exception.Create('no training sample in ' + C.Train);
      if C.Valid <> '' then
        ValidSet := TTrainSet.Create(Tok, C.Valid, C.Classes, C.Budget);
      Total := TotalUpdates(TrainSet, C);
      if C.Warmup >= Total then
        raise Exception.CreateFmt('warm-up %d must be below the %d updates', [C.Warmup, Total]);
      if C.Split < 0 then
        M := TQwenModel.Load(G, TDevice.Cpu, TDevice.Cpu, 0)
      else
        M := TQwenModel.Load(G, TDevice.Cuda(0), TDevice.Cuda(1), C.Split);
      ManualSeed(C.Seed);
      S := TLoraSet.Create(C.Rank, C.Alpha / C.Rank);
      M.AttachLora(S);
      O := TLoraAdamW.Create(S.Stores, DefaultAdamWConfig);
      ForceDirectories(C.OutDir);
      Ck := TCheckpoints.Create(C.OutDir + 'ckpt', C.Keep);
      Hash := ConfigHash(C);
      LogFile := C.OutDir + 'train-log.tsv';
      Epoch := 0;
      Pos := 0;
      U := 0;
      if C.Resume then
      begin
        Dir := Ck.Latest;
        if Dir = '' then
          raise Exception.Create('nothing to resume in ' + Ck.Root);
        St := Ck.Load(Dir, S, O);
        if St.ConfigHash <> Hash then
          raise Exception.Create('the configuration differs from the checkpoint''s: ' + Dir);
        if St.BaseSha256 <> C.BaseSha256 then
          raise Exception.Create('the base model differs from the checkpoint''s: ' + Dir);
        Epoch := St.Epoch;
        Pos := St.Position;
        U := St.Update;
        AOut.Add(Format('resumed from %s at update %d', [Dir, U]));
      end;
      while Epoch < C.Epochs do
      begin
        B := SkipBatches(EpochBatches(TrainSet, C.Seed, Epoch, C.Pool, C.Budget), Pos);
        I := 0;
        while I < Length(B) do
        begin
          T0 := Now;
          Denom := 0;
          for J := I to Min(I + C.BatchesPerUpdate, Length(B)) - 1 do
            for K := 0 to High(B[J]) do
              Denom := Denom + TrainSet[B[J][K]].LossTo - TrainSet[B[J][K]].LossFrom;
          Sum := 0;
          Tokens := 0;
          for J := I to Min(I + C.BatchesPerUpdate, Length(B)) - 1 do
            for K := 0 to High(B[J]) do
            begin
              R := SampleLossBackward(M, TrainSet[B[J][K]], Denom, C.Slice);
              Sum := Sum + R.Sum;
              Inc(Tokens, Length(TrainSet[B[J][K]].Tokens));
              Inc(Pos);
            end;
          Loss := Sum / Denom;
          if IsNan(Loss) or IsInfinite(Loss) then
          begin
            AErr.Add(Format('non-finite loss at update %d; stopped without updating', [U + 1]));
            AppendLine(LogFile, Format('%d'#9'%d'#9'%d'#9'%s', [U + 1, Epoch, Pos, G17(Loss)]));
            Exit(3);
          end;
          Inc(U);
          Norm := O.ClipGlobalNorm(C.Clip);
          Lr := ScheduleLR(C.Lr, U, C.Warmup, Total, C.MinLrRatio);
          O.Step(Lr);
          O.ZeroGrad;
          Inc(I, C.BatchesPerUpdate);
          EndOfEpoch := I >= Length(B);
          AppendLine(LogFile, Format('%d'#9'%d'#9'%d'#9'%s'#9'%s'#9'%s'#9'%d'#9'%.1f',
            [U, Epoch, Pos, G17(Loss), G17(Lr), G17(Norm), Tokens, SecondSpan(Now, T0)]));
          if EndOfEpoch or (U mod C.CkptEvery = 0) then
          begin
            Val := NaN;
            if ValidSet <> nil then
              Val := ValidationLoss(M, ValidSet, C.Slice);
            St := NewRunState;
            St.Update := U;
            { at an epoch's end the state is the next epoch's start }
            St.Epoch := Epoch + Ord(EndOfEpoch);
            if EndOfEpoch then
              St.Position := 0
            else
              St.Position := Pos;
            St.Seed := C.Seed;
            St.ConfigHash := Hash;
            St.BaseSha256 := C.BaseSha256;
            St.ValLoss := Val;
            Dir := Ck.Save(St, S, O);
            AppendLine(LogFile, Format('ckpt'#9'%d'#9'%s'#9'%s', [U, G17(Val), Dir]));
          end;
          if (C.MaxUpdates > 0) and (U >= C.MaxUpdates) then
          begin
            AOut.Add(Format('stopped after update %d of %d', [U, Total]));
            Exit(0);
          end;
        end;
        Inc(Epoch);
        Pos := 0;
      end;
      AOut.Add(Format('done: %d updates, best %s', [U, Ck.Best]));
      Result := 0;
    except
      on E: Exception do
      begin
        AErr.Add(E.Message);
        Result := 1;
      end;
    end;
  finally
    Ck.Free;
    O.Free;
    M.Free;
    S.Free;
    ValidSet.Free;
    TrainSet.Free;
    Tok.Free;
    G.Free;
  end;
end;

end.
