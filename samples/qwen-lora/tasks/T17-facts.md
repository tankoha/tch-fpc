# T17 facts (checked by the harness owner)

T17 is the training program `qlora_train` (DESIGN 8.14 "Head", "One step",
8.11 M2, 8.19 R7): the unit `TrainStep` (one sample's loss and gradient,
the validation loss), the unit `TrainCli` (the command line, the loop over
epochs and updates, the log, checkpoints and resume) and the program
`src/qlora_train.pas` (the harness owner's thin wrapper, complete; like
`qlora_logprobs`). It uses everything of M1 and M2 so far: `QwenModel`
(T11), `Lora` (T12), `Recompute` (T13), `LoraOptim` (T14), `TrainData`
(T15), `Checkpoint` (T16).

DESIGN 8.18's acceptance for T17: "on the rehearsal model: overfits 32
samples; validation loss falls on a small real set". Decisions taken by
the harness owner for this task (DESIGN 8.14 leaves them open):
- **One sample per forward** (decided by the user 2026-11-24): a batch of
  TrainData is its samples one after the other; every sample's loss is
  divided by the loss-token count of the whole update, so the gradient is
  that of the update's mean loss.
- **One sample** (`SampleLossBackward`): the trunk block-wise
  (`RecomputeForward` with the inputs on the host, then
  `RecomputeBackward`); the head only on the loss rows (positions
  `LossFrom - 1 .. LossTo - 2` predict tokens `LossFrom .. LossTo - 1`), in
  slices of `Slice` rows (256), each slice's summed cross-entropy divided
  by the denominator and sent backward on its own, then the trunk. The
  validation loss (`SampleLoss`, `ValidationLoss`) runs the whole trunk
  under NoGrad.
- **One update**: `--batches-per-update` consecutive batches of the
  epoch (default 2, DESIGN 8.14), then `ClipGlobalNorm(clip)`,
  `Step(ScheduleLR(lr, update, warm-up, total, min ratio))`, `ZeroGrad`.
  The total is the sum over the epochs of ceil(batches / batches per
  update) (TrainData's batches differ slightly between epochs). A
  non-finite loss stops the run before its update, exit code 3 (R7 says
  halve the rate and go on from the last good checkpoint; that is the
  user's decision at M4, not the program's).
- **Checkpoints** (T16) in `OUTDIR/ckpt` after every `--ckpt-every`
  updates and at the end of each epoch, with the validation loss when
  `--valid` is given (else NaN). At an epoch's end the state is the next
  epoch's start (epoch + 1, position 0). `--resume` continues from the
  latest checkpoint and refuses another configuration (`ConfigHash`: MD5
  of the train and valid files' contents and the settings that shape the
  run; not `--max-updates`, `--resume`, `--split`, `--ckpt-every`,
  `--keep`, the output directory or the base's sha256) or another base
  sha256.
- **The base's sha256**: stock FPC 3.2.2 has no sha256 (`fpsha256` is in
  FPC 3.3.1 only, measured), so it is not computed: `--base-sha256`, or
  the name of an Ollama blob `sha256-<64 hex>` (the 27B is one), else
  empty.
- **The log** `OUTDIR/train-log.tsv` is appended line by line during the
  run (a run takes hours; the output strings come at the end only).
- **Seeds**: `ManualSeed(seed)` before `AttachLora` (the A matrices); the
  epoch order is TrainData's SplitMix64 from the same seed.
- **Measured on the rehearsal model on the CPU**: one sample of T tokens
  forward and backward takes about 2 s + 6.4 ms × T (the 2 s are the
  dequantisation, paid twice: the forward and the block-wise recompute);
  8 short samples (about 80 tokens) with lr 1e-3 overfit to 3.7 % of the
  first loss in 8 updates (3.40 to 0.13), the validation loss falling
  from 2.20 to 1.80 at update 4 and rising afterwards. The M2 exit (32
  samples below 10 %) is the harness owner's run, not the ITa (it takes
  about half an hour).

Measured by the harness owner with a prototype written exactly as below:
- UT: every row of the tables below passes; both units then have 100 %
  line coverage; mutants are caught (the slice loss not divided by the denominator: TrainStep 002, 004; the loss rows shifted by one: TrainStep 002, 003, 009; the update's denominator over all tokens: TrainCli 203; no ZeroGrad after the step: 204, 205; the position not restored on resume: 204; the configuration hash not checked: 206; a non-finite loss not stopped: 207; the schedule's total one too large: 203; the epoch-end state with the old epoch: 202, 205; with the old position: 202, 205).
- ITa on the rehearsal model: 0.0896 (3.397 to 0.304 in 8 updates), 3.2e-9, 0.745 (the best validation loss 1.837 at update 2, 2.466 without adapters), 4.7e-9, 0 (ckpt-00000010), 0 (cases below).

These facts are final; do not investigate further or write probes.

## Unit `TrainStep` (interface, final; in the skeleton)

```pascal
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
```

## Implementation of `TrainStep` (write it exactly so)

The local routines (`Targets`, `SliceLoss`, `CheckSample`) are not in the
interface; write them in the implementation section where shown.

```pascal
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
```

## Unit `TrainCli` (interface, final; in the skeleton)

The unit header in the skeleton documents the command line, the run and
the output; keep it.

```pascal
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
```

## Implementation of `TrainCli` (write it exactly so)

The local constant `Usage` and routines (`AppendLine`, `G17`, and `Int`,
`Num` nested in `ParseTrainArgs`) are not in the interface; write them
where shown.

```pascal
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
```

## The program `src/qlora_train.pas` (harness owner's; complete)

It forwards the command line to `RunTrainCLI`, prints the output lines to
stdout and the error lines to stderr, and exits with the code. Nobody
changes it.

## Checks and messages (in this order)

`ETrainStep` (unit `TrainStep`), from both `SampleLossBackward` and
`SampleLoss` (through `CheckSample`, first): `CreateFmt('slice must be
positive, got %d', [ASlice])`; `CreateFmt('%s: loss span %d..%d outside
1..%d', [Id, LossFrom, LossTo, Length(Tokens)])` when LossFrom < 1, LossTo
<= LossFrom or LossTo > the length; then in `SampleLossBackward` only
`CreateFmt('denominator must be positive, got %g', [ADenom])`.

`ParseTrainArgs` (the usage error; `RunTrainCLI` prints it and the usage
line, exit 2): `<option>: value missing`; `<option>: an integer in
<lo>..<hi>, got "<v>"` (ranges: `--rank` 1..4096, `--warmup` 0..MaxInt,
`--epochs` 1..1000, `--seed` −MaxInt..MaxInt, `--pool`, `--budget`,
`--ckpt-every`, `--slice` 1..MaxInt, `--batches-per-update`, `--keep`
1..1000, `--split` 0..1000, `--max-updates` 0..High(Int64)); `<option>: a
number >= 0, got "<v>"` (`--alpha`, `--lr`, `--min-lr-ratio`, `--clip`;
NaN and ±Inf refused); `unknown option <option>`; then `expected MODEL
TRAIN OUTDIR, got <n> arguments`.

`RunTrainCLI` (exit 1, the message alone on AErr), in this order: `no
model file <model>`; from the GGUF, the tokenizer and TrainData (e.g. `no
data file <train>`); `no training sample in <train>`; `warm-up <w> must be
below the <n> updates`; from `TQwenModel.Load` (e.g. `No CUDA GPUs are
available` with `--split` and the GPUs hidden; `split 3 outside 0..2`);
with `--resume`: `nothing to resume in <outdir>ckpt/`, `the configuration
differs from the checkpoint's: <dir>`, `the base model differs from the
checkpoint's: <dir>`. A non-finite loss: AErr `non-finite loss at update
<u>; stopped without updating`, a log line `<u> <epoch> <position> nan`,
exit 3.

## UT fixture and helpers (harness owner's; nobody changes them)

The tiny model of T10/T11 (`tests/ut/QwenBlocksFixture.pas`,
`WriteTinyQwenModel(Path, 5, thTied, False)`): two blocks (linear, full),
hidden 64, the tiny tokenizer of T05 in its metadata.

`tests/ut/TrainStep_test.pas` (skeleton: helpers, runner, no test methods):
- `LoadTiny(G, M)` (the tiny model on the CPU; free M, then G);
  `NewSet(M, Seed)` (rank 4, scale 2 attached to M after `ManualSeed(3)`;
  every B := 0.1 N(0, 1) after `ManualSeed(Seed)`, B zero for Seed < 0);
  `FixSample` (tokens (7 i + 3) mod 270, i = 0..19, span 12..19, id `x`);
  `RefLoss(M, S)` (the summed cross-entropy of the span through the whole
  graph, with a graph); `Grads(S)` (copies of every A grad, then every B
  grad); `ZeroGrads(S)`; `RelDiff(A, B)` (max over the tensors of max |a −
  b| / max(max |b|, 1e-30)); `CheckStep(Self, M, Sample, Denom, Slice,
  Backward, Message)` (ETrainStep with Message); `RefBackward(M, S,
  Sample, Denom, Value)` (the reference loss's value and the grads of
  `(Ref / Denom).Backward`, the grads zeroed before and after).

`tests/ut/TrainCli_test.pas` (skeleton: helpers, runner, no test methods).
The data: `TrainFile` (8 records `t0`..`t7`, prompt `hello I`, answer
`world` + I + 1 `a`, compiled-run), `ValidFile` (1 record `v0`),
`OtherFile` (1 record of class static-only):
- `RunArgs(Out, Extra)`: `--valid ValidFile --epochs 2 --warmup 1 --lr
  1e-2 --budget 120 --pool 4 --batches-per-update 1 --ckpt-every 3 --rank
  4 --alpha 8`, then Extra, then `TinyPath TrainFile Out` (4 batches per
  epoch, 8 updates);
- `Run(Args, Out, Err)` (the exit code; the lines joined by #10);
  `NewOut` (a fresh directory, not created, ending in the delimiter);
  `UpdateLines(Out)` (the update lines of the log without the seconds
  column); `Steps(Out)` (`update epoch position` of every update line,
  joined by `|`); `CkptUpdates(Out)` (the updates of the ckpt lines);
  `ReadBack(Dir, State)` (every A, B and moment of a checkpoint read into a
  fresh set of rank 4, scale 2 on a fresh tiny model); `DiffArr`;
  `CheckParse(Self, Args, Message)`.

## Unit tests (rows; every row has one method, every method belongs to a row)

Error rows compare the whole message (`AssertEquals`); "exact" means
equal bits or a difference of 0; "rel x" means a relative difference of
at most x.

`TrainStep`, stage 1 (task T17b1); 001–009. M, G := `LoadTiny`; S :=
`NewSet(M, 6)` unless said otherwise; free M, then S, then G.

| Row | Input | Expected |
|---|---|---|
| 001 | `SampleLoss(M, FixSample, 3)` | Count 7; Sum = `RefLoss(M, FixSample)` rel 1e-5 |
| 002 | Want := `RefBackward(M, S, FixSample, 3, Ref)`; `SampleLossBackward(M, FixSample, 3, 3)` | Count 7; Sum = Ref rel 1e-5; `RelDiff(Grads(S), Want)` ≤ 1e-5; the B grads of Want not zero (max > 0.1) |
| 003 | Want := `RefBackward(.., 1, ..)`; slices 1, 3, 7, 100 (grads zeroed between) | for each: Sum = Ref rel 1e-5; grads within 1e-5 of Want |
| 004 | denominator 3; then 6; then 6 again without zeroing | grads at 6 × 2 within 1e-6 of those at 3; after the second call at 6 the grads equal those at 3 (rel 1e-6) |
| 005 | S := `NewSet(M, -1)` (B zero); M0 a second model without adapters | `SampleLoss(M0, ..).Sum` = `SampleLoss(M, ..).Sum` exact |
| 006 | `SampleLoss(M, FixSample, 3)` | no A or B grad defined afterwards |
| 007 | `CheckStep` without a set: slice 0 (both); denominator 0, −2; LossFrom 0; LossTo 12; LossTo 21 | `slice must be positive, got 0` (twice); `denominator must be positive, got 0`, `.. got -2`; `x: loss span 0..19 outside 1..20`; `x: loss span 12..12 outside 1..20`; `x: loss span 12..21 outside 1..20` |
| 008 | a TTrainSet (the tiny model's tokenizer) of three records (`hello`/`world`, `ab`/`abc abc`, `x`/`y`, class `c`); `ValidationLoss(M, V, 2)`; then an empty file | Count 3; = Σ `SampleLoss(.., 256)`.Sum / Σ Count, rel 1e-5; empty: NaN |
| 009 | FixSample with span 19..20 (the last token): `RefBackward(.., 1, ..)`, `SampleLossBackward(.., 1, 256)` | Count 1; Sum = Ref rel 1e-5; grads within 1e-5 |

`TrainCli`, stage 2 (task T17b2): `DefaultTrainConfig`, `ParseTrainArgs`,
`ShaFromName`, `ConfigHash`, `TotalUpdates`; 101–108.

| Row | Input | Expected |
|---|---|---|
| 101 | `DefaultTrainConfig` | rank 16, alpha 32, lr 1e-4, min ratio 0.1, clip 1, warm-up 50, epochs 3, seed 1, pool 64, budget 2048, batches per update 2, ckpt every 50, keep 3, slice 256, classes `compiled-run,fpc-approx`, split −1, max updates 0, resume False, the five strings '' |
| 102 | `['m.gguf', 't.jsonl', 'out']` | '' ; Model `m.gguf`, Train `t.jsonl`, OutDir `out` + delimiter, Valid '', BaseSha256 '', rank 16, resume False |
| 103 | every option: `--rank 8 --alpha 16 --lr 0.002 --warmup 5 --min-lr-ratio 0.2 --epochs 4 --seed -7 --pool 16 --budget 512 --batches-per-update 3 --clip 0.5 --ckpt-every 10 --keep 2 --slice 64 --classes a,b,c --split 12 --base-sha256 abc --max-updates 9 --resume --valid v.jsonl m t o/` | '' ; each field as given; OutDir `o/` (one delimiter) |
| 104 | `--frob 1`; `--rank` last; `--rank 0`; `--epochs x`; `--lr -1`; `--clip nan`; two and four positionals | `unknown option --frob`; `--rank: value missing`; `--rank: an integer in 1..4096, got "0"`; `--epochs: an integer in 1..1000, got "x"`; `--lr: a number >= 0, got "-1"`; `--clip: a number >= 0, got "nan"`; `expected MODEL TRAIN OUTDIR, got 2 arguments`, `.. got 4 arguments` |
| 105 | `ShaFromName` of `/usr/share/ollama/blobs/sha256-` + 64 hex (lower case); 63 digits; upper case; `model.gguf`; `sha256-<hex>/x` | the 64 hex; '' (four times) |
| 106 | `ParseTrainArgs` of a model `d/sha256-<64 f>`; then with `--base-sha256 given` | BaseSha256 the 64 `f`; `given` |
| 107 | `ConfigHash` of `--valid ValidFile TinyPath TrainFile o` | 32 characters; equal for the same arguments and with `--max-updates 3 --resume --split 1 --ckpt-every 7 --keep 9 --base-sha256 z` and another output directory; differs with `--lr 2e-4`, `--seed 2`, `--classes x`, without `--valid`, and with a train file of one more (empty) line |
| 108 | the set of TrainFile (tiny tokenizer, class compiled-run, budget 120); `TotalUpdates` for `RunArgs`, with `--batches-per-update 3`, and also `--epochs 5` | 4 batches in epochs 0 and 1; 8; 4; 10 |

`TrainCli`, stage 3 (task T17b3): `RunTrainCLI`; 201–212. O := `NewOut`.

| Row | Input | Expected |
|---|---|---|
| 201 | `Run(RunArgs(O, []))` | exit 0; out `done: 8 updates, best <O>ckpt/ckpt-00000008`; err ''; `Steps` `1 0 2\|2 0 4\|3 0 6\|4 0 8\|5 1 2\|6 1 4\|7 1 6\|8 1 8`; `CkptUpdates` `3 4 6 8`; ckpt-00000004, 6 and 8 kept, ckpt-00000003 gone |
| 202 | the same run; `ReadRunState` of ckpt-00000008 and ckpt-00000004; the validation loss of ckpt-00000008 computed with `ValidationLoss(M, V, 256)` on a fresh tiny model with the checkpoint loaded | update 8, epoch 2, position 0, seed 1, rank 4, scale 2, ConfigHash = `ConfigHash` of the arguments, base ''; ValLoss = the computed one rel 1e-9; ckpt-00000004: `4 1 0` (update, epoch, position) |
| 203 | the same run: the first and the eighth update line of the log; the mean `SampleLoss(.., 256)` of the first batch (`EpochBatches(.., 1, 0, 4, 120)[0]`) on a tiny model without adapters | 8 fields; loss = that mean rel 1e-6; lr of update 1 = `ScheduleLR(1e-2, 1, 1, 8, 0.1)`, of update 8 = `ScheduleLR(1e-2, 8, 1, 8, 0.1)` (within 1e-15) |
| 204 | A: a full run; B: `--max-updates 3`, then `--resume` | B's first part: exit 0, out `stopped after update 3 of 8`; second: exit 0, out `resumed from <B>ckpt/ckpt-00000003 at update 3` + #10 + `done: 8 updates, best <B>ckpt/ckpt-00000008`; `UpdateLines` equal; ckpt-00000008 of both: `DiffArr` 0, the same ValLoss |
| 205 | the same with `--max-updates 4` (an epoch's end) | exit 0; `UpdateLines` equal; `DiffArr` 0 |
| 206 | after `--max-updates 3`: `--resume --lr 2e-2`; `--resume --base-sha256 abc`; `--resume` in a fresh directory | exit 1, err `the configuration differs from the checkpoint's: <A>ckpt/ckpt-00000003`; exit 1, `the base model differs from the checkpoint's: ..`; exit 1, `nothing to resume in <A>ckpt/` |
| 207 | `--lr 1e30` | exit 3; err `non-finite loss at update 2; stopped without updating`; `Steps` `1 0 2\|2 0 4`; no ckpt line |
| 208 | a missing model; a missing train file; OtherFile as train; `--warmup 8` | exit 1 each: `no model file <tmp>none.gguf`; `no data file <tmp>none.jsonl`; `no training sample in <OtherFile>`; `warm-up 8 must be below the 8 updates` |
| 209 | `['--rank', '0', 'm', 't', 'o']` | exit 2; err `--rank: an integer in 1..4096, got "0"` + #10 + `usage: qlora_train [options] MODEL TRAIN OUTDIR (see the unit header)`; out '' |
| 210 | `--max-updates 100 --batches-per-update 3 --keep 1` | exit 0; out `done: 4 updates, best <O>ckpt/ckpt-00000004`; `Steps` `1 0 6\|2 0 8\|3 1 6\|4 1 8`; `CkptUpdates` `2 3 4` |
| 211 | `RunArgs` without `--valid` | exit 0; out `done: 8 updates, best ` (no best); ckpt-00000008's ValLoss NaN; every ckpt line's loss `nan` |
| 212 | `--split 1`; `--split 3` (the GPUs hidden) | exit 1, err `No CUDA GPUs are available`; exit 1, err `split 3 outside 0..2`; out '' |

Build and run (the GPUs hidden as in the gate), for `<P>` = `TrainStep`
or `TrainCli`:
`fpc -Mobjfpc -Sh -vew -l- -Fu{proj}/src -Fu{proj}/../../src -Fu{proj}/tests/ut -FU{proj}/work/bin -Fl{proj}/../../build/libtch-rocm -k-rpath={proj}/../../build/libtch-rocm {proj}/tests/ut/<P>_test.pas -o{proj}/work/bin/<P>_test`
then `HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 {proj}/work/bin/<P>_test --all --format=plain`.
Warnings from other units and libtorch's GPU warnings on stderr are
expected; so is the note "Local variable G is assigned but never used"
(the `NoGradGuard` idiom). Until a stage is written, a test program
without test methods reports `No valid tests found`; that is expected.

## ITa (prepared by the harness owner)

`QLORA_MODEL=<rehearsal GGUF> {proj}/tests/ita/T17-run.sh` builds
`src/qlora_train.pas` and `tests/ita/T17/train_check.pas` against `src/`
and runs on the CPU from a fresh directory outside the project with
`tests/ita/T17/train.jsonl` (8 short records) and `valid.jsonl` (4),
committed (made by `make-data.bb` from pieces of `src/LoraOptim.pas`).
Each run of `qlora_train` is its own process: `ovf` (`--valid`, 8 epochs
of one update, lr 1e-3, warm-up 1), `full` (2 epochs, budget 256: 5
updates per epoch, a checkpoint every 3), `part` (the same with
`--max-updates 3`, inside epoch 0, then `--resume`), `nan` (lr 1e30,
budget 256, 2 epochs); then `train_check`, whose oracles compute the losses through the
whole graph without `TrainStep`. Writes `{proj}/work/ita/T17/summary.tsv`
(case, OK/NG, value, limit, note; then the exit codes and the count of
stderr lines other than the GPU warnings and the nan run's message). OK
when value ≤ limit. About 6.5 minutes.

| Case | Compared | Limit |
|---|---|---|
| ITa-T17-001 | `ovf`: the loss of the last update divided by that of the first | 0.1 |
| ITa-T17-002 | `ovf`: the logged loss of update 1 against the mean loss of its samples through the whole graph without adapters (relative difference) | 1e-4 |
| ITa-T17-003 | `ovf`: the best checkpoint's validation loss divided by the validation loss without adapters | 0.99 |
| ITa-T17-004 | `ovf`: the best checkpoint's recorded validation loss against the whole-graph one with its adapters loaded (relative difference) | 1e-4 |
| ITa-T17-005 | `full` against `part` (resumed inside an epoch): every adapter and moment value of the latest checkpoint (max difference; + 1 if the update lines of the logs differ) | 0 |
| ITa-T17-006 | `nan`: exit code 3, two update lines (the second `nan`), no checkpoint (count of mismatches) | 0 |

Measured with the prototype: 0.0896, 3.2e-9, 0.745, 4.7e-9, 0, 0 in 6 min 14 s. With the skeleton every case is NG; with the position not restored on resume 005 is NG (the resumed run ends at ckpt-00000009). The CPU runs are deterministic: the code of the facts gives these values exactly; 001 near its limit or 005 above 0 means the code differs from the facts.

## Stages and case IDs

- Stage 1 (task T17b1): `TrainStep` (all of it); rows UT-TrainStep-001
  to 009.
- Stage 2 (task T17b2): `DefaultTrainConfig`, `ParseTrainArgs` (with
  `Int`, `Num`), `ShaFromName`, `ConfigHash`, `TotalUpdates`, the
  constant `Usage`, `AppendLine`, `G17`; rows UT-TrainCli-101 to 108.
- Stage 3 (task T17b3): `RunTrainCLI`; rows UT-TrainCli-201 to 212.
- ITa: ITa-T17-001 to 006.
