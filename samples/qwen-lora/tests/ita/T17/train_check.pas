(* ITa driver of T17 (written by the harness owner; agents run it through
  tests/ita/T17-run.sh, they do not change it).
  Usage: train_check MODEL TRAIN.jsonl VALID.jsonl WORKDIR
  WORKDIR holds the runs of qlora_train made by T17-run.sh: ovf (8 updates
  on 8 samples, with validation), full (2 epochs of 4 updates), part (3
  updates, inside epoch 0, then --resume to the end), nan (lr 1e30; its
  exit code in nan/exit). Defaults of
  qlora_train otherwise: rank 16, alpha 32, seed 1, pool 64, budget 2048.
  Prints one line per case: case<TAB>value<TAB>limit<TAB>note; OK is
  decided by the runner (value <= limit).
  The oracles do not use TrainStep: losses are computed here through the
  whole graph (one forward, one log_softmax over all positions). *)
program train_check;

{$mode objfpc}{$H+}

uses
  SysUtils, Classes, Math, Tch, TchNN, GgufFile, QwenTokenizer, QwenModel, Lora, LoraOptim,
  Checkpoint, TrainData;

var
  ModelPath, TrainPath, ValidPath, Work: string;
  G: TGgufFile;
  Tok: TQwenTokenizer;

procedure Emit(const ACase: string; AValue, ALimit: Double; const ANote: string);
begin
  WriteLn(ACase, #9, Format('%.3E', [AValue]), #9, Format('%.3E', [ALimit]), #9, ANote);
end;

{ The summed cross-entropy of the loss span of S through the whole graph. }
function WholeLoss(M: TQwenModel; const S: TTrainSample): Double;
var
  NG: IInterface;
  L: TTensor;
  I: Integer;
begin
  NG := NoGradGuard;
  L := M.Logits(M.Trunk(M.Embed(S.Tokens))).log_softmax(-1);
  Result := 0;
  for I := S.LossFrom to S.LossTo - 1 do
    Result := Result - L.select(0, I - 1).select(0, S.Tokens[I]).DoubleValue([]);
end;

function MeanLoss(M: TQwenModel; ASet: TTrainSet; const AIdx: array of Integer): Double;
var
  I: Integer;
  Sum: Double;
  N: Int64;
begin
  Sum := 0;
  N := 0;
  for I := 0 to High(AIdx) do
  begin
    Sum := Sum + WholeLoss(M, ASet[AIdx[I]]);
    Inc(N, ASet[AIdx[I]].LossTo - ASet[AIdx[I]].LossFrom);
  end;
  Result := Sum / N;
end;

function AllIdx(ASet: TTrainSet): TIntArray;
var
  I: Integer;
begin
  Result := nil;
  SetLength(Result, ASet.Count);
  for I := 0 to ASet.Count - 1 do
    Result[I] := I;
end;

function LogLines(const ADir: string): TStringList;
begin
  Result := TStringList.Create;
  if FileExists(ADir + 'train-log.tsv') then
    Result.LoadFromFile(ADir + 'train-log.tsv');
end;

{ The losses of the update lines. }
function Losses(const ADir: string): TDoubleArray;
var
  L: TStringList;
  I: Integer;
  F: TStringArray;
begin
  Result := nil;
  L := LogLines(ADir);
  try
    for I := 0 to L.Count - 1 do
      if Copy(L[I], 1, 4) <> 'ckpt' then
      begin
        F := L[I].Split([#9]);
        SetLength(Result, Length(Result) + 1);
        Result[High(Result)] := StrToFloatDef(F[3], NaN);
      end;
  finally
    L.Free;
  end;
end;

{ The update lines without the seconds column. }
function UpdateText(const ADir: string): string;
var
  L: TStringList;
  I: Integer;
  F: TStringArray;
begin
  Result := '';
  L := LogLines(ADir);
  try
    for I := 0 to L.Count - 1 do
      if Copy(L[I], 1, 4) <> 'ckpt' then
      begin
        F := L[I].Split([#9]);
        SetLength(F, Min(Length(F), 7));
        Result := Result + string.Join(#9, F) + #10;
      end;
  finally
    L.Free;
  end;
end;

procedure AppendT(var R: TDoubleArray; const T: TTensor);
var
  V: TDoubleArray;
  N, I: Integer;
begin
  V := T.detach.to_device(TDevice.Cpu, kDouble).reshape([-1]).ToDoubleArray;
  N := Length(R);
  SetLength(R, N + Length(V));
  for I := 0 to High(V) do
    R[N + I] := V[I];
end;

{ Every A, B and moment of checkpoint ADir read into a fresh set on M. }
function ReadBack(M: TQwenModel; const ADir: string): TDoubleArray;
var
  S: TLoraSet;
  O: TLoraAdamW;
  C: TCheckpoints;
  I, J: Integer;
  V: TNamedTensorArray;
begin
  Result := nil;
  S := TLoraSet.Create(16, 2.0);
  O := nil;
  C := TCheckpoints.Create(ExtractFilePath(ADir), 1000);
  try
    M.AttachLora(S);
    O := TLoraAdamW.Create(S.Stores, DefaultAdamWConfig);
    C.Load(ADir, S, O);
    for I := 0 to S.Count - 1 do
    begin
      AppendT(Result, S.Pair(I).A);
      AppendT(Result, S.Pair(I).B);
    end;
    for I := 0 to O.StoreCount - 1 do
    begin
      V := O.Moments(I).Variables;
      for J := 0 to High(V) do
        AppendT(Result, V[J].Tensor);
    end;
  finally
    C.Free;
    O.Free;
    S.Free;
  end;
end;

procedure Check;
var
  M0, M: TQwenModel;
  TS, VS: TTrainSet;
  Lo, A, B: TDoubleArray;
  Bt: TBatches;
  Base, Mine, BaseVal, Best, D: Double;
  S: TLoraSet;
  O: TLoraAdamW;
  C: TCheckpoints;
  BestDir, LastA, LastB, Note: string;
  St: TRunState;
  L: TStringList;
  Bad, I: Integer;
begin
  TS := TTrainSet.Create(Tok, TrainPath, ['compiled-run', 'fpc-approx'], 2048);
  VS := TTrainSet.Create(Tok, ValidPath, ['compiled-run', 'fpc-approx'], 2048);
  M0 := TQwenModel.Load(G, TDevice.Cpu, TDevice.Cpu, 0);
  try
    { 001: the loss of the last update against the first }
    Lo := Losses(Work + 'ovf' + PathDelim);
    if Length(Lo) < 2 then
      Emit('ITa-T17-001', Infinity, 0.1, Format('%d update lines in ovf', [Length(Lo)]))
    else
      Emit('ITa-T17-001', Lo[High(Lo)] / Lo[0], 0.1, Format('8 samples: loss %.4f at update 1, %.4f at update %d (last / first)',
        [Lo[0], Lo[High(Lo)], Length(Lo)]));
    { 002: update 1 (B zero) against the model without adapters }
    Bt := EpochBatches(TS, 1, 0, 64, 2048);
    Base := MeanLoss(M0, TS, Bt[0]);
    if Length(Lo) > 0 then
      D := Abs(Lo[0] - Base) / Base
    else
      D := Infinity;
    Emit('ITa-T17-002', D, 1e-4, Format('update 1: logged %.6f, whole graph without adapters %.6f (relative difference)',
      [IfThen(Length(Lo) > 0, Lo[0], NaN), Base]));
    { 003: the best checkpoint's validation loss against the base model's }
    BaseVal := MeanLoss(M0, VS, AllIdx(VS));
    C := TCheckpoints.Create(Work + 'ovf' + PathDelim + 'ckpt', 1000);
    try
      BestDir := C.Best;
    finally
      C.Free;
    end;
    if BestDir = '' then
    begin
      Emit('ITa-T17-003', Infinity, 0.99, 'no checkpoint with a validation loss');
      Emit('ITa-T17-004', Infinity, 1e-4, 'no checkpoint with a validation loss');
    end
    else
    begin
      St := ReadRunState(BestDir);
      Emit('ITa-T17-003', St.ValLoss / BaseVal, 0.99, Format('validation loss %.4f at %s (update %d), %.4f without adapters (best / base)',
        [St.ValLoss, ExtractFileName(BestDir), St.Update, BaseVal]));
      { 004: that validation loss computed here with the checkpoint's adapters }
      M := TQwenModel.Load(G, TDevice.Cpu, TDevice.Cpu, 0);
      S := TLoraSet.Create(16, 2.0);
      O := nil;
      C := TCheckpoints.Create(Work + 'ovf' + PathDelim + 'ckpt', 1000);
      try
        M.AttachLora(S);
        O := TLoraAdamW.Create(S.Stores, DefaultAdamWConfig);
        C.Load(BestDir, S, O);
        Mine := MeanLoss(M, VS, AllIdx(VS));
        Emit('ITa-T17-004', Abs(St.ValLoss - Mine) / Mine, 1e-4, Format('recorded %.6f, whole graph %.6f (relative difference)', [St.ValLoss, Mine]));
      finally
        C.Free;
        O.Free;
        M.Free;
        S.Free;
      end;
    end;
    { 005: one process against 3 updates + resume (inside an epoch) + the rest }
    C := TCheckpoints.Create(Work + 'full' + PathDelim + 'ckpt', 1000);
    try
      LastA := C.Latest;
    finally
      C.Free;
    end;
    C := TCheckpoints.Create(Work + 'part' + PathDelim + 'ckpt', 1000);
    try
      LastB := C.Latest;
    finally
      C.Free;
    end;
    if (LastA = '') or (ExtractFileName(LastA) <> ExtractFileName(LastB)) then
      Emit('ITa-T17-005', Infinity, 0, Format('latest checkpoints differ: "%s", "%s"', [ExtractFileName(LastA), ExtractFileName(LastB)]))
    else
    begin
      A := ReadBack(M0, LastA);
      B := ReadBack(M0, LastB);
      D := 0;
      if Length(A) <> Length(B) then
        D := Infinity
      else
        for I := 0 to High(A) do
          D := Max(D, Abs(A[I] - B[I]));
      Note := 'logs equal';
      if UpdateText(Work + 'full' + PathDelim) <> UpdateText(Work + 'part' + PathDelim) then
      begin
        D := D + 1;
        Note := 'logs differ (+1)';
      end;
      Emit('ITa-T17-005', D, 0, Format('%s: one process vs 3 updates + --resume in a new process (inside epoch 0) + the rest: max |difference| over %d adapter and moment values; %s', [ExtractFileName(LastA), Length(A), Note]));
    end;
    { 006: lr 1e30 stops with exit code 3 before the second update, no checkpoint }
    Bad := 0;
    L := TStringList.Create;
    try
      if FileExists(Work + 'nan' + PathDelim + 'exit') then
        L.LoadFromFile(Work + 'nan' + PathDelim + 'exit');
      if (L.Count = 0) or (Trim(L[0]) <> '3') then
        Inc(Bad);
    finally
      L.Free;
    end;
    Lo := Losses(Work + 'nan' + PathDelim);
    if (Length(Lo) <> 2) or not IsNan(Lo[1]) then
      Inc(Bad);
    if DirectoryExists(Work + 'nan' + PathDelim + 'ckpt' + PathDelim + 'ckpt-00000001') then
      Inc(Bad);
    Emit('ITa-T17-006', Bad, 0, 'lr 1e30: exit code 3, two log lines (the second nan), no checkpoint (count of mismatches)');
  finally
    M0.Free;
    VS.Free;
    TS.Free;
  end;
end;

begin
  if ParamCount <> 4 then
  begin
    WriteLn(StdErr, 'usage: train_check MODEL TRAIN.jsonl VALID.jsonl WORKDIR');
    Halt(2);
  end;
  ModelPath := ParamStr(1);
  TrainPath := ParamStr(2);
  ValidPath := ParamStr(3);
  Work := IncludeTrailingPathDelimiter(ParamStr(4));
  G := TGgufFile.Open(ModelPath);
  Tok := TQwenTokenizer.Create(G);
  try
    Check;
  finally
    Tok.Free;
    G.Free;
  end;
end.
