(* ITa driver of T16 (written by the harness owner; agents run it through
  tests/ita/T16-run.sh, they do not change it).
  Usage: ckpt_check MODEL PROMPTS.jsonl WORKDIR MODE
  MODEL: the rehearsal GGUF; PROMPTS: work/oracle/T11/prompts.jsonl (token
  records); WORKDIR: room for the checkpoints and the loss logs.
  MODE, run in this order by T16-run.sh, each in its own process:
    full     10 updates in one go, a checkpoint after every update into
             WORKDIR/full (keep 3); losses to WORKDIR/full-loss.tsv;
    part1    updates 1 to 5, a checkpoint after update 5 into WORKDIR/part;
    part2    a fresh run (another seed, so A differs) resumed from the
             latest checkpoint of WORKDIR/part, updates 6 to 10, a
             checkpoint after update 10; losses of both parts to
             WORKDIR/part-loss.tsv;
    compare  prints the cases.
  One update (DESIGN 8.14, on one prompt): loss = minus the next-token
  log-likelihood of chat-065 through the whole graph, Backward,
  ClipGlobalNorm(1), Step(ScheduleLR(1e-4, U, 2, 10, 0.1)), ZeroGrad.
  Rank 16, scale 2, seed 16 (part2: 17), on the CPU in fp32. The state of
  update U: epoch 0, position U, seed 16, config hash 'ita-t16', base
  sha256 'rehearsal', validation loss 1 at update 2 and 10 + U otherwise.
  compare prints one line per case: case<TAB>value<TAB>limit<TAB>note; OK
  is decided by the runner (value <= limit).
  The oracle of the resume is the uninterrupted run; the oracle of the
  retention is the rule (the last 3 and the best). *)
program ckpt_check;

{$mode objfpc}{$H+}

uses
  SysUtils, Classes, Math, fpjson, jsonparser, Tch, TchNN, GgufFile, QwenModel, Lora,
  LoraOptim, Checkpoint;

const
  Rank = 16;
  Scale = 2.0;
  BaseLR = 1e-4;
  Total = 10;

type
  TIdArray = array of Integer;

var
  ModelPath, PromptPath, WorkDir, Mode: string;

procedure Emit(const ACase: string; AValue, ALimit: Double; const ANote: string);
begin
  WriteLn(ACase, #9, Format('%.1E', [AValue]), #9, Format('%.1E', [ALimit]), #9, ANote);
end;

function PromptIds(const AId: string): TIdArray;
var
  L: TStringList;
  I, J: Integer;
  O: TJSONObject;
  A: TJSONArray;
begin
  Result := nil;
  L := TStringList.Create;
  try
    L.LoadFromFile(PromptPath);
    for I := 0 to L.Count - 1 do
    begin
      if Trim(L[I]) = '' then
        Continue;
      O := GetJSON(L[I], False) as TJSONObject;
      try
        if (O.Get('id', '') = AId) and (O.Find('tokens') <> nil) then
        begin
          A := O.Arrays['tokens'];
          SetLength(Result, A.Count);
          for J := 0 to A.Count - 1 do
            Result[J] := A.Integers[J];
          Exit;
        end;
      finally
        O.Free;
      end;
    end;
  finally
    L.Free;
  end;
  raise Exception.Create('prompt not found: ' + AId);
end;

{ next-token log-likelihood of the prompt, summed, negated }
function Loss(M: TQwenModel; const AIds: TIdArray): TTensor;
var
  T: array of Int64;
  I: Integer;
  L: TTensor;
begin
  T := nil;
  SetLength(T, Length(AIds) - 1);
  for I := 1 to High(AIds) do
    T[I - 1] := AIds[I];
  L := M.Logits(M.Trunk(M.Embed(AIds)));
  Result := -L.narrow(0, 0, Length(AIds) - 1).log_softmax(-1)
    .gather(1, TTensor.FromArray(T).view([-1, 1])).sum(kNone);
end;

function StateOf(AUpdate: Integer): TRunState;
begin
  Result := NewRunState;
  Result.Update := AUpdate;
  Result.Epoch := 0;
  Result.Position := AUpdate;
  Result.Seed := 16;
  Result.ConfigHash := 'ita-t16';
  Result.BaseSha256 := 'rehearsal';
  if AUpdate = 2 then
    Result.ValLoss := 1
  else
    Result.ValLoss := 10 + AUpdate;
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

{ Updates AFrom .. ATo; a checkpoint after every update in AEvery (0: none
  but the last). }
procedure Train(M: TQwenModel; S: TLoraSet; O: TLoraAdamW; C: TCheckpoints;
  AFrom, ATo: Integer; AEvery: Boolean; const ALossFile: string);
var
  P: TIdArray;
  U: Integer;
  L: TTensor;
begin
  P := PromptIds('chat-065');
  for U := AFrom to ATo do
  begin
    L := Loss(M, P);
    AppendLine(ALossFile, Format('%d'#9'%.17g', [U, L.DoubleValue([])]));
    L.Backward;
    O.ClipGlobalNorm(1.0);
    O.Step(ScheduleLR(BaseLR, U, 2, Total, 0.1));
    O.ZeroGrad;
    if AEvery or (U = ATo) then
      C.Save(StateOf(U), S, O);
  end;
end;

procedure RunMode(G: TGgufFile);
var
  M: TQwenModel;
  S: TLoraSet;
  O: TLoraAdamW;
  C: TCheckpoints;
  St: TRunState;
begin
  if Mode = 'part2' then
    ManualSeed(17)
  else
    ManualSeed(16);
  M := TQwenModel.Load(G, TDevice.Cpu, TDevice.Cpu, 0);
  S := TLoraSet.Create(Rank, Scale);
  O := nil;
  C := nil;
  try
    M.AttachLora(S);
    O := TLoraAdamW.Create(S.Stores, DefaultAdamWConfig);
    if Mode = 'full' then
    begin
      C := TCheckpoints.Create(WorkDir + 'full', 3);
      Train(M, S, O, C, 1, 10, True, WorkDir + 'full-loss.tsv');
    end
    else if Mode = 'part1' then
    begin
      C := TCheckpoints.Create(WorkDir + 'part', 3);
      Train(M, S, O, C, 1, 5, False, WorkDir + 'part-loss.tsv');
    end
    else
    begin
      C := TCheckpoints.Create(WorkDir + 'part', 3);
      St := C.Load(C.Latest, S, O);
      if St.Update <> 5 then
        raise Exception.CreateFmt('resumed at update %d, not 5', [St.Update]);
      Train(M, S, O, C, St.Update + 1, 10, False, WorkDir + 'part-loss.tsv');
    end;
  finally
    C.Free;
    O.Free;
    M.Free;
    S.Free;
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

{ every A and B, then every moment, of checkpoint ADir, read into a fresh
  set attached to M }
function ReadBack(M: TQwenModel; const ADir: string; out AState: TRunState): TDoubleArray;
var
  S: TLoraSet;
  O: TLoraAdamW;
  C: TCheckpoints;
  I, J: Integer;
  V: TNamedTensorArray;
begin
  Result := nil;
  S := TLoraSet.Create(Rank, Scale);
  O := nil;
  C := TCheckpoints.Create(ExtractFilePath(ADir), 3);
  try
    M.AttachLora(S);
    O := TLoraAdamW.Create(S.Stores, DefaultAdamWConfig);
    AState := C.Load(ADir, S, O);
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

function MaxDiff(const A, B: TDoubleArray): Double;
var
  I: Integer;
begin
  if Length(A) <> Length(B) then
    Exit(Infinity);
  Result := 0;
  for I := 0 to High(A) do
    Result := Max(Result, Abs(A[I] - B[I]));
end;

function Mismatches(const A, B: TRunState): Integer;
begin
  Result := Ord(A.Update <> B.Update) + Ord(A.Epoch <> B.Epoch)
    + Ord(A.Position <> B.Position) + Ord(A.Seed <> B.Seed) + Ord(A.Rank <> B.Rank)
    + Ord(A.Scale <> B.Scale)
    + Ord(A.ConfigHash <> B.ConfigHash) + Ord(A.BaseSha256 <> B.BaseSha256)
    + Ord(not ((IsNan(A.ValLoss) and IsNan(B.ValLoss)) or (A.ValLoss = B.ValLoss)));
end;

function NamesOf(const ARoot: string): string;
var
  R: TSearchRec;
  L: TStringList;
begin
  L := TStringList.Create;
  try
    if FindFirst(IncludeTrailingPathDelimiter(ARoot) + '*', faAnyFile, R) = 0 then
    try
      repeat
        if (R.Name <> '.') and (R.Name <> '..') then
          L.Add(R.Name);
      until FindNext(R) <> 0;
    finally
      FindClose(R);
    end;
    L.Sort;
    L.Delimiter := ',';
    L.StrictDelimiter := True;
    Result := L.DelimitedText;
  finally
    L.Free;
  end;
end;

function LossLines(const AFile: string): TStringList;
begin
  Result := TStringList.Create;
  Result.LoadFromFile(AFile);
end;

procedure Compare(G: TGgufFile);
var
  M: TQwenModel;
  Full, Part, Five: TDoubleArray;
  SF, SP, S5, Want: TRunState;
  Bad, I: Integer;
  LF, LP: TStringList;
  D: Double;
  Got: string;
begin
  M := TQwenModel.Load(G, TDevice.Cpu, TDevice.Cpu, 0);
  try
    Full := ReadBack(M, WorkDir + 'full' + PathDelim + 'ckpt-00000010', SF);
    Part := ReadBack(M, WorkDir + 'part' + PathDelim + 'ckpt-00000010', SP);
    Five := ReadBack(M, WorkDir + 'part' + PathDelim + 'ckpt-00000005', S5);
  finally
    M.Free;
  end;
  Emit('ITa-T16-001', MaxDiff(Full, Part), 0,
    Format('10 updates in one go vs 5 + checkpoint + resume in a new process + 5: max |difference| over every A, B and moment (%d values; updates 6-10 moved them by up to %.2e)',
      [Length(Full), MaxDiff(Part, Five)]));
  Want := StateOf(10);
  Want.Rank := Rank;
  Want.Scale := Scale;
  Bad := Mismatches(SF, Want) + Mismatches(SP, Want);
  Want := StateOf(5);
  Want.Rank := Rank;
  Want.Scale := Scale;
  Inc(Bad, Mismatches(S5, Want));
  Emit('ITa-T16-002', Bad, 0, 'state.json of full/ckpt-10, part/ckpt-10 and part/ckpt-5 against the states written (count of fields that differ)');
  Bad := 0;
  Got := NamesOf(WorkDir + 'full');
  if Got <> 'ckpt-00000002,ckpt-00000008,ckpt-00000009,ckpt-00000010' then
    Inc(Bad);
  Got := Got + ' | ' + NamesOf(WorkDir + 'part');
  if NamesOf(WorkDir + 'part') <> 'ckpt-00000005,ckpt-00000010' then
    Inc(Bad);
  Emit('ITa-T16-003', Bad, 0, 'directories left (the last 3 and the best, nothing else): ' + Got);
  LF := LossLines(WorkDir + 'full-loss.tsv');
  LP := LossLines(WorkDir + 'part-loss.tsv');
  try
    D := 0;
    if (LF.Count <> 10) or (LP.Count <> 10) then
      D := Infinity
    else
      for I := 0 to 9 do
        if LF[I] <> LP[I] then
          D := Max(D, Abs(StrToFloat(Copy(LF[I], Pos(#9, LF[I]) + 1, 40))
            - StrToFloat(Copy(LP[I], Pos(#9, LP[I]) + 1, 40))));
    Emit('ITa-T16-004', D, 0, Format('loss of every update, one go vs resumed: max |difference| (loss %s at update 1, %s at update 10)',
      [Copy(LF[0], Pos(#9, LF[0]) + 1, 8), Copy(LF[LF.Count - 1], Pos(#9, LF[LF.Count - 1]) + 1, 8)]));
  finally
    LF.Free;
    LP.Free;
  end;
end;

var
  G: TGgufFile;
begin
  if ParamCount <> 4 then
  begin
    WriteLn(StdErr, 'usage: ckpt_check MODEL PROMPTS.jsonl WORKDIR full|part1|part2|compare');
    Halt(2);
  end;
  ModelPath := ParamStr(1);
  PromptPath := ParamStr(2);
  WorkDir := IncludeTrailingPathDelimiter(ParamStr(3));
  Mode := ParamStr(4);
  G := TGgufFile.Open(ModelPath);
  try
    if Mode = 'compare' then
      Compare(G)
    else if (Mode = 'full') or (Mode = 'part1') or (Mode = 'part2') then
      RunMode(G)
    else
      raise Exception.Create('unknown mode ' + Mode);
  finally
    G.Free;
  end;
end.
