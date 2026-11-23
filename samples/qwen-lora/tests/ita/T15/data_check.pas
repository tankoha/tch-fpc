(* ITa driver of T15 (written by the harness owner; agents run it through
  tests/ita/T15-run.sh, they do not change it).
  Usage: data_check MODEL RECORDS.jsonl MODE
  MODEL: the rehearsal GGUF (its tokenizer only); RECORDS:
  tests/ita/T15/records.jsonl. Classes compiled-run and fpc-approx, at most
  2048 tokens, seed 15, pool 64, budget 2048, epochs 0 to 2.
  MODE check prints one line per case: case<TAB>value<TAB>limit<TAB>note;
  MODE dump prints the batches of epochs 0 to 2 (for 004: the runner runs
  it twice, in two processes, and compares).
  The oracles do not use TrainData for what they check: the records are
  read again here with fpjson; the expected text is rendered here from the
  template's pieces; the counts come from the file. *)
program data_check;

{$mode objfpc}{$H+}

uses
  SysUtils, Classes, Math, fpjson, jsonparser, Tch, GgufFile, QwenTokenizer, QwenChat, TrainData;

const
  TagS = '<|' + 'im_start|>';
  TagE = '<|' + 'im_end|>';
  ThinkO = '<' + 'think>';
  ThinkC = '</' + 'think>';
  MaxLen = 2048;
  Seed = 15;
  Pool = 64;
  Budget = 2048;

type
  TRec = record
    Id, Cls: string;
    Prompt, Answer: RawByteString;
  end;

var
  ModelPath, RecPath, Mode: string;
  Recs: array of TRec;

procedure Emit(const ACase: string; AValue, ALimit: Double; const ANote: string);
begin
  WriteLn(ACase, #9, Format('%.1E', [AValue]), #9, Format('%.1E', [ALimit]), #9, ANote);
end;

procedure ReadRecs;
var
  L: TStringList;
  I: Integer;
  O: TJSONObject;
begin
  L := TStringList.Create;
  try
    L.LoadFromFile(RecPath);
    Recs := nil;
    for I := 0 to L.Count - 1 do
    begin
      if Trim(L[I]) = '' then
        Continue;
      O := GetJSON(L[I], False) as TJSONObject;
      try
        SetLength(Recs, Length(Recs) + 1);
        Recs[High(Recs)].Id := O.Strings['id'];
        Recs[High(Recs)].Cls := O.Strings['class'];
        Recs[High(Recs)].Prompt := O.Strings['prompt'];
        Recs[High(Recs)].Answer := O.Strings['answer'];
      finally
        O.Free;
      end;
    end;
  finally
    L.Free;
  end;
end;

function FindRec(const AId: string): Integer;
var
  I: Integer;
begin
  for I := 0 to High(Recs) do
    if Recs[I].Id = AId then
      Exit(I);
  raise Exception.Create('no record ' + AId);
end;

function SameTokens(const A, B: TTokenArray): Boolean;
var
  I: Integer;
begin
  Result := Length(A) = Length(B);
  if Result then
    for I := 0 to High(A) do
      if A[I] <> B[I] then
        Exit(False);
end;

procedure Check(Tok: TQwenTokenizer);
var
  S: TTrainSet;
  I, J, K, R, Bad, Long, Cls, N, Row, T, E: Integer;
  Prefix, Want: RawByteString;
  Smp: TTrainSample;
  B: TBatches;
  Seen: array of Integer;
  Fill, Pos: Double;
  Ids, Tg, M: TTensor;
  Mv, Tv: TDoubleArray;
  Got: TTokenArray;
begin
  S := TTrainSet.Create(Tok, RecPath, ['compiled-run', 'fpc-approx'], MaxLen);
  try
    { 001: every sample's tokens, its loss span and the text around it }
    Bad := 0;
    for I := 0 to S.Count - 1 do
    begin
      Smp := S[I];
      R := FindRec(Smp.Id);
      Prefix := TagS + 'user'#10 + Recs[R].Prompt + TagE + #10 + TagS + 'assistant'#10
        + ThinkO + #10#10 + ThinkC + #10#10;
      Want := TrimChat(Recs[R].Answer) + TagE;
      if not SameTokens(Smp.Tokens, Tok.Encode(Prefix + Want + #10)) then
        Inc(Bad);
      if Tok.Decode(Copy(Smp.Tokens, 0, Smp.LossFrom)) <> Prefix then
        Inc(Bad);
      if Tok.Decode(Copy(Smp.Tokens, Smp.LossFrom, Smp.LossTo - Smp.LossFrom)) <> Want then
        Inc(Bad);
      if Tok.Decode(Copy(Smp.Tokens, Smp.LossTo, Length(Smp.Tokens) - Smp.LossTo)) <> #10 then
        Inc(Bad);
    end;
    Emit('ITa-T15-001', Bad, 0, Format('%d samples: tokens = Encode(text rendered here); decode of [0, LossFrom) = the prompt part, of [LossFrom, LossTo) = the trimmed answer + im_end, of the rest = one newline (count of mismatches)', [S.Count]));
    { 002: the counts against the file }
    Long := 0;
    Cls := 0;
    for I := 0 to High(Recs) do
      if (Recs[I].Cls <> 'compiled-run') and (Recs[I].Cls <> 'fpc-approx') then
        Inc(Cls)
      else if Length(Tok.Encode(TagS + 'user'#10 + Recs[I].Prompt + TagE + #10 + TagS + 'assistant'#10
        + ThinkO + #10#10 + ThinkC + #10#10 + TrimChat(Recs[I].Answer) + TagE + #10)) > MaxLen then
        Inc(Long);
    Bad := Ord(S.SkippedClass <> Cls) + Ord(S.SkippedLong <> Long)
      + Ord(S.Count <> Length(Recs) - Cls - Long);
    Emit('ITa-T15-002', Bad, 0, Format('admitted %d, skipped by class %d (file %d), too long %d (file %d), of %d records (count of mismatches)',
      [S.Count, S.SkippedClass, Cls, S.SkippedLong, Long, Length(Recs)]));
    { 003: three epochs of batches: every sample once per epoch, every batch within the budget }
    Bad := 0;
    Fill := 0;
    Pos := 0;
    N := 0;
    for E := 0 to 2 do
    begin
      B := EpochBatches(S, Seed, E, Pool, Budget);
      Seen := nil;
      SetLength(Seen, S.Count);
      for I := 0 to High(B) do
      begin
        T := 0;
        for J := 0 to High(B[I]) do
        begin
          Inc(Seen[B[I][J]]);
          T := Max(T, Length(S[B[I][J]].Tokens));
          Fill := Fill + Length(S[B[I][J]].Tokens);
        end;
        if (Length(B[I]) = 0) or (Length(B[I]) * T > Budget) then
          Inc(Bad);
        Pos := Pos + Length(B[I]) * T;
        Inc(N);
      end;
      for I := 0 to High(Seen) do
        if Seen[I] <> 1 then
          Inc(Bad);
    end;
    Emit('ITa-T15-003', Bad, 0, Format('epochs 0-2, pool %d, budget %d: %d batches, %.1f samples per batch, %.0f %% of the padded positions are tokens (count of batches over the budget or empty, and of samples not seen exactly once)',
      [Pool, Budget, N, 3 * S.Count / Max(N, 1), 100 * Fill / Max(Pos, 1)]));
    { 005: the masked targets of every row of every batch of epoch 0 }
    Bad := 0;
    B := EpochBatches(S, Seed, 0, Pool, Budget);
    for I := 0 to High(B) do
    begin
      BatchTensors(S, B[I], Tok.EosId, Ids, Tg, M);
      Mv := M.to_device(TDevice.Cpu, kDouble).reshape([-1]).ToDoubleArray;
      Tv := Tg.to_device(TDevice.Cpu, kDouble).reshape([-1]).ToDoubleArray;
      T := Tg.Shape[1];
      for Row := 0 to High(B[I]) do
      begin
        Got := nil;
        for K := 0 to T - 1 do
          if Mv[Row * T + K] <> 0 then
          begin
            SetLength(Got, Length(Got) + 1);
            Got[High(Got)] := Round(Tv[Row * T + K]);
          end;
        R := FindRec(S[B[I][Row]].Id);
        if Tok.Decode(Got) <> TrimChat(Recs[R].Answer) + TagE then
          Inc(Bad);
      end;
    end;
    Emit('ITa-T15-005', Bad, 0, 'epoch 0: in every row of every batch, the targets under the mask decode to the trimmed answer + im_end (count of rows that differ)');
  finally
    S.Free;
  end;
end;

procedure Dump(Tok: TQwenTokenizer);
var
  S: TTrainSet;
  B: TBatches;
  E, I, J: Integer;
  L: string;
begin
  S := TTrainSet.Create(Tok, RecPath, ['compiled-run', 'fpc-approx'], MaxLen);
  try
    for E := 0 to 2 do
    begin
      B := EpochBatches(S, Seed, E, Pool, Budget);
      for I := 0 to High(B) do
      begin
        L := IntToStr(E) + ':';
        for J := 0 to High(B[I]) do
          L := L + ' ' + S[B[I][J]].Id;
        WriteLn(L);
      end;
    end;
  finally
    S.Free;
  end;
end;

var
  G: TGgufFile;
  Tok: TQwenTokenizer;
begin
  if ParamCount <> 3 then
  begin
    WriteLn(StdErr, 'usage: data_check MODEL RECORDS.jsonl check|dump');
    Halt(2);
  end;
  ModelPath := ParamStr(1);
  RecPath := ParamStr(2);
  Mode := ParamStr(3);
  G := TGgufFile.Open(ModelPath);
  Tok := TQwenTokenizer.Create(G);
  try
    ReadRecs;
    if Mode = 'check' then
      Check(Tok)
    else if Mode = 'dump' then
      Dump(Tok)
    else
      raise Exception.Create('unknown mode ' + Mode);
  finally
    Tok.Free;
    G.Free;
  end;
end.
