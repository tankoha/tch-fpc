(* ITa driver of T11 (written by the harness owner; agents run it through
  tests/ita/T11-run.sh, they do not change it).
  Usage: logprobs_check MODEL OURS.jsonl EXPECTED.jsonl PROMPTS.jsonl
  OURS: qlora_logprobs --top 50 output; EXPECTED: the reference of
  work/oracle/T11 (llama-server on an all-F32 copy of the rehearsal GGUF,
  f32 KV cache; DESIGN 8.11 "H43 in detail"). Prints one line per case:
  case<TAB>value<TAB>limit<TAB>note; OK is decided by the runner (value <=
  limit). A reference token missing from our top 50 counts as 99.
  Mode "gpu" (harness owner only): OURS from --split 12 against EXPECTED =
  the CPU run's output (case 101). *)
program logprobs_check;

{$mode objfpc}{$H+}

uses
  SysUtils, Classes, Math, fpjson, jsonparser, Tch, GgufFile, Recompute, QwenModel;

type
  TRec = record
    Id: string;
    N: Integer;
    TopIds: array of Int64;
    TopLp: array of Double;
  end;
  TRecArray = array of TRec;

function LoadRecs(const APath: string): TRecArray;
var
  L: TStringList;
  I, J: Integer;
  O: TJSONObject;
  A: TJSONArray;
begin
  Result := nil;
  L := TStringList.Create;
  try
    L.LoadFromFile(APath);
    for I := 0 to L.Count - 1 do
    begin
      if Trim(L[I]) = '' then
        Continue;
      O := GetJSON(L[I], False) as TJSONObject;
      try
        SetLength(Result, Length(Result) + 1);
        with Result[High(Result)] do
        begin
          Id := O.Strings['id'];
          N := O.Integers['n'];
          A := O.Arrays['top'];
          SetLength(TopIds, A.Count);
          SetLength(TopLp, A.Count);
          for J := 0 to A.Count - 1 do
          begin
            TopIds[J] := A.Objects[J].Int64s['id'];
            TopLp[J] := A.Objects[J].Floats['logprob'];
          end;
        end;
      finally
        O.Free;
      end;
    end;
  finally
    L.Free;
  end;
end;

function FindRec(const R: TRecArray; const AId: string): Integer;
begin
  for Result := 0 to High(R) do
    if R[Result].Id = AId then
      Exit;
  Result := -1;
end;

{ max over the reference's first K tokens of |ours - ref|; 99 when one is
  missing from ours }
function TopDiff(const Ours, Ref: TRec; K: Integer): Double;
var
  I, J: Integer;
  Found: Boolean;
begin
  Result := 0;
  for I := 0 to Min(K, Length(Ref.TopIds)) - 1 do
  begin
    Found := False;
    for J := 0 to High(Ours.TopIds) do
      if Ours.TopIds[J] = Ref.TopIds[I] then
      begin
        Result := Max(Result, Abs(Ours.TopLp[J] - Ref.TopLp[I]));
        Found := True;
        Break;
      end;
    if not Found then
      Exit(99);
  end;
end;

procedure Row(const ACase: string; AValue, ALimit: Double; const ANote: string);
begin
  WriteLn(Format('%s'#9'%.1E'#9'%.1E'#9'%s', [ACase, AValue, ALimit, ANote]));
end;

function PromptIds(const APath, AId: string): TInt64Array;
var
  L: TStringList;
  I, J: Integer;
  O: TJSONObject;
  A: TJSONArray;
begin
  Result := nil;
  L := TStringList.Create;
  try
    L.LoadFromFile(APath);
    for I := 0 to L.Count - 1 do
    begin
      O := GetJSON(L[I], False) as TJSONObject;
      try
        if O.Strings['id'] = AId then
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
end;

procedure RecomputeCase(const AModel, APrompts: string);
var
  G: TGgufFile;
  M: TQwenModel;
  Saved: TSavedInputs;
  P: TInt64Array;
  Ids: array of Integer;
  X, A, B: TTensor;
  Guard: IInterface;
  I: Integer;
begin
  P := PromptIds(APrompts, 'chat-130');
  Ids := nil;
  SetLength(Ids, Length(P));
  for I := 0 to High(P) do
    Ids[I] := P[I];
  G := TGgufFile.Open(AModel);
  Saved := TSavedInputs.Create(True);
  try
    M := TQwenModel.Load(G, TDevice.Cpu, TDevice.Cpu, 0);
    try
      Guard := NoGradGuard;
      X := M.Embed(Ids);
      A := M.Trunk(X);
      B := RecomputeForward(M.Stages, X, Saved);
      Row('ITa-T11-004', (A - B).abs.max.DoubleValue([]), 0,
        Format('chat-130: Stages through RecomputeForward against Trunk (%d saved inputs)', [Saved.Count]));
    finally
      M.Free;
    end;
  finally
    Saved.Free;
    G.Free;
  end;
end;

var
  Ours, Ref: TRecArray;
  I, K, Bad, BadText, NTok: Integer;
  D, Worst, WorstText: Double;
begin
  Ours := LoadRecs(ParamStr(2));
  Ref := LoadRecs(ParamStr(3));
  if (ParamCount >= 5) and (ParamStr(5) = 'gpu') then
  begin
    Worst := 0;
    Bad := 0;
    for I := 0 to High(Ref) do
    begin
      K := FindRec(Ours, Ref[I].Id);
      if K < 0 then
        D := 99
      else
      begin
        D := TopDiff(Ours[K], Ref[I], 10);
        if Ours[K].TopIds[0] <> Ref[I].TopIds[0] then
          Inc(Bad);
      end;
      Worst := Max(Worst, D);
    end;
    Row('ITa-T11-101', Worst + Bad * 99, 1e-3,
      Format('--split 12 (cuda:0 / cuda:1) against the CPU, %d records, top-1 differs in %d', [Length(Ref), Bad]));
    Halt(0);
  end;
  Worst := 0;
  WorstText := 0;
  Bad := 0;
  BadText := 0;
  NTok := 0;
  for I := 0 to High(Ref) do
  begin
    K := FindRec(Ours, Ref[I].Id);
    if K < 0 then
      D := 99
    else
    begin
      D := TopDiff(Ours[K], Ref[I], 10);
      if Ours[K].N <> Ref[I].N then
        D := 99;
    end;
    if Copy(Ref[I].Id, 1, 5) = 'text-' then
    begin
      WorstText := Max(WorstText, D);
      if (K < 0) or (Ours[K].TopIds[0] <> Ref[I].TopIds[0]) then
        Inc(BadText);
    end
    else
    begin
      Inc(NTok);
      Worst := Max(Worst, D);
      if (K < 0) or (Ours[K].TopIds[0] <> Ref[I].TopIds[0]) then
        Inc(Bad);
    end;
  end;
  Row('ITa-T11-001', Bad, 0, Format('top-1 token differs from the reference (%d token prompts of 1-600 tokens)', [NTok]));
  Row('ITa-T11-002', Worst, 0.02, 'max |logprob - ref| over the reference''s top 10 (token prompts)');
  Row('ITa-T11-003', WorstText + BadText * 99, 0.02,
    Format('text prompts through QwenTokenizer: same length, same top-1 (%d differ), max |logprob - ref| over the top 10', [BadText]));
  RecomputeCase(ParamStr(1), ParamStr(4));
end.
