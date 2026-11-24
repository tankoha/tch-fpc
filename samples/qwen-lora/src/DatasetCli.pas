{ DatasetCli — the data set builder qlora_dataset (T19), per DESIGN.md
  8.16 and tasks/T19-facts.md (authoritative): reads the generator's raw
  JSONL, rejects by schema, id, dialect, leak, duplicate, tokenizer and
  verification, splits the accepted records into train and valid per
  dialect × task type, and writes the records, the manifest, the rejects
  and the card.
  Interface by the harness owner (tasks/T19-facts.md); the bodies are
  written in T19b2. }
unit DatasetCli;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Math, fpjson, jsonparser, md5, GgufFile, QwenTokenizer, DialectVerify,
  DatasetFilter;

const
  K_DATASET_USAGE = 'usage: qlora_dataset [options] RAW OUTDIR (see the unit header)';

type
  TDatasetConfig = record
    Raw, OutDir: string;          { OutDir ends with the path delimiter }
    Model, RefTokens: string;     { both or neither }
    Evals, LeakDirs: TStringArray;
    Seed: Int64;
    ValidPercent, Timeout: Integer;
    SpecVersion: string;
  end;

  { A text the leak check compares with: its name and shingles. }
  TLeakText = record
    Name: string;
    Sh: TShingles;
  end;
  TLeakTexts = array of TLeakText;

{ Seed 1, ValidPercent 5, Timeout 10, SpecVersion 'v1', the rest empty. }
function DefaultDatasetConfig: TDatasetConfig;

{ '' or the usage error. Options: --model F, --ref-tokens F, --eval F
  (repeatable), --leak-dir D (repeatable), --seed N, --valid-percent P
  (0..100), --timeout S (1..3600), --spec-version V; then RAW OUTDIR. }
function ParseDatasetArgs(const AArgs: array of string; out C: TDatasetConfig): string;

{ The program block of ABlocks: the first whose normalised text does not
  start with 'unit '; -1 when there is none. }
function MainBlock(const ABlocks: TStringArray): Integer;

{ The unit name of a block that starts with 'unit <name>;' (normalised),
  as written in the source; '' otherwise. }
function UnitNameOf(const ABlock: string): string;

{ Writes the blocks into ADir (main.pas for the program block, <unit>.pas
  for the others, main.expected with AExpected) and verifies main.pas as
  ADialect. AOutcome and ADetail are VerifyFile's; the result is True when
  the outcome is 'accepted', the class compiled-run or fpc-approx and the
  program block demands no compile error. AClass gets the class. }
function VerifyRecord(const R: TRawRecord; const ADir: string; ATimeout: Integer;
  const AEnv: TEnvRec; out AClass, AOutcome, ADetail: string): Boolean;

{ AOut / AErr caller-owned; AWorkRoot: where the verifier works; the
  result is the exit code (0 done, 1 error, 2 usage). }
function RunDatasetCLI(const AArgs: array of string; const AEnv: TEnvRec;
  const AWorkRoot: string; AOut, AErr: TStrings): Integer;

implementation

function DefaultDatasetConfig: TDatasetConfig;
begin
  Result := Default(TDatasetConfig);
  Result.Seed := 1;
  Result.ValidPercent := 5;
  Result.Timeout := 10;
  Result.SpecVersion := 'v1';
end;

procedure Push(var A: TStringArray; const S: string);
begin
  SetLength(A, Length(A) + 1);
  A[High(A)] := S;
end;

function ParseDatasetArgs(const AArgs: array of string; out C: TDatasetConfig): string;
var
  I: Integer;
  A, V: string;
  Pos2: TStringArray;
  N: Int64;
begin
  C := DefaultDatasetConfig;
  Pos2 := nil;
  I := 0;
  while I <= High(AArgs) do
  begin
    A := AArgs[I];
    if (A = '--model') or (A = '--ref-tokens') or (A = '--eval') or (A = '--leak-dir') or
       (A = '--seed') or (A = '--valid-percent') or (A = '--timeout') or (A = '--spec-version') then
    begin
      if I = High(AArgs) then
        Exit(A + ': value missing');
      V := AArgs[I + 1];
      Inc(I, 2);
      if A = '--model' then C.Model := V
      else if A = '--ref-tokens' then C.RefTokens := V
      else if A = '--eval' then Push(C.Evals, V)
      else if A = '--leak-dir' then Push(C.LeakDirs, V)
      else if A = '--spec-version' then C.SpecVersion := V
      else
      begin
        if not TryStrToInt64(V, N) then
          N := -1;
        if A = '--seed' then
        begin
          if not TryStrToInt64(V, C.Seed) then
            Exit(Format('--seed: an integer, got "%s"', [V]));
        end
        else if A = '--valid-percent' then
        begin
          if (N < 0) or (N > 100) then
            Exit(Format('--valid-percent: an integer in 0..100, got "%s"', [V]));
          C.ValidPercent := N;
        end
        else
        begin
          if (N < 1) or (N > 3600) then
            Exit(Format('--timeout: an integer in 1..3600, got "%s"', [V]));
          C.Timeout := N;
        end;
      end;
      Continue;
    end;
    if (Length(A) > 1) and (A[1] = '-') then
      Exit('unknown option ' + A);
    Push(Pos2, A);
    Inc(I);
  end;
  if Length(Pos2) <> 2 then
    Exit(Format('expected RAW OUTDIR, got %d arguments', [Length(Pos2)]));
  if (C.Model = '') <> (C.RefTokens = '') then
    Exit('--model and --ref-tokens go together');
  C.Raw := Pos2[0];
  C.OutDir := IncludeTrailingPathDelimiter(Pos2[1]);
  Result := '';
end;

function MainBlock(const ABlocks: TStringArray): Integer;
var
  I: Integer;
begin
  for I := 0 to High(ABlocks) do
    if Copy(NormalizeSource(ABlocks[I]), 1, 5) <> 'unit ' then
      Exit(I);
  Result := -1;
end;

function UnitNameOf(const ABlock: string): string;
var
  N: string;
  I: Integer;
begin
  Result := '';
  N := NormalizeSource(ABlock);
  if Copy(N, 1, 5) <> 'unit ' then
    Exit;
  I := 6;
  while (I <= Length(N)) and (N[I] in ['a'..'z', '0'..'9', '_', '.']) do
    Inc(I);
  N := Copy(N, 6, I - 6);
  { the name as written: find it case-insensitively in the source }
  I := Pos(N, LowerCase(ABlock));
  Result := Copy(ABlock, I, Length(N));
end;

procedure SaveText(const APath, AText: string);
var
  F: TFileStream;
begin
  F := TFileStream.Create(APath, fmCreate);
  try
    if AText <> '' then
      F.WriteBuffer(AText[1], Length(AText));
  finally
    F.Free;
  end;
end;

function VerifyRecord(const R: TRawRecord; const ADir: string; ATimeout: Integer;
  const AEnv: TEnvRec; out AClass, AOutcome, ADetail: string): Boolean;
var
  M, I: Integer;
  Line: TStringArray;
  OK: Boolean;
begin
  ForceDirectories(ADir);
  M := MainBlock(R.Blocks);
  for I := 0 to High(R.Blocks) do
    if I = M then
      SaveText(ADir + 'main.pas', R.Blocks[I])
    else
      SaveText(ADir + UnitNameOf(R.Blocks[I]) + '.pas', R.Blocks[I]);
  SaveText(ADir + 'main.expected', R.Expected);
  Line := VerifyFile(ADir + 'main.pas', R.Dialect, ATimeout, AEnv, ADir + 'build', OK).Split([#9]);
  AClass := Line[2];
  AOutcome := Line[3];
  ADetail := Line[4];
  Result := (AOutcome = OUT_ACCEPTED) and ((AClass = VS_COMPILED_RUN) or (AClass = VS_FPC_APPROX))
    and (ExtractExpectErrorText(R.Blocks[M]) = '');
end;

{ ---- the run ---- }

function ShOf(const AText: string; ASource: Boolean): TShingles;
begin
  if ASource then
    Result := Shingles(LexTokens(NormalizeSource(AText)))
  else
    Result := Shingles(LexTokens(AText));
end;

function JoinBlocks(const ABlocks: TStringArray): string;
var
  I: Integer;
begin
  Result := '';
  for I := 0 to High(ABlocks) do
    Result := Result + ABlocks[I];
end;

function ReadLines(const APath: string): TStringList;
begin
  if not FileExists(APath) then
    raise Exception.Create('no file ' + APath);
  Result := TStringList.Create;
  Result.LoadFromFile(APath);
end;

{ eval tasks: one JSON object per line with id, prompt, reference; two leak texts each }
procedure LoadEvals(const AFile: string; var APrompts, ASources: TLeakTexts);
var
  L: TStringList;
  I: Integer;
  O: TJSONObject;
begin
  L := ReadLines(AFile);
  try
    for I := 0 to L.Count - 1 do
    begin
      if Trim(L[I]) = '' then
        Continue;
      O := GetJSON(L[I], False) as TJSONObject;
      try
        SetLength(APrompts, Length(APrompts) + 1);
        APrompts[High(APrompts)].Name := O.Strings['id'];
        APrompts[High(APrompts)].Sh := ShOf(O.Strings['prompt'], False);
        SetLength(ASources, Length(ASources) + 1);
        ASources[High(ASources)].Name := O.Strings['id'];
        ASources[High(ASources)].Sh := ShOf(O.Strings['reference'], True);
      finally
        O.Free;
      end;
    end;
  finally
    L.Free;
  end;
end;

procedure LoadExamples(const ADir: string; var AFiles: TLeakTexts);
var
  F: TStringArray;
  I: Integer;
  T: string;
begin
  if not CollectPascalFiles(ADir, F) then
    raise Exception.Create('no leak directory ' + ADir);
  SortStrings(F);
  for I := 0 to High(F) do
  begin
    ReadTextFile(F[I], T);
    SetLength(AFiles, Length(AFiles) + 1);
    AFiles[High(AFiles)].Name := ExtractRelativePath(IncludeTrailingPathDelimiter(ADir), F[I]);
    AFiles[High(AFiles)].Sh := ShOf(T, True);
  end;
end;

{ the first text of A at K_NEAR or more against ASh, else '' }
function Near(const ASh: TShingles; const A: TLeakTexts): string;
var
  I: Integer;
begin
  for I := 0 to High(A) do
    if Jaccard(ASh, A[I].Sh) >= K_NEAR then
      Exit(A[I].Name);
  Result := '';
end;

function TokensOf(O: TJSONObject; const AKey: string): string;
var
  A: TJSONArray;
  I: Integer;
begin
  Result := '';
  A := O.Arrays[AKey];
  for I := 0 to A.Count - 1 do
    Result := Result + IntToStr(A.Integers[I]) + ' ';
end;

function EncodeText(T: TQwenTokenizer; const AText: string): string;
var
  A: TTokenArray;
  I: Integer;
begin
  Result := '';
  A := T.Encode(AText);
  for I := 0 to High(A) do
    Result := Result + IntToStr(A[I]) + ' ';
end;

function JsonStr(const S: string): string;
begin
  Result := '"' + StringToJSONString(S) + '"';
end;

function RunDatasetCLI(const AArgs: array of string; const AEnv: TEnvRec;
  const AWorkRoot: string; AOut, AErr: TStrings): Integer;
type
  TAcc = record
    Line, Id, Dialect, TaskType, Cls, Key: string;
    Sh: TShingles;
    Valid: Boolean;
  end;
var
  C: TDatasetConfig;
  Msg, Reason, Detail, Cls, Outcome, Line: string;
  Raw, Train, Valid, Manifest, Rejects, Card, Cells, Reasons: TStringList;
  EvalP, EvalS, Examples: TLeakTexts;
  Acc: array of TAcc;
  Ids: TStringList;
  Refs: TStringList;   { id -> JSON line of the reference tokens }
  G: TGgufFile;
  Tok: TQwenTokenizer;
  R: TRawRecord;
  Sh: TShingles;
  Key, Dup: string;
  I, J, K, N, NV, Total: Integer;
  O, Ref: TJSONObject;
  Cell: TStringList;
begin
  Msg := ParseDatasetArgs(AArgs, C);
  if Msg <> '' then
  begin
    AErr.Add(Msg);
    AErr.Add(K_DATASET_USAGE);
    Exit(2);
  end;
  Result := 1;
  Raw := nil; Train := nil; Valid := nil; Manifest := nil; Rejects := nil;
  Card := nil; Cells := nil; Reasons := nil; Ids := nil; Refs := nil;
  G := nil; Tok := nil;
  EvalP := nil; EvalS := nil; Examples := nil; Acc := nil;
  try
    try
      if not FileExists(C.Raw) then
        raise Exception.Create('no raw file ' + C.Raw);
      Raw := ReadLines(C.Raw);
      for I := 0 to High(C.Evals) do
        LoadEvals(C.Evals[I], EvalP, EvalS);
      for I := 0 to High(C.LeakDirs) do
        LoadExamples(C.LeakDirs[I], Examples);
      Refs := TStringList.Create;
      Refs.Sorted := True;
      Refs.Duplicates := dupIgnore;
      if C.Model <> '' then
      begin
        G := TGgufFile.Open(C.Model);
        Tok := TQwenTokenizer.Create(G);
        with ReadLines(C.RefTokens) do
        try
          for I := 0 to Count - 1 do
            if Trim(Strings[I]) <> '' then
            begin
              O := GetJSON(Strings[I], False) as TJSONObject;
              try
                if Refs.IndexOfName(O.Strings['id']) < 0 then   { the first line of an id }
                  Refs.Add(O.Strings['id'] + '=' + Strings[I]);
              finally
                O.Free;
              end;
            end;
        finally
          Free;
        end;
      end;
      Ids := TStringList.Create;
      Ids.Sorted := True;
      Rejects := TStringList.Create;
      Reasons := TStringList.Create;
      Total := 0;
      for I := 0 to Raw.Count - 1 do
      begin
        Line := Raw[I];
        if Trim(Line) = '' then
          Continue;
        Inc(Total);
        Detail := '';
        Cls := '';
        Reason := ParseRecord(Line, R);
        if (Reason = '') and (MainBlock(R.Blocks) < 0) then
          Reason := 'schema: answer has no program block';
        if (Reason = '') and (Ids.IndexOf(R.Id) >= 0) then
          Reason := 'duplicate-id';
        if Reason = '' then
        begin
          Ids.Add(R.Id);
          Key := DetectDialect(R.Blocks[MainBlock(R.Blocks)], 'main.pas');
          if Key <> R.Dialect then
            Reason := 'dialect: ' + Key;
        end;
        if Reason = '' then
        begin
          Sh := ShOf(JoinBlocks(R.Blocks), True);
          Dup := Near(Sh, EvalS);
          if Dup = '' then
            Dup := Near(ShOf(R.Prompt, False), EvalP);
          if Dup <> '' then
            Reason := 'leak-eval: ' + Dup;
        end;
        if Reason = '' then
        begin
          Dup := Near(Sh, Examples);
          if Dup <> '' then
            Reason := 'leak-example: ' + Dup;
        end;
        if Reason = '' then
        begin
          Key := SourceKey(R.Blocks);
          for J := 0 to High(Acc) do
            if Acc[J].Key = Key then
            begin
              Reason := 'duplicate: ' + Acc[J].Id;
              Break;
            end;
        end;
        if Reason = '' then
          for J := 0 to High(Acc) do
            if Jaccard(Sh, Acc[J].Sh) >= K_NEAR then
            begin
              Reason := 'near-duplicate: ' + Acc[J].Id;
              Break;
            end;
        if (Reason = '') and (Tok <> nil) then
        begin
          J := Refs.IndexOfName(R.Id);
          if J < 0 then
            Reason := 'tokenizer: no reference'
          else
          begin
            Ref := GetJSON(Refs.ValueFromIndex[J], False) as TJSONObject;
            try
              if (TokensOf(Ref, 'prompt') <> EncodeText(Tok, R.Prompt)) or
                 (TokensOf(Ref, 'answer') <> EncodeText(Tok, R.Answer)) then
                Reason := 'tokenizer';
            finally
              Ref.Free;
            end;
          end;
        end;
        if (Reason = '') and not VerifyRecord(R, IncludeTrailingPathDelimiter(AWorkRoot) +
          IntToStr(I + 1) + PathDelim, C.Timeout, AEnv, Cls, Outcome, Detail) then
          Reason := 'verify: ' + Outcome;
        if Reason <> '' then
        begin
          Rejects.Add(Format('{"line":%d,"id":%s,"reason":%s,"detail":%s}',
            [I + 1, JsonStr(R.Id), JsonStr(Reason), JsonStr(Detail)]));
          Key := Reason;
          if Pos(':', Key) > 0 then
            Key := Copy(Key, 1, Pos(':', Key) - 1);
          J := Reasons.IndexOfName(Key);
          if J < 0 then
            Reasons.Values[Key] := '1'
          else
            Reasons.ValueFromIndex[J] := IntToStr(StrToInt(Reasons.ValueFromIndex[J]) + 1);
          Continue;
        end;
        O := GetJSON(Line, False) as TJSONObject;
        try
          O.Add('class', Cls);
          K := Length(Acc);
          SetLength(Acc, K + 1);
          Acc[K].Line := O.AsJSON;
        finally
          O.Free;
        end;
        Acc[K].Id := R.Id;
        Acc[K].Dialect := R.Dialect;
        Acc[K].TaskType := R.TaskType;
        Acc[K].Cls := Cls;
        Acc[K].Key := SourceKey(R.Blocks);
        Acc[K].Sh := Sh;
        Acc[K].Valid := False;
      end;
      { the split: per dialect × task type, the first ValidCount by SplitKey }
      Cells := TStringList.Create;
      Cells.Sorted := True;
      Cells.Duplicates := dupIgnore;
      for I := 0 to High(Acc) do
        Cells.Add(Acc[I].Dialect + ' ' + Acc[I].TaskType);
      for K := 0 to Cells.Count - 1 do
      begin
        Cell := TStringList.Create;
        try
          Cell.Sorted := True;
          for I := 0 to High(Acc) do
            if Acc[I].Dialect + ' ' + Acc[I].TaskType = Cells[K] then
              Cell.AddObject(SplitKey(C.Seed, Acc[I].Id), TObject(PtrInt(I)));
          for J := 0 to ValidCount(Cell.Count, C.ValidPercent) - 1 do
            Acc[PtrInt(Cell.Objects[J])].Valid := True;
        finally
          Cell.Free;
        end;
      end;
      Train := TStringList.Create;
      Valid := TStringList.Create;
      Manifest := TStringList.Create;
      NV := 0;
      for I := 0 to High(Acc) do
      begin
        if Acc[I].Valid then
        begin
          Valid.Add(Acc[I].Line);
          Inc(NV);
        end
        else
          Train.Add(Acc[I].Line);
        O := GetJSON(Acc[I].Line, False) as TJSONObject;
        try
          Manifest.Add(Format('{"id":%s,"record_md5":%s,"source_md5":%s,"class":%s,"split":%s,"spec":%s,"generator":%s}',
            [JsonStr(Acc[I].Id), JsonStr(MD5Print(MD5String(Acc[I].Line))), JsonStr(Acc[I].Key),
             JsonStr(Acc[I].Cls), JsonStr(BoolToStr(Acc[I].Valid, 'valid', 'train')),
             JsonStr(C.SpecVersion), O.Objects['generator'].AsJSON]));
        finally
          O.Free;
        end;
      end;
      { the card }
      Card := TStringList.Create;
      Card.Add('# Data set card');
      Card.Add('');
      Card.Add(Format('Raw records: %d. Accepted: %d (train %d, valid %d). Rejected: %d. Yield: %.1f %%.',
        [Total, Length(Acc), Length(Acc) - NV, NV, Rejects.Count, 100.0 * Length(Acc) / Max(Total, 1)]));
      Card.Add(Format('Specification: %s. Seed: %d. Valid: %d %% per dialect × task type. Tokenizer check: %s.',
        [C.SpecVersion, C.Seed, C.ValidPercent, BoolToStr(Tok <> nil, 'QwenTokenizer against ' + ExtractFileName(C.RefTokens), 'not run')]));
      Card.Add('');
      Card.Add('| Dialect | Task type | Class | Train | Valid |');
      Card.Add('|---|---|---|---|---|');
      for K := 0 to Cells.Count - 1 do
      begin
        N := 0;
        J := 0;
        Cls := '';
        for I := 0 to High(Acc) do
          if Acc[I].Dialect + ' ' + Acc[I].TaskType = Cells[K] then
          begin
            Cls := Acc[I].Cls;
            if Acc[I].Valid then Inc(J) else Inc(N);
          end;
        Card.Add(Format('| %s | %s | %s | %d | %d |',
          [Copy(Cells[K], 1, Pos(' ', Cells[K]) - 1), Copy(Cells[K], Pos(' ', Cells[K]) + 1, MaxInt), Cls, N, J]));
      end;
      Card.Add('');
      Reasons.Sort;
      Card.Add('| Reject reason | Count |');
      Card.Add('|---|---|');
      for I := 0 to Reasons.Count - 1 do
        Card.Add(Format('| %s | %s |', [Reasons.Names[I], Reasons.ValueFromIndex[I]]));
      ForceDirectories(C.OutDir);
      Train.SaveToFile(C.OutDir + 'train.jsonl');
      Valid.SaveToFile(C.OutDir + 'valid.jsonl');
      Manifest.SaveToFile(C.OutDir + 'manifest.jsonl');
      Rejects.SaveToFile(C.OutDir + 'rejects.jsonl');
      Card.SaveToFile(C.OutDir + 'card.md');
      AOut.Add(Format('%d of %d records accepted (train %d, valid %d) into %s',
        [Length(Acc), Total, Length(Acc) - NV, NV, C.OutDir]));
      Result := 0;
    except
      on E: Exception do
        AErr.Add(E.Message);
    end;
  finally
    Tok.Free;
    G.Free;
    Refs.Free;
    Ids.Free;
    Cells.Free;
    Reasons.Free;
    Card.Free;
    Rejects.Free;
    Manifest.Free;
    Valid.Free;
    Train.Free;
    Raw.Free;
  end;
end;

end.
