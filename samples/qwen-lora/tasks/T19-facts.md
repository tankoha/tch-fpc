# T19 facts (checked by the harness owner)

T19 is the data set builder (DESIGN 8.16 "Dedup and leaks", "Provenance",
"Quotas", 8.18): the unit `DatasetFilter` (the pure parts: the record
schema, the fenced blocks, the normalised source and its key, lexical
tokens, 5-gram shingles, Jaccard, the valid split), the unit `DatasetCli`
(the command line and the run) and the program `src/qlora_dataset.pas`
(the harness owner's thin wrapper, complete; like `qlora_verify` it reads
`QLORA_FPC`, `QLORA_FPC_UNLEASHED` and `HOME`). It uses `DialectVerify`
(T06), `QwenTokenizer` (T05) and `GgufFile` (T01).

DESIGN 8.18's acceptance for T19: "on a synthetic raw set with planted
duplicates, leaks and failures, exactly the planted ones are rejected".
Decisions of the harness owner for this task (DESIGN 8.16 leaves them
open; recorded there as "As written for T19"):
- **The record schema** of 8.16 item 2, checked in order: a JSON object;
  strings `id` (non-empty), `dialect` (`objfpc`, `fpc-delphi`,
  `unleashed`, `delphi`, `oxygene`, literally), `task_type` (`write`,
  `extend`, `fix`, `convert`, `unit`), `topic`, `language`, `prompt`
  (non-empty), `answer` (non-empty), `expected_stdout`; the array
  `features`; the object `generator`; at least one fenced block in the
  answer. The program block is the first that does not start with `unit `
  (normalised); the others are units, written as `<unit name>.pas` next
  to `main.pas`.
- **The order of the checks per record** (the first failing one is the
  reason): schema; a program block; the id unique; the dialect detected
  in the program block (`DetectDialect`) equal to the record's; leak
  against the evaluation tasks (the sources against each `reference`, the
  prompt against each `prompt`); leak against the example files
  (`--leak-dir`, `*.pas`/`*.dpr` recursively); exact duplicate of an
  earlier accepted record (the source key); near duplicate of an earlier
  accepted record; the tokenizer; the verification (slowest, last). A
  duplicate keeps the earlier record.
- **Exact duplicates** by `SourceKey`: MD5 of the normalised blocks (no
  comments, lower case outside strings, white space runs as one space).
  MD5, not sha256: FPC 3.2.2 has none (as T17). **Near duplicates and
  leaks**: Jaccard of the FNV-1a 64 hashes of lexical-token 5-grams, at
  0.5 or more (8.16). Tokens: words, numbers, every other character but
  white space alone, lower case; sources normalised first.
- **The tokenizer check** without HTTP in Pascal: `--ref-tokens FILE`
  (one line `{"id","prompt","answer"}` per record: the bundled
  `llama-server`'s `/tokenize` of the prompt and the answer, special tokens
  parsed, no BOS; made by `tests/ita/T19/make-ref.bb`) against
  `QwenTokenizer.Encode` of the GGUF given by `--model`; a missing id or a
  difference rejects. The first line of an id counts. Without both
  options the check is not run (the card says so).
- **Accepted** = the verifier's outcome `accepted` with class
  `compiled-run` or `fpc-approx`, and the program block demands no compile
  error (no `expect-error:` marker; the verifier accepts such a pitfall
  file without running it). Oxygene stays `static-only` until its filter
  row (M3, a change of T06), so its records are rejected
  (`verify: static-ok`).
- **The split**: per dialect × task type, the records ordered by
  `SplitKey` (MD5 of `<seed>:<id>`), the first `ValidCount` =
  (n · percent + 50) div 100 go to valid (`--valid-percent`, default 5,
  8.16's quota); the rest to train, both in raw order.
- **Outputs** in OUTDIR: `train.jsonl`, `valid.jsonl` (the raw object plus
  `class`, as fpjson's `AsJSON`), `manifest.jsonl` (id, MD5 of the output
  line, the source key, class, split, the specification version, the
  generator object), `rejects.jsonl` (line, id, reason, the verifier's
  detail), `card.md` (totals, yield, the table dialect × task type × class
  with train and valid counts, reject reasons up to their colon with
  counts).

Measured by the harness owner with a prototype written exactly as below:
- UT: every row of the tables below passes with `-Cr`; both units then
  have 100 % line coverage (191 and 356 lines); mutants are caught (without stripping brace comments: DatasetFilter 004, 005, DatasetCli 104, 106; without lower case: DatasetFilter 004, 005, DatasetCli 104–107; the union counted twice: DatasetFilter 008, DatasetCli 106; the valid count rounded down: DatasetFilter 009; the seed left out of the split key: 010; 4-grams: 007; no prompt leak check: DatasetCli 107; no example leak check: 107; the pitfall clause dropped: 105; the last reference line of an id taken: 108; near duplicates only above 0.9: 106).
- ITa (cases below): all 0; 16 of 33 records accepted, 17 rejected
  exactly as planted, in 3 s per run.

These facts are final; do not investigate further or write probes.

## Unit `DatasetFilter` (interface, final; in the skeleton)

```pascal
const
  K_DIALECTS = 'objfpc,fpc-delphi,unleashed,delphi,oxygene';
  K_TASK_TYPES = 'write,extend,fix,convert,unit';
  K_NGRAM = 5;
  K_NEAR = 0.5;

type
  { The fields of a raw record the filter uses. }
  TRawRecord = record
    Id, Dialect, TaskType, Prompt, Answer, Expected: string;
    Blocks: TStringArray;   { the fenced blocks of Answer, in order }
  end;
  TShingles = array of QWord;   { ascending, unique }

{ '' when ALine is a valid record (R filled), else the reason
  'schema: <what>'. Checks, in order: a JSON object; the strings id
  (non-empty), dialect (one of K_DIALECTS), task_type (one of
  K_TASK_TYPES), topic, language, prompt (non-empty), answer (non-empty),
  expected_stdout; the array features; the object generator; at least one
  fenced block in answer. }
function ParseRecord(const ALine: string; out R: TRawRecord): string;

{ The closed fenced blocks of AText: a line starting with ``` opens a
  block, a line equal to ``` (trailing spaces ignored) closes it; the
  lines between, each ended by #10. An unclosed block is not returned. }
function ExtractBlocks(const AText: string): TStringArray;

{ ASrc without comments (braces, (* *), // to the line's end; not inside
  '...' strings), lower case outside strings, every run of white space
  one space, trimmed. }
function NormalizeSource(const ASrc: string): string;

{ MD5 (hex) of the normalised blocks joined by #10. }
function SourceKey(const ABlocks: TStringArray): string;

{ The tokens of LowerCase(AText): words ([a-z_][a-z0-9_]*), numbers
  ([0-9]+), every other character but white space alone. }
function LexTokens(const AText: string): TStringArray;

{ The FNV-1a 64 hashes of the K_NGRAM-grams of ATokens (joined by one
  space), ascending and unique; empty when there are fewer tokens. }
function Shingles(const ATokens: TStringArray): TShingles;

{ |A ∩ B| / |A ∪ B|; 0 when both are empty. }
function Jaccard(const A, B: TShingles): Double;

{ The records of a cell of AN that go to valid: (AN · APercent + 50) div 100. }
function ValidCount(AN, APercent: Integer): Integer;

{ The order inside a cell: MD5 (hex) of '<seed>:<id>'; the first
  ValidCount by this key go to valid. }
function SplitKey(ASeed: Int64; const AId: string): string;
```

## Implementation of `DatasetFilter` (write it exactly so)

```pascal
implementation

function InList(const AValue, AList: string): Boolean;
begin
  Result := Pos(',' + AValue + ',', ',' + AList + ',') > 0;
end;

function ParseRecord(const ALine: string; out R: TRawRecord): string;
var
  D: TJSONData;
  O: TJSONObject;

  function Str(const AKey: string; ANonEmpty: Boolean; out AValue: string): Boolean;
  var
    V: TJSONData;
  begin
    V := O.Find(AKey);
    Result := (V <> nil) and (V.JSONType = jtString);
    if Result then
      AValue := V.AsString
    else
      AValue := '';
    if Result and ANonEmpty then
      Result := AValue <> '';
  end;

var
  Topic, Lang: string;
  V: TJSONData;
begin
  R := Default(TRawRecord);
  try
    D := GetJSON(ALine, False);
  except
    on Exception do   { EJSONParser for broken text }
      Exit('schema: not a JSON object');
  end;
  try
    if D.JSONType <> jtObject then
      Exit('schema: not a JSON object');
    O := TJSONObject(D);
    if not Str('id', True, R.Id) then
      Exit('schema: id');
    if not Str('dialect', True, R.Dialect) or not InList(R.Dialect, K_DIALECTS) then
      Exit('schema: dialect');
    if not Str('task_type', True, R.TaskType) or not InList(R.TaskType, K_TASK_TYPES) then
      Exit('schema: task_type');
    if not Str('topic', False, Topic) then
      Exit('schema: topic');
    if not Str('language', False, Lang) then
      Exit('schema: language');
    if not Str('prompt', True, R.Prompt) then
      Exit('schema: prompt');
    if not Str('answer', True, R.Answer) then
      Exit('schema: answer');
    if not Str('expected_stdout', False, R.Expected) then
      Exit('schema: expected_stdout');
    V := O.Find('features');
    if (V = nil) or (V.JSONType <> jtArray) then
      Exit('schema: features');
    V := O.Find('generator');
    if (V = nil) or (V.JSONType <> jtObject) then
      Exit('schema: generator');
    R.Blocks := ExtractBlocks(R.Answer);
    if Length(R.Blocks) = 0 then
      Exit('schema: answer has no fenced block');
    Result := '';
  finally
    D.Free;
  end;
end;

function ExtractBlocks(const AText: string): TStringArray;
var
  L: TStringList;
  I, N: Integer;
  Inside: Boolean;
  Cur: string;
begin
  Result := nil;
  N := 0;
  Inside := False;
  Cur := '';
  L := TStringList.Create;
  try
    L.Text := AText;
    for I := 0 to L.Count - 1 do
      if not Inside then
      begin
        if Copy(L[I], 1, 3) = '```' then
        begin
          Inside := True;
          Cur := '';
        end;
      end
      else if TrimRight(L[I]) = '```' then
      begin
        Inside := False;
        SetLength(Result, N + 1);
        Result[N] := Cur;
        Inc(N);
      end
      else
        Cur := Cur + L[I] + #10;
  finally
    L.Free;
  end;
end;

function NormalizeSource(const ASrc: string): string;
var
  I, N: Integer;
  C: Char;
  Space: Boolean;

  procedure Put(AC: Char);
  begin
    if Space and (Result <> '') then
      Result := Result + ' ';
    Space := False;
    Result := Result + AC;
  end;

begin
  Result := '';
  Space := False;
  N := Length(ASrc);
  I := 1;
  while I <= N do
  begin
    C := ASrc[I];
    if C = '''' then
    begin
      Put(C);
      Inc(I);
      while (I <= N) and (ASrc[I] <> '''') do
      begin
        Result := Result + ASrc[I];
        Inc(I);
      end;
      if I <= N then
        Result := Result + '''';
      Inc(I);
    end
    else if C = '{' then
    begin
      while (I <= N) and (ASrc[I] <> '}') do
        Inc(I);
      Inc(I);
      Space := True;
    end
    else if (C = '(') and (I < N) and (ASrc[I + 1] = '*') then
    begin
      Inc(I, 2);
      while (I < N) and not ((ASrc[I] = '*') and (ASrc[I + 1] = ')')) do
        Inc(I);
      Inc(I, 2);
      Space := True;
    end
    else if (C = '/') and (I < N) and (ASrc[I + 1] = '/') then
    begin
      while (I <= N) and (ASrc[I] <> #10) do
        Inc(I);
      Space := True;
    end
    else if C in [#9, #10, #13, ' '] then
    begin
      Space := True;
      Inc(I);
    end
    else
    begin
      Put(LowerCase(C));
      Inc(I);
    end;
  end;
end;

function SourceKey(const ABlocks: TStringArray): string;
var
  I: Integer;
  S: string;
begin
  S := '';
  for I := 0 to High(ABlocks) do
  begin
    if I > 0 then
      S := S + #10;
    S := S + NormalizeSource(ABlocks[I]);
  end;
  Result := MD5Print(MD5String(S));
end;

function LexTokens(const AText: string): TStringArray;
var
  T: string;
  I, J, N, K: Integer;
begin
  Result := nil;
  K := 0;
  T := LowerCase(AText);
  N := Length(T);
  I := 1;
  while I <= N do
  begin
    J := I;
    if T[I] in ['a'..'z', '_'] then
      while (J <= N) and (T[J] in ['a'..'z', '0'..'9', '_']) do
        Inc(J)
    else if T[I] in ['0'..'9'] then
      while (J <= N) and (T[J] in ['0'..'9']) do
        Inc(J)
    else if T[I] in [#9, #10, #13, ' '] then
    begin
      Inc(I);
      Continue;
    end
    else
      J := I + 1;
    SetLength(Result, K + 1);
    Result[K] := Copy(T, I, J - I);
    Inc(K);
    I := J;
  end;
end;

{$push}{$Q-}{$R-}
function Fnv1a(const S: string): QWord;
var
  I: Integer;
begin
  Result := QWord($CBF29CE484222325);
  for I := 1 to Length(S) do
    Result := (Result xor Ord(S[I])) * QWord($100000001B3);
end;
{$pop}

function Shingles(const ATokens: TStringArray): TShingles;
var
  I, J, K, N: Integer;
  S: string;
  H: QWord;
begin
  Result := nil;
  N := 0;
  for I := 0 to Length(ATokens) - K_NGRAM do
  begin
    S := ATokens[I];
    for J := 1 to K_NGRAM - 1 do
      S := S + ' ' + ATokens[I + J];
    H := Fnv1a(S);
    { insertion into the ascending array, duplicates dropped }
    J := N;
    while (J > 0) and (Result[J - 1] > H) do
      Dec(J);
    if (J > 0) and (Result[J - 1] = H) then
      Continue;
    SetLength(Result, N + 1);
    for K := N downto J + 1 do
      Result[K] := Result[K - 1];
    Result[J] := H;
    Inc(N);
  end;
end;

function Jaccard(const A, B: TShingles): Double;
var
  I, J, Both: Integer;
begin
  if (Length(A) = 0) and (Length(B) = 0) then
    Exit(0);
  I := 0;
  J := 0;
  Both := 0;
  while (I < Length(A)) and (J < Length(B)) do
    if A[I] = B[J] then
    begin
      Inc(Both);
      Inc(I);
      Inc(J);
    end
    else if A[I] < B[J] then
      Inc(I)
    else
      Inc(J);
  Result := Both / (Length(A) + Length(B) - Both);
end;

function ValidCount(AN, APercent: Integer): Integer;
begin
  Result := (AN * APercent + 50) div 100;
end;

function SplitKey(ASeed: Int64; const AId: string): string;
begin
  Result := MD5Print(MD5String(IntToStr(ASeed) + ':' + AId));
end;
```

## Unit `DatasetCli` (interface, final; in the skeleton)

```pascal
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
```

## Implementation of `DatasetCli` (write it exactly so)

```pascal
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
```

## The program `src/qlora_dataset.pas` (harness owner's; complete)

It reads the verifier's compilers from the environment (as
`qlora_verify`), forwards the command line to `RunDatasetCLI` with the
work root `<temp>/qlora-dataset-<pid>`, prints the output lines to stdout
and the error lines to stderr, and exits with the code. Nobody changes it.

## Checks and messages (in this order)

`ParseRecord`: `schema: not a JSON object` (broken text or not an
object); `schema: id`, `schema: dialect`, `schema: task_type`, `schema:
topic`, `schema: language`, `schema: prompt`, `schema: answer`, `schema:
expected_stdout`, `schema: features`, `schema: generator`; `schema: answer
has no fenced block`.

`ParseDatasetArgs` (the usage error; `RunDatasetCLI` prints it and
`K_DATASET_USAGE`, exit 2): `<option>: value missing`; `--seed: an
integer, got "<v>"`; `--valid-percent: an integer in 0..100, got "<v>"`;
`--timeout: an integer in 1..3600, got "<v>"`; `unknown option <arg>`;
`expected RAW OUTDIR, got <n> arguments`; `--model and --ref-tokens go
together`.

`RunDatasetCLI` (exit 1, the message alone on AErr): `no raw file <raw>`;
`no file <path>` (an evaluation or reference file); `no leak directory
<dir>`; errors of the GGUF and the tokenizer. Reject reasons (in
`rejects.jsonl`, in the order of the checks): the schema reasons above;
`schema: answer has no program block`; `duplicate-id`; `dialect:
<detected>`; `leak-eval: <task id>`; `leak-example: <file, relative to
its directory>`; `duplicate: <id>`; `near-duplicate: <id>`; `tokenizer:
no reference`; `tokenizer`; `verify: <outcome>`. Success, exit 0: AOut
`<a> of <n> records accepted (train <t>, valid <v>) into <outdir>`.

## UT fixture and helpers (harness owner's; nobody changes them)

`tests/ut/DatasetFilter_test.pas` (skeleton: helpers, runner, no test
methods): `RecLine(Id, Dialect, TaskType, Prompt, Answer)` (a valid
record line: topic `t`, language `en`, expected_stdout `1`#10, features
`[]`, generator `{"model":"m"}`); `RecWith(Key, Json)` (a good record with
the key replaced by the JSON text, '' removes it); `ReasonOf(Line)`;
`Sh([...])` (a TShingles); `Joined(A)` (joined by `|`).

`tests/ut/DatasetCli_test.pas` (skeleton: helpers, runner, no test
methods): constants `PSum` (prints `10`), `PHello`, `PLoop` (`3 2 1 `),
`UTwice` (unit `TwiceU`), `PTwice` (uses it, prints `42`), each with its
output; `NewDir`; `Fence(Src)`; `Rec(Id, Prompt, Srcs, Expected[,
Dialect])` (a record line, answer `Code:`#10 + the fenced sources);
`SaveLines`; `FileText`; `LineCount`; `Reasons(Dir)` (the reasons of
`rejects.jsonl` joined by `|`); `TokJson(T, Text)`; `RunCli(Args, Out,
Err)` (RunDatasetCLI with `Env` and the work root `<TmpDir>work`);
`RawRec(Id, Blocks, Expected[, Dialect])`. Variables `TmpDir`,
`TinyPath` (the tiny model of `QwenBlocksFixture`, with the T05 tiny
tokenizer), `Env` (`ReadEnvRec('', '', HOME)`: the stock `fpc`).

## Unit tests (rows; every row has one method, every method belongs to a row)

Error rows compare the whole message (`AssertEquals`).

`DatasetFilter`, stage 1 (task T19b1); 001–010.

| Row | Input | Expected |
|---|---|---|
| 001 | `ParseRecord(RecLine('a1', 'fpc-delphi', 'convert', 'Do it.', 'Text'#10'```pascal'#10'program P;'#10'begin end.'#10'```'#10))` | ''; Id, Dialect, TaskType, Prompt as given; Expected `1`#10; one block `program P;`#10`begin end.`#10 |
| 002 | `ReasonOf` of: `{"id": `; `[1, 2]`; `RecWith` id removed, `""`, `7`; dialect `"turbo"`, `"OBJFPC"`; task_type `"other"`; topic removed; language `1`; prompt `""`; answer removed; expected_stdout `null`; features `"a"`; generator `[]`; answer `"no fence"`; topic `""` | `schema: not a JSON object` (twice); `schema: id` (three times); `schema: dialect` (twice); `schema: task_type`; `schema: topic`; `schema: language`; `schema: prompt`; `schema: answer`; `schema: expected_stdout`; `schema: features`; `schema: generator`; `schema: answer has no fenced block`; '' |
| 003 | `ExtractBlocks` of `intro`, a block `a`/`  b` closed by ```` ```   ````, `between`, a block `c`, an unclosed block `open`; of `no blocks`#10; of ```` ``` ````#10```` ``` ```` | two blocks `a`#10`  b`#10 and `c`#10; none; one empty block |
| 004 | `NormalizeSource` of `Program P; { note }`#10`(* old *)Begin  // tail`#10#9`WriteLn('A {b} // C');`#13#10`End.`; of `  a{x}b  `; of `'open`; of `x (* unclosed`; of '' | `program p; begin writeln('A {b} // C'); end.`; `a b`; `'open`; `x`; '' |
| 005 | `SourceKey` of [`Program P;`#10`begin end.`#10, `unit U; end.`] and [`program   p; { c }`#10#10`BEGIN END.`, `Unit u;  end.`]; then the second unit `unit v; end.`; of [`x := 'a';`] and [`x := 'A';`] | equal; = `MD5Print(MD5String('program p; begin end.'#10'unit u; end.'))`; differs; differ |
| 006 | `LexTokens` of `Foo_1 := 42+X;`; of #9`A1b`#13#10` 7 _Z `; of `  ` | `foo_1`, `:`, `=`, `42`, `+`, `x`, `;`; `a1b`, `7`, `_z`; none |
| 007 | `Shingles` of 4 tokens; of `a`..`f`; of six `x` then `y`; of six `x`; of `a`..`e` | none; 2, ascending; 2; 1; one, `$E22588B27BD8C58C` |
| 008 | `Jaccard` of [1,2,3]/[1,2,3]; [1,2]/[3,4]; [1,2,3]/[2,3,4]; [1,5]/[1,2,3,4]; []/[]; [1]/[] | 1; 0; 0.5; 0.2; 0; 0 |
| 009 | `ValidCount` (0,5), (9,5), (10,5), (29,5), (30,5), (1,25), (2,25), (6,25), (5,100), (5,0) | 0, 0, 1, 1, 2, 0, 1, 2, 5, 0 |
| 010 | `SplitKey(1, 'o1')`; `SplitKey(-3, 'Id')`; `SplitKey(0, '')` | `MD5Print(MD5String('1:o1'))`; of `'-3:Id'`; 32 characters |

`DatasetCli`, stage 2 (task T19b2); 101–109.

| Row | Input | Expected |
|---|---|---|
| 101 | `DefaultDatasetConfig` | seed 1, valid percent 5, timeout 10, spec version `v1`; Raw, OutDir, Model, RefTokens ''; no evals, no leak directories |
| 102 | every option: `--model m.gguf --ref-tokens r.jsonl --eval e1 --leak-dir x --eval e2 --leak-dir /abs/y --seed -4 --valid-percent 0 --timeout 3600 --spec-version v2 raw.jsonl out`; then `r o/` | ''; each as given (evals `e1`, `e2`; leak directories `x`, `/abs/y`); OutDir `out/`; then OutDir `o/` (one delimiter), valid percent 5 |
| 103 | `--eval` last; `--valid-percent 101`; `--valid-percent x`; `--timeout 0`; `--seed 1.5`; `--Seed 1`; one and three positionals; `--model` alone; `--ref-tokens` alone | `--eval: value missing`; `--valid-percent: an integer in 0..100, got "101"`, `.. got "x"`; `--timeout: an integer in 1..3600, got "0"`; `--seed: an integer, got "1.5"`; `unknown option --Seed`; `expected RAW OUTDIR, got 1 arguments`, `.. got 3 arguments`; `--model and --ref-tokens go together` (twice) |
| 104 | `MainBlock` of [UTwice, PTwice], [PSum], [UTwice], []; `UnitNameOf` of UTwice, of `{ c }`#10`UNIT Dotted.Name;`#10`end.`, of PSum | 1, 0, −1, −1; `TwiceU`, `Dotted.Name`, '' |
| 105 | `VerifyRecord` (timeout 10) of PSum / `10`#10; [UTwice, PTwice] / `42`#10; PSum / `11`#10; PSum with `S + Q`; the same with `{ expect-error: Identifier not found "Q" }`; `namespace N;`#10`end.`#10 as `oxygene` | True, class `compiled-run`, outcome `accepted`; True; False, `wrong-output`; False, `compile-error`, the detail contains `Identifier not found "Q"`; False with outcome `accepted`; False, class `static-only`, outcome `static-ok` |
| 106 | raw: a (PSum), b (PHello), an empty line, c (PLoop), d (UTwice + PTwice), dup (PSum with a comment), near (PHello renamed `Hello2`), wrong (`WriteLn(6 * 7)` expecting `41`); `--valid-percent 50 --seed 2` | 0; out `4 of 7 records accepted (train 2, valid 2) into <D>out/`; err ''; reasons `duplicate: a\|near-duplicate: b\|verify: wrong-output`; 2 train, 2 valid, 4 manifest lines; rejects begins `{"line":6,"id":"dup","reason":"duplicate: a","detail":""}`; the card has `Raw records: 7. Accepted: 4 (train 2, valid 2). Rejected: 3. Yield: 57.1 %.`, `\| objfpc \| write \| compiled-run \| 2 \| 2 \|`, `\| verify \| 1 \|`; train has `"class" : "compiled-run"` |
| 107 | eval e1 (prompt = b's prompt), an empty line, e2 (reference PSum); `--leak-dir` with `sub/loop.pas` = PLoop; raw a (PSum), b, c (PLoop), d | reasons `leak-eval: e2\|leak-eval: e1\|leak-example: sub/loop.pas`; out `1 of 4 records accepted (train 1, valid 0) into <D>o/` |
| 108 | raw a, b, c; reference: a right (tiny tokenizer), b's answer `[1,2,3]`, an empty line, a second line of a (`{"answer":[],"id":"a","prompt":[]}`); `--model TinyPath --ref-tokens` | 0; reasons `tokenizer\|tokenizer: no reference`; the card has `Tokenizer check: QwenTokenizer against ref.jsonl.` |
| 109 | raw: `not json`, a, a again (PHello), u (UTwice only), m (PHello as `fpc-delphi`); then `--frob`; a missing raw file; `--leak-dir <D>nodir`; `--eval <D>e.jsonl` (missing) | 0, reasons `schema: not a JSON object\|duplicate-id\|schema: answer has no program block\|dialect: objfpc`; exit 2, err `unknown option --frob`#10 + K_DATASET_USAGE, out ''; exit 1 `no raw file <D>none.jsonl`; `no leak directory <D>nodir`; `no file <D>e.jsonl` |

Build and run, for `<P>` = `DatasetFilter` or `DatasetCli`:
`fpc -Mobjfpc -Sh -vew -l- -Fu{proj}/src -Fu{proj}/../../src -Fu{proj}/tests/ut -FU{proj}/work/bin -Fl{proj}/../../build/libtch-rocm -k-rpath={proj}/../../build/libtch-rocm {proj}/tests/ut/<P>_test.pas -o{proj}/work/bin/<P>_test`
then `HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 {proj}/work/bin/<P>_test --all --format=plain`.
Warnings from other units and libtorch's GPU warnings on stderr are
expected. Until a stage is written, a test program without test methods
reports `No valid tests found`; that is expected.

## ITa (prepared by the harness owner)

`QLORA_MODEL=<rehearsal GGUF> {proj}/tests/ita/T19-run.sh` builds
`src/qlora_dataset.pas` against `src/` and runs it twice, each from a
fresh directory outside the project, on `tests/ita/T19/raw.jsonl` (16 good
records in four dialects and 17 with planted faults, made by
`make-raw.bb`), `eval.jsonl`, `examples/`, `ref-tokens.jsonl` (made by
`make-ref.bb` from the bundled llama-server, one record altered) and the
rehearsal model's tokenizer, `--valid-percent 25`; then
`tests/ita/T19/check.bb`, which recomputes the references itself. Writes
`{proj}/work/ita/T19/summary.tsv` (case, OK/NG, value, limit, note; then
the exit codes and the count of stderr lines other than the GPU
warnings). OK when value ≤ limit. Under a minute.

| Case | Compared | Limit |
|---|---|---|
| ITa-T19-001 | every raw line's outcome (accepted, or the reject reason up to its colon) against `expected.tsv` (count of mismatches) | 0 |
| ITa-T19-002 | the valid records against the split recomputed per dialect × task type (count of mismatches) | 0 |
| ITa-T19-003 | train and valid: the raw object plus its class, train in raw order (count of mismatches) | 0 |
| ITa-T19-004 | the manifest: one line per accepted record, the MD5 of its output line, its split and class (count of mismatches) | 0 |
| ITa-T19-005 | the card's totals and reject reasons against `rejects.jsonl` (count of mismatches) | 0 |
| ITa-T19-006 | a second run from another directory: the five files byte-identical (count of files that differ) | 0 |

Measured with the prototype: all 0. With the skeleton every case is NG.

## Stages and case IDs

- Stage 1 (task T19b1): `DatasetFilter` (all of it); rows
  UT-DatasetFilter-001 to 010.
- Stage 2 (task T19b2): `DatasetCli` (all of it); rows UT-DatasetCli-101
  to 109.
- ITa: ITa-T19-001 to 006.
