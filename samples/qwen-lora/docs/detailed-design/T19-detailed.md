# Detailed design — units `DatasetFilter` and `DatasetCli`, program `qlora_dataset` (T19)

Source of truth: the T19 facts (`tasks/T19-facts.md`, the interfaces and
implementations final, checked by the harness owner), together with the
T19 external design and system flow (`../external-design/T19-external.md`,
`../system-flow/T19-system.md`, T19a). This document restates and
organises the facts; it adds no new requirements. T19a is documents only:
no source code and no tests are written in this task.

## 1. Module composition

- Source files (all `{$mode objfpc}{$H+}`, in `src/`):
  - `src/DatasetFilter.pas`, unit `DatasetFilter` — the pure parts: the
    record schema (`ParseRecord`, with the local helper `InList`), the
    fenced blocks (`ExtractBlocks`), the normalised source and its key
    (`NormalizeSource`, `SourceKey`), the lexical tokens (`LexTokens`),
    the 5-gram lexical shingles (`Fnv1a` local, `Shingles`), the Jaccard
    measure (`Jaccard`) and the valid split (`ValidCount`, `SplitKey`),
    as in the facts.
  - `src/DatasetCli.pas`, unit `DatasetCli` — the command line
    (`DefaultDatasetConfig`, `ParseDatasetArgs`, with the local helper
    `Push`), the program block and the unit names (`MainBlock`,
    `UnitNameOf`), the verification (`VerifyRecord`, with the local
    helper `SaveText`), the local helpers of the run (`ShOf`,
    `JoinBlocks`, `ReadLines`, `LoadEvals`, `LoadExamples`, `Near`,
    `TokensOf`, `EncodeText`, `JsonStr`) and the run itself
    (`RunDatasetCLI`), as in the facts.
  - `src/qlora_dataset.pas`, the program — the harness owner's thin
    wrapper, **complete** (like `qlora_verify`): it reads the verifier's
    compilers from the environment (`QLORA_FPC`, `QLORA_FPC_UNLEASHED`
    and `HOME`, as with `qlora_verify`), forwards the command line to
    `RunDatasetCLI` with the work root `<temp>/qlora-dataset-<pid>`,
    prints the output lines to stdout and the error lines to stderr, and
    exits with the returned code. Nobody changes it.
- Units used (as in the facts):
  - `DialectVerify` (T06): `DetectDialect`, `VerifyFile`,
    `ExtractExpectErrorText`, the outcomes and classes `OUT_ACCEPTED`,
    `VS_COMPILED_RUN`, `VS_FPC_APPROX`, the environment block `TEnvRec`.
  - `QwenTokenizer` (T05): `TQwenTokenizer` — `Create(G)` against a GGUF,
    `Encode`.
  - `GgufFile` (T01): `TGgufFile.Open`.
  - Support units, as in the facts' code: `fpjson` (`GetJSON`,
    `TJSONObject`, `Arrays`, `Strings`, `Objects`, `AsJSON`,
    `StringToJSONString`), the FPC MD5 unit (`MD5Print`, `MD5String`),
    `Classes` (`TStringList`, `TStrings`, `TFileStream`), `SysUtils`
    (`Format`, `IntToStr`, `FileExists`, `ForceDirectories`,
    `IncludeTrailingPathDelimiter`, `Trim`, `TrimRight`,
    `SortStrings`, `ExtractRelativePath`, `ExtractFileName`).
    `CollectPascalFiles` (the recursive `*.pas`/`*.dpr` collection of
    `LoadExamples`) and `ReadTextFile` come from the project's helper
    units, as written in the facts' code.
- MD5, not sha256, for the source key, the record MD5s and the split
  key: FPC 3.2.2 has no sha256 (as T17, the facts).

## 2. Unit `DatasetFilter`

### 2.1 Constants and types (interface section, as in the facts)

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
```

- `K_DIALECTS`: the allowed dialects, compared literally by `InList`
  (`objfpc`, `fpc-delphi`, `unleashed`, `delphi`, `oxygene`).
- `K_TASK_TYPES`: the allowed task types (`write`, `extend`, `fix`,
  `convert`, `unit`).
- `K_NGRAM = 5`: the shingles are 5-grams of lexical tokens (DESIGN
  8.16).
- `K_NEAR = 0.5`: near duplicates and leaks are at Jaccard **0.5 or
  more** (DESIGN 8.16).
- `TRawRecord`: the fields of the raw record the filter uses; `Blocks`
  are the fenced blocks of `Answer`, in order.
- `TShingles`: an array of `QWord` FNV-1a hashes, **ascending and
  unique**.

### 2.2 `InList` (local)

```pascal
function InList(const AValue, AList: string): Boolean;
begin
  Result := Pos(',' + AValue + ',', ',' + AList + ',') > 0;
end;
```

The value must be **exactly one of** the comma-separated entries of
`AList` (the search is anchored by the comma on both sides, so `obj`
does not match `objfpc`, and the comparison is literal — no case
folding: `OBJFPC` is not in `K_DIALECTS`). Used by `ParseRecord` for
`dialect` and `task_type`.

### 2.3 `ParseRecord`

Interface comment, as in the facts: `''` when `ALine` is a valid record
(`R` filled), else the reason `schema: <what>`. Checks, in order: a
JSON object; the strings `id` (non-empty), `dialect` (one of
`K_DIALECTS`), `task_type` (one of `K_TASK_TYPES`), `topic`, `language`,
`prompt` (non-empty), `answer` (non-empty), `expected_stdout`; the array
`features`; the object `generator`; at least one fenced block in the
answer.

```pascal
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
```

Steps, in exactly this order:

1. `R := Default(TRawRecord)` — the record starts empty (the caller uses
   it only when the result is `''`).
2. `D := GetJSON(ALine, False)`; when the text is not JSON at all the
   exception (`EJSONParser`) is caught and the reason is
   `schema: not a JSON object` (4.4, #1).
3. When `D.JSONType <> jtObject` (e.g. `[1, 2]`) the reason is
   `schema: not a JSON object` again (4.4, #1).
4. The local `Str(AKey, ANonEmpty, AValue)`: the key must be present
   and a JSON **string** (then the value is read into `AValue`, else
   `AValue := ''`), and — when `ANonEmpty` — it must be **non-empty**.
   In this order, with these reasons:
   - `id`, non-empty, into `R.Id` → `schema: id` (4.4, #2);
   - `dialect`, non-empty, into `R.Dialect`, and `InList(R.Dialect,
     K_DIALECTS)` (literal) → `schema: dialect` (4.4, #3);
   - `task_type`, non-empty, into `R.TaskType`, and `InList(R.TaskType,
     K_TASK_TYPES)` (literal) → `schema: task_type` (4.4, #4);
   - `topic`, string (may be empty, into the local `Topic`) →
     `schema: topic` (4.4, #5);
   - `language`, string (may be empty, into the local `Lang`) →
     `schema: language` (4.4, #6);
   - `prompt`, non-empty, into `R.Prompt` → `schema: prompt` (4.4, #7);
   - `answer`, non-empty, into `R.Answer` → `schema: answer` (4.4, #8);
   - `expected_stdout`, string (may be empty, into `R.Expected`) →
     `schema: expected_stdout` (4.4, #9).
   A non-string value (a `null`, a number, an array, an object) fails
   the string test of the same check.
5. `features` must be present and a JSON **array** → `schema: features`
   (4.4, #10).
6. `generator` must be present and a JSON **object** →
   `schema: generator` (4.4, #11).
7. `R.Blocks := ExtractBlocks(R.Answer)` (2.4); when there is not a
   single closed fenced block in the answer →
   `schema: answer has no fenced block` (4.4, #12).
8. On success `Result := ''` (the record is valid and `R` is filled).
9. The `finally` frees `D` on every path.

### 2.4 `ExtractBlocks`

Interface comment, as in the facts: the **closed** fenced blocks of
`AText`; a line starting with ` ``` ` opens a block, a line equal to
` ``` ` (trailing spaces ignored) closes it; the lines between, each
ended by `#10`; an unclosed block is not returned.

```pascal
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
```

Steps, in exactly this order:

1. `Result := nil`, `N := 0`, `Inside := False`, `Cur := ''`.
2. `L.Text := AText` — the text is split into lines.
3. For each line, in order:
   - outside a block: a line whose first three characters are ` ``` `
     opens a block — the opening line may carry an info string after the
     fence (```` ```pascal ```` qualifies); `Inside := True`,
     `Cur := ''`.
   - inside a block: a line that is exactly ` ``` ` — trailing spaces
     ignored (`TrimRight`) — closes it: the block so far (`Cur`) is
     stored at index `N` and `N` is incremented; any other line inside
     is appended to the block with a line feed at the end
     (`Cur := Cur + line + #10`), **as written** (not trimmed, not
     lowercased).
   - A block that is never closed (its closing fence is missing) is
     **not** returned.
4. The `finally` frees `L`. The blocks come out in their order of
   appearance in the text.

### 2.5 `NormalizeSource`

Interface comment, as in the facts: `ASrc` without comments (braces,
`(* *)`, `//` to the line's end; **not inside** `...` strings), lower
case outside strings, every run of white space one space, trimmed.

```pascal
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
```

Steps, scanning the text character by character (the local `Put` emits
one pending space — but only when the result is not empty and a space is
pending — and then the character):

1. A **quote** opens a string: the quote is emitted (with the pending
   space, if any), then the characters are copied **verbatim** — the
   string body is not lowercased and its `{`, `(*`, `//` are **not**
   treated as comments — up to and including the closing quote (an
   unterminated string copies the rest of the text as written). No
   white-space collapsing happens inside the string.
2. A **brace** opens a brace comment: skip to and including the closing
   `}` (an unclosed brace consumes the rest of the text); a comment
   marks a pending space.
3. `(*` opens a block comment: skip to and including `*)` (an unclosed
   one consumes the rest); a pending space.
4. `//` starts a line comment: skip to just before the line feed
   (the line feed itself is then white space); a pending space.
5. A white-space character (`#9`, `#10`, `#13`, ` ' '`) only marks the
   pending space — every run of white space becomes at most one space,
   and the leading white space of the text is dropped (as is a trailing
   run, since no character follows it): the result is **trimmed**.
6. Any other character is emitted **lowercased** (outside strings).

### 2.6 `SourceKey`

Interface comment, as in the facts: MD5 (hex) of the normalised blocks
joined by `#10`.

```pascal
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
```

Steps:

1. The blocks, in order, each normalised (2.5) and joined by a single
   line feed (the normalized first block, then
   `#10 + <each block normalised>`).
2. `Result := MD5Print(MD5String(S))` — the MD5 digest in hex. MD5, not
   sha256: FPC 3.2.2 has no sha256 (as T17, the facts). Two blocks
   that differ only in comments, case or white space give the **same**
   key (an exact duplicate); a string that differs — `'a'` vs `'A'` —
   gives a different key (strings keep their case, 2.5).

### 2.7 `LexTokens`

Interface comment, as in the facts: the tokens of `LowerCase(AText)`:
words (`[a-z_][a-z0-9_]*`), numbers (`[0-9]+`), every other character
but white space alone.

```pascal
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
```

Steps, scanning `T := LowerCase(AText)`:

1. A character in `['a'..'z', '_']` starts a **word**: extend while the
   following characters are in `['a'..'z', '0'..'9', '_']`.
2. A digit starts a **number**: extend while the following characters
   are digits.
3. A white-space character (`#9`, `#10`, `#13`, ` ' '`) is skipped — it
   is **not** a token (and, since the sources are normalised before
   tokenising in the run, the only white space here is the delimiter
   between tokens).
4. Any other single character (a `:`, an `=`, a `;`, a `+`, `...`) is a
   token of its own.
5. The token `Copy(T, I, J - I)` is appended and the scan resumes after
   it. The result is `nil` for white-space-only or empty input.

### 2.8 `Fnv1a` (local)

The hash used by `Shingles`, exactly as in the facts (`{$Q-}{$R-}`
turns overflow and range checking off around the 64-bit multiply):

```pascal
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
```

Steps:

1. `Result := QWord($CBF29CE484222325)` (the starting basis).
2. For each byte, in order:
   `Result := (Result xor Ord(S[I])) * QWord($100000001B3)` — the
   multiply runs with overflow checking off (`{$Q-}{$R-}`), so it is
   taken modulo 2^64.
3. `{$push}`/`{$pop}` restore the directive state after the function.

### 2.9 `Shingles`

Interface comment, as in the facts: the FNV-1a 64 hashes of the
`K_NGRAM`-grams of `ATokens` (joined by one space), ascending and
unique; empty when there are fewer tokens.

```pascal
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
```

Steps:

1. `Result := nil`, `N := 0`. The loop runs `I := 0` to
   `Length(ATokens) - K_NGRAM` (5 with `K_NGRAM = 5`): with **fewer
   than five** tokens the loop does not run and the result is empty.
2. For each window, `S := ATokens[I]` followed by the next four tokens,
   each joined by **one space** — the 5-token gram as text.
3. `H := Fnv1a(S)` (2.8).
4. Insertion into the array, keeping it **ascending** (binary-search
   style scan from the end down to the first `Result[J-1] <= H`); when
   an equal hash is already present the gram is **dropped** (unique);
   otherwise `H` is inserted at `J` by shifting the tail, and `N` is
   incremented.

### 2.10 `Jaccard`

Interface comment, as in the facts: `|A ∩ B| / |A ∪ B|`; 0 when both
are empty.

```pascal
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
```

Steps:

1. Both empty → 0.
2. The two **sorted, unique** arrays are merged linearly; `Both`
   counts the hashes present in both (the intersection; with unique
   arrays counting a hit once is exact).
3. `Result := Both / (Length(A) + Length(B) - Both)` — `|A ∩ B| /
   |A ∪ B|` (when one side is empty the numerator is 0, the
   denominator is the other side's size, so the result is 0).
   The comparison uses whole doubles; the run rejects at
   `>= K_NEAR` (0.5).

### 2.11 `ValidCount`

Interface comment, as in the facts: the records of a cell of `AN` that
go to valid: `(AN · APercent + 50) div 100`.

```pascal
function ValidCount(AN, APercent: Integer): Integer;
begin
  Result := (AN * APercent + 50) div 100;
end;
```

The quota (DESIGN 8.16, 5 % by default) **rounded to the nearest
record**: `+ 50` before the integer division. Examples (the facts'
UT row 009): `(0,5) → 0`, `(9,5) → 0`, `(10,5) → 1`, `(29,5) → 1`,
`(30,5) → 2`, `(1,25) → 0`, `(2,25) → 1`, `(6,25) → 2`,
`(5,100) → 5`, `(5,0) → 0`.

### 2.12 `SplitKey`

Interface comment, as in the facts: the order inside a cell: MD5 (hex)
of `'<seed>:<id>'`; the first `ValidCount` by this key go to valid.

```pascal
function SplitKey(ASeed: Int64; const AId: string): string;
begin
  Result := MD5Print(MD5String(IntToStr(ASeed) + ':' + AId));
end;
```

The **seed is part of the key** (two different seeds order the same
cell differently; a missing seed collapses the ordering — the facts'
mutant list has both: `the valid count rounded down: DatasetFilter
009; the seed left out of the split key: 010`). MD5 again (no sha256
in FPC 3.2.2).

## 3. Unit `DatasetCli`

### 3.1 Constants and types (interface section, as in the facts)

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
```

- `TDatasetConfig`: the parsed command line (`OutDir` is stored with a
  trailing path delimiter; `Model` and `RefTokens` go **both or
  neither**); the arrays `Evals` and `LeakDirs` are the repeatable
  `--eval` files and `--leak-dir` directories, in the order given.
- `TLeakText` / `TLeakTexts`: the texts a leak check compares against —
  a name (the eval's `id`, or the example file relative to its
  directory) and the 5-gram shingles of its tokens.

### 3.2 `DefaultDatasetConfig`

Interface comment, as in the facts: Seed 1, ValidPercent 5, Timeout 10,
SpecVersion `'v1'`, the rest empty.

```pascal
function DefaultDatasetConfig: TDatasetConfig;
begin
  Result := Default(TDatasetConfig);
  Result.Seed := 1;
  Result.ValidPercent := 5;
  Result.Timeout := 10;
  Result.SpecVersion := 'v1';
end;
```

The defaults, in one place: the split seed **1**, the valid quota
**5 %** (DESIGN 8.16's quota), the verifier timeout **10** seconds, the
specification version **`v1`**; `Raw`, `OutDir`, `Model`, `RefTokens`
empty, no evals, no leak directories.

### 3.3 `ParseDatasetArgs` (with the local `Push`)

Interface comment, as in the facts: `''` or the usage error. Options:
`--model F`, `--ref-tokens F`, `--eval F` (repeatable), `--leak-dir D`
(repeatable), `--seed N`, `--valid-percent P` (0..100), `--timeout S`
(1..3600), `--spec-version V`; then `RAW OUTDIR`.

```pascal
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
```

Steps, in exactly this order:

1. `C := DefaultDatasetConfig()` (3.2); `Pos2 := nil`; the argument
   loop.
2. A **recognised option** — one of `--model`, `--ref-tokens`,
   `--eval`, `--leak-dir`, `--seed`, `--valid-percent`, `--timeout`,
   `--spec-version`, compared literally (case-sensitive): when it is
   the **last** argument (`I = High(AArgs)`, its value is missing) →
   `<option>: value missing` (4.2); else the next argument is the
   value and **two** arguments are consumed:
   - `--model` → `C.Model`; `--ref-tokens` → `C.RefTokens`;
     `--eval` → appended to `C.Evals` (repeatable, in order);
     `--leak-dir` → appended to `C.LeakDirs` (repeatable, in order);
     `--spec-version` → `C.SpecVersion` (all taken as given).
   - The **numeric** options, first parsed into `N` with
     `TryStrToInt64` (a non-integer makes `N = -1`):
     - `--seed`: must be an integer (re-parsed into `C.Seed`, so a
       negative value like `-4` is legal) — else
       `--seed: an integer, got "<v>"` (4.2).
     - `--valid-percent`: must be an integer in **0..100** — else
       `--valid-percent: an integer in 0..100, got "<v>"` (4.2).
     - `--timeout`: must be an integer in **1..3600** — else
       `--timeout: an integer in 1..3600, got "<v>"` (4.2).
3. Any other argument longer than one character that starts with `-`
   (`(Length(A) > 1) and (A[1] = '-')`, compared literally — no case
   folding: `--Seed`, and not just eight known names) →
   `unknown option <arg>` (4.2); a one-character dash is a positional.
4. All other arguments are **positionals**, appended to `Pos2` in
   order.
5. When the positionals are not exactly two →
   `expected RAW OUTDIR, got <n> arguments` (4.2).
6. When exactly **one** of `C.Model` / `C.RefTokens` is empty
   (`(C.Model = '') <> (C.RefTokens = '')`, the exclusive or) →
   `--model and --ref-tokens go together` (4.2) — the tokenizer check
   runs **without both** options it does not run (the facts, per the
   card).
7. `C.Raw := Pos2[0]`; `C.OutDir := IncludeTrailingPathDelimiter(Pos2[1])`
   (the output directory is stored with **one** trailing path
   delimiter; `out` and `o/` both give `o/`); `Result := ''`.

The local `Push` appends one string to the open array (used for
`--eval` and `--leak-dir`).

### 3.4 `MainBlock`

Interface comment, as in the facts: the program block of `ABlocks`: the
first whose normalised text does not start with `'unit '`; -1 when
there is none.

```pascal
function MainBlock(const ABlocks: TStringArray): Integer;
var
  I: Integer;
begin
  for I := 0 to High(ABlocks) do
    if Copy(NormalizeSource(ABlocks[I]), 1, 5) <> 'unit ' then
      Exit(I);
  Result := -1;
end;
```

The first block whose normalised (2.5) text does **not** start with
`unit ` is the **program block** — e.g. `program P;` or
`namespace N;` — and its index is returned; when **every** block is a
unit (or there are none: `High` below `Low`) the result is `-1`. This
is the facts' decision: the program block is the first that does not
start with `unit ` (normalised); the rest are units.

### 3.5 `UnitNameOf`

Interface comment, as in the facts: the unit name of a block that
starts with `'unit <name>;'` (normalised), as written in the source;
`''` otherwise.

```pascal
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
```

Steps:

1. The block is normalised (2.5); when it does not start with `unit `
   the result is `''` (the program block has no unit name).
2. The name is read from position 6 while the characters are in
   `['a'..'z', '0'..'9', '_', '.']` — letters, digits, underscore and
   **dot**, so a dotted name like `Dotted.Name` is kept whole — up to
   the `;` (or any other character).
3. The name is found **as written** by a case-insensitive search in the
   original block (`Pos` of the lowercased name in
   `LowerCase(ABlock)`) and returned with its original casing — a
   `unit TwiceU;` gives `TwiceU`, a `UNIT Dotted.Name;` gives
   `Dotted.Name`.

### 3.6 `VerifyRecord` (with the local `SaveText`)

Interface comment, as in the facts: writes the blocks into `ADir`
(`main.pas` for the program block, `<unit>.pas` for the others,
`main.expected` with `AExpected`) and verifies `main.pas` as
`ADialect`; `AOutcome` and `ADetail` are `VerifyFile`'s; the result is
True when the outcome is `'accepted'`, the class `compiled-run` or
`fpc-approx` and the program block demands no compile error;
`AClass` gets the class.

```pascal
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
```

Steps, in exactly this order:

1. `ForceDirectories(ADir)` — the record's working directory (the run
   gives one per record, 3.8 step 4).
2. `M := MainBlock(R.Blocks)` (3.4); every block is written, **as
   raw**:
   - block `M` (the program block) → `ADir + 'main.pas'`;
   - the others (the units) → `ADir + UnitNameOf(block) + '.pas'` (3.5)
     — the units are next to `main.pas`, so the program compiles
     against them (the facts' decision).
   - `main.expected` gets `R.Expected` (an empty file when the record's
     `expected_stdout` is empty).
3. `VerifyFile(ADir + 'main.pas', R.Dialect, ATimeout, AEnv,
   ADir + 'build', OK)` (T06, on the **record's** dialect, with the
   record's timeout and the environment block, a `build` subdirectory
   for the verifier's work) — its tab-separated line yields
   `AClass := Line[2]`, `AOutcome := Line[3]`, `ADetail := Line[4]`.
4. `Result := (AOutcome = OUT_ACCEPTED) and ((AClass = VS_COMPILED_RUN)
   or (AClass = VS_FPC_APPROX)) and (ExtractExpectErrorText(R.Blocks[M])
   = '')` — **accepted** means (the facts' decision):
   - the verifier's outcome is `accepted`;
   - the class is `compiled-run` or `fpc-approx` (any other class —
     in particular Oxygene's `static-only`, which stays until its
     filter row (M3, a change of T06) — fails, and the record is
     rejected with the reason `verify: <outcome>`, e.g.
     `verify: static-ok`);
   - the **program block** demands no compile error: no
     `expect-error:` marker (`ExtractExpectErrorText` — the verifier
     accepts such a pitfall file without running it, so this clause
     keeps it out of the data set).
   `AClass`, `AOutcome` and `ADetail` are returned to the run on every
   path (the reject line carries the detail).

The local `SaveText` creates the file (overwriting) and writes the
text; an empty text leaves an **empty** file.

### 3.7 Local helpers of the run

`ShOf` — the shingles of a text for the leak checks; the sources are
normalised first, the prompts are compared as written:

```pascal
function ShOf(const AText: string; ASource: Boolean): TShingles;
begin
  if ASource then
    Result := Shingles(LexTokens(NormalizeSource(AText)))
  else
    Result := Shingles(LexTokens(AText));
end;
```

`JoinBlocks` — the blocks concatenated, in order, without a separator
(the normalisation in `ShOf` removes the boundary white space):

```pascal
function JoinBlocks(const ABlocks: TStringArray): string;
var
  I: Integer;
begin
  Result := '';
  for I := 0 to High(ABlocks) do
    Result := Result + ABlocks[I];
end;
```

`ReadLines` — a non-existing file raises (the run's message
`no file <path>`, 4.3):

```pascal
function ReadLines(const APath: string): TStringList;
begin
  if not FileExists(APath) then
    raise Exception.Create('no file ' + APath);
  Result := TStringList.Create;
  Result.LoadFromFile(APath);
end;
```

`LoadEvals` — an evaluation task file: one JSON object per line with
`id`, `prompt`, `reference`; **two** leak texts each (the code
comment, as in the facts). Empty lines are skipped:

```pascal
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
```

The two texts are **both named with the eval's `id`** (the reject
reason is `leak-eval: <task id>`); the prompt's shingles are built
**without** normalisation (compared against the record's prompt as
written) and the reference's **with** it (compared against the
record's sources).

`LoadExamples` — a `--leak-dir` directory: the `*.pas`/`*.dpr` files,
**recursively** (the facts' decision), sorted, each as a leak text
named by its path relative to the directory:

```pascal
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
```

A missing directory raises `no leak directory <dir>` (4.3); the file
names in the reject reasons are **relative to the leak directory**
(e.g. `sub/loop.pas`).

`Near` — the first text of `A` at `K_NEAR` (0.5) or more against `ASh`,
else `''` (the order of `A` decides which name is reported):

```pascal
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
```

`TokensOf` — the integer token array of a key of the reference's JSON
line, as space-separated integers:

```pascal
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
```

`EncodeText` — the same shape from the live tokenizer, so the two
sides are compared as strings:

```pascal
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
```

`JsonStr` — a JSON string literal (the record's `id`, the reasons and
the detail carry no control characters, but the escaping is done
properly):

```pascal
function JsonStr(const S: string): string;
begin
  Result := '"' + StringToJSONString(S) + '"';
end;
```

### 3.8 `RunDatasetCLI` (step by step)

Interface comment, as in the facts: `AOut` / `AErr` caller-owned;
`AWorkRoot`: where the verifier works; the result is the exit code (0
done, 1 error, 2 usage).

```pascal
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

Steps, in exactly this order:

**Step 1 — the command line (usage, exit 2)**

1a. `Msg := ParseDatasetArgs(AArgs, C)` (3.3).
1b. When `Msg <> ''`: `AErr.Add(Msg)` and `AErr.Add(K_DATASET_USAGE)`,
then `Exit(2)` — the usage error is the message **followed by**
`K_DATASET_USAGE` = `usage: qlora_dataset [options] RAW OUTDIR (see the
unit header)` (4.2).

**Step 2 — the setup**

`Result := 1` — every later `Exit` without an explicit code returns 1;
only the success path sets 0. The output lists (`Train`, `Valid`,
`Manifest`, `Rejects`, `Card`, `Cells`, `Reasons`, `Ids`, `Refs`,
`Raw`) and the objects (`G`, `Tok`, the leak texts, `Acc`) are
initialised to nil (freed in step 7 on every path).

**Step 3 — the inputs (any raise: the message alone on `AErr`, exit 1)**

3a. When `not FileExists(C.Raw)` → raise `no raw file <raw>` (4.3);
else `Raw := ReadLines(C.Raw)`.
3b. Each `--eval` file, in the given order: `LoadEvals` (3.7; its
   `no file <path>` raise passes through, 4.3) — the eval leak texts
   `EvalP` (prompts) and `EvalS` (references).
3c. Each `--leak-dir`, in the given order: `LoadExamples` (3.7; its
   `no leak directory <dir>` raise passes through, 4.3) — the example
   leak texts `Examples`.
3d. When `C.Model <> ''` (the pairing of 3.3 made both options
    present): `G := TGgufFile.Open(C.Model)` (T01; its errors pass
    through, 4.3) and `Tok := TQwenTokenizer.Create(G)` (T05; its
    errors pass through, 4.3). The reference file `C.RefTokens` is
    then read (its missingness is the same `no file <path>` of
    3.7/4.3): the blank lines are skipped, each line is a JSON object
    `{"id","prompt","answer"}` (the bundled `llama-server`'s
    `/tokenize` of prompt and answer, special tokens parsed, no BOS —
    the facts), and **the first line of an id counts**: `Refs` is a
    sorted name list with `dupIgnore` and a line is added as
    `id + '=' + line` only when the id is not present yet (the later
    lines of the same id are dropped). When neither option is given
    `G` and `Tok` remain nil and the tokenizer check of step 4i is not
    run (the facts, per the card).

**Step 4 — the records, in raw order (the checks of the facts, in this
order; the first failing one is the reason)**

For each raw line `I` (in order; a blank line — `Trim = ''` — is
skipped and not counted; otherwise `Inc(Total)`, `Detail := ''`,
`Cls := ''`):

1. **Schema** — `Reason := ParseRecord(Line, R)` (2.3): one of the
   `schema: ...` reasons, 4.4 #1–#12.
2. **A program block** — when `MainBlock(R.Blocks) < 0` (3.4, every
   block is a unit or there are none) →
   `schema: answer has no program block` (4.4, #13).
3. **The id unique** — when `Ids` already contains `R.Id` →
   `duplicate-id` (4.4, #14). `Ids` gets the id in step 4, so the ids
   of records that failed check 2 do not block a later record with the
   same id.
4. **The dialect** — `Ids.Add(R.Id)`; then
   `Key := DetectDialect(R.Blocks[MainBlock(R.Blocks)], 'main.pas')`
   (T06's detection on the **program block**, as `main.pas`); when the
   detected dialect differs from the record's →
   `dialect: <detected>` (4.4, #15).
5. **The leak against the evaluation tasks** — `Sh :=
   ShOf(JoinBlocks(R.Blocks), True)` (the record's sources,
   normalised, its 5-gram shingles — computed here and reused by
   check 8); first against the eval **references**: `Dup :=
   Near(Sh, EvalS)` (Jaccard at `K_NEAR` 0.5 or more, 3.7); when there
   is no hit, against the eval **prompts**: `Dup :=
   Near(ShOf(R.Prompt, False), EvalP)` (the record's prompt as
   written); when there is a hit → `leak-eval: <task id>` (4.4, #16),
   the first eval text that reaches 0.5, in the loaded order.
6. **The leak against the example files** — `Dup := Near(Sh, Examples)`;
   when there is a hit → `leak-example: <file, relative to its
   directory>` (4.4, #17).
7. **An exact duplicate of an earlier accepted record** —
   `Key := SourceKey(R.Blocks)` (2.6); the **first** accepted record
   `Acc[J]` with the same source key → `duplicate: <id>` (4.4, #18) —
   the duplicate keeps the earlier record (the facts).
8. **A near duplicate of an earlier accepted record** — the first
   accepted record with `Jaccard(Sh, Acc[J].Sh) >= K_NEAR` →
   `near-duplicate: <id>` (4.4, #19) (0.5 or more, DESIGN 8.16).
9. **The tokenizer** — **only when `Tok <> nil`** (both
   `--model` and `--ref-tokens` were given; without both the check is
   not run, the facts): when `Refs` has no line for `R.Id` →
   `tokenizer: no reference` (4.4, #20); else, against the reference
   line **of that id** (the first one, 3d): when the
   space-separated prompt token array differs from
   `QwenTokenizer.Encode(R.Prompt)` of the GGUF's tokenizer, or the
   answer array differs from `Encode(R.Answer)` (compared as strings)
   → `tokenizer` (4.4, #21). A missing id or a **difference**
   rejects (the facts).
10. **The verification** (the slowest, last) —
    `VerifyRecord(R, IncludeTrailingPathDelimiter(AWorkRoot) +
    IntToStr(I + 1) + PathDelim, C.Timeout, AEnv, Cls, Outcome,
    Detail)` (3.6; the record's working directory is the work root —
    the program's `<temp>/qlora-dataset-<pid>` — numbered by the
    record's 1-based line index); when it returns False →
    `verify: <outcome>` (4.4, #22), with `Cls` and the verifier's
    `Detail` for the reject line.
- **Rejected** — when any `Reason` was set: one line goes to
  `Rejects`, the JSON
  `{"line":<I+1>,"id":<the record's id>,"reason":<the reason>,"detail":<the verifier's detail, '' when the verifier did not run>}`
  (the line number is 1-based; the detail is what `VerifyRecord`
  returned — the verifier's detail for a `verify:` reject, `''` for
  the earlier ones); the reason **up to its colon** is counted in
  `Reasons` (the key `schema`, `duplicate`, `near-duplicate`,
  `leak-eval`, `dialect`, `tokenizer`, `verify`… plus
  `duplicate-id`, which has no colon and counts whole); the loop goes
  to the next line.
- **Accepted** — when no `Reason` was set: the raw line is parsed once
  more, `O.Add('class', Cls)` (the verifier's class), and the record
  is appended to `Acc` with: `Line := O.AsJSON` (the **raw object
  plus `class`**, as fpjson's `AsJSON` — what goes to `train.jsonl` /
  `valid.jsonl`), `Id`, `Dialect`, `TaskType`, `Cls`, `Key :=
  SourceKey(R.Blocks)`, `Sh` (the source shingles of step 5),
  `Valid := False` (the split of step 5 may set it).

**Step 5 — the split (per dialect × task type)**

- `Cells` is the sorted, unique list of the cells
  `'dialect' + ' ' + 'tasktype'` of the accepted records.
- For each cell, its accepted records are collected in a `Cell` list
  **sorted by `SplitKey(C.Seed, Acc[I].Id)`** (2.12, the seed in the
  key) with the record's index attached; the first
  `ValidCount(n, C.ValidPercent)` of them (2.11, `(n·p + 50) div 100`)
  get `Valid := True`; the rest stay train. — DESIGN 8.16's quota: per
  dialect × task type, the records ordered by `SplitKey`, the first
  `ValidCount` go to valid (the facts).

**Step 6 — the five output files in `OUTDIR` (both splits in raw
order)**

- `NV := 0`; over `Acc` **in raw order** (`I := 0` to `High(Acc)`):
  the valid ones go to `Valid` (`NV` counts them), the rest to
  `Train` — so `train.jsonl` and `valid.jsonl` keep the raw order.
- `Manifest` gets one line per accepted record, in the same order —
  the JSON `{"id", "record_md5" (the MD5 of the record's output line),
  "source_md5" (the source key), "class", "split" (`valid` or
  `train`), "spec" (C.SpecVersion), "generator" (the generator object
  of the raw line, verbatim)}` (the facts' `manifest.jsonl`).
- `card.md` (the facts' card): the header `# Data set card`; the line
  `Raw records: <n>. Accepted: <a> (train <t>, valid <v>). Rejected:
  <r>. Yield: <x.x %>.` (the yield is `100·a / Max(Total, 1)` with one
  decimal, so a raw set with no record at all yields `0.0 %`); the
  line `Specification: <spec>. Seed: <seed>. Valid: <p> % per dialect
  × task type. Tokenizer check: <…>.` (the check is
  `QwenTokenizer against <the file name of --ref-tokens>` when it ran,
  else `not run`); the table `| Dialect | Task type | Class | Train |
  Valid |` with one row per cell (the dialect, the task type, the
  class — read from the cell's records — and the train and valid
  counts); the table `| Reject reason | Count |` with the reasons up
  to their colon, **in name order** (`Reasons.Sort`), and their
  counts.
- `ForceDirectories(C.OutDir)`; the five files are written, in this
  order: `train.jsonl`, `valid.jsonl`, `manifest.jsonl`,
  `rejects.jsonl`, `card.md` (the facts' outputs).
- **Success** — `AOut.Add(<a> of <n> records accepted (train <t>,
  valid <v>) into <outdir>)` and `Result := 0` (4.5).

**Step 7 — the error catch and the release**

- Any raise of steps 3–6 (the `no raw file` / `no file` / `no leak
  directory` of the unit and its helpers, the GGUF's and the
  tokenizer's errors, a JSON parsing error of an eval or reference
  line) is caught by `except on E: Exception do AErr.Add(E.Message)` —
  the **message alone**, one line — and `Result` stays 1 (4.3).
- The `finally` frees, in this order: `Tok`, `G`, `Refs`, `Ids`,
  `Cells`, `Reasons`, `Card`, `Rejects`, `Manifest`, `Valid`, `Train`,
  `Raw` — on every path that entered the `try` (freeing a nil
  reference is a no-op, so the usage exits of step 1 — before the
  `try` — and the early raises of step 3 are covered as well).

## 4. Checks and messages (in this order)

All the messages exactly as in the T19 facts ("Checks and messages (in
this order)"). The unit's reasons are returned as strings
(`ParseRecord` and the run); the run turns them into exit codes and
the reject lines. A UT row that expects a message compares the
**whole** message (`AssertEquals`, the facts).

### 4.1 The schema reasons (`ParseRecord`, 2.3), in this order

| # | Condition | Message |
|---|---|---|
| 1 | the text is not JSON, or not an object (the first check; the reason of both branches) | `schema: not a JSON object` |
| 2 | `id` missing, not a string, or empty | `schema: id` |
| 3 | `dialect` missing, not a string, empty, or not one of `K_DIALECTS` (literal) | `schema: dialect` |
| 4 | `task_type` missing, not a string, empty, or not one of `K_TASK_TYPES` (literal) | `schema: task_type` |
| 5 | `topic` missing or not a string (may be empty) | `schema: topic` |
| 6 | `language` missing or not a string (may be empty) | `schema: language` |
| 7 | `prompt` missing, not a string, or empty | `schema: prompt` |
| 8 | `answer` missing, not a string, or empty | `schema: answer` |
| 9 | `expected_stdout` missing or not a string (may be empty) | `schema: expected_stdout` |
| 10 | `features` missing or not an array | `schema: features` |
| 11 | `generator` missing or not an object | `schema: generator` |
| 12 | the answer has no **closed** fenced block | `schema: answer has no fenced block` |

### 4.2 The usage errors (`ParseDatasetArgs`, 3.3; `RunDatasetCLI`
prints the message **and** `K_DATASET_USAGE`, **exit 2**), in this
order

| # | Where | Condition | Message |
|---|---|---|---|
| 1 | each option | the option is the last argument (the value is missing) | `<option>: value missing` (e.g. `--eval: value missing`) |
| 2 | `--seed` | the value is not an integer | `--seed: an integer, got "<v>"` |
| 3 | `--valid-percent` | the value is not an integer in 0..100 | `--valid-percent: an integer in 0..100, got "<v>"` |
| 4 | `--timeout` | the value is not an integer in 1..3600 | `--timeout: an integer in 1..3600, got "<v>"` |
| 5 | any argument | longer than one character and starting with `-` (literal) | `unknown option <arg>` |
| 6 | after the loop | the positionals are not exactly two | `expected RAW OUTDIR, got <n> arguments` |
| 7 | after the loop | exactly one of `--model` / `--ref-tokens` given | `--model and --ref-tokens go together` |

Followed on `AErr` by `usage: qlora_dataset [options] RAW OUTDIR (see
the unit header)`.

### 4.3 The runtime errors (`RunDatasetCLI`, 3.8 steps 3–6; **exit 1**,
the message alone on `AErr`), in this order

| # | Where | Condition | Message |
|---|---|---|---|
| 1 | step 3a | `RAW` does not exist | `no raw file <raw>` |
| 2 | step 3b / 3d, `ReadLines` | an evaluation or a reference file does not exist | `no file <path>` |
| 3 | step 3c, `LoadExamples` | a `--leak-dir` directory does not exist | `no leak directory <dir>` |
| 4 | step 3d, pass-through | the GGUF's or the tokenizer's own error (opening `--model`, creating the tokenizer) | their `Message`, as is |

### 4.4 The reject reasons (in `rejects.jsonl`, 3.8 step 4; in the
order of the checks)

| # | Check (3.8 step 4) | Reason (as written in the line) |
|---|---|---|
| 1–12 | schema (2.3) | the 4.1 reasons, in the 4.1 order |
| 13 | no program block (3.4) | `schema: answer has no program block` |
| 14 | the id is already in `Ids` | `duplicate-id` |
| 15 | detected dialect ≠ the record's | `dialect: <detected>` |
| 16 | a leak text against the eval tasks | `leak-eval: <task id>` |
| 17 | a leak text against the example files | `leak-example: <file, relative to its directory>` |
| 18 | the source key of an earlier accepted record | `duplicate: <id>` |
| 19 | Jaccard ≥ 0.5 against an earlier accepted record | `near-duplicate: <id>` |
| 20 | the tokenizer ran and the id has no reference line | `tokenizer: no reference` |
| 21 | the tokenizer ran and prompt or answer differ | `tokenizer` |
| 22 | the verification failed (slowest, last) | `verify: <outcome>` (e.g. `verify: wrong-output`, `verify: compile-error`, `verify: static-ok`) |

The **card** counts the reasons up to their colon (4.4: `schema`,
`dialect`, `leak-eval`, `leak-example`, `duplicate`,
`near-duplicate`, `tokenizer`, `verify`; `duplicate-id` has no colon
and is counted whole), in name order.

### 4.5 The success line (**exit 0**, on `AOut`)

`<a> of <n> records accepted (train <t>, valid <v>) into <outdir>`
(`a` the accepted records, `n` the non-blank raw lines, `t`/`v` the
splits, `<outdir>` as stored with its trailing delimiter).

## 5. The facts' decisions (DESIGN 8.16, "As written for T19") and
where they are handled

| Decision of the facts | Handled by |
|---|---|
| the record schema of 8.16 item 2, checked in order (a JSON object; the strings `id` non-empty, `dialect`, `task_type`, `topic`, `language`, `prompt` non-empty, `answer` non-empty, `expected_stdout`; the array `features`; the object `generator`; at least one fenced block in the answer); the program block is the first block that does not start with `unit ` (normalised); the others are units, written as `<unit name>.pas` next to `main.pas` | `ParseRecord` (2.3), `ExtractBlocks` (2.4), `MainBlock` (3.4), `UnitNameOf` (3.5), `VerifyRecord` (3.6) |
| the order of the checks per record — the **first failing one** is the reason; schema, a program block, the id unique, the dialect detected in the program block equal to the record's, the leak against the eval tasks (the sources against each `reference`, the prompt against each `prompt`), the leak against the example files (`--leak-dir`, `*.pas`/`*.dpr` recursively), the exact duplicate (the source key), the near duplicate, the tokenizer, the verification (slowest, last); a duplicate keeps the earlier record | `RunDatasetCLI` step 4 (3.8), the 4.4 order |
| **exact duplicates** by `SourceKey`: MD5 of the normalised blocks (no comments, lower case outside strings, white space runs as one space); MD5, not sha256 (FPC 3.2.2 has none, as T17); **near duplicates and leaks**: Jaccard of the FNV-1a 64 hashes of lexical-token 5-grams at **0.5 or more**; tokens: words, numbers, every other character but white space alone, lower case; sources normalised first | `SourceKey` (2.6), `NormalizeSource` (2.5), `LexTokens` (2.7), `Fnv1a` (2.8), `Shingles` (2.9), `Jaccard` (2.10), `K_NGRAM`/`K_NEAR` (2.1) |
| the tokenizer check without HTTP in Pascal: `--ref-tokens FILE` (one line `{"id","prompt","answer"}` per record — the bundled `llama-server`'s `/tokenize` of the prompt and the answer, special tokens parsed, no BOS, made by `tests/ita/T19/make-ref.bb`) against `QwenTokenizer.Encode` of the GGUF given by `--model`; a missing id or a difference rejects; **the first line of an id counts**; without both options the check is not run | `RunDatasetCLI` steps 3d and 4.9 (3.8); the pairing in `ParseDatasetArgs` (3.3 step 6) |
| **accepted** = the verifier's outcome `accepted` with class `compiled-run` or `fpc-approx`, and the program block demands no compile error (no `expect-error:` marker; the verifier accepts such a pitfall file without running it); Oxygene stays `static-only` until its filter row (M3, a change of T06), so its records are rejected (`verify: static-ok`) | `VerifyRecord` (3.6 step 4) |
| the split: per dialect × task type, the records ordered by `SplitKey` (MD5 of `<seed>:<id>`), the first `ValidCount` = `(n · percent + 50) div 100` go to valid (`--valid-percent`, default 5, 8.16's quota); the rest to train, both in raw order | `ValidCount` (2.11), `SplitKey` (2.12), `RunDatasetCLI` steps 5–6 (3.8) |
| the outputs in `OUTDIR`: `train.jsonl`, `valid.jsonl` (the raw object plus `class`, as fpjson's `AsJSON`), `manifest.jsonl` (id, MD5 of the output line, the source key, class, split, the specification version, the generator object), `rejects.jsonl` (line, id, reason, the verifier's detail), `card.md` (totals, yield, the table dialect × task type × class with train and valid counts, reject reasons up to their colon with counts) | `RunDatasetCLI` step 6 (3.8) |

## 6. Change history

| Date | Change | Reason |
|---|---|---|
| T19a (initial) | First edition: module composition, the two units function by function (verbatim interfaces and implementations of the facts), the run step by step, the checks and messages in their order, the facts' decisions and where they are handled. | new units and program (T19; documents only in T19a — no source code, no tests) |
