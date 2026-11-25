# Detailed design — units `EvalReport` and `EvalCli`, program `qlora_eval` (T20)

Source of truth: the T20 facts (`tasks/T20-facts.md`, the interfaces and
implementations final, checked by the harness owner), together with the
T20 external design and system flow (`../external-design/T20-external.md`,
`../system-flow/T20-system.md`, T20a). This document restates and
organises the facts; it adds no new requirements. T20a is documents only:
no source code and no tests are written in this task.

## 1. Module composition

- Source files (all `{$mode objfpc}{$H+}`, in `src/`):
  - `src/EvalReport.pas`, unit `EvalReport` — the pure parts, as in the
    facts: the constants `K_VERIFIED`, `K_APPROX`, `K_REGRESSION`, the
    types `TJudged`, `TJudgedArray`, `TCount`, `TFlips`, the publics
    `CountOf`, `FlipsOf`, `SignTestP`, `PassRate`, `DialectsOf`,
    `Adoption`, `FirstBlock`, the locals `InList`, `InColumn`, `Above`.
    No file I/O at all; it holds the counts per column, the flips, the
    exact sign test, the adoption rule and the first fenced block.
  - `src/EvalCli.pas`, unit `EvalCli` — the command line
    (`K_EVAL_USAGE`, `TEvalTask`, `TEvalTasks`, `TEvalConfig`,
    `DefaultEvalConfig`, `ParseEvalArgs`), the file helpers (`Lines`,
    `SaveText`, and the locals of `LoadTasks`: `Str`, `BadLine`), the
    tasks (`LoadTasks`), the judging of the recorded replies
    (`JudgeReply`, `TaskIndex`, `JudgeFile`), the report (`Cell`,
    `ReportMode`) and the run itself (`RunEvalCLI`), as in the facts.
  - `src/qlora_eval.pas`, the program — the harness owner's thin
    wrapper, **complete** (like `qlora_verify`): it reads the verifier's
    compilers from the environment (`QLORA_FPC`, `QLORA_FPC_UNLEASHED`
    and `HOME`, as with `qlora_verify`), forwards the command line to
    `RunEvalCLI` with the work root `<temp>/qlora-eval-<pid>`, prints
    the output lines to stdout and the error lines to stderr, and exits
    with the returned code. Nobody changes it.
- Units used (as in the facts' code):
  - `DialectVerify` (T06): `VerifyFile` — returns the finished TSV line
    of five cells `<path> <tab> <dialect> <tab> <class> <tab> <outcome>
    <tab> <detail>` (its detailed design 2.14–2.16);
    `ExtractExpectErrorText`; the outcome constants `OUT_ACCEPTED`,
    `OUT_COMPILE_ERR`, `OUT_STATIC_OK`, `OUT_STATIC_REJECT`,
    `OUT_SKIPPED`; the environment record `TEnvRec`.
  - `fpjson` (FPC): `GetJSON`, `TJSONObject` (`Find`, `Get`, `JSONType`,
    `Items`, `AsString`), `TJSONArray`, `TJSONData` (`Free`), `jtString`,
    `jtObject`.
  - `Classes`: `TStringList` (also the `TStrings` view), `TFileStream`.
  - `SysUtils`: `Format`, `IntToStr`, `FileExists`, `ForceDirectories`,
    `IncludeTrailingPathDelimiter`, `Trim`, `TrimRight`, and the
    `TStringArray` dynamic array (as in T06's detailed design 2.17).
  - The floating-point helpers `Ln`, `Exp`, `Min`, `NaN`, `IsNan` of the
    FPC standard units, used by `EvalReport`.
- The tch-fpc units `Tch`, `TchNN`, `TchJit` are **not** used by the T20
  code: none of the facts' code has a tensor operation.

## 2. Unit `EvalReport`

### 2.1 Constants and types (interface section, as in the facts)

```pascal
const
  K_VERIFIED = 'objfpc,fpc-delphi,unleashed,oxygene';   { the aggregate's dialects }
  K_APPROX = 'delphi';                                  { reported apart }
  K_REGRESSION = 'regression';                          { the dialect of a regression task }

type
  { One judged reply. Compiled and Passed for a Pascal task (Passed: the
    program compiled, ran and printed the expected output); Passed alone
    for a regression task (its check). }
  TJudged = record
    Task, Dialect, Check: string;   { Check: '', 'tool-call' or 'contains' }
    Seed: Int64;
    Think: Boolean;
    Compiled, Passed: Boolean;
  end;
  TJudgedArray = array of TJudged;

  { Counts of one column (a dialect, the aggregate, or a regression check)
    of one model and thinking mode. }
  TCount = record
    N, Compiled, Passed: Integer;
  end;

  { The flips of one column between two models: B fail → pass, C pass →
    fail, of N paired replies. }
  TFlips = record
    N, B, C: Integer;
  end;
```

- `K_VERIFIED`: the dialects counted in the `verified` aggregate
  (`objfpc`, `fpc-delphi`, `unleashed`, `oxygene`). `K_APPROX`:
  `delphi` — reported apart, with the `(approximation)` suffix, and
  **not** in the aggregate (DESIGN 8.17, the facts). `K_REGRESSION`:
  the dialect marker of the regression tasks; such replies have no
  dialect column and are only in their check column (2.3), and their
  dialect name never appears in the report's columns (`DialectsOf`,
  2.8).
- `TJudged`: one judged reply. A Pascal task has both `Compiled` and
  `Passed` (`Passed`: the program compiled, ran and printed the
  expected output); a regression task has `Passed` alone (its check).
  `Check` is `''` for a Pascal task, `tool-call` or `contains` for a
  regression task.
- `TCount`: `N` replies of one column for one model and thinking mode,
  with their `Compiled` and `Passed` counts.
- `TFlips`: the paired flips of one column between the two models:
  `B` fail → pass, `C` pass → fail, `N` the number of pairs.

### 2.2 `InList` (local)

```pascal
function InList(const AValue, AList: string): Boolean;
begin
  Result := Pos(',' + AValue + ',', ',' + AList + ',') > 0;
end;
```

`AValue` must be **exactly one of** the comma-separated entries of
`AList` (anchored by a comma on both sides, literal, no case folding).
Used by `InColumn` (2.3, the `verified` branch) and by `Adoption`
(2.10, the dialect fall check against `K_VERIFIED`).

### 2.3 `InColumn` (local)

```pascal
function InColumn(const J: TJudged; const AColumn: string): Boolean;
begin
  if AColumn = 'verified' then
    Result := InList(J.Dialect, K_VERIFIED)
  else if J.Dialect = K_REGRESSION then
    Result := J.Check = AColumn
  else
    Result := J.Dialect = AColumn;
end;
```

In column membership, in this order:

1. `AColumn = 'verified'` — the reply counts if its dialect is one of
   `K_VERIFIED` (the aggregate: the four dialects together; `delphi`
   is **not** in it).
2. `J.Dialect = 'regression'` — the reply counts in the regression
   check column it carries: `J.Check = AColumn` (a `tool-call` reply
   only in `tool-call`, a `contains` reply only in `contains`).
3. Otherwise — exact dialect match: `J.Dialect = AColumn` (a Pascal
   reply only in its own dialect column; a Pascal reply has `Check =
   ''`, so it is in no regression column).

### 2.4 `CountOf`

Interface, as in the facts: the counts of `AJudged` for thinking mode
`AThink` and column `AColumn`: a dialect name, `'verified'` (the
dialects of `K_VERIFIED` together), or a regression check name
(`'tool-call'`, `'contains'`).

```pascal
function CountOf(const AJudged: TJudgedArray; AThink: Boolean; const AColumn: string): TCount;
var
  I: Integer;
begin
  Result := Default(TCount);
  for I := 0 to High(AJudged) do
    if (AJudged[I].Think = AThink) and InColumn(AJudged[I], AColumn) then
    begin
      Inc(Result.N);
      if AJudged[I].Compiled then
        Inc(Result.Compiled);
      if AJudged[I].Passed then
        Inc(Result.Passed);
    end;
end;
```

`N` counts the matching replies; `Compiled` and `Passed` count the
matching replies with the flag set (a regression reply never sets
`Compiled`). No raise. An empty array (or an empty column) gives
`0, 0, 0`, and `PassRate` of that is NaN (2.7).

### 2.5 `FlipsOf`

Interface, as in the facts: the flips of `AColumn` between `ABefore`
and `AAfter` for `AThink`: replies paired by task and seed; a reply
without a partner is not counted.

```pascal
function FlipsOf(const ABefore, AAfter: TJudgedArray; AThink: Boolean; const AColumn: string): TFlips;
var
  I, J: Integer;
begin
  Result := Default(TFlips);
  for I := 0 to High(ABefore) do
    if (ABefore[I].Think = AThink) and InColumn(ABefore[I], AColumn) then
      for J := 0 to High(AAfter) do
        if (AAfter[J].Think = AThink) and (AAfter[J].Task = ABefore[I].Task) and
           (AAfter[J].Seed = ABefore[I].Seed) then
        begin
          Inc(Result.N);
          if not ABefore[I].Passed and AAfter[J].Passed then
            Inc(Result.B);
          if ABefore[I].Passed and not AAfter[J].Passed then
            Inc(Result.C);
          Break;
        end;
end;
```

For each before reply of the thinking mode and column, the **first**
after reply with the same thinking mode, same task and same seed is
the partner (`Break` at the first match); `N` counts the pairs. `B`
counts the pairs that go fail → pass (`Before.Passed = False`,
`After.Passed = True`), `C` the pairs that go pass → fail. Only
`Passed` is compared, not `Compiled`. A before reply without a
partner, and an after reply without a before partner, are not counted
(DESIGN 8.17, the facts).

### 2.6 `SignTestP`

Interface, as in the facts: the exact two-sided sign test of `B`
against `C`: `min(1, 2 · sum over k = 0..min(B, C) of C(B + C, k) /
2^(B + C))`; 1 when `B + C = 0`.

```pascal
function SignTestP(AB, AC: Integer): Double;
var
  N, K: Integer;
  LogC, S: Double;
begin
  N := AB + AC;
  if N = 0 then
    Exit(1);
  S := 0;
  LogC := 0;   { ln C(N, 0) }
  for K := 0 to Min(AB, AC) do
  begin
    if K > 0 then
      LogC := LogC + Ln(N - K + 1) - Ln(K);
    S := S + Exp(LogC - N * Ln(2));
  end;
  Result := Min(1, 2 * S);
end;
```

Computed in log space so the binomials do not overflow: `LogC` is the
running `ln C(N, K)` (step `K` adds `Ln(N − K + 1) − Ln(K)`), the term
added at step `K` is `C(N, K) / 2^N` = `Exp(LogC − N·Ln(2))`, and the
two-sided probability is capped at 1: `Min(1, 2·S)`. Facts' UT
examples: `SignTestP(0,0) = 1`, `SignTestP(5,0) = 0.0625`,
`SignTestP(39,1) = 2·41/2^40`.

### 2.7 `PassRate`

Interface, as in the facts: the rate of `ACount` (`Passed / N`); NaN
when `N = 0`.

```pascal
function PassRate(const ACount: TCount): Double;
begin
  if ACount.N = 0 then
    Result := NaN
  else
    Result := ACount.Passed / ACount.N;
end;
```

NaN is the "empty series" value: it neither rises nor falls, because
`Above` (2.9) is False with a NaN on either side.

### 2.8 `DialectsOf`

Interface, as in the facts: the dialects of `AJudged` (not the
regression ones), ascending.

```pascal
function DialectsOf(const AJudged: TJudgedArray): TStringArray;
var
  L: TStringList;
  I: Integer;
begin
  L := TStringList.Create;
  try
    L.Sorted := True;
    L.Duplicates := dupIgnore;
    for I := 0 to High(AJudged) do
      if AJudged[I].Dialect <> K_REGRESSION then
        L.Add(AJudged[I].Dialect);
    Result := L.ToStringArray;
  finally
    L.Free;
  end;
end;
```

The non-regression dialect names, ascending, without duplicates
(`Sorted` + `dupIgnore`); `regression` entries are skipped, an empty
array gives an empty result, and `L` is freed on every path
(`try/finally`). Facts' UT: `unleashed, objfpc, regression, delphi,
objfpc` → `delphi|objfpc|unleashed`.

### 2.9 `Above` (local)

```pascal
{ A > B, False when either is NaN (an empty series). }
function Above(A, B: Double): Boolean;
begin
  Result := not IsNan(A) and not IsNan(B) and (A > B);
end;
```

Strict `>` with a NaN guard on **both** sides: an empty series (NaN
rate) neither rises nor falls (DESIGN 8.17, the facts). Used by
`Adoption` (2.10).

### 2.10 `Adoption`

Interface, as in the facts: the adoption rule of 8.17 for `AThink`, one
line per condition (`'yes'/'no'/'not measured'` at the end), then the
verdict line `'adopt: yes'`, `'adopt: no'` or `'adopt: undecided'`
(a condition not measured and none failed). `ASpeedRatio`: the
after/before generation speed (H49, at least 0.7); NaN when not
measured.

```pascal
function Adoption(const ABefore, AAfter: TJudgedArray; AThink: Boolean;
  ASpeedRatio: Double): TStringArray;
var
  L: TStringList;
  Rb, Ra: Double;
  F: TFlips;
  P: Double;
  D: TStringArray;
  I: Integer;
  Fall: string;
  Failed, Unmeasured: Boolean;

  procedure Cond(const AText: string; AOk: Boolean);
  begin
    L.Add(AText + ': ' + BoolToStr(AOk, 'yes', 'no'));
    if not AOk then
      Failed := True;
  end;

begin
  L := TStringList.Create;
  try
    Failed := False;
    Unmeasured := False;
    Rb := PassRate(CountOf(ABefore, AThink, 'verified'));
    Ra := PassRate(CountOf(AAfter, AThink, 'verified'));
    Cond(Format('verified aggregate rises (%.3f -> %.3f)', [Rb, Ra]), Above(Ra, Rb));
    F := FlipsOf(ABefore, AAfter, AThink, 'verified');
    P := SignTestP(F.B, F.C);
    Cond(Format('flips significant (%d fail->pass, %d pass->fail, p = %.4f)', [F.B, F.C, P]),
      (F.B > F.C) and (P < 0.05));
    Fall := '';
    D := DialectsOf(ABefore);
    for I := 0 to High(D) do
      if InList(D[I], K_VERIFIED) and
         Above(PassRate(CountOf(ABefore, AThink, D[I])) - 0.05, PassRate(CountOf(AAfter, AThink, D[I]))) then
        Fall := Fall + ' ' + D[I];
    Cond('no verified dialect falls by more than 5 points' + Fall, Fall = '');
    Cond(Format('tool-call check loses nothing (%d -> %d)', [CountOf(ABefore, AThink, 'tool-call').Passed,
      CountOf(AAfter, AThink, 'tool-call').Passed]),
      CountOf(AAfter, AThink, 'tool-call').Passed >= CountOf(ABefore, AThink, 'tool-call').Passed);
    if IsNan(ASpeedRatio) then
    begin
      L.Add('H49 speed at least 0.7 of before: not measured');
      Unmeasured := True;
    end
    else
      Cond(Format('H49 speed at least 0.7 of before (%.3f)', [ASpeedRatio]), ASpeedRatio >= 0.7);
    if Failed then
      L.Add('adopt: no')
    else if Unmeasured then
      L.Add('adopt: undecided')
    else
      L.Add('adopt: yes');
    Result := L.ToStringArray;
  finally
    L.Free;
  end;
end;
```

For one thinking mode, the lines in this order:

1. `verified aggregate rises (<rb> -> <ra>)`: the `verified` column's
   pass rate of `AAfter` above the one of `ABefore` (`Above`, 2.9).
2. `flips significant (<b> fail->pass, <c> pass->fail, p = <p>)`: the
   `verified` flips (2.5) — `B > C` (more fail → pass) and the exact
   sign test (2.6) below 0.05.
3. `no verified dialect falls by more than 5 points<fall>`: for each
   verified dialect of `DialectsOf(ABefore)` (2.8), it falls when its
   `ABefore` rate is above its `AAfter` rate by more than 5 points
   (evaluated as `Above(beforeRate − 0.05, afterRate)`); every such
   dialect is appended to the line, space-separated, and the condition
   fails. An empty series (NaN) can never fall, via `Above`.
4. `tool-call check loses nothing (<before> -> <after>)`: the
   `tool-call` column's `Passed` count of `AAfter` is not below the one
   of `ABefore` (it loses nothing; a gain is fine).
5. H49: `ASpeedRatio` NaN → the fixed line `H49 speed at least 0.7 of
   before: not measured`, and `Unmeasured := True` (it sets **no**
   `Failed`); otherwise the condition `ASpeedRatio >= 0.7`.

Then the verdict, exactly one line: `adopt: no` if a condition failed;
else `adopt: undecided` if H49 was not measured; else `adopt: yes`.
Each condition line is `text: yes`/`text: no` written by the local
`Cond(AText, AOk)` (which also sets `Failed := True` when `AOk` is
False). The speed and the MTP acceptance themselves (H49) are not
measured here — the ratio is the `--speed-ratio` input (3.3). `L` is
freed on every path (`try/finally`).

### 2.11 `FirstBlock`

Interface, as in the facts: the first closed fenced block of `AText`: a
line starting with three backquotes opens it, a line equal to three
backquotes (trailing spaces ignored) closes it; `ABlock` gets the lines
between, each ended by `#10`; `False` (`ABlock ''`) when there is none.

```pascal
function FirstBlock(const AText: string; out ABlock: string): Boolean;
var
  L: TStringList;
  I: Integer;
  Inside: Boolean;
begin
  Result := False;
  ABlock := '';
  Inside := False;
  L := TStringList.Create;
  try
    L.Text := AText;
    for I := 0 to L.Count - 1 do
      if not Inside then
        Inside := Copy(L[I], 1, 3) = '```'
      else if TrimRight(L[I]) = '```' then
        Exit(True)
      else
        ABlock := ABlock + L[I] + #10;
    ABlock := '';
  finally
    L.Free;
  end;
end;
```

Scans the lines (as `TStringList.Text` splits them): outside a block,
any line whose first three characters are three backquotes (an info
string like ```` ```pascal ```` qualifies) opens one — the fence line
itself is **not** part of the block; inside, a line equal to three
backquotes after `TrimRight` (trailing spaces ignored) closes it and
the function returns `True`, with `ABlock` holding the lines between,
each ended by `#10`; every other line inside is appended. The block is
the **first closed** one: text after its closing fence is ignored
(facts' UT 009: `a`/`  b` closed, then a second block `c` — the first
comes back). No closed block — no fence, an unclosed fence, or a fence
line with no closing one — resets `ABlock` to `''` and returns
`False`. `L` is freed on every path (`try/finally`).

## 3. Unit `EvalCli`

### 3.1 Constants and types (interface section, as in the facts)

```pascal
const
  K_EVAL_USAGE = 'usage: qlora_eval [options] TASKS BEFORE [AFTER] OUTDIR (see the unit header)';

type
  { An evaluation task: a Pascal task (Dialect one of the five, Expected
    its output) or a regression task (Dialect 'regression', Check
    'tool-call' with Tool, or 'contains' with Text). }
  TEvalTask = record
    Id, Dialect, Prompt, Expected, Check, Tool, Text: string;
  end;
  TEvalTasks = array of TEvalTask;

  TEvalConfig = record
    Tasks, Before, After, OutDir: string;   { After '' when not given; OutDir ends with the delimiter }
    Timeout: Integer;
    SpeedRatio: Double;                    { NaN when not given }
  end;
```

- `K_EVAL_USAGE`: printed after every usage error (4.1).
- `TEvalTask`: a Pascal task — `Dialect` one of the five (`objfpc`,
  `fpc-delphi`, `unleashed`, `delphi`, `oxygene`), `Expected` its
  expected output (`expected_stdout`); or a regression task —
  `Dialect = 'regression'`, `Check = 'tool-call'` with the expected
  tool name, or `Check = 'contains'` with the expected text (DESIGN
  8.17, the facts).
- `TEvalConfig`: the parsed command line. `After` stays `''` when the
  `AFTER` positional is not given; `OutDir` always ends with the
  trailing path delimiter (3.3). `SpeedRatio` is NaN when
  `--speed-ratio` is not given — H49 unmeasured (3.10).

### 3.2 `DefaultEvalConfig`

Interface, as in the facts: Timeout 10, SpeedRatio NaN, the rest empty.

```pascal
function DefaultEvalConfig: TEvalConfig;
begin
  Result := Default(TEvalConfig);
  Result.Timeout := 10;
  Result.SpeedRatio := NaN;
end;
```

The defaults of DESIGN 8.17: `Timeout = 10`, `SpeedRatio = NaN`
("not measured"), `Tasks`/`Before`/`After`/`OutDir` = `''`. `ParseEvalArgs`
(3.3) starts from this and fills in the given values.

### 3.3 `ParseEvalArgs`

Interface, as in the facts: `''` or the usage error. Options:
`--timeout S` (1..3600), `--speed-ratio R` (a number > 0); then
`TASKS BEFORE [AFTER] OUTDIR`.

```pascal
function ParseEvalArgs(const AArgs: array of string; out C: TEvalConfig): string;
var
  I: Integer;
  A, V: string;
  P: array of string;
  N: Integer;
  R: Double;
begin
  C := DefaultEvalConfig;
  P := nil;
  I := 0;
  while I <= High(AArgs) do
  begin
    A := AArgs[I];
    if (A = '--timeout') or (A = '--speed-ratio') then
    begin
      if I = High(AArgs) then
        Exit(A + ': value missing');
      V := AArgs[I + 1];
      Inc(I, 2);
      if A = '--timeout' then
      begin
        if not TryStrToInt(V, N) or (N < 1) or (N > 3600) then
          Exit(Format('--timeout: an integer in 1..3600, got "%s"', [V]));
        C.Timeout := N;
      end
      else
      begin
        if not TryStrToFloat(V, R, DefaultFormatSettings) or IsNan(R) or IsInfinite(R) or (R <= 0) then
          Exit(Format('--speed-ratio: a number > 0, got "%s"', [V]));
        C.SpeedRatio := R;
      end;
      Continue;
    end;
    if (Length(A) > 1) and (A[1] = '-') then
      Exit('unknown option ' + A);
    SetLength(P, Length(P) + 1);
    P[High(P)] := A;
    Inc(I);
  end;
  if (Length(P) < 3) or (Length(P) > 4) then
    Exit(Format('expected TASKS BEFORE [AFTER] OUTDIR, got %d arguments', [Length(P)]));
  C.Tasks := P[0];
  C.Before := P[1];
  if Length(P) = 4 then
    C.After := P[2];
  C.OutDir := IncludeTrailingPathDelimiter(P[High(P)]);
  Result := '';
end;
```

In order:

1. `C` starts at the defaults (3.2); the positionals are collected in
   `P`.
2. Per argument (the option names are literal — `--Timeout` is an
   unknown option):
   - `--timeout` or `--speed-ratio` **consumes the next argument**;
     nothing after the option → `<option>: value missing`.
     `--timeout` must be an integer in `1..3600`, else
     `--timeout: an integer in 1..3600, got "<v>"`.
     `--speed-ratio` must convert to a finite number greater than 0
     (NaN and ±Inf refused, a number ≤ 0 refused), else
     `--speed-ratio: a number > 0, got "<v>"`.
   - A longer-than-1 argument starting with `-` → `unknown option
     <arg>`.
   - Otherwise it is a positional.
3. The positionals must be 3 or 4: `TASKS`, `BEFORE`, `[AFTER]`,
   `OUTDIR` — else
   `expected TASKS BEFORE [AFTER] OUTDIR, got <n> arguments`.
   `C.OutDir` gets the trailing path delimiter; `C.After` stays `''`
   with 3 positionals.
4. `''` on success; otherwise the 4.1 message. `RunEvalCLI` (3.11)
   prints an error message **and** `K_EVAL_USAGE` and exits 2.

### 3.4 `Lines` (local)

```pascal
function Lines(const APath: string): TStringList;
begin
  if not FileExists(APath) then
    raise Exception.Create('no file ' + APath);
  Result := TStringList.Create;
  Result.LoadFromFile(APath);
end;
```

The lines of `APath`, or `Exception` `no file <path>` when the file
does not exist (4.2/4.3). Used by `LoadTasks` (3.5) and `JudgeFile`
(3.8).

### 3.5 `LoadTasks`

Interface, as in the facts: the tasks of `AFile`, one JSON object per
line (blank lines skipped). Raises `Exception`
`'task line <n>: <what>'` for a line that is not an object with the
string `id`, `dialect`, `prompt`, and for a Pascal task the string
`expected_stdout`, for a regression task `check` `'tool-call'` with the
string `tool` or `'contains'` with the string `text`.

```pascal
function LoadTasks(const AFile: string): TEvalTasks;
var
  L: TStringList;
  I, N: Integer;
  D: TJSONData;
  O: TJSONObject;
  T: TEvalTask;

  function Str(const AKey: string; out AValue: string): Boolean;
  var
    V: TJSONData;
  begin
    V := O.Find(AKey);
    Result := (V <> nil) and (V.JSONType = jtString);
    if Result then AValue := V.AsString else AValue := '';
  end;

  function BadLine(const AWhat: string): Exception;
  begin
    Result := Exception.CreateFmt('task line %d: %s', [I + 1, AWhat]);
  end;

begin
  Result := nil;
  N := 0;
  L := Lines(AFile);
  try
    for I := 0 to L.Count - 1 do
    begin
      if Trim(L[I]) = '' then
        Continue;
      try
        D := GetJSON(L[I], False);
      except
        on Exception do
          D := nil;
      end;
      try
        if (D = nil) or (D.JSONType <> jtObject) then
          raise BadLine('not a JSON object');
        O := TJSONObject(D);
        T := Default(TEvalTask);
        if not Str('id', T.Id) then raise BadLine('id');
        if not Str('dialect', T.Dialect) then raise BadLine('dialect');
        if not Str('prompt', T.Prompt) then raise BadLine('prompt');
        if T.Dialect = K_REGRESSION then
        begin
          Str('check', T.Check);
          if T.Check = 'tool-call' then
          begin
            if not Str('tool', T.Tool) then raise BadLine('tool');
          end
          else if T.Check = 'contains' then
          begin
            if not Str('text', T.Text) then raise BadLine('text');
          end
          else
            raise BadLine('check');
        end
        else if not Str('expected_stdout', T.Expected) then
          raise BadLine('expected_stdout');
        SetLength(Result, N + 1);
        Result[N] := T;
        Inc(N);
      finally
        D.Free;
      end;
    end;
  finally
    L.Free;
  end;
end;
```

Line by line (blank lines after `Trim` are skipped): parse with
`GetJSON(L[I], False)` — a parse failure leaves `D = nil` — then the
checks in the 4.2 order, the **first** failure raises:

1. `D` is a JSON object → `task line <n>: not a JSON object` (the line
   number is 1-based: `I + 1`).
2. The strings `id`, `dialect`, `prompt` — each missing or not a
   string (an empty string is **acceptable**) raises `task line <n>:
   <name>`.
3. `dialect = 'regression'`: the `check` must be the literal
   `tool-call` (then the string `tool`) or the literal `contains` (then
   the string `text`); any other `check` (or a missing/non-string
   `check`) raises `task line <n>: check`; the missing `tool`/`text`
   raise their own reasons.
4. Otherwise a Pascal task: the string `expected_stdout` (empty
   acceptable) → else `task line <n>: expected_stdout`.

The local `Str(key, value)` reads a string member (present and a
`jtString`, else False with `value := ''`); the local `BadLine(what)`
builds `task line <n>: <what>`. Each accepted task is appended to the
growing `Result` array. Raises `Exception` (caught by `RunEvalCLI`,
3.11 step 3 → exit 1). `D` is freed per line (`finally`), `L` at the
end (`finally`).

### 3.6 `JudgeReply` (with the local `SaveText`)

Interface, as in the facts: the judged reply `AReply` (one JSON object:
`task`, `seed`, `think`, `content`, optionally `tool_calls`, an array
of objects with `name` and `arguments`) of task `ATask`. A Pascal task:
the first fenced block of `content` verified in `ADir` as `main.pas`
with `main.expected`; `Compiled` unless there is no block or the
outcome is `compile-error`, `static-ok`, `static-reject` or `skipped`;
`Passed` when the outcome is `accepted` and the block has no
`expect-error` marker. A regression task: `Passed` when a tool call is
named `ATask.Tool` (tool-call), or `content` contains `ATask.Text`
literally (contains).

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

function JudgeReply(const ATask: TEvalTask; AReply: TJSONObject; const ADir: string;
  ATimeout: Integer; const AEnv: TEnvRec; out AOutcome: string): TJudged;
var
  B: string;
  OK: Boolean;
  Calls: TJSONArray;
  I: Integer;
begin
  Result := Default(TJudged);
  Result.Task := ATask.Id;
  Result.Dialect := ATask.Dialect;
  Result.Check := ATask.Check;
  Result.Seed := AReply.Get('seed', Int64(0));
  Result.Think := AReply.Get('think', False);
  if ATask.Dialect = K_REGRESSION then
  begin
    if ATask.Check = 'tool-call' then
    begin
      Calls := AReply.Get('tool_calls', TJSONArray(nil));
      if Calls <> nil then
        for I := 0 to Calls.Count - 1 do
          if (Calls.Items[I].JSONType = jtObject) and
             (TJSONObject(Calls.Items[I]).Get('name', '') = ATask.Tool) then
            Result.Passed := True;
    end
    else
      Result.Passed := Pos(ATask.Text, AReply.Get('content', '')) > 0;
    AOutcome := BoolToStr(Result.Passed, 'check-ok', 'check-failed');
    Exit;
  end;
  if not FirstBlock(AReply.Get('content', ''), B) then
  begin
    AOutcome := 'no-block';
    Exit;
  end;
  ForceDirectories(ADir);
  SaveText(ADir + 'main.pas', B);
  SaveText(ADir + 'main.expected', ATask.Expected);
  AOutcome := VerifyFile(ADir + 'main.pas', ATask.Dialect, ATimeout, AEnv, ADir + 'build', OK).Split([#9])[3];
  Result.Compiled := (AOutcome <> OUT_COMPILE_ERR) and (AOutcome <> OUT_STATIC_OK) and
    (AOutcome <> OUT_STATIC_REJECT) and (AOutcome <> OUT_SKIPPED);
  Result.Passed := (AOutcome = OUT_ACCEPTED) and (ExtractExpectErrorText(B) = '');
end;
```

`Result` is filled with the task's `Task`, `Dialect`, `Check`, and the
reply's `Seed` (`0` when absent) and `Think` (`False` when absent),
then:

- **A regression task** (`ATask.Dialect = 'regression'`):
  - `check = 'tool-call'`: `Passed` when `tool_calls` (absent →
    `TJSONArray(nil)`) has **any** element that is a JSON object whose
    `name` equals `ATask.Tool` literally. The content is not examined.
  - otherwise (`contains`): `Passed := Pos(ATask.Text, content) > 0` —
    literal, case-sensitive. The tool calls are not examined.
  - `AOutcome := 'check-ok'` when `Passed`, `'check-failed'`
    otherwise; then `Exit`. `Compiled` stays False.
- **A Pascal task** (the five dialects, including the `delphi`
  approximation and `oxygene`):
  1. `FirstBlock` (2.11) on `content`; no closed block →
     `AOutcome := 'no-block'`, `Exit` (both flags False).
  2. `ForceDirectories(ADir)`; the block is written as `ADir +
     'main.pas'` and the task's `expected_stdout` as `ADir +
     'main.expected'` (the local `SaveText`: a `TFileStream` in
     `fmCreate`, writing the whole text when it is not empty, else
     creating an empty file; freed in `finally`).
  3. `AOutcome` is the **4th cell (index 3)** of the verifier's
     finished TSV line — its outcome — of
     `VerifyFile(main.pas, ATask.Dialect, ATimeout, AEnv, ADir +
     'build', OK)` (T06, its detailed design 2.14/2.16).
  4. `Compiled` = the outcome is **not** `OUT_COMPILE_ERR`
     (`compile-error`), **not** `OUT_STATIC_OK` (`static-ok`), **not**
     `OUT_STATIC_REJECT` (`static-reject`), **not** `OUT_SKIPPED`
     (`skipped`).
  5. `Passed` = the outcome **is** `OUT_ACCEPTED` (`accepted`) and
     `ExtractExpectErrorText(B)` is `''` — the block carries no
     `expect-error:` marker.

### 3.7 `TaskIndex` (local)

```pascal
function TaskIndex(const ATasks: TEvalTasks; const AId: string): Integer;
var
  I: Integer;
begin
  for I := 0 to High(ATasks) do
    if ATasks[I].Id = AId then
      Exit(I);
  Result := -1;
end;
```

The first index whose `Id` equals `AId` literally, else `-1` (an empty
task array gives `-1`).

### 3.8 `JudgeFile` (local)

```pascal
{ judges every reply of AFile; result lines to AResults with AModel first }
function JudgeFile(const AFile, AModel: string; const ATasks: TEvalTasks; const AWorkRoot: string;
  ATimeout: Integer; const AEnv: TEnvRec; AResults: TStrings): TJudgedArray;
var
  L: TStringList;
  I, K, N: Integer;
  D: TJSONData;
  Outcome: string;
  J: TJudged;
begin
  Result := nil;
  N := 0;
  L := Lines(AFile);
  try
    for I := 0 to L.Count - 1 do
    begin
      if Trim(L[I]) = '' then
        Continue;
      try
        D := GetJSON(L[I], False);
      except
        on Exception do
          D := nil;
      end;
      try
        if (D = nil) or (D.JSONType <> jtObject) then
          raise Exception.CreateFmt('%s line %d: not a JSON object', [AFile, I + 1]);
        K := TaskIndex(ATasks, TJSONObject(D).Get('task', ''));
        if K < 0 then
          raise Exception.CreateFmt('%s line %d: no task %s', [AFile, I + 1, TJSONObject(D).Get('task', '')]);
        J := JudgeReply(ATasks[K], TJSONObject(D), IncludeTrailingPathDelimiter(AWorkRoot) + AModel +
          PathDelim + IntToStr(I + 1) + PathDelim, ATimeout, AEnv, Outcome);
        SetLength(Result, N + 1);
        Result[N] := J;
        Inc(N);
        AResults.Add(Format('%s'#9'%s'#9'%d'#9'%s'#9'%s'#9'%s'#9'%s'#9'%s',
          [AModel, J.Task, J.Seed, BoolToStr(J.Think, 'on', 'off'), J.Dialect, Outcome,
           BoolToStr(J.Compiled, '1', '0'), BoolToStr(J.Passed, '1', '0')]));
      finally
        D.Free;
      end;
    end;
  finally
    L.Free;
  end;
end;
```

Judges **every** non-blank line of `AFile`, in file order; each judged
reply is appended to `Result` and its TSV line added to `AResults`
(with `AModel` — `'before'` or `'after'` — first): `model`, `task`,
`seed`, `think` as `on`/`off`, `dialect`, the `outcome` (the
verifier's, or `no-block`, `check-ok`, `check-failed` — 4.4),
`compiled` and `passed` as `1`/`0`. The per-reply work directory is
`AWorkRoot + AModel + PathDelim + <line no., 1-based> + PathDelim`.
The checks in 4.3 order, the first failure raises: the line is a JSON
object → `<file> line <n>: not a JSON object`; its `task` name is in
the task list → `<file> line <n>: no task <id>`. A reply of an unknown
task is therefore an **error**, not a skip (facts' UT 107). `D` is
freed per line (`finally`), `L` at the end (`finally`).

### 3.9 `Cell` (local)

```pascal
function Cell(const C: TCount; ACompiled: Boolean): string;
begin
  if C.N = 0 then
    Result := '-'
  else if ACompiled then
    Result := Format('%d/%d', [C.Compiled, C.N])
  else
    Result := Format('%d/%d', [C.Passed, C.N]);
end;
```

One report cell: `-` when the column has no replies for this model and
thinking mode (`N = 0`), else `Compiled/N` for a compiled cell and
`Passed/N` for a passed cell.

### 3.10 `ReportMode` (local)

```pascal
procedure ReportMode(AReport: TStrings; const ABefore, AAfter: TJudgedArray; AHasAfter, AThink: Boolean;
  ASpeedRatio: Double);
var
  Cols: TStringArray;
  I: Integer;
  F: TFlips;
  Cb, Ca: TCount;
  Row: string;
  A: TStringArray;
begin
  AReport.Add('');
  AReport.Add('## Thinking ' + BoolToStr(AThink, 'on', 'off'));
  AReport.Add('');
  if AHasAfter then
  begin
    AReport.Add('| Column | Before compiled | Before passed | After compiled | After passed | Fail -> pass | Pass -> fail |');
    AReport.Add('|---|---|---|---|---|---|---|');
  end
  else
  begin
    AReport.Add('| Column | Compiled | Passed |');
    AReport.Add('|---|---|---|');
  end;
  Cols := DialectsOf(ABefore);
  SetLength(Cols, Length(Cols) + 3);
  Cols[High(Cols) - 2] := 'verified';
  Cols[High(Cols) - 1] := 'tool-call';
  Cols[High(Cols)] := 'contains';
  for I := 0 to High(Cols) do
  begin
    Cb := CountOf(ABefore, AThink, Cols[I]);
    Ca := CountOf(AAfter, AThink, Cols[I]);
    Row := Cols[I];
    if Pos(',' + Cols[I] + ',', ',' + K_APPROX + ',') > 0 then
      Row := Row + ' (approximation)';
    if (Cols[I] = 'tool-call') or (Cols[I] = 'contains') then
      Row := Format('| %s | - | %s', [Row, Cell(Cb, False)])
    else
      Row := Format('| %s | %s | %s', [Row, Cell(Cb, True), Cell(Cb, False)]);
    if AHasAfter then
    begin
      F := FlipsOf(ABefore, AAfter, AThink, Cols[I]);
      if (Cols[I] = 'tool-call') or (Cols[I] = 'contains') then
        Row := Row + Format(' | - | %s | %d | %d', [Cell(Ca, False), F.B, F.C])
      else
        Row := Row + Format(' | %s | %s | %d | %d', [Cell(Ca, True), Cell(Ca, False), F.B, F.C]);
    end;
    AReport.Add(Row + ' |');
  end;
  if AHasAfter then
  begin
    AReport.Add('');
    A := Adoption(ABefore, AAfter, AThink, ASpeedRatio);
    for I := 0 to High(A) do
      AReport.Add('- ' + A[I]);
  end;
end;
```

One `## Thinking on|off` section of `report.md`:

1. A blank line, the heading, a blank line.
2. The table header: with `AHasAfter` the before/after and flips
   columns (`| Column | Before compiled | Before passed | After
   compiled | After passed | Fail -> pass | Pass -> fail |`), without
   it the simple one (`| Column | Compiled | Passed |`).
3. The columns: `DialectsOf(ABefore)` (2.8, ascending, no
   `regression`) followed by `verified`, `tool-call`, `contains`.
4. One row per column:
   - the label is the column name; a column in `K_APPROX` (`delphi`)
     gets the suffix ` (approximation)`.
   - the regression columns (`tool-call`, `contains`) have no compiled
     cell (always `-`) and a passed cell `Cell(Cb, False)`; the other
     columns have `Cell(Cb, True)` (compiled) and `Cell(Cb, False)`
     (passed).
   - with `AHasAfter`: the `AAfter` cells the same way, plus the pair's
     `F.B` and `F.C` from `FlipsOf` (2.5) for that column.
5. With `AHasAfter`: a blank line, then every line of `Adoption` (2.10)
   prefixed by `- `.

Without `AHasAfter`, `AAfter` may be empty (nil); `CountOf` and
`FlipsOf` over an empty array simply do not loop.

### 3.11 `RunEvalCLI` (step by step)

Interface, as in the facts: `AOut`/`AErr` caller-owned; `AWorkRoot`:
where the verifier works; the result is the exit code (0 done, 1 error,
2 usage).

```pascal
function RunEvalCLI(const AArgs: array of string; const AEnv: TEnvRec;
  const AWorkRoot: string; AOut, AErr: TStrings): Integer;
var
  C: TEvalConfig;
  Msg: string;
  Tasks: TEvalTasks;
  Bj, Aj: TJudgedArray;
  Results, Report: TStringList;
  I, NP: Integer;
begin
  Msg := ParseEvalArgs(AArgs, C);
  if Msg <> '' then
  begin
    AErr.Add(Msg);
    AErr.Add(K_EVAL_USAGE);
    Exit(2);
  end;
  Result := 1;
  Results := TStringList.Create;
  Report := TStringList.Create;
  try
    try
      Tasks := LoadTasks(C.Tasks);
      Results.Add('model'#9'task'#9'seed'#9'think'#9'dialect'#9'outcome'#9'compiled'#9'passed');
      Bj := JudgeFile(C.Before, 'before', Tasks, AWorkRoot, C.Timeout, AEnv, Results);
      Aj := nil;
      if C.After <> '' then
        Aj := JudgeFile(C.After, 'after', Tasks, AWorkRoot, C.Timeout, AEnv, Results);
      NP := 0;
      for I := 0 to High(Tasks) do
        if Tasks[I].Dialect <> K_REGRESSION then
          Inc(NP);
      Report.Add('# Evaluation report');
      Report.Add('');
      Report.Add(Format('Tasks: %d Pascal, %d regression. Replies: before %d, after %d.',
        [NP, Length(Tasks) - NP, Length(Bj), Length(Aj)]));
      Report.Add('Compiled and passed as passed/replies; "verified" is ' + K_VERIFIED +
        ' together; ' + K_APPROX + ' is an approximation, not in it.');
      ReportMode(Report, Bj, Aj, C.After <> '', True, C.SpeedRatio);
      ReportMode(Report, Bj, Aj, C.After <> '', False, C.SpeedRatio);
      ForceDirectories(C.OutDir);
      Results.SaveToFile(C.OutDir + 'results.tsv');
      Report.SaveToFile(C.OutDir + 'report.md');
      AOut.Add(Format('%d replies judged into %s', [Length(Bj) + Length(Aj), C.OutDir]));
      Result := 0;
    except
      on E: Exception do
        AErr.Add(E.Message);
    end;
  finally
    Report.Free;
    Results.Free;
  end;
end;
```

Step by step (the `AOut`/`AErr` lines as in section 4):

1. `Msg := ParseEvalArgs(AArgs, C)`. When `Msg <> ''`: `AErr` gets the
   message and `K_EVAL_USAGE`, and the function returns **2** (4.1).
2. Otherwise prepare the error path: `Result := 1` — it stays 1 when a
   later step raises — and create `Results` and `Report` (both
   `TStringList`, freed in the outer `finally`).
3. `Tasks := LoadTasks(C.Tasks)` — 4.2 messages, raised.
4. `Results` gets the header line: `model`, `task`, `seed`, `think`,
   `dialect`, `outcome`, `compiled`, `passed` (tab-separated).
5. `Bj := JudgeFile(C.Before, 'before', Tasks, AWorkRoot, C.Timeout,
   AEnv, Results)` — 4.3 messages, raised.
6. `Aj := nil`; when `C.After <> ''`,
   `Aj := JudgeFile(C.After, 'after', ...)` — 4.3 messages, raised.
7. `NP` = the number of tasks that are **not** regression. The report
   head: `# Evaluation report`, a blank line,
   `Tasks: <np> Pascal, <nr> regression. Replies: before <b>, after
   <a>.`, and
   `Compiled and passed as passed/replies; "verified" is
   objfpc,fpc-delphi,unleashed,oxygene together; delphi is an
   approximation, not in it.`.
8. The two sections: `ReportMode(Report, Bj, Aj, C.After <> '', True,
   C.SpeedRatio)` (thinking on first), then the same with `False`
   (thinking off) (3.10). `AHasAfter = (C.After <> '')`.
9. `ForceDirectories(C.OutDir)`; `Results.SaveToFile(C.OutDir +
   'results.tsv')`; `Report.SaveToFile(C.OutDir + 'report.md')`.
10. `AOut` gets `<n> replies judged into <outdir>` with
    `n = Length(Bj) + Length(Aj)`; `Result := 0` (4.4).
11. Any raised `Exception` on the way adds its `Message` to `AErr`
    alone (4.3) and the exit stays **1**; `Report` and `Results` are
    freed in `finally` on every path.

## 4. Checks and messages (in this order)

All the messages exactly as in the T20 facts ("Checks and messages
(in this order)"). The task and reply errors are raised as
`Exception` and the run adds the message alone to `AErr` (exit 1);
the usage errors are returned and the run prints the message **and**
`K_EVAL_USAGE` (exit 2). A UT row that expects a message compares the
**whole** message (`AssertEquals`, the facts).

### 4.1 The usage errors (`ParseEvalArgs`, 3.3; `RunEvalCLI` prints the
message **and** `K_EVAL_USAGE`, **exit 2**), in this order

| # | Where | Condition | Message |
|---|---|---|---|
| 1 | `--timeout` / `--speed-ratio` | the option is the last argument (the value is missing) | `<option>: value missing` (e.g. `--timeout: value missing`) |
| 2 | `--timeout` | the value is not an integer in 1..3600 | `--timeout: an integer in 1..3600, got "<v>"` |
| 3 | `--speed-ratio` | the value is not a number, NaN, ±Inf, or not greater than 0 (NaN and ±Inf refused) | `--speed-ratio: a number > 0, got "<v>"` |
| 4 | any argument | longer than one character and starting with `-` (literal: `--Timeout` is unknown) | `unknown option <arg>` |
| 5 | after the loop | the positionals are not 3 or 4 | `expected TASKS BEFORE [AFTER] OUTDIR, got <n> arguments` |

Followed on `AErr` by
`usage: qlora_eval [options] TASKS BEFORE [AFTER] OUTDIR (see the unit
header)` (`K_EVAL_USAGE`).

### 4.2 The task errors (`LoadTasks`, 3.5; `Exception`, **exit 1** via
`RunEvalCLI`, 3.11 step 3), in this order

| # | Where | Condition | Message |
|---|---|---|---|
| 1 | `Lines` (3.4) | the tasks file does not exist | `no file <path>` |
| 2 | per line | the line is not JSON, or not an object | `task line <n>: not a JSON object` |
| 3 | per line | `id` missing or not a string (an empty string is acceptable) | `task line <n>: id` |
| 4 | per line | `dialect` missing or not a string | `task line <n>: dialect` |
| 5 | per line | `prompt` missing or not a string | `task line <n>: prompt` |
| 6 | a regression task | `check` is `tool-call` and `tool` missing or not a string | `task line <n>: tool` |
| 7 | a regression task | `check` is `contains` and `text` missing or not a string | `task line <n>: text` |
| 8 | a regression task | `check` missing, not a string, or neither `tool-call` nor `contains` (literal) | `task line <n>: check` |
| 9 | a Pascal task (dialect not `regression`) | `expected_stdout` missing or not a string (an empty string is acceptable) | `task line <n>: expected_stdout` |

### 4.3 The reply errors (`RunEvalCLI`, 3.11 steps 3–6; **exit 1**, the
message alone on `AErr`), in this order

| # | Where | Condition | Message |
|---|---|---|---|
| 1 | step 3, `LoadTasks` | a task-file error | the 4.2 messages, in the 4.2 order |
| 2 | step 5 or 6, `Lines` | the BEFORE or AFTER replies file does not exist | `no file <path>` |
| 3 | step 5 or 6, `JudgeFile` | a replies line is not JSON, or not an object | `<file> line <n>: not a JSON object` |
| 4 | step 5 or 6, `JudgeFile` | the reply's `task` name is not in the task list (a reply of an unknown task is an error, not a skip) | `<file> line <n>: no task <id>` |

### 4.4 The success output (**exit 0**, on `AOut`) and the output files

On success, `AOut` gets one line:

```
<n> replies judged into <outdir>
```

(`n` = the before replies plus the after replies; `<outdir>` as stored,
with its trailing delimiter). The files in `OUTDIR`:

- `results.tsv`: the header
  `model task seed think dialect outcome compiled passed`
  (tab-separated), then one line per reply in file order, the before
  replies first and then the after replies: `model` (`before`/`after`),
  `task`, `seed`, `think` as `on`/`off`, `dialect`, the `outcome` —
  the verifier's, or `no-block`, `check-ok`, `check-failed` — and
  `compiled` and `passed` as `1`/`0` (3.8).
- `report.md`: `# Evaluation report`, a blank line,
  `Tasks: <np> Pascal, <nr> regression. Replies: before <b>, after
  <a>.`, the legend line
  `Compiled and passed as passed/replies; "verified" is
  objfpc,fpc-delphi,unleashed,oxygene together; delphi is an
  approximation, not in it.`, and then per thinking mode — thinking on
  first, then off — the section of 3.10: the `## Thinking on|off`
  heading, the table, and, with `AFTER`, the adoption lines of 2.10
  (3.11 steps 7–9).

## 5. The facts' decisions (DESIGN 8.17, "As written for T20") and
where they are handled

| Decision of the facts | Handled by |
|---|---|
| the replies are **recorded outside the program** (`tests/ita/T20/ask.bb`: Ollama's `/api/chat`, the model's own renderer and parser, the deployed sampling parameters, seeds 1 and 2, thinking on and off, the tool `get_weather` for a tool-call task) — one reply per line: `task`, `seed`, `think`, `content`, `tool_calls` (`[{name, arguments}]`, may be missing); `qlora_eval` judges a recorded file, so the judgement is repeatable and the slow generation is not repeated | `JudgeFile` (3.8), `JudgeReply` (3.6); the reply format of 4.3/4.4 |
| a task per line: `id`, `dialect` (a Pascal dialect, or `regression`), `prompt`; a Pascal task `expected_stdout`; a regression task `check` `tool-call` with `tool`, or `contains` with `text` | `LoadTasks` (3.5), `TEvalTask` (3.1), the 4.2 order |
| judging a **Pascal** reply: its first closed fenced block (2.11) is the source (8.17), written as `main.pas` with `main.expected` and verified as the task's dialect; `Compiled` unless there is no block or the outcome is `compile-error`, `static-ok`, `static-reject` or `skipped`; `Passed` when the outcome is `accepted` and the block has no `expect-error:` marker | `JudgeReply` (3.6), `FirstBlock` (2.11) |
| judging a **regression** reply: passed when a tool call is named the task's tool (tool-call), or the content contains the task's text literally (contains) | `JudgeReply` (3.6) |
| the columns: each dialect present, `verified` (`objfpc`, `fpc-delphi`, `unleashed`, `oxygene` together; Delphi is marked an approximation and is not in it), `tool-call`, `contains`; the flips pair the replies of the two models by task and seed within a thinking mode (a reply without a partner is not counted); the sign test is exact and two-sided: `min(1, 2 · Σ_{k ≤ min(b,c)} C(b+c, k) / 2^(b+c))`, 1 when `b + c = 0` | `InColumn` (2.3), `CountOf` (2.4), `FlipsOf` (2.5), `SignTestP` (2.6), `ReportMode` (3.10) |
| the **adoption rule** (8.17) per thinking mode, one line per condition: the verified aggregate rises; the flips are significant (more fail → pass, p < 0.05); no verified dialect falls by more than 5 points; the tool-call check loses nothing; H49 (`--speed-ratio`, at least 0.7; not measured without it). Verdict `adopt: yes`, `no` (a condition failed) or `undecided` (H49 not measured, nothing failed). An empty series (NaN rate) neither rises nor falls | `Adoption` (2.10), `PassRate` (2.7), `Above` (2.9) |
| the speed and the MTP acceptance themselves (H49) are **not** measured here: the ratio is an input (`--speed-ratio`) | `ParseEvalArgs` (3.3), `TEvalConfig.SpeedRatio` (3.1), `Adoption` (2.10) |

## 6. Change history

| Date | Change | Reason |
|---|---|---|
| T20a (initial) | First edition: module composition; the two units function by function (verbatim interfaces and implementations of the facts, in the facts' order); the run step by step; the checks and messages in their order; the facts' decisions and where they are handled. | new units and program (T20; documents only in T20a — no source code, no tests) |
