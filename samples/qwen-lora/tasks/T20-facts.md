# T20 facts (checked by the harness owner)

T20 is the evaluation program (DESIGN 8.17 "Runs", "Report", "Adoption
rule", 8.18): the unit `EvalReport` (the pure parts: counts per column,
paired flips, the exact sign test, the adoption rule, the first fenced
block), the unit `EvalCli` (the command line, the tasks, judging the
recorded replies, the report) and the program `src/qlora_eval.pas` (the
harness owner's thin wrapper, complete; like `qlora_verify` it reads
`QLORA_FPC`, `QLORA_FPC_UNLEASHED` and `HOME`). It uses `DialectVerify`
(T06) only.

DESIGN 8.18's acceptance for T20: "against the rehearsal model: the report
of 8.17 from a fixed transcript of replies". Decisions of the harness
owner for this task (recorded in DESIGN 8.17 as "As written for T20"):
- **The replies are recorded outside the program** (a Babashka script,
  `tests/ita/T20/ask.bb`: Ollama's `/api/chat`, the model's own renderer
  and parser, the deployed sampling parameters, seeds 1 and 2, thinking on
  and off, the tool `get_weather` for a tool-call task). `qlora_eval`
  judges a recorded file, so the judgement is repeatable and the slow
  generation is not repeated. One reply per line: `task`, `seed`, `think`,
  `content`, `tool_calls` (`[{name, arguments}]`, may be missing).
- **A task** per line: `id`, `dialect` (a Pascal dialect, or `regression`),
  `prompt`; a Pascal task `expected_stdout`; a regression task `check`
  `tool-call` with `tool`, or `contains` with `text`.
- **Judging a Pascal reply**: its first closed fenced block is the source
  (8.17), written as `main.pas` with `main.expected` and verified as the
  task's dialect. Compiled unless there is no block or the outcome is
  `compile-error`, `static-ok`, `static-reject` or `skipped`; passed when
  the outcome is `accepted` and the block has no `expect-error:` marker.
  **A regression reply**: passed when a tool call is named the task's
  tool, or the content contains the task's text (literally).
- **The columns**: each dialect present, `verified` (objfpc, fpc-delphi,
  unleashed, oxygene together; Delphi is marked an approximation and is
  not in it), `tool-call`, `contains`. Flips pair the replies of the two
  models by task and seed within a thinking mode. The sign test is exact
  and two-sided: min(1, 2 · Σ_{k ≤ min(b,c)} C(b+c, k) / 2^(b+c)).
- **The adoption rule** (8.17) per thinking mode, one line per condition:
  the verified aggregate rises; the flips are significant (more fail →
  pass, p < 0.05); no verified dialect falls by more than 5 points; the
  tool-call check loses nothing; H49 (`--speed-ratio`, at least 0.7; not
  measured without it). Verdict `adopt: yes`, `no` (a condition failed)
  or `undecided` (H49 not measured, nothing failed). An empty series
  (NaN rate) neither rises nor falls.
- The speed and the MTP acceptance themselves (H49) are not measured
  here: the ratio is an input.

Measured by the harness owner with a prototype written exactly as below:
- UT: every row of the tables below passes with `-Cr`; both units then
  have 100 % line coverage (123 and 238 lines); mutants are caught (delphi counted as verified: EvalReport 001, 002, EvalCli 106; the seed left out of the pairing: EvalReport 002; a one-sided sign test: EvalReport 003, 005-007, EvalCli 106; a 10-point fall allowed: EvalReport 006; the tool-call check requiring a gain: 005, 007; H49 at 0.6: 006; undecided reported as yes: 007; an unclosed block taken: EvalReport 009, EvalCli 105, 106; the pitfall clause dropped: EvalCli 105; every reply counted as compiled: 105; contains without case: 105; any tool accepted: 105; a reply of an unknown task skipped: 107).
- ITa (cases below): all 0; 112 replies (56 per model; the rehearsal model passes none of the eleven Pascal tasks before or after the M2 merge, two of its unleashed replies compile; the 14 passes are regression checks); about 5 s per run.

These facts are final; do not investigate further or write probes.

## Unit `EvalReport` (interface, final; in the skeleton)

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

{ The counts of AJudged for thinking mode AThink and column AColumn: a
  dialect name, 'verified' (the dialects of K_VERIFIED together), or a
  regression check name ('tool-call', 'contains'). }
function CountOf(const AJudged: TJudgedArray; AThink: Boolean; const AColumn: string): TCount;

{ The flips of AColumn between ABefore and AAfter for AThink: replies
  paired by task and seed; a reply without a partner is not counted. }
function FlipsOf(const ABefore, AAfter: TJudgedArray; AThink: Boolean; const AColumn: string): TFlips;

{ The exact two-sided sign test of B against C:
  min(1, 2 · sum over k = 0..min(B, C) of C(B + C, k) / 2^(B + C)); 1 when B + C = 0. }
function SignTestP(AB, AC: Integer): Double;

{ The rate of ACount (Passed / N); NaN when N = 0. }
function PassRate(const ACount: TCount): Double;

{ The adoption rule of 8.17 for AThink, one line per condition
  ('yes'/'no'/'not measured' at the end), then the verdict line
  'adopt: yes', 'adopt: no' or 'adopt: undecided' (a condition not
  measured and none failed). ASpeedRatio: the after/before generation
  speed (H49, at least 0.7); NaN when not measured. }
function Adoption(const ABefore, AAfter: TJudgedArray; AThink: Boolean;
  ASpeedRatio: Double): TStringArray;

{ The dialects of AJudged (not the regression ones), ascending. }
function DialectsOf(const AJudged: TJudgedArray): TStringArray;

{ The first closed fenced block of AText: a line starting with three
  backquotes opens it, a line equal to three backquotes (trailing spaces
  ignored) closes it; ABlock gets the lines between, each ended by #10.
  False (ABlock '') when there is none. }
function FirstBlock(const AText: string; out ABlock: string): Boolean;
```

## Implementation of `EvalReport` (write it exactly so)

```pascal
implementation

function InList(const AValue, AList: string): Boolean;
begin
  Result := Pos(',' + AValue + ',', ',' + AList + ',') > 0;
end;

function InColumn(const J: TJudged; const AColumn: string): Boolean;
begin
  if AColumn = 'verified' then
    Result := InList(J.Dialect, K_VERIFIED)
  else if J.Dialect = K_REGRESSION then
    Result := J.Check = AColumn
  else
    Result := J.Dialect = AColumn;
end;

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

function PassRate(const ACount: TCount): Double;
begin
  if ACount.N = 0 then
    Result := NaN
  else
    Result := ACount.Passed / ACount.N;
end;

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

{ A > B, False when either is NaN (an empty series). }
function Above(A, B: Double): Boolean;
begin
  Result := not IsNan(A) and not IsNan(B) and (A > B);
end;

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

## Unit `EvalCli` (interface, final; in the skeleton)

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

{ Timeout 10, SpeedRatio NaN, the rest empty. }
function DefaultEvalConfig: TEvalConfig;

{ '' or the usage error. Options: --timeout S (1..3600), --speed-ratio R
  (a number > 0); then TASKS BEFORE [AFTER] OUTDIR. }
function ParseEvalArgs(const AArgs: array of string; out C: TEvalConfig): string;

{ The tasks of AFile, one JSON object per line (blank lines skipped).
  Raises Exception 'task line <n>: <what>' for a line that is not an
  object with the string id, dialect, prompt, and for a Pascal task the
  string expected_stdout, for a regression task check 'tool-call' with
  the string tool or 'contains' with the string text. }
function LoadTasks(const AFile: string): TEvalTasks;

{ The judged reply AReply (one JSON object: task, seed, think, content,
  optionally tool_calls, an array of objects with name and arguments) of task ATask. A Pascal
  task: the first fenced block of content verified in ADir as main.pas
  with main.expected; Compiled unless there is no block or the outcome is
  compile-error, static-ok, static-reject or skipped; Passed when the
  outcome is accepted and the block has no expect-error marker. A
  regression task: Passed when a tool call is named ATask.Tool
  (tool-call), or content contains ATask.Text literally (contains). }
function JudgeReply(const ATask: TEvalTask; AReply: TJSONObject; const ADir: string;
  ATimeout: Integer; const AEnv: TEnvRec; out AOutcome: string): TJudged;

{ AOut / AErr caller-owned; AWorkRoot: where the verifier works; the
  result is the exit code (0 done, 1 error, 2 usage). }
function RunEvalCLI(const AArgs: array of string; const AEnv: TEnvRec;
  const AWorkRoot: string; AOut, AErr: TStrings): Integer;
```

## Implementation of `EvalCli` (write it exactly so)

```pascal
implementation

function DefaultEvalConfig: TEvalConfig;
begin
  Result := Default(TEvalConfig);
  Result.Timeout := 10;
  Result.SpeedRatio := NaN;
end;

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

function Lines(const APath: string): TStringList;
begin
  if not FileExists(APath) then
    raise Exception.Create('no file ' + APath);
  Result := TStringList.Create;
  Result.LoadFromFile(APath);
end;

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

{ ---- the run ---- }

function TaskIndex(const ATasks: TEvalTasks; const AId: string): Integer;
var
  I: Integer;
begin
  for I := 0 to High(ATasks) do
    if ATasks[I].Id = AId then
      Exit(I);
  Result := -1;
end;

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

function Cell(const C: TCount; ACompiled: Boolean): string;
begin
  if C.N = 0 then
    Result := '-'
  else if ACompiled then
    Result := Format('%d/%d', [C.Compiled, C.N])
  else
    Result := Format('%d/%d', [C.Passed, C.N]);
end;

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

## The program `src/qlora_eval.pas` (harness owner's; complete)

It reads the verifier's compilers from the environment (as
`qlora_verify`), forwards the command line to `RunEvalCLI` with the work
root `<temp>/qlora-eval-<pid>`, prints the output lines to stdout and the
error lines to stderr, and exits with the code. Nobody changes it.

## Checks and messages (in this order)

`ParseEvalArgs` (the usage error; `RunEvalCLI` prints it and
`K_EVAL_USAGE`, exit 2): `<option>: value missing`; `--timeout: an integer
in 1..3600, got "<v>"`; `--speed-ratio: a number > 0, got "<v>"` (NaN and
±Inf refused); `unknown option <arg>`; `expected TASKS BEFORE [AFTER]
OUTDIR, got <n> arguments`.

`LoadTasks` (Exception): `no file <path>`; per line `task line <n>: not a
JSON object`, `.. id`, `.. dialect`, `.. prompt`; a regression task `..
tool`, `.. text`, `.. check`; a Pascal task `.. expected_stdout`.

`RunEvalCLI` (exit 1, the message alone on AErr): the task errors; `no
file <path>`; `<file> line <n>: not a JSON object`; `<file> line <n>: no
task <id>`. Success, exit 0: AOut `<n> replies judged into <outdir>`;
`results.tsv` (a header, then per reply in file order, before then after:
model, task, seed, think `on`/`off`, dialect, outcome — the verifier's, or
`no-block`, `check-ok`, `check-failed` — compiled and passed as 1/0);
`report.md` (the totals, then per thinking mode, on first, the table and,
with AFTER, the adoption lines).

## UT fixture and helpers (harness owner's; nobody changes them)

`tests/ut/EvalReport_test.pas` (skeleton: helpers, runner, no test
methods): `J(Task, Dialect, Seed, Think, Compiled, Passed[, Check])` (one
TJudged); `Arr([...])`; `Series(Dialect, N, Passed)` (N Pascal replies,
tasks `<dialect>-1..N`, seed 1, thinking on, the first `Passed` passed);
`Cat(A, B)`; `Joined(A)` (joined by `|`).

`tests/ut/EvalCli_test.pas` (skeleton: helpers, runner, no test methods):
the constant `PSum` (prints `10`); `NewDir`; `SaveLines`; `FileText`;
`TaskLine(Id, Dialect, Expected)`; `ReplyLine(Task, Seed, Think, Src[,
Tool])` (content `Here:` + the fenced source + `Done.`, or `I cannot.`
for an empty source; a tool call when Tool is given); `Task(Id, Dialect,
Expected[, Check, Tool, Text])`; `Judge(Task, Line, Outcome)` (JudgeReply
in a fresh directory, timeout 10); `RunCli(Args, Out, Err)` (RunEvalCLI
with `Env` and the work root `<TmpDir>work`). Variables `TmpDir`, `Env`
(`ReadEnvRec('', '', HOME)`: the stock `fpc`).

## Unit tests (rows; every row has one method, every method belongs to a row)

Error rows compare the whole message (`AssertEquals`).

`EvalReport`, stage 1 (task T20b1); 001–009.

| Row | Input | Expected |
|---|---|---|
| 001 | `CountOf` of a, b, c objfpc (passed, compiled only, neither), d fpc-delphi, e delphi, f unleashed (failed), g oxygene, a objfpc thinking off, regression t, u (tool-call, passed, failed), w (contains, passed); columns objfpc, verified, delphi, tool-call, contains (thinking on), verified (off), `pascal` | 3/2/1 (N/compiled/passed); 6/4/3; N 1; N 2, passed 1; N 1; N 1; N 0 |
| 002 | (a1 = task a, seed 1) before a1 (fail), a2 (pass), b1 (pass), c1 (fail), d1 (fail, no partner), a1 off (fail), x1 delphi (fail); after a1 (pass), a2 (fail), b1, c1 (pass), a1 off (pass), x1 (pass): `FlipsOf` objfpc on, verified on, objfpc off, delphi on | N 4, B 2, C 1; N 4; N 1, B 1; B 1 |
| 003 | `SignTestP` (0,0), (5,0), (0,6), (10,2), (2,10), (3,3), (39,1) | 1; 0.0625; 0.03125; 158/4096 (twice); 1; 2 · 41 / 2^40 |
| 004 | `PassRate` of N 0; of N 4, passed 3 | NaN; 0.75 |
| 005 | before `Series('objfpc', 10, 2)` + a passed tool call; after `Series('objfpc', 10, 9)` + the same; ratio 0.8 | `verified aggregate rises (0.200 -> 0.900): yes`, `flips significant (7 fail->pass, 0 pass->fail, p = 0.0156): yes`, `no verified dialect falls by more than 5 points: yes`, `tool-call check loses nothing (1 -> 1): yes`, `H49 speed at least 0.7 of before (0.800): yes`, `adopt: yes` |
| 006 | before objfpc 10/2 + fpc-delphi 10/10 + tool call passed; after objfpc 10/10 + fpc-delphi 10/9 + tool call failed; ratio 0.65 | `.. (0.600 -> 0.950): yes`, `.. (8 fail->pass, 1 pass->fail, p = 0.0391): yes`, `no verified dialect falls by more than 5 points fpc-delphi: no`, `tool-call check loses nothing (1 -> 0): no`, `H49 .. (0.650): no`, `adopt: no` |
| 007 | objfpc 10/2 → 10/9 without tool calls; ratio NaN; then thinking off (empty), ratio 1 | line 2 `flips significant (7 fail->pass, 0 pass->fail, p = 0.0156): yes`, `.. (0 -> 0): yes`, `H49 speed at least 0.7 of before: not measured`, `adopt: undecided`; line 6 `adopt: no` |
| 008 | `DialectsOf` of unleashed, objfpc, regression, delphi, objfpc; of nil | `delphi\|objfpc\|unleashed`; none |
| 009 | `FirstBlock` of `text`, a block `a`/`  b` closed by ```` ```  ````, then a block `c`; of an unclosed block; of `no block`; of ```` ``` ````#10```` ``` ```` | True, `a`#10`  b`#10; False, ''; False; True, '' |

`EvalCli`, stage 2 (task T20b2); 101–107.

| Row | Input | Expected |
|---|---|---|
| 101 | `DefaultEvalConfig` | timeout 10, speed ratio NaN, the four paths '' |
| 102 | `t.jsonl b.jsonl out`; `--timeout 3600 t b a --speed-ratio 0.75 o/` | '', tasks/before as given, after '', OutDir `out/`; '', after `a`, OutDir `o/`, timeout 3600, ratio 0.75 |
| 103 | `--timeout` last; `--timeout 0`; `--speed-ratio 0`, `nan`, `x`; `--Timeout 5`; two and five positionals | `--timeout: value missing`; `--timeout: an integer in 1..3600, got "0"`; `--speed-ratio: a number > 0, got "0"`, `.. "nan"`, `.. "x"`; `unknown option --Timeout`; `expected TASKS BEFORE [AFTER] OUTDIR, got 2 arguments`, `.. got 5 arguments` |
| 104 | a task file (a Pascal task, a blank line, a tool-call and a contains task); then a good line followed by each bad line: `[1]`, `{"id":`, no id, no dialect, no prompt, no expected_stdout, tool-call without tool, contains without text, check `Contains`; a missing file | 3 tasks with their fields; `task line 2: not a JSON object` (twice), `.. id`, `.. dialect`, `.. prompt`, `.. expected_stdout`, `.. tool`, `.. text`, `.. check`; `no file <D>none.jsonl` |
| 105 | `Judge` of PSum for `10`#10 (seed 2, thinking on); for `11`#10 (off); with `S + Q`; with the expect-error marker; without a block; `namespace N;` as oxygene; tool-call task with `get_weather`, `Get_Weather`, none, `tool_calls` `[1]`; contains `Paris` against `It is Paris.` and `It is paris.` | `accepted`, compiled and passed, task/dialect/seed/think as given; `wrong-output`, compiled, not passed; `compile-error`, neither; `accepted`, compiled, not passed; `no-block`, neither; `static-ok`, neither; `check-ok`, passed, not compiled, check `tool-call`; not passed `check-failed` (three times); passed; not passed |
| 106 | tasks a (`10`), b (`11`), c (delphi, `10`), r (tool-call); before: a none, b PSum, c none, r tool call, a off PSum; after: a PSum, a blank line, b PSum, c PSum with `{$APPTYPE CONSOLE}`, r none, a off PSum; `--speed-ratio 0.9`; then before only into `o2` | 0; out `10 replies judged into <D>out/`; results.tsv exactly the header and ten lines; the report has `Tasks: 3 Pascal, 1 regression. Replies: before 5, after 5.`, `\| objfpc \| 1/2 \| 0/2 \| 2/2 \| 1/2 \| 1 \| 0 \|`, `\| delphi (approximation) \| 0/1 \| 0/1 \| 1/1 \| 1/1 \| 1 \| 0 \|`, `\| verified \| 1/2 \| 0/2 \| 2/2 \| 1/2 \| 1 \| 0 \|`, `\| tool-call \| - \| 1/1 \| - \| 0/1 \| 0 \| 1 \|`, `\| contains \| - \| - \| - \| - \| 0 \| 0 \|`, the flips line with `p = 1.0000): no`, `tool-call check loses nothing (1 -> 0): no`, `H49 .. (0.900): yes`, `## Thinking off`; o2 has `\| objfpc \| 1/2 \| 0/2 \|` and no `adopt:` |
| 107 | a reply for task z; a broken second line; a missing replies file; `--frob` | exit 1 `<D>unknown.jsonl line 1: no task z`; `<D>broken.jsonl line 2: not a JSON object`; `no file <D>none.jsonl`, out ''; exit 2 `unknown option --frob`#10 + K_EVAL_USAGE |

Build and run, for `<P>` = `EvalReport` or `EvalCli`:
`fpc -Mobjfpc -Sh -vew -l- -Fu{proj}/src -Fu{proj}/../../src -Fu{proj}/tests/ut -FU{proj}/work/bin -Fl{proj}/../../build/libtch-rocm -k-rpath={proj}/../../build/libtch-rocm {proj}/tests/ut/<P>_test.pas -o{proj}/work/bin/<P>_test`
then `{proj}/work/bin/<P>_test --all --format=plain`.
Until a stage is written, a test program without test methods reports
`No valid tests found`; that is expected.

## ITa (prepared by the harness owner)

`{proj}/tests/ita/T20-run.sh` builds `src/qlora_eval.pas` and
`src/qlora_verify.pas` against `src/` and runs `qlora_eval --speed-ratio
0.9` twice, each from a fresh directory outside the project, on the fixed
records of `tests/ita/T20`: `tasks.jsonl` (eleven Pascal tasks in four
dialects, their reference solutions accepted by `qlora_verify`, and three
regression tasks; made by `make-tasks.bb`) and the replies of the
rehearsal model through Ollama before (`qlora-m2-base`) and after
(`qlora-m2`) the M2 merge (`replies-before.jsonl`, `replies-after.jsonl`,
recorded once with `ask.bb`); then `tests/ita/T20/check.bb`, which
verifies every reply again with `qlora_verify` and recomputes the report.
Writes `{proj}/work/ita/T20/summary.tsv` (case, OK/NG, value, limit,
note; then the exit codes and the count of stderr lines). OK when value ≤
limit. No model file is needed.

| Case | Compared | Limit |
|---|---|---|
| ITa-T20-001 | one result line per reply, in order (model, task, seed, think, dialect) (count of mismatches) | 0 |
| ITa-T20-002 | every reply's outcome, compiled and passed against `qlora_verify` on the first block and the recomputed checks (count of mismatches) | 0 |
| ITa-T20-003 | the report's table rows recomputed from the judgements (count missing) | 0 |
| ITa-T20-004 | the adoption lines recomputed, the sign test included (count missing) | 0 |
| ITa-T20-005 | a second run from another directory: results.tsv and report.md byte-identical (count of files that differ) | 0 |

Measured with the prototype: all 0. With the skeleton every case is NG.

## Stages and case IDs

- Stage 1 (task T20b1): `EvalReport` (all of it); rows UT-EvalReport-001
  to 009.
- Stage 2 (task T20b2): `EvalCli` (all of it); rows UT-EvalCli-101 to 107.
- ITa: ITa-T20-001 to 005.
