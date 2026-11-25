{ EvalCli — the evaluation program qlora_eval (T20), per DESIGN.md 8.17
  and tasks/T20-facts.md (authoritative): judges recorded replies (the
  first fenced block of a Pascal task's reply through DialectVerify; a
  regression task's check), writes one result line per reply and the
  report of 8.17 with the adoption rule.
  Interface by the harness owner (tasks/T20-facts.md); the bodies are
  written in T20b2. }
unit EvalCli;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Math, fpjson, jsonparser, DialectVerify, EvalReport;

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

end.
