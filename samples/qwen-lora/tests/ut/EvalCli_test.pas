{ UT program for src/EvalCli.pas: the rows UT-EvalCli-NNN of
  tests/ut/cases/EvalCli.cases.md. Compiles and runs small objfpc
  programs with the stock fpc (as DialectVerify's UT).
  Invoked with `--all --format=plain` (see .sdlc/ut-coverage.sh).

  Skeleton by the harness owner; the test methods are written in T20b2. }
program EvalCli_test;

{$mode objfpc}{$H+}

uses
  SysUtils, Classes, Math, fpjson, jsonparser, DialectVerify, EvalReport, EvalCli,
  fpcunit, testregistry, consoletestrunner;

type
  TEvalCliTest = class(TTestCase)
  published
    procedure TestUT_EvalCli_101;
    procedure TestUT_EvalCli_102;
    procedure TestUT_EvalCli_103;
    procedure TestUT_EvalCli_104;
    procedure TestUT_EvalCli_105;
    procedure TestUT_EvalCli_106;
    procedure TestUT_EvalCli_107;
  end;

{ ---- helpers (harness owner's; nobody changes them) ---- }

var
  TmpDir: string;
  Env: TEnvRec;
  Fresh: Integer = 0;

const
  { a small objfpc program printing 10 }
  PSum = 'program Sum;' + #10 + '{$mode objfpc}{$H+}' + #10 + 'var' + #10 + '  I, S: Integer;' + #10 +
    'begin' + #10 + '  S := 0;' + #10 + '  for I := 1 to 4 do' + #10 + '    S := S + I;' + #10 +
    '  WriteLn(S);' + #10 + 'end.' + #10;

{ A fresh directory under TmpDir (created), ending in the delimiter. }
function NewDir: string;
begin
  Inc(Fresh);
  Result := TmpDir + 'd' + IntToStr(Fresh) + PathDelim;
  ForceDirectories(Result);
end;

procedure SaveLines(const APath: string; const ALines: array of string);
var
  L: TStringList;
  I: Integer;
begin
  L := TStringList.Create;
  try
    for I := 0 to High(ALines) do
      L.Add(ALines[I]);
    L.SaveToFile(APath);
  finally
    L.Free;
  end;
end;

function FileText(const APath: string): string;
var
  L: TStringList;
begin
  L := TStringList.Create;
  try
    L.LoadFromFile(APath);
    Result := L.Text;
  finally
    L.Free;
  end;
end;

{ A Pascal task line. }
function TaskLine(const AId, ADialect, AExpected: string): string;
begin
  Result := Format('{"id":"%s","dialect":"%s","prompt":"Write it.","expected_stdout":"%s"}',
    [AId, ADialect, StringToJSONString(AExpected)]);
end;

{ A reply line: task, seed, think, content (a reply around the fenced
  source ASrc; '' gives a reply without a block), optional tool call. }
function ReplyLine(const ATask: string; ASeed: Integer; AThink: Boolean; const ASrc: string;
  const ATool: string = ''): string;
var
  C: string;
begin
  if ASrc = '' then
    C := 'I cannot.'
  else
    C := 'Here:' + #10 + '```pascal' + #10 + ASrc + '```' + #10 + 'Done.';
  Result := Format('{"task":"%s","seed":%d,"think":%s,"content":"%s"',
    [ATask, ASeed, BoolToStr(AThink, 'true', 'false'), StringToJSONString(C)]);
  if ATool <> '' then
    Result := Result + Format(',"tool_calls":[{"name":"%s","arguments":{}}]', [ATool]);
  Result := Result + '}';
end;

function Task(const AId, ADialect, AExpected: string; const ACheck: string = '';
  const ATool: string = ''; const AText: string = ''): TEvalTask;
begin
  Result := Default(TEvalTask);
  Result.Id := AId;
  Result.Dialect := ADialect;
  Result.Prompt := 'p';
  Result.Expected := AExpected;
  Result.Check := ACheck;
  Result.Tool := ATool;
  Result.Text := AText;
end;

{ JudgeReply of the reply line ALine for ATask in a fresh directory. }
function Judge(const ATask: TEvalTask; const ALine: string; out AOutcome: string): TJudged;
var
  O: TJSONObject;
begin
  O := GetJSON(ALine, False) as TJSONObject;
  try
    Result := JudgeReply(ATask, O, NewDir, 10, Env, AOutcome);
  finally
    O.Free;
  end;
end;

{ RunEvalCLI with Env and the work root <TmpDir>work; AOut / AErr the
  lines joined by #10; the exit code. }
function RunCli(const AArgs: array of string; out AOut, AErr: string): Integer;
var
  O, E: TStringList;
begin
  O := TStringList.Create;
  E := TStringList.Create;
  try
    Result := RunEvalCLI(AArgs, Env, TmpDir + 'work', O, E);
    AOut := string.Join(#10, O.ToStringArray);
    AErr := string.Join(#10, E.ToStringArray);
  finally
    O.Free;
    E.Free;
  end;
end;

{ ---- test methods ---- }

{ UT-EvalCli-101: DefaultEvalConfig }
procedure TEvalCliTest.TestUT_EvalCli_101;
var
  C: TEvalConfig;
begin
  C := DefaultEvalConfig;
  AssertEquals(10, C.Timeout);
  AssertTrue('speed ratio is NaN', IsNan(C.SpeedRatio));
  AssertEquals('', C.Tasks);
  AssertEquals('', C.Before);
  AssertEquals('', C.After);
  AssertEquals('', C.OutDir);
end;

{ UT-EvalCli-102: ParseEvalArgs, valid }
procedure TEvalCliTest.TestUT_EvalCli_102;
var
  C: TEvalConfig;
begin
  AssertEquals('', ParseEvalArgs(['t.jsonl', 'b.jsonl', 'out'], C));
  AssertEquals('t.jsonl', C.Tasks);
  AssertEquals('b.jsonl', C.Before);
  AssertEquals('', C.After);
  AssertEquals('out' + PathDelim, C.OutDir);
  AssertEquals('',
    ParseEvalArgs(['--timeout', '3600', 't', 'b', 'a', '--speed-ratio', '0.75', 'o/'], C));
  AssertEquals('a', C.After);
  AssertEquals('o/', C.OutDir);
  AssertEquals(3600, C.Timeout);
  AssertEquals(0.75, C.SpeedRatio);
end;

{ UT-EvalCli-103: ParseEvalArgs, errors }
procedure TEvalCliTest.TestUT_EvalCli_103;
var
  C: TEvalConfig;
begin
  AssertEquals('--timeout: value missing', ParseEvalArgs(['--timeout'], C));
  AssertEquals('--timeout: an integer in 1..3600, got "0"',
    ParseEvalArgs(['--timeout', '0'], C));
  AssertEquals('--speed-ratio: a number > 0, got "0"',
    ParseEvalArgs(['--speed-ratio', '0'], C));
  AssertEquals('--speed-ratio: a number > 0, got "nan"',
    ParseEvalArgs(['--speed-ratio', 'nan'], C));
  AssertEquals('--speed-ratio: a number > 0, got "x"',
    ParseEvalArgs(['--speed-ratio', 'x'], C));
  AssertEquals('unknown option --Timeout', ParseEvalArgs(['--Timeout', '5'], C));
  AssertEquals('expected TASKS BEFORE [AFTER] OUTDIR, got 2 arguments',
    ParseEvalArgs(['t', 'b'], C));
  AssertEquals('expected TASKS BEFORE [AFTER] OUTDIR, got 5 arguments',
    ParseEvalArgs(['t', 'b', 'a', 'o', 'x'], C));
end;

{ UT-EvalCli-104: LoadTasks }
procedure TEvalCliTest.TestUT_EvalCli_104;
var
  Tasks: TEvalTasks;
  D, Bad: string;

  procedure ExpectBadLine(const ALine, AMsg: string);
  begin
    SaveLines(Bad, [TaskLine('a', 'objfpc', '10'), ALine]);
    try
      LoadTasks(Bad);
      Fail('no Exception');
    except
      on E: Exception do
        AssertEquals(AMsg, E.Message);
    end;
  end;

begin
  D := NewDir;
  Bad := D + 'bad.jsonl';
  SaveLines(D + 'tasks.jsonl', [TaskLine('a', 'objfpc', '10'), '',
    '{"id":"t","dialect":"regression","prompt":"p","check":"tool-call","tool":"get_weather"}',
    '{"id":"c","dialect":"regression","prompt":"p","check":"contains","text":"Paris"}']);
  Tasks := LoadTasks(D + 'tasks.jsonl');
  AssertEquals(3, Length(Tasks));
  AssertEquals('a', Tasks[0].Id);
  AssertEquals('objfpc', Tasks[0].Dialect);
  AssertEquals('Write it.', Tasks[0].Prompt);
  AssertEquals('10', Tasks[0].Expected);
  AssertEquals('t', Tasks[1].Id);
  AssertEquals('regression', Tasks[1].Dialect);
  AssertEquals('tool-call', Tasks[1].Check);
  AssertEquals('get_weather', Tasks[1].Tool);
  AssertEquals('c', Tasks[2].Id);
  AssertEquals('contains', Tasks[2].Check);
  AssertEquals('Paris', Tasks[2].Text);
  ExpectBadLine('[1]', 'task line 2: not a JSON object');
  ExpectBadLine('{"id":', 'task line 2: not a JSON object');
  ExpectBadLine('{"dialect":"objfpc","prompt":"p","expected_stdout":"1"}', 'task line 2: id');
  ExpectBadLine('{"id":"a","prompt":"p","expected_stdout":"1"}', 'task line 2: dialect');
  ExpectBadLine('{"id":"a","dialect":"objfpc","expected_stdout":"1"}', 'task line 2: prompt');
  ExpectBadLine('{"id":"a","dialect":"objfpc","prompt":"p"}', 'task line 2: expected_stdout');
  ExpectBadLine('{"id":"t","dialect":"regression","prompt":"p","check":"tool-call"}', 'task line 2: tool');
  ExpectBadLine('{"id":"c","dialect":"regression","prompt":"p","check":"contains"}', 'task line 2: text');
  ExpectBadLine('{"id":"c","dialect":"regression","prompt":"p","check":"Contains"}', 'task line 2: check');
  try
    LoadTasks(D + 'none.jsonl');
    Fail('no Exception');
  except
    on E: Exception do
      AssertEquals('no file ' + D + 'none.jsonl', E.Message);
  end;
end;

{ UT-EvalCli-105: JudgeReply, Pascal and regression }
procedure TEvalCliTest.TestUT_EvalCli_105;
var
  J: TJudged;
  O: string;
begin
  J := Judge(Task('a', 'objfpc', '10'#10), ReplyLine('a', 2, True, PSum), O);
  AssertEquals('accepted', O);
  AssertTrue('compiled', J.Compiled);
  AssertTrue('passed', J.Passed);
  AssertEquals('a', J.Task);
  AssertEquals('objfpc', J.Dialect);
  AssertEquals(2, J.Seed);
  AssertTrue('thinking on', J.Think);
  J := Judge(Task('b', 'objfpc', '11'#10), ReplyLine('b', 10, False, PSum), O);
  AssertEquals('wrong-output', O);
  AssertTrue('compiled', J.Compiled);
  AssertTrue('not passed', not J.Passed);
  J := Judge(Task('a', 'objfpc', '10'#10),
    ReplyLine('a', 1, True, 'program S;' + #10 + '{$mode objfpc}{$H+}' + #10 +
      'var S: Integer;' + #10 + 'begin' + #10 + '  S := S + Q;' + #10 + 'end.' + #10), O);
  AssertEquals('compile-error', O);
  AssertTrue('not compiled', not J.Compiled);
  AssertTrue('not passed', not J.Passed);
  J := Judge(Task('a', 'objfpc', '10'#10),
    ReplyLine('a', 1, True, 'program Bad;' + #10 + '{$mode objfpc}{$H+}' + #10 + 'begin' + #10 +
      '  Q := 1; { expect-error: Identifier not found "Q" }' + #10 + 'end.' + #10), O);
  AssertEquals('accepted', O);
  AssertTrue('compiled', J.Compiled);
  AssertTrue('not passed', not J.Passed);
  J := Judge(Task('a', 'objfpc', '10'#10), ReplyLine('a', 1, True, ''), O);
  AssertEquals('no-block', O);
  AssertTrue('not compiled', not J.Compiled);
  AssertTrue('not passed', not J.Passed);
  J := Judge(Task('o', 'oxygene', '10'),
    ReplyLine('o', 1, True, 'namespace N;' + #10), O);
  AssertEquals('static-ok', O);
  AssertTrue('not compiled', not J.Compiled);
  AssertTrue('not passed', not J.Passed);
  J := Judge(Task('r', 'regression', '', 'tool-call', 'get_weather'),
    ReplyLine('r', 1, True, '', 'get_weather'), O);
  AssertEquals('check-ok', O);
  AssertTrue('passed', J.Passed);
  AssertTrue('not compiled', not J.Compiled);
  AssertEquals('tool-call', J.Check);
  J := Judge(Task('r', 'regression', '', 'tool-call', 'get_weather'),
    ReplyLine('r', 1, True, '', 'Get_Weather'), O);
  AssertEquals('check-failed', O);
  AssertTrue('not passed', not J.Passed);
  J := Judge(Task('r', 'regression', '', 'tool-call', 'get_weather'),
    ReplyLine('r', 1, True, ''), O);
  AssertEquals('check-failed', O);
  AssertTrue('not passed', not J.Passed);
  J := Judge(Task('r', 'regression', '', 'tool-call', 'get_weather'),
    '{"task":"r","seed":1,"think":true,"content":"I cannot.","tool_calls":[1]}', O);
  AssertEquals('check-failed', O);
  AssertTrue('not passed', not J.Passed);
  J := Judge(Task('c', 'regression', '', 'contains', '', 'Paris'),
    ReplyLine('c', 1, True, 'It is Paris.'), O);
  AssertEquals('check-ok', O);
  AssertTrue('passed', J.Passed);
  J := Judge(Task('c', 'regression', '', 'contains', '', 'Paris'),
    ReplyLine('c', 1, True, 'It is paris.'), O);
  AssertEquals('check-failed', O);
  AssertTrue('not passed', not J.Passed);
end;

{ UT-EvalCli-106: RunEvalCLI, valid run }
procedure TEvalCliTest.TestUT_EvalCli_106;
var
  D, Tasks, Before, After, O, E, Report: string;
  PCon: string;
begin
  D := NewDir;
  Tasks := D + 'tasks.jsonl';
  Before := D + 'before.jsonl';
  After := D + 'after.jsonl';
  SaveLines(Tasks, [
    TaskLine('a', 'objfpc', '10'#10),
    TaskLine('b', 'objfpc', '11'#10),
    TaskLine('c', 'delphi', '10'#10),
    '{"id":"r","dialect":"regression","prompt":"p","check":"tool-call","tool":"get_weather"}']);
  SaveLines(Before, [
    ReplyLine('a', 2, True, ''),
    ReplyLine('b', 1, True, PSum),
    ReplyLine('c', 1, True, ''),
    ReplyLine('r', 1, True, '', 'get_weather'),
    ReplyLine('a', 2, False, PSum)]);
  PCon := 'program Sum;' + #10 + '{$mode objfpc}{$H+}' + #10 + '{$APPTYPE CONSOLE}' + #10 +
    'var' + #10 + '  I, S: Integer;' + #10 + 'begin' + #10 + '  S := 0;' + #10 +
    '  for I := 1 to 4 do' + #10 + '    S := S + I;' + #10 + '  WriteLn(S);' + #10 + 'end.' + #10;
  SaveLines(After, [
    ReplyLine('a', 2, True, PSum),
    '',
    ReplyLine('b', 1, True, PSum),
    ReplyLine('c', 1, True, PCon),
    ReplyLine('r', 1, True, ''),
    ReplyLine('a', 2, False, PSum)]);
  AssertEquals(0, RunCli([Tasks, Before, '--speed-ratio', '0.9', After, D + 'out'], O, E));
  AssertEquals('', E);
  AssertEquals('10 replies judged into ' + D + 'out/', O);
  AssertEquals(
    'model'#9'task'#9'seed'#9'think'#9'dialect'#9'outcome'#9'compiled'#9'passed' + #10 +
    'before'#9'a'#9'2'#9'on'#9'objfpc'#9'no-block'#9'0'#9'0' + #10 +
    'before'#9'b'#9'1'#9'on'#9'objfpc'#9'wrong-output'#9'1'#9'0' + #10 +
    'before'#9'c'#9'1'#9'on'#9'delphi'#9'no-block'#9'0'#9'0' + #10 +
    'before'#9'r'#9'1'#9'on'#9'regression'#9'check-ok'#9'0'#9'1' + #10 +
    'before'#9'a'#9'2'#9'off'#9'objfpc'#9'accepted'#9'1'#9'1' + #10 +
    'after'#9'a'#9'2'#9'on'#9'objfpc'#9'accepted'#9'1'#9'1' + #10 +
    'after'#9'b'#9'1'#9'on'#9'objfpc'#9'wrong-output'#9'1'#9'0' + #10 +
    'after'#9'c'#9'1'#9'on'#9'delphi'#9'accepted'#9'1'#9'1' + #10 +
    'after'#9'r'#9'1'#9'on'#9'regression'#9'check-failed'#9'0'#9'0' + #10 +
    'after'#9'a'#9'2'#9'off'#9'objfpc'#9'accepted'#9'1'#9'1' + #10,
    FileText(D + 'out/results.tsv'));
  Report := FileText(D + 'out/report.md');
  AssertTrue('totals', Pos('Tasks: 3 Pascal, 1 regression. Replies: before 5, after 5.', Report) > 0);
  AssertTrue('objfpc row', Pos('| objfpc | 1/2 | 0/2 | 2/2 | 1/2 | 1 | 0 |', Report) > 0);
  AssertTrue('delphi row', Pos('| delphi (approximation) | 0/1 | 0/1 | 1/1 | 1/1 | 1 | 0 |', Report) > 0);
  AssertTrue('verified row', Pos('| verified | 1/2 | 0/2 | 2/2 | 1/2 | 1 | 0 |', Report) > 0);
  AssertTrue('tool-call row', Pos('| tool-call | - | 1/1 | - | 0/1 | 0 | 1 |', Report) > 0);
  AssertTrue('contains row', Pos('| contains | - | - | - | - | 0 | 0 |', Report) > 0);
  AssertTrue('flips line', Pos('- flips significant (1 fail->pass, 0 pass->fail, p = 1.0000): no', Report) > 0);
  AssertTrue('tool-call line', Pos('tool-call check loses nothing (1 -> 0): no', Report) > 0);
  AssertTrue('H49 line', Pos('H49 speed at least 0.7 of before (0.900): yes', Report) > 0);
  AssertTrue('thinking off section', Pos('## Thinking off', Report) > 0);
  AssertEquals(0, RunCli([Tasks, Before, D + 'o2'], O, E));
  AssertEquals('', E);
  AssertEquals('5 replies judged into ' + D + 'o2/', O);
  Report := FileText(D + 'o2/report.md');
  AssertTrue('o2 objfpc row', Pos('| objfpc | 1/2 | 0/2 |', Report) > 0);
  AssertTrue('no adopt line', Pos('adopt:', Report) = 0);
end;

{ UT-EvalCli-107: RunEvalCLI, errors }
procedure TEvalCliTest.TestUT_EvalCli_107;
var
  D, Tasks, O, E: string;
begin
  D := NewDir;
  Tasks := D + 'tasks.jsonl';
  SaveLines(Tasks, [TaskLine('a', 'objfpc', '10'#10)]);
  SaveLines(D + 'unknown.jsonl', ['{"task":"z","seed":1,"think":true,"content":"hi"}']);
  AssertEquals(1, RunCli([Tasks, D + 'unknown.jsonl', D + 'out'], O, E));
  AssertEquals('', O);
  AssertEquals(D + 'unknown.jsonl line 1: no task z', E);
  SaveLines(D + 'broken.jsonl', [ReplyLine('a', 1, True, ''), '{"task":"a"']);
  AssertEquals(1, RunCli([Tasks, D + 'broken.jsonl', D + 'out'], O, E));
  AssertEquals(D + 'broken.jsonl line 2: not a JSON object', E);
  AssertEquals(1, RunCli([Tasks, D + 'none.jsonl', D + 'out'], O, E));
  AssertEquals('', O);
  AssertEquals('no file ' + D + 'none.jsonl', E);
  AssertEquals(2, RunCli(['--frob', Tasks, D + 'before.jsonl', D + 'out'], O, E));
  AssertEquals('', O);
  AssertEquals('unknown option --frob'#10 + K_EVAL_USAGE, E);
end;

var
  App: TTestRunner;
begin
  TmpDir := GetTempDir + 'evalcli-ut-' + IntToStr(GetProcessID) + PathDelim;
  ForceDirectories(TmpDir);
  Env := ReadEnvRec('', '', GetEnvironmentVariable('HOME'));
  RegisterTest(TEvalCliTest);
  App := TTestRunner.Create(nil);
  App.Initialize;
  App.Title := 'EvalCli UT';
  App.Run;
  App.Free;
end.
