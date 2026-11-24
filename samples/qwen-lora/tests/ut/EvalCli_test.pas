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
