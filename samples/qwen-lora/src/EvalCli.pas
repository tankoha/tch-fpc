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
  raise Exception.Create('DefaultEvalConfig: not implemented yet');
end;

function ParseEvalArgs(const AArgs: array of string; out C: TEvalConfig): string;
begin
  Result := '';
  raise Exception.Create('ParseEvalArgs: not implemented yet');
end;

function LoadTasks(const AFile: string): TEvalTasks;
begin
  Result := nil;
  raise Exception.Create('LoadTasks: not implemented yet');
end;

function JudgeReply(const ATask: TEvalTask; AReply: TJSONObject; const ADir: string;
  ATimeout: Integer; const AEnv: TEnvRec; out AOutcome: string): TJudged;
begin
  Result := Default(TJudged);
  raise Exception.Create('JudgeReply: not implemented yet');
end;

function RunEvalCLI(const AArgs: array of string; const AEnv: TEnvRec;
  const AWorkRoot: string; AOut, AErr: TStrings): Integer;
begin
  Result := 0;
  raise Exception.Create('RunEvalCLI: not implemented yet');
end;

end.
