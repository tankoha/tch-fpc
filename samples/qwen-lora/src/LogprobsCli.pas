(* LogprobsCli (T11): the behaviour of qlora_logprobs. Command line:
    qlora_logprobs [--top N] [--split S] MODEL FILE
  FILE: JSONL; every non-empty line is {"id": ..., "tokens": [ids]} or
  {"id": ..., "text": "..."} (text encoded by QwenTokenizer). One output
  line per record: {"id":"..","n":T,"top":[{"id":I,"logprob":L},...]}, the
  N most likely next tokens after the whole prompt (natural log of the
  softmax over the vocabulary; default N 20, 1..100). Without --split the
  model runs on the CPU; with --split S blocks 0..S-1 run on cuda:0 and the
  rest and the head on cuda:1.
  Exit codes: 0 done, 1 error (message on stderr), 2 usage.
  tasks/T11-facts.md.

  Skeleton by the harness owner (interface final); the routines are
  written in T11b3. *)
unit LogprobsCli;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils;

{ AOut / AErr are filled (caller-created, caller-owned); the result is the
  exit code. }
function RunLogprobsCLI(const AArgs: array of string; AOut, AErr: TStrings): Integer;

{ One output line (the format above) from ids and log-probabilities. }
function TopLine(const AId: string; ACount: Integer; const AIds: array of Int64;
  const ALogProbs: array of Double): string;

implementation

uses
  fpjson, jsonparser, Tch, GgufFile, QwenTokenizer, QwenModel;

function TopLine(const AId: string; ACount: Integer; const AIds: array of Int64;
  const ALogProbs: array of Double): string;
begin
  Result := '';
  raise Exception.Create('TopLine: not implemented yet');
end;

function RunLogprobsCLI(const AArgs: array of string; AOut, AErr: TStrings): Integer;
begin
  Result := 1;
  raise Exception.Create('RunLogprobsCLI: not implemented yet');
end;

end.
