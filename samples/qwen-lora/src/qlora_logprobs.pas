(* qlora_logprobs (T11) — thin wrapper around LogprobsCli (written by the
  harness owner; complete). Command line (tasks/T11-facts.md):
    qlora_logprobs [--top N] [--split S] MODEL FILE
  FILE: JSONL, every non-empty line {"id": ..., "tokens": [...]} or
  {"id": ..., "text": "..."}; one JSON line per record with the N most
  likely next tokens and their log-probabilities (H43).
  Exit codes: 0 done, 1 error (message on stderr), 2 usage.
  Keep this body to forwarding only: kcov does not trace child processes,
  so the behaviour stays in the unit where the UT measures it. *)
program qlora_logprobs;

{$mode objfpc}{$H+}

uses
  SysUtils, Classes, LogprobsCli;

var
  Args: array of string;
  O, E: TStringList;
  I, Code: Integer;
begin
  SetLength(Args, ParamCount);
  for I := 1 to ParamCount do
    Args[I - 1] := ParamStr(I);
  O := TStringList.Create;
  E := TStringList.Create;
  try
    Code := RunLogprobsCLI(Args, O, E);
    for I := 0 to O.Count - 1 do
      WriteLn(O[I]);
    for I := 0 to E.Count - 1 do
      WriteLn(StdErr, E[I]);
  finally
    O.Free;
    E.Free;
  end;
  Halt(Code);
end.
