{ qlora_tokenize (T05) — thin wrapper around TokenizeCli (written by the
  harness owner; complete). Command line (tasks/T05-facts.md):
    qlora_tokenize [--jsonl] MODEL FILE
  Plain: the whole FILE (bytes, UTF-8) is encoded; one line of token ids.
  --jsonl: every non-empty line of FILE is {"id": ..., "text": ...}; one
  line id<TAB>ids per record (id defaults to the line number).
  Exit codes: 0 done, 1 error (message on stderr), 2 usage.
  Keep this body to forwarding only: kcov does not trace child processes,
  so the behaviour stays in the unit where the UT measures it. }
program qlora_tokenize;

{$mode objfpc}{$H+}

uses
  SysUtils, Classes, TokenizeCli;

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
    Code := RunTokenizeCLI(Args, O, E);
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
