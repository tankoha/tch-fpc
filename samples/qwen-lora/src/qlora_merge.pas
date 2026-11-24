(* qlora_merge (T18) — thin wrapper around GgufMerge (written by the
  harness owner; complete). Command line (tasks/T18-facts.md):
    qlora_merge [--base-sha256 H] BASE CKPT OUT
  Exit codes: 0 done, 1 error (message on stderr), 2 usage. *)
program qlora_merge;

{$mode objfpc}{$H+}

uses
  SysUtils, Classes, GgufMerge;

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
    Code := RunMergeCLI(Args, O, E);
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
