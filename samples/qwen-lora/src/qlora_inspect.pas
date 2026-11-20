{ qlora_inspect (T01c) — thin CLI wrapper around the InspectCli unit.

  Command line (facts file / T01-detailed 2.12):
    qlora_inspect [--kv] [--tensors] FILE

  The logic lives in the unit: RunInspectCLI (T01-detailed 2.12) parses the
  argument vector, opens the GGUF file with GgufFile.Open, fills AOut with
  the summary / type / kv / tensor lines and AErr with usage and error
  lines, and returns the exit code (2 = usage error, 1 = file error or
  data_end <> file_size, 0 = inspected, data_end = file_size).

  Keep this body to forwarding only (R20): kcov does not trace child
  processes, so every behavior line stays in the unit where the UT measures
  it in process (UT-InspectCli-001..011); this program only copies
  ParamStr(1..ParamCount) into an array, calls RunInspectCLI, writes the
  AOut lines to stdout and the AErr lines to stderr, and Halts with the
  returned code. The wrapper contract itself is pinned by
  tests/ut/qlora_inspect_test.pas (UT-qlora_inspect-001..003). }
program qlora_inspect;

{$mode objfpc}{$H+}

uses
  SysUtils, Classes, InspectCli;

var
  I: LongInt;
  Args: array of string;
  AOut, AErr: TStringList;
  Code: Integer;
begin
  SetLength(Args, ParamCount);
  for I := 0 to ParamCount - 1 do
    Args[I] := ParamStr(I + 1);
  AOut := TStringList.Create;
  AErr := TStringList.Create;
  try
    Code := InspectCli.RunInspectCLI(Args, AOut, AErr);
    for I := 0 to AOut.Count - 1 do
      Writeln(AOut[I]);
    for I := 0 to AErr.Count - 1 do
      Writeln(StdErr, AErr[I]);
  finally
    AOut.Free;
    AErr.Free;
  end;
  Halt(Code);
end.
