(* qlora_dataset (T19) — thin wrapper around DatasetCli (written by the
  harness owner; complete). Command line (tasks/T19-facts.md):
    qlora_dataset [options] RAW OUTDIR
  Reads QLORA_FPC / QLORA_FPC_UNLEASHED / HOME (the verifier's compilers,
  as qlora_verify) at start-up; the verifier works in a fresh directory
  under the system's temporary directory.
  Exit codes: 0 done, 1 error (message on stderr), 2 usage. *)
program qlora_dataset;

{$mode objfpc}{$H+}

uses
  SysUtils, Classes, DialectVerify, DatasetCli;

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
    Code := RunDatasetCLI(Args,
      ReadEnvRec(GetEnvironmentVariable('QLORA_FPC'),
        GetEnvironmentVariable('QLORA_FPC_UNLEASHED'), GetEnvironmentVariable('HOME')),
      GetTempDir + 'qlora-dataset-' + IntToStr(GetProcessID), O, E);
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
