(* qlora_train (T17) — thin wrapper around TrainCli (written by the
  harness owner; complete). Command line (tasks/T17-facts.md):
    qlora_train [options] MODEL TRAIN OUTDIR
  TRAIN: TrainData records; OUTDIR: train-log.tsv and ckpt/ (TrainCli).
  Exit codes: 0 done, 1 error (message on stderr), 2 usage, 3 non-finite loss.
  Keep this body to forwarding only: kcov does not trace child processes,
  so the behaviour stays in the unit where the UT measures it. *)
program qlora_train;

{$mode objfpc}{$H+}

uses
  SysUtils, Classes, TrainCli;

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
    Code := RunTrainCLI(Args, O, E);
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
