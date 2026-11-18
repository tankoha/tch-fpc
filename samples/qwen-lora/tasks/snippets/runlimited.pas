program runlimited;
{ Checked sample (FPC 3.2.2): run a program with a time limit, stdout and
  stderr captured together, exit code and timeout reported. Copy the
  function RunLimited into the unit; the program part only shows its use. }
{$mode objfpc}{$H+}

uses
  SysUtils, Classes, Process;

{ Runs AExe with AArgs in ADir. Returns False when the program could not be
  started. On timeout the process is killed and ATimedOut is True. }
function RunLimited(const AExe: string; const AArgs: array of string;
  const ADir: string; ATimeoutMs: Integer; out AOutput: string;
  out AExitCode: Integer; out ATimedOut: Boolean): Boolean;
var
  P: TProcess;
  Buf: array[0..4095] of Byte;
  N: LongInt;
  OutS: TStringStream;
  T0: QWord;
  A: string;
begin
  AOutput := '';
  AExitCode := -1;
  ATimedOut := False;
  OutS := TStringStream.Create('');
  P := TProcess.Create(nil);
  try
    P.Executable := AExe;
    for A in AArgs do
      P.Parameters.Add(A);
    if ADir <> '' then
      P.CurrentDirectory := ADir;
    P.Options := [poUsePipes, poStderrToOutPut];
    try
      P.Execute;
    except
      on E: Exception do
      begin
        AOutput := E.Message;
        Exit(False);
      end;
    end;
    T0 := GetTickCount64;
    repeat
      while P.Output.NumBytesAvailable > 0 do
      begin
        N := P.Output.Read(Buf, SizeOf(Buf));
        if N > 0 then
          OutS.Write(Buf, N);
      end;
      if not P.Running then
        Break;
      if GetTickCount64 - T0 > QWord(ATimeoutMs) then
      begin
        ATimedOut := True;
        P.Terminate(9);
        Break;
      end;
      Sleep(10);
    until False;
    { read what is left after the process ended }
    repeat
      N := P.Output.Read(Buf, SizeOf(Buf));
      if N > 0 then
        OutS.Write(Buf, N);
    until N <= 0;
    if not ATimedOut then
      AExitCode := P.ExitCode;  { ExitStatus is the raw wait status (3 shl 8) }
    AOutput := OutS.DataString;
    Result := True;
  finally
    P.Free;
    OutS.Free;
  end;
end;

procedure Show(const AExe: string; const AArgs: array of string; AMs: Integer);
var
  O: string;
  C: Integer;
  T, S: Boolean;
begin
  S := RunLimited(AExe, AArgs, '', AMs, O, C, T);
  WriteLn('started=', S, ' exit=', C, ' timeout=', T, ' out=', QuotedStr(Trim(O)));
end;

begin
  Show('/bin/sh', ['-c', 'echo out; echo err 1>&2; exit 3'], 2000);
  Show('/bin/sleep', ['5'], 300);
  Show('/no/such/program', [], 1000);
end.
