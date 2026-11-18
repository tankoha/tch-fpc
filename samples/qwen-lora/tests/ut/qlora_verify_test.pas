{ UT for program qlora_verify (T06c) — one published Test method per case row
  of tests/ut/cases/qlora_verify.cases.md.
  Each case compiles the program with the system fpc into a scratch dir and
  runs it with the project root as the working directory (so its work root
  work/verify lands in the project's git-ignored scratch).
  Invoked with `--all --format=plain` (see .sdlc/ut-coverage.sh). }
program qlora_verify_test;

{$mode objfpc}{$H+}

uses
  SysUtils, Classes, Process, fpcunit, testregistry, consoletestrunner;

const
  K_USAGE_LINE = 'usage: qlora_verify [--dialect D] [--timeout SECONDS] PATH...';

type
  TTestQloraVerify = class(TTestCase)
  public
    procedure WriteFile(const APath, AContent: string);
    function BuildProg(const AScratch: string): string;
    function RunProg(const AExe: string; const AArgs: TStringArray;
      out AOut: string): Integer;
    function Which(const AName: string): string;
  published
    procedure TestUT_qlora_verify_001;
    procedure TestUT_qlora_verify_002;
    procedure TestUT_qlora_verify_003;
  end;

procedure TTestQloraVerify.WriteFile(const APath, AContent: string);
var
  D: string;
  F: TFileStream;
begin
  D := ExtractFileDir(APath);
  if D <> '' then
    ForceDirectories(D)
  else
    D := '.';
  F := TFileStream.Create(APath, fmCreate);
  try
    if Length(AContent) > 0 then
      F.Write(AContent[1], Length(AContent));
  finally
    F.Free;
  end;
end;

{ Compile src/qlora_verify.pas with the system fpc into AScratch; returns the
  exe path on success (empty otherwise). DialectVerify has no tch dependency,
  so no library rpath is needed. }
function TTestQloraVerify.BuildProg(const AScratch: string): string;
var
  P: TProcess;
  S: TStringStream;
  Buf: array[1..4096] of Char;
  N: Integer;
  T0: QWord;
begin
  Result := '';
  ForceDirectories(AScratch);
  P := TProcess.Create(nil);
  S := TStringStream.Create('');
  try
    P.Executable := 'fpc';
    P.Parameters.Add('-Mobjfpc');
    P.Parameters.Add('-Sh');
    P.Parameters.Add('-vew');
    P.Parameters.Add('-l-');
    P.Parameters.Add('-Fu' + GetCurrentDir + '/src');
    P.Parameters.Add('-FU' + AScratch);
    P.Parameters.Add('-FE' + AScratch);
    P.Parameters.Add(GetCurrentDir + '/src/qlora_verify.pas');
    P.Options := [poUsePipes, poStderrToOutPut];
    try
      P.Execute;
    except
      Exit;
    end;
    T0 := GetTickCount64;
    while P.Running and (GetTickCount64 - T0 < QWord(120000)) do
      Sleep(50);
    if P.Running then
      P.Terminate(9);
    repeat
      N := P.Output.Read(Buf, SizeOf(Buf));
      if N > 0 then
        S.Write(Buf, N);
    until N <= 0;
    if (P.ExitCode = 0) and FileExists(AScratch + '/qlora_verify') then
      Result := AScratch + '/qlora_verify';
  finally
    P.Free;
    S.Free;
  end;
end;

{ Run AExe (inherited working directory) with AArgs; captured output in AOut,
  exit code in Result. Same poll/drain shape as DialectVerify.RunLimited. }
function TTestQloraVerify.RunProg(const AExe: string; const AArgs: TStringArray;
  out AOut: string): Integer;
var
  P: TProcess;
  S: TStringStream;
  Buf: array[1..4096] of Char;
  I, N: Integer;
  T0: QWord;
begin
  AOut := '';
  P := TProcess.Create(nil);
  S := TStringStream.Create('');
  try
    P.Executable := AExe;
    for I := 0 to High(AArgs) do
      P.Parameters.Add(AArgs[I]);
    P.Options := [poUsePipes, poStderrToOutPut];
    try
      P.Execute;
      T0 := GetTickCount64;
      while P.Running and (GetTickCount64 - T0 < QWord(60000)) do
        Sleep(50);
      if P.Running then
        P.Terminate(9);
      repeat
        N := P.Output.Read(Buf, SizeOf(Buf));
        if N > 0 then
          S.Write(Buf, N);
      until N <= 0;
      AOut := S.DataString;
      Result := P.ExitCode;
    except
      Result := -1;
    end;
  finally
    P.Free;
    S.Free;
  end;
end;

function TTestQloraVerify.Which(const AName: string): string;
var
  Out: string;
  Args: TStringArray;
begin
  SetLength(Args, 1);
  Args[0] := AName;
  if RunProg('which', Args, Out) = 0 then
    Result := Trim(Out)
  else
    Result := '';
end;

{ UT-qlora_verify-001: --help → usage line once, exit 0. }
procedure TTestQloraVerify.TestUT_qlora_verify_001;
var
  Exe, Out: string;
  Args: TStringArray;
begin
  Exe := BuildProg('work/ut/qv-001');
  AssertTrue(Exe <> '');
  SetLength(Args, 1);
  Args[0] := '--help';
  AssertEquals(0, RunProg(Exe, Args, Out));
  AssertEquals(K_USAGE_LINE + #10, Out);
end;

{ UT-qlora_verify-002: no arguments → usage error, exit 2 (exercises the
  program's `if not VerifyMain` branch and the empty-args print loop). }
procedure TTestQloraVerify.TestUT_qlora_verify_002;
var
  Exe, Out: string;
  Args: TStringArray;
begin
  Exe := BuildProg('work/ut/qv-002');
  AssertTrue(Exe <> '');
  SetLength(Args, 0);
  AssertEquals(2, RunProg(Exe, Args, Out));
  AssertEquals(K_USAGE_LINE + #10, Out);
end;

{ UT-qlora_verify-003: real end-to-end run — objfpc example compiled by the
  real fpc into work/verify/<n> and executed there (T06c fix: the program now
  runs INSIDE <out>); accepted line on stdout, exit 0.
  The unit's IsPathUsable (2.12) needs an existing file, so a bare 'fpc'
  default is unresolvable by design; the real caller supplies the absolute
  compiler path, here via the verified sh -c VAR=val env path. }
procedure TTestQloraVerify.TestUT_qlora_verify_003;
var
  Exe, Out, P1, Fpc: string;
  Args: TStringArray;
begin
  Exe := BuildProg('work/ut/qv-003');
  AssertTrue(Exe <> '');
  P1 := 'work/ut/qv-003/s.pas';
  WriteFile(P1, '{$mode objfpc}'#10
    + 'program s;'#10
    + 'begin'#10
    + '  Writeln(''Hello'');' + #10
    + 'end.'#10);
  WriteFile('work/ut/qv-003/s.expected', 'Hello'#10);
  Fpc := Which('fpc');
  AssertTrue(Fpc <> '');
  SetLength(Args, 2);
  Args[0] := '-c';
  Args[1] := 'QLORA_FPC=' + Fpc + ' ' + Exe + ' --dialect objfpc ' + P1;
  AssertEquals(0, RunProg('/bin/sh', Args, Out));
  AssertEquals(P1 + #9 + 'objfpc' + #9 + 'compiled-run' + #9 + 'accepted' + #9 +
    #10, Out);
end;

var
  App: TTestRunner;
begin
  RegisterTest(TTestQloraVerify);
  App := TTestRunner.Create(nil);
  App.Initialize;
  App.Title := 'qlora_verify UT (T06c)';
  App.Run;
  App.Free;
end.
