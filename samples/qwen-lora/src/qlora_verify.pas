{ qlora_verify (T06c) — thin CLI wrapper around the DialectVerify unit.

  Command line (facts file):
    qlora_verify [--dialect D] [--timeout SECONDS] PATH...

  The logic lives in the unit: RunVerifyCLI (2.22) forwards the whole
  argument vector to VerifyMain (work root work/verify) and prints the
  result lines, returning the exit code (2 = usage error, 0/1 = verify
  verdicts). This program is the only environment reader in the design: it
  reads QLORA_FPC / QLORA_FPC_UNLEASHED / HOME at start-up, builds the
  TEnvRec with ReadEnvRec, and Halts with the code RunVerifyCLI returns.

  Keep this body to forwarding only: kcov does not trace child processes,
  so every behavior line stays in the unit where the UT measures it in
  process (UT-DialectVerify-134..136); the binary contract itself is pinned
  by tests/ut/qlora_verify_test.pas (UT-qlora_verify-001..003). }
program qlora_verify;

{$mode objfpc}{$H+}

uses
  SysUtils, DialectVerify;

var
  I: LongInt;
  Args: array of string;
begin
  SetLength(Args, ParamCount);
  for I := 0 to ParamCount - 1 do
    Args[I] := ParamStr(I + 1);
  Halt(DialectVerify.RunVerifyCLI(Args, 'work/verify',
    DialectVerify.ReadEnvRec(GetEnvironmentVariable('QLORA_FPC'),
      GetEnvironmentVariable('QLORA_FPC_UNLEASHED'),
      GetEnvironmentVariable('HOME'))));
end.
