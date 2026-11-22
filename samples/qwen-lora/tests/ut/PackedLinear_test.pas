{ UT program for src/PackedLinear.pas: the rows UT-PackedLinear-NNN of
  tests/ut/cases/PackedLinear.cases.md (stage 1 from 001: TPackedLinear; stage 1 also 012 to 015: LoadPacked, LoadDense).
  The tiny qwen35 GGUF of tests/ut/QwenBlocksFixture.pas is written once
  at start (TinyPath); the scalar oracle is tests/ut/QwenRefScalar.pas.
  Runs on the CPU (.sdlc/ut-coverage.sh hides the GPUs).
  Invoked with `--all --format=plain` (see .sdlc/ut-coverage.sh).

  Skeleton by the harness owner; the test methods are written in T10b. }
program PackedLinear_test;

{$mode objfpc}{$H+}

uses
  SysUtils, Math, Tch, GgufFile, GgufQuant, PackedLinear, QwenBlocksFixture,
  DeltaRuleFixture, QwenRefScalar,
  fpcunit, testregistry, consoletestrunner;

type
  TPackedLinearTest = class(TTestCase)
  public
    { The tensor's values as Double, row-major (moved to the CPU). }
    class function Arr(const T: TTensor): TDoubleArray; static;
    { Max |a - b| over two arrays of the same length. }
    class function DiffArr(const A, B: TDoubleArray): Double; static;
  published
  end;

var
  { the tiny model, written by WriteTinyQwen(TinyPath, 5) before the tests }
  TinyPath: string;

class function TPackedLinearTest.Arr(const T: TTensor): TDoubleArray;
var
  F: TTensor;
  I: Integer;
begin
  F := T.to_device(TDevice.Cpu, kDouble).reshape([-1]);
  Result := nil;
  SetLength(Result, F.Shape[0]);
  for I := 0 to High(Result) do
    Result[I] := F.DoubleValue([I]);
end;

class function TPackedLinearTest.DiffArr(const A, B: TDoubleArray): Double;
var
  I: Integer;
begin
  Result := 0;
  for I := 0 to High(A) do
    Result := Max(Result, Abs(A[I] - B[I]));
end;

var
  App: TTestRunner;
begin
  TinyPath := GetTempDir + 'tiny-qwen-ut-' + IntToStr(GetProcessID) + '.gguf';
  WriteTinyQwen(TinyPath, 5);
  RegisterTest(TPackedLinearTest);
  App := TTestRunner.Create(nil);
  App.Initialize;
  App.Title := 'PackedLinear UT';
  App.Run;
  App.Free;
  DeleteFile(TinyPath);
end.
