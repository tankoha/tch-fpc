{ UT program for src/QwenBlocks.pas: the rows UT-QwenBlocks-NNN of
  tests/ut/cases/QwenBlocks.cases.md (stage 2 from 101: ReadQwenDims, IsFullBlock, RmsNorm, RopeText; stage 3 from 201: the full block; stage 4 from 211: the linear block).
  The tiny qwen35 GGUF of tests/ut/QwenBlocksFixture.pas is written once
  at start (TinyPath); the scalar oracle is tests/ut/QwenRefScalar.pas.
  Runs on the CPU (.sdlc/ut-coverage.sh hides the GPUs).
  Invoked with `--all --format=plain` (see .sdlc/ut-coverage.sh).

  Skeleton by the harness owner; the test methods are written in T10b. }
program QwenBlocks_test;

{$mode objfpc}{$H+}

uses
  SysUtils, Math, Tch, GgufFile, PackedLinear, QwenBlocks, TokenizerFixture, QwenBlocksFixture,
  DeltaRuleFixture, QwenRefScalar,
  fpcunit, testregistry, consoletestrunner;

type
  TQwenBlocksTest = class(TTestCase)
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

class function TQwenBlocksTest.Arr(const T: TTensor): TDoubleArray;
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

class function TQwenBlocksTest.DiffArr(const A, B: TDoubleArray): Double;
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
  RegisterTest(TQwenBlocksTest);
  App := TTestRunner.Create(nil);
  App.Initialize;
  App.Title := 'QwenBlocks UT';
  App.Run;
  App.Free;
  DeleteFile(TinyPath);
end.
