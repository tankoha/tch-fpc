{ UT program for src/QwenModel.pas: the rows UT-QwenModel-NNN of
  tests/ut/cases/QwenModel.cases.md (stage 1 from 001: Load, Embed, Stages,
  HeadChunk; stage 2 from 101: Trunk, Logits, LastLogProbs).
  The tiny qwen35 models of tests/ut/QwenBlocksFixture.pas
  (WriteTinyQwenModel) are written once at start; the scalar oracle is
  RefLogits of tests/ut/QwenRefScalar.pas.
  Runs on the CPU (.sdlc/ut-coverage.sh hides the GPUs).
  Invoked with `--all --format=plain` (see .sdlc/ut-coverage.sh).

  Skeleton by the harness owner; the test methods are written in T11b. }
program QwenModel_test;

{$mode objfpc}{$H+}

uses
  SysUtils, Math, Tch, GgufFile, GgufQuant, PackedLinear, QwenBlocks, Recompute, QwenModel,
  TokenizerFixture, QwenBlocksFixture, QwenRefScalar, DeltaRuleFixture,
  fpcunit, testregistry, consoletestrunner;

type
  TQwenModelTest = class(TTestCase)
  public
    { The tensor's values as Double, row-major (moved to the CPU). }
    class function Arr(const T: TTensor): TDoubleArray; static;
    { Max |a - b| over two arrays of the same length. }
    class function DiffArr(const A, B: TDoubleArray): Double; static;
    { Opens APath and loads the model on the CPU (split 0). }
    class function LoadCpu(const APath: string; out G: TGgufFile): TQwenModel; static;
  published
  end;

const
  Ids8: array[0..7] of Integer = (5, 17, 2, 33, 8, 0, 273, 121);

var
  { written before the tests by WriteTinyQwenModel(.., 5, ...) }
  TiedPath, OwnPath, NextPath, NonePath, BadEmbdPath, BadOutPath: string;

class function TQwenModelTest.Arr(const T: TTensor): TDoubleArray;
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

class function TQwenModelTest.DiffArr(const A, B: TDoubleArray): Double;
var
  I: Integer;
begin
  Result := 0;
  for I := 0 to High(A) do
    Result := Max(Result, Abs(A[I] - B[I]));
end;

class function TQwenModelTest.LoadCpu(const APath: string; out G: TGgufFile): TQwenModel;
begin
  G := TGgufFile.Open(APath);
  try
    Result := TQwenModel.Load(G, TDevice.Cpu, TDevice.Cpu, 0);
  except
    G.Free;
    raise;
  end;
end;

{ Max |logits - RefLogits| for AIds on the CPU model of APath (HeadChunk
  AChunk when > 0); raises when the logits are not [T, 274]. }
function LogitsDiff(const APath: string; const AIds: array of Integer; AChunk: Int64 = 0): Double;
var
  G: TGgufFile;
  M: TQwenModel;
  L: TTensor;
begin
  M := TQwenModelTest.LoadCpu(APath, G);
  try
    if AChunk > 0 then
      M.HeadChunk := AChunk;
    L := M.Logits(M.Trunk(M.Embed(AIds)));
    if ShapeText(L) <> Format('[%d, 274]', [Length(AIds)]) then
      raise Exception.Create('shape ' + ShapeText(L));
    Result := TQwenModelTest.DiffArr(TQwenModelTest.Arr(L), RefLogits(G, RefDims(G), AIds));
  finally
    M.Free;
    G.Free;
  end;
end;

var
  App: TTestRunner;
  Base: string;
begin
  Base := GetTempDir + 'tiny-qwenmodel-ut-' + IntToStr(GetProcessID);
  TiedPath := Base + '-tied.gguf';
  OwnPath := Base + '-own.gguf';
  NextPath := Base + '-next.gguf';
  NonePath := Base + '-none.gguf';
  BadEmbdPath := Base + '-badembd.gguf';
  BadOutPath := Base + '-badout.gguf';
  WriteTinyQwenModel(TiedPath, 5, thTied, False);
  WriteTinyQwenModel(OwnPath, 5, thOwn, False);
  WriteTinyQwenModel(NextPath, 5, thTied, True);
  WriteTinyQwenModel(NonePath, 5, thNone, False);
  WriteTinyQwenModel(BadEmbdPath, 5, thBadEmbd, False);
  WriteTinyQwenModel(BadOutPath, 5, thBadOutput, False);
  RegisterTest(TQwenModelTest);
  App := TTestRunner.Create(nil);
  App.Initialize;
  App.Title := 'QwenModel UT';
  App.Run;
  App.Free;
  DeleteFile(TiedPath);
  DeleteFile(OwnPath);
  DeleteFile(NextPath);
  DeleteFile(NonePath);
  DeleteFile(BadEmbdPath);
  DeleteFile(BadOutPath);
end.
