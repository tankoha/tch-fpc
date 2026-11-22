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
  DeltaRuleFixture, QwenRefScalar, QwenBlocksMetaFixture,
  fpcunit, testregistry, consoletestrunner;

type
  TQwenBlocksTest = class(TTestCase)
  public
    { The tensor's values as Double, row-major (moved to the CPU). }
    class function Arr(const T: TTensor): TDoubleArray; static;
    { Max |a - b| over two arrays of the same length. }
    class function DiffArr(const A, B: TDoubleArray): Double; static;
  published
    procedure TestUT_QwenBlocks_101;
    procedure TestUT_QwenBlocks_102;
    procedure TestUT_QwenBlocks_103;
    procedure TestUT_QwenBlocks_104;
    procedure TestUT_QwenBlocks_105;
    procedure TestUT_QwenBlocks_106;
    procedure TestUT_QwenBlocks_107;
    procedure TestUT_QwenBlocks_201;
    procedure TestUT_QwenBlocks_202;
    procedure TestUT_QwenBlocks_203;
    procedure TestUT_QwenBlocks_204;
    procedure TestUT_QwenBlocks_211;
    procedure TestUT_QwenBlocks_212;
    procedure TestUT_QwenBlocks_213;
    procedure TestUT_QwenBlocks_214;
    procedure TestUT_QwenBlocks_215;
    procedure TestUT_QwenBlocks_216;
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

{ UT-QwenBlocks-101: ReadQwenDims, the tiny model. }
procedure TQwenBlocksTest.TestUT_QwenBlocks_101;
var
  G: TGgufFile;
  D: TQwenDims;
begin
  G := TGgufFile.Open(TinyPath);
  try
    D := ReadQwenDims(G);
    AssertEquals('Blocks', 2, D.Blocks);
    AssertEquals('Hidden', 64, D.Hidden);
    AssertEquals('Ffn', 96, D.Ffn);
    AssertEquals('HeadsQ', 4, D.HeadsQ);
    AssertEquals('HeadsKV', 2, D.HeadsKV);
    AssertEquals('HeadDim', 32, D.HeadDim);
    AssertEquals('Rot', 16, D.Rot);
    AssertEquals('RopeBase', 10000, D.RopeBase, 1e-12);
    AssertEquals('Eps', 1e-6, D.Eps, 1e-12);
    AssertEquals('SsmHeadsK', 2, D.SsmHeadsK);
    AssertEquals('SsmHeadsV', 4, D.SsmHeadsV);
    AssertEquals('SsmDim', 16, D.SsmDim);
    AssertEquals('ConvKernel', 4, D.ConvKernel);
    AssertEquals('FullInterval', 2, D.FullInterval);
  finally
    G.Free;
  end;
end;

{ UT-QwenBlocks-102: ReadQwenDims, metadata key missing. }
procedure TQwenBlocksTest.TestUT_QwenBlocks_102;
var
  G: TGgufFile;
  P: string;
begin
  P := GetTempDir + 'tiny-tok.gguf';
  WriteTinyGguf(P);
  G := TGgufFile.Open(P);
  try
    try
      ReadQwenDims(G);
      Fail('no EQwenBlock');
    except
      on E: EQwenBlock do
        AssertEquals('metadata key missing: qwen35.block_count', E.Message);
    end;
  finally
    G.Free;
  end;
  DeleteFile(P);
end;

{ UT-QwenBlocks-103: IsFullBlock, interval 4. }
procedure TQwenBlocksTest.TestUT_QwenBlocks_103;
var
  G: TGgufFile;
  D: TQwenDims;
  Expect: array of Boolean;
  I: Integer;
begin
  G := TGgufFile.Open(TinyPath);
  try
    D := ReadQwenDims(G);
    D.FullInterval := 4;
    Expect := [False, False, False, True, False, False, False, True];
    for I := 0 to 7 do
      AssertEquals('IsFullBlock(D, ' + IntToStr(I) + ')', Expect[I], IsFullBlock(D, I));
  finally
    G.Free;
  end;
end;

{ UT-QwenBlocks-104: RmsNorm, hand-computed. }
procedure TQwenBlocksTest.TestUT_QwenBlocks_104;
var
  X, W: TTensor;
begin
  X := TTensor.FromArray([3.0, 4.0]).view([1, 2]);
  W := TTensor.FromArray([1.0, 2.0]);
  AssertTrue('diff <= 1e-6',
    DiffArr(Arr(RmsNorm(X, W, 1e-6)), RefRms([3.0, 4.0], [1.0, 2.0], 1e-6)) <= 1e-6);
end;

{ UT-QwenBlocks-105: RopeText, positions 0..4. }
procedure TQwenBlocksTest.TestUT_QwenBlocks_105;
var
  X, R: TTensor;
begin
  ManualSeed(31);
  X := TTensor.randn([5, 3, 32]);
  R := RopeText(X, 16, 10000);
  AssertTrue('diff <= 1e-6',
    DiffArr(Arr(R), RefRope(Arr(X), 5, 3, 32, 16, 10000, 0)) <= 1e-6);
  AssertTrue('tail exact',
    DiffArr(Arr(R.narrow(2, 16, 16)), Arr(X.narrow(2, 16, 16))) = 0);
end;

{ UT-QwenBlocks-106: RopeText, Pos0 = 7. }
procedure TQwenBlocksTest.TestUT_QwenBlocks_106;
var
  X, R: TTensor;
begin
  ManualSeed(31);
  X := TTensor.randn([5, 3, 32]);
  R := RopeText(X, 16, 10000, 7);
  AssertTrue('diff <= 1e-6',
    DiffArr(Arr(R), RefRope(Arr(X), 5, 3, 32, 16, 10000, 7)) <= 1e-6);
end;

{ UT-QwenBlocks-107: RopeText, position 0. }
procedure TQwenBlocksTest.TestUT_QwenBlocks_107;
var
  X, R: TTensor;
begin
  ManualSeed(31);
  X := TTensor.randn([5, 3, 32]);
  R := RopeText(X.narrow(0, 0, 1), 16, 10000);
  AssertTrue('exact',
    DiffArr(Arr(R), Arr(X.narrow(0, 0, 1))) = 0);
end;

{ UT-QwenBlocks-201: TFullBlock, device and forward T=8. }
procedure TQwenBlocksTest.TestUT_QwenBlocks_201;
var
  G: TGgufFile;
  D: TQwenDims;
  RD: TRefDims;
  B: TFullBlock;
  H: TTensor;
begin
  G := TGgufFile.Open(TinyPath);
  try
    D := ReadQwenDims(G);
    RD := RefDims(G);
    B := TFullBlock.Load(G, 1, D, TDevice.Cpu);
    try
      AssertEquals('device', 'cpu', B.Device.ToString);
      ManualSeed(48);
      H := TTensor.randn([8, 64]);
      AssertTrue('diff <= 1e-4',
        DiffArr(Arr(B.Forward(H)), RefBlock(G, 1, RD, Arr(H), 8)) <= 1e-4);
    finally
      B.Free;
    end;
  finally
    G.Free;
  end;
end;

{ UT-QwenBlocks-202: TFullBlock, forward T=70. }
procedure TQwenBlocksTest.TestUT_QwenBlocks_202;
var
  G: TGgufFile;
  D: TQwenDims;
  RD: TRefDims;
  B: TFullBlock;
  H: TTensor;
begin
  G := TGgufFile.Open(TinyPath);
  try
    D := ReadQwenDims(G);
    RD := RefDims(G);
    B := TFullBlock.Load(G, 1, D, TDevice.Cpu);
    try
      ManualSeed(110);
      H := TTensor.randn([70, 64]);
      AssertTrue('diff <= 1e-4',
        DiffArr(Arr(B.Forward(H)), RefBlock(G, 1, RD, Arr(H), 70)) <= 1e-4);
    finally
      B.Free;
    end;
  finally
    G.Free;
  end;
end;

{ UT-QwenBlocks-203: TFullBlock, causality. }
procedure TQwenBlocksTest.TestUT_QwenBlocks_203;
var
  G: TGgufFile;
  D: TQwenDims;
  B: TFullBlock;
  H, Full, Part: TTensor;
begin
  G := TGgufFile.Open(TinyPath);
  try
    D := ReadQwenDims(G);
    B := TFullBlock.Load(G, 1, D, TDevice.Cpu);
    try
      ManualSeed(48);
      H := TTensor.randn([8, 64]);
      Full := B.Forward(H);
      Part := B.Forward(H.narrow(0, 0, 5));
      AssertTrue('first 5 rows agree <= 1e-5',
        DiffArr(Arr(Full.narrow(0, 0, 5)), Arr(Part)) <= 1e-5);
    finally
      B.Free;
    end;
  finally
    G.Free;
  end;
end;

{ UT-QwenBlocks-204: TFullBlock, backward. }
procedure TQwenBlocksTest.TestUT_QwenBlocks_204;
var
  G: TGgufFile;
  D: TQwenDims;
  B: TFullBlock;
  H: TTensor;
begin
  G := TGgufFile.Open(TinyPath);
  try
    D := ReadQwenDims(G);
    B := TFullBlock.Load(G, 1, D, TDevice.Cpu);
    try
      ManualSeed(48);
      H := TTensor.randn([8, 64]);
      H := H.detach.Copy.requires_grad_(True);
      B.Forward(H).sum(kNone).Backward;
      AssertEquals('grad shape', '[8, 64]', ShapeText(H.grad));
      AssertTrue('grad finite', H.grad.isfinite.all.DoubleValue([]) = 1.0);
    finally
      B.Free;
    end;
  finally
    G.Free;
  end;
end;

{ UT-QwenBlocks-211: TLinearBlock, forward T=8. }
procedure TQwenBlocksTest.TestUT_QwenBlocks_211;
var
  G: TGgufFile;
  D: TQwenDims;
  RD: TRefDims;
  B: TLinearBlock;
  H: TTensor;
begin
  G := TGgufFile.Open(TinyPath);
  try
    D := ReadQwenDims(G);
    RD := RefDims(G);
    B := TLinearBlock.Load(G, 0, D, TDevice.Cpu);
    try
      ManualSeed(48);
      H := TTensor.randn([8, 64]);
      AssertTrue('diff <= 1e-4',
        DiffArr(Arr(B.Forward(H)), RefBlock(G, 0, RD, Arr(H), 8)) <= 1e-4);
    finally
      B.Free;
    end;
  finally
    G.Free;
  end;
end;

{ UT-QwenBlocks-212: TLinearBlock, forward T=70 chunking. }
procedure TQwenBlocksTest.TestUT_QwenBlocks_212;
var
  G: TGgufFile;
  D: TQwenDims;
  RD: TRefDims;
  B: TLinearBlock;
  H: TTensor;
begin
  G := TGgufFile.Open(TinyPath);
  try
    D := ReadQwenDims(G);
    RD := RefDims(G);
    B := TLinearBlock.Load(G, 0, D, TDevice.Cpu);
    try
      ManualSeed(110);
      H := TTensor.randn([70, 64]);
      AssertTrue('diff <= 1e-4',
        DiffArr(Arr(B.Forward(H)), RefBlock(G, 0, RD, Arr(H), 70)) <= 1e-4);
    finally
      B.Free;
    end;
  finally
    G.Free;
  end;
end;

{ UT-QwenBlocks-213: TLinearBlock, causality. }
procedure TQwenBlocksTest.TestUT_QwenBlocks_213;
var
  G: TGgufFile;
  D: TQwenDims;
  B: TLinearBlock;
  H, Full, Part: TTensor;
begin
  G := TGgufFile.Open(TinyPath);
  try
    D := ReadQwenDims(G);
    B := TLinearBlock.Load(G, 0, D, TDevice.Cpu);
    try
      ManualSeed(48);
      H := TTensor.randn([8, 64]);
      Full := B.Forward(H);
      Part := B.Forward(H.narrow(0, 0, 5));
      AssertTrue('first 5 rows agree <= 1e-5',
        DiffArr(Arr(Full.narrow(0, 0, 5)), Arr(Part)) <= 1e-5);
    finally
      B.Free;
    end;
  finally
    G.Free;
  end;
end;

{ UT-QwenBlocks-214: TLinearBlock, backward. }
procedure TQwenBlocksTest.TestUT_QwenBlocks_214;
var
  G: TGgufFile;
  D: TQwenDims;
  B: TLinearBlock;
  H: TTensor;
begin
  G := TGgufFile.Open(TinyPath);
  try
    D := ReadQwenDims(G);
    B := TLinearBlock.Load(G, 0, D, TDevice.Cpu);
    try
      ManualSeed(48);
      H := TTensor.randn([8, 64]);
      H := H.detach.Copy.requires_grad_(True);
      B.Forward(H).sum(kNone).Backward;
      AssertEquals('grad shape', '[8, 64]', ShapeText(H.grad));
      AssertTrue('grad finite', H.grad.isfinite.all.DoubleValue([]) = 1.0);
    finally
      B.Free;
    end;
  finally
    G.Free;
  end;
end;

{ UT-QwenBlocks-215: LoadBlock, class names. }
procedure TQwenBlocksTest.TestUT_QwenBlocks_215;
var
  G: TGgufFile;
  D: TQwenDims;
  B0, B1: TQwenBlock;
begin
  G := TGgufFile.Open(TinyPath);
  try
    D := ReadQwenDims(G);
    B0 := LoadBlock(G, 0, D, TDevice.Cpu);
    B1 := LoadBlock(G, 1, D, TDevice.Cpu);
    try
      AssertEquals('block 0', 'TLinearBlock', B0.ClassName);
      AssertEquals('block 1', 'TFullBlock', B1.ClassName);
    finally
      B0.Free;
      B1.Free;
    end;
  finally
    G.Free;
  end;
end;

{ UT-QwenBlocks-216: ReadQwenDims, float metadata key missing. }
procedure TQwenBlocksTest.TestUT_QwenBlocks_216;
var
  G: TGgufFile;
  P: string;
begin
  P := GetTempDir + 'tiny-nokey.gguf';
  WriteQwenMetaNoRopeBase(P);
  G := TGgufFile.Open(P);
  try
    try
      ReadQwenDims(G);
      Fail('no EQwenBlock');
    except
      on E: EQwenBlock do
        AssertEquals('metadata key missing: qwen35.rope.freq_base', E.Message);
    end;
  finally
    G.Free;
  end;
  DeleteFile(P);
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
