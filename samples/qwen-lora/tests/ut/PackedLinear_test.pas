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
    procedure TestUT_PackedLinear_001;
    procedure TestUT_PackedLinear_002;
    procedure TestUT_PackedLinear_003;
    procedure TestUT_PackedLinear_004;
    procedure TestUT_PackedLinear_005;
    procedure TestUT_PackedLinear_006;
    procedure TestUT_PackedLinear_007;
    procedure TestUT_PackedLinear_008;
    procedure TestUT_PackedLinear_009;
    procedure TestUT_PackedLinear_010;
    procedure TestUT_PackedLinear_011;
    procedure TestUT_PackedLinear_012;
    procedure TestUT_PackedLinear_013;
    procedure TestUT_PackedLinear_014;
    procedure TestUT_PackedLinear_015;
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

{ UT-PackedLinear-001: Create, the Q8_0 matrix. }
procedure TPackedLinearTest.TestUT_PackedLinear_001;
var
  V: TSingleArray;
  Bytes: TBytes;
  Raw: TTensor;
  L: TPackedLinear;
  I: Integer;
begin
  SetLength(V, 64);
  for I := 0 to 63 do
    V[I] := I / 10 - 3;
  Bytes := QuantizeQ8_0(V);
  Raw := TTensor.FromBytes(Bytes[0], [68], kUInt8);
  L := TPackedLinear.Create(Raw, 8, 2, 32);
  try
    AssertEquals('OutDim', 2, L.OutDim);
    AssertEquals('InDim', 32, L.InDim);
    AssertEquals('TType', 8, L.TType);
    AssertEquals('Device', 'cpu', L.Device.ToString);
  finally
    L.Free;
  end;
end;

{ UT-PackedLinear-002: Weight of the Q8_0 matrix. }
procedure TPackedLinearTest.TestUT_PackedLinear_002;
var
  V: TSingleArray;
  Bytes: TBytes;
  Raw: TTensor;
  D: TSingleArray;
  R: TDoubleArray;
  L: TPackedLinear;
  I: Integer;
begin
  SetLength(V, 64);
  for I := 0 to 63 do
    V[I] := I / 10 - 3;
  Bytes := QuantizeQ8_0(V);
  Raw := TTensor.FromBytes(Bytes[0], [68], kUInt8);
  L := TPackedLinear.Create(Raw, 8, 2, 32);
  try
    AssertEquals('shape', '[2, 32]', ShapeText(L.Weight));
    D := DequantizeBytes(8, Bytes, 64);
    SetLength(R, 64);
    for I := 0 to 63 do
      R[I] := D[I];
    AssertTrue('values exact', DiffArr(Arr(L.Weight), R) = 0);
  finally
    L.Free;
  end;
end;

{ UT-PackedLinear-003: Forward, X [3, 32]. }
procedure TPackedLinearTest.TestUT_PackedLinear_003;
const
  Xv: array[0..2] of Double = (1.0, -2.0, 0.5);
var
  V: TSingleArray;
  Bytes: TBytes;
  Raw: TTensor;
  D: TSingleArray;
  L: TPackedLinear;
  X, Y: TTensor;
  I, J, K: Integer;
  S: Double;
begin
  SetLength(V, 64);
  for I := 0 to 63 do
    V[I] := I / 10 - 3;
  Bytes := QuantizeQ8_0(V);
  Raw := TTensor.FromBytes(Bytes[0], [68], kUInt8);
  L := TPackedLinear.Create(Raw, 8, 2, 32);
  X := TTensor.FromArray([1.0, -2.0, 0.5]).view([3, 1]) * TTensor.ones([1, 32]);
  D := DequantizeBytes(8, Bytes, 64);
  try
    Y := L.Forward(X);
    AssertEquals('shape', '[3, 2]', ShapeText(Y));
    for I := 0 to 2 do
      for J := 0 to 1 do
      begin
        S := 0;
        for K := 0 to 31 do
          S := S + D[J * 32 + K] * Xv[I];
        AssertTrue('Y[' + IntToStr(I) + ',' + IntToStr(J) + ']',
          Abs(Y.DoubleValue([I, J]) - S) <= 1.0e-5);
      end;
  finally
    L.Free;
  end;
end;

{ UT-PackedLinear-004: Forward, 3-D X. }
procedure TPackedLinearTest.TestUT_PackedLinear_004;
var
  V: TSingleArray;
  Bytes: TBytes;
  Raw: TTensor;
  L: TPackedLinear;
  I: Integer;
begin
  SetLength(V, 64);
  for I := 0 to 63 do
    V[I] := I / 10 - 3;
  Bytes := QuantizeQ8_0(V);
  Raw := TTensor.FromBytes(Bytes[0], [68], kUInt8);
  L := TPackedLinear.Create(Raw, 8, 2, 32);
  try
    AssertEquals('shape', '[2, 5, 2]', ShapeText(L.Forward(TTensor.ones([2, 5, 32]))));
  finally
    L.Free;
  end;
end;

{ UT-PackedLinear-005: F32 case. }
procedure TPackedLinearTest.TestUT_PackedLinear_005;
var
  S: array[0..5] of Single;
  Bytes: TBytes;
  Raw: TTensor;
  L: TPackedLinear;
  I: Integer;
  W: TTensor;
begin
  for I := 0 to 5 do
    S[I] := I - 2.5;
  SetLength(Bytes, 24);
  Move(S[0], Bytes[0], SizeOf(S));
  Raw := TTensor.FromBytes(Bytes[0], [24], kUInt8);
  L := TPackedLinear.Create(Raw, 0, 2, 3);
  try
    W := L.Weight;
    AssertEquals('shape', '[2, 3]', ShapeText(W));
    AssertEquals('w[0,0]', -2.5, W.DoubleValue([0, 0]));
    AssertEquals('w[0,1]', -1.5, W.DoubleValue([0, 1]));
    AssertEquals('w[0,2]', -0.5, W.DoubleValue([0, 2]));
    AssertEquals('w[1,0]', 0.5, W.DoubleValue([1, 0]));
    AssertEquals('w[1,1]', 1.5, W.DoubleValue([1, 1]));
    AssertEquals('w[1,2]', 2.5, W.DoubleValue([1, 2]));
  finally
    L.Free;
  end;
end;

{ UT-PackedLinear-006: Create, wrong kind (kFloat). }
procedure TPackedLinearTest.TestUT_PackedLinear_006;
begin
  try
    TPackedLinear.Create(TTensor.zeros([34]), 8, 1, 32);
    Fail('no EPackedLinear');
  except
    on E: EPackedLinear do
      AssertEquals('the packed bytes must be a 1-D kUInt8 tensor, got [34]', E.Message);
  end;
end;

{ UT-PackedLinear-007: Create, 2-D raw. }
procedure TPackedLinearTest.TestUT_PackedLinear_007;
begin
  try
    TPackedLinear.Create(TTensor.zeros([2, 34], kUInt8), 8, 1, 32);
    Fail('no EPackedLinear');
  except
    on E: EPackedLinear do
      AssertEquals('the packed bytes must be a 1-D kUInt8 tensor, got [2, 34]', E.Message);
  end;
end;

{ UT-PackedLinear-008: Create, dimension 0. }
procedure TPackedLinearTest.TestUT_PackedLinear_008;
begin
  try
    TPackedLinear.Create(TTensor.zeros([34], kUInt8), 8, 0, 32);
    Fail('no EPackedLinear');
  except
    on E: EPackedLinear do
      AssertEquals('dimensions must be positive, got 0 x 32', E.Message);
  end;
end;

{ UT-PackedLinear-009: Create, wrong byte count. }
procedure TPackedLinearTest.TestUT_PackedLinear_009;
begin
  try
    TPackedLinear.Create(TTensor.zeros([34], kUInt8), 8, 2, 32);
    Fail('no EPackedLinear');
  except
    on E: EPackedLinear do
      AssertEquals('[34] holds 34 bytes, 2 x 32 of type q8_0 need 68', E.Message);
  end;
end;

{ UT-PackedLinear-010: Forward, X not kFloat. }
procedure TPackedLinearTest.TestUT_PackedLinear_010;
begin
  try
    TPackedLinear.Create(TTensor.zeros([34], kUInt8), 8, 1, 32)
      .Forward(TTensor.zeros([3, 32], kDouble));
    Fail('no EPackedLinear');
  except
    on E: EPackedLinear do
      AssertEquals('X must be kFloat (fp32)', E.Message);
  end;
end;

{ UT-PackedLinear-011: Forward, wrong last dim. }
procedure TPackedLinearTest.TestUT_PackedLinear_011;
begin
  try
    TPackedLinear.Create(TTensor.zeros([34], kUInt8), 8, 1, 32)
      .Forward(TTensor.zeros([3, 31]));
    Fail('no EPackedLinear');
  except
    on E: EPackedLinear do
      AssertEquals('X must end in 32, got [3, 31]', E.Message);
  end;
end;

{ UT-PackedLinear-012: LoadPacked, ffn_down. }
procedure TPackedLinearTest.TestUT_PackedLinear_012;
var
  G: TGgufFile;
  L: TPackedLinear;
begin
  G := TGgufFile.Open(TinyPath);
  try
    L := LoadPacked(G, 'blk.0.ffn_down.weight', TDevice.Cpu);
    try
      AssertEquals('OutDim', 64, L.OutDim);
      AssertEquals('InDim', 96, L.InDim);
      AssertEquals('TType', 8, L.TType);
    finally
      L.Free;
    end;
  finally
    G.Free;
  end;
end;

{ UT-PackedLinear-013: LoadDense, ssm_conv1d. }
procedure TPackedLinearTest.TestUT_PackedLinear_013;
var
  G: TGgufFile;
  W: TTensor;
begin
  G := TGgufFile.Open(TinyPath);
  try
    W := LoadDense(G, 'blk.0.ssm_conv1d.weight', TDevice.Cpu);
    AssertEquals('shape', '[128, 4]', ShapeText(W));
    AssertTrue('kind kFloat', W.Kind = kFloat);
  finally
    G.Free;
  end;
end;

{ UT-PackedLinear-014: LoadPacked, tensor not found. }
procedure TPackedLinearTest.TestUT_PackedLinear_014;
var
  G: TGgufFile;
begin
  G := TGgufFile.Open(TinyPath);
  try
    try
      LoadPacked(G, 'blk.9.x', TDevice.Cpu);
      Fail('no EPackedLinear');
    except
      on E: EPackedLinear do
        AssertEquals('tensor not in the file: blk.9.x', E.Message);
    end;
  finally
    G.Free;
  end;
end;

{ UT-PackedLinear-015: LoadPacked, not a matrix. }
procedure TPackedLinearTest.TestUT_PackedLinear_015;
var
  G: TGgufFile;
begin
  G := TGgufFile.Open(TinyPath);
  try
    try
      LoadPacked(G, 'blk.0.attn_norm.weight', TDevice.Cpu);
      Fail('no EPackedLinear');
    except
      on E: EPackedLinear do
        AssertEquals('blk.0.attn_norm.weight is not a matrix (1 dimensions)', E.Message);
    end;
  finally
    G.Free;
  end;
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
