{ UT program for src/PackedLinear.pas: the rows UT-PackedLinear-NNN of
  tests/ut/cases/PackedLinear.cases.md (stage 1 from 001: TPackedLinear; stage 1 also 012 to 015: LoadPacked, LoadDense; T12 from 101: the LoRA pair of a matrix).
  The tiny qwen35 GGUF of tests/ut/QwenBlocksFixture.pas is written once
  at start (TinyPath); the scalar oracle is tests/ut/QwenRefScalar.pas.
  Runs on the CPU (.sdlc/ut-coverage.sh hides the GPUs).
  Invoked with `--all --format=plain` (see .sdlc/ut-coverage.sh).

  Skeleton by the harness owner; the test methods are written in T10b and T12b. 
  (T21 from 301: the compute kind; the methods are written in T21b.)}
program PackedLinear_test;

{$mode objfpc}{$H+}

uses
  SysUtils, Math, Tch, TchNN, GgufFile, GgufQuant, PackedLinear, Lora, QwenBlocksFixture,
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
    procedure TestUT_PackedLinear_101;
    procedure TestUT_PackedLinear_102;
    procedure TestUT_PackedLinear_103;
    procedure TestUT_PackedLinear_104;
    procedure TestUT_PackedLinear_105;
    procedure TestUT_PackedLinear_106;
    procedure TestUT_PackedLinear_107;
    procedure TestUT_PackedLinear_301;
    procedure TestUT_PackedLinear_302;
    procedure TestUT_PackedLinear_303;
    procedure TestUT_PackedLinear_304;
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

{ stage 2 (T12): the LoRA pair of a matrix }

{ an F32 matrix 6 x 8, W[o, c] = 0.01 (8 o + c) - 0.2 }
function NewLinear: TPackedLinear;
var
  V: array[0..47] of Single;
  I: Integer;
begin
  for I := 0 to 47 do
    V[I] := 0.01 * I - 0.2;
  Result := TPackedLinear.Create(TTensor.FromBytes(V[0], [48 * 4], kUInt8), 0, 6, 8);
end;

{ X [5, 8], X[t, c] = sin(1 + 8 t + c) }
function FixX: TTensor;
var
  V: array of Double;
  I: Integer;
begin
  V := nil;
  SetLength(V, 40);
  for I := 0 to 39 do
    V[I] := Sin(1 + I);
  Result := TTensor.FromArray(V).view([5, 8]);
end;

{ the pair 'q' (AIn -> AOut, rank 3, scale 2) in S }
function NewPair(S: TVarStore; AIn, AOut: Int64): TLoraPair;
begin
  Result := TLoraPair.Create(S.Root / 'q', 'q', AIn, AOut, 3, 2.0);
end;

{ B[o, k] = 0.03 (o - k) }
procedure SetB(P: TLoraPair);
var
  G: IInterface;
  V: array of Double;
  O, K: Integer;
begin
  V := nil;
  SetLength(V, P.OutDim * 3);
  for O := 0 to P.OutDim - 1 do
    for K := 0 to 2 do
      V[O * 3 + K] := 0.03 * (O - K);
  G := NoGradGuard;
  P.B.CopyFrom(TTensor.FromArray(V).view([P.OutDim, 3]));
end;

procedure CheckSetLora(T: TTestCase; AIn, AOut: Int64; const AMessage: string);
var
  L: TPackedLinear;
  S: TVarStore;
  P: TLoraPair;
begin
  L := NewLinear;
  S := TVarStore.Create(TDevice.Cpu);
  P := NewPair(S, AIn, AOut);
  try
    try
      L.Lora := P;
      T.Fail('no EPackedLinear');
    except
      on E: EPackedLinear do
        T.AssertEquals(AMessage, E.Message);
    end;
    T.AssertTrue(L.Lora = nil);
  finally
    L.Free;
    P.Free;
    S.Free;
  end;
end;

{ UT-PackedLinear-101: Lora, default. }
procedure TPackedLinearTest.TestUT_PackedLinear_101;
var
  L: TPackedLinear;
begin
  L := NewLinear;
  try
    AssertTrue('L.Lora = nil', L.Lora = nil);
  finally
    L.Free;
  end;
end;

{ UT-PackedLinear-102: Forward with a pair, B zero. }
procedure TPackedLinearTest.TestUT_PackedLinear_102;
var
  L: TPackedLinear;
  S: TVarStore;
  P: TLoraPair;
  Y0, Y1: TTensor;
begin
  L := NewLinear;
  S := TVarStore.Create(TDevice.Cpu);
  P := NewPair(S, 8, 6);
  try
    Y0 := L.Forward(FixX);
    L.Lora := P;
    AssertTrue('L.Lora = P', L.Lora = P);
    Y1 := L.Forward(FixX);
    AssertTrue('equals Y0 (exact)', DiffArr(Arr(Y0), Arr(Y1)) = 0);
  finally
    L.Free;
    P.Free;
    S.Free;
  end;
end;

{ UT-PackedLinear-103: Forward, the delta vs the references. }
procedure TPackedLinearTest.TestUT_PackedLinear_103;
var
  L: TPackedLinear;
  S: TVarStore;
  P: TLoraPair;
  X, Y: TTensor;
  G: IInterface;
begin
  ManualSeed(2);
  L := NewLinear;
  S := TVarStore.Create(TDevice.Cpu);
  P := NewPair(S, 8, 6);
  SetB(P);
  L.Lora := P;
  try
    G := NoGradGuard;
    X := FixX;
    Y := L.Forward(X);
    AssertTrue('vs X@W + P.Delta(X)',
      DiffArr(Arr(Y), Arr(X.matmul(L.Weight.t) + P.Delta(X))) <= 1.0e-6);
    AssertTrue('vs X@(W + MergedDelta)',
      DiffArr(Arr(Y), Arr(X.matmul((L.Weight + P.MergedDelta).t))) <= 1.0e-5);
    AssertTrue('differs from the base forward',
      DiffArr(Arr(Y), Arr(X.matmul(L.Weight.t))) > 0.1);
  finally
    L.Free;
    P.Free;
    S.Free;
  end;
end;

{ UT-PackedLinear-104: SetLora, InDim differs. }
procedure TPackedLinearTest.TestUT_PackedLinear_104;
begin
  CheckSetLora(Self, 7, 6, 'LoRA pair q is 7 -> 6, the matrix 8 -> 6');
end;

{ UT-PackedLinear-105: SetLora, OutDim differs. }
procedure TPackedLinearTest.TestUT_PackedLinear_105;
begin
  CheckSetLora(Self, 8, 5, 'LoRA pair q is 8 -> 5, the matrix 8 -> 6');
end;

{ UT-PackedLinear-106: Lora := nil removes the pair. }
procedure TPackedLinearTest.TestUT_PackedLinear_106;
var
  L: TPackedLinear;
  S: TVarStore;
  P: TLoraPair;
  Y0, Y1: TTensor;
begin
  L := NewLinear;
  S := TVarStore.Create(TDevice.Cpu);
  P := NewPair(S, 8, 6);
  try
    Y0 := L.Forward(FixX);
    SetB(P);
    L.Lora := P;
    L.Lora := nil;
    AssertTrue('L.Lora = nil', L.Lora = nil);
    Y1 := L.Forward(FixX);
    AssertTrue('equals Y0 (exact)', DiffArr(Arr(Y0), Arr(Y1)) = 0);
  finally
    L.Free;
    P.Free;
    S.Free;
  end;
end;

{ UT-PackedLinear-107: Backward through Forward. }
procedure TPackedLinearTest.TestUT_PackedLinear_107;
var
  L: TPackedLinear;
  S: TVarStore;
  P: TLoraPair;
  V: TDoubleArray;
  M: Double;
  I: Integer;
begin
  ManualSeed(2);
  L := NewLinear;
  S := TVarStore.Create(TDevice.Cpu);
  P := NewPair(S, 8, 6);
  try
    L.Lora := P;
    L.Forward(FixX).sum(kNone).Backward;
    AssertEquals('B.grad shape', '[6, 3]', ShapeText(P.B.grad));
    V := Arr(P.B.grad);
    M := 0;
    for I := 0 to High(V) do
      M := Max(M, Abs(V[I]));
    AssertTrue('max |B.grad| > 0.1', M > 0.1);
    V := Arr(P.A.grad);
    M := 0;
    for I := 0 to High(V) do
      M := Max(M, Abs(V[I]));
    AssertTrue('max |A.grad| 0 (exact)', M = 0);
  finally
    L.Free;
    P.Free;
    S.Free;
  end;
end;

{ UT-PackedLinear-301: Compute, the setter and the default. }
procedure TPackedLinearTest.TestUT_PackedLinear_301;
var
  L: TPackedLinear;
begin
  L := NewLinear;
  try
    AssertTrue('default kFloat', L.Compute = kFloat);
    L.Compute := kBFloat16;
    AssertTrue('kBFloat16', L.Compute = kBFloat16);
    try
      L.Compute := kHalf;
      Fail('no EPackedLinear');
    except
      on E: EPackedLinear do
      begin
        AssertEquals('compute must be kFloat or kBFloat16', E.Message);
        AssertTrue('still kBFloat16', L.Compute = kBFloat16);
      end;
    end;
    try
      L.Compute := kDouble;
      Fail('no EPackedLinear');
    except
      on E: EPackedLinear do
        AssertEquals('compute must be kFloat or kBFloat16', E.Message);
    end;
    L.Compute := kFloat;
    AssertTrue('kFloat', L.Compute = kFloat);
  finally
    L.Free;
  end;
end;

{ UT-PackedLinear-302: Forward, the bf16 path. }
procedure TPackedLinearTest.TestUT_PackedLinear_302;
var
  L: TPackedLinear;
  X, Y, Y32: TTensor;
begin
  L := NewLinear;
  try
    X := FixX.totype(kFloat);
    Y32 := L.Forward(X);
    L.Compute := kBFloat16;
    Y := L.Forward(X);
    AssertTrue('Y kind kFloat', Y.Kind = kFloat);
    AssertEquals('shape', '[5, 6]', ShapeText(Y));
    AssertTrue('equals the bf16 product (exact)',
      DiffArr(Arr(Y),
        Arr(X.totype(kBFloat16).matmul(L.Weight.totype(kBFloat16).transpose(0, 1)).totype(kFloat))) = 0);
    AssertTrue('max |Y - Y32| > 0', DiffArr(Arr(Y), Arr(Y32)) > 0);
    AssertTrue('max |Y - Y32| <= 0.02', DiffArr(Arr(Y), Arr(Y32)) <= 0.02);
    L.Compute := kFloat;
    AssertTrue('Forward(X) equals Y32 again (exact)', DiffArr(Arr(L.Forward(X)), Arr(Y32)) = 0);
  finally
    L.Free;
  end;
end;

{ UT-PackedLinear-303: Forward, bf16 with a LoRA pair and backward. }
procedure TPackedLinearTest.TestUT_PackedLinear_303;
var
  L: TPackedLinear;
  S: TVarStore;
  P: TLoraPair;
  X, Xc, Y: TTensor;
  V: TDoubleArray;
  I: Integer;
  SumSq: Double;
begin
  L := NewLinear;
  S := TVarStore.Create(TDevice.Cpu);
  P := NewPair(S, 8, 6);
  try
    X := FixX.totype(kFloat);
    SetB(P);
    L.Lora := P;
    L.Compute := kBFloat16;
    Xc := X.totype(kBFloat16);
    Y := L.Forward(X);
    Y.sum(kNone).Backward;
    AssertTrue('Y kind kFloat', Y.Kind = kFloat);
    AssertTrue('equals bf16 product + Delta (exact)',
      DiffArr(Arr(Y),
        Arr((Xc.matmul(L.Weight.totype(kBFloat16).transpose(0, 1)) + P.Delta(Xc)).totype(kFloat))) = 0);
    AssertTrue('A.grad kind kFloat', P.A.grad.Kind = kFloat);
    V := Arr(P.A.grad);
    SumSq := 0;
    for I := 0 to High(V) do
      SumSq := SumSq + V[I] * V[I];
    AssertTrue('A.grad sum of squares > 0', SumSq > 0);
    AssertTrue('B.grad kind kFloat', P.B.grad.Kind = kFloat);
    V := Arr(P.B.grad);
    SumSq := 0;
    for I := 0 to High(V) do
      SumSq := SumSq + V[I] * V[I];
    AssertTrue('B.grad sum of squares > 0', SumSq > 0);
  finally
    L.Free;
    P.Free;
    S.Free;
  end;
end;

{ UT-PackedLinear-304: Forward, bf16 and a non-fp32 X. }
procedure TPackedLinearTest.TestUT_PackedLinear_304;
var
  L: TPackedLinear;
begin
  L := NewLinear;
  try
    L.Compute := kBFloat16;
    try
      L.Forward(FixX.totype(kBFloat16));
      Fail('no EPackedLinear');
    except
      on E: EPackedLinear do
        AssertEquals('X must be kFloat (fp32)', E.Message);
    end;
  finally
    L.Free;
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
