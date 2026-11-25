{ UT program for src/Lora.pas: the rows UT-Lora-NNN of
  tests/ut/cases/Lora.cases.md (stage 1 from 001: LoraPath and TLoraPair;
  stage 2 from 101: TLoraSet). Runs on the CPU (.sdlc/ut-coverage.sh hides
  the GPUs). Invoked with `--all --format=plain` (see .sdlc/ut-coverage.sh).

  Skeleton by the harness owner; the test methods are written in T12b. 
  (T21: row 301, the initial values drawn on the CPU; the methods are written in T21b.)}
program Lora_test;

{$mode objfpc}{$H+}

uses
  SysUtils, Math, Tch, TchNN, Lora, DeltaRuleFixture,
  fpcunit, testregistry, consoletestrunner;

type
  TLoraTest = class(TTestCase)
  public
    { The tensor's values as Double, row-major (moved to the CPU). }
    class function Arr(const T: TTensor): TDoubleArray; static;
    { Max |a - b| over two arrays of the same length. }
    class function DiffArr(const A, B: TDoubleArray): Double; static;
    { Copies AValues (row-major) into the leaf T without a graph. }
    class procedure SetT(const T: TTensor; const AValues: TDoubleArray); static;
    { The pair's fixed values: A[k, c] = 0.1 (k + 1) - 0.05 c,
      B[o, k] = 0.03 (o - k). }
    class procedure FixPair(P: TLoraPair); static;
    { X [ARows, ACols], X[t, c] = sin(1 + t ACols + c). }
    class function FixX(ARows, ACols: Int64): TTensor; static;
    { s·((X·Aᵀ)·Bᵀ) in Double, row-major [rows of X, OutDim]. }
    class function RefDelta(P: TLoraPair; const X: TTensor): TDoubleArray; static;
  published
    { UT-Lora-001 }
    procedure TestUT_Lora_001;
    { UT-Lora-002 }
    procedure TestUT_Lora_002;
    { UT-Lora-003 }
    procedure TestUT_Lora_003;
    { UT-Lora-004 }
    procedure TestUT_Lora_004;
    { UT-Lora-005 }
    procedure TestUT_Lora_005;
    { UT-Lora-006 }
    procedure TestUT_Lora_006;
    { UT-Lora-007 }
    procedure TestUT_Lora_007;
    { UT-Lora-008 }
    procedure TestUT_Lora_008;
    { UT-Lora-009 }
    procedure TestUT_Lora_009;
    { UT-Lora-010 }
    procedure TestUT_Lora_010;
    { UT-Lora-011 }
    procedure TestUT_Lora_011;
    { UT-Lora-012 }
    procedure TestUT_Lora_012;
    { UT-Lora-013 }
    procedure TestUT_Lora_013;
    { UT-Lora-014 }
    procedure TestUT_Lora_014;
    { UT-Lora-015 }
    procedure TestUT_Lora_015;
    { UT-Lora-016 }
    procedure TestUT_Lora_016;
    { UT-Lora-101 }
    procedure TestUT_Lora_101;
    { UT-Lora-102 }
    procedure TestUT_Lora_102;
    { UT-Lora-103 }
    procedure TestUT_Lora_103;
    { UT-Lora-104 }
    procedure TestUT_Lora_104;
    { UT-Lora-105 }
    procedure TestUT_Lora_105;
    { UT-Lora-106 }
    procedure TestUT_Lora_106;
    { UT-Lora-107 }
    procedure TestUT_Lora_107;
    { UT-Lora-108 }
    procedure TestUT_Lora_108;
    { UT-Lora-109 }
    procedure TestUT_Lora_109;
    { UT-Lora-110 }
    procedure TestUT_Lora_110;
    { UT-Lora-111 }
    procedure TestUT_Lora_111;
    { UT-Lora-112 }
    procedure TestUT_Lora_112;
    { UT-Lora-113 }
    procedure TestUT_Lora_113;
    { UT-Lora-114 }
    procedure TestUT_Lora_114;
    { UT-Lora-301 }
    procedure TestUT_Lora_301;
  end;

var
  { a fresh directory for the files of the rows (Save, Load) }
  TmpDir: string;

class function TLoraTest.Arr(const T: TTensor): TDoubleArray;
begin
  Result := T.detach.to_device(TDevice.Cpu, kDouble).reshape([-1]).ToDoubleArray;
end;

class function TLoraTest.DiffArr(const A, B: TDoubleArray): Double;
var
  I: Integer;
begin
  Result := 0;
  for I := 0 to High(A) do
    Result := Max(Result, Abs(A[I] - B[I]));
end;

class procedure TLoraTest.SetT(const T: TTensor; const AValues: TDoubleArray);
var
  G: IInterface;
begin
  G := NoGradGuard;
  T.CopyFrom(TTensor.FromArray(AValues).view(T.Shape));
end;

class procedure TLoraTest.FixPair(P: TLoraPair);
var
  V: TDoubleArray;
  K, C, O: Int64;
begin
  V := nil;
  SetLength(V, P.Rank * P.InDim);
  for K := 0 to P.Rank - 1 do
    for C := 0 to P.InDim - 1 do
      V[K * P.InDim + C] := 0.1 * (K + 1) - 0.05 * C;
  SetT(P.A, V);
  V := nil;
  SetLength(V, P.OutDim * P.Rank);
  for O := 0 to P.OutDim - 1 do
    for K := 0 to P.Rank - 1 do
      V[O * P.Rank + K] := 0.03 * (O - K);
  SetT(P.B, V);
end;

class function TLoraTest.FixX(ARows, ACols: Int64): TTensor;
var
  V: TDoubleArray;
  I: Int64;
begin
  V := nil;
  SetLength(V, ARows * ACols);
  for I := 0 to High(V) do
    V[I] := Sin(1 + I);
  Result := TTensor.FromArray(V).view([ARows, ACols]);
end;

class function TLoraTest.RefDelta(P: TLoraPair; const X: TTensor): TDoubleArray;
var
  XA, A, B, U: TDoubleArray;
  T, Rows, K, C, O: Int64;
  Acc: Double;
begin
  XA := Arr(X);
  A := Arr(P.A);
  B := Arr(P.B);
  Rows := Length(XA) div P.InDim;
  U := nil;
  SetLength(U, P.Rank);
  Result := nil;
  SetLength(Result, Rows * P.OutDim);
  for T := 0 to Rows - 1 do
  begin
    for K := 0 to P.Rank - 1 do
    begin
      Acc := 0;
      for C := 0 to P.InDim - 1 do
        Acc := Acc + XA[T * P.InDim + C] * A[K * P.InDim + C];
      U[K] := Acc;
    end;
    for O := 0 to P.OutDim - 1 do
    begin
      Acc := 0;
      for K := 0 to P.Rank - 1 do
        Acc := Acc + U[K] * B[O * P.Rank + K];
      Result[T * P.OutDim + O] := P.Scale * Acc;
    end;
  end;
end;

{ a pair 'p' (8 -> 6, rank 3, scale 2) in its own store; the caller frees both }
function NewPair(out S: TVarStore): TLoraPair;
begin
  S := TVarStore.Create(TDevice.Cpu);
  Result := TLoraPair.Create(S.Root / 'p', 'p', 8, 6, 3, 2.0);
end;

procedure CheckCreate(T: TTestCase; AIn, AOut: Int64; const AMessage: string);
var
  S: TVarStore;
begin
  S := TVarStore.Create(TDevice.Cpu);
  try
    try
      TLoraPair.Create(S.Root / 'p', 'p', AIn, AOut, 3, 2.0).Free;
      T.Fail('no ELora');
    except
      on E: ELora do
        T.AssertEquals(AMessage, E.Message);
    end;
    T.AssertEquals(0, S.Count);
  finally
    S.Free;
  end;
end;

procedure CheckDelta(T: TTestCase; const X: TTensor);
var
  S: TVarStore;
  P: TLoraPair;
begin
  P := NewPair(S);
  try
    try
      P.Delta(X);
      T.Fail('no ELora');
    except
      on E: ELora do
        T.AssertEquals('p: X must end in 8', E.Message);
    end;
  finally
    P.Free;
    S.Free;
  end;
end;

{ UT-Lora-001 }
procedure TLoraTest.TestUT_Lora_001;
var
  S: TVarStore;
begin
  S := TVarStore.Create(TDevice.Cpu);
  try
    AssertEquals('blk.3.attn_q.lora_a', LoraPath(S.Root, 'blk.3.attn_q').FullName('lora_a'));
  finally
    S.Free;
  end;
end;

{ UT-Lora-002 }
procedure TLoraTest.TestUT_Lora_002;
var
  S: TVarStore;
begin
  S := TVarStore.Create(TDevice.Cpu);
  try
    AssertEquals('x.y', LoraPath(S.Root, 'x').FullName('y'));
  finally
    S.Free;
  end;
end;

{ UT-Lora-003 }
procedure TLoraTest.TestUT_Lora_003;
var
  S: TVarStore;
begin
  S := TVarStore.Create(TDevice.Cpu);
  try
    try
      LoraPath(S.Root, 'blk..q');
      Fail('no ETorchError');
    except
      on E: ETorchError do
        AssertEquals('sub name cannot be empty or contain "." or "|": ""', E.Message);
    end;
  finally
    S.Free;
  end;
end;

{ UT-Lora-004 }
procedure TLoraTest.TestUT_Lora_004;
var
  S: TVarStore;
  P: TLoraPair;
  N: TNamedTensorArray;
begin
  P := NewPair(S);
  try
    AssertEquals('p', P.Name);
    AssertEquals(8, P.InDim);
    AssertEquals(6, P.OutDim);
    AssertEquals(3, P.Rank);
    AssertTrue(Format('scale %.6f', [P.Scale]), P.Scale = 2.0);
    AssertEquals('[3, 8]', ShapeText(P.A));
    AssertEquals('[6, 3]', ShapeText(P.B));
    AssertTrue('A requires grad', P.A.RequiresGrad);
    AssertTrue('B requires grad', P.B.RequiresGrad);
    N := S.Variables;
    AssertEquals(2, Length(N));
    AssertEquals('p.lora_a', N[0].Name);
    AssertEquals('p.lora_b', N[1].Name);
    AssertEquals(2, Length(S.TrainableVariables));
  finally
    P.Free;
    S.Free;
  end;
end;

{ UT-Lora-005 }
procedure TLoraTest.TestUT_Lora_005;
var
  S: TVarStore;
  P: TLoraPair;
  A, B: TDoubleArray;
  I: Integer;
  DMax: Double;
begin
  ManualSeed(1);
  P := NewPair(S);
  try
    A := Arr(P.A);
    B := Arr(P.B);
    DMax := 0;
    for I := 0 to High(A) do
    begin
      AssertTrue(Format('A[%d] = %.9f', [I, A[I]]), Abs(A[I]) <= 1.0 / Sqrt(8.0) + 1.0e-7);
      if Abs(A[I]) > DMax then DMax := Abs(A[I]);
    end;
    AssertTrue(Format('max |A| = %.9f', [DMax]), DMax > 0.3);
    DMax := 0;
    for I := 0 to High(B) do
      if Abs(B[I]) > DMax then DMax := Abs(B[I]);
    AssertTrue(Format('max |B| = %.9f', [DMax]), DMax = 0.0);
  finally
    P.Free;
    S.Free;
  end;
end;

{ UT-Lora-006 }
procedure TLoraTest.TestUT_Lora_006;
begin
  CheckCreate(Self, 0, 6, 'p: dimensions must be positive, got 0 -> 6');
end;

{ UT-Lora-007 }
procedure TLoraTest.TestUT_Lora_007;
begin
  CheckCreate(Self, 8, -1, 'p: dimensions must be positive, got 8 -> -1');
end;

{ UT-Lora-008 }
procedure TLoraTest.TestUT_Lora_008;
var
  S: TVarStore;
  P: TLoraPair;
  R: TTensor;
  A: TDoubleArray;
  I: Integer;
  DMax: Double;
begin
  P := NewPair(S);
  try
    R := P.Delta(FixX(5, 8));
    AssertEquals('[5, 6]', ShapeText(R));
    A := Arr(R);
    DMax := 0;
    for I := 0 to High(A) do
      if Abs(A[I]) > DMax then DMax := Abs(A[I]);
    AssertTrue(Format('max |R| = %.9f', [DMax]), DMax = 0.0);
  finally
    P.Free;
    S.Free;
  end;
end;

{ UT-Lora-009 }
procedure TLoraTest.TestUT_Lora_009;
var
  S: TVarStore;
  P: TLoraPair;
  X: TTensor;
  D: Double;
begin
  P := NewPair(S);
  try
    FixPair(P);
    X := FixX(5, 8);
    D := DiffArr(Arr(P.Delta(X)), RefDelta(P, X));
    AssertTrue(Format('diff %.3e', [D]), D <= 1e-6);
  finally
    P.Free;
    S.Free;
  end;
end;

{ UT-Lora-010 }
procedure TLoraTest.TestUT_Lora_010;
var
  S: TVarStore;
  P: TLoraPair;
  X, R: TTensor;
  D: Double;
begin
  P := NewPair(S);
  try
    FixPair(P);
    X := FixX(6, 8);
    R := P.Delta(X.view([2, 3, 8]));
    AssertEquals('[2, 3, 6]', ShapeText(R));
    D := DiffArr(Arr(R), RefDelta(P, X));
    AssertTrue(Format('diff %.3e', [D]), D <= 1e-6);
  finally
    P.Free;
    S.Free;
  end;
end;

{ UT-Lora-011 }
procedure TLoraTest.TestUT_Lora_011;
begin
  CheckDelta(Self, TTensor.zeros([5, 7]));
end;

{ UT-Lora-012 }
procedure TLoraTest.TestUT_Lora_012;
begin
  CheckDelta(Self, TTensor.zeros([5, 7]).sum(kNone));
end;

{ UT-Lora-013 }
procedure TLoraTest.TestUT_Lora_013;
var
  S: TVarStore;
  P: TLoraPair;
  M: TTensor;
  A, B, Ref: TDoubleArray;
  O, K, C: Int64;
  D: Double;
begin
  P := NewPair(S);
  try
    FixPair(P);
    M := P.MergedDelta;
    AssertEquals('[6, 8]', ShapeText(M));
    AssertFalse('MergedDelta requires grad', M.RequiresGrad);
    AssertTrue('MergedDelta kind', M.Kind = kFloat);
    A := Arr(P.A);
    B := Arr(P.B);
    SetLength(Ref, P.OutDim * P.InDim);
    for O := 0 to P.OutDim - 1 do
      for C := 0 to P.InDim - 1 do
      begin
        Ref[O * P.InDim + C] := 0;
        for K := 0 to P.Rank - 1 do
          Ref[O * P.InDim + C] := Ref[O * P.InDim + C] + 2.0 * B[O * P.Rank + K] * A[K * P.InDim + C];
      end;
    D := DiffArr(Arr(M), Ref);
    AssertTrue(Format('diff %.3e', [D]), D <= 1e-6);
  finally
    P.Free;
    S.Free;
  end;
end;

{ UT-Lora-014 }
procedure TLoraTest.TestUT_Lora_014;
var
  S: TVarStore;
  P: TLoraPair;
  X: TTensor;
  D: Double;
begin
  P := NewPair(S);
  try
    FixPair(P);
    X := FixX(5, 8);
    D := DiffArr(Arr(X.matmul(P.MergedDelta.t)), Arr(P.Delta(X)));
    AssertTrue(Format('diff %.3e', [D]), D <= 1e-5);
  finally
    P.Free;
    S.Free;
  end;
end;

{ UT-Lora-015 }
procedure TLoraTest.TestUT_Lora_015;
var
  S: TVarStore;
  P: TLoraPair;
  X: TTensor;
  AG, BG, XA, AA: TDoubleArray;
  O, K, T2, C: Int64;
  Rows: Int64;
  Acc, DMax: Double;
begin
  ManualSeed(1);
  P := NewPair(S);
  try
    X := FixX(5, 8);
    P.Delta(X).sum(kNone).Backward;
    AG := Arr(P.A.grad);
    DMax := 0;
    for C := 0 to High(AG) do
      if Abs(AG[C]) > DMax then DMax := Abs(AG[C]);
    AssertTrue(Format('max |A.grad| = %.9f', [DMax]), DMax = 0.0);
    BG := Arr(P.B.grad);
    XA := Arr(X);
    AA := Arr(P.A);
    Rows := Length(XA) div P.InDim;
    for O := 0 to P.OutDim - 1 do
      for K := 0 to P.Rank - 1 do
      begin
        Acc := 0;
        for T2 := 0 to Rows - 1 do
          for C := 0 to P.InDim - 1 do
            Acc := Acc + XA[T2 * P.InDim + C] * AA[K * P.InDim + C];
        Acc := 2.0 * Acc;
        AssertTrue(Format('B.grad[%d, %d] = %.9f, ref %.9f', [O, K, BG[O * P.Rank + K], Acc]),
          Abs(BG[O * P.Rank + K] - Acc) <= 1e-5);
      end;
  finally
    P.Free;
    S.Free;
  end;
end;

{ UT-Lora-016 }
procedure TLoraTest.TestUT_Lora_016;
var
  S: TVarStore;
  P: TLoraPair;
  X, R: TTensor;
  D: Double;
begin
  P := NewPair(S);
  try
    FixPair(P);
    X := FixX(5, 8).totype(kDouble);
    R := P.Delta(X);
    AssertTrue('Delta kind', R.Kind = kDouble);
    AssertTrue('A kind', P.A.Kind = kFloat);
    D := DiffArr(Arr(R), RefDelta(P, X));
    AssertTrue(Format('diff %.3e', [D]), D <= 1e-12);
  finally
    P.Free;
    S.Free;
  end;
end;

{ stage 2: TLoraSet }

{ a set (rank 4, scale 2) with blk.0.attn_qkv (64 -> 128) and blk.1.attn_q (64 -> 256) }
function NewSet: TLoraSet;
begin
  Result := TLoraSet.Create(4, 2.0);
  Result.Add('blk.0.attn_qkv', 64, 128, TDevice.Cpu);
  Result.Add('blk.1.attn_q', 64, 256, TDevice.Cpu);
end;

procedure CheckPair(T: TTestCase; S: TLoraSet; AIndex: Integer; const AMessage: string);
begin
  try
    S.Pair(AIndex);
    T.Fail('no ELora');
  except
    on E: ELora do
      T.AssertEquals(AMessage, E.Message);
  end;
end;

{ UT-Lora-101 }
procedure TLoraTest.TestUT_Lora_101;
var
  S: TLoraSet;
begin
  S := TLoraSet.Create(4, 2.0);
  try
    AssertEquals(4, S.Rank);
    AssertTrue(Format('scale %.6f', [S.Scale]), S.Scale = 2.0);
    AssertEquals(0, S.Count);
    AssertEquals(0, Length(S.Stores));
    AssertEquals(0, S.ParameterCount);
  finally
    S.Free;
  end;
end;

{ UT-Lora-102 }
procedure TLoraTest.TestUT_Lora_102;
begin
  try
    TLoraSet.Create(0, 2.0).Free;
    Fail('no ELora');
  except
    on E: ELora do
      AssertEquals('rank must be positive, got 0', E.Message);
  end;
end;

{ UT-Lora-103 }
procedure TLoraTest.TestUT_Lora_103;
var
  S: TLoraSet;
  N: TNamedTensorArray;
begin
  S := NewSet;
  try
    AssertEquals(2, S.Count);
    AssertEquals('blk.0.attn_qkv', S.Pair(0).Name);
    AssertEquals('blk.1.attn_q', S.Pair(1).Name);
    AssertEquals(4, S.Pair(1).Rank);
    AssertTrue(Format('scale %.6f', [S.Pair(1).Scale]), S.Pair(1).Scale = 2.0);
    AssertEquals(256, S.Pair(1).OutDim);
    AssertEquals(1, Length(S.Stores));
    AssertEquals('cpu', S.Stores[0].Device.ToString);
    N := S.Stores[0].Variables;
    AssertEquals(4, Length(N));
    AssertEquals('blk.0.attn_qkv.lora_a', N[0].Name);
    AssertEquals('blk.0.attn_qkv.lora_b', N[1].Name);
    AssertEquals('blk.1.attn_q.lora_a', N[2].Name);
    AssertEquals('blk.1.attn_q.lora_b', N[3].Name);
    AssertEquals(2048, S.ParameterCount);
  finally
    S.Free;
  end;
end;

{ UT-Lora-104 }
procedure TLoraTest.TestUT_Lora_104;
var
  S: TLoraSet;
begin
  S := NewSet;
  try
    try
      S.Add('blk.0.attn_qkv', 64, 128, TDevice.Cpu);
      Fail('no ELora');
    except
      on E: ELora do
        AssertEquals('pair already present: blk.0.attn_qkv', E.Message);
    end;
    AssertEquals(2, S.Count);
  finally
    S.Free;
  end;
end;

{ UT-Lora-105 }
procedure TLoraTest.TestUT_Lora_105;
var
  S: TLoraSet;
begin
  S := NewSet;
  try
    AssertTrue('Find lower-case', S.Find('blk.1.attn_q') = S.Pair(1));
    AssertFalse('Find upper-case is nil', S.Find('BLK.1.ATTN_Q') <> nil);
    AssertFalse('Find empty is nil', S.Find('') <> nil);
    S.Add('BLK.1.ATTN_Q', 64, 256, TDevice.Cpu);
    AssertEquals(3, S.Count);
    AssertTrue('Find upper-case after Add', S.Find('BLK.1.ATTN_Q') = S.Pair(2));
  finally
    S.Free;
  end;
end;

{ UT-Lora-106 }
procedure TLoraTest.TestUT_Lora_106;
var
  S: TLoraSet;
begin
  S := NewSet;
  try
    CheckPair(Self, S, 2, 'pair 2 outside 0..1');
    CheckPair(Self, S, -1, 'pair -1 outside 0..1');
  finally
    S.Free;
  end;
end;

{ UT-Lora-107 }
procedure TLoraTest.TestUT_Lora_107;
var
  S: TLoraSet;
begin
  S := TLoraSet.Create(4, 2.0);
  try
    CheckPair(Self, S, 0, 'pair 0 outside 0..-1');
  finally
    S.Free;
  end;
end;

{ UT-Lora-108 }
procedure TLoraTest.TestUT_Lora_108;
var
  S, S2: TLoraSet;
  I: Integer;
  G: IInterface;
begin
  ManualSeed(3);
  S := NewSet;
  S2 := NewSet;
  try
    G := NoGradGuard;
    for I := 0 to S.Count - 1 do
      S.Pair(I).B.CopyFrom(TTensor.randn(S.Pair(I).B.Shape));
    G := nil;
    S.Save(TmpDir + 'a.ot');
    S2.Load(TmpDir + 'a.ot');
    AssertTrue('a.ot exists', FileExists(TmpDir + 'a.ot'));
    for I := 0 to S.Count - 1 do
    begin
      AssertTrue(Format('A[%d] exact', [I]), DiffArr(Arr(S2.Pair(I).A), Arr(S.Pair(I).A)) = 0.0);
      AssertTrue(Format('B[%d] exact', [I]), DiffArr(Arr(S2.Pair(I).B), Arr(S.Pair(I).B)) = 0.0);
    end;
  finally
    S2.Free;
    S.Free;
  end;
end;

{ UT-Lora-109 }
procedure TLoraTest.TestUT_Lora_109;
var
  S, S2: TLoraSet;
begin
  S := TLoraSet.Create(4, 2.0);
  S2 := TLoraSet.Create(3, 2.0);
  S.Add('blk.0.attn_qkv', 64, 128, TDevice.Cpu);
  S.Add('blk.1.attn_q', 64, 256, TDevice.Cpu);
  S2.Add('blk.0.attn_qkv', 64, 128, TDevice.Cpu);
  S2.Add('blk.1.attn_q', 64, 256, TDevice.Cpu);
  try
    S.Save(TmpDir + 'b.ot');
    try
      S2.Load(TmpDir + 'b.ot');
      Fail('no ETorchNativeError');
    except
      on E: ETorchNativeError do
        AssertEquals('The size of tensor a (3) must match the size of tensor b (4)' +
          ' at non-singleton dimension 0', E.Message);
    end;
  finally
    S2.Free;
    S.Free;
  end;
end;

{ UT-Lora-110 }
procedure TLoraTest.TestUT_Lora_110;
var
  S: TLoraSet;
begin
  S := NewSet;
  try
    try
      S.Load(TmpDir + 'missing.ot');
      Fail('no ETorchNativeError');
    except
      on E: ETorchNativeError do
        AssertEquals('open file failed because of errno 2 on fopen: No such file or directory, ' +
          'file path: ' + TmpDir + 'missing.ot', E.Message);
    end;
  finally
    S.Free;
  end;
end;

{ UT-Lora-111 }
procedure TLoraTest.TestUT_Lora_111;
var
  S, S2: TLoraSet;
begin
  S := NewSet;
  S2 := NewSet;
  S2.Add('blk.2.attn_qkv', 64, 128, TDevice.Cpu);
  try
    S.Save(TmpDir + 'c.ot');
    try
      S2.Load(TmpDir + 'c.ot');
      Fail('no ETorchError');
    except
      on E: ETorchError do
        AssertEquals('variable blk.2.attn_qkv.lora_a not found in ' + TmpDir + 'c.ot', E.Message);
    end;
  finally
    S2.Free;
    S.Free;
  end;
end;

{ UT-Lora-112 }
procedure TLoraTest.TestUT_Lora_112;
var
  S, S2: TLoraSet;
  OldDir: string;
begin
  ManualSeed(4);
  OldDir := GetCurrentDir;
  S := NewSet;
  S2 := NewSet;
  try
    try
      SetCurrentDir(TmpDir);
      S.Save('rel.ot');
      S2.Load('rel.ot');
    finally
      SetCurrentDir(OldDir);
    end;
    AssertTrue('rel.ot exists', FileExists(TmpDir + 'rel.ot'));
    AssertTrue('Pair(1).A exact', DiffArr(Arr(S2.Pair(1).A), Arr(S.Pair(1).A)) = 0.0);
  finally
    S2.Free;
    S.Free;
  end;
end;

{ UT-Lora-113 }
procedure TLoraTest.TestUT_Lora_113;
var
  S, S2: TLoraSet;
  A0: TTensor;
begin
  S := NewSet;
  S2 := NewSet;
  try
    A0 := S2.Pair(0).A;
    S.Save(TmpDir + 'd.ot');
    S2.Load(TmpDir + 'd.ot');
    AssertTrue('same handle', S2.Pair(0).A.SameHandle(A0));
    AssertTrue('A requires grad', S2.Pair(0).A.RequiresGrad);
    AssertTrue('B requires grad', S2.Pair(0).B.RequiresGrad);
    AssertTrue('A0 exact vs S''s A', DiffArr(Arr(S2.Pair(0).A), Arr(S.Pair(0).A)) = 0.0);
  finally
    S2.Free;
    S.Free;
  end;
end;

{ UT-Lora-114 }
procedure TLoraTest.TestUT_Lora_114;
var
  S: TLoraSet;
begin
  S := TLoraSet.Create(4, 2.0);
  try
    try
      S.Add('', 64, 128, TDevice.Cpu);
      Fail('no ETorchError');
    except
      on E: ETorchError do
        AssertEquals('sub name cannot be empty or contain "." or "|": ""', E.Message);
    end;
  finally
    S.Free;
  end;
end;

{ UT-Lora-301 }
procedure TLoraTest.TestUT_Lora_301;
var
  S1, S2: TVarStore;
  P: TLoraPair;
  N1, N2, Old: TTensor;
  B: TDoubleArray;
  I: Integer;
  M: Double;
begin
  ManualSeed(7);
  S1 := TVarStore.Create(TDevice.Cpu);
  P := TLoraPair.Create(S1.Root / 'p', 'p', 8, 6, 3, 2.0);
  try
    N1 := TTensor.rand([4], kFloat, TDevice.Cpu);
    ManualSeed(7);
    S2 := TVarStore.Create(TDevice.Cpu);
    try
      Old := S2.Root.Uniform('x', [3, 8], -1 / Sqrt(8), 1 / Sqrt(8));
      N2 := TTensor.rand([4], kFloat, TDevice.Cpu);
      AssertTrue('A equals Old (exact)', DiffArr(Arr(P.A), Arr(Old)) = 0.0);
      AssertTrue('N1 equals N2 (exact)', DiffArr(Arr(N1), Arr(N2)) = 0.0);
      AssertTrue('A requires grad', P.A.RequiresGrad);
      AssertEquals('S1.Count', 2, S1.Count);
      B := Arr(P.B);
      M := 0;
      for I := 0 to High(B) do
        M := Max(M, Abs(B[I]));
      AssertTrue('B all zero', M = 0.0);
    finally
      S2.Free;
    end;
  finally
    P.Free;
    S1.Free;
  end;
end;

var
  App: TTestRunner;
begin
  TmpDir := GetTempDir + 'lora-ut-' + IntToStr(GetProcessID) + PathDelim;
  ForceDirectories(TmpDir);
  RegisterTest(TLoraTest);
  App := TTestRunner.Create(nil);
  App.Initialize;
  App.Title := 'Lora UT';
  App.Run;
  App.Free;
  DeleteFile(TmpDir + 'a.ot');
  DeleteFile(TmpDir + 'b.ot');
  DeleteFile(TmpDir + 'c.ot');
  DeleteFile(TmpDir + 'd.ot');
  DeleteFile(TmpDir + 'rel.ot');
  RemoveDir(TmpDir);
end.
