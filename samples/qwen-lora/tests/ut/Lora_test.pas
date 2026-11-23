{ UT program for src/Lora.pas: the rows UT-Lora-NNN of
  tests/ut/cases/Lora.cases.md (stage 1 from 001: LoraPath and TLoraPair;
  stage 2 from 101: TLoraSet). Runs on the CPU (.sdlc/ut-coverage.sh hides
  the GPUs). Invoked with `--all --format=plain` (see .sdlc/ut-coverage.sh).

  Skeleton by the harness owner; the test methods are written in T12b. }
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
