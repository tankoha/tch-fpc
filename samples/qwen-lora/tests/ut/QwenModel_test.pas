{ UT program for src/QwenModel.pas: the rows UT-QwenModel-NNN of
  tests/ut/cases/QwenModel.cases.md (stage 1 from 001: Load, Embed, Stages,
  HeadChunk; stage 2 from 101: Trunk, Logits, LastLogProbs; T12 from 201: AttachLora).
  The tiny qwen35 models of tests/ut/QwenBlocksFixture.pas
  (WriteTinyQwenModel) are written once at start; the scalar oracle is
  RefLogits of tests/ut/QwenRefScalar.pas.
  Runs on the CPU (.sdlc/ut-coverage.sh hides the GPUs).
  Invoked with `--all --format=plain` (see .sdlc/ut-coverage.sh).

  Skeleton by the harness owner; the test methods are written in T11b and T12b. }
program QwenModel_test;

{$mode objfpc}{$H+}

uses
  SysUtils, Math, Tch, GgufFile, GgufQuant, PackedLinear, QwenBlocks, Recompute, QwenModel, Lora,
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
    procedure TestUT_QwenModel_001;
    procedure TestUT_QwenModel_002;
    procedure TestUT_QwenModel_003;
    procedure TestUT_QwenModel_004;
    procedure TestUT_QwenModel_005;
    procedure TestUT_QwenModel_006;
    procedure TestUT_QwenModel_007;
    procedure TestUT_QwenModel_008;
    procedure TestUT_QwenModel_009;
    procedure TestUT_QwenModel_010;
    procedure TestUT_QwenModel_011;
    procedure TestUT_QwenModel_012;
    procedure TestUT_QwenModel_013;
    procedure TestUT_QwenModel_014;
    procedure TestUT_QwenModel_015;
    procedure TestUT_QwenModel_016;
    procedure TestUT_QwenModel_101;
    procedure TestUT_QwenModel_102;
    procedure TestUT_QwenModel_103;
    procedure TestUT_QwenModel_104;
    procedure TestUT_QwenModel_105;
    procedure TestUT_QwenModel_106;
    procedure TestUT_QwenModel_107;
    procedure TestUT_QwenModel_108;
    procedure TestUT_QwenModel_109;
    procedure TestUT_QwenModel_110;
    procedure TestUT_QwenModel_201;
    procedure TestUT_QwenModel_202;
    procedure TestUT_QwenModel_203;
    procedure TestUT_QwenModel_204;
    procedure TestUT_QwenModel_205;
    procedure TestUT_QwenModel_206;
    procedure TestUT_QwenModel_207;
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

{ The log-softmax of the ARow'th row of R (AVocab values wide), computed in
  Double per the case table: R[t, j] - max - Ln(Σ exp(R[t, k] - max)). }
function Lp(const R: TDoubleArray; ARow, AVocab: Integer): TDoubleArray;
var
  I: Integer;
  Mx, S: Double;
begin
  Mx := -Infinity;
  for I := 0 to AVocab - 1 do
    Mx := Max(Mx, R[ARow * AVocab + I]);
  S := 0;
  for I := 0 to AVocab - 1 do
    S := S + Exp(R[ARow * AVocab + I] - Mx);
  Result := nil;
  SetLength(Result, AVocab);
  for I := 0 to AVocab - 1 do
    Result[I] := R[ARow * AVocab + I] - Mx - Ln(S);
end;

{ UT-QwenModel-001: Load, tied head. }
procedure TQwenModelTest.TestUT_QwenModel_001;
var
  G: TGgufFile;
  M: TQwenModel;
begin
  M := LoadCpu(TiedPath, G);
  try
    AssertEquals(2, M.TrunkBlocks);
    AssertEquals(0, M.Split);
    AssertEquals(274, M.Vocab);
    AssertTrue(M.Tied);
    AssertEquals(8192, M.HeadChunk);
    AssertEquals(64, M.Dims.Hidden);
  finally
    M.Free;
    G.Free;
  end;
end;

{ UT-QwenModel-002: Load, own head. }
procedure TQwenModelTest.TestUT_QwenModel_002;
var
  G: TGgufFile;
  M: TQwenModel;
begin
  M := LoadCpu(OwnPath, G);
  try
    AssertFalse(M.Tied);
    AssertEquals(274, M.Vocab);
  finally
    M.Free;
    G.Free;
  end;
end;

{ UT-QwenModel-003: Load, NextN file. }
procedure TQwenModelTest.TestUT_QwenModel_003;
var
  G: TGgufFile;
  M: TQwenModel;
  D: TQwenDims;
begin
  M := LoadCpu(NextPath, G);
  try
    D := ReadQwenDims(G);
    AssertEquals(3, D.Blocks);
    AssertEquals(2, TrunkBlockCount(G, ReadQwenDims(G)));
    AssertEquals(2, M.TrunkBlocks);
  finally
    M.Free;
    G.Free;
  end;
end;

{ UT-QwenModel-004: TrunkBlockCount on an opened file, no model. }
procedure TQwenModelTest.TestUT_QwenModel_004;
var
  G: TGgufFile;
begin
  G := TGgufFile.Open(TiedPath);
  try
    AssertEquals(2, TrunkBlockCount(G, ReadQwenDims(G)));
  finally
    G.Free;
  end;
end;

{ UT-QwenModel-005: Load, split 3. }
procedure TQwenModelTest.TestUT_QwenModel_005;
var
  G: TGgufFile;
begin
  G := TGgufFile.Open(TiedPath);
  try
    try
      TQwenModel.Load(G, TDevice.Cpu, TDevice.Cpu, 3);
      Fail('no EQwenModel');
    except
      on E: EQwenModel do
        AssertEquals('split 3 outside 0..2', E.Message);
    end;
  finally
    G.Free;
  end;
end;

{ UT-QwenModel-006: Load, split -1. }
procedure TQwenModelTest.TestUT_QwenModel_006;
var
  G: TGgufFile;
begin
  G := TGgufFile.Open(TiedPath);
  try
    try
      TQwenModel.Load(G, TDevice.Cpu, TDevice.Cpu, -1);
      Fail('no EQwenModel');
    except
      on E: EQwenModel do
        AssertEquals('split -1 outside 0..2', E.Message);
    end;
  finally
    G.Free;
  end;
end;

{ UT-QwenModel-007: Load, no token_embd. }
procedure TQwenModelTest.TestUT_QwenModel_007;
var
  G: TGgufFile;
begin
  G := TGgufFile.Open(NonePath);
  try
    try
      TQwenModel.Load(G, TDevice.Cpu, TDevice.Cpu, 0);
      Fail('no EPackedLinear');
    except
      on E: EPackedLinear do
        AssertEquals('tensor not in the file: token_embd.weight', E.Message);
    end;
  finally
    G.Free;
  end;
end;

{ UT-QwenModel-008: Load, bad embedding shape. }
procedure TQwenModelTest.TestUT_QwenModel_008;
var
  G: TGgufFile;
begin
  G := TGgufFile.Open(BadEmbdPath);
  try
    try
      TQwenModel.Load(G, TDevice.Cpu, TDevice.Cpu, 0);
      Fail('no EQwenModel');
    except
      on E: EQwenModel do
        AssertEquals('token_embd.weight must be [vocab, hidden]', E.Message);
    end;
  finally
    G.Free;
  end;
end;

{ UT-QwenModel-009: Load, bad output shape. }
procedure TQwenModelTest.TestUT_QwenModel_009;
var
  G: TGgufFile;
begin
  G := TGgufFile.Open(BadOutPath);
  try
    try
      TQwenModel.Load(G, TDevice.Cpu, TDevice.Cpu, 0);
      Fail('no EQwenModel');
    except
      on E: EQwenModel do
        AssertEquals('output.weight must be [vocab, hidden]', E.Message);
    end;
  finally
    G.Free;
  end;
end;

{ UT-QwenModel-010: Embed, exact rows. }
procedure TQwenModelTest.TestUT_QwenModel_010;
var
  G: TGgufFile;
  M: TQwenModel;
  E: TTensor;
  Bytes: TBytes;
  Ref: TSingleArray;
  Got: TDoubleArray;
  Ids: array of Integer;
  K, C: Integer;
begin
  M := LoadCpu(TiedPath, G);
  try
    Ids := [3, 0, 273];
    E := M.Embed(Ids);
    AssertEquals('[3, 64]', ShapeText(E));
    G.ReadTensorBytes(G.FindTensor('token_embd.weight'), Bytes);
    Ref := DequantizeBytes(8, Bytes, 274 * 64);
    Got := Arr(E);
    for K := 0 to 2 do
      for C := 0 to 63 do
        AssertEquals(0.0, Got[K * 64 + C] - Double(Ref[Ids[K] * 64 + C]), 0.0);
  finally
    M.Free;
    G.Free;
  end;
end;

{ UT-QwenModel-011: Embed, empty array. }
procedure TQwenModelTest.TestUT_QwenModel_011;
var
  G: TGgufFile;
  M: TQwenModel;
  Empty: array of Integer;
begin
  M := LoadCpu(TiedPath, G);
  try
    try
      M.Embed(Empty);
      Fail('no EQwenModel');
    except
      on E: EQwenModel do
        AssertEquals('no token ids', E.Message);
    end;
  finally
    M.Free;
    G.Free;
  end;
end;

{ UT-QwenModel-012: Embed, id 274. }
procedure TQwenModelTest.TestUT_QwenModel_012;
var
  G: TGgufFile;
  M: TQwenModel;
begin
  M := LoadCpu(TiedPath, G);
  try
    try
      M.Embed([274]);
      Fail('no EQwenModel');
    except
      on E: EQwenModel do
        AssertEquals('token id 274 outside 0..273', E.Message);
    end;
  finally
    M.Free;
    G.Free;
  end;
end;

{ UT-QwenModel-013: Embed, id -1. }
procedure TQwenModelTest.TestUT_QwenModel_013;
var
  G: TGgufFile;
  M: TQwenModel;
begin
  M := LoadCpu(TiedPath, G);
  try
    try
      M.Embed([5, -1]);
      Fail('no EQwenModel');
    except
      on E: EQwenModel do
        AssertEquals('token id -1 outside 0..273', E.Message);
    end;
  finally
    M.Free;
    G.Free;
  end;
end;

{ UT-QwenModel-014: Stages. }
procedure TQwenModelTest.TestUT_QwenModel_014;
var
  G: TGgufFile;
  M: TQwenModel;
  S: TStageArray;
begin
  M := LoadCpu(TiedPath, G);
  try
    S := M.Stages;
    AssertEquals(2, Length(S));
    AssertEquals('TLinearBlock', S[0].ClassName);
    AssertEquals('TFullBlock', S[1].ClassName);
    AssertEquals('cpu', S[0].Device.ToString);
  finally
    M.Free;
    G.Free;
  end;
end;

{ UT-QwenModel-015: Load, split 1 and 2 on one file. }
procedure TQwenModelTest.TestUT_QwenModel_015;
var
  G: TGgufFile;
  M1, M2: TQwenModel;
begin
  G := TGgufFile.Open(TiedPath);
  try
    M1 := TQwenModel.Load(G, TDevice.Cpu, TDevice.Cpu, 1);
    try
      AssertEquals(1, M1.Split);
    finally
      M1.Free;
    end;
    M2 := TQwenModel.Load(G, TDevice.Cpu, TDevice.Cpu, 2);
    try
      AssertEquals(2, M2.Split);
    finally
      M2.Free;
    end;
  finally
    G.Free;
  end;
end;

{ UT-QwenModel-016: HeadChunk 0. }
procedure TQwenModelTest.TestUT_QwenModel_016;
var
  G: TGgufFile;
  M: TQwenModel;
begin
  M := LoadCpu(TiedPath, G);
  try
    try
      M.HeadChunk := 0;
      Fail('no EQwenModel');
    except
      on E: EQwenModel do
        AssertEquals('head chunk must be positive, got 0', E.Message);
    end;
  finally
    M.Free;
    G.Free;
  end;
end;

{ UT-QwenModel-101: Trunk + Logits vs the scalar oracle, tied head. }
procedure TQwenModelTest.TestUT_QwenModel_101;
var
  D: Double;
begin
  D := LogitsDiff(TiedPath, Ids8);
  AssertTrue(Format('diff %.3e', [D]), D <= 1e-4);
end;

{ UT-QwenModel-102: Trunk + Logits vs the scalar oracle, own head. }
procedure TQwenModelTest.TestUT_QwenModel_102;
var
  D: Double;
begin
  D := LogitsDiff(OwnPath, Ids8);
  AssertTrue(Format('diff %.3e', [D]), D <= 1e-4);
end;

{ UT-QwenModel-103: Trunk + Logits vs the scalar oracle, NextN file. }
procedure TQwenModelTest.TestUT_QwenModel_103;
var
  D: Double;
begin
  D := LogitsDiff(NextPath, Ids8);
  AssertTrue(Format('diff %.3e', [D]), D <= 1e-4);
end;

{ UT-QwenModel-104: Trunk + Logits vs the oracle, 70 tokens (two delta-rule chunks). }
procedure TQwenModelTest.TestUT_QwenModel_104;
var
  D: Double;
  Ids: array of Integer;
  I: Integer;
begin
  Ids := nil;
  SetLength(Ids, 70);
  for I := 0 to High(Ids) do
    Ids[I] := I * 37 mod 274;
  D := LogitsDiff(TiedPath, Ids);
  AssertTrue(Format('diff %.3e', [D]), D <= 1e-4);
end;

{ UT-QwenModel-105: HeadChunk 100: three chunks (100, 100, 74). }
procedure TQwenModelTest.TestUT_QwenModel_105;
var
  G: TGgufFile;
  M: TQwenModel;
  H, A, B: TTensor;
  D: Double;
begin
  M := LoadCpu(TiedPath, G);
  try
    H := M.Trunk(M.Embed(Ids8));
    A := M.Logits(H);
    M.HeadChunk := 100;
    AssertEquals(100, M.HeadChunk);
    B := M.Logits(H);
    AssertEquals('[8, 274]', ShapeText(B));
    D := DiffArr(Arr(A), Arr(B));
    AssertTrue(Format('diff %.3e', [D]), D <= 1e-5);
  finally
    M.Free;
    G.Free;
  end;
end;

{ UT-QwenModel-106: LastLogProbs vs the oracle's row 7, tied head. }
procedure TQwenModelTest.TestUT_QwenModel_106;
var
  G: TGgufFile;
  M: TQwenModel;
  LPt: TTensor;
  R: TDoubleArray;
  D: Double;
begin
  M := LoadCpu(TiedPath, G);
  try
    LPt := M.LastLogProbs(Ids8);
    AssertEquals('[274]', ShapeText(LPt));
    AssertFalse(LPt.RequiresGrad);
    R := RefLogits(G, RefDims(G), Ids8);
    D := DiffArr(Arr(LPt), Lp(R, 7, 274));
    AssertTrue(Format('diff %.3e', [D]), D <= 1e-4);
  finally
    M.Free;
    G.Free;
  end;
end;

{ UT-QwenModel-107: Logits, H not [T, Hidden]. }
procedure TQwenModelTest.TestUT_QwenModel_107;
var
  G: TGgufFile;
  M: TQwenModel;
begin
  M := LoadCpu(TiedPath, G);
  try
    try
      M.Logits(TTensor.zeros([3, 32]));
      Fail('no EQwenModel');
    except
      on E: EQwenModel do
        AssertEquals('H must be [T, 64]', E.Message);
    end;
  finally
    M.Free;
    G.Free;
  end;
end;

{ UT-QwenModel-108: Logits, H 1-D. }
procedure TQwenModelTest.TestUT_QwenModel_108;
var
  G: TGgufFile;
  M: TQwenModel;
begin
  M := LoadCpu(TiedPath, G);
  try
    try
      M.Logits(TTensor.zeros([64]));
      Fail('no EQwenModel');
    except
      on E: EQwenModel do
        AssertEquals('H must be [T, 64]', E.Message);
    end;
  finally
    M.Free;
    G.Free;
  end;
end;

{ UT-QwenModel-109: backward through the whole model. }
procedure TQwenModelTest.TestUT_QwenModel_109;
var
  G: TGgufFile;
  M: TQwenModel;
  X: TTensor;
begin
  M := LoadCpu(TiedPath, G);
  try
    X := M.Embed(Ids8).detach.Copy.requires_grad_(True);
    M.Logits(M.Trunk(X)).sum(kNone).Backward;
    AssertEquals('[8, 64]', ShapeText(X.grad));
    AssertTrue('grad finite', X.grad.isfinite.all.DoubleValue([]) = 1.0);
  finally
    M.Free;
    G.Free;
  end;
end;

{ UT-QwenModel-110: LastLogProbs with a single token. }
procedure TQwenModelTest.TestUT_QwenModel_110;
var
  G: TGgufFile;
  M: TQwenModel;
  LPt: TTensor;
  R: TDoubleArray;
  D: Double;
begin
  M := LoadCpu(TiedPath, G);
  try
    LPt := M.LastLogProbs([5]);
    R := RefLogits(G, RefDims(G), [5]);
    D := DiffArr(Arr(LPt), Lp(R, 0, 274));
    AssertTrue(Format('diff %.3e', [D]), D <= 1e-4);
  finally
    M.Free;
    G.Free;
  end;
end;

{ stage 3 (T12): AttachLora }

{ the pairs' names, comma-separated }
function PairNames(S: TLoraSet): string;
var
  I: Integer;
begin
  Result := '';
  for I := 0 to S.Count - 1 do
  begin
    if I > 0 then
      Result := Result + ',';
    Result := Result + S.Pair(I).Name;
  end;
end;

{ every B := 0.1 N(0, 1), after ManualSeed(6) }
procedure RandomB(S: TLoraSet);
var
  G: IInterface;
  I: Integer;
begin
  ManualSeed(6);
  G := NoGradGuard;
  for I := 0 to S.Count - 1 do
    S.Pair(I).B.CopyFrom(TTensor.randn([S.Pair(I).OutDim, S.Rank]) * 0.1);
end;

function NoGradLogits(M: TQwenModel): TTensor;
var
  G: IInterface;
begin
  G := NoGradGuard;
  Result := M.Logits(M.Trunk(M.Embed(Ids8)));
end;

const
  TinyPairs = 'blk.0.attn_qkv,blk.0.attn_gate,blk.0.ssm_out,blk.1.attn_q,blk.1.attn_k,blk.1.attn_v,blk.1.attn_output';

{ UT-QwenModel-207: AttachLora, split 1 on one device. }
procedure TQwenModelTest.TestUT_QwenModel_207;
var
  G: TGgufFile;
  M: TQwenModel;
  S: TLoraSet;
begin
  G := TGgufFile.Open(TiedPath);
  M := TQwenModel.Load(G, TDevice.Cpu, TDevice.Cpu, 1);
  S := TLoraSet.Create(4, 2.0);
  try
    AssertEquals(7, M.AttachLora(S));
    AssertEquals(1, Length(S.Stores));
    AssertEquals(14, S.Stores[0].Count);
  finally
    M.Free;
    S.Free;
    G.Free;
  end;
end;

{ UT-QwenModel-206: AttachLora, NextN file (no MTP block). }
procedure TQwenModelTest.TestUT_QwenModel_206;
var
  G: TGgufFile;
  M: TQwenModel;
  S: TLoraSet;
begin
  M := LoadCpu(NextPath, G);
  S := TLoraSet.Create(4, 2.0);
  try
    AssertEquals(7, M.AttachLora(S));
    AssertEquals(TinyPairs, PairNames(S));
  finally
    M.Free;
    S.Free;
    G.Free;
  end;
end;

{ UT-QwenModel-205: AttachLora twice on the same set. }
procedure TQwenModelTest.TestUT_QwenModel_205;
var
  G: TGgufFile;
  M: TQwenModel;
  S: TLoraSet;
begin
  M := LoadCpu(TiedPath, G);
  S := TLoraSet.Create(4, 2.0);
  try
    M.AttachLora(S);
    try
      M.AttachLora(S);
      Fail('no ELora');
    except
      on E: ELora do
        AssertEquals('pair already present: blk.0.attn_qkv', E.Message);
    end;
    AssertEquals(7, S.Count);
  finally
    M.Free;
    S.Free;
    G.Free;
  end;
end;

{ UT-QwenModel-204: AttachLora, block-wise grads vs the whole graph. }
procedure TQwenModelTest.TestUT_QwenModel_204;
var
  G: TGgufFile;
  M: TQwenModel;
  S: TLoraSet;
  H: TTensor;
  Saved: TSavedInputs;
  GA, GB: array of TDoubleArray;
  V: TDoubleArray;
  D, BMax: Double;
  I, J: Integer;
begin
  M := LoadCpu(TiedPath, G);
  S := TLoraSet.Create(4, 2.0);
  Saved := TSavedInputs.Create(True);
  try
    M.AttachLora(S);
    RandomB(S);
    M.Logits(M.Trunk(M.Embed(Ids8))).log_softmax(-1).sum(kNone).Backward;
    SetLength(GA, S.Count);
    SetLength(GB, S.Count);
    for I := 0 to S.Count - 1 do
    begin
      GA[I] := Arr(S.Pair(I).A.grad);
      GB[I] := Arr(S.Pair(I).B.grad);
      S.Pair(I).A.ZeroGrad;
      S.Pair(I).B.ZeroGrad;
    end;
    H := RecomputeForward(M.Stages, M.Embed(Ids8), Saved).detach.requires_grad_(True);
    M.Logits(H).log_softmax(-1).sum(kNone).Backward;
    RecomputeBackward(M.Stages, Saved, H.grad);
    for I := 0 to S.Count - 1 do
    begin
      V := Arr(S.Pair(I).B.grad);
      BMax := 0;
      for J := 0 to High(V) do
        if Abs(V[J]) > BMax then BMax := Abs(V[J]);
      AssertTrue(Format('pair %d: B.grad max |.| = %.3e > 0', [I, BMax]), BMax > 0);
      D := DiffArr(Arr(S.Pair(I).A.grad), GA[I]);
      AssertTrue(Format('pair %d: A.grad diff %.3e', [I, D]), D <= 1.0e-6);
      D := DiffArr(Arr(S.Pair(I).B.grad), GB[I]);
      AssertTrue(Format('pair %d: B.grad diff %.3e', [I, D]), D <= 1.0e-6);
    end;
  finally
    M.Free;
    S.Free;
    Saved.Free;
    G.Free;
  end;
end;

{ UT-QwenModel-203: AttachLora, random B changes the output. }
procedure TQwenModelTest.TestUT_QwenModel_203;
var
  G: TGgufFile;
  M, M0: TQwenModel;
  S: TLoraSet;
  H: TTensor;
  Saved: TSavedInputs;
begin
  M := LoadCpu(TiedPath, G);
  M0 := TQwenModel.Load(G, TDevice.Cpu, TDevice.Cpu, 0);
  S := TLoraSet.Create(4, 2.0);
  Saved := TSavedInputs.Create(True);
  try
    M.AttachLora(S);
    RandomB(S);
    H := RecomputeForward(M.Stages, M.Embed(Ids8), Saved);
    AssertTrue('random B moves the logits by more than 0.1',
      DiffArr(Arr(NoGradLogits(M)), Arr(NoGradLogits(M0))) > 0.1);
    AssertTrue('RecomputeForward equals Trunk (exact)',
      DiffArr(Arr(H), Arr(M.Trunk(M.Embed(Ids8)))) = 0);
  finally
    M.Free;
    M0.Free;
    S.Free;
    Saved.Free;
    G.Free;
  end;
end;

{ UT-QwenModel-202: AttachLora, output unchanged with B zero. }
procedure TQwenModelTest.TestUT_QwenModel_202;
var
  G: TGgufFile;
  M, M0: TQwenModel;
  S: TLoraSet;
begin
  M := LoadCpu(TiedPath, G);
  M0 := TQwenModel.Load(G, TDevice.Cpu, TDevice.Cpu, 0);
  S := TLoraSet.Create(4, 2.0);
  try
    M.AttachLora(S);
    AssertTrue('zero B leaves the logits unchanged (exact)',
      DiffArr(Arr(NoGradLogits(M)), Arr(NoGradLogits(M0))) = 0);
  finally
    M.Free;
    M0.Free;
    S.Free;
    G.Free;
  end;
end;

{ UT-QwenModel-201: AttachLora, counts, names and attachment. }
procedure TQwenModelTest.TestUT_QwenModel_201;
var
  G: TGgufFile;
  M: TQwenModel;
  S: TLoraSet;
  B0, B1: TQwenBlock;
begin
  M := LoadCpu(TiedPath, G);
  S := TLoraSet.Create(4, 2.0);
  try
    AssertEquals(7, M.AttachLora(S));
    AssertEquals(7, S.Count);
    AssertEquals(TinyPairs, PairNames(S));
    AssertEquals(4864, S.ParameterCount);
    B0 := M.Stages[0] as TQwenBlock;
    B1 := M.Stages[1] as TQwenBlock;
    AssertTrue('blk.0 attn_qkv has pair 0', B0.Target('attn_qkv').Lora = S.Pair(0));
    AssertTrue('blk.0 ssm_out has pair 2', B0.Target('ssm_out').Lora = S.Pair(2));
    AssertTrue('blk.0 ssm_beta untouched', B0.Target('ssm_beta').Lora = nil);
    AssertTrue('blk.0 ffn_gate untouched', B0.Target('ffn_gate').Lora = nil);
    AssertTrue('blk.1 attn_output has pair 6', B1.Target('attn_output').Lora = S.Pair(6));
  finally
    M.Free;
    S.Free;
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
