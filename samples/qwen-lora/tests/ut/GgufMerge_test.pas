{ UT program for src/GgufMerge.pas: the rows UT-GgufMerge-NNN of
  tests/ut/cases/GgufMerge.cases.md. Runs on the CPU (.sdlc/ut-coverage.sh
  hides the GPUs) on the tiny model of tests/ut/QwenBlocksFixture.pas.
  Invoked with `--all --format=plain` (see .sdlc/ut-coverage.sh).

  Skeleton by the harness owner; the test methods are written in T18b. }
program GgufMerge_test;

{$mode objfpc}{$H+}

uses
  SysUtils, Classes, Math, Tch, GgufFile, GgufQuant, QwenModel, Lora, Checkpoint,
  GgufMerge, QwenBlocksFixture, fpcunit, testregistry, consoletestrunner;

type
  TGgufMergeTest = class(TTestCase)
  published
    procedure TestUT_GgufMerge_001;
    procedure TestUT_GgufMerge_002;
    procedure TestUT_GgufMerge_003;
    procedure TestUT_GgufMerge_004;
    procedure TestUT_GgufMerge_005;
    procedure TestUT_GgufMerge_006;
    procedure TestUT_GgufMerge_007;
    procedure TestUT_GgufMerge_008;
    procedure TestUT_GgufMerge_009;
    procedure TestUT_GgufMerge_010;
    procedure TestUT_GgufMerge_011;
    procedure TestUT_GgufMerge_012;
    procedure TestUT_GgufMerge_013;
    procedure TestUT_GgufMerge_014;
    procedure TestUT_GgufMerge_101;
    procedure TestUT_GgufMerge_102;
    procedure TestUT_GgufMerge_103;
    procedure TestUT_GgufMerge_104;
    procedure TestUT_GgufMerge_105;
    procedure TestUT_GgufMerge_106;
  end;

{ ---- helpers (harness owner's; nobody changes them) ---- }

var
  TinyPath, NextNPath, TmpDir: string;
  Fresh: Integer = 0;

{ A fresh directory under TmpDir (created), ending in the delimiter. }
function NewDir: string;
begin
  Inc(Fresh);
  Result := TmpDir + 'd' + IntToStr(Fresh) + PathDelim;
  ForceDirectories(Result);
end;

function FileBytes(const APath: string): TBytes;
var
  F: TFileStream;
begin
  Result := nil;
  F := TFileStream.Create(APath, fmOpenRead or fmShareDenyNone);
  try
    SetLength(Result, F.Size);
    if F.Size > 0 then
      F.ReadBuffer(Result[0], F.Size);
  finally
    F.Free;
  end;
end;

procedure SaveBytes(const APath: string; const B: TBytes);
var
  F: TFileStream;
begin
  F := TFileStream.Create(APath, fmCreate);
  try
    if Length(B) > 0 then
      F.WriteBuffer(B[0], Length(B));
  finally
    F.Free;
  end;
end;

function SameBytes(const A, B: TBytes): Boolean;
begin
  Result := (Length(A) = Length(B)) and
    ((Length(A) = 0) or CompareMem(@A[0], @B[0], Length(A)));
end;

{ A copy of ASrc at ADst with every occurrence of AOld[i] replaced by
  ANew[i] (same lengths), in the order given. }
procedure PatchedCopy(const ASrc, ADst: string; const AOld, ANew: array of RawByteString);
var
  B: TBytes;
  I, J, K: Integer;
  Hit: Boolean;
begin
  B := FileBytes(ASrc);
  for K := 0 to High(AOld) do
    for I := 0 to Length(B) - Length(AOld[K]) do
    begin
      Hit := True;
      for J := 1 to Length(AOld[K]) do
        if B[I + J - 1] <> Ord(AOld[K][J]) then
        begin
          Hit := False;
          Break;
        end;
      if Hit then
        for J := 1 to Length(ANew[K]) do
          B[I + J - 1] := Ord(ANew[K][J]);
    end;
  SaveBytes(ADst, B);
end;

{ A set of rank 4, scale 2 on the CPU with one pair per target (A
  initialised after ManualSeed(3)); every B := 0.1 N(0, 1) after
  ManualSeed(ASeed); ASeed < 0: B stays zero. }
function NewSet(const T: TMergeTargets; ASeed: Integer): TLoraSet;
var
  NG: IInterface;
  I: Integer;
begin
  ManualSeed(3);
  Result := TLoraSet.Create(4, 2.0);
  for I := 0 to High(T) do
    Result.Add(T[I].Name, T[I].InDim, T[I].OutDim, TDevice.Cpu);
  if ASeed >= 0 then
  begin
    NG := NoGradGuard;
    ManualSeed(ASeed);
    for I := 0 to Result.Count - 1 do
      Result.Pair(I).B.CopyFrom(TTensor.randn([Result.Pair(I).OutDim, Result.Pair(I).Rank]) * 0.1);
  end;
end;

{ ADir/lora.ot from S and ADir/state.json: update 8, rank and scale of S,
  BaseSha256 ASha, the rest NewRunState. }
procedure SaveCkpt(const ADir: string; S: TLoraSet; const ASha: string);
var
  St: TRunState;
  L: TStringList;
begin
  ForceDirectories(ADir);
  S.Save(IncludeTrailingPathDelimiter(ADir) + 'lora.ot');
  St := NewRunState;
  St.Update := 8;
  St.Rank := S.Rank;
  St.Scale := S.Scale;
  St.BaseSha256 := ASha;
  L := TStringList.Create;
  try
    L.Text := RunStateToJson(St);
    L.SaveToFile(IncludeTrailingPathDelimiter(ADir) + 'state.json');
  finally
    L.Free;
  end;
end;

{ The scalar reference of a merge in Double: W (dequantised) +
  scale · Σ_r B[o, r] · A[r, i], row-major [AOut, AIn]. }
function RefMerged(ATType: Cardinal; const ABytes: TBytes; AIn, AOut: Int64;
  P: TLoraPair): TDoubleArray;
var
  W, A, B: TSingleArray;
  O, I, R: Int64;
  S: Double;
begin
  Result := nil;
  W := DequantizeBytes(ATType, ABytes, AIn * AOut);
  A := P.A.ToSingleArray;
  B := P.B.ToSingleArray;
  SetLength(Result, AIn * AOut);
  for O := 0 to AOut - 1 do
    for I := 0 to AIn - 1 do
    begin
      S := 0;
      for R := 0 to P.Rank - 1 do
        S := S + Double(B[O * P.Rank + R]) * A[R * AIn + I];
      Result[O * AIn + I] := W[O * AIn + I] + P.Scale * S;
    end;
end;

{ Max over the elements of |dequantised Q8_0 - ARef| / the block's d (fp16). }
function StepErr(const AQ8: TBytes; const ARef: TDoubleArray): Double;
var
  Q: TSingleArray;
  J, K: Integer;
  D: Double;
begin
  Result := 0;
  Q := DequantizeBytes(GgufTypeQ8_0, AQ8, Length(ARef));
  for J := 0 to Length(ARef) div 32 - 1 do
  begin
    D := Fp16ToSingle(PWord(@AQ8[J * 34])^);
    for K := J * 32 to J * 32 + 31 do
      if D > 0 then
        Result := Max(Result, Abs(Q[K] - ARef[K]) / D)
      else if ARef[K] <> 0 then
        Result := Infinity;
  end;
end;

{ The bytes of tensor AName of AGguf. }
function TensorBytes(AGguf: TGgufFile; const AName: RawByteString): TBytes;
begin
  AGguf.ReadTensorBytes(AGguf.FindTensor(AName), Result);
end;

{ The mean |Δ logprob| over the top 10 of AA's last position, AB against
  AA, for tokens (7 i + 3) mod 270, i = 0..19. }
function TopDiff(AA, AB: TQwenModel): Double;
var
  NG: IInterface;
  T: array of Integer;
  I: Integer;
  LA, LB: TTensor;
  Idx: TInt64Array;
begin
  NG := NoGradGuard;
  T := nil;
  SetLength(T, 20);
  for I := 0 to 19 do
    T[I] := (7 * I + 3) mod 270;
  LA := AA.Logits(AA.Trunk(AA.Embed(T))).select(0, 19).log_softmax(-1);
  LB := AB.Logits(AB.Trunk(AB.Embed(T))).select(0, 19).log_softmax(-1);
  Idx := LA.argsort(0, True).narrow(0, 0, 10).ToInt64Array;
  Result := 0;
  for I := 0 to 9 do
    Result := Result + Abs(LA.select(0, Idx[I]).DoubleValue([]) - LB.select(0, Idx[I]).DoubleValue([]));
  Result := Result / 10;
end;

{ RunCli: RunMergeCLI; AOut / AErr get the lines joined by #10; the exit code. }
function RunCli(const AArgs: array of string; out AOut, AErr: string): Integer;
var
  O, E: TStringList;
begin
  O := TStringList.Create;
  E := TStringList.Create;
  try
    Result := RunMergeCLI(AArgs, O, E);
    AOut := string.Join(#10, O.ToStringArray);
    AErr := string.Join(#10, E.ToStringArray);
  finally
    O.Free;
    E.Free;
  end;
end;

{ UT-GgufMerge-001 — TinyPath, the 7 targets. }
procedure TGgufMergeTest.TestUT_GgufMerge_001;
const
  InD: array[0..6] of Int64 = (64, 64, 64, 64, 64, 64, 128);
  OutD: array[0..6] of Int64 = (128, 64, 64, 256, 64, 64, 64);
var
  G: TGgufFile;
  T: TMergeTargets;
  I: Integer;
begin
  G := TGgufFile.Open(TinyPath);
  try
    T := MergeTargets(G);
    AssertEquals(7, Length(T));
    AssertEquals('blk.0.attn_qkv', T[0].Name);
    AssertEquals('blk.0.attn_gate', T[1].Name);
    AssertEquals('blk.0.ssm_out', T[2].Name);
    AssertEquals('blk.1.attn_q', T[3].Name);
    AssertEquals('blk.1.attn_k', T[4].Name);
    AssertEquals('blk.1.attn_v', T[5].Name);
    AssertEquals('blk.1.attn_output', T[6].Name);
    for I := 0 to 6 do
    begin
      AssertEquals(G.FindTensor(T[I].Name + '.weight'), T[I].Tensor);
      AssertEquals(InD[I], T[I].InDim);
      AssertEquals(OutD[I], T[I].OutDim);
    end;
  finally
    G.Free;
  end;
end;

{ UT-GgufMerge-002 — NextNPath, one MTP block. }
procedure TGgufMergeTest.TestUT_GgufMerge_002;
var
  G: TGgufFile;
  T: TMergeTargets;
begin
  G := TGgufFile.Open(NextNPath);
  try
    T := MergeTargets(G);
    AssertEquals(7, Length(T));
    AssertEquals('blk.0.attn_qkv', T[0].Name);
    AssertEquals('blk.1.attn_output', T[6].Name);
  finally
    G.Free;
  end;
end;

{ UT-GgufMerge-003 — patched copy: missing tensor, 1-dim tensor. }
procedure TGgufMergeTest.TestUT_GgufMerge_003;
var
  D: string;
  G: TGgufFile;
begin
  D := NewDir;
  PatchedCopy(TinyPath, D + 'pk.gguf',
    ['blk.1.attn_k.weight'], ['blk.1.attn_X.weight']);
  G := TGgufFile.Open(D + 'pk.gguf');
  try
    try
      MergeTargets(G);
      Fail('no EGgufMerge');
    except
      on E: EGgufMerge do
        AssertEquals('no tensor blk.1.attn_k.weight', E.Message);
    end;
  finally
    G.Free;
  end;
  PatchedCopy(TinyPath, D + 'p1d.gguf',
    ['blk.0.attn_gate.weight', 'blk.0.attn_norm.weight'],
    ['blk.0.attn_XXXX.weight', 'blk.0.attn_gate.weight']);
  G := TGgufFile.Open(D + 'p1d.gguf');
  try
    try
      MergeTargets(G);
      Fail('no EGgufMerge');
    except
      on E: EGgufMerge do
        AssertEquals('blk.0.attn_gate.weight has 1 dimensions, expected 2', E.Message);
    end;
  finally
    G.Free;
  end;
end;

{ UT-GgufMerge-004 — Q8_0 merge of blk.0.attn_qkv. }
procedure TGgufMergeTest.TestUT_GgufMerge_004;
var
  G: TGgufFile;
  T: TMergeTargets;
  S: TLoraSet;
  P: TLoraPair;
  B, Q: TBytes;
  R: TDoubleArray;
begin
  G := TGgufFile.Open(TinyPath);
  try
    T := MergeTargets(G);
    S := NewSet(T, 6);
    try
      AssertEquals(8, G.Tensor[T[0].Tensor].TType);
      P := S.Find('blk.0.attn_qkv');
      AssertTrue(P <> nil);
      B := TensorBytes(G, 'blk.0.attn_qkv.weight');
      Q := MergeTensor(8, B, 64, 128, P);
      AssertEquals(8704, Length(Q));
      R := RefMerged(8, B, 64, 128, P);
      AssertTrue(StepErr(Q, R) <= 0.6);
      AssertFalse(SameBytes(Q, B));
    finally
      S.Free;
    end;
  finally
    G.Free;
  end;
end;

{ UT-GgufMerge-005 — F16 merge of blk.0.attn_gate. }
procedure TGgufMergeTest.TestUT_GgufMerge_005;
var
  G: TGgufFile;
  T: TMergeTargets;
  S: TLoraSet;
  P: TLoraPair;
  B, Q: TBytes;
  R: TDoubleArray;
begin
  G := TGgufFile.Open(TinyPath);
  try
    T := MergeTargets(G);
    S := NewSet(T, 6);
    try
      AssertEquals(1, G.Tensor[T[1].Tensor].TType);
      P := S.Find('blk.0.attn_gate');
      AssertTrue(P <> nil);
      B := TensorBytes(G, 'blk.0.attn_gate.weight');
      Q := MergeTensor(1, B, 64, 64, P);
      AssertEquals(4352, Length(Q));
      R := RefMerged(1, B, 64, 64, P);
      AssertTrue(StepErr(Q, R) <= 0.6);
    finally
      S.Free;
    end;
  finally
    G.Free;
  end;
end;

{ UT-GgufMerge-006 — all-delta B of zero: the bytes are unchanged / round trip. }
procedure TGgufMergeTest.TestUT_GgufMerge_006;
var
  G: TGgufFile;
  T: TMergeTargets;
  S: TLoraSet;
  P: TLoraPair;
  BQ, QQ, BF, QF: TBytes;
  QRef: TBytes;
begin
  G := TGgufFile.Open(TinyPath);
  try
    T := MergeTargets(G);
    S := NewSet(T, -1);
    try
      P := S.Find('blk.0.attn_qkv');
      AssertTrue(P <> nil);
      BQ := TensorBytes(G, 'blk.0.attn_qkv.weight');
      QQ := MergeTensor(8, BQ, 64, 128, P);
      AssertTrue(SameBytes(QQ, BQ));
      P := S.Find('blk.0.attn_gate');
      AssertTrue(P <> nil);
      BF := TensorBytes(G, 'blk.0.attn_gate.weight');
      QF := MergeTensor(1, BF, 64, 64, P);
      QRef := QuantizeQ8_0(DequantizeBytes(1, BF, 4096));
      AssertTrue(SameBytes(QF, QRef));
    finally
      S.Free;
    end;
  finally
    G.Free;
  end;
end;

{ UT-GgufMerge-007 — adapter dimensions of another tensor. }
procedure TGgufMergeTest.TestUT_GgufMerge_007;
var
  G: TGgufFile;
  T: TMergeTargets;
  S: TLoraSet;
  P: TLoraPair;
  B: TBytes;
begin
  G := TGgufFile.Open(TinyPath);
  try
    T := MergeTargets(G);
    S := NewSet(T, 6);
    try
      P := S.Find('blk.0.attn_gate');
      AssertTrue(P <> nil);
      B := TensorBytes(G, 'blk.0.attn_qkv.weight');
      try
        MergeTensor(8, B, 64, 128, P);
        Fail('no EGgufMerge');
      except
        on E: EGgufMerge do
          AssertEquals('blk.0.attn_gate: adapter 64 -> 64, tensor 64 -> 128', E.Message);
      end;
    finally
      S.Free;
    end;
  finally
    G.Free;
  end;
end;

{ UT-GgufMerge-008 — load from the written checkpoint. }
procedure TGgufMergeTest.TestUT_GgufMerge_008;
var
  D: string;
  G: TGgufFile;
  T: TMergeTargets;
  S, L: TLoraSet;
  S0: TLoraSet;
  St: TRunState;
  I, IA: Integer;
  AA, BB: TSingleArray;
  M: Double;
begin
  D := NewDir;
  G := TGgufFile.Open(TinyPath);
  try
    T := MergeTargets(G);
    S := NewSet(T, 6);
    try
      SaveCkpt(D, S, '');
      St := ReadRunState(D);
      L := LoadAdapters(T, D, St);
      try
        AssertEquals(7, L.Count);
        AssertEquals(4, L.Rank);
        AssertEquals(2.0, L.Scale);
        S0 := NewSet(T, 6);
        try
          for I := 0 to L.Count - 1 do
          begin
            AssertEquals(T[I].Name, L.Pair(I).Name);
            AA := L.Pair(I).A.ToSingleArray;
            BB := S0.Pair(I).A.ToSingleArray;
            M := 0.0;
            for IA := 0 to High(AA) do
              M := Max(M, Abs(AA[IA] - BB[IA]));
            AssertEquals(0.0, M);
            AA := L.Pair(I).B.ToSingleArray;
            BB := S0.Pair(I).B.ToSingleArray;
            M := 0.0;
            for IA := 0 to High(AA) do
              M := Max(M, Abs(AA[IA] - BB[IA]));
            AssertEquals(0.0, M);
          end;
        finally
          S0.Free;
        end;
      finally
        L.Free;
      end;
    finally
      S.Free;
    end;
  finally
    G.Free;
  end;
end;

{ UT-GgufMerge-009 — lora.ot does not exist. }
procedure TGgufMergeTest.TestUT_GgufMerge_009;
var
  D: string;
  G: TGgufFile;
  T: TMergeTargets;
  S: TLoraSet;
  St: TRunState;
begin
  D := NewDir;
  G := TGgufFile.Open(TinyPath);
  try
    T := MergeTargets(G);
    S := NewSet(T, 6);
    try
      SaveCkpt(D, S, '');
      DeleteFile(D + 'lora.ot');
      St := ReadRunState(D);
      try
        LoadAdapters(T, D, St);
        Fail('no ETorchNativeError');
      except
        on E: ETorchNativeError do
          AssertEquals('open file failed because of errno 2 on fopen: No such file or directory, file path: ' + D + 'lora.ot', E.Message);
      end;
    finally
      S.Free;
    end;
  finally
    G.Free;
  end;
end;

{ UT-GgufMerge-010 — ASet is nil: a byte-identical copy. }
procedure TGgufMergeTest.TestUT_GgufMerge_010;
var
  D: string;
  G: TGgufFile;
  T: TMergeTargets;
  N: Integer;
  A, B: TBytes;
begin
  D := NewDir;
  G := TGgufFile.Open(TinyPath);
  try
    T := MergeTargets(G);
    N := WriteMerged(G, T, nil, D + 'id.gguf');
    AssertEquals(0, N);
    AssertFalse(FileExists(D + 'id.gguf.part'));
    A := FileBytes(D + 'id.gguf');
    B := FileBytes(TinyPath);
    AssertTrue(SameBytes(A, B));
  finally
    G.Free;
  end;
end;

{ UT-GgufMerge-011 — all 7 targets are merged. }
procedure TGgufMergeTest.TestUT_GgufMerge_011;
var
  D: string;
  G, H: TGgufFile;
  T: TMergeTargets;
  S: TLoraSet;
  N: Integer;
  I, J: Integer;
  Name: RawByteString;
  IsTarget: Boolean;
  B, E: TBytes;
begin
  D := NewDir;
  G := TGgufFile.Open(TinyPath);
  try
    T := MergeTargets(G);
    S := NewSet(T, 6);
    try
      N := WriteMerged(G, T, S, D + 'm.gguf');
      AssertEquals(7, N);
      AssertFalse(FileExists(D + 'm.gguf.part'));
      H := TGgufFile.Open(D + 'm.gguf');
      try
        AssertEquals(G.TensorCount, H.TensorCount);
        AssertEquals(G.KVCount, H.KVCount);
        AssertEquals(G.KVStart, H.KVStart);
        AssertEquals(G.KVEnd, H.KVEnd);
        AssertTrue(CompareMem(@FileBytes(G.Path)[Cardinal(G.KVStart)],
          @FileBytes(D + 'm.gguf')[Cardinal(H.KVStart)],
          Cardinal(G.KVEnd - G.KVStart)));
        for I := 0 to G.TensorCount - 1 do
        begin
          Name := G.Tensor[I].Name;
          AssertEquals(Name, H.Tensor[I].Name);
          IsTarget := False;
          for J := 0 to High(T) do
            if T[J].Tensor = I then
              IsTarget := True;
          if IsTarget then
          begin
            AssertEquals(8, H.Tensor[I].TType);
            E := MergeTensor(G.Tensor[I].TType, TensorBytes(G, Name),
              G.Tensor[I].Dims[0], G.Tensor[I].Dims[1],
              S.Find(Copy(Name, 1, Length(Name) - 7)));
            B := TensorBytes(H, Name);
            AssertTrue(SameBytes(B, E));
          end
          else
          begin
            AssertEquals(G.Tensor[I].TType, H.Tensor[I].TType);
            AssertTrue(SameBytes(TensorBytes(H, Name), TensorBytes(G, Name)));
          end;
        end;
      finally
        H.Free;
      end;
    finally
      S.Free;
    end;
  finally
    G.Free;
  end;
end;

{ UT-GgufMerge-012 — a set that does not contain blk.0.attn_gate. }
procedure TGgufMergeTest.TestUT_GgufMerge_012;
var
  D: string;
  G: TGgufFile;
  T: TMergeTargets;
  P: TLoraSet;
begin
  D := NewDir;
  G := TGgufFile.Open(TinyPath);
  try
    T := MergeTargets(G);
    P := TLoraSet.Create(4, 2.0);
    try
      P.Add('blk.0.attn_qkv', 64, 128, TDevice.Cpu);
      try
        WriteMerged(G, T, P, D + 'm.gguf');
        Fail('no EGgufMerge');
      except
        on E: EGgufMerge do
          AssertEquals('no adapter blk.0.attn_gate', E.Message);
      end;
      AssertFalse(FileExists(D + 'm.gguf'));
      AssertFalse(FileExists(D + 'm.gguf.part'));
    finally
      P.Free;
    end;
  finally
    G.Free;
  end;
end;

{ UT-GgufMerge-013 — fail before write; rename fails; overwrite on rename. }
procedure TGgufMergeTest.TestUT_GgufMerge_013;
const
  MB: array[0..2] of Byte = (1, 2, 3);
  NB: array[0..2] of Byte = (4, 5, 6);
var
  D: string;
  G: TGgufFile;
  T: TMergeTargets;
  S: TLoraSet;
  N: Integer;
  A, B: TBytes;
begin
  D := NewDir;
  G := TGgufFile.Open(TinyPath);
  try
    T := MergeTargets(G);
    S := NewSet(T, 6);
    try
      try
        WriteMerged(G, T, S, D + 'none/m.gguf');
        Fail('no EGgufError');
      except
        on E: EGgufError do
          AssertEquals('TGgufWriter.BeginWrite: cannot create file ' + D + 'none/m.gguf.part', E.Message);
      end;
      ForceDirectories(D + 'dir');
      try
        WriteMerged(G, T, S, D + 'dir');
        Fail('no EGgufMerge');
      except
        on E: EGgufMerge do
        begin
          AssertEquals('cannot rename ' + D + 'dir.part to ' + D + 'dir', E.Message);
          AssertFalse(FileExists(D + 'dir.part'));
        end;
      end;
      SaveBytes(D + 'm.gguf', MB);
      N := WriteMerged(G, T, S, D + 'm.gguf');
      AssertEquals(7, N);
      SaveBytes(D + 'n.gguf', NB);
      N := WriteMerged(G, T, S, D + 'n.gguf');
      AssertEquals(7, N);
      A := FileBytes(D + 'm.gguf');
      B := FileBytes(D + 'n.gguf');
      AssertTrue(SameBytes(A, B));
    finally
      S.Free;
    end;
  finally
    G.Free;
  end;
end;

{ UT-GgufMerge-014 — effect and fidelity on the tiny model. }
procedure TGgufMergeTest.TestUT_GgufMerge_014;
var
  D: string;
  G, G0, GMA, GMM: TGgufFile;
  T: TMergeTargets;
  S, SA: TLoraSet;
  St: TRunState;
  M0, MA, MM: TQwenModel;
  Effect, Fid: Double;
begin
  D := NewDir;
  G0 := TGgufFile.Open(TinyPath);
  M0 := TQwenModel.Load(G0, TDevice.Cpu, TDevice.Cpu, 0);
  G0.Free;
  G := TGgufFile.Open(TinyPath);
  T := MergeTargets(G);
  S := NewSet(T, 6);
  try
    SaveCkpt(D, S, '');
    WriteMerged(G, T, S, D + 'mm.gguf');
    GMM := TGgufFile.Open(D + 'mm.gguf');
    MM := TQwenModel.Load(GMM, TDevice.Cpu, TDevice.Cpu, 0);
    GMM.Free;
    GMA := TGgufFile.Open(TinyPath);
    MA := TQwenModel.Load(GMA, TDevice.Cpu, TDevice.Cpu, 0);
    GMA.Free;
    St := ReadRunState(D);
    SA := TLoraSet.Create(St.Rank, St.Scale);
    MA.AttachLora(SA);
    SA.Load(IncludeTrailingPathDelimiter(D) + 'lora.ot');
    Effect := TopDiff(M0, MA);
    AssertTrue(Effect > 0.5);
    Fid := TopDiff(MA, MM);
    AssertTrue(Fid <= 0.1 * Effect);
    MA.Free;
    SA.Free;
  finally
    S.Free;
    G.Free;
    MM.Free;
    M0.Free;
  end;
end;

{ UT-GgufMerge-101 — success: BASE CKPT OUT. }
procedure TGgufMergeTest.TestUT_GgufMerge_101;
var
  D: string;
  G: TGgufFile;
  T: TMergeTargets;
  S: TLoraSet;
  N: Integer;
  O, E: string;
  A, B: TBytes;
begin
  D := NewDir;
  G := TGgufFile.Open(TinyPath);
  T := MergeTargets(G);
  S := NewSet(T, 6);
  try
    N := WriteMerged(G, T, S, D + 'ref.gguf');
    AssertEquals(7, N);
    SaveCkpt(D + 'c', S, '');
    AssertEquals(0, RunCli([TinyPath, D + 'c', D + 'm.gguf'], O, E));
    AssertEquals('merged 7 tensors (rank 4, scale 2) into ' + D + 'm.gguf', O);
    AssertEquals('', E);
    A := FileBytes(D + 'm.gguf');
    B := FileBytes(D + 'ref.gguf');
    AssertTrue(SameBytes(A, B));
  finally
    S.Free;
    G.Free;
  end;
end;

{ UT-GgufMerge-102 — usage errors, exit 2. }
procedure TGgufMergeTest.TestUT_GgufMerge_102;
var
  A: array of string;
  O, E: string;
begin
  A := nil;
  AssertEquals(2, RunCli(A, O, E));
  AssertEquals('expected BASE CKPT OUT, got 0 arguments', E);
  AssertEquals('', O);
  AssertEquals(2, RunCli(['a', 'b'], O, E));
  AssertEquals('expected BASE CKPT OUT, got 2 arguments', E);
  AssertEquals('', O);
  AssertEquals(2, RunCli(['a', 'b', 'c', 'd'], O, E));
  AssertEquals('expected BASE CKPT OUT, got 4 arguments', E);
  AssertEquals('', O);
  AssertEquals(2, RunCli(['--frob', 'a', 'b', 'c'], O, E));
  AssertEquals('unknown option --frob', E);
  AssertEquals('', O);
  AssertEquals(2, RunCli(['a', 'b', 'c', '--base-sha256'], O, E));
  AssertEquals('--base-sha256: value missing', E);
  AssertEquals('', O);
  AssertEquals(2, RunCli(['--BASE-SHA256', 'x', 'a', 'b', 'c'], O, E));
  AssertEquals('unknown option --BASE-SHA256', E);
  AssertEquals('', O);
end;

{ UT-GgufMerge-103 — no base; output is the base. }
procedure TGgufMergeTest.TestUT_GgufMerge_103;
var
  D: string;
  O, E: string;
  Cwd: string;
begin
  D := NewDir;
  AssertEquals(1, RunCli([TmpDir + 'none.gguf', D + 'c', D + 'm.gguf'], O, E));
  AssertEquals('no model file ' + TmpDir + 'none.gguf', E);
  AssertEquals('', O);
  AssertFalse(FileExists(D + 'm.gguf'));
  AssertEquals(1, RunCli([TinyPath, D + 'c', TinyPath], O, E));
  AssertEquals('the output must not be the base model', E);
  AssertFalse(FileExists(D + 'm.gguf'));
  AssertEquals(1, RunCli([TinyPath, D + 'c', D + '../tiny.gguf'], O, E));
  AssertEquals('the output must not be the base model', E);
  AssertFalse(FileExists(D + 'm.gguf'));
  Cwd := GetCurrentDir;
  SetCurrentDir(TmpDir);
  try
    AssertEquals(1, RunCli([TinyPath, D + 'c', 'tiny.gguf'], O, E));
    AssertEquals('the output must not be the base model', E);
    AssertEquals('', O);
  finally
    SetCurrentDir(Cwd);
  end;
  AssertFalse(FileExists(D + 'm.gguf'));
end;

{ UT-GgufMerge-104 — an empty directory as the checkpoint. }
procedure TGgufMergeTest.TestUT_GgufMerge_104;
var
  D: string;
  O, E: string;
begin
  D := NewDir;
  AssertEquals(1, RunCli([TinyPath, D, D + 'm.gguf'], O, E));
  AssertEquals('not a checkpoint: ' + D, E);
  AssertEquals('', O);
  AssertFalse(FileExists(D + 'm.gguf'));
end;

{ UT-GgufMerge-105 — the base''s sha256 against the checkpoint''s. }
procedure TGgufMergeTest.TestUT_GgufMerge_105;
var
  D: string;
  G: TGgufFile;
  T: TMergeTargets;
  S: TLoraSet;
  O, E: string;
  Ha, Hb, Hu: string;
begin
  D := NewDir;
  Ha := StringOfChar('a', 64);
  Hb := StringOfChar('b', 64);
  Hu := StringOfChar('A', 64);
  G := TGgufFile.Open(TinyPath);
  T := MergeTargets(G);
  S := NewSet(T, 6);
  try
    SaveCkpt(D + 'c', S, Ha);
    SaveCkpt(D + 'e', S, '');
    SaveBytes(D + 'sha256-' + Ha, FileBytes(TinyPath));
    SaveBytes(D + 'sha256-' + Hb, FileBytes(TinyPath));
    AssertEquals(1, RunCli(['--base-sha256', Hb, D + 'sha256-' + Ha, D + 'c', D + 'o1.gguf'], O, E));
    AssertEquals('the base model differs from the checkpoint''s: ' + D + 'c', E);
    AssertFalse(FileExists(D + 'o1.gguf'));
    AssertEquals(1, RunCli(['--base-sha256', Hu, D + 'sha256-' + Ha, D + 'c', D + 'o2.gguf'], O, E));
    AssertEquals('the base model differs from the checkpoint''s: ' + D + 'c', E);
    AssertFalse(FileExists(D + 'o2.gguf'));
    AssertEquals(1, RunCli([D + 'sha256-' + Hb, D + 'c', D + 'o3.gguf'], O, E));
    AssertEquals('the base model differs from the checkpoint''s: ' + D + 'c', E);
    AssertFalse(FileExists(D + 'o3.gguf'));
    AssertEquals(0, RunCli(['--base-sha256', Ha, D + 'sha256-' + Ha, D + 'c', D + 'o4.gguf'], O, E));
    AssertEquals('', E);
    AssertEquals(0, RunCli([D + 'sha256-' + Ha, D + 'c', D + 'o5.gguf'], O, E));
    AssertEquals('', E);
    AssertEquals(0, RunCli([TinyPath, D + 'c', D + 'o6.gguf'], O, E));
    AssertEquals('', E);
    AssertEquals(0, RunCli(['--base-sha256', Hb, TinyPath, D + 'e', D + 'o7.gguf'], O, E));
    AssertEquals('', E);
  finally
    S.Free;
    G.Free;
  end;
end;

{ UT-GgufMerge-106 — lora.ot missing at run time. }
procedure TGgufMergeTest.TestUT_GgufMerge_106;
var
  D: string;
  G: TGgufFile;
  T: TMergeTargets;
  S: TLoraSet;
  O, E: string;
begin
  D := NewDir;
  G := TGgufFile.Open(TinyPath);
  T := MergeTargets(G);
  S := NewSet(T, 6);
  try
    SaveCkpt(D + 'c', S, '');
    DeleteFile(D + 'c/lora.ot');
    AssertEquals(1, RunCli([TinyPath, D + 'c', D + 'm.gguf'], O, E));
    AssertEquals('open file failed because of errno 2 on fopen: No such file or directory, file path: ' + D + 'c/lora.ot', E);
    AssertEquals('', O);
    AssertFalse(FileExists(D + 'm.gguf'));
  finally
    S.Free;
    G.Free;
  end;
end;

var
  App: TTestRunner;
begin
  TmpDir := GetTempDir + 'ggufmerge-ut-' + IntToStr(GetProcessID) + PathDelim;
  ForceDirectories(TmpDir);
  TinyPath := TmpDir + 'tiny.gguf';
  NextNPath := TmpDir + 'nextn.gguf';
  WriteTinyQwenModel(TinyPath, 5, thTied, False);
  WriteTinyQwenModel(NextNPath, 5, thTied, True);
  RegisterTest(TGgufMergeTest);
  App := TTestRunner.Create(nil);
  App.Initialize;
  App.Title := 'GgufMerge UT';
  App.Run;
  App.Free;
end.
