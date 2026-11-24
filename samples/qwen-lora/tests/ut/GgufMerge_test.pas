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
