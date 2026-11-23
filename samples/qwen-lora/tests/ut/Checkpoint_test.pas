{ UT program for src/Checkpoint.pas: the rows UT-Checkpoint-NNN of
  tests/ut/cases/Checkpoint.cases.md (stage 1 from 001: TRunState and
  state.json; stage 2 from 101: TCheckpoints). Runs on the CPU
  (.sdlc/ut-coverage.sh hides the GPUs). Invoked with `--all
  --format=plain` (see .sdlc/ut-coverage.sh).

  Skeleton by the harness owner; the test methods are written in T16b. }
program Checkpoint_test;

{$mode objfpc}{$H+}

uses
  SysUtils, Classes, Math, BaseUnix, Tch, TchNN, Lora, LoraOptim, Checkpoint,
  fpcunit, testregistry, consoletestrunner;

type
  TCheckpointTest = class(TTestCase)
  public
    { The state of the rows: Update 5, Epoch 1, Position 37, Seed 16,
      Rank 0, Scale 0, ConfigHash 'cfg-1', BaseSha256 'abc',
      ValLoss 1/3. }
    class function FixState: TRunState; static;
    { A run: ManualSeed(ASeed); the set (rank 4, scale 2) with the pairs
      blk.0.attn_qkv 8 -> 6 and blk.1.attn_q 8 -> 10 on the CPU; AdamW
      (DefaultAdamWConfig) over its stores. Free O, then S. }
    class procedure NewRun(ASeed: Integer; out S: TLoraSet; out O: TLoraAdamW); static;
    { Updates AFrom .. AFrom + ACount - 1 of a toy problem: loss = sum over
      the pairs of ((Delta(X) - Y)^2).sum, X [5, 8] = sin(1 + i),
      Y = cos(1 + i) of the pair's shape; Backward, ClipGlobalNorm(1),
      Step(ScheduleLR(0.01, U, 2, 10, 0.1)), ZeroGrad. }
    class procedure Train(S: TLoraSet; O: TLoraAdamW; AFrom, ACount: Integer); static;
    { Every A and B of S, then every moment of O, as Double. }
    class function Snapshot(S: TLoraSet; O: TLoraAdamW): TDoubleArray; static;
    { Max |a - b|; Infinity when the lengths differ. }
    class function DiffArr(const A, B: TDoubleArray): Double; static;
    { A fresh directory under TmpDir (not created), ending in the delimiter. }
    class function NewRoot: string; static;
    { The file names of the paths, joined by commas. }
    class function Names(const A: TStringArray): string; static;
  published
  end;

var
  TmpDir: string;
  RootNo: Integer = 0;

class function TCheckpointTest.FixState: TRunState;
begin
  Result := NewRunState;
  Result.Update := 5;
  Result.Epoch := 1;
  Result.Position := 37;
  Result.Seed := 16;
  Result.ConfigHash := 'cfg-1';
  Result.BaseSha256 := 'abc';
  Result.ValLoss := 1 / 3;
end;

class procedure TCheckpointTest.NewRun(ASeed: Integer; out S: TLoraSet; out O: TLoraAdamW);
begin
  ManualSeed(ASeed);
  S := TLoraSet.Create(4, 2.0);
  S.Add('blk.0.attn_qkv', 8, 6, TDevice.Cpu);
  S.Add('blk.1.attn_q', 8, 10, TDevice.Cpu);
  O := TLoraAdamW.Create(S.Stores, DefaultAdamWConfig);
end;

function SinT(ARows, ACols: Int64): TTensor;
begin
  Result := (TTensor.arange(ARows * ACols, kFloat, TDevice.Cpu) + 1).sin.view([ARows, ACols]);
end;

function CosT(ARows, ACols: Int64): TTensor;
begin
  Result := (TTensor.arange(ARows * ACols, kFloat, TDevice.Cpu) + 1).cos.view([ARows, ACols]);
end;

class procedure TCheckpointTest.Train(S: TLoraSet; O: TLoraAdamW; AFrom, ACount: Integer);
var
  U, I: Integer;
  L: TTensor;
begin
  for U := AFrom to AFrom + ACount - 1 do
  begin
    L := TTensor.zeros([]);
    for I := 0 to S.Count - 1 do
      L := L + (S.Pair(I).Delta(SinT(5, 8)) - CosT(5, S.Pair(I).OutDim)).square.sum(kNone);
    L.Backward;
    O.ClipGlobalNorm(1.0);
    O.Step(ScheduleLR(0.01, U, 2, 10, 0.1));
    O.ZeroGrad;
  end;
end;

procedure AppendT(var R: TDoubleArray; const T: TTensor);
var
  V: TDoubleArray;
  N, I: Integer;
begin
  V := T.detach.to_device(TDevice.Cpu, kDouble).reshape([-1]).ToDoubleArray;
  N := Length(R);
  SetLength(R, N + Length(V));
  for I := 0 to High(V) do
    R[N + I] := V[I];
end;

class function TCheckpointTest.Snapshot(S: TLoraSet; O: TLoraAdamW): TDoubleArray;
var
  I, J: Integer;
  V: TNamedTensorArray;
begin
  Result := nil;
  for I := 0 to S.Count - 1 do
  begin
    AppendT(Result, S.Pair(I).A);
    AppendT(Result, S.Pair(I).B);
  end;
  for I := 0 to O.StoreCount - 1 do
  begin
    V := O.Moments(I).Variables;
    for J := 0 to High(V) do
      AppendT(Result, V[J].Tensor);
  end;
end;

class function TCheckpointTest.DiffArr(const A, B: TDoubleArray): Double;
var
  I: Integer;
begin
  if Length(A) <> Length(B) then
    Exit(Infinity);
  Result := 0;
  for I := 0 to High(A) do
    Result := Max(Result, Abs(A[I] - B[I]));
end;

class function TCheckpointTest.NewRoot: string;
begin
  Inc(RootNo);
  Result := TmpDir + 'r' + IntToStr(RootNo) + PathDelim;
end;

class function TCheckpointTest.Names(const A: TStringArray): string;
var
  I: Integer;
begin
  Result := '';
  for I := 0 to High(A) do
  begin
    if I > 0 then
      Result := Result + ',';
    Result := Result + ExtractFileName(A[I]);
  end;
end;

{ Writes the file AFile with the text 'x'. }
procedure TouchFile(const AFile: string);
var
  L: TStringList;
begin
  L := TStringList.Create;
  try
    L.Text := 'x';
    L.SaveToFile(AFile);
  finally
    L.Free;
  end;
end;

{ A and B have the same bits (also for NaN). }
{ The tensor's values as Double, row-major. }
function Arr(const T: TTensor): TDoubleArray;
begin
  Result := nil;
  AppendT(Result, T);
end;

{ Moments AFrom .. ATo (in the store's order m.0, v.0, m.1, ...) of store 0 of O. }
function SliceMoments(O: TLoraAdamW; AFrom, ATo: Integer): TDoubleArray;
var
  I: Integer;
begin
  Result := nil;
  for I := AFrom to ATo do
    AppendT(Result, O.Moments(0).Variables[I].Tensor);
end;

{ A and B have the same bits (also for NaN). }
function SameBits(A, B: Double): Boolean;
begin
  Result := CompareMem(@A, @B, SizeOf(Double));
end;

{ The JSON of FixState with Rank 4, Scale 2 and val_loss 0.5;
  key AKey renamed to x<AKey> when AValue is '', else given the value AValue
  (its old value stays under y<AKey>). }
function FixJson(const AKey, AValue: string): string;
begin
  Result := '{ "update" : 5, "epoch" : 1, "position" : 37, "seed" : 16, "rank" : 4, '
    + '"scale" : 2.0, "config_hash" : "cfg-1", "base_sha256" : "abc", '
    + '"val_loss" : 0.5 }';
  if AValue = '' then
    Result := StringReplace(Result, '"' + AKey + '" : ', '"x' + AKey + '" : ', [])
  else
  begin
    Result := StringReplace(Result, '"' + AKey + '" : ', '"' + AKey + '" : ' + AValue + ', "y' + AKey + '" : ', []);
  end;
end;

{ RunStateFromJson(AText) raises ECheckpoint with the message AMessage. }
procedure CheckJsonError(T: TTestCase; const AText, AMessage: string);
begin
  try
    RunStateFromJson(AText);
    T.Fail('no ECheckpoint');
  except
    on E: ECheckpoint do
      T.AssertEquals(AMessage, E.Message);
  end;
end;

{ Saves FixState of NewRun(16) and loads it into a set of ARank, AScale
  with the same pairs and AdamW over its store (twice when ATwice); expects
  ECheckpoint with AMessage, '<dir>' replaced by the checkpoint directory. }
procedure CheckLoad(T: TTestCase; ARank: Int64; AScale: Double; ATwice: Boolean; const AMessage: string);
var
  C: TCheckpoints;
  S, S2: TLoraSet;
  O, O2: TLoraAdamW;
  D: string;
begin
  TCheckpointTest.NewRun(16, S, O);
  S2 := TLoraSet.Create(ARank, AScale);
  S2.Add('blk.0.attn_qkv', 8, 6, TDevice.Cpu);
  S2.Add('blk.1.attn_q', 8, 10, TDevice.Cpu);
  if ATwice then
    O2 := TLoraAdamW.Create([S2.Stores[0], S2.Stores[0]], DefaultAdamWConfig)
  else
    O2 := TLoraAdamW.Create(S2.Stores, DefaultAdamWConfig);
  C := TCheckpoints.Create(TCheckpointTest.NewRoot, 3);
  try
    D := C.Save(TCheckpointTest.FixState, S, O);
    try
      C.Load(D, S2, O2);
      T.Fail('no ECheckpoint');
    except
      on E: ECheckpoint do
        T.AssertEquals(StringReplace(AMessage, '<dir>', D, []), E.Message);
    end;
  finally
    C.Free;
    O2.Free;
    S2.Free;
    O.Free;
    S.Free;
  end;
end;

procedure RemoveTree(const ADir: string);
var
  R: TSearchRec;
  D: string;
begin
  D := IncludeTrailingPathDelimiter(ADir);
  if FindFirst(D + '*', faAnyFile, R) = 0 then
  try
    repeat
      if (R.Name = '.') or (R.Name = '..') then
        Continue;
      if (R.Attr and faDirectory) <> 0 then
        RemoveTree(D + R.Name)
      else
        DeleteFile(D + R.Name);
    until FindNext(R) <> 0;
  finally
    FindClose(R);
  end;
  RemoveDir(ADir);
end;

var
  App: TTestRunner;
begin
  TmpDir := GetTempDir + 'checkpoint-ut-' + IntToStr(GetProcessID) + PathDelim;
  ForceDirectories(TmpDir);
  RegisterTest(TCheckpointTest);
  App := TTestRunner.Create(nil);
  App.Initialize;
  App.Title := 'Checkpoint UT';
  App.Run;
  App.Free;
  RemoveTree(TmpDir);
end.
