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
    procedure TestUT_Checkpoint_001;
    procedure TestUT_Checkpoint_002;
    procedure TestUT_Checkpoint_003;
    procedure TestUT_Checkpoint_004;
    procedure TestUT_Checkpoint_005;
    procedure TestUT_Checkpoint_006;
    procedure TestUT_Checkpoint_007;
    procedure TestUT_Checkpoint_008;
    procedure TestUT_Checkpoint_009;
    procedure TestUT_Checkpoint_010;
    procedure TestUT_Checkpoint_011;
    procedure TestUT_Checkpoint_012;
    procedure TestUT_Checkpoint_101;
    procedure TestUT_Checkpoint_102;
    procedure TestUT_Checkpoint_103;
    procedure TestUT_Checkpoint_104;
    procedure TestUT_Checkpoint_105;
    procedure TestUT_Checkpoint_106;
    procedure TestUT_Checkpoint_107;
    procedure TestUT_Checkpoint_108;
    procedure TestUT_Checkpoint_109;
    procedure TestUT_Checkpoint_110;
    procedure TestUT_Checkpoint_111;
    procedure TestUT_Checkpoint_112;
    procedure TestUT_Checkpoint_113;
    procedure TestUT_Checkpoint_114;
    procedure TestUT_Checkpoint_115;
    procedure TestUT_Checkpoint_116;
    procedure TestUT_Checkpoint_117;
    procedure TestUT_Checkpoint_118;
    procedure TestUT_Checkpoint_119;
    procedure TestUT_Checkpoint_120;
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

{ UT-Checkpoint-001: defaults of NewRunState. }
procedure TCheckpointTest.TestUT_Checkpoint_001;
var
  S: TRunState;
begin
  S := NewRunState;
  AssertEquals(Int64(0), S.Update);
  AssertEquals(Int64(0), S.Epoch);
  AssertEquals(Int64(0), S.Position);
  AssertEquals(Int64(0), S.Seed);
  AssertEquals(Int64(0), S.Rank);
  AssertEquals(0.0, S.Scale);
  AssertEquals('', S.ConfigHash);
  AssertEquals('', S.BaseSha256);
  AssertTrue(IsNan(S.ValLoss));
end;

{ UT-Checkpoint-002: RunStateToJson of the row's state (Rank 4, Scale 2)
  is exactly the facts' text (LineEnding after each line, braces alone). }
procedure TCheckpointTest.TestUT_Checkpoint_002;
var
  S: TRunState;
  E: string;
begin
  S := FixState;
  S.Rank := 4;
  S.Scale := 2;
  E := '{' + LineEnding +
       '  "update" : 5,' + LineEnding +
       '  "epoch" : 1,' + LineEnding +
       '  "position" : 37,' + LineEnding +
       '  "seed" : 16,' + LineEnding +
       '  "rank" : 4,' + LineEnding +
       '  "scale" : 2.0000000000000000E+000,' + LineEnding +
       '  "config_hash" : "cfg-1",' + LineEnding +
       '  "base_sha256" : "abc",' + LineEnding +
       '  "val_loss" : 3.3333333333333331E-001' + LineEnding +
       '}' + LineEnding;
  AssertEquals(E, RunStateToJson(S));
end;

{ UT-Checkpoint-003: JSON round trip with a big seed, an odd config hash
  and scale 0.1: all fields equal (Scale and ValLoss by identical bits). }
procedure TCheckpointTest.TestUT_Checkpoint_003;
var
  S, R: TRunState;
begin
  S := FixState;
  S.Seed := 9007199254740993;
  S.Rank := 16;
  S.Scale := 0.1;
  S.ConfigHash := 'a"b\c/~ ';
  R := RunStateFromJson(RunStateToJson(S));
  AssertEquals(S.Update, R.Update);
  AssertEquals(S.Epoch, R.Epoch);
  AssertEquals(S.Position, R.Position);
  AssertEquals(S.Seed, R.Seed);
  AssertEquals(S.Rank, R.Rank);
  AssertEquals('a"b\c/~ ', R.ConfigHash);
  AssertEquals(S.BaseSha256, R.BaseSha256);
  AssertTrue('scale bits differ', SameBits(S.Scale, R.Scale));
  AssertTrue('val_loss bits differ', SameBits(S.ValLoss, R.ValLoss));
end;

{ UT-Checkpoint-004: a NaN, +Inf or -Inf val_loss serializes to null and
  reads back as NaN. }
procedure TCheckpointTest.TestUT_Checkpoint_004;
var
  S, R: TRunState;
  J: string;
  I: Integer;
  Vals: array[0..2] of Double;
begin
  Vals[0] := NaN;
  Vals[1] := Infinity;
  Vals[2] := -Infinity;
  for I := 0 to 2 do
  begin
    S := FixState;
    S.ValLoss := Vals[I];
    J := RunStateToJson(S);
    AssertTrue('val_loss must serialize as null', Pos('"val_loss" : null', J) > 0);
    R := RunStateFromJson(J);
    AssertTrue('NaN / Inf must read back as NaN', IsNan(R.ValLoss));
  end;
end;

{ UT-Checkpoint-005: seed missing (renamed) -> 'state.json: missing or bad seed'. }
procedure TCheckpointTest.TestUT_Checkpoint_005;
begin
  CheckJsonError(Self, FixJson('seed', ''), 'state.json: missing or bad seed');
end;

{ UT-Checkpoint-006: update a number with a decimal point
  -> 'state.json: missing or bad update'. }
procedure TCheckpointTest.TestUT_Checkpoint_006;
begin
  CheckJsonError(Self, FixJson('update', '5.0'), 'state.json: missing or bad update');
end;

{ UT-Checkpoint-007: config_hash the integer 7
  -> 'state.json: missing or bad config_hash'. }
procedure TCheckpointTest.TestUT_Checkpoint_007;
begin
  CheckJsonError(Self, FixJson('config_hash', '7'), 'state.json: missing or bad config_hash');
end;

{ UT-Checkpoint-008: scale null -> 'state.json: missing or bad scale';
  val_loss the string "0.5" -> 'state.json: missing or bad val_loss'. }
procedure TCheckpointTest.TestUT_Checkpoint_008;
begin
  CheckJsonError(Self, FixJson('scale', 'null'), 'state.json: missing or bad scale');
  CheckJsonError(Self, FixJson('val_loss', '"0.5"'), 'state.json: missing or bad val_loss');
end;

{ UT-Checkpoint-009: a truncated object (a parser error) and the array
  [1, 2] both -> 'state.json: not a JSON object'. }
procedure TCheckpointTest.TestUT_Checkpoint_009;
begin
  CheckJsonError(Self, '{ "update" : ', 'state.json: not a JSON object');
  CheckJsonError(Self, '[1, 2]', 'state.json: not a JSON object');
end;

{ UT-Checkpoint-010: the extra key (yepoch) is ignored; the row's values read. }
procedure TCheckpointTest.TestUT_Checkpoint_010;
var
  R: TRunState;
begin
  R := RunStateFromJson(FixJson('epoch', '3'));
  AssertEquals(Int64(3), R.Epoch);
  AssertEquals(Int64(4), R.Rank);
  AssertEquals(2.0, R.Scale);
  AssertEquals(0.5, R.ValLoss);
end;

{ UT-Checkpoint-011: ReadRunState of a fresh (empty) directory D
  -> ECheckpoint 'not a checkpoint: <D>'. }
procedure TCheckpointTest.TestUT_Checkpoint_011;
var
  D: string;
begin
  D := NewRoot;
  ForceDirectories(D);
  try
    try
      ReadRunState(D);
      Fail('no ECheckpoint');
    except
      on E: ECheckpoint do
        AssertEquals('not a checkpoint: ' + D, E.Message);
    end;
  finally
    RemoveTree(D);
  end;
end;

{ UT-Checkpoint-012: RunStateToJson with a non-ASCII base_sha256 / config_hash
  -> ECheckpoint '<key> is not printable ASCII'. }
procedure TCheckpointTest.TestUT_Checkpoint_012;
var
  S: TRunState;
begin
  S := FixState;
  S.BaseSha256 := 'ab'#9;
  try
    RunStateToJson(S);
    Fail('no ECheckpoint');
  except
    on E: ECheckpoint do
      AssertEquals('base_sha256 is not printable ASCII', E.Message);
  end;
  S := FixState;
  S.ConfigHash := 'h'#200;
  try
    RunStateToJson(S);
    Fail('no ECheckpoint');
  except
    on E: ECheckpoint do
      AssertEquals('config_hash is not printable ASCII', E.Message);
  end;
end;

{ UT-Checkpoint-101: Create with a root that has no trailing delimiter
  creates the directory; Root ends in the delimiter; Keep 3; nothing to list. }
procedure TCheckpointTest.TestUT_Checkpoint_101;
var
  C: TCheckpoints;
  R: string;
  A: TStringArray;
begin
  R := NewRoot + 'a/b';
  C := TCheckpoints.Create(R, 3);
  try
    AssertTrue(DirectoryExists(R));
    AssertEquals(IncludeTrailingPathDelimiter(R), C.Root);
    AssertEquals(3, C.Keep);
    A := C.List;
    AssertEquals(0, Length(A));
    AssertEquals('', C.Latest);
    AssertEquals('', C.Best);
  finally
    C.Free;
  end;
end;

{ UT-Checkpoint-102: Keep 0 -> 'keep must be at least 1, got 0'; a root
  under an existing file -> 'cannot create <root>'. }
procedure TCheckpointTest.TestUT_Checkpoint_102;
var
  C: TCheckpoints;
  R: string;
begin
  try
    C := TCheckpoints.Create(NewRoot, 0);
    C.Free;
    Fail('no ECheckpoint');
  except
    on E: ECheckpoint do
      AssertEquals('keep must be at least 1, got 0', E.Message);
  end;
  R := NewRoot;
  ForceDirectories(R);
  try
    TouchFile(R + 'f');
    try
      C := TCheckpoints.Create(R + 'f/x', 1);
      C.Free;
      Fail('no ECheckpoint');
    except
      on E: ECheckpoint do
        AssertEquals('cannot create ' + R + 'f/x/', E.Message);
    end;
  finally
    RemoveTree(R);
  end;
end;

{ UT-Checkpoint-103: Save writes state.json, lora.ot and moments.ot into
  ckpt-00000005 (no tmp-ckpt-00000005); the state reads back FixState with
  Rank 4 and Scale 2; Latest and Best are the checkpoint. }
procedure TCheckpointTest.TestUT_Checkpoint_103;
var
  C: TCheckpoints;
  S: TLoraSet;
  O: TLoraAdamW;
  D: string;
  R: TRunState;
begin
  NewRun(16, S, O);
  C := TCheckpoints.Create(NewRoot, 3);
  try
    D := C.Save(FixState, S, O);
    AssertEquals(C.Root + 'ckpt-00000005', D);
    AssertTrue(FileExists(D + PathDelim + 'state.json'));
    AssertTrue(FileExists(D + PathDelim + 'lora.ot'));
    AssertTrue(FileExists(D + PathDelim + 'moments.ot'));
    AssertFalse(DirectoryExists(C.Root + 'tmp-ckpt-00000005'));
    R := ReadRunState(D);
    AssertEquals(Int64(5), R.Update);
    AssertEquals(Int64(1), R.Epoch);
    AssertEquals(Int64(37), R.Position);
    AssertEquals(Int64(16), R.Seed);
    AssertEquals(Int64(4), R.Rank);
    AssertTrue(SameBits(2.0, R.Scale));
    AssertEquals('cfg-1', R.ConfigHash);
    AssertEquals('abc', R.BaseSha256);
    AssertTrue(SameBits(FixState.ValLoss, R.ValLoss));
    AssertEquals(D, C.Latest);
    AssertEquals(D, C.Best);
  finally
    C.Free;
    O.Free;
    S.Free;
  end;
end;

{ UT-Checkpoint-104: Load restores the set, the moments and StepCount of a
  different run (S2/O2), exactly. }
procedure TCheckpointTest.TestUT_Checkpoint_104;
var
  C: TCheckpoints;
  S, S2: TLoraSet;
  O, O2: TLoraAdamW;
  D: string;
  R: TRunState;
  Want, Now: TDoubleArray;
begin
  NewRun(16, S, O);
  NewRun(17, S2, O2);
  C := TCheckpoints.Create(NewRoot, 3);
  try
    Train(S, O, 1, 5);
    Want := Snapshot(S, O);
    AssertTrue('S2/O2 must differ from the trained run', DiffArr(Want, Snapshot(S2, O2)) > 0.01);
    D := C.Save(FixState, S, O);
    R := C.Load(D, S2, O2);
    Now := Snapshot(S2, O2);
    AssertEquals(Length(Want), Length(Now));
    AssertTrue('snapshot must be exact', DiffArr(Want, Now) = 0.0);
    AssertEquals(Int64(5), O2.StepCount);
    AssertEquals(Int64(5), R.Update);
    AssertEquals(Int64(4), R.Rank);
    AssertEquals('cfg-1', R.ConfigHash);
  finally
    C.Free;
    O2.Free;
    S2.Free;
    O.Free;
    S.Free;
  end;
end;

{ UT-Checkpoint-105: Save leaves S and O (parameters and moments) unchanged. }
procedure TCheckpointTest.TestUT_Checkpoint_105;
var
  C: TCheckpoints;
  S: TLoraSet;
  O: TLoraAdamW;
  D: string;
  Want, Now: TDoubleArray;
begin
  NewRun(16, S, O);
  C := TCheckpoints.Create(NewRoot, 3);
  try
    Train(S, O, 1, 3);
    Want := Snapshot(S, O);
    D := C.Save(FixState, S, O);
    AssertTrue(DirectoryExists(D));
    Now := Snapshot(S, O);
    AssertEquals(Length(Want), Length(Now));
    AssertTrue('snapshot must be exact', DiffArr(Want, Now) = 0.0);
    AssertEquals(Int64(3), O.StepCount);
  finally
    C.Free;
    O.Free;
    S.Free;
  end;
end;

{ UT-Checkpoint-106: a second Save for the same update
  -> ECheckpoint 'checkpoint exists: <dir>'. }
procedure TCheckpointTest.TestUT_Checkpoint_106;
var
  C: TCheckpoints;
  S: TLoraSet;
  O: TLoraAdamW;
begin
  NewRun(16, S, O);
  C := TCheckpoints.Create(NewRoot, 3);
  try
    C.Save(FixState, S, O);
    try
      C.Save(FixState, S, O);
      Fail('no ECheckpoint');
    except
      on E: ECheckpoint do
        AssertEquals('checkpoint exists: ' + C.Root + 'ckpt-00000005', E.Message);
    end;
  finally
    C.Free;
    O.Free;
    S.Free;
  end;
end;

{ UT-Checkpoint-107: the set has rank 3
  -> ECheckpoint '<dir>: rank 4 scale 2, the set rank 3 scale 2'. }
procedure TCheckpointTest.TestUT_Checkpoint_107;
begin
  CheckLoad(Self, 3, 2.0, False, '<dir>: rank 4 scale 2, the set rank 3 scale 2');
end;

{ UT-Checkpoint-108: the set has scale 0.5
  -> ECheckpoint '<dir>: rank 4 scale 2, the set rank 4 scale 0.5'. }
procedure TCheckpointTest.TestUT_Checkpoint_108;
begin
  CheckLoad(Self, 4, 0.5, False, '<dir>: rank 4 scale 2, the set rank 4 scale 0.5');
end;

{ UT-Checkpoint-109: an optimiser with the store twice
  -> ECheckpoint 'the optimiser has 2 stores, the set 1'. }
procedure TCheckpointTest.TestUT_Checkpoint_109;
begin
  CheckLoad(Self, 4, 2.0, True, 'the optimiser has 2 stores, the set 1');
end;

{ UT-Checkpoint-110: Load of a directory without state.json
  -> ECheckpoint 'not a checkpoint: <dir>'. }
procedure TCheckpointTest.TestUT_Checkpoint_110;
var
  C: TCheckpoints;
  S: TLoraSet;
  O: TLoraAdamW;
  R, D: string;
begin
  NewRun(16, S, O);
  R := NewRoot;
  C := TCheckpoints.Create(R, 3);
  try
    ForceDirectories(R + 'ckpt-00000001');
    D := R + 'ckpt-00000001';
    try
      C.Load(D, S, O);
      Fail('no ECheckpoint');
    except
      on E: ECheckpoint do
        AssertEquals('not a checkpoint: ' + D, E.Message);
    end;
  finally
    C.Free;
    O.Free;
    S.Free;
  end;
end;

{ UT-Checkpoint-111: Load with moments.ot deleted passes the ETorchNativeError
  of TVarStore.Load through unchanged. }
procedure TCheckpointTest.TestUT_Checkpoint_111;
var
  C: TCheckpoints;
  S: TLoraSet;
  O: TLoraAdamW;
  D: string;
begin
  NewRun(16, S, O);
  C := TCheckpoints.Create(NewRoot, 3);
  try
    D := C.Save(FixState, S, O);
    DeleteFile(D + PathDelim + 'moments.ot');
    try
      C.Load(D, S, O);
      Fail('no ETorchNativeError');
    except
      on E: ETorchNativeError do
        AssertEquals('open file failed because of errno 2 on fopen: ' +
          'No such file or directory, file path: ' + D + PathDelim + 'moments.ot',
          E.Message);
    end;
  finally
    C.Free;
    O.Free;
    S.Free;
  end;
end;

{ UT-Checkpoint-112: with Keep 2 the last two updates and the best
  (lowest finite ValLoss) survive. }
procedure TCheckpointTest.TestUT_Checkpoint_112;
var
  C: TCheckpoints;
  S: TLoraSet;
  O: TLoraAdamW;
  U: Integer;
  St: TRunState;
  Vals: array[1..5] of Double;
begin
  Vals[1] := 5;
  Vals[2] := 1;
  Vals[3] := 4;
  Vals[4] := 3;
  Vals[5] := 2;
  NewRun(16, S, O);
  C := TCheckpoints.Create(NewRoot, 2);
  try
    for U := 1 to 5 do
    begin
      St := FixState;
      St.Update := U;
      St.ValLoss := Vals[U];
      C.Save(St, S, O);
    end;
    AssertEquals('ckpt-00000002,ckpt-00000004,ckpt-00000005', Names(C.List));
    AssertEquals(C.Root + 'ckpt-00000002', C.Best);
    AssertEquals(C.Root + 'ckpt-00000005', C.Latest);
  finally
    C.Free;
    O.Free;
    S.Free;
  end;
end;

{ UT-Checkpoint-113: with Keep 1 and NaN losses only the last one survives;
  a tie on the finite loss keeps the earlier update. }
procedure TCheckpointTest.TestUT_Checkpoint_113;
var
  C: TCheckpoints;
  S: TLoraSet;
  O: TLoraAdamW;
  U: Integer;
  St: TRunState;
begin
  NewRun(16, S, O);
  C := TCheckpoints.Create(NewRoot, 1);
  try
    for U := 1 to 3 do
    begin
      St := FixState;
      St.Update := U;
      St.ValLoss := NaN;
      C.Save(St, S, O);
    end;
    AssertEquals('ckpt-00000003', Names(C.List));
    AssertEquals('', C.Best);
    for U := 4 to 5 do
    begin
      St := FixState;
      St.Update := U;
      St.ValLoss := 3;
      C.Save(St, S, O);
    end;
    AssertEquals('ckpt-00000004,ckpt-00000005', Names(C.List));
    AssertEquals(C.Root + 'ckpt-00000004', C.Best);
  finally
    C.Free;
    O.Free;
    S.Free;
  end;
end;

{ UT-Checkpoint-114: tmp-ckpt-* entries left by an interrupted save are
  removed by the next save. }
procedure TCheckpointTest.TestUT_Checkpoint_114;
var
  C: TCheckpoints;
  S: TLoraSet;
  O: TLoraAdamW;
  R: string;
begin
  NewRun(16, S, O);
  R := NewRoot;
  C := TCheckpoints.Create(R, 3);
  try
    ForceDirectories(R + 'tmp-ckpt-00000009' + PathDelim + 'sub');
    TouchFile(R + 'tmp-ckpt-00000009' + PathDelim + 'state.json');
    TouchFile(R + 'tmp-ckpt-00000009' + PathDelim + 'sub' + PathDelim + 'x');
    TouchFile(R + 'tmp-ckpt-00000008');
    AssertEquals(0, Length(C.List));
    AssertEquals('', C.Latest);
    C.Save(FixState, S, O);
    AssertFalse(DirectoryExists(R + 'tmp-ckpt-00000009'));
    AssertFalse(FileExists(R + 'tmp-ckpt-00000008'));
    AssertEquals('ckpt-00000005', Names(C.List));
  finally
    C.Free;
    O.Free;
    S.Free;
  end;
end;

{ UT-Checkpoint-115: List only returns complete checkpoints
  (ckpt-<8 digits> with a state.json, a directory). }
procedure TCheckpointTest.TestUT_Checkpoint_115;
var
  C: TCheckpoints;
  S: TLoraSet;
  O: TLoraAdamW;
  R: string;
  D: TStringArray;
  I: Integer;
begin
  NewRun(16, S, O);
  R := NewRoot;
  C := TCheckpoints.Create(R, 3);
  try
    C.Save(FixState, S, O);
    ForceDirectories(R + 'ckpt-00000003');
    D := ['ckpt-000000001', 'ckpt--0000001', 'ckpt-0000000x'];
    for I := 0 to High(D) do
    begin
      ForceDirectories(R + D[I]);
      TouchFile(R + D[I] + PathDelim + 'state.json');
    end;
    TouchFile(R + 'ckpt-00000007');
    AssertEquals('ckpt-00000005', Names(C.List));
  finally
    C.Free;
    O.Free;
    S.Free;
  end;
end;

{ UT-Checkpoint-116: a file named ckpt-<update> blocks the rename
  -> ECheckpoint 'cannot rename <tmp>'; nothing is listed. }
procedure TCheckpointTest.TestUT_Checkpoint_116;
var
  C: TCheckpoints;
  S: TLoraSet;
  O: TLoraAdamW;
  R: string;
  St: TRunState;
begin
  NewRun(16, S, O);
  R := NewRoot;
  C := TCheckpoints.Create(R, 3);
  try
    TouchFile(R + 'ckpt-00000007');
    St := FixState;
    St.Update := 7;
    try
      C.Save(St, S, O);
      Fail('no ECheckpoint');
    except
      on E: ECheckpoint do
        AssertEquals('cannot rename ' + R + 'tmp-ckpt-00000007', E.Message);
    end;
    AssertEquals(0, Length(C.List));
  finally
    C.Free;
    O.Free;
    S.Free;
  end;
end;

{ UT-Checkpoint-117: a read-only root
  -> ECheckpoint 'cannot create <root>tmp-ckpt-00000005'. }
procedure TCheckpointTest.TestUT_Checkpoint_117;
var
  C: TCheckpoints;
  S: TLoraSet;
  O: TLoraAdamW;
  R: string;
begin
  NewRun(16, S, O);
  R := NewRoot;
  C := TCheckpoints.Create(R, 3);
  FpChmod(R, 555);
  try
    try
      C.Save(FixState, S, O);
      Fail('no ECheckpoint');
    except
      on E: ECheckpoint do
        AssertEquals('cannot create ' + R + 'tmp-ckpt-00000005', E.Message);
    end;
  finally
    FpChmod(R, 755);
    C.Free;
    O.Free;
    S.Free;
  end;
end;

{ UT-Checkpoint-118: exact resume: 10 updates in one go give the same
  parameters and moments as 5 updates, a save, a resume and 5 more. }
procedure TCheckpointTest.TestUT_Checkpoint_118;
var
  C: TCheckpoints;
  S, S2: TLoraSet;
  O, O2: TLoraAdamW;
  St: TRunState;
  Want, Now: TDoubleArray;
begin
  C := TCheckpoints.Create(NewRoot, 3);
  try
    NewRun(16, S, O);
    try
      Train(S, O, 1, 10);
      Want := Snapshot(S, O);
      AssertTrue('snapshot of the one-go run must be non-empty', Length(Want) > 0);
    finally
      O.Free;
      S.Free;
    end;
    NewRun(16, S, O);
    try
      Train(S, O, 1, 5);
      C.Save(FixState, S, O);
    finally
      O.Free;
      S.Free;
    end;
    NewRun(18, S2, O2);
    try
      St := C.Load(C.Latest, S2, O2);
      Train(S2, O2, St.Update + 1, 5);
      Now := Snapshot(S2, O2);
      AssertEquals(Length(Want), Length(Now));
      AssertTrue('snapshot must be exact', DiffArr(Want, Now) = 0.0);
      AssertEquals(Int64(10), O2.StepCount);
    finally
      O2.Free;
      S2.Free;
    end;
  finally
    C.Free;
  end;
end;

{ UT-Checkpoint-119: the moments are stored by variable name, so a set with
  the pairs in the other order loads them to the right variables. }
procedure TCheckpointTest.TestUT_Checkpoint_119;
var
  C: TCheckpoints;
  S, S2: TLoraSet;
  O, O2: TLoraAdamW;
  D: string;
  Mp, Mq: TDoubleArray;
begin
  ManualSeed(16);
  S := TLoraSet.Create(4, 2.0);
  S.Add('p', 8, 6, TDevice.Cpu);
  S.Add('q', 8, 6, TDevice.Cpu);
  O := TLoraAdamW.Create(S.Stores, DefaultAdamWConfig);
  S2 := TLoraSet.Create(4, 2.0);
  S2.Add('q', 8, 6, TDevice.Cpu);
  S2.Add('p', 8, 6, TDevice.Cpu);
  O2 := TLoraAdamW.Create(S2.Stores, DefaultAdamWConfig);
  C := TCheckpoints.Create(NewRoot, 3);
  try
    Train(S, O, 1, 3);
    Mp := SliceMoments(O, 0, 3);
    Mq := SliceMoments(O, 4, 7);
    AssertTrue('p and q moments must differ', DiffArr(Mp, Mq) > 1e-6);
    D := C.Save(FixState, S, O);
    C.Load(D, S2, O2);
    AssertTrue('O2 0..3 must equal q moments', DiffArr(SliceMoments(O2, 0, 3), Mq) = 0.0);
    AssertTrue('O2 4..7 must equal p moments', DiffArr(SliceMoments(O2, 4, 7), Mp) = 0.0);
    AssertTrue('A of p must be equal', DiffArr(Arr(S.Pair(0).A), Arr(S2.Pair(1).A)) = 0.0);
  finally
    C.Free;
    O2.Free;
    S2.Free;
    O.Free;
    S.Free;
  end;
end;

{ UT-Checkpoint-120: a pair added after the optimiser was created
  -> ECheckpoint 'store 0: the optimiser has 8 moments, the set 6 variables'. }
procedure TCheckpointTest.TestUT_Checkpoint_120;
var
  C: TCheckpoints;
  S: TLoraSet;
  O: TLoraAdamW;
  D: string;
begin
  NewRun(16, S, O);
  C := TCheckpoints.Create(NewRoot, 3);
  try
    S.Add('blk.2.attn_q', 8, 10, TDevice.Cpu);
    try
      D := C.Save(FixState, S, O);
      Fail('no ECheckpoint');
    except
      on E: ECheckpoint do
        AssertEquals('store 0: the optimiser has 8 moments, the set 6 variables', E.Message);
    end;
    AssertEquals(0, Length(C.List));
  finally
    C.Free;
    O.Free;
    S.Free;
  end;
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
