# T16 facts (checked by the harness owner)

T16 is the unit `Checkpoint` (DESIGN 8.14 "Checkpoint", 8.12): saving and
restoring everything a LoRA run needs to continue exactly, the adapters
(T12's `TLoraSet`), the AdamW moments (T14's `TLoraAdamW`) and the run
state (`state.json`). fp32 on the CPU; nothing in it depends on the
device map, so the same code serves the two-GPU split of the 27B.

DESIGN 8.18's acceptance for T16: "exact resume" (8.14: ten updates in one
go and five + resume + five give identical parameters on the CPU).
Decisions taken by the harness owner for this task (DESIGN 8.14 leaves
them open):
- Layout: `<root>/ckpt-<update, 8 digits>/` holds `state.json`, `lora.ot`
  (`TLoraSet.Save`) and `moments.ot`. A save writes
  `<root>/tmp-ckpt-<update>/` and renames it, so a directory named
  `ckpt-*` with a `state.json` is always complete; `tmp-ckpt-*` entries
  left by an interrupted save are removed by the next save.
- The moments are stored **by variable name**, like the adapters: one
  archive through a temporary store on the CPU, variables
  `<variable>.adam_m` and `<variable>.adam_v` (e.g.
  `blk.12.attn_qkv.lora_a.adam_m`). T14's moment store `I` holds `m.<J>`,
  `v.<J>` for variable `J` of parameter store `I` (insertion order), so a
  per-store file would depend on the device map: resuming with another
  split would load the moments of other matrices, silently where the
  shapes agree (consecutive linear blocks). The name-keyed file has no
  such dependency (UT row 119 adds the pairs in the other order).
- Retention: after each save, the last `Keep` checkpoints by update and
  the one with the lowest finite validation loss (the earlier on a tie)
  are kept; every other `ckpt-*` directory is deleted. DESIGN 8.14: keep 3.
- `state.json`: the keys of `TRunState` below (fpjson `FormatJSON`, so
  numbers round-trip exactly); a non-finite validation loss (NaN, ±Inf)
  is written as `null` and read back as NaN. `config_hash` and
  `base_sha256` are printable ASCII (fpjson writes other characters
  encoded twice), checked before writing. Unknown keys are ignored when
  reading. The trainer (T17) computes the hash and the sha256; T16 only
  stores them. `Save` takes rank and scale from the set; `Load` refuses a
  set of another rank or scale.
- The optimiser passed to `Save` and `Load` must be the one created over
  the set's stores (`TLoraAdamW.Create(ASet.Stores, ...)`) after the last
  `Add`: as many stores, two moments per variable; checked.

Measured by the harness owner with a prototype written exactly as below:
- UT: every row of the tables below passes; the unit then has 100 % line
  coverage; mutants are caught (`Load` not restoring `StepCount`: 104, 118; `Load` not reading the moments: 104, 111, 118, 119; the moments saved by position (reversed): 104, 118, 119; `Prune` deleting the best too: 112, 113; stale `tmp-ckpt-*` entries left: 114; ±Inf not written as `null`: 004; `Save` without the optimiser check: 120).
- ITa on the rehearsal model: 0, 0, 0, 0; the 14 745 600 adapter and moment values moved by up to 1.3e-2 in updates 6–10, the loss fell from 245.9 to 92.7 (cases below); with `Load` not restoring `StepCount` 001 (3.4e-3) and 004 (12) are NG, with `Load` not reading the moments 001 (2.0e-4) and 004 (4.2), with `Prune` deleting the best 003; with the skeleton every case is NG.

These facts are final; do not investigate further or write probes.

## Unit `Checkpoint` (interface, final; in the skeleton)

```pascal
type
  ECheckpoint = class(Exception);

  TRunState = record
    Update: Int64;        // optimiser updates done
    Epoch: Int64;         // epoch in progress, from 0
    Position: Int64;      // samples of this epoch's order consumed
    Seed: Int64;          // the run's shuffle seed
    Rank: Int64;          // of the adapters (TLoraSet.Rank)
    Scale: Double;        // of the adapters (TLoraSet.Scale)
    ConfigHash: string;   // the trainer's configuration hash (printable ASCII)
    BaseSha256: string;   // sha256 of the base GGUF (printable ASCII)
    ValLoss: Double;      // validation loss; NaN when not measured (written as null, as is ±Inf)
  end;

  TCheckpoints = class
  private
    FRoot: string;
    FKeep: Integer;
    function DirOf(AUpdate: Int64): string;
    procedure RemoveTree(const ADir: string);
    procedure Prune;
  public
    { Creates ARoot when missing. Raises ECheckpoint when AKeep < 1. }
    constructor Create(const ARoot: string; AKeep: Integer);
    { Writes the checkpoint of AState.Update; returns its directory. }
    function Save(const AState: TRunState; ASet: TLoraSet; AOptim: TLoraAdamW): string;
    { Restores ASet and AOptim (moments and StepCount) from ADir; returns its state. }
    function Load(const ADir: string; ASet: TLoraSet; AOptim: TLoraAdamW): TRunState;
    { The complete checkpoints, ascending by update (full paths). }
    function List: TStringArray;
    { The newest checkpoint; '' when there is none. }
    function Latest: string;
    { The checkpoint with the lowest finite validation loss (the earlier on
      a tie); '' when there is none. }
    function Best: string;
    property Root: string read FRoot;
    property Keep: Integer read FKeep;
  end;

{ Update etc. 0, Rank 0, Scale 0, strings empty, ValLoss NaN. }
function NewRunState: TRunState;
{ Raises ECheckpoint when ConfigHash or BaseSha256 has a character outside #32..#126. }
function RunStateToJson(const S: TRunState): string;
function RunStateFromJson(const AText: string): TRunState;
{ The state of checkpoint directory ADir (for qlora_merge). }
function ReadRunState(const ADir: string): TRunState;
```

## Implementation of `Checkpoint` (write it exactly so)

The routines in the order of the unit. The local routines (`GetInt`,
`GetFloat`, `GetStr`, `CheckAscii`, `ReadText`, `WriteText`,
`CheckOptim`, `MomentsOf`, `SaveMoments`, `LoadMoments`) are not in the
interface; write them in the implementation section where shown.

```pascal
implementation

const
  StateFile = 'state.json';
  LoraFile = 'lora.ot';
  MomentFile = 'moments.ot';

function NewRunState: TRunState;
begin
  Result.Update := 0;
  Result.Epoch := 0;
  Result.Position := 0;
  Result.Seed := 0;
  Result.Rank := 0;
  Result.Scale := 0;
  Result.ConfigHash := '';
  Result.BaseSha256 := '';
  Result.ValLoss := NaN;
end;

procedure CheckAscii(const AKey, AValue: string);
var
  I: Integer;
begin
  for I := 1 to Length(AValue) do
    if (AValue[I] < #32) or (AValue[I] > #126) then
      raise ECheckpoint.Create(AKey + ' is not printable ASCII');
end;

function RunStateToJson(const S: TRunState): string;
var
  O: TJSONObject;
begin
  CheckAscii('config_hash', S.ConfigHash);
  CheckAscii('base_sha256', S.BaseSha256);
  O := TJSONObject.Create;
  try
    O.Add('update', S.Update);
    O.Add('epoch', S.Epoch);
    O.Add('position', S.Position);
    O.Add('seed', S.Seed);
    O.Add('rank', S.Rank);
    O.Add('scale', S.Scale);
    O.Add('config_hash', S.ConfigHash);
    O.Add('base_sha256', S.BaseSha256);
    if IsNan(S.ValLoss) or IsInfinite(S.ValLoss) then
      O.Add('val_loss', TJSONNull.Create)
    else
      O.Add('val_loss', S.ValLoss);
    Result := O.FormatJSON + LineEnding;
  finally
    O.Free;
  end;
end;

function GetInt(O: TJSONObject; const AKey: string): Int64;
var
  D: TJSONData;
begin
  D := O.Find(AKey);
  if (D = nil) or not ((D is TJSONIntegerNumber) or (D is TJSONInt64Number)) then
    raise ECheckpoint.Create('state.json: missing or bad ' + AKey);
  Result := D.AsInt64;
end;

function GetFloat(O: TJSONObject; const AKey: string; AllowNull: Boolean): Double;
var
  D: TJSONData;
begin
  D := O.Find(AKey);
  if (D <> nil) and AllowNull and (D.JSONType = jtNull) then
    Exit(NaN);
  if (D = nil) or (D.JSONType <> jtNumber) then
    raise ECheckpoint.Create('state.json: missing or bad ' + AKey);
  Result := D.AsFloat;
end;

function GetStr(O: TJSONObject; const AKey: string): string;
var
  D: TJSONData;
begin
  D := O.Find(AKey);
  if (D = nil) or (D.JSONType <> jtString) then
    raise ECheckpoint.Create('state.json: missing or bad ' + AKey);
  Result := D.AsString;
end;

function RunStateFromJson(const AText: string): TRunState;
var
  D: TJSONData;
  O: TJSONObject;
begin
  try
    D := GetJSON(AText);
  except
    on E: EJSONParser do
      raise ECheckpoint.Create('state.json: not a JSON object');
  end;
  try
    if not (D is TJSONObject) then
      raise ECheckpoint.Create('state.json: not a JSON object');
    O := TJSONObject(D);
    Result.Update := GetInt(O, 'update');
    Result.Epoch := GetInt(O, 'epoch');
    Result.Position := GetInt(O, 'position');
    Result.Seed := GetInt(O, 'seed');
    Result.Rank := GetInt(O, 'rank');
    Result.Scale := GetFloat(O, 'scale', False);
    Result.ConfigHash := GetStr(O, 'config_hash');
    Result.BaseSha256 := GetStr(O, 'base_sha256');
    Result.ValLoss := GetFloat(O, 'val_loss', True);
  finally
    D.Free;
  end;
end;

function ReadText(const AFile: string): string;
var
  L: TStringList;
begin
  L := TStringList.Create;
  try
    L.LoadFromFile(AFile);
    Result := L.Text;
  finally
    L.Free;
  end;
end;

procedure WriteText(const AFile, AText: string);
var
  L: TStringList;
begin
  L := TStringList.Create;
  try
    L.Text := AText;
    L.SaveToFile(AFile);
  finally
    L.Free;
  end;
end;

function ReadRunState(const ADir: string): TRunState;
var
  F: string;
begin
  F := IncludeTrailingPathDelimiter(ADir) + StateFile;
  if not FileExists(F) then
    raise ECheckpoint.Create('not a checkpoint: ' + ADir);
  Result := RunStateFromJson(ReadText(F));
end;

{ AOptim must be the optimiser of ASet's stores: as many stores, two
  moments per variable. }
procedure CheckOptim(ASet: TLoraSet; AOptim: TLoraAdamW);
var
  St: TVarStoreArray;
  I: Integer;
begin
  St := ASet.Stores;
  if Length(St) <> AOptim.StoreCount then
    raise ECheckpoint.CreateFmt('the optimiser has %d stores, the set %d',
      [AOptim.StoreCount, Length(St)]);
  for I := 0 to High(St) do
    if AOptim.Moments(I).Count <> 2 * St[I].Count then
      raise ECheckpoint.CreateFmt('store %d: the optimiser has %d moments, the set %d variables',
        [I, AOptim.Moments(I).Count, St[I].Count]);
end;

{ The moments of variable J of store I: m.<J> and v.<J> of the optimiser's
  moment store I (LoraOptim). }
procedure MomentsOf(AOptim: TLoraAdamW; I, J: Integer; out M, V: TTensor);
begin
  AOptim.Moments(I).TryGet('m.' + IntToStr(J), M);
  AOptim.Moments(I).TryGet('v.' + IntToStr(J), V);
end;

procedure SaveMoments(const AFile: string; ASet: TLoraSet; AOptim: TLoraAdamW);
var
  St: TVarStoreArray;
  S: TVarStore;
  Vars: TNamedTensorArray;
  I, J: Integer;
  P: TVarPath;
  M, V: TTensor;
  G: IInterface;
begin
  G := NoGradGuard;
  St := ASet.Stores;
  S := TVarStore.Create(TDevice.Cpu);
  try
    for I := 0 to High(St) do
    begin
      Vars := St[I].Variables;
      for J := 0 to High(Vars) do
      begin
        MomentsOf(AOptim, I, J, M, V);
        P := LoraPath(S.Root, Vars[J].Name);
        P.Add('adam_m', M.&to(TDevice.Cpu).detach.Copy, False);
        P.Add('adam_v', V.&to(TDevice.Cpu).detach.Copy, False);
      end;
    end;
    S.Save(AFile);
  finally
    S.Free;
  end;
end;

procedure LoadMoments(const AFile: string; ASet: TLoraSet; AOptim: TLoraAdamW);
var
  St: TVarStoreArray;
  S: TVarStore;
  Vars: TNamedTensorArray;
  I, J: Integer;
  P: TVarPath;
  M, V, Fm, Fv: TTensor;
  G: IInterface;
begin
  St := ASet.Stores;
  S := TVarStore.Create(TDevice.Cpu);
  try
    for I := 0 to High(St) do
    begin
      Vars := St[I].Variables;
      for J := 0 to High(Vars) do
      begin
        P := LoraPath(S.Root, Vars[J].Name);
        P.ZerosNoTrain('adam_m', Vars[J].Tensor.Shape);
        P.ZerosNoTrain('adam_v', Vars[J].Tensor.Shape);
      end;
    end;
    S.Load(AFile);
    G := NoGradGuard;
    for I := 0 to High(St) do
    begin
      Vars := St[I].Variables;
      for J := 0 to High(Vars) do
      begin
        MomentsOf(AOptim, I, J, M, V);
        S.TryGet(Vars[J].Name + '.adam_m', Fm);
        S.TryGet(Vars[J].Name + '.adam_v', Fv);
        M.CopyFrom(Fm);
        V.CopyFrom(Fv);
      end;
    end;
  finally
    S.Free;
  end;
end;

{ TCheckpoints }

constructor TCheckpoints.Create(const ARoot: string; AKeep: Integer);
begin
  inherited Create;
  if AKeep < 1 then
    raise ECheckpoint.CreateFmt('keep must be at least 1, got %d', [AKeep]);
  FRoot := IncludeTrailingPathDelimiter(ARoot);
  FKeep := AKeep;
  if not ForceDirectories(FRoot) then
    raise ECheckpoint.Create('cannot create ' + FRoot);
end;

function TCheckpoints.DirOf(AUpdate: Int64): string;
begin
  Result := FRoot + Format('ckpt-%.8d', [AUpdate]);
end;

procedure TCheckpoints.RemoveTree(const ADir: string);
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

function TCheckpoints.List: TStringArray;
var
  R: TSearchRec;
  L: TStringList;
  I: Integer;
  N: Int64;
begin
  L := TStringList.Create;
  try
    if FindFirst(FRoot + 'ckpt-*', faDirectory, R) = 0 then
    try
      repeat
        if ((R.Attr and faDirectory) <> 0) and (Length(R.Name) = 13)
          and TryStrToInt64(Copy(R.Name, 6, 8), N) and (N >= 0)
          and FileExists(FRoot + R.Name + PathDelim + StateFile) then
          L.Add(R.Name);
      until FindNext(R) <> 0;
    finally
      FindClose(R);
    end;
    L.Sort;
    SetLength(Result, L.Count);
    for I := 0 to L.Count - 1 do
      Result[I] := FRoot + L[I];
  finally
    L.Free;
  end;
end;

function TCheckpoints.Latest: string;
var
  A: TStringArray;
begin
  A := List;
  if Length(A) = 0 then
    Exit('');
  Result := A[High(A)];
end;

function TCheckpoints.Best: string;
var
  A: TStringArray;
  I: Integer;
  V, BestV: Double;
begin
  Result := '';
  BestV := Infinity;
  A := List;
  for I := 0 to High(A) do
  begin
    V := ReadRunState(A[I]).ValLoss;
    if not IsNan(V) and (V < BestV) then
    begin
      BestV := V;
      Result := A[I];
    end;
  end;
end;

procedure TCheckpoints.Prune;
var
  A: TStringArray;
  B: string;
  I: Integer;
begin
  A := List;
  B := Best;
  for I := 0 to High(A) - FKeep do
    if A[I] <> B then
      RemoveTree(A[I]);
end;

function TCheckpoints.Save(const AState: TRunState; ASet: TLoraSet; AOptim: TLoraAdamW): string;
var
  R: TSearchRec;
  T: string;
  S: TRunState;
begin
  CheckOptim(ASet, AOptim);
  Result := DirOf(AState.Update);
  if DirectoryExists(Result) then
    raise ECheckpoint.Create('checkpoint exists: ' + Result);
  { leftovers of an interrupted save }
  if FindFirst(FRoot + 'tmp-ckpt-*', faAnyFile, R) = 0 then
  try
    repeat
      if (R.Attr and faDirectory) <> 0 then
        RemoveTree(FRoot + R.Name)
      else
        DeleteFile(FRoot + R.Name);
    until FindNext(R) <> 0;
  finally
    FindClose(R);
  end;
  T := FRoot + 'tmp-' + ExtractFileName(Result);
  if not ForceDirectories(T) then
    raise ECheckpoint.Create('cannot create ' + T);
  S := AState;
  S.Rank := ASet.Rank;
  S.Scale := ASet.Scale;
  ASet.Save(T + PathDelim + LoraFile);
  SaveMoments(T + PathDelim + MomentFile, ASet, AOptim);
  WriteText(T + PathDelim + StateFile, RunStateToJson(S));
  if not RenameFile(T, Result) then
    raise ECheckpoint.Create('cannot rename ' + T);
  Prune;
end;

function TCheckpoints.Load(const ADir: string; ASet: TLoraSet; AOptim: TLoraAdamW): TRunState;
var
  D: string;
begin
  Result := ReadRunState(ADir);
  if (Result.Rank <> ASet.Rank) or (Result.Scale <> ASet.Scale) then
    raise ECheckpoint.CreateFmt('%s: rank %d scale %g, the set rank %d scale %g',
      [ADir, Result.Rank, Result.Scale, ASet.Rank, ASet.Scale]);
  CheckOptim(ASet, AOptim);
  D := IncludeTrailingPathDelimiter(ADir);
  ASet.Load(D + LoraFile);
  LoadMoments(D + MomentFile, ASet, AOptim);
  AOptim.StepCount := Result.Update;
end;
```

## Checks and messages (in this order)

`ECheckpoint` (unit `Checkpoint`):
- `RunStateToJson`: `<key> is not printable ASCII` for `config_hash`, then
  `base_sha256`, when a character is outside #32..#126.
- `RunStateFromJson`: `state.json: not a JSON object` when the text does
  not parse (`EJSONParser`) or is not an object; then, key by key in the
  order `update`, `epoch`, `position`, `seed`, `rank`, `scale`,
  `config_hash`, `base_sha256`, `val_loss`: `state.json: missing or bad
  <key>` when the key is missing or of the wrong JSON type (an integer
  key needs an integer number, `5.0` is bad; `scale` a number; the two
  strings a string; `val_loss` a number or `null`).
- `ReadRunState` (and `Load`): `not a checkpoint: <dir>` (the directory as
  given) when `<dir>/state.json` does not exist.
- `TCheckpoints.Create`: `CreateFmt('keep must be at least 1, got %d',
  [AKeep])`; `cannot create <root>` (with the delimiter) when the root
  cannot be created.
- `Save`, in this order: the optimiser check (below); `checkpoint exists:
  <dir>` when `ckpt-<update>` exists as a directory; `cannot create
  <tmp>` when the temporary directory cannot be created; `cannot rename
  <tmp>` when the rename fails (e.g. a file named `ckpt-<update>`; the
  temporary directory stays until the next save).
- `Load`, in this order: `not a checkpoint: <dir>`; `CreateFmt('%s: rank %d
  scale %g, the set rank %d scale %g', [ADir, state rank, state scale,
  set rank, set scale])`; the optimiser check.
- The optimiser check: `CreateFmt('the optimiser has %d stores, the set
  %d', ...)`, then per store `CreateFmt('store %d: the optimiser has %d
  moments, the set %d variables', ...)`.
- From tch-fpc, passed on unchanged: a missing `lora.ot` or `moments.ot`
  (`ETorchNativeError` `open file failed because of errno 2 on fopen: No
  such file or directory, file path: <file>`), a file without one of the
  variables (`ETorchError` `variable <name> not found in <file>`), another
  shape (`ETorchNativeError` from the copy).

## UT fixture and helpers (harness owner's; nobody changes them)

`tests/ut/Checkpoint_test.pas` (skeleton: helpers, runner, no test
methods). No model; a small set and its optimiser:
- `FixState`: Update 5, Epoch 1, Position 37, Seed 16, Rank 0, Scale 0,
  ConfigHash `cfg-1`, BaseSha256 `abc`, ValLoss 1/3.
- `NewRun(ASeed, S, O)`: `ManualSeed(ASeed)`; S := a set of rank 4, scale
  2 with the pairs `blk.0.attn_qkv` 8 -> 6 and `blk.1.attn_q` 8 -> 10 on
  the CPU; O := `TLoraAdamW.Create(S.Stores, DefaultAdamWConfig)`. Free O,
  then S.
- `Train(S, O, AFrom, ACount)`: updates AFrom .. AFrom + ACount − 1 of a
  toy problem (loss = Σ over the pairs of ((Delta(X) − Y)²).sum, X [5, 8]
  = sin(1 + i), Y = cos(1 + i); Backward, `ClipGlobalNorm(1)`,
  `Step(ScheduleLR(0.01, U, 2, 10, 0.1))`, `ZeroGrad`).
- `Snapshot(S, O)`: every A and B, then every moment, as Double;
  `DiffArr(A, B)` (max |a − b|, Infinity for other lengths); `Arr(T)`;
  `SliceMoments(O, From, To)` (moment variables From..To of store 0, in
  the store's order m.0, v.0, m.1, ...).
- `NewRoot` (a fresh directory under the test's temporary directory, not
  created, ending in the delimiter); `Names(A)` (the file names joined by
  commas); `TouchFile(F)`; `SameBits(A, B)`.
- `FixJson(Key, Value)`: the JSON of FixState with Rank 4, Scale 2 and
  val_loss 0.5, key Key renamed to x<Key> when Value is '', else given
  Value (the old value stays under y<Key>); `CheckJsonError(Self, Text,
  Message)`.
- `CheckLoad(Self, Rank, Scale, Twice, Message)`: saves FixState of
  `NewRun(16)` and loads it into a set of that rank and scale with the
  same pairs and AdamW over its store (the store twice when Twice);
  expects ECheckpoint with Message, `<dir>` replaced by the checkpoint
  directory.
- Each test frees what it creates in `try ... finally`.

## Unit tests (rows; every row has one method, every method belongs to a row)

Error rows catch the named exception class and compare the whole message
(`AssertEquals`); a row fails if no exception comes. "exact" means equal
strings, equal bits or a difference of 0.

`Checkpoint`, stage 1 (task T16b1): `NewRunState`, `RunStateToJson`,
`RunStateFromJson`, `ReadRunState`; 001–012.

| Row | Input | Expected |
|---|---|---|
| 001 | `NewRunState` | Update, Epoch, Position, Seed, Rank 0; Scale 0; ConfigHash, BaseSha256 ''; ValLoss NaN |
| 002 | FixState with Rank 4, Scale 2: `RunStateToJson` | exactly `{`, then one line per key with two spaces of indent: `"update" : 5,` `"epoch" : 1,` `"position" : 37,` `"seed" : 16,` `"rank" : 4,` `"scale" : 2.0000000000000000E+000,` `"config_hash" : "cfg-1",` `"base_sha256" : "abc",` `"val_loss" : 3.3333333333333331E-001`, then `}` and a line end (LineEnding after every line) |
| 003 | FixState with Seed 9007199254740993, Rank 16, Scale 0.1, ConfigHash `a"b\c/~ ` (with the trailing space): `RunStateFromJson(RunStateToJson(S))` | every field equal; Scale and ValLoss with the same bits |
| 004 | FixState with ValLoss NaN, then +Inf, then −Inf | the JSON contains `"val_loss" : null` (NaN and +Inf); read back NaN (NaN and −Inf) |
| 005 | `FixJson('seed', '')` | ECheckpoint `state.json: missing or bad seed` |
| 006 | `FixJson('update', '5.0')` | ECheckpoint `state.json: missing or bad update` |
| 007 | `FixJson('config_hash', '7')` | ECheckpoint `state.json: missing or bad config_hash` |
| 008 | `FixJson('scale', 'null')`; `FixJson('val_loss', '"0.5"')` | ECheckpoint `state.json: missing or bad scale`; `state.json: missing or bad val_loss` |
| 009 | `{ "update" : ` (cut); `[1, 2]` | both ECheckpoint `state.json: not a JSON object` |
| 010 | `FixJson('epoch', '3')` (an extra key `yepoch`) | Epoch 3, Rank 4, Scale 2, ValLoss 0.5 |
| 011 | `ReadRunState(D)`, D an existing empty directory | ECheckpoint `not a checkpoint: <D>` |
| 012 | FixState with BaseSha256 `ab` + #9; then FixState with ConfigHash `h` + #200: `RunStateToJson` | ECheckpoint `base_sha256 is not printable ASCII`; `config_hash is not printable ASCII` |

`Checkpoint`, stage 2 (task T16b2): `TCheckpoints`; 101–120. S, O :=
`NewRun(16)` unless said otherwise; C := `TCheckpoints.Create(NewRoot,
3)` unless said otherwise; free C, then O, then S.

| Row | Input | Expected |
|---|---|---|
| 101 | `TCheckpoints.Create(NewRoot + 'a/b', 3)` (no delimiter at the end) | the directory exists; Root = it + delimiter; Keep 3; `List` empty; `Latest` ''; `Best` '' |
| 102 | `Create(NewRoot, 0)`; R := an existing directory with a file `f`: `Create(R + 'f/x', 1)` | ECheckpoint `keep must be at least 1, got 0`; ECheckpoint `cannot create <R>f/x/` |
| 103 | `D := C.Save(FixState, S, O)` | D = Root + `ckpt-00000005`; `state.json`, `lora.ot`, `moments.ot` in D; no `tmp-ckpt-00000005`; `ReadRunState(D)`: FixState with Rank 4, Scale 2 (ValLoss same bits); `Latest` = `Best` = D |
| 104 | `NewRun(17)` as S2, O2; `Train(S, O, 1, 5)`; Want := `Snapshot(S, O)`; D := `C.Save(FixState, S, O)`; `C.Load(D, S2, O2)` | before the load S2, O2 differ from Want by > 0.01; after: `Snapshot(S2, O2)` = Want (exact); O2.StepCount 5; the state returned: Update 5, Rank 4, ConfigHash `cfg-1` |
| 105 | `Train(S, O, 1, 3)`; Want := `Snapshot(S, O)`; `C.Save(FixState, S, O)` | `Snapshot(S, O)` = Want (exact); O.StepCount 3 |
| 106 | `C.Save(FixState, S, O)` twice | the second: ECheckpoint `checkpoint exists: <Root>ckpt-00000005` |
| 107 | `CheckLoad(Self, 3, 2.0, False, ..)` | ECheckpoint `<dir>: rank 4 scale 2, the set rank 3 scale 2` |
| 108 | `CheckLoad(Self, 4, 0.5, False, ..)` | ECheckpoint `<dir>: rank 4 scale 2, the set rank 4 scale 0.5` |
| 109 | `CheckLoad(Self, 4, 2.0, True, ..)` | ECheckpoint `the optimiser has 2 stores, the set 1` |
| 110 | a directory `ckpt-00000001` without `state.json` under Root: `C.Load(it, S, O)` | ECheckpoint `not a checkpoint: <Root>ckpt-00000001` |
| 111 | D := `C.Save(FixState, S, O)`; delete `D/moments.ot`; `C.Load(D, S, O)` | ETorchNativeError `open file failed because of errno 2 on fopen: No such file or directory, file path: <D>/moments.ot` |
| 112 | C with Keep 2; saves of updates 1..5 with ValLoss 5, 1, 4, 3, 2 | `Names(C.List)` `ckpt-00000002,ckpt-00000004,ckpt-00000005`; `Best` Root + `ckpt-00000002`; `Latest` Root + `ckpt-00000005` |
| 113 | C with Keep 1; saves of updates 1..3 with ValLoss NaN; then 4 and 5 with ValLoss 3 | after 3: `Names(C.List)` `ckpt-00000003`, `Best` ''; after 5: `ckpt-00000004,ckpt-00000005` (a tie keeps the earlier), `Best` Root + `ckpt-00000004` |
| 114 | under Root: a directory `tmp-ckpt-00000009` with `state.json` and `sub/x`, a file `tmp-ckpt-00000008`; then `C.Save(FixState, S, O)` | before the save: `List` empty, `Latest` ''; after: both entries gone; `Names(C.List)` `ckpt-00000005` |
| 115 | `C.Save(FixState, S, O)`; under Root: directories `ckpt-00000003` (empty), `ckpt-000000001`, `ckpt--0000001`, `ckpt-0000000x` (each with a `state.json`), a file `ckpt-00000007` | `Names(C.List)` `ckpt-00000005` |
| 116 | a file `ckpt-00000007` under Root; `C.Save` of FixState with Update 7 | ECheckpoint `cannot rename <Root>tmp-ckpt-00000007`; `List` empty |
| 117 | Root made read-only (`FpChmod(Root, &555)`, restored to `&755` in `finally`); `C.Save(FixState, S, O)` | ECheckpoint `cannot create <Root>tmp-ckpt-00000005` |
| 118 | one go: `NewRun(16)`, `Train(.., 1, 10)`, Want := `Snapshot`; resumed: `NewRun(16)`, `Train(.., 1, 5)`, `C.Save(FixState, ..)`, free; `NewRun(18)` as S2, O2; St := `C.Load(C.Latest, S2, O2)`; `Train(S2, O2, St.Update + 1, 5)` | `Snapshot(S2, O2)` = Want (exact); O2.StepCount 10 |
| 119 | `ManualSeed(16)`; S: rank 4, scale 2, pairs `p` then `q`, both 8 -> 6, O over S.Stores; S2 the same with `q` then `p`, O2 over S2.Stores; `Train(S, O, 1, 3)`; D := `C.Save(FixState, S, O)`; `C.Load(D, S2, O2)` | the moments of p (O's m.0, v.0, m.1, v.1) differ from those of q by > 1e-6; O2's moments 4..7 equal p's, 0..3 equal q's (exact); A of p equal (exact) |
| 120 | `S.Add('blk.2.attn_q', 8, 10, TDevice.Cpu)` after O was created; `C.Save(FixState, S, O)` | ECheckpoint `store 0: the optimiser has 8 moments, the set 6 variables`; `List` empty |

Build and run (the GPUs hidden as in the gate):
`fpc -Mobjfpc -Sh -vew -l- -Fu{proj}/src -Fu{proj}/../../src -Fu{proj}/tests/ut -FU{proj}/work/bin -Fl{proj}/../../build/libtch-rocm -k-rpath={proj}/../../build/libtch-rocm {proj}/tests/ut/Checkpoint_test.pas -o{proj}/work/bin/Checkpoint_test`
then `HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 {proj}/work/bin/Checkpoint_test --all --format=plain`.
Warnings from other units (`GgufQuant`, `GgufQuantT`, `QwenTokenizer`,
`generics.dictionaries.inc`) and libtorch's GPU warnings on stderr are
expected; so are the notes "Local variable G is assigned but never used"
(the `NoGradGuard` idiom). Until a stage is written, the test program
without test methods reports `No valid tests found`; that is expected.

## ITa (prepared by the harness owner)

`QLORA_MODEL=<rehearsal GGUF> {proj}/tests/ita/T16-run.sh` builds
`tests/ita/T16/ckpt_check.pas` against `src/` and runs it four times on
the CPU from a fresh directory outside the project, each run its own
process: `full` (10 updates in one go, a checkpoint after each, keep 3),
`part1` (updates 1–5, a checkpoint), `part2` (a fresh run with another
seed, resumed from the latest checkpoint, updates 6–10, a checkpoint),
`compare`. One update: minus the log-likelihood of prompt chat-065 of
`work/oracle/T11` through the whole graph, Backward, clip 1, AdamW at
`ScheduleLR(1e-4, U, 2, 10, 0.1)`. Rank 16, scale 2. The driver sets the
validation loss of update U to 1 at update 2 and 10 + U otherwise. Writes
`{proj}/work/ita/T16/summary.tsv` (case, OK/NG, value, limit, note; then
the four exit codes and the count of stderr lines other than the GPU
warnings). OK when value ≤ limit. About two minutes; the checkpoints (about
0.5 GB) are deleted afterwards.

| Case | Compared | Limit |
|---|---|---|
| ITa-T16-001 | every A, B and moment of `full/ckpt-00000010` against `part/ckpt-00000010` (both read back into fresh sets) | 0 (exact) |
| ITa-T16-002 | `state.json` of `full/ckpt-10`, `part/ckpt-10` and `part/ckpt-5` against the states written (count of differing fields) | 0 |
| ITa-T16-003 | entries left: `full` exactly `ckpt-00000002,ckpt-00000008,ckpt-00000009,ckpt-00000010` (the last 3 and the best), `part` exactly `ckpt-00000005,ckpt-00000010` (count of mismatches) | 0 |
| ITa-T16-004 | the loss of every update, one go against resumed (`full-loss.tsv`, `part-loss.tsv`) | 0 (exact) |

Measured with the prototype: 0, 0, 0, 0 (bit-identical: the CPU runs are deterministic for the same shapes). A different value in 001 or 004 is a defect, not noise.

## Stages and case IDs

- Stage 1 (task T16b1): `NewRunState`, `RunStateToJson` (with
  `CheckAscii`), `RunStateFromJson` (with `GetInt`, `GetFloat`, `GetStr`),
  `ReadText`, `WriteText`, `ReadRunState`; rows UT-Checkpoint-001 to 012.
- Stage 2 (task T16b2): `CheckOptim`, `MomentsOf`, `SaveMoments`,
  `LoadMoments` and `TCheckpoints` (all of it); rows UT-Checkpoint-101 to
  120.
- ITa: ITa-T16-001 to 004.
