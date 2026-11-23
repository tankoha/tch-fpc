# Detailed Design — Unit `Checkpoint` (T16)

Source of truth: `../external-design/T16-external.md`,
`../system-flow/T16-system.md` and the T16 facts
(`tasks/T16-facts.md`). This document restates and organises them; it adds
no new requirements.

## 1. Module composition

- Source file: `src/Checkpoint.pas`, `{$mode objfpc}{$H+}`.
- Units used: `Tch`, `TchNN` (stores, `NoGradGuard`, `TDevice`,
  `LoraPath`), `Lora` (`TLoraSet`), `LoraOptim` (`TLoraAdamW`), `fpjson`
  (JSON), `Math` (`NaN`, `Infinity`, `IsNan`, `IsInfinite`),
  `System.Classes` (`TStringList`, `TStringArray`), `System.SysUtils`
  (`Format`, file and directory operations).
- Declared:
  - `ECheckpoint = class(Exception)` — the unit's error class.
  - `TRunState` — the run-state record (3, 4.1).
  - `TCheckpoints` — the checkpoint manager: private fields `FRoot`,
    `FKeep` and private methods `DirOf`, `RemoveTree`, `Prune`; public
    `Create`, `Save`, `Load`, `List`, `Latest`, `Best`; properties `Root`,
    `Keep`.
  - Functions (interface section): `NewRunState`, `RunStateToJson`,
    `RunStateFromJson`, `ReadRunState`.
  - Local routines (implementation section only): `CheckAscii`, `GetInt`,
    `GetFloat`, `GetStr`, `ReadText`, `WriteText`, `CheckOptim`,
    `MomentsOf`, `SaveMoments`, `LoadMoments`.
  - Constants (implementation section): `StateFile = 'state.json'`,
    `LoraFile = 'lora.ot'`, `MomentFile = 'moments.ot'`.

## 2. Function specifications

| Function | Arguments | Returns | Raises | Preconditions |
|---|---|---|---|---|
| `NewRunState` | — | the initial `TRunState` (zeros, `ValLoss` NaN) | — | — |
| `CheckAscii` (local) | `const AKey, AValue: string` | — | `ECheckpoint` (4.7, #1) | — |
| `RunStateToJson` | `const S: TRunState` | the `state.json` text + `LineEnding` | `ECheckpoint` (4.7, #1) | — |
| `GetInt` (local) | `O: TJSONObject; const AKey: string` | `Int64` | `ECheckpoint` (4.7, #2) | `O` alive |
| `GetFloat` (local) | `O: TJSONObject; const AKey: string; AllowNull: Boolean` | `Double` (`NaN` for `null` when `AllowNull`) | `ECheckpoint` (4.7, #2) | `O` alive |
| `GetStr` (local) | `O: TJSONObject; const AKey: string` | `string` | `ECheckpoint` (4.7, #2) | `O` alive |
| `RunStateFromJson` | `const AText: string` | the parsed `TRunState` | `ECheckpoint` (4.7, #2) | — |
| `ReadText` (local) | `const AFile: string` | the file's text | — | the file exists |
| `WriteText` (local) | `const AFile, AText: string` | — | — | the directory writable |
| `ReadRunState` | `const ADir: string` | the state of `ADir` | `ECheckpoint` (4.7, #2–#3) | — |
| `CheckOptim` (local) | `ASet: TLoraSet; AOptim: TLoraAdamW` | — | `ECheckpoint` (4.7, #7) | both alive |
| `MomentsOf` (local) | `AOptim: TLoraAdamW; I, J: Integer; out M, V: TTensor` | the moments of variable `J` of store `I` | tch-fpc (4.7, #8) | `I` in range |
| `SaveMoments` (local) | `const AFile: string; ASet: TLoraSet; AOptim: TLoraAdamW` | — | tch-fpc (4.7, #8) | `CheckOptim` already passed |
| `LoadMoments` (local) | `const AFile: string; ASet: TLoraSet; AOptim: TLoraAdamW` | — | tch-fpc (4.7, #8) | `CheckOptim` already passed |
| `TCheckpoints.Create` | `const ARoot: string; AKeep: Integer` | new manager; `ARoot` created when missing | `ECheckpoint` (4.7, #4) | — |
| `DirOf` (private) | `AUpdate: Int64` | `<root>/ckpt-<update, 8 digits>` | — | created |
| `RemoveTree` (private) | `const ADir: string` | — | — | — |
| `List` | — | the complete checkpoints, ascending (full paths) | — | created |
| `Latest` | — | the newest; `''` when none | — | created |
| `Best` | — | the lowest finite validation loss (the earlier on a tie); `''` when none | `ECheckpoint` (through `ReadRunState`, 4.7, #2–#3) | created |
| `Prune` (private) | — | — | — | created |
| `Save` | `const AState: TRunState; ASet: TLoraSet; AOptim: TLoraAdamW` | the checkpoint directory | `ECheckpoint` (4.7, #5, #7), tch-fpc (4.7, #8) | created; no `ckpt-<update>` yet |
| `Load` | `const ADir: string; ASet: TLoraSet; AOptim: TLoraAdamW` | the state of `ADir` | `ECheckpoint` (4.7, #3, #6, #7), tch-fpc (4.7, #8) | a complete checkpoint directory |

`TTensor` is a managed record: never freed by this unit. `TJSONObject`,
`TVarStore` and `TStringList` objects are freed in `try … finally`. The
tch-fpc errors (4.7, #8) are raised by `TLoraSet.Save`/`Load`,
`TVarStore.Save`/`Load` and the tensor copies, and pass through unchanged.

## 3. Data structures

- Constants (implementation section): `StateFile = 'state.json'`,
  `LoraFile = 'lora.ot'`, `MomentFile = 'moments.ot'`.
- `TRunState` record, fields (4.1):

  | Field | Type | Meaning (external design, "Keys of `state.json`") |
  |---|---|---|
  | `Update` | `Int64` | optimiser updates done |
  | `Epoch` | `Int64` | epoch in progress, from 0 |
  | `Position` | `Int64` | samples of this epoch's order consumed |
  | `Seed` | `Int64` | the run's shuffle seed |
  | `Rank` | `Int64` | of the adapters (`TLoraSet.Rank`) |
  | `Scale` | `Double` | of the adapters (`TLoraSet.Scale`) |
  | `ConfigHash` | `string` | the trainer's configuration hash (printable ASCII) |
  | `BaseSha256` | `string` | sha256 of the base GGUF (printable ASCII) |
  | `ValLoss` | `Double` | validation loss; NaN when not measured; non-finite written as `null` |

- `TCheckpoints` fields (private): `FRoot: string` (root with trailing
  delimiter), `FKeep: Integer`.
- Checkpoint directory: `<root>/ckpt-<update, 8 digits>/` holding exactly
  `state.json`, `lora.ot`, `moments.ot`. The 8-digit update makes the
  lexical order equal to the update order. A save works in
  `<root>/tmp-ckpt-<update>/` first and renames it to the final name.
- `state.json`: a JSON object with the 9 keys of the table above, written
  by `fpjson FormatJSON` (numbers round-trip exactly).
- `lora.ot`: a `TVarStore` archive (`TLoraSet.Save`); the variables are
  the LoRA pair parameters, named by their path (e.g.
  `blk.0.attn_qkv.lora_a`).
- `moments.ot`: a `TVarStore` archive; for every variable of the set, the
  two variables `<variable>.adam_m` and `<variable>.adam_v` (e.g.
  `blk.12.attn_qkv.lora_a.adam_m`). Keyed by variable name, not by the
  variable's index in a store, so the mapping does not depend on the device
  map or the variable order (external design, "Why the moments are stored
  by variable name"; UT row 119 loads them into a set with the reversed
  order).

## 4. Processing detail

### 4.1 `TRunState` and `NewRunState`

The record (interface section):

```pascal
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
```

The initial state is zero everywhere, `ValLoss` NaN:

```pascal
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
```

Notes:

- `ConfigHash` and `BaseSha256` are computed by the trainer (T17); this
  unit only stores them (external design, "Keys of `state.json`").
- The `Rank` and `Scale` written into `state.json` are taken from the
  adapter set at save time, not from `AState` (4.6 `Save`).
- `ValLoss` is NaN until a validation loss is measured; NaN and ±Inf are
  written as JSON `null` (4.2) and read back as NaN (4.3).

### 4.2 `RunStateToJson` (with `CheckAscii`)

Both strings are checked before any writing, `config_hash` first, then
`base_sha256` (4.7, #1):

```pascal
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
```

Notes:

- The keys are added in the order `update`, `epoch`, `position`, `seed`,
  `rank`, `scale`, `config_hash`, `base_sha256`, `val_loss`.
- `val_loss` is a JSON `null` when `IsNan(S.ValLoss) or
  IsInfinite(S.ValLoss)` (NaN, +Inf, −Inf), otherwise the exact number.
- `FormatJSON` gives the exact round-trip of the numbers (external design,
  "Keys of `state.json`"); a `LineEnding` is appended after the closing
  brace.
- The printable-ASCII check covers the characters `#32..#126`; the first
  offending character raises (external design, "Errors").

### 4.3 `RunStateFromJson` (with `GetInt`, `GetFloat`, `GetStr`)

```pascal
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
```

Notes:

- Text that does not parse (`EJSONParser`) and text that parses to a
  non-object (e.g. `[1, 2]`) both raise `state.json: not a JSON object`
  (4.7, #2).
- The nine keys are read in the order `update`, `epoch`, `position`, `seed`,
  `rank`, `scale`, `config_hash`, `base_sha256`, `val_loss`.
- Type rules (a violation raises `state.json: missing or bad <key>`): the
  five integer keys need an integer number — `5.0` is bad; `scale` needs a
  number (`null` is bad, `AllowNull = False`); `config_hash` and
  `base_sha256` need strings; `val_loss` needs a number or `null`
  (`AllowNull = True`, `null` → NaN).
- Unknown keys are ignored: only the nine listed keys are read
  (external design, "Keys of `state.json`").
- `D` is the parsed document; it is freed in `finally`.

### 4.4 `ReadText`, `WriteText`, `ReadRunState`

```pascal
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
```

Notes:

- `ReadRunState` is the entry point for `qlora_merge` (T18) and for
  `Load`: it needs only the directory, then reads
  `<dir>/state.json` (4.7, #3; the error message contains `ADir` as given).
- `WriteText` is the only place `state.json` is written (in `Save`, 4.6).

### 4.5 `CheckOptim`, `MomentsOf`, `SaveMoments`, `LoadMoments`

```pascal
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
```

Notes:

- `CheckOptim`: the optimiser passed to `Save` and `Load` must be the one
  created over the set's stores (`TLoraAdamW.Create(ASet.Stores, …)`) after
  the last `Add`. It is checked twice, in this order (4.7, #7): the store
  count, then — for each store `I`, the first offending one raised — the
  moment count of `2 × variables`.
- `MomentsOf` names the moments by the variable's index `J` inside the
  optimiser's moment store `I` (`m.<J>`, `v.<J>`); `TryGet` does not raise
  because `CheckOptim` guarantees the count.
- `SaveMoments`: temporary store on the CPU; per variable the two moments
  are detached, moved to the CPU and copied into the store under the names
  `<variable>.adam_m` / `<variable>.adam_v` (through `LoraPath(S.Root,
  Vars[J].Name)`) — by the variable's **name**, so the archive does not
  depend on the device map (external design, "Why the moments are stored by
  variable name"); the store is freed in `finally`. `G` keeps the
  `NoGradGuard` alive for the whole block (a bare `NoGradGuard;` statement
  has no effect).
- `LoadMoments`: first the placeholder variables are created with the exact
  shape of every variable of the set (`ZerosNoTrain`), then `S.Load(AFile)`
  fills them — a file without one of the variables or with another shape
  then raises the tch-fpc error (4.7, #8); afterwards the values are copied
  back into the optimiser's moments by variable name, under the guard.
- The `Copy`/`CopyFrom`/`&to` idiom follows the tch-fpc rules: `Copy` is
  new storage, in-place updates of the leaf moments only under
  `NoGradGuard`, `TTensor` is a managed record and never freed.

### 4.6 `TCheckpoints`

#### `Create`

```pascal
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
```

- `AKeep < 1` raises before anything is created (4.7, #4).
- `FRoot` always ends in the path delimiter; the root is created when
  missing, and `cannot create <root>` is the message with the delimiter
  (4.7, #4).
- The retention default is 3 (DESIGN 8.14, external design "Retention
  rule"); it is a property of the instance, not of the unit.

#### `DirOf`

```pascal
function TCheckpoints.DirOf(AUpdate: Int64): string;
begin
  Result := FRoot + Format('ckpt-%.8d', [AUpdate]);
end;
```

8 digits (e.g. `ckpt-00000005`), so the lexical order of the names equals
the update order (external design, "Layout of a checkpoint directory").

#### `RemoveTree`

```pascal
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
```

Removes a tree recursively: sub-directories first, files per directory,
then the directory itself; used for pruning (retention rule) and for the
stale `tmp-ckpt-*` cleanup.

#### `List`

```pascal
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
```

A name is a **complete checkpoint** only when it is a directory named
`ckpt-` + 8 digits (13 characters, digits parse as an `Int64` ≥ 0) and
holds a `state.json`. This is the "always complete" invariant of the
save-by-rename (external design, "Save through a temporary directory and
rename"): `ckpt-00000003` as an empty directory, `ckpt-000000001`,
`ckpt--0000001`, `ckpt-0000000x`, and a *file* `ckpt-00000007` are all
excluded (UT row 115). Sorted names → ascending by update; the result is
full paths.

#### `Latest`

```pascal
function TCheckpoints.Latest: string;
var
  A: TStringArray;
begin
  A := List;
  if Length(A) = 0 then
    Exit('');
  Result := A[High(A)];
end;
```

The newest complete checkpoint; `''` when there is none (T17 resumes from
it, system flow).

#### `Best`

```pascal
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
```

Walks the checkpoints in ascending update order and replaces the candidate
only on a strictly smaller finite loss: NaNs are skipped, and on a tie the
earlier checkpoint stays (external design, "Retention rule"; UT row 113).
`''` when no checkpoint has a finite loss.

#### `Prune`

```pascal
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
```

After each save, the last `Keep` checkpoints by update stay; the one with
the lowest finite validation loss (`Best`) additionally stays when it is
among the rest, and every other `ckpt-*` directory is deleted (external
design, "Retention rule"; UT rows 112, 113). When `High(A) - FKeep < 0`
(the loop bound is negative) nothing is deleted.

#### `Save`

```pascal
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
```

Steps, in exactly this order (the error order is 4.7, #5 with the
optimiser check of #7):

1. `CheckOptim(ASet, AOptim)` — the optimiser must be the one of the set's
   stores (4.5).
2. `Result := DirOf(AState.Update)`; a directory `ckpt-<update>` that
   already exists raises `checkpoint exists: <dir>`.
3. The entries left by an interrupted save are removed before the new one:
   every `tmp-ckpt-*` directory through `RemoveTree`, every `tmp-ckpt-*`
   file through `DeleteFile` (external design, "Save through a temporary
   directory and rename"; UT row 114).
4. `T := FRoot + 'tmp-' + ExtractFileName(Result)` =
   `<root>/tmp-ckpt-<update, 8 digits>`; it is created, `cannot create
   <tmp>` otherwise (e.g. a read-only root, UT row 117).
5. `S := AState` with `Rank` and `Scale` **from the set** (`S.Rank :=
   ASet.Rank; S.Scale := ASet.Scale`) — the state written is the caller's,
   corrected by the set (external design, "Keys of `state.json`").
6. `lora.ot`: `ASet.Save` (tch-fpc errors pass through, 4.7 #8).
7. `moments.ot`: `SaveMoments` (4.5).
8. `state.json`: `WriteText(T + PathDelim + StateFile, RunStateToJson(S))`
   — `state.json` is the last file written into the temporary directory, so
   after the rename a `ckpt-*` directory with a `state.json` is always
   complete (4.6 `List`).
9. `RenameFile(T, Result)`, `cannot rename <tmp>` otherwise (e.g. when a
   **file** named `ckpt-<update>` sits in the root, UT row 116); the
   temporary directory stays until the next save.
10. `Prune` (retention rule).

The returned `Result` is the final checkpoint directory.

#### `Load`

```pascal
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

Steps, in exactly this order (the error order is 4.7, #6 with #3 and the
optimiser check of #7):

1. `Result := ReadRunState(ADir)` — `not a checkpoint: <dir>` when
   `<dir>/state.json` is missing (4.4); the state is returned even when a
   later step raises, and is what a caller of `qlora_merge` needs (T18).
2. `Load` refuses a set of another rank or scale (external design, "Keys of
   `state.json`"): `CreateFmt('%s: rank %d scale %g, the set rank %d scale
   %g', [ADir, Result.Rank, Result.Scale, ASet.Rank, ASet.Scale])`.
3. `CheckOptim(ASet, AOptim)`.
4. `ASet.Load(D + LoraFile)` restores the adapters; `LoadMoments(D +
   MomentFile, ASet, AOptim)` restores the moments into the optimiser's
   moment stores (4.5); tch-fpc errors for a missing file or variable pass
   through unchanged (4.7, #8).
5. `AOptim.StepCount := Result.Update` — the optimiser's step counter is
   part of the resume (the exact-resume acceptance of DESIGN 8.14, UT rows
   104, 118: without it the resumed run diverges).
6. The returned state is the one from `state.json` of `ADir`.

### 4.7 Checks and messages

All raised as `ECheckpoint` unless the row says otherwise, in this order
(exact messages from the T16 facts):

| # | Where | Condition (checked in this order) | Message |
|---|---|---|---|
| 1 | `RunStateToJson` | a character outside `#32..#126` in `config_hash`, then in `base_sha256` | `<key> is not printable ASCII` |
| 2 | `RunStateFromJson` | the text does not parse (`EJSONParser`) or is not an object; then, key by key in the order `update`, `epoch`, `position`, `seed`, `rank`, `scale`, `config_hash`, `base_sha256`, `val_loss`: the key is missing or of the wrong JSON type (an integer key needs an integer number, `5.0` is bad; `scale` a number; the two strings a string; `val_loss` a number or `null`) | `state.json: not a JSON object`; `state.json: missing or bad <key>` |
| 3 | `ReadRunState` (and `Load`) | `<dir>/state.json` does not exist | `not a checkpoint: <dir>` (the directory as given) |
| 4 | `TCheckpoints.Create` | `AKeep < 1`; the root cannot be created | `CreateFmt('keep must be at least 1, got %d', [AKeep])`; `cannot create <root>` (with the delimiter) |
| 5 | `Save`, in this order | the optimiser check (#7); `ckpt-<update>` exists as a directory; the temporary directory cannot be created; the rename fails (e.g. a file named `ckpt-<update>`; the temporary directory stays until the next save) | #7 messages; `checkpoint exists: <dir>`; `cannot create <tmp>`; `cannot rename <tmp>` |
| 6 | `Load`, in this order | `not a checkpoint: <dir>` (#3); `result.Rank <> ASet.Rank` or `result.Scale <> ASet.Scale`; the optimiser check (#7) | #3 message; `CreateFmt('%s: rank %d scale %g, the set rank %d scale %g', [ADir, Result.Rank, Result.Scale, ASet.Rank, ASet.Scale])`; #7 messages |
| 7 | the optimiser check (`CheckOptim`) | the store counts differ; then, per store `I`, the moment counts differ — `2 × variables` expected | `CreateFmt('the optimiser has %d stores, the set %d', [AOptim.StoreCount, Length(St)])`; then `CreateFmt('store %d: the optimiser has %d moments, the set %d variables', [I, AOptim.Moments(I).Count, St[I].Count])` |
| 8 | from tch-fpc, passed on unchanged | a missing `lora.ot` or `moments.ot`; a file without one of the variables; another shape | `ETorchNativeError`: `open file failed because of errno 2 on fopen: No such file or directory, file path: <file>`; `ETorchError`: `variable <name> not found in <file>`; `ETorchNativeError` from the copy |

Notes:

- `Save` raises no `ECheckpoint` for the two missing files: those raise the
  tch-fpc errors of #8 through `ASet.Save`/`SaveMoments`.
- A row in a UT case table that expects an error catches the named
  exception class and compares the whole message; a row fails when no
  exception comes (T16 facts, "Unit tests").

### 4.8 Business rules of the external design and where they are handled

| Business rule (`T16-external.md`) | Handled by |
|---|---|
| `<root>/ckpt-<update, 8 digits>/` holds `state.json`, `lora.ot`, `moments.ot`; lexical order = update order | `DirOf` (4.6), `Save` steps 6–8 (4.6), `List` filter (4.6) |
| save through `tmp-ckpt-<update>/` and rename; `ckpt-*` with `state.json` always complete | `Save` steps 4 and 9 (4.6), `List` (4.6) |
| stale `tmp-ckpt-*` entries are removed by the next save | `Save` step 3 (4.6) |
| retention: last `Keep` by update + lowest finite val loss (earlier on a tie); every other `ckpt-*` deleted; `AKeep ≥ 1`; default 3 | `Create` (4.6), `Best` (4.6), `Prune` (4.6) |
| the 9 keys of `state.json`, numbers round-trip exactly | `RunStateToJson` (4.2), `RunStateFromJson` (4.3) |
| non-finite `val_loss` (NaN, ±Inf) written as `null`, read back as NaN | `RunStateToJson` (`IsNan or IsInfinite` → `TJSONNull`, 4.2), `GetFloat` (`jtNull` → NaN, 4.3) |
| `config_hash` / `base_sha256` printable ASCII, checked before writing | `RunStateToJson` → `CheckAscii` (4.2, 4.7 #1) |
| unknown keys ignored when reading | `RunStateFromJson` reads only the nine keys in order (4.3) |
| the trainer computes the hash and the sha256; T16 only stores them | `RunStateToJson` / `RunStateFromJson` pass the strings through unchanged (4.2, 4.3) |
| `Save` takes rank and scale from the set; `Load` refuses a set of another rank or scale | `Save` step 5 (`S.Rank := ASet.Rank; S.Scale := ASet.Scale`), `Load` step 2 (4.6, 4.7 #5–#6) |
| the optimiser must be the one created over the set's stores (store count, two moments per variable) | `CheckOptim` (4.5, 4.7 #7) |
| the moments are stored by variable name, not by store position | `SaveMoments`, `LoadMoments` (4.5) |
| resume is exact: adapters, moments and `StepCount` restored | `Load` steps 4–5 (4.6) |
| the error list (`T16-external.md`, "Errors") | 4.7 (#1–#8) |

## 5. Change history

| Date | Change | Reason |
|---|---|---|
| T16a (initial) | First edition, from the external design and the T16 facts. | new unit |
