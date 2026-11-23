{$mode objfpc}{$H+}
{ Checkpoints of a LoRA run (DESIGN 8.14 "Checkpoint", T16).
  Purpose:
    Saves and restores everything a run needs to continue exactly: the
    adapters (TLoraSet.Save), the AdamW moments and the run state
    (state.json). Adapters and moments are stored by variable name through
    a store on the CPU, so a checkpoint does not depend on the device map.
  Layout:
    <root>/ckpt-<update, 8 digits>/ with state.json, lora.ot and
    moments.ot (<variable>.adam_m and .adam_v). Written into <root>/tmp-ckpt-<update>/ and renamed, so
    a directory named ckpt-* is always complete; tmp-ckpt-* entries left
    by an interrupted save are removed by the next save.
  Retention:
    The last Keep checkpoints by update and the one with the lowest finite
    validation loss are kept; every other is deleted after each save.
  Interface by the harness owner (tasks/T16-facts.md); the bodies are
  written in T16b. }
unit Checkpoint;

interface

uses
  SysUtils, Classes, Math, fpjson, jsonparser, Tch, TchNN, Lora, LoraOptim;

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

end.
