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

function NewRunState: TRunState;
begin
  Result := Default(TRunState);
  raise ECheckpoint.Create('NewRunState: not implemented yet');
end;

function RunStateToJson(const S: TRunState): string;
begin
  Result := '';
  raise ECheckpoint.Create('RunStateToJson: not implemented yet');
end;

function RunStateFromJson(const AText: string): TRunState;
begin
  Result := Default(TRunState);
  raise ECheckpoint.Create('RunStateFromJson: not implemented yet');
end;

function ReadRunState(const ADir: string): TRunState;
begin
  Result := Default(TRunState);
  raise ECheckpoint.Create('ReadRunState: not implemented yet');
end;

constructor TCheckpoints.Create(const ARoot: string; AKeep: Integer);
begin
  inherited Create;
  raise ECheckpoint.Create('TCheckpoints.Create: not implemented yet');
end;

function TCheckpoints.DirOf(AUpdate: Int64): string;
begin
  Result := '';
  raise ECheckpoint.Create('TCheckpoints.DirOf: not implemented yet');
end;

procedure TCheckpoints.RemoveTree(const ADir: string);
begin
  raise ECheckpoint.Create('TCheckpoints.RemoveTree: not implemented yet');
end;

function TCheckpoints.List: TStringArray;
begin
  Result := nil;
  raise ECheckpoint.Create('TCheckpoints.List: not implemented yet');
end;

function TCheckpoints.Latest: string;
begin
  Result := '';
  raise ECheckpoint.Create('TCheckpoints.Latest: not implemented yet');
end;

function TCheckpoints.Best: string;
begin
  Result := '';
  raise ECheckpoint.Create('TCheckpoints.Best: not implemented yet');
end;

procedure TCheckpoints.Prune;
begin
  raise ECheckpoint.Create('TCheckpoints.Prune: not implemented yet');
end;

function TCheckpoints.Save(const AState: TRunState; ASet: TLoraSet; AOptim: TLoraAdamW): string;
begin
  Result := '';
  raise ECheckpoint.Create('TCheckpoints.Save: not implemented yet');
end;

function TCheckpoints.Load(const ADir: string; ASet: TLoraSet; AOptim: TLoraAdamW): TRunState;
begin
  Result := Default(TRunState);
  raise ECheckpoint.Create('TCheckpoints.Load: not implemented yet');
end;

end.
