{ T13: the block-wise backward by recomputation (DESIGN 8.14, "Recompute").

  RecomputeForward runs every stage without a graph (NoGradGuard) and keeps
  each stage's input in a TSavedInputs (on the host or on the stage's
  device). RecomputeBackward goes from the last stage to the first: it
  runs the stage again with a graph from its saved input, back-propagates
  sum(y * g), and passes x.grad on as the next g. Parameter gradients
  accumulate in the stages' own leaves; the result is dL/dX0.

  Independent of Qwen: a stage is anything with a device and a forward.
  Tensors cross devices (&to) inside the two loops only.

  Interface by the harness owner (tasks/T13-facts.md); the bodies are
  written in T13b. }
unit Recompute;

{$mode objfpc}{$H+}

interface

uses
  SysUtils, Tch;

type
  ERecompute = class(Exception);

  { One step of the stack (a trunk block): its device and its forward. }
  TStage = class abstract
  public
    function Device: TDevice; virtual; abstract;
    function Forward(const X: TTensor): TTensor; virtual; abstract;
  end;
  TStageArray = array of TStage;

  { One saved input per stage, detached; on the CPU when OnHost. }
  TSavedInputs = class
  private
    FOnHost: Boolean;
    FItems: array of TTensor;
  public
    constructor Create(AOnHost: Boolean);
    procedure Clear;
    procedure Add(const X: TTensor);
    function Count: Integer;
    function Item(I: Integer): TTensor;
    property OnHost: Boolean read FOnHost;
  end;

{ Runs all stages under NoGradGuard, keeps every stage's input (Saved is
  cleared first), returns the last output. Raises ERecompute. }
function RecomputeForward(const Stages: TStageArray; const X0: TTensor;
  Saved: TSavedInputs): TTensor;

{ From dL/d(last output), per stage from the last to the first:
  x := saved.detach + requires_grad; y := Stage.Forward(x);
  (y * g).sum.Backward; g := x.grad. Returns dL/dX0. Raises ERecompute. }
function RecomputeBackward(const Stages: TStageArray; Saved: TSavedInputs;
  const GradOut: TTensor): TTensor;

implementation

{ '[2, 3]' }
function ShapeStr(const T: TTensor): string;
var
  S: TInt64Array;
  I: Integer;
begin
  S := T.Shape;
  Result := '[';
  for I := 0 to High(S) do
  begin
    if I > 0 then
      Result := Result + ', ';
    Result := Result + IntToStr(S[I]);
  end;
  Result := Result + ']';
end;

constructor TSavedInputs.Create(AOnHost: Boolean);
begin
  inherited Create;
  FOnHost := AOnHost;
end;

procedure TSavedInputs.Clear;
begin
  SetLength(FItems, 0);
end;

procedure TSavedInputs.Add(const X: TTensor);
var
  N: Integer;
begin
  N := Length(FItems);
  SetLength(FItems, N + 1);
  if FOnHost then
    FItems[N] := X.detach.&to(TDevice.Cpu)
  else
    FItems[N] := X.detach;
end;

function TSavedInputs.Count: Integer;
begin
  Result := Length(FItems);
end;

function TSavedInputs.Item(I: Integer): TTensor;
begin
  if (I < 0) or (I >= Count) then
    raise ERecompute.CreateFmt('saved input %d out of range (count %d)', [I, Count]);
  Result := FItems[I];
end;

function RecomputeForward(const Stages: TStageArray; const X0: TTensor;
  Saved: TSavedInputs): TTensor;
var
  G: IInterface;
  X: TTensor;
  I: Integer;
begin
  if Length(Stages) = 0 then
    raise ERecompute.Create('no stages');
  if not X0.Defined then
    raise ERecompute.Create('X0 is undefined');
  Saved.Clear;
  G := NoGradGuard;
  X := X0;
  for I := 0 to High(Stages) do
  begin
    X := X.&to(Stages[I].Device);
    Saved.Add(X);
    X := Stages[I].Forward(X);
  end;
  Result := X;
end;

function RecomputeBackward(const Stages: TStageArray; Saved: TSavedInputs;
  const GradOut: TTensor): TTensor;
var
  G, X, Y: TTensor;
  I: Integer;
begin
  if Saved.Count <> Length(Stages) then
    raise ERecompute.CreateFmt('saved inputs: %d, stages: %d',
      [Saved.Count, Length(Stages)]);
  if not GradOut.Defined then
    raise ERecompute.Create('GradOut is undefined');
  G := GradOut;
  for I := High(Stages) downto 0 do
  begin
    X := Saved.Item(I).&to(Stages[I].Device).detach.requires_grad_(True);
    Y := Stages[I].Forward(X);
    G := G.&to(Stages[I].Device);
    if ShapeStr(Y) <> ShapeStr(G) then
      raise ERecompute.CreateFmt('stage %d: output %s, gradient %s',
        [I, ShapeStr(Y), ShapeStr(G)]);
    (Y * G).sum.Backward;
    G := X.grad;
  end;
  Result := G;
end;

end.
