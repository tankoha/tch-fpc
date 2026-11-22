{ T13 UT/ITa fixture (harness owner's): small test stages and the
  reference whole-graph backward. Not code under test. }
unit RecomputeFixture;

{$mode objfpc}{$H+}

interface

uses
  SysUtils, Math, Tch, Recompute;

type
  { y = x * w; w [1] a leaf that requires grad (kFloat), on the CPU. For
    hand-computed rows: dL/dw = sum(x * g), dL/dx = w * g. }
  TScaleStage = class(TStage)
  public
    W: TTensor;
    Calls: Integer;
    LastRequiresGrad: Boolean;
    constructor Create(AW: Double);
    function Device: TDevice; override;
    function Forward(const X: TTensor): TTensor; override;
  end;

  { y = tanh(x W + b); W [D, D], b [D] leaves that require grad. Counts
    its forwards and whether the last one built a graph. }
  TAffineStage = class(TStage)
  private
    FDevice: TDevice;
  public
    W, B: TTensor;
    Calls: Integer;
    LastRequiresGrad: Boolean;
    constructor Create(ADim: Int64; AKind: TKind; ADevice: TDevice);
    function Device: TDevice; override;
    function Forward(const X: TTensor): TTensor; override;
    { The same parameter values (copied, new leaves) on ADevice in AKind. }
    function CloneTo(AKind: TKind; ADevice: TDevice): TAffineStage;
  end;

{ N stages of width D, parameters from ManualSeed(Seed); stage I on
  Devices[I * Length(Devices) div N]. }
function MakeStages(Seed: Int64; N: Integer; D: Int64; AKind: TKind;
  const Devices: array of TDevice): TStageArray;
function CloneStages(const S: TStageArray; AKind: TKind;
  const Devices: array of TDevice): TStageArray;
procedure FreeStages(var S: TStageArray);
{ Reference: one ordinary forward with graph and one backward of
  sum(y * GradOut); returns dL/dX0 (X0 is copied into a new leaf). }
function WholeGraphBackward(const S: TStageArray; const X0, GradOut: TTensor): TTensor;
{ Max |a - b| over all elements, both moved to the CPU as Double. }
function MaxAbsDiff(const A, B: TTensor): Double;
{ Max over stages of MaxAbsDiff of W.grad and of b.grad. }
function MaxParamGradDiff(const A, B: TStageArray): Double;
function ShapeText(const T: TTensor): string;
{ Values as FloatToStr, separated by one space ('6 12'). }
function ValuesText(const T: TTensor): string;

implementation

constructor TScaleStage.Create(AW: Double);
begin
  inherited Create;
  W := TTensor.FromArray([AW]).requires_grad_(True);
end;

function TScaleStage.Device: TDevice;
begin
  Result := TDevice.Cpu;
end;

function TScaleStage.Forward(const X: TTensor): TTensor;
begin
  Inc(Calls);
  Result := X * W;
  LastRequiresGrad := Result.RequiresGrad;
end;

function ValuesText(const T: TTensor): string;
var
  D: TDoubleArray;
  I: Integer;
begin
  D := T.to_device(TDevice.Cpu, kDouble).reshape([-1]).ToDoubleArray;
  Result := '';
  for I := 0 to High(D) do
  begin
    if I > 0 then
      Result := Result + ' ';
    Result := Result + FloatToStr(D[I]);
  end;
end;

constructor TAffineStage.Create(ADim: Int64; AKind: TKind; ADevice: TDevice);
begin
  inherited Create;
  FDevice := ADevice;
  W := (TTensor.randn([ADim, ADim], AKind) / Sqrt(ADim)).to_device(ADevice, AKind).requires_grad_(True);
  B := (TTensor.randn([ADim], AKind) * 0.1).to_device(ADevice, AKind).requires_grad_(True);
end;

function TAffineStage.Device: TDevice;
begin
  Result := FDevice;
end;

function TAffineStage.Forward(const X: TTensor): TTensor;
begin
  Inc(Calls);
  Result := (X.matmul(W) + B).tanh;
  LastRequiresGrad := Result.RequiresGrad;
end;

function TAffineStage.CloneTo(AKind: TKind; ADevice: TDevice): TAffineStage;
begin
  Result := TAffineStage.Create(1, AKind, ADevice);
  Result.W := W.detach.to_device(ADevice, AKind).Copy.requires_grad_(True);
  Result.B := B.detach.to_device(ADevice, AKind).Copy.requires_grad_(True);
end;

function DeviceFor(I, N: Integer; const Devices: array of TDevice): TDevice;
begin
  Result := Devices[I * Length(Devices) div N];
end;

function MakeStages(Seed: Int64; N: Integer; D: Int64; AKind: TKind;
  const Devices: array of TDevice): TStageArray;
var
  I: Integer;
begin
  Result := nil;
  ManualSeed(Seed);
  SetLength(Result, N);
  for I := 0 to N - 1 do
    Result[I] := TAffineStage.Create(D, AKind, DeviceFor(I, N, Devices));
end;

function CloneStages(const S: TStageArray; AKind: TKind;
  const Devices: array of TDevice): TStageArray;
var
  I: Integer;
begin
  Result := nil;
  SetLength(Result, Length(S));
  for I := 0 to High(S) do
    Result[I] := (S[I] as TAffineStage).CloneTo(AKind, DeviceFor(I, Length(S), Devices));
end;

procedure FreeStages(var S: TStageArray);
var
  I: Integer;
begin
  for I := 0 to High(S) do
    S[I].Free;
  SetLength(S, 0);
end;

function WholeGraphBackward(const S: TStageArray; const X0, GradOut: TTensor): TTensor;
var
  X, Leaf: TTensor;
  I: Integer;
begin
  Leaf := X0.detach.to_device(S[0].Device, X0.Kind).Copy.requires_grad_(True);
  X := Leaf;
  for I := 0 to High(S) do
    X := S[I].Forward(X.&to(S[I].Device));
  (X * GradOut.&to(X.Device)).sum.Backward;
  Result := Leaf.grad;
end;

function MaxAbsDiff(const A, B: TTensor): Double;
begin
  Result := (A.to_device(TDevice.Cpu, kDouble) - B.to_device(TDevice.Cpu, kDouble)).abs.max.DoubleValue([]);
end;

function MaxParamGradDiff(const A, B: TStageArray): Double;
var
  I: Integer;
begin
  Result := 0;
  for I := 0 to High(A) do
  begin
    Result := Max(Result, MaxAbsDiff((A[I] as TAffineStage).W.grad, (B[I] as TAffineStage).W.grad));
    Result := Max(Result, MaxAbsDiff((A[I] as TAffineStage).B.grad, (B[I] as TAffineStage).B.grad));
  end;
end;

function ShapeText(const T: TTensor): string;
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

end.
