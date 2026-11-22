{ T14: the sample's own AdamW (DESIGN 8.14, "LoraOptim").

  TOptimizer cannot save its state, so the two moments of every trainable
  variable are non-trainable variables of a second TVarStore per parameter
  store (same device and kind; names 'm.<j>' and 'v.<j>', j the index in
  the store's TrainableVariables), saved and loaded with TVarStore.Save /
  Load. The update is PyTorch's AdamW (decoupled weight decay, bias
  correction) under NoGradGuard; variables without a gradient are skipped.
  ClipGlobalNorm clips over all stores together. ScheduleLR gives the
  learning rate per update: linear warm-up, then cosine decay to MinRatio.

  Interface by the harness owner (tasks/T14-facts.md); the bodies are
  written in T14b. }
unit LoraOptim;

{$mode objfpc}{$H+}

interface

uses
  SysUtils, Math, Tch, TchNN;

type
  ELoraOptim = class(Exception);

  TLoraAdamWConfig = record
    Beta1, Beta2, Eps, WeightDecay: Double;
  end;

  TLoraAdamW = class
  private
    FConfig: TLoraAdamWConfig;
    FParams: array of TTensorArray;   // per store: its trainable variables
    FM, FV: array of TTensorArray;    // per store: the moments
    FMoments: array of TVarStore;     // owned, one per store
    FStepCount: Int64;
  public
    { One moment store per parameter store; the stores must already hold
      all their trainable variables. Raises ELoraOptim. }
    constructor Create(const Stores: array of TVarStore; const AConfig: TLoraAdamWConfig);
    destructor Destroy; override;
    procedure ZeroGrad;
    { Scales every gradient by MaxNorm / (norm + 1e-6) when that is below
      1; returns the global norm before clipping. }
    function ClipGlobalNorm(MaxNorm: Double): Double;
    { One AdamW update with this learning rate; StepCount + 1. }
    procedure Step(LR: Double);
    function StoreCount: Integer;
    { The moment store of parameter store I (for Save / Load). }
    function Moments(I: Integer): TVarStore;
    property StepCount: Int64 read FStepCount write FStepCount;
  end;

{ Beta1 0.9, Beta2 0.999, Eps 1e-8, WeightDecay 0.01 }
function DefaultAdamWConfig: TLoraAdamWConfig;
{ Learning rate of update Update (1-based) of Total: Base * Update / Warmup
  while Update <= Warmup, then Base * (MinRatio + (1 - MinRatio) * 0.5 *
  (1 + cos(pi * (Update - Warmup) / (Total - Warmup)))). Raises ELoraOptim. }
function ScheduleLR(Base: Double; Update, Warmup, Total: Int64; MinRatio: Double): Double;

implementation

function DefaultAdamWConfig: TLoraAdamWConfig;
begin
  Result.Beta1 := 0.9;
  Result.Beta2 := 0.999;
  Result.Eps := 1e-8;
  Result.WeightDecay := 0.01;
end;

function ScheduleLR(Base: Double; Update, Warmup, Total: Int64; MinRatio: Double): Double;
var
  P: Double;
begin
  if (Update < 1) or (Update > Total) then
    raise ELoraOptim.CreateFmt('update %d outside 1..%d', [Update, Total]);
  if (Warmup < 0) or (Warmup >= Total) then
    raise ELoraOptim.CreateFmt('warm-up %d outside 0..%d', [Warmup, Total - 1]);
  if Update <= Warmup then
    Exit(Base * Update / Warmup);
  P := (Update - Warmup) / (Total - Warmup);
  Result := Base * (MinRatio + (1 - MinRatio) * 0.5 * (1 + Cos(Pi * P)));
end;

constructor TLoraAdamW.Create(const Stores: array of TVarStore; const AConfig: TLoraAdamWConfig);
var
  I, J: Integer;
  P: TTensorArray;
  Root: TVarPath;
begin
  inherited Create;
  if Length(Stores) = 0 then
    raise ELoraOptim.Create('no stores');
  FConfig := AConfig;
  SetLength(FParams, Length(Stores));
  SetLength(FM, Length(Stores));
  SetLength(FV, Length(Stores));
  SetLength(FMoments, Length(Stores));
  for I := 0 to High(Stores) do
  begin
    P := Stores[I].TrainableVariables;
    if Length(P) = 0 then
      raise ELoraOptim.CreateFmt('store %d has no trainable variables', [I]);
    FParams[I] := P;
    FMoments[I] := TVarStore.Create(Stores[I].Device, Stores[I].Kind);
    Root := FMoments[I].Root;
    SetLength(FM[I], Length(P));
    SetLength(FV[I], Length(P));
    for J := 0 to High(P) do
    begin
      FM[I][J] := Root.Sub('m').ZerosNoTrain(IntToStr(J), P[J].Shape);
      FV[I][J] := Root.Sub('v').ZerosNoTrain(IntToStr(J), P[J].Shape);
    end;
  end;
end;

destructor TLoraAdamW.Destroy;
var
  I: Integer;
begin
  for I := 0 to High(FMoments) do
    FMoments[I].Free;
  inherited Destroy;
end;

procedure TLoraAdamW.ZeroGrad;
var
  I, J: Integer;
begin
  for I := 0 to High(FParams) do
    for J := 0 to High(FParams[I]) do
      FParams[I][J].ZeroGrad;
end;

function TLoraAdamW.ClipGlobalNorm(MaxNorm: Double): Double;
var
  G: IInterface;
  Gr: TTensor;
  I, J: Integer;
  Sq, Coef: Double;
begin
  G := NoGradGuard;
  Sq := 0;
  for I := 0 to High(FParams) do
    for J := 0 to High(FParams[I]) do
    begin
      Gr := FParams[I][J].grad;
      if Gr.Defined then
        Sq := Sq + Sqr(Gr.norm.DoubleValue([]));
    end;
  Result := Sqrt(Sq);
  Coef := MaxNorm / (Result + 1e-6);
  if Coef < 1 then
    for I := 0 to High(FParams) do
      for J := 0 to High(FParams[I]) do
      begin
        Gr := FParams[I][J].grad;
        if Gr.Defined then
          Gr.mul_scalar_(Coef);
      end;
end;

procedure TLoraAdamW.Step(LR: Double);
var
  G: IInterface;
  P, Gr, M, V: TTensor;
  I, J: Integer;
  Bc1, Bc2: Double;
begin
  G := NoGradGuard;
  Inc(FStepCount);
  Bc1 := 1 - Power(FConfig.Beta1, FStepCount);
  Bc2 := 1 - Power(FConfig.Beta2, FStepCount);
  for I := 0 to High(FParams) do
    for J := 0 to High(FParams[I]) do
    begin
      P := FParams[I][J];
      Gr := P.grad;
      if not Gr.Defined then
        Continue;
      M := FM[I][J];
      V := FV[I][J];
      P.mul_scalar_(1 - LR * FConfig.WeightDecay);
      M.mul_scalar_(FConfig.Beta1).add_(Gr * (1 - FConfig.Beta1));
      V.mul_scalar_(FConfig.Beta2).add_(Gr * Gr * (1 - FConfig.Beta2));
      P.add_(M / (V.sqrt / Sqrt(Bc2) + FConfig.Eps) * (-LR / Bc1));
    end;
end;

function TLoraAdamW.StoreCount: Integer;
begin
  Result := Length(FMoments);
end;

function TLoraAdamW.Moments(I: Integer): TVarStore;
begin
  if (I < 0) or (I > High(FMoments)) then
    raise ELoraOptim.CreateFmt('store %d out of range (count %d)', [I, Length(FMoments)]);
  Result := FMoments[I];
end;

end.
