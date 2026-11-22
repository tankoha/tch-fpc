{ T14 UT/ITa fixture (harness owner's): a toy least-squares problem and
  the reference run with tch-fpc's TOptimizer.AdamW (libtorch's AdamW).
  Not code under test. }
unit LoraOptimFixture;

{$mode objfpc}{$H+}

interface

uses
  SysUtils, Tch, TchNN;

{ ManualSeed(Seed); X := randn([16, 4]), Y := randn([16, 3]) in AKind. }
procedure MakeToyData(Seed: Int64; AKind: TKind; out X, Y: TTensor);
{ A CPU store of AKind with 'w' [4, 3] (ManualSeed(Seed), RandnStandard)
  and 'b' [3] (zeros), both trainable. Equal seeds give equal values. }
function MakeToyStore(Seed: Int64; AKind: TKind; out W, B: TTensor): TVarStore;
{ mean((X W + B - Y)^2) }
function ToyLoss(const W, B, X, Y: TTensor): TTensor;
{ The oracle: Steps updates of TOptimizer.AdamW(LR, 0.9, 0.999, WD, 1e-8)
  on MakeToyStore(StoreSeed) with MakeToyData(DataSeed); per update
  ZeroGrad, backward, ClipGradNorm(Clip) when Clip > 0, Step. Returns
  copies of the final w and b. }
procedure ReferenceRun(StoreSeed, DataSeed: Int64; AKind: TKind; Steps: Integer;
  LR, WD, Clip: Double; out W, B: TTensor);
{ Max |a - b| over all elements, as Double on the CPU. }
function MaxAbsDiff(const A, B: TTensor): Double;
{ Values as FloatToStr, separated by one space ('0.05 0.1'). }
function ValuesText(const T: TTensor): string;

implementation

procedure MakeToyData(Seed: Int64; AKind: TKind; out X, Y: TTensor);
begin
  ManualSeed(Seed);
  X := TTensor.randn([16, 4], AKind);
  Y := TTensor.randn([16, 3], AKind);
end;

function MakeToyStore(Seed: Int64; AKind: TKind; out W, B: TTensor): TVarStore;
begin
  ManualSeed(Seed);
  Result := TVarStore.Create(TDevice.Cpu, AKind);
  W := Result.Root.RandnStandard('w', [4, 3]);
  B := Result.Root.Zeros('b', [3]);
end;

function ToyLoss(const W, B, X, Y: TTensor): TTensor;
begin
  Result := ((X.matmul(W) + B - Y) ** 2).mean(kNone);
end;

procedure ReferenceRun(StoreSeed, DataSeed: Int64; AKind: TKind; Steps: Integer;
  LR, WD, Clip: Double; out W, B: TTensor);
var
  S: TVarStore;
  Opt: TOptimizer;
  X, Y, SW, SB: TTensor;
  I: Integer;
begin
  S := MakeToyStore(StoreSeed, AKind, SW, SB);
  MakeToyData(DataSeed, AKind, X, Y);
  Opt := TOptimizer.AdamW(S, LR, 0.9, 0.999, WD, 1e-8);
  try
    for I := 1 to Steps do
    begin
      Opt.ZeroGrad;
      ToyLoss(SW, SB, X, Y).Backward;
      if Clip > 0 then
        Opt.ClipGradNorm(Clip);
      Opt.Step;
    end;
    W := SW.detach.Copy;
    B := SB.detach.Copy;
  finally
    Opt.Free;
    S.Free;
  end;
end;

function MaxAbsDiff(const A, B: TTensor): Double;
begin
  Result := (A.to_device(TDevice.Cpu, kDouble) - B.to_device(TDevice.Cpu, kDouble)).abs.max.DoubleValue([]);
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

end.
