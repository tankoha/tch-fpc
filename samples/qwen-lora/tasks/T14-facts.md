# T14 facts (checked by the harness owner)

T14 is the unit `LoraOptim`: the sample's own AdamW (DESIGN 8.14,
"LoraOptim"). `TOptimizer` cannot save its state, so the two moments of
every trainable variable live as non-trainable variables in a second
`TVarStore` per parameter store and are saved with `TVarStore.Save`. The
update is PyTorch's AdamW (decoupled weight decay, bias correction) under
`NoGradGuard`; `ClipGlobalNorm` clips over all stores together;
`ScheduleLR` gives the learning rate of each update (linear warm-up, then
cosine decay). T16 (checkpoints) and T17 (the training step) use it.

Measured by the harness owner with a prototype written exactly as below,
on the CPU (GPUs hidden): 20 updates on a toy problem follow
`TOptimizer.AdamW` (libtorch's) within 1.2e-7 in fp32 and 1e-15 in fp64,
also with clipping and with the schedule; two stores clip and update
exactly as one store with the same variables; save, load and resume give
exactly the uninterrupted result. Every row of "Unit tests" gives the
value in its row; the messages are the prototype's. These facts are
final; do not investigate further or write probes.

## Unit `LoraOptim` (interface, final; in the skeleton)

```pascal
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
    constructor Create(const Stores: array of TVarStore; const AConfig: TLoraAdamWConfig);
    destructor Destroy; override;
    procedure ZeroGrad;
    function ClipGlobalNorm(MaxNorm: Double): Double;
    procedure Step(LR: Double);
    function StoreCount: Integer;
    function Moments(I: Integer): TVarStore;
    property StepCount: Int64 read FStepCount write FStepCount;
  end;

function DefaultAdamWConfig: TLoraAdamWConfig;
function ScheduleLR(Base: Double; Update, Warmup, Total: Int64; MinRatio: Double): Double;
```

## Implementation (write it exactly so; the parity rows depend on it)

`DefaultAdamWConfig`: `Beta1 := 0.9; Beta2 := 0.999; Eps := 1e-8;
WeightDecay := 0.01`.

`ScheduleLR` (after its checks; `P: Double`):

```pascal
  if Update <= Warmup then
    Exit(Base * Update / Warmup);
  P := (Update - Warmup) / (Total - Warmup);
  Result := Base * (MinRatio + (1 - MinRatio) * 0.5 * (1 + Cos(Pi * P)));
```

`Create` (`inherited Create` first, then its checks in the loop; `P:
TTensorArray`, `Root: TVarPath`):

```pascal
  FConfig := AConfig;
  SetLength(FParams, Length(Stores));
  SetLength(FM, Length(Stores));
  SetLength(FV, Length(Stores));
  SetLength(FMoments, Length(Stores));
  for I := 0 to High(Stores) do
  begin
    P := Stores[I].TrainableVariables;
    { check: no trainable variables }
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
```

The moment names are therefore `m.0`, `v.0`, `m.1`, `v.1`, ... in this
order. `Destroy`: free every `FMoments[I]`, then `inherited Destroy`.
`ZeroGrad`: `ZeroGrad` of every `FParams[I][J]`. `StoreCount`:
`Length(FMoments)`. `Moments(I)`: its check, then `FMoments[I]`.

`ClipGlobalNorm` (`G: IInterface; Gr: TTensor; Sq, Coef: Double`):

```pascal
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
```

`Step` (`G: IInterface; P, Gr, M, V: TTensor; Bc1, Bc2: Double`):

```pascal
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
```

(`Power` is from unit `Math`, already in the `uses`.)

## Checks and messages (ELoraOptim; in this order)

- `ScheduleLR`: 1. `(Update < 1) or (Update > Total)`:
  `CreateFmt('update %d outside 1..%d', [Update, Total])`; 2.
  `(Warmup < 0) or (Warmup >= Total)`:
  `CreateFmt('warm-up %d outside 0..%d', [Warmup, Total - 1])`.
- `Create`: 1. `Length(Stores) = 0`: `no stores`; 2. in the loop, a store
  without trainable variables:
  `CreateFmt('store %d has no trainable variables', [I])`.
- `Moments(I)` with `I < 0` or `I > High(FMoments)`:
  `CreateFmt('store %d out of range (count %d)', [I, Length(FMoments)])`.

## UT fixture (`tests/ut/LoraOptimFixture.pas`, harness owner's)

Used by the UT and the ITa; nobody changes it.
- `MakeToyData(Seed, Kind, X, Y)`: `ManualSeed(Seed)`, X [16, 4], Y [16, 3].
- `MakeToyStore(Seed, Kind, W, B)`: a CPU store with trainable `w` [4, 3]
  (random from the seed) and `b` [3] (zeros); equal seeds, equal values.
- `ToyLoss(W, B, X, Y)`: mean((X W + B − Y)²).
- `ReferenceRun(StoreSeed, DataSeed, Kind, Steps, LR, WD, Clip, W, B)`:
  the oracle, `TOptimizer.AdamW(LR, 0.9, 0.999, WD, 1e-8)` on the toy
  problem (per update `ZeroGrad`, backward, `ClipGradNorm(Clip)` when
  Clip > 0, `Step`); returns copies of the final w and b.
- `MaxAbsDiff(A, B)`, `ValuesText(T)` (`FloatToStr` of each value, single
  spaces: `'0.05'`).

The test skeleton `tests/ut/LoraOptim_test.pas` has `OneVarStore(N,
AOnes, P)`: a CPU kDouble store with one trainable variable `p` [N] of
ones (AOnes) or zeros. Each test frees what it creates (optimizer first,
then the stores) in `try ... finally`.

## Unit tests (rows; every row has one method, every method belongs to a row)

A tolerance row passes when the difference is ≤ the limit
(`AssertEquals(Expected, Actual, Delta)` for a Double, or
`AssertTrue(MaxAbsDiff(..) <= limit)`); "exact" means the exact strings or
a difference of 0. Error rows catch `ELoraOptim` and compare the whole
message (`AssertEquals`); a row fails if no exception comes
(`Fail('no ELoraOptim')`). `Cfg` is `DefaultAdamWConfig` unless the row
says otherwise.

Stage 1, `DefaultAdamWConfig` and `ScheduleLR` (001–010):

| Row | Input | Expected |
|---|---|---|
| 001 | `DefaultAdamWConfig` | Beta1 0.9, Beta2 0.999, Eps 1e-8, WeightDecay 0.01 (exact) |
| 002 | `ScheduleLR(1e-4, 1, 50, 1000, 0.1)` | 2e-6 (delta 1e-15) |
| 003 | `ScheduleLR(1e-4, 50, 50, 1000, 0.1)` | 1e-4 (delta 1e-15) |
| 004 | `ScheduleLR(1e-4, 525, 50, 1000, 0.1)` | 5.5e-5 (delta 1e-15) |
| 005 | `ScheduleLR(1e-4, 1000, 50, 1000, 0.1)` | 1e-5 (delta 1e-15) |
| 006 | `ScheduleLR(1e-4, 1, 0, 3, 0.1)` (no warm-up) | 7.75e-5 (delta 1e-15) |
| 007 | `ScheduleLR(1e-4, 0, 50, 1000, 0.1)` | `update 0 outside 1..1000` |
| 008 | `ScheduleLR(1e-4, 1001, 50, 1000, 0.1)` | `update 1001 outside 1..1000` |
| 009 | `ScheduleLR(1e-4, 10, 1000, 1000, 0.1)` | `warm-up 1000 outside 0..999` |
| 010 | `ScheduleLR(1e-4, 10, -1, 1000, 0.1)` | `warm-up -1 outside 0..999` |

Stage 2, `Create`, the moment stores, `ZeroGrad` (011–016):

| Row | Input | Expected |
|---|---|---|
| 011 | S1 := `MakeToyStore(3, kDouble, W, B)`; S2 := `TVarStore.Create(TDevice.Cpu, kFloat)` with `S2.Root.Zeros('c', [2])`; `TLoraAdamW.Create([S1, S2], Cfg)` | `StoreCount` 2; `Moments(0).Count` 4, `Moments(1).Count` 2; `Moments(0).Variables[0..3].Name` `m.0`, `v.0`, `m.1`, `v.1`; `Length(Moments(0).TrainableVariables)` 0; `Moments(0).Kind` kDouble, `Moments(1).Kind` kFloat; `Moments(0).Variables[0].Tensor.abs.sum(kNone)` 0; `StepCount` 0 |
| 012 | `TLoraAdamW.Create([], Cfg)` | `no stores` |
| 013 | S1 as 011; S2 a kDouble store with only `S2.Root.ZerosNoTrain('z', [2])`; `Create([S1, S2], Cfg)` | `store 1 has no trainable variables` |
| 014 | S1 as 011, optimizer on it; `MakeToyData(4, kDouble, X, Y)`; `ToyLoss(W, B, X, Y).Backward`; `ZeroGrad` | before `ZeroGrad` `W.grad.abs.sum(kNone)` > 0; after it `ValuesText` of `W.grad.abs.sum(kNone)` and of `B.grad.abs.sum(kNone)` `'0'` |
| 015 | the optimizer of 011; `Moments(2)` | `store 2 out of range (count 2)` |
| 016 | the optimizer of 011; `Moments(-1)` | `store -1 out of range (count 2)` |

Stage 3, `Step` and `ClipGlobalNorm` (101–112):

| Row | Input | Expected |
|---|---|---|
| 101 | `OneVarStore(1, True, P)`; Cfg with `Eps := 0`; `ZeroGrad`; `(P * 0.5).sum(kNone).Backward`; `Step(0.1)` | `P.DoubleValue([0])` 0.899 (delta 1e-12); `ValuesText` of `Moments(0).Variables[0].Tensor` (m) `'0.05'`, of `Variables[1]` (v) `'0.00025'`; `StepCount` 1 |
| 102 | as 101, then the same `ZeroGrad`, backward, `Step(0.1)` again | P 0.798101 (delta 1e-12); m `'0.095'`, v `'0.00049975'`; `StepCount` 2 |
| 103 | `MakeToyStore(5, kFloat, W, B)`, `MakeToyData(6, kFloat, X, Y)`; 20 × (`ZeroGrad`, `ToyLoss(..).Backward`, `Step(0.05)`); `ReferenceRun(5, 6, kFloat, 20, 0.05, 0.01, 0, RW, RB)` | `MaxAbsDiff(W, RW)` and `(B, RB)` ≤ 1e-6 (measured 8.9e-8, 1.0e-7); `StepCount` 20 |
| 104 | as 103 in kDouble | ≤ 1e-12 (measured 8.3e-16, 1.9e-16) |
| 105 | `MakeToyStore(7, kDouble, W, B)` and U := `S.Root.Ones('u', [2])` (trainable, not in the loss); `MakeToyData(8, kDouble, X, Y)`; one update with `Step(0.1)` | `ValuesText(U)` `'1 1'` (no grad: skipped, also no weight decay); `Moments(0).Variables[4].Name` `m.2`, its values `'0 0'`, `Variables[5]` (v.2) `'0 0'` |
| 106 | `OneVarStore(1, True, P)`; backward of `(P * 0.5).sum(kNone)`; `Step(0)` | `ValuesText(P)` `'1'`; m `'0.05'`; `StepCount` 1 |
| 107 | `OneVarStore(2, False, P)`; backward of `(P * TTensor.FromArray([3.0, 4], kDouble)).sum(kNone)`; `ClipGlobalNorm(1)` | result 5 (delta 1e-12); `MaxAbsDiff(P.grad, FromArray([0.6, 0.8], kDouble))` ≤ 1e-6 (measured 1.6e-7: the factor is 1 / (5 + 1e-6)) |
| 108 | as 107 with `ClipGlobalNorm(10)` | result 5; `ValuesText(P.grad)` `'3 4'` (unchanged) |
| 109 | two kDouble stores, `p` [1] in the first and `c` [1] in the second (`Root.Zeros`); `Create([S1, S2], Cfg)`; backward of `(P * 3.0 + C * 4.0).sum(kNone)`; `ClipGlobalNorm(1)` | result 5; P.grad 0.6 and C.grad 0.8, each within 1e-6 |
| 110 | `MakeToyStore(9, kFloat, ..)`, `MakeToyData(10, kFloat, ..)`; 20 × (`ZeroGrad`, backward, `ClipGlobalNorm(0.5)`, `Step(0.05)`); `ReferenceRun(9, 10, kFloat, 20, 0.05, 0.01, 0.5, RW, RB)` | ≤ 1e-6 (measured 1.8e-7, 6.0e-8) |
| 111 | SU := `MakeToyStore(11, kDouble, W, B)`; S2, S3 kDouble stores: `ManualSeed(11)`, `W2 := S2.Root.RandnStandard('w', [4, 3])`, `B2 := S3.Root.Zeros('b', [3])`; `MakeToyData(12, kDouble, X, Y)`; optimizers on `[SU]` and `[S2, S3]`; 20 × (each: `ZeroGrad`, backward, `ClipGlobalNorm(0.5)`, `Step(0.05)`) | exact: `MaxAbsDiff(W, W2)` and `(B, B2)` 0 |
| 112 | `OneVarStore(2, True, P)`; no backward; `ClipGlobalNorm(1)`, then `Step(0.1)` | result 0; `ValuesText(P)` `'1 1'`; `StepCount` 1 |

Build and run (the GPUs hidden as in the gate):
`fpc -Mobjfpc -Sh -vew -l- -Fu{proj}/src -Fu{proj}/../../src -Fu{proj}/tests/ut -FU{proj}/work/bin -Fl{proj}/../../build/libtch-rocm -k-rpath={proj}/../../build/libtch-rocm {proj}/tests/ut/LoraOptim_test.pas -o{proj}/work/bin/LoraOptim_test`
then `HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 {proj}/work/bin/LoraOptim_test --all --format=plain`.
Expected compiler warnings from `TchNN` (`generics.dictionaries.inc ...
Constructing a class "TCustomDictionaryEnumerator..." with abstract
method`) and libtorch's warnings about the hidden GPUs on stderr are not
errors; ignore them.

## ITa (prepared by the harness owner)

`{proj}/tests/ita/T14-run.sh` (no variable needed) builds
`tests/ita/T14/loraoptim_check.pas` against `src/`, runs it on the CPU
from a fresh directory outside the project and writes
`{proj}/work/ita/T14/summary.tsv` (case, OK/NG, value, limit, note; then
the driver's exit code and the count of stderr lines other than the GPU
warnings). OK when value ≤ limit. A few seconds. The oracle is
`TOptimizer.AdamW` / `ClipGradNorm` (libtorch), outside the unit.

| Case | Compared | Limit |
|---|---|---|
| ITa-T14-001 | fp32, 20 updates, lr 0.05, wd 0.01, against `TOptimizer.AdamW` | 1e-6 |
| ITa-T14-002 | fp64, 200 updates, against `TOptimizer.AdamW` | 1e-12 |
| ITa-T14-003 | fp32, 20 updates with clipping at 0.5, against `ClipGradNorm` + AdamW | 1e-6 |
| ITa-T14-004 | fp64, 50 updates with clipping: two stores (w; b) against one store (w, b) | 0 |
| ITa-T14-005 | fp32, 100 updates with `ScheduleLR(0.05, u, 10, 100, 0.1)` on both sides | 1e-6 |
| ITa-T14-006 | fp32: 10 updates against 5 + save (store and `Moments(0)`) + load into a new store and optimizer + `StepCount := 5` + 5 | 0 |
| ITa-T14-007 | fp64: gradients after `ClipGlobalNorm(0.1)` against `ClipGradNorm(0.1)`, and the returned norm | 1e-12 |

Measured with the prototype: 1.2e-7, 1.1e-16, 2.4e-7, 0, 6.0e-8, 0, 0.
With the skeleton every case is NG; mutants are caught (no bias
correction: 001–003, 005; weight decay added to the gradient instead of
decoupled: 001–003, 005; a wrong clipping norm: 007). Cases 101–102 (GPU)
are the harness owner's (`T14-run.sh gpu`); agents do not run them.

## Stages and case IDs

- Stage 1 (task T14b1): `DefaultAdamWConfig`, `ScheduleLR`; rows
  UT-LoraOptim-001 to 010.
- Stage 2 (task T14b2): `Create`, `Destroy`, `ZeroGrad`, `StoreCount`,
  `Moments`; rows UT-LoraOptim-011 to 016.
- Stage 3 (task T14b3): `ClipGlobalNorm`, `Step`; rows UT-LoraOptim-101
  to 112.
- ITa: ITa-T14-001 to 007.
