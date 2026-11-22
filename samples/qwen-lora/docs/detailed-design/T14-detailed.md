# Detailed Design — Unit `LoraOptim` (T14)

Source of truth: `../external-design/T14-external.md` and the T14 facts
(`tasks/T14-facts.md`). This document restates and organises them; it adds no
new requirements.

## 1. Module composition

- Source file: `src/LoraOptim.pas`, `{$mode objfpc}{$H+}`.
- Units used: `Tch`, `TchNN`, `Math` (`Power` is from `Math`, already in the
  `uses`).
- Declared:
  - `ELoraOptim = class(Exception)` — the unit's error class.
  - `TLoraAdamWConfig` — record with `Beta1, Beta2, Eps, WeightDecay: Double`.
  - `TLoraAdamW` — the optimizer class; owned moment stores in `FMoments`,
    per-store parameter lists in `FParams`, per-store moment lists in `FM`
    (first moment) and `FV` (second moment), `FStepCount`.
  - `function DefaultAdamWConfig: TLoraAdamWConfig`
  - `function ScheduleLR(Base: Double; Update, Warmup, Total: Int64; MinRatio: Double): Double`

## 2. Function specifications

| Function | Arguments | Returns | Errors | Preconditions |
|---|---|---|---|---|
| `DefaultAdamWConfig` | — | the default `TLoraAdamWConfig` | — | — |
| `ScheduleLR` | `Base: Double; Update, Warmup, Total: Int64; MinRatio: Double` | learning rate for update `Update` | `ELoraOptim` (section 6) | — |
| `Create` | `const Stores: array of TVarStore; const AConfig: TLoraAdamWConfig` | new optimizer | `ELoraOptim` (section 6) | each store alive |
| `Destroy` | — | — | — | — |
| `ZeroGrad` | — | — | — | created |
| `ClipGlobalNorm` | `MaxNorm: Double` | the global norm before clipping | — | gradients defined from a backward |
| `Step` | `LR: Double` | — | — | gradients defined; `StepCount` current |
| `StoreCount` | — | number of parameter stores | — | — |
| `Moments(I)` | `I: Integer` | the `I`-th moment store (owned by the optimizer) | `ELoraOptim` (section 6) | — |
| `StepCount` (property) | — | `Int64` | — | readable and writable |

`TTensor` is a managed record: it is never freed by this unit. `FMoments[I]`
is the owned object and is freed in `Destroy`.

## 3. Data structures

- `TLoraAdamWConfig`: `Beta1, Beta2, Eps, WeightDecay: Double`.
- `TLoraAdamW` fields (private):
  - `FConfig: TLoraAdamWConfig`
  - `FParams: array of TTensorArray` — per store: its trainable variables
  - `FM: array of TTensorArray` — per store: the first moments
  - `FV: array of TTensorArray` — per store: the second moments
  - `FMoments: array of TVarStore` — owned, one per store
  - `FStepCount: Int64`
- Moment store layout: for store `I` with `N` trainable variables (indices
  `0..N-1`), the moment store's variables are created in this order —
  `FM[I][0] = m.0`, `FV[I][0] = v.0`, `FM[I][1] = m.1`, `FV[I][1] = v.1`,
  … up to `m.(N-1)`, `v.(N-1)`. All moment variables are non-trainable
  (`ZerosNoTrain`), with the same shape as the parameter they belong to.
- Moment stores: one `TVarStore` per parameter store, created with the
  parameter store's `Device` and `Kind` (e.g. kDouble stays kDouble, kFloat
  stays kFloat). They are saved and loaded through `TVarStore.Save` /
  `TVarStore.Load` (that is T16's job, not this unit's).

## 4. Processing detail

### 4.1 `DefaultAdamWConfig` and `ScheduleLR`

`DefaultAdamWConfig` (set the four fields, in this order):

```pascal
Beta1 := 0.9; Beta2 := 0.999; Eps := 1e-8; WeightDecay := 0.01
```

`ScheduleLR` (after its checks, section 6, with a local `P: Double`):

```pascal
if Update <= Warmup then
  Exit(Base * Update / Warmup);
P := (Update - Warmup) / (Total - Warmup);
Result := Base * (MinRatio + (1 - MinRatio) * 0.5 * (1 + Cos(Pi * P)));
```

- `Update = 1` and `Warmup = 0`: the `if` is false (`0 <= 0` is false), so
  the decay formula is used with `P = 1/Total`.
- `Update = Warmup`: returns `Base` (full learning rate at the end of warm-up).
- `Update = Total`: `P = 1`, `Cos(Pi) = -1`, result `Base * MinRatio`.

### 4.2 `Create`, `Destroy`, `ZeroGrad`, `StoreCount`, `Moments`

`Create` (`inherited Create` first, then its checks, section 6; locals `P:
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

Notes:

- The check "no trainable variables in store `I`" is raised inside the loop,
  before `FMoments[I]` is created; its message is in section 6.
- The moment names are therefore `m.0`, `v.0`, `m.1`, `v.1`, … in this order.
- `Destroy`: free every `FMoments[I]`, then `inherited Destroy`.
- `ZeroGrad`: `ZeroGrad` of every `FParams[I][J]`.
- `StoreCount`: `Length(FMoments)`.
- `Moments(I)`: its check (section 6), then return `FMoments[I]`.

### 4.3 `ClipGlobalNorm` (locals `G: IInterface; Gr: TTensor; Sq, Coef: Double`)

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

Notes:

- The norm is computed over **all stores together** (that is why the unit
  must own one `Step` and one `ClipGlobalNorm` over several stores);
  variables without a defined gradient are skipped in both passes.
- `Result` is the norm before clipping and is always returned.
- Gradients are scaled in place only when `Coef < 1` (i.e. the norm exceeds
  `MaxNorm`); the guard keeps the in-place updates out of autograd.

### 4.4 `Step` (locals `G: IInterface; P, Gr, M, V: TTensor; Bc1, Bc2: Double`)

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

Notes:

- `FStepCount` is incremented once per `Step`, before the bias corrections;
  the bias corrections use the count **after** the increment, and `Moments`
  of a parameter that was never updated are not touched (no gradient →
  `Continue`), so a parameter with no gradient keeps its value, its moments
  at zero, and receives no weight decay either.
- Weight decay is decoupled: it multiplies the parameter itself, it is not
  added to the gradient.
- Parity: measured against `TOptimizer.AdamW` (libtorch) on the toy problem
  in the UT fixture (`ReferenceRun`): within 1.2e-7 in fp32 and 1e-15 in
  fp64, also with clipping and the schedule; two stores update exactly the
  same as one store with the same variables; save, load and resume give the
  uninterrupted result.

### 4.5 Checks and messages (all `ELoraOptim`, in this order)

| # | Where | Condition (checked in this order) | Message |
|---|---|---|---|
| 1 | `ScheduleLR` | `(Update < 1) or (Update > Total)` | `CreateFmt('update %d outside 1..%d', [Update, Total])` |
| 2 | `ScheduleLR` | `(Warmup < 0) or (Warmup >= Total)` | `CreateFmt('warm-up %d outside 0..%d', [Warmup, Total - 1])` |
| 3 | `Create` | `Length(Stores) = 0` | `no stores` |
| 4 | `Create`, in the loop | a store without trainable variables | `CreateFmt('store %d has no trainable variables', [I])` |
| 5 | `Moments(I)` | `I < 0` or `I > High(FMoments)` | `CreateFmt('store %d out of range (count %d)', [I, Length(FMoments)])` |

- `ScheduleLR` checks #1 before #2, so `ScheduleLR(1e-4, 0, -1, 1000, 0.1)`
  reports the update-range message, not the warm-up one.
- In `Create`, check #3 comes first (the empty array raises before the loop);
  the per-store check is raised for the first offending index `I`.

## 5. Change history

| Date | Change |
|---|---|
| T14 (initial) | First edition, from the external design and the T14 facts. |
