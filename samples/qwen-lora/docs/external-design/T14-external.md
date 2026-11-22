# External Design — Unit `LoraOptim` (T14)

## Purpose

`TLoraAdamW` is the sample's own AdamW optimizer. `TOptimizer` cannot save its state, so the two moments (first and second moment) of every trainable variable live as non-trainable variables in a second `TVarStore` per parameter store. These moment stores are saved with `TVarStore.Save` and loaded with `TVarStore.Load`, enabling checkpoint save and resume. The update is PyTorch's AdamW (decoupled weight decay, bias correction). `ClipGlobalNorm` clips gradients over all stores together. `ScheduleLR` gives the learning rate of each update (linear warm-up, then cosine decay).

## The AdamW Update Rule

For each trainable parameter p with gradient g:

1. Weight decay (decoupled): `p := p * (1 - lr * weight_decay)`
2. First moment: `m := m * beta1 + g * (1 - beta1)`
3. Second moment: `v := v * beta2 + g^2 * (1 - beta2)`
4. Bias correction: `bc1 := 1 - beta1^t`, `bc2 := 1 - beta2^t` (t = step count)
5. Update: `p := p - m / (sqrt(v) / sqrt(bc2) + eps) * (lr / bc1)`

All moment operations are under `NoGradGuard`.

## Learning Rate Schedule

`ScheduleLR(Base, Update, Warmup, Total, MinRatio)`:

1. Warm-up phase (Update <= Warmup): `lr := Base * Update / Warmup`
2. Decay phase: `p := (Update - Warmup) / (Total - Warmup)`, then `lr := Base * (MinRatio + (1 - MinRatio) * 0.5 * (1 + cos(pi * p)))`

## Interface

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

## Default Configuration

`DefaultAdamWConfig` returns:
- `Beta1 := 0.9`
- `Beta2 := 0.999`
- `Eps := 1e-8`
- `WeightDecay := 0.01`

## Errors (ELoraOptim)

In order:

1. `ScheduleLR`:
   - `(Update < 1) or (Update > Total)`: `CreateFmt('update %d outside 1..%d', [Update, Total])`
   - `(Warmup < 0) or (Warmup >= Total)`: `CreateFmt('warm-up %d outside 0..%d', [Warmup, Total - 1])`

2. `Create`:
   - `Length(Stores) = 0`: `no stores`
   - A store without trainable variables: `CreateFmt('store %d has no trainable variables', [I])`

3. `Moments(I)`:
   - `I < 0` or `I > High(FMoments)`: `CreateFmt('store %d out of range (count %d)', [I, Length(FMoments)])`

## Moment Store Layout

For store I with trainable variables at indices 0 to N-1, the moments are:
- `FM[I][0]` named `m.0`, `FV[I][0]` named `v.0`
- `FM[I][1]` named `m.1`, `FV[I][1]` named `v.1`
- ... up to `m.(N-1)`, `v.(N-1)`

The moment variables are non-trainable (`ZerosNoTrain`). Each moment store has the same device and scalar kind as its parameter store.
