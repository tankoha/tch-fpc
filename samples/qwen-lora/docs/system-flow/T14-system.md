# System Flow — Unit `LoraOptim` (T14)

## Callers

### T17: Training Step

The training step (T17) uses `TLoraAdamW` to update the LoRA parameters of both GPUs:

1. Every second batch:
   - For each GPU's LoRA parameter store:
     - `ZeroGrad` on the optimizer
     - Backward pass
     - `ClipGlobalNorm` on the optimizer (both stores together)
     - `Step` with the current learning rate from `ScheduleLR`

2. The moment stores are saved at checkpoints (T16) together with the parameter stores.

### T16: Checkpoint

The checkpoint routine (T16) saves and restores the optimizer state:

1. Save:
   - `TVarStore.Save` on each parameter store
   - `TVarStore.Save` on each moment store (`Moments(I)`)
   - Write `StepCount` to a side file

2. Load:
   - `TVarStore.Load` on each parameter store
   - `TVarStore.Load` on each moment store
   - Set `StepCount` from the side file

### Unit Tests

The unit tests (T14b) verify `TLoraAdamW` against hand-computed values and against libtorch's `TOptimizer.AdamW`:

1. Stage 1 tests: `DefaultAdamWConfig` returns the expected values; `ScheduleLR` computes the correct learning rate for each phase and rejects invalid arguments.

2. Stage 2 tests: `Create` builds the moment stores correctly; `ZeroGrad` clears gradients; `Moments` returns the correct store and rejects out-of-range indices.

3. Stage 3 tests: 
   - Small hand-computed cases verify the update rule (tolerance 1e-12).
   - Larger cases run 20 updates and compare against `TOptimizer.AdamW` (tolerance 1e-6 in fp32, 1e-12 in fp64).
   - `ClipGlobalNorm` matches libtorch's `ClipGradNorm` (tolerance 1e-6).
   - Two stores clipping and updating together give the same result as one store with the same variables (exact match).

### ITa Driver

The ITa driver `tests/ita/T14/loraoptim_check.pas` runs a toy problem on the CPU:

1. Build and run via `<project root>/tests/ita/T14-run.sh`.
2. Compares against `TOptimizer.AdamW` and `ClipGradNorm` from libtorch.
3. Tests fp32 and fp64, with and without clipping, with schedule, and save/load/resume.
4. Result file: `<project root>/work/ita/T14/summary.tsv`.

## Test Fixtures

### `tests/ut/LoraOptimFixture.pas`

Provided by the harness owner, used by UT and ITa:

- `MakeToyData(Seed, Kind, X, Y)`: Creates toy input data X [16, 4] and Y [16, 3].
- `MakeToyStore(Seed, Kind, W, B)`: Creates a CPU store with trainable `w` [4, 3] (random from seed) and `b` [3] (zeros).
- `ToyLoss(W, B, X, Y)`: Computes mean((X W + B - Y)^2).
- `ReferenceRun(StoreSeed, DataSeed, Kind, Steps, LR, WD, Clip, W, B)`: Oracle using `TOptimizer.AdamW`.
- `MaxAbsDiff(A, B)`, `ValuesText(T)`: Utility functions.

### Test Skeleton

`tests/ut/LoraOptim_test.pas` provides:

- `OneVarStore(N, AOnes, P)`: Creates a CPU kDouble store with one trainable variable `p` [N] of ones or zeros.
- Each test frees what it creates in `try ... finally` (optimizer first, then stores).

## Verification

The optimizer follows PyTorch's AdamW exactly:

- Decoupled weight decay (applied to the parameter, not the gradient).
- Bias correction on the moments.
- `ClipGlobalNorm` matches libtorch's `ClipGradNorm`.
- Multiple stores are clipped and updated as if all their variables were in one store.
- Save/load/resume gives exactly the same result as uninterrupted training.
