# UT cases — unit `LoraOptim` (T14)

`Cfg` is `DefaultAdamWConfig` unless a row says otherwise. A tolerance row
passes when the difference is ≤ the limit (`AssertEquals(Expected, Actual,
Delta)` for a Double, or `AssertTrue(MaxAbsDiff(..) <= limit)`); "exact"
means the exact strings or a difference of 0. Error rows catch `ELoraOptim`
and compare the whole message (`AssertEquals`); a row fails if no exception
comes (`Fail('no ELoraOptim')`).

| Case ID | Title | Target function | Input (short) | Expected result | Notes |
|---|---|---|---|---|---|
| UT-LoraOptim-001 | default config | DefaultAdamWConfig | `DefaultAdamWConfig` | Beta1 0.9, Beta2 0.999, Eps 1e-8, WeightDecay 0.01 (exact) | Stage 1 |
| UT-LoraOptim-002 | warm-up, first update | ScheduleLR | `ScheduleLR(1e-4, 1, 50, 1000, 0.1)` | 2e-6 (delta 1e-15) | Stage 1 |
| UT-LoraOptim-003 | warm-up, last update | ScheduleLR | `ScheduleLR(1e-4, 50, 50, 1000, 0.1)` | 1e-4 (delta 1e-15) | Stage 1 |
| UT-LoraOptim-004 | cosine decay, middle | ScheduleLR | `ScheduleLR(1e-4, 525, 50, 1000, 0.1)` | 5.5e-5 (delta 1e-15) | Stage 1 |
| UT-LoraOptim-005 | decay, total update | ScheduleLR | `ScheduleLR(1e-4, 1000, 50, 1000, 0.1)` | 1e-5 (delta 1e-15) | Stage 1 |
| UT-LoraOptim-006 | no warm-up | ScheduleLR | `ScheduleLR(1e-4, 1, 0, 3, 0.1)` (no warm-up) | 7.75e-5 (delta 1e-15) | Stage 1 |
| UT-LoraOptim-007 | update below range | ScheduleLR | `ScheduleLR(1e-4, 0, 50, 1000, 0.1)` | `update 0 outside 1..1000` | Stage 1 |
| UT-LoraOptim-008 | update above range | ScheduleLR | `ScheduleLR(1e-4, 1001, 50, 1000, 0.1)` | `update 1001 outside 1..1000` | Stage 1 |
| UT-LoraOptim-009 | warm-up above range | ScheduleLR | `ScheduleLR(1e-4, 10, 1000, 1000, 0.1)` | `warm-up 1000 outside 0..999` | Stage 1 |
| UT-LoraOptim-010 | warm-up negative | ScheduleLR | `ScheduleLR(1e-4, 10, -1, 1000, 0.1)` | `warm-up -1 outside 0..999` | Stage 1 |
| UT-LoraOptim-011 | Create, two stores, moment layout | Create | S1 := `MakeToyStore(3, kDouble, W, B)`; S2 := `TVarStore.Create(TDevice.Cpu, kFloat)` with `S2.Root.Zeros('c', [2])`; `TLoraAdamW.Create([S1, S2], Cfg)` | `StoreCount` 2; `Moments(0).Count` 4, `Moments(1).Count` 2; `Moments(0).Variables[0..3].Name` `m.0`, `v.0`, `m.1`, `v.1`; `Length(Moments(0).TrainableVariables)` 0; `Moments(0).Kind` kDouble, `Moments(1).Kind` kFloat; `Moments(0).Variables[0].Tensor.abs.sum(kNone)` 0; `StepCount` 0 | Stage 2 |
| UT-LoraOptim-012 | Create, no stores | Create | `TLoraAdamW.Create([], Cfg)` | `no stores` | Stage 2 |
| UT-LoraOptim-013 | Create, store without trainable variables | Create | S1 as 011; S2 a kDouble store with only `S2.Root.ZerosNoTrain('z', [2])`; `Create([S1, S2], Cfg)` | `store 1 has no trainable variables` | Stage 2 |
| UT-LoraOptim-014 | ZeroGrad | ZeroGrad | S1 as 011, optimizer on it; `MakeToyData(4, kDouble, X, Y)`; `ToyLoss(W, B, X, Y).Backward`; `ZeroGrad` | before `ZeroGrad` `W.grad.abs.sum(kNone)` > 0; after it `ValuesText` of `W.grad.abs.sum(kNone)` and of `B.grad.abs.sum(kNone)` `'0'` | Stage 2 |
| UT-LoraOptim-015 | Moments, index above range | Moments | the optimizer of 011; `Moments(2)` | `store 2 out of range (count 2)` | Stage 2 |
| UT-LoraOptim-016 | Moments, index below range | Moments | the optimizer of 011; `Moments(-1)` | `store -1 out of range (count 2)` | Stage 2 |
| UT-LoraOptim-101 | Step, one step, hand values | Step | `OneVarStore(1, True, P)`; Cfg with `Eps := 0`; `ZeroGrad`; `(P * 0.5).sum(kNone).Backward`; `Step(0.1)` | `P.DoubleValue([0])` 0.899 (delta 1e-12); `ValuesText` of `Moments(0).Variables[0].Tensor` (m) `'0.05'`, of `Variables[1]` (v) `'0.00025'`; `StepCount` 1 | Stage 3 |
| UT-LoraOptim-102 | Step, second step, hand values | Step | as 101, then the same `ZeroGrad`, backward, `Step(0.1)` again | P 0.798101 (delta 1e-12); m `'0.095'`, v `'0.00049975'`; `StepCount` 2 | Stage 3 |
| UT-LoraOptim-103 | Step, parity AdamW fp32 | Step | `MakeToyStore(5, kFloat, W, B)`, `MakeToyData(6, kFloat, X, Y)`; 20 × (`ZeroGrad`, `ToyLoss(..).Backward`, `Step(0.05)`); `ReferenceRun(5, 6, kFloat, 20, 0.05, 0.01, 0, RW, RB)` | `MaxAbsDiff(W, RW)` and `(B, RB)` ≤ 1e-6 (measured 8.9e-8, 1.0e-7); `StepCount` 20 | Stage 3; oracle `TOptimizer.AdamW` |
| UT-LoraOptim-104 | Step, parity AdamW fp64 | Step | as 103 in kDouble | ≤ 1e-12 (measured 8.3e-16, 1.9e-16) | Stage 3; oracle `TOptimizer.AdamW` |
| UT-LoraOptim-105 | Step, variable without grad skipped | Step | `MakeToyStore(7, kDouble, W, B)` and U := `S.Root.Ones('u', [2])` (trainable, not in the loss); `MakeToyData(8, kDouble, X, Y)`; one update with `Step(0.1)` | `ValuesText(U)` `'1 1'` (no grad: skipped, also no weight decay); `Moments(0).Variables[4].Name` `m.2`, its values `'0 0'`, `Variables[5]` (v.2) `'0 0'` | Stage 3 |
| UT-LoraOptim-106 | Step, zero LR | Step | `OneVarStore(1, True, P)`; backward of `(P * 0.5).sum(kNone)`; `Step(0)` | `ValuesText(P)` `'1'`; m `'0.05'`; `StepCount` 1 | Stage 3 |
| UT-LoraOptim-107 | ClipGlobalNorm, clips | ClipGlobalNorm | `OneVarStore(2, False, P)`; backward of `(P * TTensor.FromArray([3.0, 4], kDouble)).sum(kNone)`; `ClipGlobalNorm(1)` | result 5 (delta 1e-12); `MaxAbsDiff(P.grad, FromArray([0.6, 0.8], kDouble))` ≤ 1e-6 (measured 1.6e-7: the factor is 1 / (5 + 1e-6)) | Stage 3 |
| UT-LoraOptim-108 | ClipGlobalNorm, no clipping | ClipGlobalNorm | as 107 with `ClipGlobalNorm(10)` | result 5; `ValuesText(P.grad)` `'3 4'` (unchanged) | Stage 3 |
| UT-LoraOptim-109 | ClipGlobalNorm, two stores | ClipGlobalNorm | two kDouble stores, `p` [1] in the first and `c` [1] in the second (`Root.Zeros`); `Create([S1, S2], Cfg)`; backward of `(P * 3.0 + C * 4.0).sum(kNone)`; `ClipGlobalNorm(1)` | result 5; P.grad 0.6 and C.grad 0.8, each within 1e-6 | Stage 3 |
| UT-LoraOptim-110 | parity with clipping, fp32 | Step/ClipGlobalNorm | `MakeToyStore(9, kFloat, ..)`, `MakeToyData(10, kFloat, ..)`; 20 × (`ZeroGrad`, backward, `ClipGlobalNorm(0.5)`, `Step(0.05)`); `ReferenceRun(9, 10, kFloat, 20, 0.05, 0.01, 0.5, RW, RB)` | ≤ 1e-6 (measured 1.8e-7, 6.0e-8) | Stage 3; oracle `ClipGradNorm` + `TOptimizer.AdamW` |
| UT-LoraOptim-111 | two stores equal one store | Step/ClipGlobalNorm | SU := `MakeToyStore(11, kDouble, W, B)`; S2, S3 kDouble stores: `ManualSeed(11)`, `W2 := S2.Root.RandnStandard('w', [4, 3])`, `B2 := S3.Root.Zeros('b', [3])`; `MakeToyData(12, kDouble, X, Y)`; optimizers on `[SU]` and `[S2, S3]`; 20 × (each: `ZeroGrad`, backward, `ClipGlobalNorm(0.5)`, `Step(0.05)`) | exact: `MaxAbsDiff(W, W2)` and `(B, B2)` 0 | Stage 3 |
| UT-LoraOptim-112 | no gradients: clip and step | ClipGlobalNorm/Step | `OneVarStore(2, True, P)`; no backward; `ClipGlobalNorm(1)`, then `Step(0.1)` | result 0; `ValuesText(P)` `'1 1'`; `StepCount` 1 | Stage 3 |
