# ITa cases — unit `LoraOptim` (T14)

| Case ID | Title | Expected (short) |
|---|---|---|
| ITa-T14-001 | fp32, 20 updates, lr 0.05, wd 0.01, against TOptimizer.AdamW | ≤ 1e-6 |
| ITa-T14-002 | fp64, 200 updates, against TOptimizer.AdamW | ≤ 1e-12 |
| ITa-T14-003 | fp32, 20 updates with clipping at 0.5, against ClipGradNorm + AdamW | ≤ 1e-6 |
| ITa-T14-004 | fp64, 50 updates with clipping: two stores (w; b) against one store (w, b) | 0 |
| ITa-T14-005 | fp32, 100 updates with ScheduleLR(0.05, u, 10, 100, 0.1) on both sides | ≤ 1e-6 |
| ITa-T14-006 | fp32: 10 updates against 5 + save (store and Moments(0)) + load into a new store and optimizer + StepCount := 5 + 5 | 0 |
| ITa-T14-007 | fp64: gradients after ClipGlobalNorm(0.1) against ClipGradNorm(0.1), and the returned norm | ≤ 1e-12 |

**Command**: `<project root>/tests/ita/T14-run.sh`

**Result file**: `<project root>/work/ita/T14/summary.tsv`
