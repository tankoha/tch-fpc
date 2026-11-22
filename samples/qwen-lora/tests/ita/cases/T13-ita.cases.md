# ITa cases — unit `Recompute` (T13)

| Case ID | Title | Expected (short) |
|---|---|---|
| ITa-T13-001 | fp64, 8 stages, width 16, 4 rows, inputs on device: max of parameter and input gradient differences | ≤ 1e-10 |
| ITa-T13-002 | fp64, 64 stages, width 64, 32 rows, inputs on host | ≤ 1e-10 |
| ITa-T13-003 | fp32 against fp32, 64 stages, width 256, 128 rows | ≤ 1e-6 |
| ITa-T13-004 | count of violations: forward without graph, one call per stage, 16 saved inputs without grad; backward one more call with graph | 0 |
| ITa-T13-005 | two backwards: W.grad equals twice the reference | ≤ 1e-10 |

## 実行コマンド

`<project root>/tests/ita/T13-run.sh`

## 結果ファイル

`<project root>/work/ita/T13/summary.tsv`

形式: case, OK/NG, value, limit, note
