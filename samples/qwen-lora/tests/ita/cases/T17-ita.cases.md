# ITa cases — program `qlora_train` (T17)

| Case ID | Title | Expected (short) |
|---|---|---|
| ITa-T17-001 | Overfit: loss of last update / first update | ≤ 0.1 |
| ITa-T17-002 | Logged loss matches whole-graph computation (relative) | ≤ 1e-4 |
| ITa-T17-003 | Best validation loss / validation loss without adapters | ≤ 0.99 |
| ITa-T17-004 | Recorded validation loss matches whole-graph one (relative) | ≤ 1e-4 |
| ITa-T17-005 | Resume: adapters and moments equal (max difference) | 0 |
| ITa-T17-006 | Non-finite loss: exit code 3, update lines, no checkpoint | 0 mismatches |

## Command

```
QLORA_MODEL=<rehearsal GGUF> <project root>/tests/ita/T17-run.sh
```

## Result file

```
<project root>/work/ita/T17/summary.tsv
```

Columns: case, OK/NG, value, limit, note. OK when value ≤ limit.

## Notes

- Four runs: `ovf` (8 epochs of one update, lr 1e-3, warm-up 1), `full`
  (2 epochs, budget 256, 5 updates per epoch, checkpoint every 3), `part`
  (resume from update 3), `nan` (lr 1e30).
- Data: `tests/ita/T17/train.jsonl` (8 short records), `valid.jsonl` (4).
- About 6.5 minutes on the CPU.
- Measured: 0.0896, 3.2e-9, 0.745, 4.7e-9, 0, 0.
