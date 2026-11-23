# ITa cases — T16 Checkpoint

| Case ID | Title | Expected (short) |
|---|---|---|
| ITa-T16-001 | every A, B and moment of `full/ckpt-00000010` against `part/ckpt-00000010` (both read back into fresh sets) | 0 (exact) |
| ITa-T16-002 | `state.json` of `full/ckpt-10`, `part/ckpt-10` and `part/ckpt-5` against the states written (count of differing fields) | 0 |
| ITa-T16-003 | entries left: `full` exactly `ckpt-00000002,ckpt-00000008,ckpt-00000009,ckpt-00000010` (the last 3 and the best), `part` exactly `ckpt-00000005,ckpt-00000010` (count of mismatches) | 0 |
| ITa-T16-004 | the loss of every update, one go against resumed (`full-loss.tsv`, `part-loss.tsv`) | 0 (exact) |

Command:

```
QLORA_MODEL=<rehearsal GGUF> <project root>/tests/ita/T16-run.sh
```

Result file:

```
<project root>/work/ita/T16/summary.tsv
```
