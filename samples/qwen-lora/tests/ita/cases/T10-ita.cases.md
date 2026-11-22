# ITa cases — T10 (PackedLinear, QwenBlocks)

| Case ID | Title | Expected (short) |
|---|---|---|
| ITa-T10-001 | Every block of the rehearsal model (24), T 8, against the scalar version | ≤ 1e-4 |
| ITa-T10-002 | Rehearsal blocks 0 (linear) and 3 (full), T 70 | ≤ 1e-4 |
| ITa-T10-003 | The tiny model's two blocks, T 70 (tiling, GQA) | ≤ 1e-4 |
| ITa-T10-004 | Rehearsal blocks 0-3 chained, T 16, against the scalar chain | ≤ 1e-4 |
| ITa-T10-005 | Rehearsal blocks 0-3 as Recompute stages: dL/dX0 against an ordinary backward | ≤ 1e-5 |

Run command:

```
QLORA_MODEL=<rehearsal GGUF> <project root>/tests/ita/T10-run.sh
```

Result file:

```
<project root>/work/ita/T10/summary.tsv
```

The summary.tsv columns are: case, OK/NG, value, limit, note. OK when value ≤ limit. The driver's exit code and the count of stderr lines (excluding GPU warnings) follow the table.
