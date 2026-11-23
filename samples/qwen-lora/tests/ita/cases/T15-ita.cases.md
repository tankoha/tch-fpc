# ITa cases — unit `TrainData` (T15)

| Case ID | Title | Expected (short) |
|---|---|---|
| ITa-T15-001 | Token encoding and loss span match rendered text | 0 mismatches |
| ITa-T15-002 | Admission counts match file contents | 0 mismatches |
| ITa-T15-003 | Batches respect budget, samples seen once per epoch | 0 |
| ITa-T15-004 | Batches identical across processes | 0 (equal) |
| ITa-T15-005 | Mask targets decode to trimmed answer + im_end | 0 mismatches |

## Command

```
QLORA_MODEL=<rehearsal GGUF> <project root>/tests/ita/T15-run.sh
```

## Result file

```
<project root>/work/ita/T15/summary.tsv
```

Columns: case, OK/NG, value, limit, note. OK when value ≤ limit.

## Notes

- 42 records in `tests/ita/T15/records.jsonl`.
- Classes: `compiled-run`, `fpc-approx` (default); 3 are `static-only`.
- Two records exceed 2048 tokens.
- Expected: 37 samples admitted, 33 batches over three epochs.
- Three processes: once `check`, twice `dump`.
