# ITa cases — qlora_eval (T20)

| Case ID | Title | Expected (short) |
|---|---|---|
| ITa-T20-001 | Result lines match reply order (model, task, seed, think, dialect) | 0 mismatches |
| ITa-T20-002 | Outcome, compiled, passed match verifier recomputation | 0 mismatches |
| ITa-T20-003 | Report table rows recomputed correctly | 0 missing |
| ITa-T20-004 | Adoption lines recomputed, sign test included | 0 missing |
| ITa-T20-005 | Second run: results.tsv and report.md byte-identical | 0 files differ |

## Test Execution

```
<project root>/tests/ita/T20-run.sh
```

Result file: `<project root>/work/ita/T20/summary.tsv`
