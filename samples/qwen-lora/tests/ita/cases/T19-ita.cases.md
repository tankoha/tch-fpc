# ITa cases — qlora_dataset (T19)

| Case ID | Title | Expected (short) |
|---|---|---|
| ITa-T19-001 | Record outcomes vs expected.tsv | 0 mismatches |
| ITa-T19-002 | Valid records vs split recomputed | 0 mismatches |
| ITa-T19-003 | train/valid format, raw order | 0 mismatches |
| ITa-T19-004 | manifest: MD5, split, class | 0 mismatches |
| ITa-T19-005 | card totals vs rejects.jsonl | 0 mismatches |
| ITa-T19-006 | Second run: files byte-identical | 0 files differ |

## Test Execution

```
QLORA_MODEL=<rehearsal GGUF> <project root>/tests/ita/T19-run.sh
```

Result file: `<project root>/work/ita/T19/summary.tsv`
