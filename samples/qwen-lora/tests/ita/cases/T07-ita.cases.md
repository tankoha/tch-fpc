# ITa cases — unit `QwenChat` (T07)

| Case ID | Title | Expected (short) |
|---|---|---|
| ITa-T07-001 | Renderings match `/apply-template` of 14 conversations | 0 different |
| ITa-T07-002 | Training sample encodings match `/tokenize` of 5 samples | 0 different |
| ITa-T07-003 | Loss spans are exactly last answer + `<e>` of 5 samples | 0 different |
| ITa-T07-004 | Rejected conversations raise `EChatError` of 4 cases | 0 missed |

## Execution

Command:
```
QLORA_MODEL=<rehearsal GGUF> <project root>/tests/ita/T07-run.sh
```

Result file:
```
<project root>/work/ita/T07/summary.tsv
```

The driver compares against the bundled `llama-server` with the 27B model's
chat template. Value ≤ limit means OK.
