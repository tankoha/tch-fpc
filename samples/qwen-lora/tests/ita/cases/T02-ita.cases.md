| Case ID   | Title                                      | Expected (short)                     |
|-----------|-------------------------------------------|--------------------------------------|
| ITa-T02-001 | f32 tensors: sha256 of the fp32 bytes      | 133 lines, no diff, correct SHA256   |
| ITa-T02-002 | q8_0 tensors: dequantised                 | 36 lines, no diff, correct SHA256    |
| ITa-T02-003 | q4_K tensors: dequantised                 | 98 lines, no diff, correct SHA256    |
| ITa-T02-004 | q5_K tensors: dequantised                 | 36 lines, no diff, correct SHA256    |
| ITa-T02-005 | q6_K tensors: dequantised                 | 17 lines, no diff, correct SHA256    |
| ITa-T02-006 | All tensors dequantised, then Q8_0        | 284 lines, no diff, correct SHA256   |
| ITa-T02-007 | q8_0 tensors re-quantised equal to file   | 36 lines, no diff, correct bytes     |

---

Run all cases with:
```bash
QLORA_MODEL=<absolute path of the rehearsal GGUF> <project root>/tests/ita/T02-run.sh
```

Result file:
`<project root>/work/ita/T02/summary.tsv`