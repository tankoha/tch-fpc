| Case ID | Title | Expected (short) |
|---|---|---|
| ITa-T03-001 | Dequantise f32 tensor matches reference | match reference |
| ITa-T03-002 | Dequantise f16 tensor matches reference | match reference |
| ITa-T03-003 | Dequantise bf16 tensor matches reference | match reference |
| ITa-T03-004 | Dequantise q8_0 tensor matches reference | match reference |
| ITa-T03-005 | Dequantise q4_K tensor matches reference | match reference |
| ITa-T03-006 | Dequantise q5_K tensor matches reference | match reference |
| ITa-T03-007 | Dequantise q6_K tensor matches reference | match reference |

```bash
QLORA_MODEL=<absolute path of the rehearsal GGUF> <project root>/tests/ita/T03-run.sh
```

Result file: `<project root>/work/ita/T03/summary.tsv`
