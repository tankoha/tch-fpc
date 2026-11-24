# T18 ITa Test Cases

| Case ID | Title | Expected (short) |
|---|---|---|
| ITa-T18-001 | WriteMerged without adapters vs base (byte-identical) | 0 |
| ITa-T18-002 | Merged tensor dequantised vs fp32 merge, max diff / Q8_0 step | ≤ 1 |
| ITa-T18-003 | Mean \|Δ logprob\|: merged vs adapter, divided by floor vs base | ≤ 1.5 |
| ITa-T18-004 | Mean loss: 1 − (base − merged) / (base − adapter) | ≤ 0.1 |
| ITa-T18-005 | H47: max \|Pascal − llama-server\| over top 10 | ≤ 0.02 |
| ITa-T18-006 | Merged file vs prototype (sha256 comparison) | 0 |

## Execution

```bash
QLORA_MODEL=<rehearsal GGUF> <project root>/tests/ita/T18-run.sh
```

## Result File

`<project root>/work/ita/T18/summary.tsv`

Columns: case, OK/NG, value, limit, note; then exit codes and stderr line count.
