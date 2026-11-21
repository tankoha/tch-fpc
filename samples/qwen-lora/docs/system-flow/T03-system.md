## Callers
- **T10** `PackedLinear` de‑quantises weight tensors on a device using `GgufQuantT.DequantizeT`.
- **T18** quantises merged tensors on a device using `GgufQuantT.QuantizeQ8_0T`.

## Unit Tests
The UT compares the output of `GgufQuantT` with the reference implementation `GgufQuant` from T02 for all supported types and error cases.

## ITa Driver
`tests/ita/T03-run.sh` builds and runs `tests/ita/T03/quantt_check.pas`. It de‑quantises every tensor of the rehearsal GGUF model, re‑quantises it, and compares the resulting bytes with the oracle tables from T02 (`work/oracle/T02/`). The summary is written to `work/ita/T03/summary.tsv`.
