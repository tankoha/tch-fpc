# T02: GgufQuant (system flow)

## Callers
- **T03 (`GgufQuantT`):** Dequantises tensors for training via `DequantizeRow`/`DequantizeBytes`.
- **T10:** Dequantises weights via `DequantizeRow`/`DequantizeBytes`.
- **T18:** Quantises merged tensors to Q8_0 via `QuantizeRowQ8_0`/`QuantizeQ8_0`.
- **ITa driver (`tests/ita/T02/quant_check.pas`):**
  - Reads every tensor from the GGUF file via `GgufFile`,
  - Dequantises it via `GgufQuant`,
  - Re-quantises Q8_0 compatible tensors via `QuantizeQ8_0`,
  - Compares results with the oracle `dequant-sha256.tsv` and `q8_0-sha256.tsv`.

## Flow
1. `GgufFile` reads tensor metadata and bytes from the GGUF file.
2. Caller passes tensor type, byte array, and element count to `GgufQuant`.
3. `GgufQuant` dequantises or quantises in scalar operations.
4. Result is used by:
   - Training (T03, T10),
   - Merging (T18),
   - ITa validation (bit-for-bit agreement with upstream).