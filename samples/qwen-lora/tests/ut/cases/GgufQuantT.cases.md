# UT cases — unit `GgufQuantT` (T03)

Unit under test: `src/GgufQuantT.pas` — the `DequantizeT` and `QuantizeQ8_0T` entry points (and the internal per-type dequantisers / Q8_0 quantiser). `DequantizeT` converts raw GGUF bytes to single-precision floating point tensors on the same device. `QuantizeQ8_0T` quantises single-precision inputs to Q8_0 blocks. Both are tensor-level equivalents of T02 `GgufQuant`.

Base: `tasks/T03-facts.md` — sections "Dequantisation, per type", "Q8_0 quantisation", "Unit `GgufQuantT` (interface, final)" and "Unit tests". The facts are final and state every error/message. No other design documents are used.

Test arrangement (facts "Unit tests"): every case creates its own test data (bytes or tensors) and compares `DequantizeT` / `QuantizeQ8_0T` **bit for bit** with the result from T02's unit `GgufQuant`. UT never needs the rehearsal model, the GPU or `QLORA_MODEL`.

| Case ID | Title | Target function | Input (short) | Expected result | Notes |
|---|---|---|---|---|---|
| UT-GgufQuantT-001 | f32 random blocks (512 × 64) | DequantizeT | RandSeed = 1001; 512 blocks × 64 values | Same bits as GgufQuant | Random blocks; seed fixes reproduction |
| UT-GgufQuantT-002 | f16 random blocks (512 × 32) | DequantizeT | RandSeed = 1002; 512 blocks × 32 values | Same bits as GgufQuant | Random blocks; seed fixes reproduction |
| UT-GgufQuantT-003 | bf16 random blocks (512 × 32) | DequantizeT | RandSeed = 1003; 512 blocks × 32 values | Same bits as GgufQuant | Random blocks; seed fixes reproduction |
| UT-GgufQuantT-004 | q8_0 random blocks (512 × 32) | DequantizeT | RandSeed = 1004; 512 blocks × 32 values | Same bits as GgufQuant | Random blocks; seed fixes reproduction |
| UT-GgufQuantT-005 | q4_K random blocks (512 × 256) | DequantizeT | RandSeed = 1005; 512 blocks × 256 values | Same bits as GgufQuant | Random blocks; seed fixes reproduction |
| UT-GgufQuantT-006 | q5_K random blocks (512 × 256) | DequantizeT | RandSeed = 1006; 512 blocks × 256 values | Same bits as GgufQuant | Random blocks; seed fixes reproduction |
| UT-GgufQuantT-007 | q6_K random blocks (512 × 256) | DequantizeT | RandSeed = 1007; 512 blocks × 256 values | Same bits as GgufQuant | Random blocks; seed fixes reproduction |
| UT-GgufQuantT-008 | f32 hand case | DequantizeT | See f32 block from UT-GgufQuant-011 | Same bits as GgufQuant | From T02 hand case; exact fp32 |
| UT-GgufQuantT-009 | f16 hand case | DequantizeT | See f16 block from UT-GgufQuant-012 | Same bits as GgufQuant | From T02 hand case; exact fp32 |
| UT-GgufQuantT-010 | bf16 hand case | DequantizeT | See bf16 block from UT-GgufQuant-013 | Same bits as GgufQuant | From T02 hand case; exact fp32 |
| UT-GgufQuantT-011 | q8_0 hand case | DequantizeT | See q8_0 block from UT-GgufQuant-014 | Same bits as GgufQuant | From T02 hand case; exact fp32 |
| UT-GgufQuantT-012 | q4_K hand case | DequantizeT | See q4_K block from UT-GgufQuant-101 | Same bits as GgufQuant | From T02 hand case; exact fp32 |
| UT-GgufQuantT-013 | q5_K hand case | DequantizeT | See q5_K block from UT-GgufQuant-102 | Same bits as GgufQuant | From T02 hand case; exact fp32 |
| UT-GgufQuantT-014 | q6_K hand case | DequantizeT | See q6_K block from UT-GgufQuant-103 | Same bits as GgufQuant | From T02 hand case; exact fp32 |
| UT-GgufQuantT-015 | Unsupported tensor type (2) | DequantizeT | Type = 2, count = 64 | EGgufError (names the type) | Unsupported type |
| UT-GgufQuantT-016 | Count not a multiple of block size (q8_0) | DequantizeT | Type = 8 (q8_0), count = 33 | EGgufError (names type and count) | Block size violation |
| UT-GgufQuantT-017 | Wrong byte length | DequantizeT | Type = 8, count = 32, length = 33 | EGgufError (names both lengths) | Byte length mismatch |
| UT-GgufQuantT-018 | Result tensor shape and kind (f32) | DequantizeT | Valid f32 block (16 bytes) | Shape=[4], Kind=kFloat | Verify result shape/kind |
| UT-GgufQuantT-019 | Result tensor shape and kind (bf16) | DequantizeT | Valid bf16 block (16 bytes) | Shape=[8], Kind=kFloat | Verify result shape/kind |
| UT-GgufQuantT-020 | Result tensor device matches input | DequantizeT | Valid block on CPU | Result on CPU | Device consistency |
| UT-GgufQuantT-021 | ARaw not kUInt8 | DequantizeT | Kind = kFloat, type = 0, count = 4 | EGgufError | ARaw kind mismatch |
| UT-GgufQuantT-022 | ARaw not 1-D | DequantizeT | 2-D kUInt8 tensor, type = 0, count = 4 | EGgufError | ARaw shape mismatch |
| UT-GgufQuantT-023 | Count not a multiple (q4_K) | DequantizeT | Type = 12 (q4_K), count = 255 | EGgufError (names the count) | Block size violation |
| UT-GgufQuantT-024 | Count not a multiple (q6_K) | DequantizeT | Type = 14 (q6_K), count = 255 | EGgufError (names the count) | Block size violation |
| UT-GgufQuantT-101 | Q8_0 random finite blocks (544) | QuantizeQ8_0T | RandSeed = 1101; 544 blocks × 32 values | Same bytes as GgufQuant | Finite random; seed fixes reproduction |
| UT-GgufQuantT-102 | Q8_0 random zeros and tiny values | QuantizeQ8_0T | RandSeed = 1102; zeros, 1e-44, 1e-41 | Same bytes as GgufQuant | Special magnitude cases |
| UT-GgufQuantT-103 | Q8_0 hand case | QuantizeQ8_0T | See input from UT-GgufQuant-019 | Same bytes as GgufQuant | From T02 hand case |
| UT-GgufQuantT-104 | Non-finite input values | QuantizeQ8_0T | Contains +inf, -inf, NaN | EGgufError | Non-finite inputs forbidden |
| UT-GgufQuantT-105 | Input length not multiple of 32 | QuantizeQ8_0T | Length = 33 | EGgufError (mentions length) | Length constraint |
| UT-GgufQuantT-106 | Result tensor length and kind | QuantizeQ8_0T | Valid input (32 elements) | Length=34, Kind=kUInt8 | Verify result properties |
| UT-GgufQuantT-107 | Result tensor device matches input | QuantizeQ8_0T | Valid input tensor on CPU | Result on CPU | Device consistency |
| UT-GgufQuantT-108 | AX not kFloat (kDouble tensor) | QuantizeQ8_0T | Double tensor, length 32 | EGgufError | Wrong tensor kind |
| UT-GgufQuantT-109 | AX not 1-D | QuantizeQ8_0T | kFloat tensor shape [2,32] | EGgufError | AX shape mismatch |