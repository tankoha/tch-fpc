# Program flow — `GgufQuant` (T02)

Companion to `docs/detailed-design/T02-detailed.md`; source of truth:
`tasks/T02-facts.md`.

## `DequantizeRow(ATType, ASrc, ACount, ADst)`

- Resolve `ATType`: an unsupported type id raises `EGgufError` (message
  `"unsupported tensor type N"`).
- `ACount` not a multiple of the type's block size: raise `EGgufError`
  (message names the type and the count).
- Dequantize the row with the per-type dequantiser below, element by
  element for f32/f16/bf16 and block by block for q8_0/q4_K/q5_K/q6_K —
  the `ACount / block` blocks one after the other, element order as in the
  file.

## `DequantizeBytes(ATType, ABytes, ACount)`

- `Length(ABytes) <> GgufTensorBytes(ATType, [ACount])`: raise
  `EGgufError` (message names both lengths).
- Call `DequantizeRow` (validation and dequantisation) over `ABytes` and
  return the resulting `TSingleArray` of `ACount` elements.

## `QuantizeRowQ8_0(ASrc, ACount, ADst)`

- Non-finite input: raise `EGgufError`.
- For each of the `ACount / 32` blocks, in order: call the Q8_0 block
  quantiser on the next 32 `Singles` → 34 bytes, block after block into
  `ADst`.

## `QuantizeQ8_0(AValues)`

- `Length(AValues) mod 32 <> 0`: raise `EGgufError`.
- Call `QuantizeRowQ8_0` over `AValues` and return the
  `Length div 32 * 34` bytes.

## Helpers — role and call sites

- Per-type dequantisers for f32 (0), f16 (1), q8_0 (8) and bf16 (30):
  implemented as the per-type branches of `DequantizeRow` (the detailed
  design's `DequantF32` / `DequantF16` / `DequantQ8_0` / `DequantBf16`
  rules: one element each for the element-wise types, one 34-byte block
  for q8_0), each branch reading from the block's first byte;
  `DequantizeBytes` reaches them through `DequantizeRow`.
  T02b2 adds the q4_K (12), q5_K (13) and q6_K (14) branches.
- `Q8_0NonFinite(AX)` (internal): the bit-level non-finite test (the
  eight exponent bits all 1, via `Move`); called per element from
  `QuantizeRowQ8_0`, which then raises `EGgufError`.
- `GgufTinyD` (constant, the `Single` 2^-128): the guard
  `0 < D <= GgufTinyD` inside `QuantizeRowQ8_0`, checked **before**
  computing `1 / D`.
- Q8_0 block quantiser: the per-block steps inside `QuantizeRowQ8_0`'s
  block loop (`Amax` → `D := Amax / 127` → the `0 < D <= 2^-128` guard →
  `d := SingleToFp16(D)` → the 32 quants via `RoundHalfAway`);
  `QuantizeQ8_0` reaches it through `QuantizeRowQ8_0`.
- `ScaleMin(J, Scales)`: (T02b2) returns the 6-bit scale and minimum of
  group `J` (0..7) from the 12-byte scale field; called from
  `DequantQ4_K` and `DequantQ5_K` for the two scale slots of each of the
  four groups (`Is` and `Is + 1`).
- Conversions `Fp16ToSingle`, `SingleToFp16`, `Bf16ToSingle` and
  `RoundHalfAway`: called exactly where the detailed design's steps name
  them (dequantisers use the forward conversions; the quantiser uses
  `SingleToFp16` and `RoundHalfAway`).
