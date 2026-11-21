# Detailed design — T02, unit `GgufQuant` (scalar dequantisation and Q8_0 quantisation)

The unit `GgufQuant` holds the scalar dequantisers for the tensor types of
T01 and the Q8_0 quantiser. It is the reference for T03 (`GgufQuantT`, the
same in tensor ops) and for T18 (the merge writes `Q8_0`), so it must agree
with upstream `llama.cpp` **bit for bit**; ITa (`T02-run.sh`) compares with
`diff` against the upstream C reference in `work/oracle/T02/`.

- Source of truth: `tasks/T02-facts.md` (taken from upstream
  `ggml/src/ggml-quants.c` at `39febd2ad`). Where this document differs
  from the facts, the facts win.
- Reused from `GgufFile` (T01, unchanged): `EGgufError`, `GgufTypeName`,
  `GgufTensorBytes`.
- UT compares the resulting `Singles` by their `Cardinal` bits, never with
  a tolerance (facts "Unit tests").

## General rules (all routines)

- Compute in `Single` exactly as written: one operation at a time, in the
  order shown, left to right; integer operands are converted to `Single`
  where they meet a `Single`. No reordering, factoring or precomputation.
- Reinterpret bits (`Single` ↔ `Cardinal`) with `Move` or a pointer cast
  (`PCardinal(@X)^`), not `absolute` on a global variable.
- Bytes are little-endian. `fp16` is IEEE binary16; `bf16` is the top 16
  bits of a binary32.

## `Fp16ToSingle(AH: Word): Single`

Purpose: convert an `fp16` value stored in a `Word` to the equivalent
`Single` (exact; every `fp16` is a `Single`).

Inputs:

- `AH` — `Word` — the 16-bit `fp16` value.

Steps:

1. Extract the fields: `E := (AH shr 10) and $1F`; `M := AH and $3FF`;
   sign bit `S := AH and $8000`.
2. Branch on `E`, in the facts' order:
   - `E = 0` (zero or subnormal): the result is `M * (1.0 / 16777216.0)`
     (that is `M·2^-24`, exact), negated when `S <> 0` (so `$8000` gives
     `−0.0`).
   - `E = 31`, `M = 0` (infinity): bits `(S shl 16) or $7F800000`.
   - `E = 31`, `M <> 0` (NaN): bits `(S shl 16) or $7FC00000 or (M shl 13)`
     (the quiet bit is set, as C does; so `$7C01` gives `$7FC02000`).
   - otherwise (normal): bits `(S shl 16) or ((E + 112) shl 23) or (M shl 13)`.
3. "Bits" in step 2 are the `Cardinal` bits of the returned `Single`, set
   by the bit reinterpretation in the general rules.

Errors: none (facts "Conversions").

Output:

- result — `Single` — the exact value of the `fp16` input.

Cases:

- UT-GgufQuant-001
- UT-GgufQuant-002
- UT-GgufQuant-003

## `SingleToFp16(AX: Single): Word`

Purpose: convert a `Single` to `fp16`, round to nearest, ties to even
(what C's `(_Float16)x` and the F16C instruction do).

Inputs:

- `AX` — `Single` — any fp32 value, including NaN, ±infinity, denormals.

Steps:

1. Bit view of `AX` as `U`: `Sign := (U shr 16) and $8000`;
   `E := (U shr 23) and $FF`; `M := U and $7FFFFF`.
2. Branch on `E`, in the facts' order:
   - `E = 255`: NaN (`M <> 0`) gives `Sign or $7E00 or (M shr 13)` (sign
     preserved, quiet NaN with only the top 3 mantissa bits); infinity
     gives `Sign or $7C00`.
   - `E >= 143` (`|X| >= 65536`): `Sign or $7C00` (infinity).
   - `E >= 113` (normal `fp16`): `R := ((E - 112) shl 10) or (M shr 13)`;
     `Rem := M and $1FFF`; if `Rem > $1000`, or `Rem = $1000` and `R` is
     odd, `R := R + 1` (a carry into the exponent is correct, up to
     `$7C00` = infinity; so 65504 stays 65504 = `$7BFF`, 65520 becomes
     infinity). Result `Sign or R`.
   - `E < 102`: `Sign` (±0).
   - otherwise (subnormal `fp16`): `M := M or $800000`;
     `Shift := 126 - E` (14..24); `R := M shr Shift`;
     `Half := 1 shl (Shift - 1)`; `Rem := M and ((1 shl Shift) - 1)`; if
     `Rem > Half`, or `Rem = Half` and `R` is odd, `R := R + 1`. Result
     `Sign or R`.

Errors: none (facts "Conversions").

Output:

- result — `Word` — the rounded `fp16` bits.

Cases:

- UT-GgufQuant-004
- UT-GgufQuant-005
- UT-GgufQuant-006
- UT-GgufQuant-007
- UT-GgufQuant-008

## `Bf16ToSingle(AH: Word): Single`

Purpose: convert a `bf16` value (the top 16 bits of a binary32) to its
`Single` with its lower 16 bits zeroed.

Inputs:

- `AH` — `Word` — the 16-bit `bf16` value.

Steps:

1. The result's bits are `Cardinal(AH) shl 16` (bit reinterpretation per
   the general rules).

Errors: none.

Output:

- result — `Single` — the `bf16` value with the low 16 mantissa bits zero.

Cases:

- UT-GgufQuant-009

## `RoundHalfAway(AX: Single): Single`

Purpose: round a `Single` to the nearest integer, halves away from zero
(C's `roundf`).

Inputs:

- `AX` — `Single` — finite only (the facts state only finite `X` are
  passed to it).

Steps:

1. `T := Int(AX)`.
2. If `Abs(AX - T) >= 0.5`, then `T := T + 1` when `AX > 0`, `T - 1` when
   `AX < 0`.
3. Result `T`.

Note: `Round` (halves to even) and `Trunc(AX + 0.5)` are not equivalent —
`Trunc(0.49999997 + 0.5)` gives 1 (facts "Conversions").

Errors: none.

Output:

- result — `Single` — the rounded value (for inputs in the test range,
  an integer-valued `Single`; the sign of zero is preserved).

Cases:

- UT-GgufQuant-010

## `DequantizeRow(ATType, ASrc, ACount, ADst)`

Purpose: dequantize a row of `ACount` elements of tensor type `ATType`
from a byte buffer into `Singles`.

Inputs:

- `ATType` — `Cardinal` — a GGUF tensor type id as per T01.
- `ASrc` — `PByte` — the row's bytes: `ACount / block` blocks one after
  the other; type-specific block layout in the per-type sections below.
- `ACount` — `Int64` — the element count; a multiple of the type's block
  size.
- `ADst` — `PSingle` — destination; the caller provides room for `ACount`
  `Singles`.

Steps:

1. Resolve `ATType`. An unsupported type id raises `EGgufError` with the
   message `"unsupported tensor type N"` (`N` = the id) (facts, unit
   interface).
2. If `ACount` is not a multiple of the type's block size, raise
   `EGgufError`; the message names the type and the count (facts, unit
   interface).
3. Dequantise starting at `ASrc` into `ADst`:
   - for the element-wise types (f32, f16, bf16), convert each element in
     element order with the per-type rule below;
   - for the block types (q8_0, q4_K, q5_K, q6_K), run the per-block rule
     below once per block, `ACount / block` blocks one after the other.
   - the element order is the order in the file.

Errors:

- `EGgufError`, message `"unsupported tensor type N"` (facts, unit
  interface).
- `EGgufError`, a message naming the type and the count, when `ACount` is
  not a multiple of the block size (facts, unit interface).

Output:

- `ADst` — `ACount` `Singles` written (bit-for-bit as specified).

Cases:

- UT-GgufQuant-015
- UT-GgufQuant-016
- UT-GgufQuant-017

## `DequantF32` (type id 0)

Purpose: dequantize one f32 element (trivial conversion).

Inputs:

- 4 bytes — the little-endian `Single` (4 bytes per element; element
  order is the order in the file).

Steps:

1. The output element is the 4 bytes reinterpreted as a `Single` (bit
   reinterpretation per the general rules).

Errors: none.

Output:

- 1 `Single`.

Cases:

- UT-GgufQuant-011

## `DequantF16` (type id 1)

Purpose: dequantize f16 elements to `Singles`.

Inputs:

- 2 bytes per element — `fp16` values (element order is the order in the
  file).

Steps:

1. For each element: `Fp16ToSingle` of the `Word`.

Errors: none.

Output:

- one `Single` per element.

Cases:

- UT-GgufQuant-012

## `DequantBf16` (type id 30)

Purpose: dequantize bf16 elements to `Singles`.

Inputs:

- 2 bytes per element — `bf16` values (element order is the order in the
  file).

Steps:

1. For each element: `Bf16ToSingle` of the `Word`.

Errors: none.

Output:

- one `Single` per element.

Cases:

- UT-GgufQuant-013

## `DequantQ8_0` (type id 8)

Purpose: dequantize one q8_0 block of 32 elements.

Inputs (34 bytes per block):

- `d` — fp16 — bytes 0..1
- `qs` — 32 × int8 — bytes 2..33

Steps:

1. `D := Fp16ToSingle(d)`.
2. For `j = 0..31`: next y := `qs[j] * D`.

Errors: none.

Output:

- 32 `Singles`.

Cases:

- UT-GgufQuant-014

## `DequantQ4_K` (type id 12)

Purpose: dequantize one q4_K block of 256 elements.

Inputs (144 bytes per block):

- `d` — fp16 — bytes 0..1
- `dmin` — fp16 — bytes 2..3
- `scales` — 12 bytes — bytes 4..15 (layout per facts)
- `qs` — 128 bytes — bytes 16..143

Steps (per block):

1. `D := Fp16ToSingle(d)`; `Mn := Fp16ToSingle(dmin)`.
2. `Q :=` start of `qs`; `Is := 0`; repeat four times (64 elements each):
   - `(Sc, M) := ScaleMin(Is, scales)`; `D1 := D * Sc`; `M1 := Mn * M`;
   - `(Sc, M) := ScaleMin(Is + 1, scales)`; `D2 := D * Sc`; `M2 := Mn * M`;
   - for `L = 0..31`: next y := `D1 * (Q[L] and $F) - M1`;
   - for `L = 0..31`: next y := `D2 * (Q[L] shr 4) - M2`;
   - `Q := Q + 32`; `Is := Is + 2`.

Errors: none.

Output:

- 256 `Singles`.

Cases:

- UT-GgufQuant-101

## `DequantQ5_K` (type id 13)

Purpose: dequantize one q5_K block of 256 elements.

Inputs (176 bytes per block):

- `d` — fp16 — bytes 0..1
- `dmin` — fp16 — bytes 2..3
- `scales` — 12 bytes — bytes 4..15 (layout per facts)
- `qh` — 32 bytes — bytes 16..47
- `qs` — 128 bytes — bytes 48..175

Steps (per block), as q4_K with `Q :=` start of `qs`, and two bit masks
`U1 := 1`, `U2 := 2` that start at each block:

1. `D := Fp16ToSingle(d)`; `Mn := Fp16ToSingle(dmin)`; `Q` and `Is` as in
   q4_K; repeat four times (64 elements each):
   - `(Sc, M) := ScaleMin(Is, scales)`; `D1 := D * Sc`; `M1 := Mn * M`;
   - `(Sc, M) := ScaleMin(Is + 1, scales)`; `D2 := D * Sc`; `M2 := Mn * M`;
   - first 32: for `L = 0..31`: next y :=
     `D1 * ((Q[L] and $F) + H1) - M1`, where `H1 := 16` if
     `(qh[L] and U1) <> 0` else 0 (`qh` indexed by `L`, 0..31; the same
     32 bytes for all four groups);
   - next 32: for `L = 0..31`: next y :=
     `D2 * ((Q[L] shr 4) + H2) - M2`, `H2` likewise with `U2`;
   - after the group: `Q := Q + 32`; `Is := Is + 2`; `U1 := U1 shl 2`;
     `U2 := U2 shl 2`.
2. The sum `( … ) + H` is an integer (0..31) converted to `Single` once
   (facts).

Errors: none.

Output:

- 256 `Singles`.

Cases:

- UT-GgufQuant-102

## `DequantQ6_K` (type id 14)

Purpose: dequantize one q6_K block of 256 elements.

Inputs (210 bytes per block):

- `ql` — 128 bytes — bytes 0..127
- `qh` — 64 bytes — bytes 128..191
- `scales` — 16 × int8 — bytes 192..207
- `d` — fp16 — bytes 208..209

Steps:

1. `D := Fp16ToSingle(d)`.
2. Two halves of 128 elements; `QL`, `QH`, `SC` start at `ql`, `qh`,
   `scales`; for each half, for `L = 0..31` with `Is := L div 16`:
   - `Q1 := ((QL[L] and $F) or (((QH[L] shr 0) and 3) shl 4)) - 32`
   - `Q2 := ((QL[L + 32] and $F) or (((QH[L] shr 2) and 3) shl 4)) - 32`
   - `Q3 := ((QL[L] shr 4) or (((QH[L] shr 4) and 3) shl 4)) - 32`
   - `Q4 := ((QL[L + 32] shr 4) or (((QH[L] shr 6) and 3) shl 4)) - 32`
   - `y[L] := D * SC[Is] * Q1`; `y[L + 32] := D * SC[Is + 2] * Q2`;
     `y[L + 64] := D * SC[Is + 4] * Q3`; `y[L + 96] := D * SC[Is + 6] * Q4`
     (`SC` entries are int8; `D * SC[..]` first, then `* Q`).
3. After each half: `y := y + 128`; `QL := QL + 64`; `QH := QH + 32`;
   `SC := SC + 8`.

Errors: none.

Output:

- 256 `Singles`.

Cases:

- UT-GgufQuant-103

## `ScaleMin(J, Scales)` (internal helper)

Purpose: return the 6-bit scale and 6-bit minimum for the `J`-th group
(`J = 0..7`) within a K-quant's 12-byte scale field.

Inputs:

- `J` — group index, 0..7
- `Scales` — 12 bytes — the K-quant scale bytes (layout per facts)

Steps (two 6-bit fields packed into the bytes):

- `J < 4`: `Sc := Scales[J] and 63`; `M := Scales[J + 4] and 63`.
- `J >= 4`:
  - `Sc := (Scales[J + 4] and $F) or ((Scales[J - 4] shr 6) shl 4)`
  - `M := (Scales[J + 4] shr 4) or ((Scales[J] shr 6) shl 4)`

Errors: none.

Output:

- `(Sc, M)` — the 6-bit scale and minimum for group `J`.

Called from: `DequantQ4_K` and `DequantQ5_K` (twice per group: `Is` and
`Is + 1`).

Cases:

- UT-GgufQuant-101
- UT-GgufQuant-102

## `DequantizeBytes(ATType, ABytes, ACount)`

Purpose: dequantize a whole tensor's bytes into a `TSingleArray`.

Inputs:

- `ATType` — `Cardinal` — a tensor type id as per T01.
- `ABytes` — `TBytes` — the tensor's bytes; `Length(ABytes)` must equal
  `GgufTensorBytes(ATType, [ACount])`.
- `ACount` — `Int64` — the element count.

Steps:

1. `Length(ABytes)` must equal `GgufTensorBytes(ATType, [ACount])`;
   otherwise raise `EGgufError` naming both lengths (facts, unit
   interface).
2. Run the same dequantisation as `DequantizeRow` over `ABytes`, into a
   new `TSingleArray` of `ACount` elements.

Errors:

- `EGgufError`, a message naming both lengths, when
  `Length(ABytes) <> GgufTensorBytes(ATType, [ACount])` (facts, unit
  interface).
- the `DequantizeRow` errors above (unsupported type; count not a
  multiple of the block size) apply to the dequantisation itself.

Output:

- result — `TSingleArray` — `ACount` `Singles`.

Cases:

- UT-GgufQuant-018

## `QuantizeRowQ8_0(ASrc, ACount, ADst)`

Purpose: quantize a row of `ACount` (a multiple of 32) `Singles` to Q8_0
blocks.

Inputs:

- `ASrc` — `PSingle` — `ACount` `Singles`; all finite (facts "Q8_0
  quantisation").
- `ACount` — `Int64` — a multiple of 32.
- `ADst` — `PByte` — the caller provides room for
  `ACount div 32 * 34` bytes.

Steps:

1. Non-finite input raises `EGgufError` (facts "Q8_0 quantisation").
2. For each block of 32 elements, in order: run the Q8_0 block quantiser
   below, block after block starting at `ADst`.

Errors:

- `EGgufError` for non-finite input (facts "Q8_0 quantisation").

Output:

- `ADst` — `ACount div 32` blocks of 34 bytes: `d` fp16 (block bytes
  0..1), `qs` 32 × int8 (block bytes 2..33), in element order.

Cases:

- UT-GgufQuant-019
- UT-GgufQuant-020
- UT-GgufQuant-021
- UT-GgufQuant-022
- UT-GgufQuant-023
- UT-GgufQuant-024

## Q8_0 block quantiser (internal helper)

Purpose: quantize 32 finite `Singles` into one 34-byte q8_0 block.

Inputs:

- `x[0..31]` — 32 finite `Singles`.

Steps (facts' order):

1. `Amax := 0`; for each `j`: `Amax := Max(Amax, Abs(x[j]))`
   (`Math.Max` on `Singles`).
2. `D := Amax / 127`. Check step 3 **before** computing `1 / D`; then
   `Id := 1 / D` if `D <> 0`, else `0`.
3. If `0 < D <= 2^-128` (the `Single` `2.93873588e-39`,
   `Single(1) / 2^128`; `1 / D` is infinite there and finite for the next
   larger `Single`), do not compute `1 / D`: store `d := 0` and every
   `q := 0` (upstream's result is undefined there; this choice
   dequantises to 0, within half a step). No floating-point exception may
   be raised for any finite input.
4. `d := SingleToFp16(D)` (block bytes 0..1).
5. For each `j`: `X0 := x[j] * Id`; `q[j] := Trunc(RoundHalfAway(X0))` as
   int8 (always in -127..127).

Note: the quantiser multiplies by `1 / D`; it does **not** divide by `D`
(DESIGN 8.15 says `x / d`; the two differ in the last bit for some
inputs, and upstream multiplies) (facts "Q8_0 quantisation").

Errors:

- `EGgufError` for non-finite input (facts "Q8_0 quantisation").

Output:

- one 34-byte block: `d` fp16 (bytes 0..1), `qs` 32 × int8 (bytes 2..33).

Cases:

- UT-GgufQuant-019
- UT-GgufQuant-020
- UT-GgufQuant-021
- UT-GgufQuant-022
- UT-GgufQuant-023
- UT-GgufQuant-024

## `QuantizeQ8_0(AValues)`

Purpose: quantize an in-memory array of values to Q8_0 bytes.

Inputs:

- `AValues` — `TSingleArray` — the elements to quantize; the length must
  be a multiple of 32.

Steps:

1. If `Length(AValues) mod 32 <> 0`, raise `EGgufError` (facts, unit
   interface).
2. Run `QuantizeRowQ8_0` over all elements of `AValues`; the result is
  `Length(AValues) div 32 * 34` bytes.

Errors:

- `EGgufError` if `Length mod 32 <> 0` (facts, unit interface).
- `EGgufError` for non-finite input (facts "Q8_0 quantisation").

Output:

- result — `TBytes` — `Length(AValues) div 32` blocks of 34 bytes each.

Cases:

- UT-GgufQuant-025
