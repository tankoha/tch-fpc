# UT cases — unit `GgufQuant` (T02)

Unit under test: `src/GgufQuant.pas` — the `Fp16ToSingle`, `SingleToFp16`,
`Bf16ToSingle`, `RoundHalfAway`, `DequantizeRow`, `DequantizeBytes`,
`QuantizeRowQ8_0`, `QuantizeQ8_0` entry points (and the internal per-type
dequantisers / Q8_0 quantiser). `Fp16ToSingle`, `SingleToFp16`, the type
ids, the block-byte counts and `GgufTensorBytes` are already covered by
`GgufFile` (T01); this table covers the *quanti/singularity* side only.
Base: `tasks/T02-facts.md` — sections "Block layouts and
dequantisation", "Q8_0 quantisation", "API" and "Unit tests", and the
errors they fix. **Note:** there is no `docs/detailed-design/T02-detailed.md`
for this step; the facts are final and state every error/message (they are
the authoritative list). The messages are asserted exactly where the facts
fix them and as "EGgufError (names …)" otherwise.

Test arrangement (facts "Unit tests"): every case builds its block bytes
in the test and compares the dequantised/quantised result **bit for bit**
(compare the `Cardinal` bits of the `Single`s, never with a tolerance).
The hand-computed blocks use `d = 1.0` (fp16 `$3C00`) and `dmin = 0.5`
(fp16 `$3800`) so every result is an exact binary fraction and is written
out fully in "Hand-computed blocks" below. UT never needs the rehearsal
model, the GPU or `QLORA_MODEL`.

| Case ID | Title | Input (short) | Expected result | Notes |
|---|---|---|---|---|
| UT-GgufQuant-001 | Fp16ToSingle: finite values incl. ±0, normal, max finite | $0000, $8000, $3C00 (1.0), $C000 (−2.0), $7BFF (65504) | +0→$00000000, −0→$80000000, 1.0→$3F800000, −2.0→$C0000000, 65504→$477FE000 (bit-exact; corrected 2026-11-21: $4C000000 is the bits of 33554432.0 — see the corrected f16 row el6) | facts "Unit tests" |
| UT-GgufQuant-002 | Fp16ToSingle: infinities and NaNs | $7C00 (+inf), $FC00 (−inf), qNaN $7E00, sNaN $7C01 | +inf→$7F800000, −inf→$FF800000, $7E00→$7FC00000, $7C01→$7FC02000 (bit-exact) | facts "Unit tests" |
| UT-GgufQuant-003 | Fp16ToSingle: subnormals (smallest, largest, negative) | $0001, $03FF, $8001 | $0001→2^-24, $03FF→largest subnormal, $8001→−(smallest subnormal) (bit-exact) | facts "Unit tests" |
| UT-GgufQuant-004 | SingleToFp16: the finite values above back to their bits | 1.0, −2.0, 65504.0, +0.0, −0.0 | 1.0→$3C00, −2.0→$C000, 65504→$7BFF, +0→$0000, −0→$8000 | facts "Unit tests" |
| UT-GgufQuant-005 | SingleToFp16: infinities and NaNs | +inf, −inf, a quiet NaN, a negative NaN | +inf→$7C00, −inf→$FC00, NaN→a quiet NaN (sign preserved) | facts "Unit tests" |
| UT-GgufQuant-006 | SingleToFp16: ties round to even | 1 + 2^-11; 1 + 3·2^-11 | 1+2^-11→$3C00, 1+3·2^-11→$3C02 (both exact, tie-to-even) | facts "Unit tests" |
| UT-GgufQuant-007 | SingleToFp16: max finite / overflow boundary | 65519; 65520 | 65519→$7BFF, 65520→$7C00 (rounds up to +inf) (bit-exact) | facts "Unit tests" |
| UT-GgufQuant-008 | SingleToFp16: subnormal boundary (underflow + rounding) | 2^-25; 2^-25 + 2^-35 | 2^-25→$0000, 2^-25+2^-35→$0001 (bit-exact) | facts "Unit tests" |
| UT-GgufQuant-009 | Bf16ToSingle: finite + inf | $3F80 (1.0), $C000 (−2.0), $7F80 (+inf) | 1.0→$3F800000, −2.0→$C0000000, +inf→$7F800000 (bit-exact) | facts "Unit tests" |
| UT-GgufQuant-010 | RoundHalfAway: halves away from zero, non-halves, −0.0 | 0.5, −0.5, 1.5, 2.5, −2.5, 0.49999997, 126.5, −0.0 | 0.5→1, −0.5→−1, 1.5→2, 2.5→3, −2.5→−3, 0.49999997→0, 126.5→127, −0.0→−0.0 (bit-exact) | facts "Unit tests"; §57–60 |
| UT-GgufQuant-011 | DequantizeRow f32 block (4 elements, exact) | see f32 block, input 16 bytes | see f32 expected (4 exact fp32) | facts "each type: one hand-computed block" |
| UT-GgufQuant-012 | DequantizeRow f16 block (8 elements, exact) | see f16 block, input 16 bytes | see f16 expected (8 exact fp32) | facts "each type: one hand-computed block" |
| UT-GgufQuant-013 | DequantizeRow bf16 block (8 elements, exact) | see bf16 block, input 16 bytes | see bf16 expected (8 exact fp32) | facts "each type: one hand-computed block" |
| UT-GgufQuant-014 | DequantizeRow q8_0 block (32 elements, negative quant) | see q8_0 block, input 34 bytes | see q8_0 expected (32 exact fp32, includes negatives) | facts "for q8_0 a negative quant" |
| UT-GgufQuant-015 | DequantizeRow two blocks (second block aligns) | two q8_0 blocks, 64 elements | see 2-block expected (64 exact fp32) | facts "two blocks (the second block starts at the right byte)" |
| UT-GgufQuant-016 | DequantizeRow unsupported type (2) | AType = 2 | EGgufError; message names the type ("unsupported tensor type 2") | facts "an unsupported type (2)" |
| UT-GgufQuant-017 | DequantizeRow count not a multiple of the block size | q8_0 (32/block), ACount = 33 | EGgufError; message names the type and the count | facts "a count that is not a multiple of the block size" |
| UT-GgufQuant-018 | DequantizeBytes wrong byte length | q8_0, ACount = 32 (needs 34 bytes) but ALen = 33 | EGgufError; message names both lengths | facts "DequantizeBytes with a wrong byte length" |
| UT-GgufQuant-019 | QuantizeRowQ8_0: Amax = 127 (d = 1.0, q = x) | x block with one element 127, others small; D = 1.0 | d = $3C00 (00 3C); q bytes = the x values as int8 (q = x) | facts "a block whose Amax is 127 (d = 1.0, q = x)" |
| UT-GgufQuant-020 | QuantizeRowQ8_0: block of zeros | x = 0 for all 32 | d = $0000 (00 00); all q = 0 (32 × 00) | facts "a block of zeros (d = 0, q = 0)" |
| UT-GgufQuant-021 | QuantizeRowQ8_0: exact half after scaling | x with 0.5 and −0.5, and 127 (Amax = 127, D = 1.0) | q(+0.5) = 1, q(−0.5) = −1, q(127) = 127, rest 0 | facts "0.5 gives q = 1; −0.5 gives −1" |
| UT-GgufQuant-022 | QuantizeRowQ8_0: tiny Amax (no exception) | Amax = 1e-40 (D <= 2^-128) | d = 0 ($0000), every q = 0, no floating-point exception | facts "tiny Amax (1e-40: d = 0, q = 0, no exception)" |
| UT-GgufQuant-023 | QuantizeRowQ8_0: non-finite input | one element +inf (or NaN) | EGgufError | facts "a non-finite input (EGgufError)" |
| UT-GgufQuant-024 | QuantizeRowQ8_0: dequantise∘quantise within half a step | a block of varied finite x; D = Amax/127 | for every j, |y[j] − x[j]| ≤ D/2 + 1 ulp | facts "within half a step (\|y − x\| <= D/2 + 1 ulp)" |
| UT-GgufQuant-025 | QuantizeQ8_0: length not a multiple of 32 | TSingleArray of Length 33 | EGgufError ("if Length mod 32 <> 0") | facts "QuantizeQ8_0 with a length that is not a multiple of 32" |
| UT-GgufQuant-026 | SingleToFp16: \|X\| >= 65536 gives infinity | 65536.0, 1.0e10, −1.0e10 | 65536.0→$7C00, 1.0e10→$7C00, −1.0e10→$FC00 (bit-exact) | facts "overflow: \|X\| >= 65536 → (signed) infinity" |
| UT-GgufQuant-027 | QuantizeRowQ8_0: values in [1, 2) are finite | a 32-element block: x[0] = 1.5, x[1] = −1.25, the rest 0 | d = $220C, q[0] = 127, q[1] = −106; output bytes 0C 22 7F 96 followed by 30 bytes 00 (bit-exact) | BUG-006 |
| UT-GgufQuant-101 | DequantizeRow q4_K block (256 elements, exact) | see q4_K block, input 144 bytes | see q4_K expected (256 exact fp32) | Stage 2; facts "for q4_K/q5_K the scales … J < 4 and J >= 4" |
| UT-GgufQuant-102 | DequantizeRow q5_K block (256 elements, exact) | see q5_K block, input 176 bytes | see q5_K expected (256 exact fp32) | Stage 2; facts "for q5_K the high bit set and clear in each of the four groups" |
| UT-GgufQuant-103 | DequantizeRow q6_K block (256 elements, exact) | see q6_K block, input 210 bytes | see q6_K expected (256 exact fp32) | Stage 2; facts "for q6_K every one of Q1..Q4 and a negative scale" |

## Hand-computed blocks

All blocks use `d = $3C00` (D = 1.0) and, where present, `dmin = $3800`
(Mn = 0.5) so every result is an exact binary fraction. Bytes are shown
little-endian as they appear in the file. "el N" = the Nth output element
(0-based); "× n" = that value on n consecutive elements.

### `ScaleMin(J, Scales)` for the K-quant scale bytes below

The shared 12 scale bytes (little-endian) are
`82 C1 03 C5 40 40 00 02 24 35 FF 27`, i.e. `S[0..11]` =
130, 193, 3, 197, 64, 64, 0, 2, 36, 53, 255, 39.

| J | Sc (D-side) | M (min-side) | D1/M1 (D=1, Mn=0.5) |
|---|---|---|---|
| 0 | 2 | 0 | 2 / 0 |
| 1 | 1 | 0 | 1 / 0 |
| 2 | 3 | 0 | 3 / 0 |
| 3 | 5 | 2 | 5 / 1 |
| 4 | 36 | 18 | 36 / 9 |
| 5 | 53 | 19 | 53 / 9.5 |
| 6 | 15 | 15 | 15 / 7.5 |
| 7 | 55 | 2 | 55 / 1 |

(Groups of the K-types use `Is = 0,2,4,6`: group g uses `D1 = D·Sc(2g)`,
`M1 = Mn·M(2g)` for the low nibbles and `D2 = D·Sc(2g+1)`,
`M2 = Mn·M(2g+1)` for the high nibbles.)

### f32 block — `DequantizeRow(0, src, 4, dst)`, 16 input bytes

Input (16 bytes): `00 00 C0 3F  00 00 12 C0  00 00 00 3F  00 00 00 80`
(el0 = $3FC00000, el1 = $C0120000, el2 = $3F000000, el3 = $80000000).

| el | expected fp32 (bit-exact) |
|---|---|
| 0 | 1.5 ($3FC00000) |
| 1 | −2.28125 ($C0120000) |
| 2 | 0.5 ($3F000000) |
| 3 | −0.0 ($80000000) |

### f16 block — `DequantizeRow(1, src, 8, dst)`, 16 input bytes

Input (16 bytes): `00 00  00 80  00 3C  00 C0  00 7C  00 FC  FF 7B  00 7E`
(el0..7 = $0000, $8000, $3C00, $C000, $7C00, $FC00, $7BFF, $7E00).

| el | expected fp32 (bit-exact) |
|---|---|
| 0 | +0.0 ($00000000) |
| 1 | −0.0 ($80000000) |
| 2 | 1.0 ($3F800000) |
| 3 | −2.0 ($C0000000) |
| 4 | +inf ($7F800000) |
| 5 | −inf ($FF800000) |
| 6 | 65504 ($477FE000) |
| 7 | qNaN ($7FC00000) |

### bf16 block — `DequantizeRow(30, src, 8, dst)`, 16 input bytes

Input (16 bytes): `80 3F  00 C0  00 40  00 3F  00 BF  80 7F  80 FF  7F 7F`
(el0..7 = $3F80, $C000, $4000, $3F00, $BF00, $7F80, $FF80, $7F7F).

| el | expected fp32 (bit-exact) |
|---|---|
| 0 | 1.0 ($3F800000) |
| 1 | −2.0 ($C0000000) |
| 2 | 2.0 ($40000000) |
| 3 | 0.5 ($3F000000) |
| 4 | −0.5 ($BF000000) |
| 5 | +inf ($7F800000) |
| 6 | −inf ($FF800000) |
| 7 | 3.3895314e38, finite ($7F7F0000) |

### q8_0 block — `DequantizeRow(8, src, 32, dst)`, 34 input bytes

Layout: d: fp16 (bytes 0..1), qs: 32×int8 (bytes 2..33). D = 1·(d/1) =
Fp16ToSingle(d) = 1.0 (d = $3C00), so y[j] = qs[j]·1.0.

Input (34 bytes):
d = `00 3C`; qs(32) = `7F 7F 00 05 FB 08 F8 96 60 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00`.

| el | qs[j] | expected (bit-exact) |
|---|---|---|
| 0 | 127 | 127 (0x42FE0000) |
| 1 | 127 | 127 (0x42FE0000) |
| 2 | 0 | 0 |
| 3 | 5 | 5 (0x40A00000) |
| 4 | −5 | −5 (0xC0A00000) |
| 5 | 8 | 8 (0x41000000) |
| 6 | −8 | −8 (0xC1000000) |
| 7 | −106 (byte 0x96) | −106 (0xC2D40000) |
| 8 | 96 | 96 (0x42C00000) |
| 9..31 | 0 | 0 |

(negative quant at el4/el6) — all exact.

### Two-block q8_0 — `DequantizeRow(8, src, 64, dst)`

Block A (elements 0..31): d = $3C00 (1.0), qs[0] = 127 (0x7F), qs[1] = −1 (0xFF), rest 0.
Block B (elements 32..63): d = $4000 (2.0), qs[0] = 1 (0x01), qs[1] = −3 (0xFD), rest 0.

| el | expected (bit-exact) |
|---|---|
| 0 | 127 |
| 1 | −1 |
| 2..31 | 0 |
| 32 | 2 (0x40000000) |
| 33 | −6 (0xC0C00000) |
| 34..63 | 0 |

This exercises that the second block's `d` (2.0) and quants start at byte 34
and produce the right values (block A uses d = 1.0, block B uses d = 2.0).

### q4_K block — `DequantizeRow(12, src, 256, dst)`, 144 input bytes

Layout: d (2, $3C00=1.0), dmin (2, $3800=0.5), scales (12, as above),
qs (128). In **each** group's 32 qs bytes the first 4 bytes (group-local
L=0..3) are `$21` (low nibble = 1, high nibble = 2) and the remaining
28 bytes are `$00`; groups do not overlap.

Input (144 bytes) = d `00 3C` ‖ dmin `00 38` ‖ scales
`82 C1 03 C5 40 40 00 02 24 35 FF 27` ‖ qs, where the 32-byte group g (g=0..3,
at file bytes 32g..32g+31) is `21 21 21 21` ‖ `00 ×28`.

| group g | Is | el (low nib, L0..3) | el (low nib, L4..31) | el (high nib, L0..3) | el (high nib, L4..31) |
|---|---|---|---|---|---|
| 0 | 0 | +2 (×4) | 0 (×28) | +2 (×4) | 0 (×28) |
| 1 | 2 | +3 (×4) | 0 (×28) | +9 (×4) | −1 (×28) |
| 2 | 4 | +27 (×4) | −9 (×28) | +96.5 (×4) | −9.5 (×28) |
| 3 | 6 | +7.5 (×4) | −7.5 (×28) | +109 (×4) | −1 (×28) |

By absolute element index (256 values, all exact):
`el0..3 = 2`; `el4..31 = 0`; `el32..35 = 2`; `el36..63 = 0`;
`el64..67 = 3`; `el68..95 = 0`; `el96..99 = 9`; `el100..127 = −1`;
`el128..131 = 27`; `el132..159 = −9`; `el160..163 = 96.5`; `el164..191 = −9.5`;
`el192..195 = 7.5`; `el196..223 = −7.5`; `el224..227 = 109`; `el228..255 = −1`.

(Covers `ScaleMin` at J < 4 (groups 0,1: J0..J3) and J ≥ 4 (groups 2,3:
J4..J7, the high-bit branch) and negative min-side results.)

### q5_K block — `DequantizeRow(13, src, 256, dst)`, 176 input bytes

Layout: d ($3C00=1.0), dmin ($3800=0.5), scales (12, as above), qh (32),
qs (128). qh bytes 0..3 are `$FF, $55, $AA, $00`; qh bytes 4..31 are `$00`.
qs is laid out exactly as in the q4_K block (per group L0..3 = `$21`,
rest `$00`). The high-bit mask for group g is `U1 = 4^g`, `U2 = 2·U1`.

Input (176 bytes) = d `00 3C` ‖ dmin `00 38` ‖ scales `82 C1 03 C5 40 40 00 02 24 35 FF 27`
‖ qh `FF 55 AA 00` ‖ `00 ×28` ‖ qs (128, as in the q4_K block).

| group g | el (low nib, L0..3) | el (low nib, L4..31) | el (high nib, L0..3) | el (high nib, L4..31) |
|---|---|---|---|---|
| 0 | +34, +34, +2, +2 | 0 (×28) | +18, +2, +18, +2 | 0 (×28) |
| 1 | +51, +51, +3, +3 | 0 (×28) | +89, +9, +89, +9 | −1 (×28) |
| 2 | +603, +603, +27, +27 | −9 (×28) | +944.5, +96.5, +944.5, +96.5 | −9.5 (×28) |
| 3 | +247.5, +247.5, +7.5, +7.5 | −7.5 (×28) | +989, +109, +989, +109 | −1 (×28) |

By absolute element index (256 values, all exact):
`el0..3 = 34,34,2,2`; `el4..31 = 0`; `el32..35 = 18,2,18,2`; `el36..63 = 0`;
`el64..67 = 51,51,3,3`; `el68..95 = 0`; `el96..99 = 89,9,89,9`; `el100..127 = −1`;
`el128..131 = 603,603,27,27`; `el132..159 = −9`; `el160..163 = 944.5,96.5,944.5,96.5`; `el164..191 = −9.5`;
`el192..195 = 247.5,247.5,7.5,7.5`; `el196..223 = −7.5`; `el224..227 = 989,109,989,109`; `el228..255 = −1`.

(At L0 = `$FF` the H bit is set in all groups; at L1 = `$55` only U1 (low)
is set; at L2 = `$AA` only U2 (high) is set; at L3 = `$00` it is clear — so
in each of the four groups the high bit is set for some L and clear for
others.)

### q6_K block — `DequantizeRow(14, src, 256, dst)`, 210 input bytes

Layout: ql (128), qh (64), scales (16×int8), d (2, $3C00=1.0).
ql[0] = `$12`, ql[32] = `$34`, all other ql = `$00`.
qh[0] = `$0F`, all other qh = `$00`.

| scale idx | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| value | −2 | 3 | −1 | 5 | 7 | −3 | 4 | −6 | 1 | −4 | 2 | 8 | −5 | 6 | −1 | 10 |
| byte (int8) | FE | 03 | FF | 05 | 07 | FD | 04 | FA | 01 | FC | 02 | 08 | FB | 06 | FF | 0A |

Input (210 bytes) = ql (128, `12` at byte 0, `34` at byte 32, `00` elsewhere)
‖ qh (64, `0F` at byte 0, `00` elsewhere) ‖ scales (16, FE 03 FF 05 07 FD
04 FA 01 FC 02 08 FB 06 FF 0A) ‖ d `00 3C`.

With D = 1.0 the half 0 produces `Q1 = 18, Q2 = 20, Q3 = −31, Q4 = −29`
(non-zero, so every one of Q1..Q4 appears) and the negative scale
`SC[0] = −2`:
half 0 → `el0 = −2·18 = −36`, `el32 = −1·20 = −20`, `el64 = 7·(−31) = −217`,
`el96 = 4·(−29) = −116`.

Full expected (256 values, all exact):
`el0 = −36`; `el1..15 = 64`; `el16..31 = −96`; `el32 = −20`;
`el33..47 = 32`; `el48..63 = −160`; `el64 = −217`; `el65..79 = −224`;
`el80..95 = 96`; `el96 = −116`; `el97..111 = −128`; `el112..127 = 192`;
`el128..143 = −32`; `el144..159 = 128`; `el160..175 = −64`; `el176..191 = −256`;
`el192..207 = 160`; `el208..223 = −192`; `el224..239 = 32`; `el240..255 = −320`.

(The second half uses scales idx 8..15 and all-zero quants → Q = −32; the
special element in each of the four Q lanes is in the first half.)

## Coverage against the facts' "Unit tests" list

- Fp16ToSingle (±0, subnormals, 1.0, −2.0, 65504, ±inf, qNaN, sNaN, negative
  subnormal) → UT-GgufQuant-001..003.
- SingleToFp16 (finite round-trip, ties-to-even, 65519/65520, 2^-25
  boundary, −0.0, NaN sign) → UT-GgufQuant-004..008.
- Bf16ToSingle (1.0, −2.0, +inf) → UT-GgufQuant-009.
- RoundHalfAway (8 listed values) → UT-GgufQuant-010.
- one hand-computed block per type, with type-specific coverage (q8_0
  negative quant; q4_K/q5_K ScaleMin J<4 and J>=4; q5_K high bit set/clear in
  each group; q6_K Q1..Q4 + negative scale) → UT-GgufQuant-011..014 (Stage 1)
  and -101..-103 (Stage 2), all in "Hand-computed blocks".
- DequantizeRow two blocks / unsupported type (2) / count not a multiple of
  the block size; DequantizeBytes wrong byte length → UT-GgufQuant-015..018.
- QuantizeRowQ8_0 Amax=127, zeros, exact half, tiny Amax, non-finite,
  round-trip within half a step; QuantizeQ8_0 length not ×32 →
  UT-GgufQuant-019..025.

All errors the facts fix (unsupported type 2, count not a multiple of the
block size, DequantizeBytes wrong length, QuantizeRowQ8_0 non-finite,
QuantizeQ8_0 length mod 32) are present: UT-GgufQuant-016, -017, -018, -023,
-025.

## Corrections by the harness owner (2026-11-21)

Checked against the facts with an independent implementation: the K blocks
(q4_K, q5_K, q6_K) were right; four stage-1 entries were wrong and are
corrected above: the f32 input bytes (now the little-endian bytes of the
listed values), f16 el6 (65504 is $477FE000), bf16 el7 ($7F7F0000 is a
finite value, not a NaN) and q8_0 el7 (byte $96 is the int8 −106).
