# T02 facts (checked by the harness owner)

T02 is the unit `GgufQuant`: the scalar dequantisers of the tensor types of
T01 and the Q8_0 quantiser. It is the reference for T03 (`GgufQuantT`, the
same in tensor ops) and T18 (the merge writes Q8_0), so it must agree with
upstream llama.cpp **bit for bit**. These facts were taken from upstream
`ggml/src/ggml-quants.c` at `39febd2ad` and checked by the harness owner: a
Pascal version written exactly as below reproduced the upstream C code on
every tensor of the rehearsal model (320 dequantised, 284 re-quantised, all
equal); `SingleToFp16` agrees with C on all 2^32 floats, `Fp16ToSingle`
on all 65536 halves, and `RoundHalfAway` with C's `roundf` on about a
million finite floats spread over the whole range. They are final; where a
T02 document differs, these facts win. Do not investigate the format further, do not read other units'
sources (except the `GgufFile` interface below) and do not write probes.

## Arithmetic (applies to every routine)

- Compute in `Single` (fp32) exactly as written: one operation at a time,
  in the order shown, left to right; integer operands are converted to
  `Single` where they meet a `Single`. FPC 3.2.2 on x86_64 does this with
  plain expressions (no `Double` temporaries, no `Math` routines other
  than those named here). Do not reorder, factor out or precompute.
- To reinterpret bits (Single <-> Cardinal) use `Move` or a pointer cast
  (`PCardinal(@X)^`), not `absolute` on a global variable: with -O2 FPC may
  keep the old value in a register (measured).
- Bytes are little-endian. `fp16` is IEEE binary16, `bf16` the top 16 bits
  of a binary32.

## Conversions

`Fp16ToSingle(H: Word): Single` (exact; every fp16 is a Single):
- E := (H shr 10) and $1F; M := H and $3FF; sign bit S := H and $8000.
- E = 0 (zero or subnormal): result := M * (1.0 / 16777216.0) (that is
  M·2^-24, exact), negated when S <> 0 (so $8000 gives -0.0).
- E = 31, M = 0: infinity, bits (S shl 16) or $7F800000.
- E = 31, M <> 0: NaN, bits (S shl 16) or $7FC00000 or (M shl 13) (the
  quiet bit is set, as C does; so $7C01 gives $7FC02000).
- otherwise: bits (S shl 16) or ((E + 112) shl 23) or (M shl 13).

`SingleToFp16(X: Single): Word`, round to nearest, ties to even (what C's
`(_Float16)x` and the F16C instruction do). With U the bits of X,
Sign := (U shr 16) and $8000, E := (U shr 23) and $FF, M := U and $7FFFFF:
- E = 255: NaN (M <> 0) gives Sign or $7E00 or (M shr 13); infinity gives
  Sign or $7C00.
- E >= 143 (|X| >= 65536): Sign or $7C00 (infinity).
- E >= 113 (normal fp16): R := ((E - 112) shl 10) or (M shr 13);
  Rem := M and $1FFF; if Rem > $1000, or Rem = $1000 and R is odd, R := R + 1
  (a carry into the exponent is correct, up to $7C00 = infinity);
  result Sign or R. So 65504 stays 65504 ($7BFF), 65520 becomes infinity.
- E < 102: Sign (±0).
- otherwise (subnormal fp16): M := M or $800000; Shift := 126 - E (14..24);
  R := M shr Shift; Half := 1 shl (Shift - 1); Rem := M and ((1 shl Shift) - 1);
  if Rem > Half, or Rem = Half and R is odd, R := R + 1; result Sign or R.

`Bf16ToSingle(H: Word): Single`: the bits `Cardinal(H) shl 16`.

`RoundHalfAway(X: Single): Single` = C's `roundf` (halves away from zero):
T := Int(X); if Abs(X - T) >= 0.5 then T := T + 1 when X > 0, T - 1 when
X < 0; result T. (`Round` rounds halves to even and `Trunc(X + 0.5)` is
wrong for 0.49999997, which gives 1.) Only finite X are passed to it.

## Block layouts and dequantisation

Type ids, block size (elements) and bytes as in T01. Element order is the
order in the file; a row of `ACount` elements is `ACount / block` blocks
one after the other.

- **f32 (0):** the 4 bytes are the Single.
- **f16 (1):** `Fp16ToSingle` of each Word. **bf16 (30):** `Bf16ToSingle`.
- **q8_0 (8), 34 bytes:** d: fp16 (bytes 0..1), qs: 32 × int8 (bytes 2..33).
  D := Fp16ToSingle(d); y[j] := qs[j] * D for j = 0..31.
- **q4_K (12), 144 bytes:** d: fp16 (0..1), dmin: fp16 (2..3),
  scales: 12 bytes (4..15), qs: 128 bytes (16..143).
- **q5_K (13), 176 bytes:** d (0..1), dmin (2..3), scales: 12 bytes (4..15),
  qh: 32 bytes (16..47), qs: 128 bytes (48..175).
- **q6_K (14), 210 bytes:** ql: 128 bytes (0..127), qh: 64 bytes (128..191),
  scales: 16 × int8 (192..207), d: fp16 (208..209).

`ScaleMin(J, Scales) -> (Sc, M)` (6-bit scale and min, J = 0..7, bytes):
- J < 4: Sc := Scales[J] and 63; M := Scales[J + 4] and 63.
- J >= 4: Sc := (Scales[J + 4] and $F) or ((Scales[J - 4] shr 6) shl 4);
  M := (Scales[J + 4] shr 4) or ((Scales[J] shr 6) shl 4).

**q4_K**, per block: D := Fp16ToSingle(d); Mn := Fp16ToSingle(dmin);
Q := start of qs; Is := 0; four times (64 elements each):
- (Sc, M) := ScaleMin(Is, scales); D1 := D * Sc; M1 := Mn * M;
- (Sc, M) := ScaleMin(Is + 1, scales); D2 := D * Sc; M2 := Mn * M;
- for L = 0..31: next y := D1 * (Q[L] and $F) - M1;
- for L = 0..31: next y := D2 * (Q[L] shr 4) - M2;
- Q := Q + 32; Is := Is + 2.

**q5_K**, per block: as q4_K with Q := start of qs, and two bit masks
U1 := 1, U2 := 2 that start at each block:
- first 32: next y := D1 * ((Q[L] and $F) + H1) - M1, where H1 := 16 if
  (qh[L] and U1) <> 0 else 0 (qh indexed by L, 0..31, the same 32 bytes
  for all four groups);
- next 32: next y := D2 * ((Q[L] shr 4) + H2) - M2, H2 likewise with U2;
- after the group: Q := Q + 32; Is := Is + 2; U1 := U1 shl 2; U2 := U2 shl 2.
The sum `(… ) + H` is an integer (0..31) converted to Single once.

**q6_K**, per block: D := Fp16ToSingle(d); two halves of 128 elements;
QL, QH, SC start at ql, qh, scales; for each half, for L = 0..31 with
Is := L div 16:
- Q1 := ((QL[L] and $F) or (((QH[L] shr 0) and 3) shl 4)) - 32
- Q2 := ((QL[L + 32] and $F) or (((QH[L] shr 2) and 3) shl 4)) - 32
- Q3 := ((QL[L] shr 4) or (((QH[L] shr 4) and 3) shl 4)) - 32
- Q4 := ((QL[L + 32] shr 4) or (((QH[L] shr 6) and 3) shl 4)) - 32
- y[L] := D * SC[Is] * Q1; y[L + 32] := D * SC[Is + 2] * Q2;
  y[L + 64] := D * SC[Is + 4] * Q3; y[L + 96] := D * SC[Is + 6] * Q4
  (SC entries are int8; `D * SC[..]` first, then `* Q`);
- after the half: y := y + 128; QL := QL + 64; QH := QH + 32; SC := SC + 8.

## Q8_0 quantisation

Per block of 32 inputs x[0..31] (finite Singles):
- Amax := 0; for each j: Amax := Max(Amax, Abs(x[j])) (`Math.Max` on Singles).
- D := Amax / 127. Check the next item **before** computing 1 / D. Then
  Id := 1 / D if D <> 0, else 0.
- If 0 < D <= 2^-128 (the Single 2.93873588e-39, `Single(1) / 2^128`;
  measured: 1 / D is infinite there and finite for the next larger Single),
  do not compute 1 / D: store d := 0 and every q := 0 (upstream's result is
  undefined there; this choice dequantises to 0, within half a step). No
  floating-point exception may be raised for any finite input.
- d := SingleToFp16(D) (bytes 0..1).
- for each j: X0 := x[j] * Id; q[j] := Trunc(RoundHalfAway(X0)) as int8
  (always in -127..127).

Note: the quantiser multiplies by `1 / D`; it does **not** divide by D
(DESIGN 8.15 says `x / d`; the two differ in the last bit for some inputs,
and upstream multiplies). A non-finite input raises `EGgufError`.

## Unit `GgufQuant` (interface, final)

```pascal
unit GgufQuant;
{$mode objfpc}{$H+}
interface
uses SysUtils, GgufFile;   { EGgufError, GgufTensorBytes, GgufTypeName }
type
  TSingleArray = array of Single;

function Fp16ToSingle(AH: Word): Single;
function SingleToFp16(AX: Single): Word;
function Bf16ToSingle(AH: Word): Single;
function RoundHalfAway(AX: Single): Single;

{ ACount elements of type ATType from ASrc^ into ADst^. Raises EGgufError
  for an unsupported type ("unsupported tensor type N") or an ACount that is
  not a multiple of the block size (message names the type and the count). }
procedure DequantizeRow(ATType: Cardinal; ASrc: PByte; ACount: Int64; ADst: PSingle);
{ The same for a tensor's bytes; Length(ABytes) must equal
  GgufTensorBytes(ATType, [ACount]), else EGgufError naming both lengths. }
function DequantizeBytes(ATType: Cardinal; const ABytes: TBytes; ACount: Int64): TSingleArray;

{ ACount (a multiple of 32) Singles from ASrc^ into ACount div 32 * 34 bytes at ADst^. }
procedure QuantizeRowQ8_0(ASrc: PSingle; ACount: Int64; ADst: PByte);
function QuantizeQ8_0(const AValues: TSingleArray): TBytes;   { EGgufError if Length mod 32 <> 0 }
```

`GgufFile` (T01, final) supplies `EGgufError`, `GgufTypeName(ATType)` and
`GgufTensorBytes(ATType; const ADims: array of QWord): QWord` (raises for
an unknown type or a bad row length). Reuse them; do not change GgufFile.

## Unit tests

UT never need the model. Each case builds its block bytes in the test and
compares the result **bit for bit** (compare the Cardinal bits of Singles,
not with a tolerance). At least:
- `Fp16ToSingle`: +0, -0 ($8000), the smallest subnormal ($0001 = 2^-24),
  the largest subnormal ($03FF), 1.0 ($3C00), -2.0 ($C000), 65504 ($7BFF),
  +inf ($7C00), -inf ($FC00), a quiet NaN ($7E00 gives $7FC00000), a
  signalling one ($7C01 gives $7FC02000), a negative subnormal ($8001).
- `SingleToFp16`: the values above back to their bits; ties to even
  (1 + 2^-11 gives $3C00, 1 + 3·2^-11 gives $3C02); 65519 gives $7BFF and
  65520 gives $7C00; 2^-25 gives $0000 and 2^-25 + 2^-35 gives $0001;
  -0.0 gives $8000; a NaN keeps the sign and gives a NaN.
- `Bf16ToSingle`: 1.0 ($3F80), -2.0 ($C000), $7F80 (+inf).
- `RoundHalfAway`: 0.5 → 1, -0.5 → -1, 1.5 → 2, 2.5 → 3, -2.5 → -3,
  0.49999997 → 0, 126.5 → 127, -0.0 → -0.0.
- each type: one hand-computed block whose expected values are worked out
  in the case table from the layout above (choose d, dmin, scales and
  quants so that the results are exact and easy to check, for example
  d = 1.0, dmin = 0.5), and covering: for q4_K/q5_K the scales of index
  J < 4 and J >= 4 (the high bits of `ScaleMin`); for q5_K the high bit set
  and clear in each of the four groups; for q6_K every one of Q1..Q4 and a
  negative scale; for q8_0 a negative quant.
- `DequantizeRow` with two blocks (the second block starts at the right
  byte); an unsupported type (2); a count that is not a multiple of the
  block size; `DequantizeBytes` with a wrong byte length.
- `QuantizeRowQ8_0`: a block whose Amax is 127 (d = 1.0, q = x); a block of
  zeros (d = 0, q = 0); a block with an exact half after scaling (x = 0.5,
  Amax = 127 gives q = 1; -0.5 gives -1); a block with a tiny Amax
  (1e-40: d = 0, q = 0, no exception); a non-finite input (EGgufError);
  dequantise∘quantise of a block of varied values is within half a step
  (|y - x| <= D / 2 + 1 ulp) of the input; `QuantizeQ8_0` with a length
  that is not a multiple of 32.

## ITa (the oracle, prepared by the harness owner)

`{proj}/work/oracle/T02/` holds the upstream C reference (`quantref.c`)
and its results on the rehearsal model: `dequant-sha256.tsv`
(`<name><TAB><type><TAB><elements><TAB><sha256 of the fp32 bytes>`, 320
tensors) and `q8_0-sha256.tsv` (the same for the dequantised tensor
re-quantised to Q8_0, 284 tensors whose element count is a multiple of 32).
`{proj}/tests/ita/T02-run.sh` (written by the harness owner; do not change
it) builds `{proj}/tests/ita/T02/quant_check.pas` against `src/`, runs it on
`$QLORA_MODEL` from a `mktemp -d` directory outside the project, compares
with the oracle and writes `{proj}/work/ita/T02/summary.tsv`:
`<case ID><TAB><OK|NG><TAB><diff lines><TAB><lines ours><TAB><lines oracle><TAB><note>`.
Cases:

| Case ID | Compares |
|---|---|
| ITa-T02-001 | f32 tensors: sha256 of the fp32 bytes (133 lines) |
| ITa-T02-002 | q8_0 tensors: dequantised (36) |
| ITa-T02-003 | q4_K tensors: dequantised (98) |
| ITa-T02-004 | q5_K tensors: dequantised (36) |
| ITa-T02-005 | q6_K tensors: dequantised (17) |
| ITa-T02-006 | every tensor dequantised, then quantised to Q8_0 (284) |
| ITa-T02-007 | the 36 q8_0 tensors re-quantised equal their own bytes in the file (36) |

A case is OK only when `diff` prints nothing and both sides have the
expected number of lines (not zero).
