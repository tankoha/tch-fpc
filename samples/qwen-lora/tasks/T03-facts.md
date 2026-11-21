# T03 facts (checked by the harness owner)

T03 is the unit `GgufQuantT`: the dequantisers and the Q8_0 quantiser of T02
written in libtorch tensor ops (unit `Tch` of tch-fpc), so that whole
weight matrices are dequantised on a device (T10 `PackedLinear`) and merged
tensors are quantised in bulk (T18). It must agree with T02 `GgufQuant`
**bit for bit**.

Measured by the harness owner with a prototype written exactly as below, on
the CPU (GPUs hidden): every tensor of the rehearsal model dequantises (320)
and re-quantises (284) to the same bytes as the upstream C reference, in
35 s; 512 fully random blocks per type (including NaN and infinite scales)
and 544 random Q8_0 blocks (magnitudes 1e-45 to 3e38, zeros, exact halves,
tiny blocks) give the same bits as T02. These facts are final; do not
investigate further or write probes.

## Arithmetic

- Tensors are fp32 (`kFloat`) for every arithmetic step; the result of a
  dequantisation is fp32. Converting to bf16 for compute is the caller's
  business (T10), not this unit's.
- Write each formula as the products and differences below, in that order
  and grouping: `(D * Sc) * Q - (DMin * Mn)`, not `D * (Sc * Q)`. libtorch's
  elementwise `*`, `-` are single IEEE operations on the CPU, so the bits
  equal T02's.
- Raw bytes are a 1-D `kUInt8` tensor; reinterpret with `view_dtype` on a
  `contiguous` tensor (`kHalf`, `kBFloat16`, `kInt8`, `kFloat`), never by
  arithmetic. `totype(kFloat)` of a half/bfloat16 tensor equals T02's
  `Fp16ToSingle`/`Bf16ToSingle` (measured).
- NaN results: when a block's scales are NaN, both units give NaN, but
  the sign and payload may differ (libtorch picks a different operand's NaN
  in `a * b - c * d`; seen in T03b1b, seed 1005, q4_K). Tests therefore
  count two NaNs as equal and compare every other value by its bits. Real
  model files have no NaN scales.

## Operations used (all exist in `Tch`)

`narrow(dim, start, length)`, `reshape`, `view`, `contiguous`, `select`,
`unsqueeze`, `TTensor.cat([...], dim)`, `TTensor.stack([...], dim)`,
`view_dtype(kind)`, `totype(kind)`, `bitwise_and(int)`,
`bitwise_or_tensor(t)`, `bitwise_right_shift_tensor_scalar(n)`,
`bitwise_left_shift_tensor_scalar(n)`, operators `*`, `-`, `+`, `/` (also
with a Double), `abs`, `amax([dim], keepdim)`, `reciprocal`, `trunc`,
`sign`, `ge`, `le`, `gt` (with a scalar), `logical_and`, `logical_not`,
`where_scalarother(condition, value)` (keeps self where condition is true,
else value), `isfinite`, `all`, `Shape`, `Kind`,
`TTensor.FromBytes(Data, [n], kUInt8)`, `CopyData`. `Cols(T, A, B)` below
means `T.narrow(1, A, B - A)`.

## Dequantisation, per type (ARaw: uint8 [bytes]; N = element count)

- **f32 (0):** `ARaw.contiguous.view_dtype(kFloat)`.
- **f16 (1):** `ARaw.contiguous.view_dtype(kHalf).totype(kFloat)`.
- **bf16 (30):** `ARaw.contiguous.view_dtype(kBFloat16).totype(kFloat)`.
- **q8_0 (8):** R := ARaw.reshape([N/32, 34]);
  D := Cols(R,0,2).contiguous.view_dtype(kHalf).totype(kFloat) ([nb,1]);
  Q := Cols(R,2,34).contiguous.view_dtype(kInt8).totype(kFloat);
  result (Q * D).reshape([N]).
- **q4_K (12) and q5_K (13):** R := ARaw.reshape([nb, 144 or 176]);
  D, DMin from bytes 0..1 and 2..3 as for q8_0, viewed [nb,1,1];
  scales S = Cols(R,4,16) give Sc, Mn ([nb,8] uint8):
  `Sc = cat([S[0..3] and 63, (S[8..11] and 15) or ((S[0..3] shr 6) shl 4)], 1)`,
  `Mn = cat([S[4..7] and 63, (S[8..11] shr 4) or ((S[4..7] shr 6) shl 4)], 1)`;
  Qs = the 128 quant bytes (q4_K: Cols(R,16,144); q5_K: Cols(R,48,176))
  reshaped [nb,4,32]; Q = stack([Qs and 15, Qs shr 4], 2) → [nb,4,2,32].
  q5_K only: Qh = Cols(R,16,48) reshaped [nb,1,32]; the high bit of group g
  (g = 0..3) is bit 2g of Qh for the low nibbles and bit 2g+1 for the high
  nibbles: H = stack over g of stack([(Qh shr 2g) and 1, (Qh shr (2g+1)) and 1], 2),
  shaped [nb,4,2,32]; Q := Q or (H shl 4).
  Then Q reshaped [nb,8,32] and
  result `((D * Sc.totype(kFloat).unsqueeze(2)) * Q.totype(kFloat) - DMin * Mn.totype(kFloat).unsqueeze(2)).reshape([N])`.
- **q6_K (14):** R := ARaw.reshape([nb,210]); Ql = Cols(R,0,128) reshaped
  [nb,2,2,32]; Qh = Cols(R,128,192) reshaped [nb,2,32]; Sc =
  Cols(R,192,208).contiguous.view_dtype(kInt8) ([nb,16]); D from bytes
  208..209 viewed [nb,1,1]; A = Ql.select(2,0), B = Ql.select(2,1);
  Q = stack([(A and 15) or ((Qh and 3) shl 4),
             (B and 15) or (((Qh shr 2) and 3) shl 4),
             (A shr 4) or (((Qh shr 4) and 3) shl 4),
             (B shr 4) or (((Qh shr 6) and 3) shl 4)], 2) reshaped [nb,16,16];
  result `((D * Sc.totype(kFloat).unsqueeze(2)) * (Q.totype(kFloat) - 32)).reshape([N])`.

## Q8_0 quantisation (AX: fp32 [n], n mod 32 = 0)

X := AX.reshape([nb,32]); Amax := X.abs.amax([1], True); D := Amax / 127.0;
Tiny := D.le(2.9387358770557188e-39) (that is D <= 2^-128; zero blocks are
also "tiny" and give the same bytes); Id := D.reciprocal.where_scalarother(
D.gt(0.0).logical_and(Tiny.logical_not), 0.0); X0 := X * Id; T := X0.trunc;
Q := T + X0.sign * (X0 - T).abs.ge(0.5).totype(kFloat) (round half away
from zero; `round` rounds half to even); Q := Q.where_scalarother(
Tiny.logical_not, 0.0); DH := D.where_scalarother(Tiny.logical_not, 0.0)
.totype(kHalf).view_dtype(kUInt8) ([nb,2]); result
cat([DH, Q.totype(kInt8).view_dtype(kUInt8)], 1).reshape([nb * 34]).

## Unit `GgufQuantT` (interface, final)

```pascal
unit GgufQuantT;
{$mode objfpc}{$H+}
interface
uses SysUtils, Tch, GgufFile;   { EGgufError, GgufTensorBytes, GgufTypeName }

{ ARaw: kUInt8, 1-D, the tensor's GGUF bytes, on any device. Result: fp32
  [ACount] on the same device. EGgufError for: ARaw not kUInt8 or not 1-D;
  an unsupported type ("unsupported tensor type N"); ACount not a multiple
  of the block size; ARaw's length <> GgufTensorBytes(ATType, [ACount])
  (message names both lengths). }
function DequantizeT(const ARaw: TTensor; ATType: Cardinal; ACount: Int64): TTensor;

{ AX: kFloat, 1-D, length a multiple of 32, every value finite (checked with
  isfinite.all). Result: kUInt8 [length div 32 * 34] on AX's device.
  EGgufError otherwise. }
function QuantizeQ8_0T(const AX: TTensor): TTensor;
```

## Unit tests

UT never need the model and run on the CPU (`.sdlc/ut-coverage.sh` hides
the GPUs). The oracle is T02's unit `GgufQuant` (final): the test program
uses both units and compares bits.
- For each type (0, 1, 30, 8, 12, 13, 14): 512 blocks of random bytes
  (`RandSeed` fixed per case so a failure reproduces; f32 uses 512 × 64
  values), `DequantizeT` versus `GgufQuant.DequantizeBytes`, every element
  equal as `Cardinal` bits.
- Q8_0: 544 blocks of random finite values over many magnitudes (zeros,
  halves such as `(Random(255) - 127) / 2`, 1e-41, 3e38, 1e-30..1e30),
  one block of tiny values (1e-44) and one of zeros; `QuantizeQ8_0T`
  versus `GgufQuant.QuantizeQ8_0`, byte for byte.
- One hand case per type copied from T02's case table (its bytes and hex
  expected values), so a failure points at the formula.
- Every error of the interface, one case each (message checked to name
  what the interface says).
- `DequantizeT` result shape, kind (`kFloat`) and device; `QuantizeQ8_0T`
  result length and kind.

## ITa (prepared by the harness owner)

`{proj}/tests/ita/T03-run.sh` builds `{proj}/tests/ita/T03/quantt_check.pas`
and compares every tensor of `$QLORA_MODEL`, dequantised and re-quantised
with `GgufQuantT` on the CPU, with the same oracle tables as T02
(`{proj}/work/oracle/T02/`), writing `{proj}/work/ita/T03/summary.tsv` in
T02's format. Cases ITa-T03-001 to 007 correspond to ITa-T02-001 to 007.

The script's stderr (`errors.txt`) holds ROCm warnings such as
"Sequence '-1' in HIP_VISIBLE_DEVICES=-1 not recognized": they come from
hiding the GPUs and are expected; the driver's exit code is what counts.

## Stages and case IDs

- Stage 1 (T03b, delegation 1): `DequantizeT` with its errors; UT rows
  `UT-GgufQuantT-001` to `099`, test program `tests/ut/GgufQuantT_test.pas`.
- Stage 2 (T03b, delegation 2): `QuantizeQ8_0T` with its errors; UT rows
  from `UT-GgufQuantT-101`, in the same test program.
