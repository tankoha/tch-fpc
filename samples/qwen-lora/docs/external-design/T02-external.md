# T02: GgufQuant (external design)

## Purpose
Unit `GgufQuant` provides scalar dequantisers for the GGUF tensor types and the Q8_0 quantiser. It is the reference for:
- T03 (`GgufQuantT`: the same dequantisers in tensor operations),
- T10 (dequantised weights for training),
- T18 (merged tensors quantised to Q8_0).

Every routine computes in `Single` (fp32) exactly as documented, matching upstream `ggml/src/ggml-quants.c` bit-for-bit.

## Interface
```pascal
unit GgufQuant;
{$mode objfpc}{$H+}
interface
uses SysUtils, GgufFile;   // EGgufError, GgufTensorBytes, GgufTypeName

type
  TSingleArray = array of Single;

// Conversions
function Fp16ToSingle(AH: Word): Single;
function SingleToFp16(AX: Single): Word;
function Bf16ToSingle(AH: Word): Single;
function RoundHalfAway(AX: Single): Single;

// Dequantisers
procedure DequantizeRow(ATType: Cardinal; ASrc: PByte; ACount: Int64; ADst: PSingle);
function DequantizeBytes(ATType: Cardinal; const ABytes: TBytes; ACount: Int64): TSingleArray;

// Q8_0 quantiser
procedure QuantizeRowQ8_0(ASrc: PSingle; ACount: Int64; ADst: PByte);
function QuantizeQ8_0(const AValues: TSingleArray): TBytes;
```

## Errors
All routines raise `EGgufError` with descriptive messages:
- Unsupported tensor type (`GgufTypeName(ATType)` is used),
- `ACount` not a multiple of the block size (message names the type and `ACount`),
- For `DequantizeBytes`: byte length ≠ `GgufTensorBytes(ATType, [ACount])`,
- For `QuantizeQ8_0`: `Length(AValues)` not a multiple of 32,
- For `QuantizeRowQ8_0`/`QuantizeQ8_0`: non-finite input in `ASrc`.

`GgufFile`'s `EGgufError`, `GgufTypeName`, and `GgufTensorBytes` are reused.