## Purpose
The unit `GgufQuantT` provides de‑quantisation of GGUF tensor bytes and Q8_0 quantisation of fp32 tensors using libtorch tensor operations. It must reproduce the behaviour of the existing `GgufQuant` unit (T02) **bit‑for‑bit** for all supported types.

## Interface
```pascal
unit GgufQuantT;
{$mode objfpc}{$H+}
interface
uses SysUtils, Tch, GgufFile;   { EGgufError, GgufTensorBytes, GgufTypeName }

{ Dequantise raw GGUF bytes to fp32 tensors }
function DequantizeT(const ARaw: TTensor; ATType: Cardinal; ACount: Int64): TTensor;

{ Quantise fp32 tensors to Q8_0 GGUF bytes }
function QuantizeQ8_0T(const AX: TTensor): TTensor;
```
### Errors (EGgufError)
- `ARaw` not `kUInt8` or not 1‑D.
- Unsupported tensor type (message "unsupported tensor type N").
- `ACount` not a multiple of the block size for the type.
- Length of `ARaw` does not match `GgufTensorBytes(ATType, [ACount])`.
- For `QuantizeQ8_0T`: `AX` not `kFloat`, not 1‑D, length not a multiple of 32, or contains non‑finite values.
