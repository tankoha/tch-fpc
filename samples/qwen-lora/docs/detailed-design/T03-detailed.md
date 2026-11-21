# Detailed Design: GgufQuantT Unit

## 1. Module Structure

### Source Files
- `src/GgufQuantT.pas` - Main unit implementing tensor-based dequantization and quantization

### Functions
- `DequantizeT(const ARaw: TTensor; ATType: Cardinal; ACount: Int64): TTensor` - Dequantizes GGUF tensor data using libtorch tensor operations
- `QuantizeQ8_0T(const AX: TTensor): TTensor` - Quantizes fp32 tensor data to Q8_0 format using libtorch tensor operations

## 2. Function Specifications

### DequantizeT
- **Purpose**: Converts quantized GGUF tensor bytes to fp32 tensor using libtorch operations
- **Parameters**: 
  - `ARaw`: kUInt8, 1-D tensor containing the GGUF tensor bytes on any device
  - `ATType`: Cardinal representing the GGUF tensor type
  - `ACount`: Int64 number of elements in the resulting tensor
- **Returns**: fp32 [ACount] tensor on the same device as ARaw
- **Exceptions**: EGgufError raised for:
  - ARaw not kUInt8 or not 1-D
  - Unsupported tensor type ("unsupported tensor type N")
  - ACount not a multiple of block size
  - ARaw's length ≠ GgufTensorBytes(ATType, [ACount])
- **Preconditions**: 
  - ARaw is a valid 1-D kUInt8 tensor
  - ATType is a supported GGUF tensor type
  - ACount is consistent with ARaw's length and tensor type

### QuantizeQ8_0T
- **Purpose**: Converts fp32 tensor to Q8_0 quantized format using libtorch operations
- **Parameters**: 
  - `AX`: kFloat, 1-D tensor with length multiple of 32, all values finite
- **Returns**: kUInt8 tensor [length div 32 * 34] on same device as AX
- **Exceptions**: EGgufError raised for:
  - AX not kFloat or not 1-D
  - AX length not multiple of 32
  - Non-finite values in AX
- **Preconditions**: 
  - AX is a valid 1-D kFloat tensor
  - All values in AX are finite (isfinite.all = True)
  - Length of AX is a multiple of 32

## 3. Data Structures

### Supported GGUF Tensor Types
- f32 (0): 32-bit floating point
- f16 (1): 16-bit floating point
- bf16 (30): Brain floating point
- q8_0 (8): 8-bit quantization with 32-element blocks
- q4_K (12): 4-bit quantization with 256-element blocks
- q5_K (13): 5-bit quantization with 256-element blocks
- q6_K (14): 6-bit quantization with 256-element blocks

### Internal Tensor Representations
All operations use fp32 tensors for arithmetic steps to maintain bit-for-bit compatibility with T02 GgufQuant unit.

## 4. Processing Details

### Dequantization by Type

#### f32 (0)
- Raw bytes interpreted directly as fp32 values
- Operation sequence: `ARaw.contiguous.view_dtype(kFloat)`

#### f16 (1)
- Raw bytes interpreted as f16, converted to fp32
- Operation sequence: `ARaw.contiguous.view_dtype(kHalf).totype(kFloat)`

#### bf16 (30)
- Raw bytes interpreted as bf16, converted to fp32
- Operation sequence: `ARaw.contiguous.view_dtype(kBFloat16).totype(kFloat)`

#### q8_0 (8)
- Block size: 34 bytes (2 scale bytes + 32 quantized values)
- Operation sequence:
  1. Reshape: `R := ARaw.reshape([N/32, 34])`
  2. Extract scales: `D := Cols(R,0,2).contiguous.view_dtype(kHalf).totype(kFloat)` ([nb,1])
  3. Extract quantized values: `Q := Cols(R,2,34).contiguous.view_dtype(kInt8).totype(kFloat)`
  4. Compute result: `(Q * D).reshape([N])`

#### q4_K (12) and q5_K (13)
- Block size: 144 bytes (q4_K) or 176 bytes (q5_K)
- Operation sequence:
  1. Reshape: `R := ARaw.reshape([nb, 144 or 176])`
  2. Extract scales D, DMin from bytes 0..1 and 2..3 as for q8_0, viewed [nb,1,1]
  3. Extract scale components S = Cols(R,4,16) giving Sc, Mn ([nb,8] uint8):
     - `Sc = cat([S[0..3] and 63, (S[8..11] and 15) or ((S[0..3] shr 6) shl 4)], 1)`
     - `Mn = cat([S[4..7] and 63, (S[8..11] shr 4) or ((S[4..7] shr 6) shl 4)], 1)`
  4. Extract quantized values Qs (q4_K: Cols(R,16,144); q5_K: Cols(R,48,176)), reshaped [nb,4,32]
  5. Separate low/high nibbles: `Q = stack([Qs and 15, Qs shr 4], 2)` → [nb,4,2,32]
  6. For q5_K only: Apply high bits from Qh = Cols(R,16,48) reshaped [nb,1,32]:
     - High bit of group g (g = 0..3) is bit 2g of Qh for low nibbles and bit 2g+1 for high nibbles:
     - `H = stack over g of stack([(Qh shr 2g) and 1, (Qh shr (2g+1)) and 1], 2)`, shaped [nb,4,2,32]
     - `Q := Q or (H shl 4)`
  7. Reshape Q to [nb,8,32]
  8. Compute result: `((D * Sc.totype(kFloat).unsqueeze(2)) * Q.totype(kFloat) - DMin * Mn.totype(kFloat).unsqueeze(2)).reshape([N])`

#### q6_K (14)
- Block size: 210 bytes
- Operation sequence:
  1. Reshape: `R := ARaw.reshape([nb,210])`
  2. Extract components:
     - `Ql = Cols(R,0,128)` reshaped [nb,2,2,32]
     - `Qh = Cols(R,128,192)` reshaped [nb,2,32]
     - `Sc = Cols(R,192,208).contiguous.view_dtype(kInt8)` ([nb,16])
     - `D` from bytes 208..209 viewed [nb,1,1]
  3. Reconstruct quantized values:
     - `A = Ql.select(2,0)`, `B = Ql.select(2,1)`
     - `Q = stack([(A and 15) or ((Qh and 3) shl 4), (B and 15) or (((Qh shr 2) and 3) shl 4), (A shr 4) or (((Qh shr 4) and 3) shl 4), (B shr 4) or (((Qh shr 6) and 3) shl 4)], 2)` reshaped [nb,16,16]
  4. Compute result: `((D * Sc.totype(kFloat).unsqueeze(2)) * (Q.totype(kFloat) - 32)).reshape([N])`

### Q8_0 Quantization

#### Input Requirements
- AX: kFloat, 1-D tensor with length multiple of 32
- All values must be finite (isfinite.all = True)

#### Operation Sequence
1. Reshape: `X := AX.reshape([nb,32])`
2. Calculate maximum absolute values: `Amax := X.abs.amax([1], True)`
3. Calculate scales: `D := Amax / 127.0`
4. Identify tiny values: `Tiny := D.le(2.9387358770557188e-39)` (D <= 2^-128)
5. Calculate inverse scales: `Id := D.reciprocal.where_scalarother(D.gt(0.0).logical_and(Tiny.logical_not), 0.0)`
6. Scale input: `X0 := X * Id`
7. Truncate: `T := X0.trunc`
8. Round half away from zero:
   - `Q := T + X0.sign * (X0 - T).abs.ge(0.5).totype(kFloat)`
   - `Q := Q.where_scalarother(Tiny.logical_not, 0.0)`
9. Prepare scale results: `DH := D.where_scalarother(Tiny.logical_not, 0.0).totype(kHalf).view_dtype(kUInt8)` ([nb,2])
10. Combine scales and quantized values: `cat([DH, Q.totype(kInt8).view_dtype(kUInt8)], 1).reshape([nb * 34])`

### Error Conditions

#### DequantizeT Errors
1. ARaw not kUInt8 or not 1-D: Check tensor properties before processing
2. Unsupported tensor type: Validate ATType against known types (0, 1, 8, 12, 13, 14, 30)
3. ACount not a multiple of block size: Validate based on tensor type block sizes
4. ARaw length mismatch: Verify ARaw's length equals GgufTensorBytes(AType, [ACount])

#### QuantizeQ8_0T Errors
1. AX not kFloat or not 1-D: Check tensor properties before processing
2. AX length not multiple of 32: Validate input tensor dimensions
3. Non-finite values in AX: Check with isfinite.all before processing

## 5. Change History
- 2026-11-21: Initial version based on T03 facts