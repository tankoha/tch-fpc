{$mode objfpc}{$H+}
unit GgufQuantT;
{ T02's dequantisers and Q8_0 quantiser in libtorch tensor ops (T03), bit
  for bit equal to GgufQuant: see tasks/T03-facts.md and
  docs/detailed-design/T03-detailed.md.

  Skeleton by the harness owner (the interface of the facts); the routines
  are written in T03b. }

interface

uses
  SysUtils, Tch, GgufFile;   { EGgufError, GgufTensorBytes, GgufTypeName }

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

implementation

function DequantizeT(const ARaw: TTensor; ATType: Cardinal; ACount: Int64): TTensor;
var
 ExpectedBytes: Int64;
 nb: Int64;
 R, D, DMin, Q, Sc, Mn, S, Qs, Qh, H, Ql, A, B, Tmp1, Tmp2, X, Amax, Tiny, Id, X0, Ttemp, DH, Qbytes: TTensor;
 begin
  // Check that ARaw is kUInt8 and 1-D
  if ARaw.Kind <> kUInt8 then
    raise EGgufError.Create('DequantizeT: ARaw not kUInt8');
  if Length(ARaw.Shape) <> 1 then
    raise EGgufError.Create('DequantizeT: ARaw not 1-D');
    
  // Check supported types
  case ATType of
    0, 1, 8, 12, 13, 14, 30: ; // Supported types
  end;
  
  // Check block size requirements
  case ATType of
    0: ; // f32 block size check removed
    1: ; // f16 block size check removed
    8: if ACount mod 32 <> 0 then  // q8_0: 32 values per block
         raise EGgufError.CreateFmt('DequantizeT: count %d not a multiple of block size for type %d', [ACount, ATType]);
    12, 13: if ACount mod 256 <> 0 then  // q4_K, q5_K: 256 values per block
              raise EGgufError.CreateFmt('DequantizeT: count %d not a multiple of block size for type %d', [ACount, ATType]);
    14: if ACount mod 256 <> 0 then  // q6_K: 256 values per block
          raise EGgufError.CreateFmt('DequantizeT: count %d not a multiple of block size for type %d', [ACount, ATType]);
    30: if ACount mod 1 <> 0 then  // bf16: 2 bytes per value
          raise EGgufError.CreateFmt('DequantizeT: count %d not a multiple of block size for type %d', [ACount, ATType]);
  end;
  
  // Check byte length
  ExpectedBytes := GgufTensorBytes(ATType, [ACount]);
  if ARaw.Shape[0] <> ExpectedBytes then
    raise EGgufError.CreateFmt('DequantizeT: ARaw length %d <> expected %d', [ARaw.Shape[0], ExpectedBytes]);
  
  // Implementation by type
  case ATType of
    0: // f32
      begin
        Result := ARaw.contiguous.view_dtype(kFloat);
        Exit;
      end;
    1: // f16
      begin
        Result := ARaw.contiguous.view_dtype(kHalf).totype(kFloat);
        Exit;
      end;
    30: // bf16
      begin
        Result := ARaw.contiguous.view_dtype(kBFloat16).totype(kFloat);
        Exit;
      end;
    8: // q8_0
      begin
        nb := ACount div 32;
        R := ARaw.reshape([nb, 34]);
        D := R.narrow(1, 0, 2).contiguous.view_dtype(kHalf).totype(kFloat);
        Q := R.narrow(1, 2, 32).contiguous.view_dtype(kInt8).totype(kFloat);
        Result := (Q * D).reshape([ACount]);
        Exit;
      end;
    12: // q4_K
      begin
        nb := ACount div 256;
        R := ARaw.reshape([nb, 144]);
        D := R.narrow(1,0,2).contiguous.view_dtype(kHalf).totype(kFloat).reshape([nb,1,1]);
        DMin := R.narrow(1,2,2).contiguous.view_dtype(kHalf).totype(kFloat).reshape([nb,1,1]);
        S := R.narrow(1,4,12);
        // Sc
        Sc := TTensor.cat([
          S.narrow(1,0,4).bitwise_and(63),
          (S.narrow(1,8,4).bitwise_and(15)).bitwise_or_tensor((S.narrow(1,0,4).bitwise_right_shift_tensor_scalar(6)).bitwise_left_shift_tensor_scalar(4))
        ], 1);
        // Mn
        Mn := TTensor.cat([
          S.narrow(1,4,4).bitwise_and(63),
          (S.narrow(1,8,4).bitwise_right_shift_tensor_scalar(4)).bitwise_or_tensor((S.narrow(1,4,4).bitwise_right_shift_tensor_scalar(6)).bitwise_left_shift_tensor_scalar(4))
        ], 1);
        // Qs and Q
        Qs := R.narrow(1,16,128).reshape([nb,4,32]);
        Q := TTensor.stack([Qs.bitwise_and(15), Qs.bitwise_right_shift_tensor_scalar(4)], 2).reshape([nb,8,32]);
        Result := ((D * Sc.totype(kFloat).unsqueeze(2)) * Q.totype(kFloat) - DMin * Mn.totype(kFloat).unsqueeze(2)).reshape([ACount]);
        Exit;
      end;
    13: // q5_K
      begin
        nb := ACount div 256;
        R := ARaw.reshape([nb, 176]);
        D := R.narrow(1,0,2).contiguous.view_dtype(kHalf).totype(kFloat).reshape([nb,1,1]);
        DMin := R.narrow(1,2,2).contiguous.view_dtype(kHalf).totype(kFloat).reshape([nb,1,1]);
        S := R.narrow(1,4,12);
        // Sc
        Sc := TTensor.cat([
          S.narrow(1,0,4).bitwise_and(63),
          (S.narrow(1,8,4).bitwise_and(15)).bitwise_or_tensor((S.narrow(1,0,4).bitwise_right_shift_tensor_scalar(6)).bitwise_left_shift_tensor_scalar(4))
        ], 1);
        // Mn
        Mn := TTensor.cat([
          S.narrow(1,4,4).bitwise_and(63),
          (S.narrow(1,8,4).bitwise_right_shift_tensor_scalar(4)).bitwise_or_tensor((S.narrow(1,4,4).bitwise_right_shift_tensor_scalar(6)).bitwise_left_shift_tensor_scalar(4))
        ], 1);
        // Qs and Q
        Qs := R.narrow(1,48,128).reshape([nb,4,32]);
        Q := TTensor.stack([Qs.bitwise_and(15), Qs.bitwise_right_shift_tensor_scalar(4)], 2);
        // Qh high bits
        Qh := R.narrow(1,16,32).reshape([nb,1,32]);
        // Build H from Qh bits
        H := TTensor.cat([
          TTensor.stack([Qh.bitwise_right_shift_tensor_scalar(0).bitwise_and(1), Qh.bitwise_right_shift_tensor_scalar(1).bitwise_and(1)], 2),
          TTensor.stack([Qh.bitwise_right_shift_tensor_scalar(2).bitwise_and(1), Qh.bitwise_right_shift_tensor_scalar(3).bitwise_and(1)], 2),
          TTensor.stack([Qh.bitwise_right_shift_tensor_scalar(4).bitwise_and(1), Qh.bitwise_right_shift_tensor_scalar(5).bitwise_and(1)], 2),
          TTensor.stack([Qh.bitwise_right_shift_tensor_scalar(6).bitwise_and(1), Qh.bitwise_right_shift_tensor_scalar(7).bitwise_and(1)], 2)
        ], 1);
        Q := Q.bitwise_or_tensor(H.bitwise_left_shift_tensor_scalar(4));
        Q := Q.reshape([nb,8,32]);
        Result := ((D * Sc.totype(kFloat).unsqueeze(2)) * Q.totype(kFloat) - DMin * Mn.totype(kFloat).unsqueeze(2)).reshape([ACount]);
        Exit;
      end;
    14: // q6_K
      begin
        nb := ACount div 256;
        R := ARaw.reshape([nb, 210]);
        // Ql and Qh extraction
        Ql := R.narrow(1,0,128).reshape([nb,2,2,32]);
        Qh := R.narrow(1,128,64).reshape([nb,2,32]);
        Sc := R.narrow(1,192,16).contiguous.view_dtype(kInt8);
        D := R.narrow(1,208,2).contiguous.view_dtype(kHalf).totype(kFloat).reshape([nb,1,1]);
        // A and B selects
        A := Ql.select(2,0);
        B := Ql.select(2,1);
        // Build Q parts
        Q := TTensor.stack([
          (A.bitwise_and(15)).bitwise_or_tensor( (Qh.bitwise_and(3)).bitwise_left_shift_tensor_scalar(4) ),
          (B.bitwise_and(15)).bitwise_or_tensor( (Qh.bitwise_right_shift_tensor_scalar(2).bitwise_and(3)).bitwise_left_shift_tensor_scalar(4) ),
          (A.bitwise_right_shift_tensor_scalar(4)).bitwise_or_tensor( (Qh.bitwise_right_shift_tensor_scalar(4).bitwise_and(3)).bitwise_left_shift_tensor_scalar(4) ),
          (B.bitwise_right_shift_tensor_scalar(4)).bitwise_or_tensor( (Qh.bitwise_right_shift_tensor_scalar(6).bitwise_and(3)).bitwise_left_shift_tensor_scalar(4) )
        ], 2).reshape([nb,16,16]);
        Result := ((D * Sc.totype(kFloat).unsqueeze(2)) * (Q.totype(kFloat) - 32)).reshape([ACount]);
        Exit;
      end;

  end;
end;

function QuantizeQ8_0T(const AX: TTensor): TTensor;
var
  nb: Int64;
  D, Q, X, Amax, Tiny, Id, X0, Ttemp, DH, Qbytes: TTensor;
begin
  // AX must be kFloat, 1-D, length a multiple of 32, all finite
  if AX.Kind <> kFloat then
    raise EGgufError.Create('QuantizeQ8_0T: AX not kFloat');
  if Length(AX.Shape) <> 1 then
    raise EGgufError.Create('QuantizeQ8_0T: AX not 1-D');
  if (AX.Numel mod 32) <> 0 then
    raise EGgufError.CreateFmt('QuantizeQ8_0T: length %d not a multiple of 32', [AX.Numel]);
  if AX.isfinite.all.DoubleValue([]) = 0.0 then
    raise EGgufError.Create('QuantizeQ8_0T: non-finite values');
  // Reshape to [nb,32]
  nb := AX.Numel div 32;
  X := AX.reshape([nb, 32]);
  // Compute scale D per block
  Amax := X.abs.amax([1], True); // shape [nb,1]
  D := Amax / 127.0;
  // Tiny detection
  Tiny := D.le(2.9387358770557188e-39);
  // Id where D > 0 and not tiny, else 0
  Id := D.reciprocal.where_scalarother(D.gt(0.0).logical_and(Tiny.logical_not), 0.0);
  // Scale values
  X0 := X * Id;
  // Rounding half away from zero
  Ttemp := X0.trunc;
  Q := Ttemp + X0.sign * (X0 - Ttemp).abs.ge(0.5).totype(kFloat);
  // Zero out tiny blocks
  Q := Q.where_scalarother(Tiny.logical_not, 0.0);
  // Encode D as half and view as uint8 bytes (2 bytes per block)
  DH := D.where_scalarother(Tiny.logical_not, 0.0).totype(kHalf).view_dtype(kUInt8);
  // Q to int8 then view as uint8 bytes (32 bytes per block)
  Qbytes := Q.totype(kInt8).view_dtype(kUInt8);
  // Concatenate D bytes and Q bytes along dimension 1
  Result := TTensor.cat([DH, Qbytes], 1).reshape([nb * 34]);
end;

end.
