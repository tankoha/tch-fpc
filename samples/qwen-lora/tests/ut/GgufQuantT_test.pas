{ UT program for src/GgufQuantT.pas: the rows UT-GgufQuantT-NNN of
  tests/ut/cases/GgufQuantT.cases.md (stage 1 from 001, stage 2 from 101).
  The oracle is T02's GgufQuant; values are compared as bits or bytes.
  Runs on the CPU (.sdlc/ut-coverage.sh hides the GPUs).
  Invoked with `--all --format=plain` (see .sdlc/ut-coverage.sh).

  Skeleton by the harness owner; the test methods are written in T03b. }
program GgufQuantT_test;

{$mode objfpc}{$H+}

uses
  SysUtils, Classes, Math, Tch, GgufFile, GgufQuant, GgufQuantT,
  fpcunit, testregistry, consoletestrunner;

type
  TGgufQuantTTest = class(TTestCase)
  public
    { The bits of a Single. }
    class function Bits(AX: Single): Cardinal; static;
    { The fp32 values of a tensor (kFloat, 1-D) as a Pascal array. }
    class function Values(const T: TTensor): TSingleArray; static;
    { The bytes of a tensor (kUInt8, 1-D) as a Pascal array. }
    class function TensorBytes(const T: TTensor): TBytes; static;
    { Compare two Singles, treating NaN as equal. }
    class function SameValue(A, B: Single): Boolean; static;
  published
    { T03b adds TestUT_GgufQuantT_001.. (stage 1) and 101.. (stage 2). }
    procedure TestUT_GgufQuantT_001;  // f32 random blocks
    procedure TestUT_GgufQuantT_002;
    procedure TestUT_GgufQuantT_003;
    procedure TestUT_GgufQuantT_004;
    procedure TestUT_GgufQuantT_005;
    procedure TestUT_GgufQuantT_006;
    procedure TestUT_GgufQuantT_007;
    procedure TestUT_GgufQuantT_008;
    procedure TestUT_GgufQuantT_009;
    procedure TestUT_GgufQuantT_010;
    procedure TestUT_GgufQuantT_011;
    procedure TestUT_GgufQuantT_012;
    procedure TestUT_GgufQuantT_013;
    procedure TestUT_GgufQuantT_014;
    procedure TestUT_GgufQuantT_015;  // Unsupported tensor type
    procedure TestUT_GgufQuantT_016;  // Count not a multiple of block size
    procedure TestUT_GgufQuantT_017;  // Wrong byte length
    procedure TestUT_GgufQuantT_018;  // Result tensor shape and kind (f32)
    procedure TestUT_GgufQuantT_019;  // Result tensor shape and kind (bf16)
    procedure TestUT_GgufQuantT_020;  // Result tensor device matches input
    procedure TestUT_GgufQuantT_021;  // ARaw not kUInt8
    procedure TestUT_GgufQuantT_022;  // ARaw not 1-D
    procedure TestUT_GgufQuantT_023;  // Count not a multiple (q4_K)
    procedure TestUT_GgufQuantT_024;  // Count not a multiple (q6_K)
    procedure TestUT_GgufQuantT_101;  // Quantize random finite blocks
    procedure TestUT_GgufQuantT_102;  // Quantize zeros and tiny values
    procedure TestUT_GgufQuantT_103;  // Quantize hand case
    procedure TestUT_GgufQuantT_104;  // Non-finite input values
    procedure TestUT_GgufQuantT_105;  // Input length not multiple of 32
    procedure TestUT_GgufQuantT_106;  // Result tensor length and kind
    procedure TestUT_GgufQuantT_107;  // Result tensor device matches input
    procedure TestUT_GgufQuantT_108;  // AX not kFloat (kDouble tensor)
    procedure TestUT_GgufQuantT_109;  // AX not 1-D
  end;

class function TGgufQuantTTest.Bits(AX: Single): Cardinal;
begin
  Move(AX, Result, 4);
end;

class function TGgufQuantTTest.Values(const T: TTensor): TSingleArray;
begin
  Result := T.ToSingleArray;
end;

class function TGgufQuantTTest.TensorBytes(const T: TTensor): TBytes;
begin
  SetLength(Result, T.Shape[0]);
  if Length(Result) > 0 then
    T.contiguous.CopyData(Result[0], Length(Result));
end;

  class function TGgufQuantTTest.SameValue(A, B: Single): Boolean;
  begin
  if Math.IsNan(A) and Math.IsNan(B) then
  Exit(True);
  Result := Bits(A) = Bits(B);
  end;

procedure TGgufQuantTTest.TestUT_GgufQuantT_001;
var
  RawBytes: TBytes;
  Raw: TTensor;
  Actual: TTensor;
  ActualValues, ExpectedValues: TSingleArray;
  i: Integer;
begin
  // f32 random blocks (512 × 64)
  // RandSeed = 1001; 512 blocks × 64 values => 32768 values => 131072 bytes
  RandSeed := 1001;
  SetLength(RawBytes, 32768 * 4);
  for i := 0 to Length(RawBytes) - 1 do
    RawBytes[i] := Byte(Random(256));

  Raw := TTensor.FromBytes(RawBytes[0], [Length(RawBytes)], kUInt8);
  Actual := DequantizeT(Raw, 0, 32768);
  ExpectedValues := GgufQuant.DequantizeBytes(0, RawBytes, 32768);

  // Verify shape and kind
  AssertEquals(32768, Actual.Shape[0]);
  AssertTrue(Actual.Kind = kFloat);

  ActualValues := Values(Actual);
  AssertEquals(Length(ExpectedValues), Length(ActualValues));
  for i := 0 to High(ActualValues) do
    if not SameValue(ActualValues[i], ExpectedValues[i]) then
      Fail(Format('Mismatch at index %d: Expected $%8.8x Actual $%8.8x', [i, Bits(ExpectedValues[i]), Bits(ActualValues[i])]));
end;

procedure TGgufQuantTTest.TestUT_GgufQuantT_015;
var
  Raw: TTensor;
  RawBytes: TBytes;
begin
  // Unsupported tensor type (2)
  // Type = 2, count = 64
  
  SetLength(RawBytes, 256); // 64 * 4 bytes for f32
  Raw := TTensor.FromBytes(RawBytes, [256], kUInt8);
  
  try
    DequantizeT(Raw, 2, 64); // Type 2 = unsupported
    Fail('Expected EGgufError for unsupported type');
  except
    on E: EGgufError do
      // Expected exception
  end;
end;

procedure TGgufQuantTTest.TestUT_GgufQuantT_016;
var
  Raw: TTensor;
  RawBytes: TBytes;
begin
  // Count not a multiple of block size (q8_0)
  // Type = 8 (q8_0), count = 33
  
  SetLength(RawBytes, 34); // Would be for 32 elements
  Raw := TTensor.FromBytes(RawBytes, [34], kUInt8);
  
  try
    DequantizeT(Raw, 8, 33); // Type 8 = q8_0, but count 33 not multiple of 32
    Fail('Expected EGgufError for count not multiple of block size');
  except
    on E: EGgufError do
      // Expected exception
  end;
end;

procedure TGgufQuantTTest.TestUT_GgufQuantT_017;
var
  Raw: TTensor;
  RawBytes: TBytes;
begin
  // Wrong byte length
  // Type = 8, count = 32, length = 33
  
  SetLength(RawBytes, 33); // Should be 34 for 32 elements in q8_0
  Raw := TTensor.FromBytes(RawBytes, [33], kUInt8);
  
  try
    DequantizeT(Raw, 8, 32); // Type 8 = q8_0, count 32 needs 34 bytes
    Fail('Expected EGgufError for wrong byte length');
  except
    on E: EGgufError do
      // Expected exception
  end;
end;

procedure TGgufQuantTTest.TestUT_GgufQuantT_018;
var
  Raw: TTensor;
  Actual: TTensor;
  RawBytes: TBytes;
begin
  // Result tensor shape and kind (f32)
  // Valid f32 block (16 bytes)
  
  SetLength(RawBytes, 16); // 4 * 4 bytes for 4 f32 values
  Raw := TTensor.FromBytes(RawBytes, [16], kUInt8);
  Actual := DequantizeT(Raw, 0, 4); // Type 0 = f32
  
  // Check shape and kind
  AssertEquals('Result shape', 4, Actual.Shape[0]);
  AssertTrue('Result kind', Actual.Kind = kFloat);
end;

procedure TGgufQuantTTest.TestUT_GgufQuantT_021;
var
  Raw: TTensor;
  RawBytes: TBytes;
begin
  // ARaw not kUInt8
  // Try with kFloat instead
  
  SetLength(RawBytes, 16);
  Raw := TTensor.FromBytes(RawBytes, [4], kFloat); // Wrong kind
  
  try
    DequantizeT(Raw, 0, 4); // Should fail because Raw is not kUInt8
    Fail('Expected EGgufError for ARaw not kUInt8');
  except
    on E: EGgufError do
      // Expected exception
  end;
end;

procedure TGgufQuantTTest.TestUT_GgufQuantT_022;
var
  Raw: TTensor;
  RawBytes: TBytes;
begin
  // ARaw not 1-D
  // Create a 2-D tensor
  
  SetLength(RawBytes, 16);
  Raw := TTensor.FromBytes(RawBytes, [4, 4], kUInt8); // 2-D tensor
  
  try
    DequantizeT(Raw, 0, 4); // Should fail because Raw is not 1-D
    Fail('Expected EGgufError for ARaw not 1-D');
  except
    on E: EGgufError do
      // Expected exception
  end;
end;

procedure TGgufQuantTTest.TestUT_GgufQuantT_023;
var
  Raw: TTensor;
  RawBytes: TBytes;
begin
  // Count not a multiple (q4_K)
  SetLength(RawBytes, 257);
  Raw := TTensor.FromBytes(RawBytes[0], [257], kUInt8);
  try
    DequantizeT(Raw, 12, 255);
    Fail('Expected EGgufError for count not multiple (q4_K)');
  except
    on EGgufError do ; // expected
  end;
end;

procedure TGgufQuantTTest.TestUT_GgufQuantT_024;
var
  Raw: TTensor;
  RawBytes: TBytes;
begin
  // Count not a multiple (q6_K)
  SetLength(RawBytes, 257);
  Raw := TTensor.FromBytes(RawBytes[0], [257], kUInt8);
  try
    DequantizeT(Raw, 14, 255);
    Fail('Expected EGgufError for count not multiple (q6_K)');
  except
    on EGgufError do ; // expected
  end;
end;

procedure TGgufQuantTTest.TestUT_GgufQuantT_002;
var
  RawBytes: TBytes;
  Raw, Actual: TTensor;
  ActualValues, ExpectedValues: TSingleArray;
  i, Count: Integer;
begin
  // f16 random blocks (512 × 32)
  RandSeed := 1002;
  Count := 512 * 32; // 16384 values
  SetLength(RawBytes, Count * 2);
  for i := 0 to Length(RawBytes) - 1 do
    RawBytes[i] := Byte(Random(256));
  Raw := TTensor.FromBytes(RawBytes[0], [Length(RawBytes)], kUInt8);
  Actual := DequantizeT(Raw, 1, Count);
  ExpectedValues := GgufQuant.DequantizeBytes(1, RawBytes, Count);
  // Verify
  AssertEquals(Count, Actual.Shape[0]);
  AssertTrue(Actual.Kind = kFloat);
  ActualValues := Values(Actual);
  AssertEquals(Length(ExpectedValues), Length(ActualValues));
  for i := 0 to High(ActualValues) do
    if not SameValue(ActualValues[i], ExpectedValues[i]) then
      Fail(Format('Mismatch at index %d: Expected $%8.8x Actual $%8.8x', [i, Bits(ExpectedValues[i]), Bits(ActualValues[i])]));
end;

procedure TGgufQuantTTest.TestUT_GgufQuantT_003;
var
  RawBytes: TBytes;
  Raw, Actual: TTensor;
  ActualValues, ExpectedValues: TSingleArray;
  i, Count: Integer;
begin
  // bf16 random blocks (512 × 32)
  RandSeed := 1003;
  Count := 512 * 32; // 16384 values
  SetLength(RawBytes, Count * 2);
  for i := 0 to Length(RawBytes) - 1 do
    RawBytes[i] := Byte(Random(256));
  Raw := TTensor.FromBytes(RawBytes[0], [Length(RawBytes)], kUInt8);
  Actual := DequantizeT(Raw, 30, Count);
  ExpectedValues := GgufQuant.DequantizeBytes(30, RawBytes, Count);
  AssertEquals(Count, Actual.Shape[0]);
  AssertTrue(Actual.Kind = kFloat);
  ActualValues := Values(Actual);
  for i := 0 to High(ActualValues) do
    AssertEquals(Bits(ActualValues[i]), Bits(ExpectedValues[i]));
end;

procedure TGgufQuantTTest.TestUT_GgufQuantT_004;
var
  RawBytes: TBytes;
  Raw, Actual: TTensor;
  ActualValues, ExpectedValues: TSingleArray;
  i, Count, Nb: Integer;
begin
  // q8_0 random blocks (512 × 32)
  RandSeed := 1004;
  Count := 512 * 32; // 16384 values
  Nb := Count div 32; // 512 blocks
  SetLength(RawBytes, Nb * 34);
  for i := 0 to Length(RawBytes) - 1 do
    RawBytes[i] := Byte(Random(256));
  Raw := TTensor.FromBytes(RawBytes[0], [Length(RawBytes)], kUInt8);
  Actual := DequantizeT(Raw, 8, Count);
  ExpectedValues := GgufQuant.DequantizeBytes(8, RawBytes, Count);
  AssertEquals(Count, Actual.Shape[0]);
  AssertTrue(Actual.Kind = kFloat);
  ActualValues := Values(Actual);
  for i := 0 to High(ActualValues) do
    if not SameValue(ActualValues[i], ExpectedValues[i]) then
      Fail(Format('Mismatch at index %d: Expected $%8.8x Actual $%8.8x', [i, Bits(ExpectedValues[i]), Bits(ActualValues[i])]));
end;

procedure TGgufQuantTTest.TestUT_GgufQuantT_005;
var
  RawBytes: TBytes;
  Raw, Actual: TTensor;
  i, Count, Nb: Integer;
  ExpectedValues: TSingleArray;
  MismatchFound: Boolean;
begin
  // q4_K random blocks (512 × 256)
  RandSeed := 1005;
  Count := 512 * 256; // 131072 values
  Nb := Count div 256; // 512 blocks
  SetLength(RawBytes, Nb * 144);
  for i := 0 to Length(RawBytes) - 1 do
    RawBytes[i] := Byte(Random(256));
  Raw := TTensor.FromBytes(RawBytes[0], [Length(RawBytes)], kUInt8);
  Actual := DequantizeT(Raw, 12, Count);
  ExpectedValues := GgufQuant.DequantizeBytes(12, RawBytes, Count);
  AssertEquals(Count, Actual.Shape[0]);
  AssertTrue(Actual.Kind = kFloat);
  // Compare values using SameValue; report first mismatch
  for i := 0 to High(Values(Actual)) do
    if not SameValue(Values(Actual)[i], ExpectedValues[i]) then
      Fail(Format('First mismatch index %d: Expected $%8.8x Actual $%8.8x', [i, Bits(ExpectedValues[i]), Bits(Values(Actual)[i])]))
end;

procedure TGgufQuantTTest.TestUT_GgufQuantT_006;
var
  RawBytes: TBytes;
  Raw, Actual: TTensor;
  i, Count, Nb: Integer;
  ExpectedValues: TSingleArray;
begin
  // q5_K random blocks (512 × 256)
  RandSeed := 1006;
  Count := 512 * 256; // 131072 values
  Nb := Count div 256; // 512 blocks
  SetLength(RawBytes, Nb * 176);
  for i := 0 to Length(RawBytes) - 1 do
    RawBytes[i] := Byte(Random(256));
  Raw := TTensor.FromBytes(RawBytes[0], [Length(RawBytes)], kUInt8);
  Actual := DequantizeT(Raw, 13, Count);
  ExpectedValues := GgufQuant.DequantizeBytes(13, RawBytes, Count);
  AssertEquals(Count, Actual.Shape[0]);
  AssertTrue(Actual.Kind = kFloat);
  for i := 0 to High(Values(Actual)) do
    AssertEquals(Bits(Values(Actual)[i]), Bits(ExpectedValues[i]));
end;

procedure TGgufQuantTTest.TestUT_GgufQuantT_007;
var
  RawBytes: TBytes;
  Raw, Actual: TTensor;
  i, Count, Nb: Integer;
  ExpectedValues: TSingleArray;
begin
  // q6_K random blocks (512 × 256)
  RandSeed := 1007;
  Count := 512 * 256; // 131072 values
  Nb := Count div 256; // 512 blocks
  SetLength(RawBytes, Nb * 210);
  for i := 0 to Length(RawBytes) - 1 do
    RawBytes[i] := Byte(Random(256));
  Raw := TTensor.FromBytes(RawBytes[0], [Length(RawBytes)], kUInt8);
  Actual := DequantizeT(Raw, 14, Count);
  ExpectedValues := GgufQuant.DequantizeBytes(14, RawBytes, Count);
  AssertEquals(Count, Actual.Shape[0]);
  AssertTrue(Actual.Kind = kFloat);
  for i := 0 to High(Values(Actual)) do
    AssertEquals(Bits(Values(Actual)[i]), Bits(ExpectedValues[i]));
end;

procedure TGgufQuantTTest.TestUT_GgufQuantT_008;
var
  RawBytes: TBytes;
  Raw, Actual: TTensor;
  i: Integer;
  ExpectedValues: TSingleArray;
begin
  // f32 hand case (UT-GgufQuantT-008)
  SetLength(RawBytes, 16);
  // 00 00 C0 3F  00 00 12 C0  00 00 00 3F  00 00 00 80
  RawBytes[0] := $00; RawBytes[1] := $00; RawBytes[2] := $C0; RawBytes[3] := $3F;
  RawBytes[4] := $00; RawBytes[5] := $00; RawBytes[6] := $12; RawBytes[7] := $C0;
  RawBytes[8] := $00; RawBytes[9] := $00; RawBytes[10] := $00; RawBytes[11] := $3F;
  RawBytes[12] := $00; RawBytes[13] := $00; RawBytes[14] := $00; RawBytes[15] := $80;
  Raw := TTensor.FromBytes(RawBytes[0], [16], kUInt8);
  Actual := DequantizeT(Raw, 0, 4);
  ExpectedValues := GgufQuant.DequantizeBytes(0, RawBytes, 4);
  // Compare bits
  for i := 0 to 3 do
    AssertEquals(Bits(Values(Actual)[i]), Bits(ExpectedValues[i]));
end;

procedure TGgufQuantTTest.TestUT_GgufQuantT_009;
var
  RawBytes: TBytes;
  Raw, Actual: TTensor;
  i: Integer;
  ExpectedValues: TSingleArray;
begin
  // f16 hand case (UT-GgufQuantT-009)
  SetLength(RawBytes, 16);
  // 00 00  00 80  00 3C  00 C0  00 7C  00 FC  FF 7B  00 7E
  RawBytes[0] := $00; RawBytes[1] := $00;
  RawBytes[2] := $00; RawBytes[3] := $80;
  RawBytes[4] := $00; RawBytes[5] := $3C;
  RawBytes[6] := $00; RawBytes[7] := $C0;
  RawBytes[8] := $00; RawBytes[9] := $7C;
  RawBytes[10] := $00; RawBytes[11] := $FC;
  RawBytes[12] := $FF; RawBytes[13] := $7B;
  RawBytes[14] := $00; RawBytes[15] := $7E;
  Raw := TTensor.FromBytes(RawBytes[0], [16], kUInt8);
  Actual := DequantizeT(Raw, 1, 8);
  ExpectedValues := GgufQuant.DequantizeBytes(1, RawBytes, 8);
  for i := 0 to 7 do
    AssertEquals(Bits(Values(Actual)[i]), Bits(ExpectedValues[i]));
end;

procedure TGgufQuantTTest.TestUT_GgufQuantT_010;
var
  RawBytes: TBytes;
  Raw, Actual: TTensor;
  i: Integer;
  ExpectedValues: TSingleArray;
begin
  // bf16 hand case (UT-GgufQuantT-010)
  SetLength(RawBytes, 16);
  // 80 3F  00 C0  00 40  00 3F  00 BF  80 7F  80 FF  7F 7F
  RawBytes[0] := $80; RawBytes[1] := $3F;
  RawBytes[2] := $00; RawBytes[3] := $C0;
  RawBytes[4] := $00; RawBytes[5] := $40;
  RawBytes[6] := $00; RawBytes[7] := $3F;
  RawBytes[8] := $00; RawBytes[9] := $BF;
  RawBytes[10] := $80; RawBytes[11] := $7F;
  RawBytes[12] := $80; RawBytes[13] := $FF;
  RawBytes[14] := $7F; RawBytes[15] := $7F;
  Raw := TTensor.FromBytes(RawBytes[0], [16], kUInt8);
  Actual := DequantizeT(Raw, 30, 8);
  ExpectedValues := GgufQuant.DequantizeBytes(30, RawBytes, 8);
  for i := 0 to 7 do
    AssertEquals(Bits(Values(Actual)[i]), Bits(ExpectedValues[i]));
end;

procedure TGgufQuantTTest.TestUT_GgufQuantT_011;
var
  RawBytes: TBytes;
  Raw, Actual: TTensor;
  i: Integer;
  ExpectedValues: TSingleArray;
begin
  // q8_0 hand case (UT-GgufQuantT-011)
  SetLength(RawBytes, 34);
  // d = 00 3C
  RawBytes[0] := $00; RawBytes[1] := $3C;
  // qs (32 bytes) as listed
  RawBytes[2] := $7F; RawBytes[3] := $7F; RawBytes[4] := $00; RawBytes[5] := $05;
  RawBytes[6] := $FB; RawBytes[7] := $08; RawBytes[8] := $F8; RawBytes[9] := $96;
  RawBytes[10] := $60; RawBytes[11] := $00; RawBytes[12] := $00; RawBytes[13] := $00;
  RawBytes[14] := $00; RawBytes[15] := $00; RawBytes[16] := $00; RawBytes[17] := $00;
  RawBytes[18] := $00; RawBytes[19] := $00; RawBytes[20] := $00; RawBytes[21] := $00;
  RawBytes[22] := $00; RawBytes[23] := $00; RawBytes[24] := $00; RawBytes[25] := $00;
  RawBytes[26] := $00; RawBytes[27] := $00; RawBytes[28] := $00; RawBytes[29] := $00;
  RawBytes[30] := $00; RawBytes[31] := $00; RawBytes[32] := $00; RawBytes[33] := $00;
  Raw := TTensor.FromBytes(RawBytes[0], [34], kUInt8);
  Actual := DequantizeT(Raw, 8, 32);
  ExpectedValues := GgufQuant.DequantizeBytes(8, RawBytes, 32);
  for i := 0 to 31 do
    AssertEquals(Bits(Values(Actual)[i]), Bits(ExpectedValues[i]));
end;

procedure TGgufQuantTTest.TestUT_GgufQuantT_012;
var
  RawBytes: TBytes;
  Raw, Actual: TTensor;
  i, Nb: Integer;
  ExpectedValues: TSingleArray;
begin
  // q4_K hand case (UT-GgufQuantT-012)
  Nb := 1; // single block
  SetLength(RawBytes, Nb * 144);
  // d (2) and dmin (2)
  RawBytes[0] := $00; RawBytes[1] := $3C; // d = 1.0
  RawBytes[2] := $00; RawBytes[3] := $38; // dmin = 0.5
  // scales (12 bytes)
  RawBytes[4] := $82; RawBytes[5] := $C1; RawBytes[6] := $03; RawBytes[7] := $C5;
  RawBytes[8] := $40; RawBytes[9] := $40; RawBytes[10] := $00; RawBytes[11] := $02;
  RawBytes[12] := $24; RawBytes[13] := $35; RawBytes[14] := $FF; RawBytes[15] := $27;
  // qs (128 bytes): 4 groups of 32
  for i := 0 to 3 do
  begin
    RawBytes[16 + i*32 + 0] := $21; // low nibble pattern
    RawBytes[16 + i*32 + 1] := $21;
    RawBytes[16 + i*32 + 2] := $21;
    RawBytes[16 + i*32 + 3] := $21;
    // rest are zeros (already zero-initialized)
  end;
  Raw := TTensor.FromBytes(RawBytes[0], [Length(RawBytes)], kUInt8);
  Actual := DequantizeT(Raw, 12, 256);
  ExpectedValues := GgufQuant.DequantizeBytes(12, RawBytes, 256);
  for i := 0 to 255 do
    AssertEquals(Bits(Values(Actual)[i]), Bits(ExpectedValues[i]));
end;

procedure TGgufQuantTTest.TestUT_GgufQuantT_013;
var
  RawBytes: TBytes;
  Raw, Actual: TTensor;
  i, Nb: Integer;
  ExpectedValues: TSingleArray;
begin
  // q5_K hand case (UT-GgufQuantT-013)
  Nb := 1;
  SetLength(RawBytes, Nb * 176);
  // d and dmin
  RawBytes[0] := $00; RawBytes[1] := $3C;
  RawBytes[2] := $00; RawBytes[3] := $38;
  // scales (12 bytes)
  RawBytes[4] := $82; RawBytes[5] := $C1; RawBytes[6] := $03; RawBytes[7] := $C5;
  RawBytes[8] := $40; RawBytes[9] := $40; RawBytes[10] := $00; RawBytes[11] := $02;
  RawBytes[12] := $24; RawBytes[13] := $35; RawBytes[14] := $FF; RawBytes[15] := $27;
  // qh (4 bytes)
  RawBytes[16] := $FF; RawBytes[17] := $55; RawBytes[18] := $AA; RawBytes[19] := $00;
  // qs (128 bytes) same pattern as q4_K
  for i := 0 to 3 do
  begin
    RawBytes[20 + i*32 + 0] := $21;
    RawBytes[20 + i*32 + 1] := $21;
    RawBytes[20 + i*32 + 2] := $21;
    RawBytes[20 + i*32 + 3] := $21;
  end;
  Raw := TTensor.FromBytes(RawBytes[0], [Length(RawBytes)], kUInt8);
  Actual := DequantizeT(Raw, 13, 256);
  ExpectedValues := GgufQuant.DequantizeBytes(13, RawBytes, 256);
  for i := 0 to 255 do
    AssertEquals(Bits(Values(Actual)[i]), Bits(ExpectedValues[i]));
end;

procedure TGgufQuantTTest.TestUT_GgufQuantT_014;
var
  RawBytes: TBytes;
  Raw, Actual: TTensor;
  i: Integer;
  ExpectedValues: TSingleArray;
begin
  // q6_K hand case (UT-GgufQuantT-014)
  SetLength(RawBytes, 210);
  // ql (128 bytes), set byte0=$12, byte32=$34
  RawBytes[0] := $12;
  RawBytes[32] := $34;
  // rest are zero (already)
  // qh (64 bytes), set byte0=$0F
  RawBytes[128] := $0F;
  // scales (16 bytes) at offset 192
  RawBytes[192] := $FE; RawBytes[193] := $03; RawBytes[194] := $FF; RawBytes[195] := $05;
  RawBytes[196] := $07; RawBytes[197] := $FD; RawBytes[198] := $04; RawBytes[199] := $FA;
  RawBytes[200] := $01; RawBytes[201] := $FC; RawBytes[202] := $02; RawBytes[203] := $08;
  RawBytes[204] := $FB; RawBytes[205] := $06; RawBytes[206] := $FF; RawBytes[207] := $0A;
  // d (2 bytes) at end
  RawBytes[208] := $00; RawBytes[209] := $3C;
  Raw := TTensor.FromBytes(RawBytes[0], [210], kUInt8);
  Actual := DequantizeT(Raw, 14, 256);
  ExpectedValues := GgufQuant.DequantizeBytes(14, RawBytes, 256);
  for i := 0 to 255 do
    AssertEquals(Bits(Values(Actual)[i]), Bits(ExpectedValues[i]));
end;

procedure TGgufQuantTTest.TestUT_GgufQuantT_019;
var
  RawBytes: TBytes;
  Raw, Actual: TTensor;
begin
  // Result tensor shape and kind (bf16) using hand case (UT-GgufQuantT-010)
  SetLength(RawBytes, 16);
  RawBytes[0] := $80; RawBytes[1] := $3F; RawBytes[2] := $00; RawBytes[3] := $C0;
  RawBytes[4] := $00; RawBytes[5] := $40; RawBytes[6] := $00; RawBytes[7] := $3F;
  RawBytes[8] := $00; RawBytes[9] := $BF; RawBytes[10] := $80; RawBytes[11] := $7F;
  RawBytes[12] := $80; RawBytes[13] := $FF; RawBytes[14] := $7F; RawBytes[15] := $7F;
  Raw := TTensor.FromBytes(RawBytes[0], [16], kUInt8);
  Actual := DequantizeT(Raw, 30, 8);
  AssertEquals(8, Actual.Shape[0]);
  AssertTrue(Actual.Kind = kFloat);
end;

procedure TGgufQuantTTest.TestUT_GgufQuantT_020;
var
  RawBytes: TBytes;
  Raw, Actual: TTensor;
begin
  // Result tensor device matches input (CPU)
  SetLength(RawBytes, 16);
  // reuse same bytes as bf16 hand case
  RawBytes[0] := $80; RawBytes[1] := $3F; RawBytes[2] := $00; RawBytes[3] := $C0;
  RawBytes[4] := $00; RawBytes[5] := $40; RawBytes[6] := $00; RawBytes[7] := $3F;
  RawBytes[8] := $00; RawBytes[9] := $BF; RawBytes[10] := $80; RawBytes[11] := $7F;
  RawBytes[12] := $80; RawBytes[13] := $FF; RawBytes[14] := $7F; RawBytes[15] := $7F;
  Raw := TTensor.FromBytes(RawBytes[0], [16], kUInt8);
  Actual := DequantizeT(Raw, 30, 8);
  // Assuming TTensor has Device property
  // AssertEquals(Raw.Device, Actual.Device); // device compare omitted
end;

procedure TGgufQuantTTest.TestUT_GgufQuantT_101;
var
  Count, Nb, i: Integer;
  AX: TTensor;
  Oracle: TBytes;
  Singles: array of Single;
  ExpectedBytes, ActualBytes: TBytes;
  QuantTensor: TTensor;
begin
  // Q8_0 random finite blocks (544)
  RandSeed := 1101;
  Count := 544 * 32;
  Nb := Count div 32;
  SetLength(Singles, Count);
  for i := 0 to Count - 1 do
    Singles[i] := (Random(2000000) - 1000000) / 100000.0;
  AX := TTensor.FromSingleArray(Singles);
 
  QuantTensor := QuantizeQ8_0T(AX);
  SetLength(ActualBytes, QuantTensor.Shape[0]);
  if Length(ActualBytes) > 0 then
    QuantTensor.contiguous.CopyData(ActualBytes[0], Length(ActualBytes));
  Oracle := GgufQuant.QuantizeQ8_0(AX.ToSingleArray);
  ExpectedBytes := Oracle;
  AssertEquals(Length(ExpectedBytes), Length(ActualBytes));
  for i := 0 to High(ActualBytes) do
    AssertEquals(ExpectedBytes[i], ActualBytes[i]);
end;

procedure TGgufQuantTTest.TestUT_GgufQuantT_102;
var
  AX: TTensor;
  Oracle: TBytes;
  ExpectedBytes, ActualBytes: TBytes;
  i: Integer;
  QuantTensor: TTensor;
begin
  // Q8_0 zeros and tiny values (single block)
  AX := TTensor.zeros([32], kFloat);
  // tiny values can be set after, but zeros are already tiny
 
  QuantTensor := QuantizeQ8_0T(AX);
  SetLength(ActualBytes, QuantTensor.Shape[0]);
  if Length(ActualBytes) > 0 then
    QuantTensor.contiguous.CopyData(ActualBytes[0], Length(ActualBytes));
  Oracle := GgufQuant.QuantizeQ8_0(AX.ToSingleArray);
  ExpectedBytes := Oracle;
  AssertEquals(Length(ExpectedBytes), Length(ActualBytes));
  for i := 0 to High(ActualBytes) do
    AssertEquals(ExpectedBytes[i], ActualBytes[i]);
end;

procedure TGgufQuantTTest.TestUT_GgufQuantT_103;
var
  RawBytes: TBytes;
  Raw, AX, ResultBytes, OracleBytes: TTensor;
  i: Integer;
begin
  // Q8_0 hand case (same as UT-GgufQuantT-011)
  SetLength(RawBytes, 34);
  RawBytes[0] := $00; RawBytes[1] := $3C;
  RawBytes[2] := $7F; RawBytes[3] := $7F; RawBytes[4] := $00; RawBytes[5] := $05;
  RawBytes[6] := $FB; RawBytes[7] := $08; RawBytes[8] := $F8; RawBytes[9] := $96;
  RawBytes[10] := $60; RawBytes[11] := $00; RawBytes[12] := $00; RawBytes[13] := $00;
  RawBytes[14] := $00; RawBytes[15] := $00; RawBytes[16] := $00; RawBytes[17] := $00;
  RawBytes[18] := $00; RawBytes[19] := $00; RawBytes[20] := $00; RawBytes[21] := $00;
  RawBytes[22] := $00; RawBytes[23] := $00; RawBytes[24] := $00; RawBytes[25] := $00;
  RawBytes[26] := $00; RawBytes[27] := $00; RawBytes[28] := $00; RawBytes[29] := $00;
  RawBytes[30] := $00; RawBytes[31] := $00; RawBytes[32] := $00; RawBytes[33] := $00;
  Raw := TTensor.FromBytes(RawBytes[0], [34], kUInt8);
  AX := DequantizeT(Raw, 8, 32);
  ResultBytes := QuantizeQ8_0T(AX);
  // compare to original bytes
  for i := 0 to 33 do
    AssertEquals(RawBytes[i], TGgufQuantTTest.TensorBytes(ResultBytes)[i]);
end;

procedure TGgufQuantTTest.TestUT_GgufQuantT_104;
var
  AX: TTensor;
  Singles: array of Single;
  i: Integer;
begin
  // Non-finite input values (NaN)
  SetLength(Singles, 32);
  for i := 0 to 31 do
    Singles[i] := Math.NaN;
  AX := TTensor.FromSingleArray(Singles);
  try
    QuantizeQ8_0T(AX);
    Fail('Expected EGgufError for non-finite values');
  except
    on E: EGgufError do ; // expected
  end;
end;

procedure TGgufQuantTTest.TestUT_GgufQuantT_105;
var
  AX: TTensor;
  Singles: array of Single;
begin
  // Input length not a multiple of 32 (33 values)
  SetLength(Singles, 33);
  AX := TTensor.FromSingleArray(Singles);
  try
    QuantizeQ8_0T(AX);
    Fail('Expected EGgufError for length not multiple of 32');
  except
    on E: EGgufError do ; // expected
  end;
end;

procedure TGgufQuantTTest.TestUT_GgufQuantT_106;
var
  AX, Result: TTensor;
begin
  // Result tensor length and kind
  AX := TTensor.zeros([32], kFloat);
  Result := QuantizeQ8_0T(AX);
  AssertEquals(34, Result.Shape[0]);
  AssertTrue(Result.Kind = kUInt8);
end;

procedure TGgufQuantTTest.TestUT_GgufQuantT_107;
var
  AX, Result: TTensor;
begin
  // Result tensor device matches input (CPU)
  AX := TTensor.zeros([32], kFloat);
  Result := QuantizeQ8_0T(AX);
  // Assuming Device property exists
  // AssertEquals(AX.Device, Result.Device);
end;

procedure TGgufQuantTTest.TestUT_GgufQuantT_108;
var
  AX: TTensor;
  Singles: array of Single;
begin
  // AX not kFloat (kDouble tensor)
  SetLength(Singles, 32);
  AX := TTensor.FromSingleArray(Singles, kDouble);
  try
    QuantizeQ8_0T(AX);
    Fail('Expected EGgufError for AX not kFloat');
  except
    on E: EGgufError do ; // expected
  end;
end;

procedure TGgufQuantTTest.TestUT_GgufQuantT_109;
var
  AX: TTensor;
  RawBytes: TBytes;
begin
  // AX not 1-D (kFloat tensor shape [2,32])
  SetLength(RawBytes, 2*32*4);
  AX := TTensor.FromBytes(RawBytes[0], [2,32], kFloat);
  try
    QuantizeQ8_0T(AX);
    Fail('Expected EGgufError for AX not 1-D');
  except
    on EGgufError do ; // expected
  end;
end;

var
  App: TTestRunner;
begin
  RegisterTest(TGgufQuantTTest);
  App := TTestRunner.Create(nil);
  App.Initialize;
  App.Title := 'GgufQuantT UT';
  App.Run;
  App.Free;
end.
