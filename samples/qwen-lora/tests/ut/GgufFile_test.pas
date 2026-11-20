{ UT program for src/GgufFile.pas — T01b stage 1 (T01b1) and stage 2 (T01b2):
  the rows UT-GgufFile-001..049 (048 is [retired]) of
  tests/ut/cases/GgufFile.cases.md.
  Stage 2 adds the tensor directory: GgufTypeName, GgufTensorBytes, the tensor
  read in TGgufFile.Open (names, dims, type, offset, DataOffset and the
  checks), FindTensor, ReadTensorBytes, and the DataOffset / TensorCount /
  Tensor[] properties.
  The test builds its own small GGUF files under work/ut/gguf/ with the
  writer kept in this program (work/ is git-ignored; the gate wipes
  work/ut before running, so every file is written again on every run).
  Invoked with `--all --format=plain` (see .sdlc/ut-coverage.sh). }
program GgufFile_test;

{$mode objfpc}{$H+}

uses
  SysUtils, Classes, GgufFile, fpcunit, testregistry, consoletestrunner;

const
  K_DIR = 'work/ut/gguf';

  { 23.5 "héllo" as raw UTF-8 bytes, code-page independent:
    h = 0x68, é = 0xC3 0xA9, l = 0x6C, l = 0x6C, o = 0x6F. }
  K_HELLO_UTF8: array[0..5] of Byte = (104, 195, 169, 108, 108, 111);

  { All 64 bits set (002's uint64 value; QWord(-1) avoids a const-fold
    warning on the hex literal). }
  K_MAX_U64: QWord = QWord(-1);

type
  TGgufFileTest = class(TTestCase)
  public
    procedure WU32(S: TBytesStream; V: LongWord);
    procedure WU64(S: TBytesStream; V: QWord);
    procedure WB(S: TBytesStream; V: Byte);
    procedure WStr(S: TBytesStream; const A: string);
    procedure WVal(S: TBytesStream; const AKey: string;
      AVT: LongWord; ARaw: QWord; AWidth: Integer);
    procedure WValF32(S: TBytesStream; const AKey: string; AVal: Single);
    procedure WValF64(S: TBytesStream; const AKey: string; AVal: Double);
    procedure WValStr(S: TBytesStream; const AKey, AVal: string);
    procedure WValStrRaw(S: TBytesStream; const AKey: string;
      const ABytes: array of Byte);
    procedure WHeader(S: TBytesStream; AVersion: LongWord;
      ANTensors, ANKV: QWord);
    procedure WArrInt(S: TBytesStream; const AKey: string;
      AElemVT: LongWord; const AVals: array of Int64);
    procedure WArrF32(S: TBytesStream; const AKey: string;
      const AVals: array of Single);
    procedure WArrF64(S: TBytesStream; const AKey: string;
      const AVals: array of Double);
    procedure WArrStr(S: TBytesStream; const AKey: string;
      const AVals: array of string);
    procedure WArrBad(S: TBytesStream; const AKey: string;
      AElemVT: LongWord; AN: QWord);
    procedure WTensor(S: TBytesStream; const AName: string;
      ANDims: LongWord; const ADims: array of QWord;
      ATType: LongWord; AOffset: QWord);
    procedure PadTo(S: TBytesStream; ATarget: QWord);
    function MakePat(AN: Integer; ABase: Byte): TBytes;
  published
    procedure TestUT_GgufFile_001;
    procedure TestUT_GgufFile_002;
    procedure TestUT_GgufFile_003;
    procedure TestUT_GgufFile_004;
    procedure TestUT_GgufFile_005;
    procedure TestUT_GgufFile_006;
    procedure TestUT_GgufFile_007;
    procedure TestUT_GgufFile_008;
    procedure TestUT_GgufFile_009;
    procedure TestUT_GgufFile_010;
    procedure TestUT_GgufFile_011;
    procedure TestUT_GgufFile_012;
    procedure TestUT_GgufFile_013;
    procedure TestUT_GgufFile_014;
    procedure TestUT_GgufFile_015;
    procedure TestUT_GgufFile_016;
    procedure TestUT_GgufFile_017;
    procedure TestUT_GgufFile_018;
    procedure TestUT_GgufFile_019;
    procedure TestUT_GgufFile_020;
    procedure TestUT_GgufFile_021;
    procedure TestUT_GgufFile_022;
    procedure TestUT_GgufFile_023;
    procedure TestUT_GgufFile_024;
    procedure TestUT_GgufFile_025;
    procedure TestUT_GgufFile_026;
    procedure TestUT_GgufFile_027;
    procedure TestUT_GgufFile_028;
    procedure TestUT_GgufFile_029;
    procedure TestUT_GgufFile_030;
    procedure TestUT_GgufFile_031;
    procedure TestUT_GgufFile_032;
    procedure TestUT_GgufFile_033;
    procedure TestUT_GgufFile_034;
    procedure TestUT_GgufFile_035;
    procedure TestUT_GgufFile_036;
    procedure TestUT_GgufFile_037;
    procedure TestUT_GgufFile_038;
    procedure TestUT_GgufFile_039;
    procedure TestUT_GgufFile_040;
    procedure TestUT_GgufFile_041;
    procedure TestUT_GgufFile_042;
    procedure TestUT_GgufFile_043;
    procedure TestUT_GgufFile_044;
    procedure TestUT_GgufFile_045;
    procedure TestUT_GgufFile_046;
    procedure TestUT_GgufFile_047;
    procedure TestUT_GgufFile_049;
  end;

{ --- the writer (little-endian, exactly the GGUF wire layout) --- }

procedure TGgufFileTest.WU32(S: TBytesStream; V: LongWord);
begin
  S.WriteBuffer(V, 4);
end;

procedure TGgufFileTest.WU64(S: TBytesStream; V: QWord);
begin
  S.WriteBuffer(V, 8);
end;

procedure TGgufFileTest.WB(S: TBytesStream; V: Byte);
begin
  S.WriteBuffer(V, 1);
end;

{ A GGUF string: UInt64 number of bytes, then that many bytes. }
procedure TGgufFileTest.WStr(S: TBytesStream; const A: string);
var
  C: QWord;
begin
  C := Length(A);
  WU64(S, C);
  if C > 0 then
    S.WriteBuffer(A[1], LongInt(C));
end;

{ A scalar entry: key, value type (facts' wire codes 0..12), then
  AWidth bytes with the low twos-complement bytes of ARaw. }
procedure TGgufFileTest.WVal(S: TBytesStream; const AKey: string;
  AVT: LongWord; ARaw: QWord; AWidth: Integer);
var
  Buf: array[0..7] of Byte;
begin
  WStr(S, AKey);
  WU32(S, AVT);
  Move(ARaw, Buf, AWidth);
  S.WriteBuffer(Buf, AWidth);
end;

procedure TGgufFileTest.WValF32(S: TBytesStream; const AKey: string; AVal: Single);
begin
  WStr(S, AKey);
  WU32(S, 6);
  S.WriteBuffer(AVal, 4);
end;

procedure TGgufFileTest.WValF64(S: TBytesStream; const AKey: string; AVal: Double);
begin
  WStr(S, AKey);
  WU32(S, 12);
  S.WriteBuffer(AVal, 8);
end;

procedure TGgufFileTest.WValStr(S: TBytesStream; const AKey, AVal: string);
begin
  WStr(S, AKey);
  WU32(S, 8);
  WStr(S, AVal);
end;

procedure TGgufFileTest.WValStrRaw(S: TBytesStream; const AKey: string;
  const ABytes: array of Byte);
begin
  WStr(S, AKey);
  WU32(S, 8);
  WU64(S, Length(ABytes));
  S.WriteBuffer(ABytes[0], Length(ABytes));
end;

procedure TGgufFileTest.WHeader(S: TBytesStream; AVersion: LongWord;
  ANTensors, ANKV: QWord);
begin
  WU32(S, $46554747); { "GGUF" }
  WU32(S, AVersion);
  WU64(S, ANTensors);
  WU64(S, ANKV);
end;

procedure TGgufFileTest.WArrInt(S: TBytesStream; const AKey: string;
  AElemVT: LongWord; const AVals: array of Int64);
var
  I, W: Integer;
  T: array[0..7] of Byte;
begin
  WStr(S, AKey);
  WU32(S, 9); { array }
  WU32(S, AElemVT);
  WU64(S, Length(AVals));
  case AElemVT of
    0, 1, 7: { uint8, int8, bool }
      W := 1;
    2, 3: { uint16, int16 }
      W := 2;
    4, 5: { uint32, int32 }
      W := 4;
  else { uint64, int64 }
      W := 8;
  end;
  for I := Low(AVals) to High(AVals) do
  begin
    Move(AVals[I], T, W);
    S.WriteBuffer(T, W);
  end;
end;

procedure TGgufFileTest.WArrF32(S: TBytesStream; const AKey: string;
  const AVals: array of Single);
var
  I: Integer;
begin
  WStr(S, AKey);
  WU32(S, 9);
  WU32(S, 6);
  WU64(S, Length(AVals));
  for I := Low(AVals) to High(AVals) do
    S.WriteBuffer(AVals[I], 4);
end;

procedure TGgufFileTest.WArrF64(S: TBytesStream; const AKey: string;
  const AVals: array of Double);
var
  I: Integer;
begin
  WStr(S, AKey);
  WU32(S, 9);
  WU32(S, 12);
  WU64(S, Length(AVals));
  for I := Low(AVals) to High(AVals) do
    S.WriteBuffer(AVals[I], 8);
end;

procedure TGgufFileTest.WArrStr(S: TBytesStream; const AKey: string;
  const AVals: array of string);
var
  I: Integer;
begin
  WStr(S, AKey);
  WU32(S, 9);
  WU32(S, 8);
  WU64(S, Length(AVals));
  for I := Low(AVals) to High(AVals) do
    WStr(S, AVals[I]);
end;

{ 011: an entry whose array element type is array itself (9). }
procedure TGgufFileTest.WArrBad(S: TBytesStream; const AKey: string;
  AElemVT: LongWord; AN: QWord);
var
  X: Byte;
begin
  WStr(S, AKey);
  WU32(S, 9);
  WU32(S, AElemVT);
  WU64(S, AN);
  X := 255;
  S.WriteBuffer(X, 1);
end;

{ A byte-exact RawByteString from raw bytes (no code page involved). }
function BytesToRBS(const A: array of Byte): RawByteString;
var
  B: TBytes;
  I: Integer;
begin
  SetLength(B, Length(A));
  for I := 0 to High(A) do
    B[I] := A[I];
  SetString(Result, PAnsiChar(@B[0]), Length(B));
end;

{ A tensor directory entry (2.6/8 wire order): name (length-prefixed),
  n_dims, that many UInt64 dimensions (padded with 1 if the caller passed
  fewer), the tensor type, the offset relative to the data section. }
procedure TGgufFileTest.WTensor(S: TBytesStream; const AName: string;
  ANDims: LongWord; const ADims: array of QWord;
  ATType: LongWord; AOffset: QWord);
var
  I: Integer;
begin
  WStr(S, AName);
  WU32(S, ANDims);
  for I := 1 to ANDims do
  begin
    if I <= Length(ADims) then
      WU64(S, ADims[I - 1])
    else
      WU64(S, 1);
  end;
  WU32(S, ATType);
  WU64(S, AOffset);
end;

{ Zero pads the stream up to ATarget (the 2.6/8 alignment padding
  between the directory and the data section). }
procedure TGgufFileTest.PadTo(S: TBytesStream; ATarget: QWord);
var
  B: Byte;
begin
  B := 0;
  while S.Position < ATarget do
    S.WriteBuffer(B, 1);
end;

{ A distinct data-section fill: N bytes, byte i = (ABase + i) mod 256. }
function TGgufFileTest.MakePat(AN: Integer; ABase: Byte): TBytes;
var
  I: Integer;
begin
  SetLength(Result, AN);
  for I := 0 to AN - 1 do
    Result[I] := Byte((LongWord(ABase) + LongWord(I)) mod 256);
end;

{ --- the tests (one published method per case row of this stage) --- }
{ 001: all six unsigned/signed integer types, representative values incl. negatives. }
procedure TGgufFileTest.TestUT_GgufFile_001;
var
  S: TBytesStream;
  P: string;
  F: TGgufFile;
begin
  P := K_DIR + '/UT_GgufFile_001.gguf';
  S := TBytesStream.Create;
  try
    WHeader(S, 3, 0, 6);
    WVal(S, 'u8', 0, 200, 1);
    WVal(S, 'i8', 1, 251, 1);
    WVal(S, 'u16', 2, 65535, 2);
    WVal(S, 'i16', 3, 32768, 2);
    WVal(S, 'u32', 4, 4294967295, 4);
    WVal(S, 'i32', 5, 2147483648, 4);
    S.SaveToFile(P);
  finally
    S.Free;
  end;
  F := TGgufFile.Open(P);
  try
    AssertEquals(6, F.KVCount);
    AssertTrue(F.KV[0].VType = gvUInt8);
    AssertTrue(F.KV[0].Count = 1);
    AssertTrue(F.KV[0].IntVal = 200);
    AssertTrue(F.KV[1].VType = gvInt8);
    AssertTrue(F.KV[1].IntVal = -5);
    AssertTrue(F.KV[2].VType = gvUInt16);
    AssertTrue(F.KV[2].IntVal = 65535);
    AssertTrue(F.KV[3].VType = gvInt16);
    AssertTrue(F.KV[3].IntVal = -32768);
    AssertTrue(F.KV[4].VType = gvUInt32);
    AssertTrue(F.KV[4].IntVal = 4294967295);
    AssertTrue(F.KV[5].VType = gvInt32);
    AssertTrue(F.KV[5].IntVal = -2147483648);
  finally
    F.Free;
  end;
end;
{ 002: uint64 $FFFFFFFFFFFFFFFF (stored as-is via typecast) and int64 -5. }
procedure TGgufFileTest.TestUT_GgufFile_002;
var
  S: TBytesStream;
  P: string;
  F: TGgufFile;
begin
  P := K_DIR + '/UT_GgufFile_002.gguf';
  S := TBytesStream.Create;
  try
    WHeader(S, 3, 0, 2);
    WVal(S, 'u64', 10, K_MAX_U64, 8);
    WVal(S, 'i64', 11, QWord(Int64(-5)), 8);
    S.SaveToFile(P);
  finally
    S.Free;
  end;
  F := TGgufFile.Open(P);
  try
    AssertEquals(2, F.KVCount);
    AssertTrue(F.KV[0].VType = gvUInt64);
    AssertTrue(F.KV[0].IntVal = -1);
    AssertTrue(F.KV[1].VType = gvInt64);
    AssertTrue(F.KV[1].IntVal = -5);
  finally
    F.Free;
  end;
end;
{ 003: float32 1.5f and float64 2.25 stored in FloatVal (exact binary values). }
procedure TGgufFileTest.TestUT_GgufFile_003;
var
  S: TBytesStream;
  P: string;
  F: TGgufFile;
begin
  P := K_DIR + '/UT_GgufFile_003.gguf';
  S := TBytesStream.Create;
  try
    WHeader(S, 3, 0, 2);
    WValF32(S, 'f32', 1.5);
    WValF64(S, 'f64', 2.25);
    S.SaveToFile(P);
  finally
    S.Free;
  end;
  F := TGgufFile.Open(P);
  try
    AssertEquals(2, F.KVCount);
    AssertTrue(F.KV[0].VType = gvFloat32);
    AssertTrue(F.KV[0].FloatVal = 1.5);
    AssertTrue(F.KV[1].VType = gvFloat64);
    AssertTrue(F.KV[1].FloatVal = 2.25);
  finally
    F.Free;
  end;
end;
{ 004: bool with 0 and 1. }
procedure TGgufFileTest.TestUT_GgufFile_004;
var
  S: TBytesStream;
  P: string;
  F: TGgufFile;
begin
  P := K_DIR + '/UT_GgufFile_004.gguf';
  S := TBytesStream.Create;
  try
    WHeader(S, 3, 0, 2);
    WVal(S, 'b0', 7, 0, 1);
    WVal(S, 'b1', 7, 1, 1);
    S.SaveToFile(P);
  finally
    S.Free;
  end;
  F := TGgufFile.Open(P);
  try
    AssertEquals(2, F.KVCount);
    AssertTrue(F.KV[0].VType = gvBool);
    AssertTrue(F.KV[0].IntVal = 0);
    AssertTrue(F.KV[1].VType = gvBool);
    AssertTrue(F.KV[1].IntVal = 1);
  finally
    F.Free;
  end;
end;
{ 005: string "hello", empty string, and 23.5 "héllo" (byte-for-byte, never converted). }
procedure TGgufFileTest.TestUT_GgufFile_005;
var
  S: TBytesStream;
  P: string;
  F: TGgufFile;
begin
  P := K_DIR + '/UT_GgufFile_005.gguf';
  S := TBytesStream.Create;
  try
    WHeader(S, 3, 0, 3);
    WValStr(S, 's', 'hello');
    WValStr(S, 'e', '');
    WValStrRaw(S, 'u', K_HELLO_UTF8);
    S.SaveToFile(P);
  finally
    S.Free;
  end;
  F := TGgufFile.Open(P);
  try
    AssertEquals(3, F.KVCount);
    AssertTrue(F.KV[0].VType = gvString);
    AssertTrue(F.KV[0].Count = 1);
    AssertTrue(F.KV[0].StrVal = 'hello');
    AssertTrue(F.KV[1].VType = gvString);
    AssertTrue(F.KV[1].StrVal = '');
    AssertTrue(Length(F.KV[1].StrVal) = 0);
    AssertTrue(F.KV[2].VType = gvString);
    AssertTrue(F.KV[2].StrVal = BytesToRBS(K_HELLO_UTF8));
  finally
    F.Free;
  end;
end;
{ 006: arrays of all integer types and bool, 2-3 elements each, exact values and order. }
procedure TGgufFileTest.TestUT_GgufFile_006;
var
  S: TBytesStream;
  P: string;
  F: TGgufFile;
begin
  P := K_DIR + '/UT_GgufFile_006.gguf';
  S := TBytesStream.Create;
  try
    WHeader(S, 3, 0, 9);
    WArrInt(S, 'a_u8', 0, [1, 255]);
    WArrInt(S, 'a_i8', 1, [-1, 127]);
    WArrInt(S, 'a_u16', 2, [65535]);
    WArrInt(S, 'a_i16', 3, [-32768, 0, 32767]);
    WArrInt(S, 'a_u32', 4, [4294967295]);
    WArrInt(S, 'a_i32', 5, [-2147483648]);
    WArrInt(S, 'a_u64', 10, [-1, 9223372036854775807]);
    WArrInt(S, 'a_i64', 11, [-5]);
    WArrInt(S, 'a_b', 7, [1, 0]);
    S.SaveToFile(P);
  finally
    S.Free;
  end;
  F := TGgufFile.Open(P);
  try
    AssertEquals(9, F.KVCount);
    AssertTrue(F.KV[0].VType = gvArray);
    AssertTrue(F.KV[0].ArrType = gvUInt8);
    AssertTrue(F.KV[0].Count = 2);
    AssertTrue(F.KV[0].Ints[0] = 1);
    AssertTrue(F.KV[0].Ints[1] = 255);
    AssertTrue(F.KV[1].ArrType = gvInt8);
    AssertTrue(F.KV[1].Ints[0] = -1);
    AssertTrue(F.KV[1].Ints[1] = 127);
    AssertTrue(F.KV[2].ArrType = gvUInt16);
    AssertTrue(F.KV[2].Count = 1);
    AssertTrue(F.KV[2].Ints[0] = 65535);
    AssertTrue(F.KV[3].ArrType = gvInt16);
    AssertTrue(F.KV[3].Count = 3);
    AssertTrue(F.KV[3].Ints[0] = -32768);
    AssertTrue(F.KV[3].Ints[1] = 0);
    AssertTrue(F.KV[3].Ints[2] = 32767);
    AssertTrue(F.KV[4].ArrType = gvUInt32);
    AssertTrue(F.KV[4].Ints[0] = 4294967295);
    AssertTrue(F.KV[5].ArrType = gvInt32);
    AssertTrue(F.KV[5].Ints[0] = -2147483648);
    AssertTrue(F.KV[6].ArrType = gvUInt64);
    AssertTrue(F.KV[6].Ints[0] = -1);
    AssertTrue(F.KV[6].Ints[1] = 9223372036854775807);
    AssertTrue(F.KV[7].ArrType = gvInt64);
    AssertTrue(F.KV[7].Ints[0] = -5);
    AssertTrue(F.KV[8].ArrType = gvBool);
    AssertTrue(F.KV[8].Ints[0] = 1);
    AssertTrue(F.KV[8].Ints[1] = 0);
  finally
    F.Free;
  end;
end;
{ 007: array of float32 [1.5, -0.25, 2.5] and array of float64 [2.25, -3.5], exact. }
procedure TGgufFileTest.TestUT_GgufFile_007;
var
  S: TBytesStream;
  P: string;
  F: TGgufFile;
begin
  P := K_DIR + '/UT_GgufFile_007.gguf';
  S := TBytesStream.Create;
  try
    WHeader(S, 3, 0, 2);
    WArrF32(S, 'a32', [1.5, -0.25, 2.5]);
    WArrF64(S, 'a64', [2.25, -3.5]);
    S.SaveToFile(P);
  finally
    S.Free;
  end;
  F := TGgufFile.Open(P);
  try
    AssertEquals(2, F.KVCount);
    AssertTrue(F.KV[0].VType = gvArray);
    AssertTrue(F.KV[0].ArrType = gvFloat32);
    AssertTrue(F.KV[0].Count = 3);
    AssertTrue(F.KV[0].Floats[0] = 1.5);
    AssertTrue(F.KV[0].Floats[1] = -0.25);
    AssertTrue(F.KV[0].Floats[2] = 2.5);
    AssertTrue(F.KV[1].ArrType = gvFloat64);
    AssertTrue(F.KV[1].Count = 2);
    AssertTrue(F.KV[1].Floats[0] = 2.25);
    AssertTrue(F.KV[1].Floats[1] = -3.5);
  finally
    F.Free;
  end;
end;
{ 008: array of strings ['alpha', '', 'beta']. }
procedure TGgufFileTest.TestUT_GgufFile_008;
var
  S: TBytesStream;
  P: string;
  F: TGgufFile;
begin
  P := K_DIR + '/UT_GgufFile_008.gguf';
  S := TBytesStream.Create;
  try
    WHeader(S, 3, 0, 1);
    WArrStr(S, 'ast', ['alpha', '', 'beta']);
    S.SaveToFile(P);
  finally
    S.Free;
  end;
  F := TGgufFile.Open(P);
  try
    AssertEquals(1, F.KVCount);
    AssertTrue(F.KV[0].VType = gvArray);
    AssertTrue(F.KV[0].ArrType = gvString);
    AssertTrue(F.KV[0].Count = 3);
    AssertTrue(F.KV[0].Strs[0] = 'alpha');
    AssertTrue(F.KV[0].Strs[1] = '');
    AssertTrue(F.KV[0].Strs[2] = 'beta');
  finally
    F.Free;
  end;
end;
{ 009: empty array: count 0, no elements. }
procedure TGgufFileTest.TestUT_GgufFile_009;
var
  S: TBytesStream;
  P: string;
  F: TGgufFile;
  E: array of Int64;
begin
  P := K_DIR + '/UT_GgufFile_009.gguf';
  S := TBytesStream.Create;
  SetLength(E, 0);
  try
    WHeader(S, 3, 0, 1);
    WArrInt(S, 'ae', 0, E);
    S.SaveToFile(P);
  finally
    S.Free;
  end;
  F := TGgufFile.Open(P);
  try
    AssertEquals(1, F.KVCount);
    AssertTrue(F.KV[0].VType = gvArray);
    AssertTrue(F.KV[0].ArrType = gvUInt8);
    AssertTrue(F.KV[0].Count = 0);
    AssertTrue(Length(F.KV[0].Ints) = 0);
  finally
    F.Free;
  end;
end;
{ 010: unknown value type 13 -> EGgufError naming key, type and position. }
procedure TGgufFileTest.TestUT_GgufFile_010;
var
  S: TBytesStream;
  P: string;
begin
  P := K_DIR + '/UT_GgufFile_010.gguf';
  S := TBytesStream.Create;
  try
    WHeader(S, 3, 0, 1);
    WVal(S, 'badtype', 13, 255, 1);
    S.SaveToFile(P);
  finally
    S.Free;
  end;
  try
    TGgufFile.Open(P).Free;
    raise Exception.Create('UT-GgufFile-010: EGgufError expected (unknown value type 13)');
  except
    on E: EGgufError do
    begin
      AssertTrue(Pos('badtype', E.Message) > 0);
      AssertTrue(Pos('position', E.Message) > 0);
    end;
  end;
end;
{ 011: array element type is array itself (9) -> EGgufError naming key and type. }
procedure TGgufFileTest.TestUT_GgufFile_011;
var
  S: TBytesStream;
  P: string;
begin
  P := K_DIR + '/UT_GgufFile_011.gguf';
  S := TBytesStream.Create;
  try
    WHeader(S, 3, 0, 1);
    WArrBad(S, 'aa', 9, 1);
    S.SaveToFile(P);
  finally
    S.Free;
  end;
  try
    TGgufFile.Open(P).Free;
    raise Exception.Create('UT-GgufFile-011: EGgufError expected (array element type "array")');
  except
    on E: EGgufError do
    begin
      AssertTrue(Pos('aa', E.Message) > 0);
      AssertTrue(Pos('position', E.Message) > 0);
    end;
  end;
end;
{ 012: key lookup is literal, case-sensitive: "MODEL" found, "model" not. }
procedure TGgufFileTest.TestUT_GgufFile_012;
var
  S: TBytesStream;
  P: string;
  F: TGgufFile;
begin
  P := K_DIR + '/UT_GgufFile_012.gguf';
  S := TBytesStream.Create;
  try
    WHeader(S, 3, 0, 2);
    WVal(S, 'MODEL', 0, 1, 1);
    WVal(S, 'model.x', 0, 2, 1);
    S.SaveToFile(P);
  finally
    S.Free;
  end;
  F := TGgufFile.Open(P);
  try
    AssertTrue(F.KV[0].Key = 'MODEL');
    AssertEquals(0, F.FindKey('MODEL'));
    AssertEquals(-1, F.FindKey('model'));
    AssertEquals(-1, F.FindKey('ModeL'));
    AssertEquals(1, F.FindKey('model.x'));
  finally
    F.Free;
  end;
end;
{ 013: duplicate key -> index of the first occurrence; missing key -> -1; KV[i] intact. }
procedure TGgufFileTest.TestUT_GgufFile_013;
var
  S: TBytesStream;
  P: string;
  F: TGgufFile;
begin
  P := K_DIR + '/UT_GgufFile_013.gguf';
  S := TBytesStream.Create;
  try
    WHeader(S, 3, 0, 2);
    WVal(S, 'k', 0, 1, 1);
    WVal(S, 'k', 0, 2, 1);
    S.SaveToFile(P);
  finally
    S.Free;
  end;
  F := TGgufFile.Open(P);
  try
    AssertEquals(2, F.KVCount);
    AssertEquals(0, F.FindKey('k'));
    AssertEquals(-1, F.FindKey('nope'));
    AssertTrue(F.KV[0].IntVal = 1);
    AssertTrue(F.KV[1].IntVal = 2);
  finally
    F.Free;
  end;
end;
{ 015: bad magic -> EGgufError; nothing published. }
procedure TGgufFileTest.TestUT_GgufFile_015;
var
  S: TBytesStream;
  P: string;
  I: Integer;
begin
  P := K_DIR + '/UT_GgufFile_015.gguf';
  S := TBytesStream.Create;
  try
    for I := 1 to 4 do
      WB(S, 88); { "XXXX" }
    WU32(S, 2);
    WU64(S, 0);
    WU64(S, 0);
    S.SaveToFile(P);
  finally
    S.Free;
  end;
  try
    TGgufFile.Open(P).Free;
    raise Exception.Create('UT-GgufFile-015: EGgufError expected (bad magic "XXXX")');
  except
    on E: EGgufError do
    begin
      AssertTrue(Pos('position 0', E.Message) > 0);
      AssertTrue(Pos('GGUF', E.Message) > 0);
    end;
  end;
end;
{ 016: version 1 and version 4 -> EGgufError naming the version. }
procedure TGgufFileTest.TestUT_GgufFile_016;
var
  S: TBytesStream;
  P: string;
begin
  P := K_DIR + '/UT_GgufFile_016_v1.gguf';
  S := TBytesStream.Create;
  try
    WHeader(S, 1, 0, 0);
    S.SaveToFile(P);
  finally
    S.Free;
  end;
  try
    TGgufFile.Open(P).Free;
    raise Exception.Create('UT-GgufFile-016: EGgufError expected (version 1)');
  except
    on E: EGgufError do
      AssertTrue(Pos('version 1', E.Message) > 0);
  end;
  S := TBytesStream.Create;
  try
    WHeader(S, 4, 0, 0);
    S.SaveToFile(P + '.4');
  finally
    S.Free;
  end;
  try
    TGgufFile.Open(P + '.4').Free;
    raise Exception.Create('UT-GgufFile-016: EGgufError expected (version 4)');
  except
    on E: EGgufError do
      AssertTrue(Pos('version 4', E.Message) > 0);
  end;
end;
{ 017: version 2 and version 3 both fine (a rehearsal file is version 3). }
procedure TGgufFileTest.TestUT_GgufFile_017;
var
  S: TBytesStream;
  P: string;
  F: TGgufFile;
begin
  P := K_DIR + '/UT_GgufFile_017_v2.gguf';
  S := TBytesStream.Create;
  try
    WHeader(S, 2, 0, 0);
    S.SaveToFile(P);
  finally
    S.Free;
  end;
  F := TGgufFile.Open(P);
  try
    AssertEquals(2, F.Version);
    AssertEquals(0, F.KVCount);
  finally
    F.Free;
  end;
  S := TBytesStream.Create;
  try
    WHeader(S, 3, 0, 0);
    S.SaveToFile(P + '.3');
  finally
    S.Free;
  end;
  F := TGgufFile.Open(P + '.3');
  try
    AssertEquals(3, F.Version);
    AssertEquals(0, F.KVCount);
  finally
    F.Free;
  end;
end;
{ 018: empty file -> EGgufError. }
procedure TGgufFileTest.TestUT_GgufFile_018;
var
  S: TBytesStream;
  P: string;
begin
  P := K_DIR + '/UT_GgufFile_018.gguf';
  S := TBytesStream.Create;
  try
    S.SaveToFile(P); { 0 bytes }
  finally
    S.Free;
  end;
  try
    TGgufFile.Open(P).Free;
    raise Exception.Create('UT-GgufFile-018: EGgufError expected (empty file)');
  except
    on E: EGgufError do
      AssertTrue(Length(E.Message) > 0);
  end;
end;
{ 019: file that cuts inside the header -> EGgufError. }
procedure TGgufFileTest.TestUT_GgufFile_019;
var
  S: TBytesStream;
  P: string;
begin
  P := K_DIR + '/UT_GgufFile_019.gguf';
  S := TBytesStream.Create;
  try
    WHeader(S, 3, 0, 0);
    S.Size := 20; { cut 4 bytes into the header (header is 24) }
    S.SaveToFile(P);
  finally
    S.Free;
  end;
  try
    TGgufFile.Open(P).Free;
    raise Exception.Create('UT-GgufFile-019: EGgufError expected (header cut)');
  except
    on E: EGgufError do
      AssertTrue(Length(E.Message) > 0);
  end;
end;
{ 020: file that cuts inside a key string -> EGgufError. }
procedure TGgufFileTest.TestUT_GgufFile_020;
var
  S: TBytesStream;
  P: string;
begin
  P := K_DIR + '/UT_GgufFile_020.gguf';
  S := TBytesStream.Create;
  try
    WHeader(S, 3, 0, 2);
    WVal(S, 'key1', 0, 1, 1);
    WStr(S, 'key2'); { key 2: length + bytes are written, nothing more:
                        the file ends inside the second key string area and
                        reading its value must fail. }
    S.SaveToFile(P);
  finally
    S.Free;
  end;
  try
    TGgufFile.Open(P).Free;
    raise Exception.Create('UT-GgufFile-020: EGgufError expected (key string cut)');
  except
    on E: EGgufError do
      AssertTrue(Length(E.Message) > 0);
  end;
end;
{ 023: Path and FileSize mirror the file that was opened. }
procedure TGgufFileTest.TestUT_GgufFile_023;
var
  S: TBytesStream;
  P: string;
  F: TGgufFile;
  ASize: Integer;
begin
  P := K_DIR + '/UT_GgufFile_023.gguf';
  S := TBytesStream.Create;
  try
    WHeader(S, 3, 0, 1);
    WVal(S, 'a', 0, 7, 1);
    S.SaveToFile(P);
    ASize := S.Size;
  finally
    S.Free;
  end;
  AssertEquals(24 + (8 + 1) + 4 + 1, ASize); { header + key + u32 + u8: exact }
  F := TGgufFile.Open(P);
  try
    AssertTrue(F.Path = P);
    AssertEquals(QWord(ASize), F.FileSize);
  finally
    F.Free;
  end;
end;
{ 024: general.alignment drives the rule: 16 and 4096; wrong type, zero and non-power-of-two -> EGgufError. }
procedure TGgufFileTest.TestUT_GgufFile_024;
var
  S: TBytesStream;
  P: string;
  F: TGgufFile;
begin
  P := K_DIR + '/UT_GgufFile_024_a16.gguf';
  S := TBytesStream.Create;
  try
    WHeader(S, 3, 0, 1);
    WVal(S, 'general.alignment', 4, 16, 4);
    S.SaveToFile(P);
  finally
    S.Free;
  end;
  F := TGgufFile.Open(P);
  try
    AssertEquals(16, F.Alignment);
  finally
    F.Free;
  end;
  S := TBytesStream.Create;
  try
    WHeader(S, 3, 0, 1);
    WVal(S, 'general.alignment', 4, 4096, 4);
    S.SaveToFile(P + '.4096');
  finally
    S.Free;
  end;
  F := TGgufFile.Open(P + '.4096');
  try
    AssertEquals(4096, F.Alignment);
  finally
    F.Free;
  end;
  S := TBytesStream.Create;
  try
    WHeader(S, 3, 0, 1);
    WVal(S, 'general.alignment', 4, 0, 4);
    S.SaveToFile(P + '.0');
  finally
    S.Free;
  end;
  try
    TGgufFile.Open(P + '.0').Free;
    raise Exception.Create('UT-GgufFile-024: EGgufError expected (alignment 0)');
  except
    on E: EGgufError do
      AssertTrue(Pos('general.alignment', E.Message) > 0);
  end;
  S := TBytesStream.Create;
  try
    WHeader(S, 3, 0, 1);
    WVal(S, 'general.alignment', 4, 48, 4);
    S.SaveToFile(P + '.48');
  finally
    S.Free;
  end;
  try
    TGgufFile.Open(P + '.48').Free;
    raise Exception.Create('UT-GgufFile-024: EGgufError expected (alignment 48)');
  except
    on E: EGgufError do
      AssertTrue(Pos('general.alignment', E.Message) > 0);
  end;
  S := TBytesStream.Create;
  try
    WHeader(S, 3, 0, 1);
    WVal(S, 'general.alignment', 5, 16, 4); { int32, not uint32 }
    S.SaveToFile(P + '.i32');
  finally
    S.Free;
  end;
  try
    TGgufFile.Open(P + '.i32').Free;
    raise Exception.Create('UT-GgufFile-024: EGgufError expected (alignment int32)');
  except
    on E: EGgufError do
      AssertTrue(Pos('general.alignment', E.Message) > 0);
  end;
end;
{ 025: general.alignment not a power of two (48) -> EGgufError. }
procedure TGgufFileTest.TestUT_GgufFile_025;
var
  S: TBytesStream;
  P: string;
begin
  P := K_DIR + '/UT_GgufFile_025.gguf';
  S := TBytesStream.Create;
  try
    WHeader(S, 3, 0, 1);
    WVal(S, 'general.alignment', 4, 48, 4);
    S.SaveToFile(P);
  finally
    S.Free;
  end;
  try
    TGgufFile.Open(P).Free;
    raise Exception.Create('UT-GgufFile-025: EGgufError expected (alignment 48)');
  except
    on E: EGgufError do
      AssertTrue(Pos('general.alignment', E.Message) > 0);
  end;
end;
{ 041: Open with an absolute path -> Path = the string as given; FileSize correct. }
procedure TGgufFileTest.TestUT_GgufFile_041;
var
  S: TBytesStream;
  P: string;
  F: TGgufFile;
begin
  { absolute: the current directory plus the relative test-directory name }
  P := GetCurrentDir + '/' + K_DIR + '/UT_GgufFile_041.gguf';
  S := TBytesStream.Create;
  try
    WHeader(S, 3, 0, 1);
    WVal(S, 'n', 2, 12345, 2);
    S.SaveToFile(P);
  finally
    S.Free;
  end;
  AssertTrue(FileExists(P));
  F := TGgufFile.Open(P);
  try
    AssertEquals(P, F.Path); { byte for byte, as given }
    AssertEquals(24 + (8 + 1) + 4 + 2, F.FileSize); { header + key 'n' + u16 }
    AssertEquals(12345, LongInt(F.KV[0].IntVal));
  finally
    F.Free;
  end;
end;
{ 042: chdir into the parent dir and Open with the bare name: OK,
  Path = the relative string as given, values identical to the absolute case. }
procedure TGgufFileTest.TestUT_GgufFile_042;
var
  S: TBytesStream;
  P, OldDir, Bare: string;
  FA, FR: TGgufFile;
begin
  P := GetCurrentDir + '/' + K_DIR + '/UT_GgufFile_042.gguf';
  S := TBytesStream.Create;
  try
    WHeader(S, 3, 0, 1);
    WVal(S, 'n', 2, 12345, 2);
    S.SaveToFile(P);
  finally
    S.Free;
  end;
  FA := TGgufFile.Open(P); { the absolute case }
  try
    OldDir := GetCurrentDir;
    SetCurrentDir(K_DIR); { chdir into the parent directory }
    try
      Bare := 'UT_GgufFile_042.gguf';
      FR := TGgufFile.Open(Bare);
      try
        AssertEquals(Bare, FR.Path); { the relative string as given }
        AssertEquals(FA.FileSize, FR.FileSize);
        AssertEquals(FA.Version, FR.Version);
        AssertEquals(FA.Alignment, FR.Alignment);
        AssertEquals(FA.KVCount, FR.KVCount);
        AssertEquals(FA.KV[0].IntVal, FR.KV[0].IntVal);
        AssertTrue(FA.KV[0].Key = FR.KV[0].Key);
      finally
        FR.Free;
      end;
    finally
      SetCurrentDir(OldDir); { back, for the other tests }
    end;
  finally
    FA.Free;
  end;
end;
{ 043: KVStart always 24; KVEnd = 24 + the exact bytes of both entries. }
procedure TGgufFileTest.TestUT_GgufFile_043;
var
  S: TBytesStream;
  P: string;
  F: TGgufFile;
begin
  { entry 1: key "ab" (8+2) + uint8 (4+1)      = 15 bytes
    entry 2: key "xyz" (8+3) + string "hi" (4 + 8+2) = 25 bytes
    KVEnd = 24 + 15 + 25 = 64 }
  P := K_DIR + '/UT_GgufFile_043.gguf';
  S := TBytesStream.Create;
  try
    WHeader(S, 3, 0, 2);
    WVal(S, 'ab', 0, 1, 1);
    WValStr(S, 'xyz', 'hi');
    S.SaveToFile(P);
  finally
    S.Free;
  end;
  F := TGgufFile.Open(P);
  try
    AssertEquals(24, F.KVStart);
    AssertEquals(64, F.KVEnd);
    AssertEquals(64, F.FileSize); { no tensors: the file ends at KVEnd }
  finally
    F.Free;
  end;
end;
{ 045: GgufValueTypeName names every type, ordinals 0..12, wire order. }
procedure TGgufFileTest.TestUT_GgufFile_045;
const
  Names: array[0..12] of string = (
    'uint8', 'int8', 'uint16', 'int16', 'uint32', 'int32',
    'float32', 'bool', 'string', 'array', 'uint64', 'int64', 'float64');
var
  I: Byte;
begin
  for I := 0 to 12 do
    AssertEquals(Names[I], GgufValueTypeName(TGgufValueType(I)));
end;

{ 014: FindTensor exact, case-sensitive; absent; 'wt' and 'WT' distinguishable. }
procedure TGgufFileTest.TestUT_GgufFile_014;
var
  S: TBytesStream;
  P: string;
  F: TGgufFile;
  A, B: TBytes;
begin
  P := K_DIR + '/UT_GgufFile_014.gguf';
  S := TBytesStream.Create;
  try
    WHeader(S, 3, 2, 0);
    WTensor(S, 'wt', 2, [4, 3], 0, 0);
    WTensor(S, 'WT', 2, [4, 3], 0, 64); { 48 bytes rounded up to 32 -> 64 }
    PadTo(S, 128); { 24 + 42 + 42 = 108 after the directory; 108 -> 128 }
    A := MakePat(48, 1);
    S.WriteBuffer(A[0], 48);
    PadTo(S, 192); { data-section padding 48 -> 64, where "WT" lives }
    B := MakePat(48, 100);
    S.WriteBuffer(B[0], 48);
    S.SaveToFile(P);
  finally
    S.Free;
  end;
  F := TGgufFile.Open(P);
  try
    AssertEquals(2, F.TensorCount);
    AssertEquals(0, F.FindTensor('wt'));
    AssertEquals(1, F.FindTensor('WT'));
    AssertEquals(-1, F.FindTensor('nope'));
    AssertEquals(-1, F.FindTensor('wT'));
  finally
    F.Free;
  end;
end;
{ 021: general.alignment absent -> 32; the directory end (65) is not a multiple of 32. }
procedure TGgufFileTest.TestUT_GgufFile_021;
var
  S: TBytesStream;
  P: string;
  F: TGgufFile;
  A: TBytes;
begin
  P := K_DIR + '/UT_GgufFile_021.gguf';
  S := TBytesStream.Create;
  try
    WHeader(S, 3, 1, 0);
    WTensor(S, 't', 2, [4, 3], 0, 0);
    PadTo(S, 96); { 24 + 41 = 65 after the directory; 65 rounded up to 32 -> 96 }
    A := MakePat(48, 7);
    S.WriteBuffer(A[0], 48);
    S.SaveToFile(P);
  finally
    S.Free;
  end;
  F := TGgufFile.Open(P);
  try
    AssertEquals(32, F.Alignment);
    AssertEquals(96, F.DataOffset);
    AssertEquals(1, F.TensorCount);
    AssertTrue(F.Tensor[0].Name = 't');
    AssertEquals(2, F.Tensor[0].NDims);
    AssertEquals(4, F.Tensor[0].Dims[0]);
    AssertEquals(3, F.Tensor[0].Dims[1]);
    AssertEquals(0, F.Tensor[0].TType);
    AssertEquals(0, F.Tensor[0].Offset);
    AssertEquals(48, F.Tensor[0].NBytes);
  finally
    F.Free;
  end;
end;
{ 022: general.alignment = 64 present -> used; the directory end (100) lies strictly between two multiples of 64. }
procedure TGgufFileTest.TestUT_GgufFile_022;
var
  S: TBytesStream;
  P: string;
  F: TGgufFile;
  A: TBytes;
begin
  P := K_DIR + '/UT_GgufFile_022.gguf';
  S := TBytesStream.Create;
  try
    WHeader(S, 3, 1, 1);
    WVal(S, 'general.alignment', 4, 64, 4);
    WTensor(S, 't', 2, [4, 3], 0, 0);
    PadTo(S, 128); { 24 + 35 + 41 = 100 after the directory; 100 -> 128 (multiple of 64) }
    A := MakePat(48, 50);
    S.WriteBuffer(A[0], 48);
    S.SaveToFile(P);
  finally
    S.Free;
  end;
  F := TGgufFile.Open(P);
  try
    AssertEquals(64, F.Alignment);
    AssertEquals(128, F.DataOffset);
    AssertEquals(1, F.TensorCount);
    AssertEquals(48, F.Tensor[0].NBytes);
    AssertEquals(0, F.Tensor[0].Offset);
  finally
    F.Free;
  end;
end;
{ 026: an f32 tensor with dims [4, 3] opens with NBytes = 48. }
procedure TGgufFileTest.TestUT_GgufFile_026;
var
  S: TBytesStream;
  P: string;
  F: TGgufFile;
  A: TBytes;
begin
  P := K_DIR + '/UT_GgufFile_026.gguf';
  S := TBytesStream.Create;
  try
    WHeader(S, 3, 1, 0);
    WTensor(S, 'f32', 2, [4, 3], 0, 0);
    PadTo(S, 96); { 67 -> 96 }
    A := MakePat(48, 10);
    S.WriteBuffer(A[0], 48);
    S.SaveToFile(P);
  finally
    S.Free;
  end;
  F := TGgufFile.Open(P);
  try
    AssertEquals(1, F.TensorCount);
    AssertEquals(48, F.Tensor[0].NBytes);
    AssertEquals(0, F.Tensor[0].TType);
    AssertTrue(F.Tensor[0].Name = 'f32');
    AssertEquals(0, F.Tensor[0].Offset);
    AssertEquals(96, F.DataOffset);
  finally
    F.Free;
  end;
end;
{ 027: an f16 tensor with dims [6, 4] -> NBytes = 48 (block 1, 2 bytes per element). }
procedure TGgufFileTest.TestUT_GgufFile_027;
var
  S: TBytesStream;
  P: string;
  F: TGgufFile;
  A: TBytes;
begin
  P := K_DIR + '/UT_GgufFile_027.gguf';
  S := TBytesStream.Create;
  try
    WHeader(S, 3, 1, 0);
    WTensor(S, 'f16', 2, [6, 4], 1, 0);
    PadTo(S, 96); { 67 -> 96 }
    A := MakePat(48, 20);
    S.WriteBuffer(A[0], 48);
    S.SaveToFile(P);
  finally
    S.Free;
  end;
  F := TGgufFile.Open(P);
  try
    AssertEquals(1, F.TensorCount);
    AssertEquals(48, F.Tensor[0].NBytes);
    AssertEquals(1, F.Tensor[0].TType);
    AssertEquals(6, F.Tensor[0].Dims[0]);
    AssertEquals(4, F.Tensor[0].Dims[1]);
  finally
    F.Free;
  end;
end;
{ 028: a q8_0 tensor with dims [32] -> NBytes = 34. }
procedure TGgufFileTest.TestUT_GgufFile_028;
var
  S: TBytesStream;
  P: string;
  F: TGgufFile;
  A: TBytes;
begin
  P := K_DIR + '/UT_GgufFile_028.gguf';
  S := TBytesStream.Create;
  try
    WHeader(S, 3, 1, 0);
    WTensor(S, 'q80', 1, [32], 8, 0);
    PadTo(S, 64); { 59 -> 64 }
    A := MakePat(34, 30);
    S.WriteBuffer(A[0], 34);
    S.SaveToFile(P);
  finally
    S.Free;
  end;
  F := TGgufFile.Open(P);
  try
    AssertEquals(1, F.TensorCount);
    AssertEquals(34, F.Tensor[0].NBytes);
    AssertEquals(8, F.Tensor[0].TType);
    AssertEquals(1, F.Tensor[0].NDims);
    AssertEquals(32, F.Tensor[0].Dims[0]);
  finally
    F.Free;
  end;
end;

{ 029: a q4_K tensor with dims [256, 2] -> NBytes = 288 (2 blocks of 144). }
procedure TGgufFileTest.TestUT_GgufFile_029;
var
  S: TBytesStream;
  P: string;
  F: TGgufFile;
  A: TBytes;
begin
  P := K_DIR + '/UT_GgufFile_029.gguf';
  S := TBytesStream.Create;
  try
    WHeader(S, 3, 1, 0);
    WTensor(S, 'q4K', 2, [256, 2], 12, 0);
    PadTo(S, 96); { 67 -> 96 }
    A := MakePat(288, 40);
    S.WriteBuffer(A[0], 288);
    S.SaveToFile(P);
  finally
    S.Free;
  end;
  F := TGgufFile.Open(P);
  try
    AssertEquals(1, F.TensorCount);
    AssertEquals(288, F.Tensor[0].NBytes);
    AssertEquals(12, F.Tensor[0].TType);
    AssertEquals(2, F.Tensor[0].NDims);
    AssertEquals(256, F.Tensor[0].Dims[0]);
    AssertEquals(2, F.Tensor[0].Dims[1]);
  finally
    F.Free;
  end;
end;
{ 030: a q5_K tensor with dims [256] -> NBytes = 176. }
procedure TGgufFileTest.TestUT_GgufFile_030;
var
  S: TBytesStream;
  P: string;
  F: TGgufFile;
  A: TBytes;
begin
  P := K_DIR + '/UT_GgufFile_030.gguf';
  S := TBytesStream.Create;
  try
    WHeader(S, 3, 1, 0);
    WTensor(S, 'q5K', 1, [256], 13, 0);
    PadTo(S, 64); { 59 -> 64 }
    A := MakePat(176, 60);
    S.WriteBuffer(A[0], 176);
    S.SaveToFile(P);
  finally
    S.Free;
  end;
  F := TGgufFile.Open(P);
  try
    AssertEquals(1, F.TensorCount);
    AssertEquals(176, F.Tensor[0].NBytes);
    AssertEquals(13, F.Tensor[0].TType);
  finally
    F.Free;
  end;
end;
{ 031: a q6_K tensor with dims [256, 3, 2] -> NBytes = 1260 (6 blocks of 210). }
procedure TGgufFileTest.TestUT_GgufFile_031;
var
  S: TBytesStream;
  P: string;
  F: TGgufFile;
  A: TBytes;
begin
  P := K_DIR + '/UT_GgufFile_031.gguf';
  S := TBytesStream.Create;
  try
    WHeader(S, 3, 1, 0);
    WTensor(S, 'q6K', 3, [256, 3, 2], 14, 0);
    PadTo(S, 96); { 75 -> 96 }
    A := MakePat(1260, 80);
    S.WriteBuffer(A[0], 1260);
    S.SaveToFile(P);
  finally
    S.Free;
  end;
  F := TGgufFile.Open(P);
  try
    AssertEquals(1, F.TensorCount);
    AssertEquals(1260, F.Tensor[0].NBytes);
    AssertEquals(14, F.Tensor[0].TType);
    AssertEquals(3, F.Tensor[0].NDims);
    AssertEquals(256, F.Tensor[0].Dims[0]);
    AssertEquals(3, F.Tensor[0].Dims[1]);
    AssertEquals(2, F.Tensor[0].Dims[2]);
  finally
    F.Free;
  end;
end;
{ 032: a bf16 tensor with dims [8] -> NBytes = 16 (block 1, 2 bytes per element). }
procedure TGgufFileTest.TestUT_GgufFile_032;
var
  S: TBytesStream;
  P: string;
  F: TGgufFile;
  A: TBytes;
begin
  P := K_DIR + '/UT_GgufFile_032.gguf';
  S := TBytesStream.Create;
  try
    WHeader(S, 3, 1, 0);
    WTensor(S, 'bf', 1, [8], 30, 0);
    PadTo(S, 64); { 58 -> 64 }
    A := MakePat(16, 90);
    S.WriteBuffer(A[0], 16);
    S.SaveToFile(P);
  finally
    S.Free;
  end;
  F := TGgufFile.Open(P);
  try
    AssertEquals(1, F.TensorCount);
    AssertEquals(16, F.Tensor[0].NBytes);
    AssertEquals(30, F.Tensor[0].TType);
    AssertTrue(F.Tensor[0].Name = 'bf');
  finally
    F.Free;
  end;
end;
{ 033: an unsupported tensor type (5) -> EGgufError "unsupported tensor type 5", the tensor named. }
procedure TGgufFileTest.TestUT_GgufFile_033;
var
  S: TBytesStream;
  P: string;
begin
  P := K_DIR + '/UT_GgufFile_033.gguf';
  S := TBytesStream.Create;
  try
    WHeader(S, 3, 1, 0);
    WTensor(S, 'bad', 1, [4], 5, 0);
    S.SaveToFile(P);
  finally
    S.Free;
  end;
  try
    TGgufFile.Open(P).Free;
    raise Exception.Create('UT-GgufFile-033: EGgufError expected (tensor type 5)');
  except
    on E: EGgufError do
    begin
      AssertTrue(Pos('unsupported tensor type 5', E.Message) > 0);
      AssertTrue(Pos('bad', E.Message) > 0);
    end;
  end;
end;
{ 034: a row length that is not a multiple of the block size: q8_0 [16, 1] and q4_K [128]; both name the tensor. }
procedure TGgufFileTest.TestUT_GgufFile_034;
var
  S: TBytesStream;
  P: string;
begin
  P := K_DIR + '/UT_GgufFile_034.gguf';
  S := TBytesStream.Create;
  try
    WHeader(S, 3, 1, 0);
    WTensor(S, 't8', 2, [16, 1], 8, 0);
    S.SaveToFile(P);
  finally
    S.Free;
  end;
  try
    TGgufFile.Open(P).Free;
    raise Exception.Create('UT-GgufFile-034: EGgufError expected (q8_0 row length 16)');
  except
    on E: EGgufError do
    begin
      AssertTrue(Pos('t8', E.Message) > 0);
      AssertTrue(Pos('row length', E.Message) > 0);
      AssertTrue(Pos('block size', E.Message) > 0);
    end;
  end;
  S := TBytesStream.Create;
  try
    WHeader(S, 3, 1, 0);
    WTensor(S, 'tK', 1, [128], 12, 0);
    S.SaveToFile(P + '.K');
  finally
    S.Free;
  end;
  try
    TGgufFile.Open(P + '.K').Free;
    raise Exception.Create('UT-GgufFile-034: EGgufError expected (q4_K row length 128)');
  except
    on E: EGgufError do
    begin
      AssertTrue(Pos('tK', E.Message) > 0);
      AssertTrue(Pos('row length', E.Message) > 0);
      AssertTrue(Pos('block size', E.Message) > 0);
    end;
  end;
end;
{ 035: n_dims = 0 rejected -> EGgufError (n_dims must be 1..4), the tensor named. }
procedure TGgufFileTest.TestUT_GgufFile_035;
var
  S: TBytesStream;
  P: string;
begin
  P := K_DIR + '/UT_GgufFile_035.gguf';
  S := TBytesStream.Create;
  try
    WHeader(S, 3, 1, 0);
    WTensor(S, 'n0', 0, [1], 0, 0);
    S.SaveToFile(P);
  finally
    S.Free;
  end;
  try
    TGgufFile.Open(P).Free;
    raise Exception.Create('UT-GgufFile-035: EGgufError expected (n_dims = 0)');
  except
    on E: EGgufError do
    begin
      AssertTrue(Pos('n0', E.Message) > 0);
      AssertTrue(Pos('n_dims', E.Message) > 0);
    end;
  end;
end;
{ 036: n_dims = 5 rejected -> EGgufError, the tensor named. }
procedure TGgufFileTest.TestUT_GgufFile_036;
var
  S: TBytesStream;
  P: string;
begin
  P := K_DIR + '/UT_GgufFile_036.gguf';
  S := TBytesStream.Create;
  try
    WHeader(S, 3, 1, 0);
    WTensor(S, 'n5', 5, [1, 1, 1, 1, 1], 0, 0);
    S.SaveToFile(P);
  finally
    S.Free;
  end;
  try
    TGgufFile.Open(P).Free;
    raise Exception.Create('UT-GgufFile-036: EGgufError expected (n_dims = 5)');
  except
    on E: EGgufError do
    begin
      AssertTrue(Pos('n5', E.Message) > 0);
      AssertTrue(Pos('n_dims', E.Message) > 0);
    end;
  end;
end;

{ 037: offsets correct with padding: f32 [4] (16 B) at 0, then f32 [8] (32 B) at 32 (alignment 32). }
procedure TGgufFileTest.TestUT_GgufFile_037;
var
  S: TBytesStream;
  P: string;
  F: TGgufFile;
  A, B: TBytes;
begin
  P := K_DIR + '/UT_GgufFile_037.gguf';
  S := TBytesStream.Create;
  try
    WHeader(S, 3, 2, 0);
    WTensor(S, 'a', 1, [4], 0, 0);
    WTensor(S, 'b', 1, [8], 0, 32); { 16 rounded up to 32 -> 32 }
    PadTo(S, 96); { 90 -> 96 }
    A := MakePat(16, 5);
    S.WriteBuffer(A[0], 16);
    PadTo(S, 128); { data-section padding 16 -> 32 }
    B := MakePat(32, 45);
    S.WriteBuffer(B[0], 32);
    S.SaveToFile(P);
  finally
    S.Free;
  end;
  F := TGgufFile.Open(P);
  try
    AssertEquals(2, F.TensorCount);
    AssertEquals(0, F.Tensor[0].Offset);
    AssertEquals(32, F.Tensor[1].Offset);
    AssertEquals(16, F.Tensor[0].NBytes);
    AssertEquals(32, F.Tensor[1].NBytes);
    AssertEquals(96, F.DataOffset);
  finally
    F.Free;
  end;
end;
{ 038: as 037 but the second tensor at the wrong offset 17 -> EGgufError naming the tensor and position. }
procedure TGgufFileTest.TestUT_GgufFile_038;
var
  S: TBytesStream;
  P: string;
  A, B: TBytes;
begin
  P := K_DIR + '/UT_GgufFile_038.gguf';
  S := TBytesStream.Create;
  try
    WHeader(S, 3, 2, 0);
    WTensor(S, 'a', 1, [4], 0, 0);
    WTensor(S, 'b', 1, [8], 0, 17); { wrong: 16 rounded up to 32 is 32, not 17 }
    PadTo(S, 96);
    A := MakePat(16, 5);
    S.WriteBuffer(A[0], 16);
    B := MakePat(32, 45);
    S.WriteBuffer(B[0], 32);
    S.SaveToFile(P);
  finally
    S.Free;
  end;
  try
    TGgufFile.Open(P).Free;
    raise Exception.Create('UT-GgufFile-038: EGgufError expected (offset 17)');
  except
    on E: EGgufError do
    begin
      AssertTrue(Pos('b', E.Message) > 0);
      AssertTrue(Pos('offset', E.Message) > 0);
      AssertTrue(Pos('position', E.Message) > 0);
    end;
  end;
end;
{ 039: a tensor past the end of the file: f32 [4] (16 B) declared, the file ends 10 bytes after the data section. }
procedure TGgufFileTest.TestUT_GgufFile_039;
var
  S: TBytesStream;
  P: string;
  A: TBytes;
begin
  P := K_DIR + '/UT_GgufFile_039.gguf';
  S := TBytesStream.Create;
  try
    WHeader(S, 3, 1, 0);
    WTensor(S, 'xt', 1, [4], 0, 0);
    PadTo(S, 64); { 58 -> 64 }
    A := MakePat(10, 99); { only 10 data bytes, 16 needed }
    S.WriteBuffer(A[0], 10);
    S.SaveToFile(P);
  finally
    S.Free;
  end;
  try
    TGgufFile.Open(P).Free;
    raise Exception.Create('UT-GgufFile-039: EGgufError expected (extent)');
  except
    on E: EGgufError do
    begin
      AssertTrue(Pos('xt', E.Message) > 0);
      AssertTrue(Pos('position', E.Message) > 0);
      AssertTrue(Pos('bytes', E.Message) > 0);
    end;
  end;
end;
{ 040: zero tensors: TensorCount = 0; DataOffset = the KV end rounded up (38 -> 64); no raise. }
procedure TGgufFileTest.TestUT_GgufFile_040;
var
  S: TBytesStream;
  P: string;
  F: TGgufFile;
begin
  P := K_DIR + '/UT_GgufFile_040.gguf';
  S := TBytesStream.Create;
  try
    WHeader(S, 3, 0, 1);
    WVal(S, 'k', 0, 1, 1);
    S.SaveToFile(P); { the file ends after the KV entries; no data section }
  finally
    S.Free;
  end;
  F := TGgufFile.Open(P);
  try
    AssertEquals(0, F.TensorCount);
    AssertEquals(1, F.KVCount);
    AssertEquals(64, F.DataOffset);
  finally
    F.Free;
  end;
end;
{ 044: ReadTensorBytes returns exactly the bytes written: two tensors with distinct patterns, each read twice. }
procedure TGgufFileTest.TestUT_GgufFile_044;
var
  S: TBytesStream;
  P: string;
  F: TGgufFile;
  A, B, R: TBytes;
  I: Integer;
begin
  P := K_DIR + '/UT_GgufFile_044.gguf';
  S := TBytesStream.Create;
  try
    WHeader(S, 3, 2, 0);
    WTensor(S, 'A', 2, [4, 3], 0, 0);
    WTensor(S, 'B', 2, [6, 4], 1, 64); { 48 rounded up to 32 -> 64 }
    PadTo(S, 128); { 106 -> 128 }
    A := MakePat(48, 10);
    S.WriteBuffer(A[0], 48);
    PadTo(S, 192); { data-section padding between the tensors }
    B := MakePat(48, 200);
    S.WriteBuffer(B[0], 48);
    S.SaveToFile(P);
  finally
    S.Free;
  end;
  F := TGgufFile.Open(P);
  try
    R := nil;
    F.ReadTensorBytes(0, R);
    AssertEquals(48, Length(R));
    for I := 0 to 47 do
      AssertEquals(Integer(A[I]), Integer(R[I]));
    F.ReadTensorBytes(1, R);
    AssertEquals(48, Length(R));
    for I := 0 to 47 do
      AssertEquals(Integer(B[I]), Integer(R[I]));
    F.ReadTensorBytes(0, R); { the second round: the stream repositioned again }
    for I := 0 to 47 do
      AssertEquals(Integer(A[I]), Integer(R[I]));
    F.ReadTensorBytes(1, R);
    for I := 0 to 47 do
      AssertEquals(Integer(B[I]), Integer(R[I]));
  finally
    F.Free;
  end;
end;
{ 046: GgufTypeName: the seven known ids and two unknown ones. }
procedure TGgufFileTest.TestUT_GgufFile_046;
begin
  AssertTrue(GgufTypeName(0) = 'f32');
  AssertTrue(GgufTypeName(1) = 'f16');
  AssertTrue(GgufTypeName(8) = 'q8_0');
  AssertTrue(GgufTypeName(12) = 'q4_K');
  AssertTrue(GgufTypeName(13) = 'q5_K');
  AssertTrue(GgufTypeName(14) = 'q6_K');
  AssertTrue(GgufTypeName(30) = 'bf16');
  AssertTrue(GgufTypeName(5) = 'type5');
  AssertTrue(GgufTypeName(999) = 'type999');
end;
{ 047: GgufTensorBytes: two valid calls, an unknown type, a bad row length. }
procedure TGgufFileTest.TestUT_GgufFile_047;
var
  D: array of QWord;
begin
  SetLength(D, 1);
  D[0] := 1;
  AssertEquals(QWord(4), GgufTensorBytes(0, D));
  D[0] := 32;
  AssertEquals(QWord(34), GgufTensorBytes(8, D));
  D[0] := 4;
  try
    GgufTensorBytes(5, D);
    raise Exception.Create('UT-GgufFile-047: EGgufError expected (type 5)');
  except
    on E: EGgufError do
      AssertTrue(Pos('unsupported tensor type 5', E.Message) > 0);
  end;
  D[0] := 128;
  try
    GgufTensorBytes(12, D);
    raise Exception.Create('UT-GgufFile-047: EGgufError expected (row length 128)');
  except
    on E: EGgufError do
    begin
      AssertTrue(Pos('row length', E.Message) > 0);
      AssertTrue(Pos('block size', E.Message) > 0);
    end;
  end;
end;
{ 049: the last tensor ends one byte past the end of the file: f32 [4] (16 B)
  at offset 0, only 15 of the 16 data bytes present (file = 79 bytes,
  TEnd = 64 + 0 + 16 = 80). The message must quote the real file size,
  not the 0 left by the pre-raise Destroy (BUG-005). }
procedure TGgufFileTest.TestUT_GgufFile_049;
var
  S: TBytesStream;
  P: string;
  A: TBytes;
  N, Q: Integer;
  Size: string;
begin
  P := K_DIR + '/UT_GgufFile_049.gguf';
  S := TBytesStream.Create;
  try
    WHeader(S, 3, 1, 0);
    WTensor(S, 'xt', 1, [4], 0, 0);
    PadTo(S, 64); { 58 -> 64 }
    A := MakePat(15, 99); { only 15 data bytes, 16 needed }
    S.WriteBuffer(A[0], 15);
    S.SaveToFile(P);
  finally
    S.Free;
  end;
  try
    TGgufFile.Open(P).Free;
    raise Exception.Create('UT-GgufFile-049: EGgufError expected (extent, 1 byte short)');
  except
    on E: EGgufError do
    begin
      AssertTrue(Pos('xt', E.Message) > 0);
      AssertTrue(Pos('position', E.Message) > 0);
      AssertTrue(Pos(' bytes', E.Message) > 0);
      { the quoted file size: the digits right before ' bytes' (2.6/11) }
      N := Pos(' bytes', E.Message);
      AssertTrue(N > 1);
      Q := N;
      while (Q > 1) and (E.Message[Q - 1] >= '0') and (E.Message[Q - 1] <= '9') do
        Dec(Q);
      Size := Copy(E.Message, Q, N - Q);
      AssertTrue(Size <> '0'); { BUG-005: the size is the real one, not 0 }
      AssertEquals(79, StrToInt(Size)); { the file really is 79 bytes }
    end;
  end;
end;

{ Main section (T06 pattern): run the registered test suite. }
var
  App: TTestRunner;
begin
  ForceDirectories(K_DIR);
  RegisterTest(TGgufFileTest);
  App := TTestRunner.Create(nil);
  App.Initialize;
  App.Title := 'GgufFile UT (T01b1 stage 1)';
  App.Run;
  App.Free;
end.
