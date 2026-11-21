{ UT program for src/GgufQuant.pas, stage 1: the rows UT-GgufQuant-001..025
  of tests/ut/cases/GgufQuant.cases.md (stage 2, rows 101.., is
  tests/ut/GgufQuantK_test.pas). One published method per case row, named
  TestUT_GgufQuant_NNN. Singles are compared by their bits (Bits below).
  Invoked with `--all --format=plain` (see .sdlc/ut-coverage.sh).

  Skeleton by the harness owner; the test methods are written in T02b1a
  (001..010) and T02b1b (011..025). }
program GgufQuant_test;

{$mode objfpc}{$H+}

uses
  SysUtils, Classes, Math, GgufFile, GgufQuant, fpcunit, testregistry, consoletestrunner;

type
  TGgufQuantTest = class(TTestCase)
  public
    { The bits of a Single (no tolerance anywhere in this program). }
    class function Bits(AX: Single): Cardinal; static;
    { A Single from its bits. }
    class function FromBits(AU: Cardinal): Single; static;
  published
    procedure TestUT_GgufQuant_001;
    procedure TestUT_GgufQuant_002;
    procedure TestUT_GgufQuant_003;
    procedure TestUT_GgufQuant_004;
    procedure TestUT_GgufQuant_005;
    procedure TestUT_GgufQuant_006;
    procedure TestUT_GgufQuant_007;
    procedure TestUT_GgufQuant_008;
    procedure TestUT_GgufQuant_009;
    { T02b1a continues here (010); T02b1b adds TestUT_GgufQuant_011..026. }
    procedure TestUT_GgufQuant_010;
    { T02b1b: rows 011..026 (facts; case table lines). }
    procedure TestUT_GgufQuant_011;
    procedure TestUT_GgufQuant_012;
    procedure TestUT_GgufQuant_013;
    procedure TestUT_GgufQuant_014;
    procedure TestUT_GgufQuant_015;
    procedure TestUT_GgufQuant_016;
    procedure TestUT_GgufQuant_017;
    procedure TestUT_GgufQuant_018;
    procedure TestUT_GgufQuant_019;
    procedure TestUT_GgufQuant_020;
    procedure TestUT_GgufQuant_021;
    procedure TestUT_GgufQuant_022;
    procedure TestUT_GgufQuant_023;
    procedure TestUT_GgufQuant_024;
    procedure TestUT_GgufQuant_025;
    procedure TestUT_GgufQuant_026;
    procedure TestUT_GgufQuant_027;
  end;

class function TGgufQuantTest.Bits(AX: Single): Cardinal;
begin
  Move(AX, Result, 4);
end;

class function TGgufQuantTest.FromBits(AU: Cardinal): Single;
begin
  Move(AU, Result, 4);
end;

procedure TGgufQuantTest.TestUT_GgufQuant_001;
begin
  AssertEquals('UT-GgufQuant-001 $0000 -> +0.0', QWord(0), QWord(Bits(Fp16ToSingle($0000))));
  AssertEquals('UT-GgufQuant-001 $8000 -> -0.0', QWord($80000000), QWord(Bits(Fp16ToSingle($8000))));
  AssertEquals('UT-GgufQuant-001 $3C00 -> 1.0', QWord($3F800000), QWord(Bits(Fp16ToSingle($3C00))));
  AssertEquals('UT-GgufQuant-001 $C000 -> -2.0', QWord($C0000000), QWord(Bits(Fp16ToSingle($C000))));
  AssertEquals('UT-GgufQuant-001 $7BFF -> exact 65504', QWord(Bits(65504.0)), QWord(Bits(Fp16ToSingle($7BFF))));
  AssertEquals('UT-GgufQuant-001 $7BFF -> $477FE000', QWord($477FE000), QWord(Bits(Fp16ToSingle($7BFF))));
end;

procedure TGgufQuantTest.TestUT_GgufQuant_002;
begin
  AssertEquals('UT-GgufQuant-002 $7C00 -> +inf', QWord($7F800000), QWord(Bits(Fp16ToSingle($7C00))));
  AssertEquals('UT-GgufQuant-002 $FC00 -> -inf', QWord($FF800000), QWord(Bits(Fp16ToSingle($FC00))));
  AssertEquals('UT-GgufQuant-002 $7E00 -> qNaN $7FC00000', QWord($7FC00000), QWord(Bits(Fp16ToSingle($7E00))));
  AssertEquals('UT-GgufQuant-002 $7C01 -> sNaN $7FC02000', QWord($7FC02000), QWord(Bits(Fp16ToSingle($7C01))));
end;

procedure TGgufQuantTest.TestUT_GgufQuant_003;
begin
  AssertEquals('UT-GgufQuant-003 $0001 -> 2^-24', QWord(Bits(1.0 / 16777216.0)), QWord(Bits(Fp16ToSingle($0001))));
  AssertEquals('UT-GgufQuant-003 $03FF -> 1023*2^-24', QWord(Bits(1023.0 / 16777216.0)), QWord(Bits(Fp16ToSingle($03FF))));
  AssertEquals('UT-GgufQuant-003 $8001 -> -(2^-24)', QWord(Bits(-1.0 / 16777216.0)), QWord(Bits(Fp16ToSingle($8001))));
end;

procedure TGgufQuantTest.TestUT_GgufQuant_004;
begin
  AssertEquals('UT-GgufQuant-004 1.0 -> $3C00', QWord($3C00), QWord(SingleToFp16(1.0)));
  AssertEquals('UT-GgufQuant-004 -2.0 -> $C000', QWord($C000), QWord(SingleToFp16(-2.0)));
  AssertEquals('UT-GgufQuant-004 65504.0 -> $7BFF', QWord($7BFF), QWord(SingleToFp16(65504.0)));
  AssertEquals('UT-GgufQuant-004 +0.0 -> $0000', QWord(0), QWord(SingleToFp16(0.0)));
  AssertEquals('UT-GgufQuant-004 -0.0 -> $8000', QWord($8000), QWord(SingleToFp16(-0.0)));
end;

procedure TGgufQuantTest.TestUT_GgufQuant_005;
var
  R: Word;
begin
  AssertEquals('UT-GgufQuant-005 +inf -> $7C00', QWord($7C00), QWord(SingleToFp16(FromBits($7F800000))));
  AssertEquals('UT-GgufQuant-005 -inf -> $FC00', QWord($FC00), QWord(SingleToFp16(FromBits($FF800000))));
  R := SingleToFp16(FromBits($7FC00000));
  AssertTrue('UT-GgufQuant-005 qNaN +: is a NaN', (R and $7C00) = $7C00);
  AssertTrue('UT-GgufQuant-005 qNaN +: quiet bit set', (R and $200) <> 0);
  AssertEquals('UT-GgufQuant-005 qNaN +: sign 0', QWord(0), QWord(R and $8000));
  R := SingleToFp16(FromBits($FFC00000));
  AssertTrue('UT-GgufQuant-005 qNaN -: is a NaN', (R and $7C00) = $7C00);
  AssertTrue('UT-GgufQuant-005 qNaN -: quiet bit set', (R and $200) <> 0);
  AssertEquals('UT-GgufQuant-005 qNaN -: sign 1', QWord($8000), QWord(R and $8000));
end;

procedure TGgufQuantTest.TestUT_GgufQuant_006;
begin
  AssertEquals('UT-GgufQuant-006 1+2^-11 -> $3C00 (tie stays)', QWord($3C00), QWord(SingleToFp16(1.0 + 1.0 / 2048.0)));
  AssertEquals('UT-GgufQuant-006 1+3*2^-11 -> $3C02 (tie rounds up)', QWord($3C02), QWord(SingleToFp16(1.0 + 3.0 / 2048.0)));
end;

procedure TGgufQuantTest.TestUT_GgufQuant_007;
begin
  AssertEquals('UT-GgufQuant-007 65519 -> $7BFF (65504, max finite)', QWord($7BFF), QWord(SingleToFp16(65519.0)));
  AssertEquals('UT-GgufQuant-007 65520 -> $7C00 (tie to +inf)', QWord($7C00), QWord(SingleToFp16(65520.0)));
end;

procedure TGgufQuantTest.TestUT_GgufQuant_008;
var
  X: Single;
begin
  X := 1.0 / 33554432.0;
  AssertEquals('UT-GgufQuant-008 2^-25 -> $0000 (tie, to even, flush)', QWord(0), QWord(SingleToFp16(X)));
  AssertEquals('UT-GgufQuant-008 2^-25+2^-35 -> $0001 (above tie, up)', QWord($0001), QWord(SingleToFp16(X + 1.0 / 34359738368.0)));
end;

procedure TGgufQuantTest.TestUT_GgufQuant_009;
begin
  AssertEquals('UT-GgufQuant-009 $3F80 -> 1.0', QWord($3F800000), QWord(Bits(Bf16ToSingle($3F80))));
  AssertEquals('UT-GgufQuant-009 $C000 -> -2.0', QWord($C0000000), QWord(Bits(Bf16ToSingle($C000))));
  AssertEquals('UT-GgufQuant-009 $7F80 -> +inf', QWord($7F800000), QWord(Bits(Bf16ToSingle($7F80))));
end;

procedure TGgufQuantTest.TestUT_GgufQuant_010;
begin
  AssertEquals('UT-GgufQuant-010 0.5 -> 1', QWord(Bits(1.0)), QWord(Bits(RoundHalfAway(0.5))));
  AssertEquals('UT-GgufQuant-010 -0.5 -> -1', QWord(Bits(-1.0)), QWord(Bits(RoundHalfAway(-0.5))));
  AssertEquals('UT-GgufQuant-010 1.5 -> 2', QWord(Bits(2.0)), QWord(Bits(RoundHalfAway(1.5))));
  AssertEquals('UT-GgufQuant-010 2.5 -> 3', QWord(Bits(3.0)), QWord(Bits(RoundHalfAway(2.5))));
  AssertEquals('UT-GgufQuant-010 -2.5 -> -3', QWord(Bits(-3.0)), QWord(Bits(RoundHalfAway(-2.5))));
  AssertEquals('UT-GgufQuant-010 0.49999997 -> 0', QWord(0), QWord(Bits(RoundHalfAway(0.49999997))));
  AssertEquals('UT-GgufQuant-010 126.5 -> 127', QWord(Bits(127.0)), QWord(Bits(RoundHalfAway(126.5))));
  AssertEquals('UT-GgufQuant-010 -0.0 -> -0.0', QWord($80000000), QWord(Bits(RoundHalfAway(-0.0))));
end;

procedure TGgufQuantTest.TestUT_GgufQuant_011;
var
  B: TBytes;
  Y: array[0..3] of Single;
begin
  B := [$00, $00, $C0, $3F, $00, $00, $12, $C0, $00, $00, $00, $3F, $00, $00, $00, $80];
  DequantizeRow(0, @B[0], 4, @Y[0]);
  AssertEquals('UT-GgufQuant-011 el0  1.5', QWord($3FC00000), QWord(Bits(Y[0])));
  AssertEquals('UT-GgufQuant-011 el1 -2.28125', QWord($C0120000), QWord(Bits(Y[1])));
  AssertEquals('UT-GgufQuant-011 el2  0.5', QWord($3F000000), QWord(Bits(Y[2])));
  AssertEquals('UT-GgufQuant-011 el3 -0.0', QWord($80000000), QWord(Bits(Y[3])));
end;

procedure TGgufQuantTest.TestUT_GgufQuant_012;
var
  B: TBytes;
  Y: array[0..7] of Single;
begin
  B := [$00, $00, $00, $80, $00, $3C, $00, $C0, $00, $7C, $00, $FC, $FF, $7B, $00, $7E];
  DequantizeRow(1, @B[0], 8, @Y[0]);
  AssertEquals('UT-GgufQuant-012 el0  +0.0', QWord($00000000), QWord(Bits(Y[0])));
  AssertEquals('UT-GgufQuant-012 el1  -0.0', QWord($80000000), QWord(Bits(Y[1])));
  AssertEquals('UT-GgufQuant-012 el2  1.0', QWord($3F800000), QWord(Bits(Y[2])));
  AssertEquals('UT-GgufQuant-012 el3 -2.0', QWord($C0000000), QWord(Bits(Y[3])));
  AssertEquals('UT-GgufQuant-012 el4  +inf', QWord($7F800000), QWord(Bits(Y[4])));
  AssertEquals('UT-GgufQuant-012 el5  -inf', QWord($FF800000), QWord(Bits(Y[5])));
  AssertEquals('UT-GgufQuant-012 el6  65504', QWord($477FE000), QWord(Bits(Y[6])));
  AssertEquals('UT-GgufQuant-012 el7 qNaN $7E00', QWord($7FC00000), QWord(Bits(Y[7])));
end;

procedure TGgufQuantTest.TestUT_GgufQuant_013;
var
  B: TBytes;
  Y: array[0..7] of Single;
begin
  B := [$80, $3F, $00, $C0, $00, $40, $00, $3F, $00, $BF, $80, $7F, $80, $FF, $7F, $7F];
  DequantizeRow(30, @B[0], 8, @Y[0]);
  AssertEquals('UT-GgufQuant-013 el0  1.0', QWord($3F800000), QWord(Bits(Y[0])));
  AssertEquals('UT-GgufQuant-013 el1 -2.0', QWord($C0000000), QWord(Bits(Y[1])));
  AssertEquals('UT-GgufQuant-013 el2  2.0', QWord($40000000), QWord(Bits(Y[2])));
  AssertEquals('UT-GgufQuant-013 el3  0.5', QWord($3F000000), QWord(Bits(Y[3])));
  AssertEquals('UT-GgufQuant-013 el4 -0.5', QWord($BF000000), QWord(Bits(Y[4])));
  AssertEquals('UT-GgufQuant-013 el5  +inf', QWord($7F800000), QWord(Bits(Y[5])));
  AssertEquals('UT-GgufQuant-013 el6  -inf', QWord($FF800000), QWord(Bits(Y[6])));
  AssertEquals('UT-GgufQuant-013 el7 qNaN $7F7F', QWord($7F7F0000), QWord(Bits(Y[7])));
end;

procedure TGgufQuantTest.TestUT_GgufQuant_014;
var
  B: TBytes;
  Y: array[0..31] of Single;
  J: Integer;
begin
  { d = $3C00 (1.0); qs = 127, 127, 0, 5, -5, 8, -8, -106, 96, then 23 zeros. }
  SetLength(B, 34);
  FillChar(B[0], Length(B), 0);
  B[0] := $00;
  B[1] := $3C;
  B[2] := $7F;
  B[3] := $7F;
  B[4] := $00;
  B[5] := $05;
  B[6] := $FB;
  B[7] := $08;
  B[8] := $F8;
  B[9] := $96;
  B[10] := $60;
  AssertEquals('UT-GgufQuant-014 34 bytes', Int64(34), Int64(Length(B)));
  DequantizeRow(8, @B[0], 32, @Y[0]);
  AssertEquals('UT-GgufQuant-014 el0  127', QWord($42FE0000), QWord(Bits(Y[0])));
  AssertEquals('UT-GgufQuant-014 el1  127', QWord($42FE0000), QWord(Bits(Y[1])));
  AssertEquals('UT-GgufQuant-014 el2   0', QWord($00000000), QWord(Bits(Y[2])));
  AssertEquals('UT-GgufQuant-014 el3    5', QWord($40A00000), QWord(Bits(Y[3])));
  AssertEquals('UT-GgufQuant-014 el4   -5', QWord($C0A00000), QWord(Bits(Y[4])));
  AssertEquals('UT-GgufQuant-014 el5    8', QWord($41000000), QWord(Bits(Y[5])));
  AssertEquals('UT-GgufQuant-014 el6   -8', QWord($C1000000), QWord(Bits(Y[6])));
  AssertEquals('UT-GgufQuant-014 el7 -106', QWord($C2D40000), QWord(Bits(Y[7])));
  AssertEquals('UT-GgufQuant-014 el8   96', QWord($42C00000), QWord(Bits(Y[8])));
  for J := 9 to 31 do
    AssertEquals('UT-GgufQuant-014 el' + IntToStr(J) + ' = 0',
                 QWord($00000000), QWord(Bits(Y[J])));
end;

procedure TGgufQuantTest.TestUT_GgufQuant_015;
var
  B: TBytes;
  Y: array[0..63] of Single;
  J: Integer;
begin
  { Block A: d = $3C00 (1.0), qs[0] = 127, qs[1] = -1, rest 0. }
  { Block B (bytes 34..67): d = $4000 (2.0), qs[0] = 1, qs[1] = -3, rest 0. }
  SetLength(B, 68);
  FillChar(B[0], Length(B), 0);
  B[0] := $00;
  B[1] := $3C;
  B[2] := $7F;
  B[3] := $FF;
  B[34] := $00;
  B[35] := $40;
  B[36] := $01;
  B[37] := $FD;
  AssertEquals('UT-GgufQuant-015 68 bytes', Int64(68), Int64(Length(B)));
  DequantizeRow(8, @B[0], 64, @Y[0]);
  AssertEquals('UT-GgufQuant-015 el0   127', QWord($42FE0000), QWord(Bits(Y[0])));
  AssertEquals('UT-GgufQuant-015 el1    -1', QWord($BF800000), QWord(Bits(Y[1])));
  for J := 2 to 31 do
    AssertEquals('UT-GgufQuant-015 el' + IntToStr(J) + ' = 0',
                 QWord($00000000), QWord(Bits(Y[J])));
  AssertEquals('UT-GgufQuant-015 el32  2.0', QWord($40000000), QWord(Bits(Y[32])));
  AssertEquals('UT-GgufQuant-015 el33  -6.0', QWord($C0C00000), QWord(Bits(Y[33])));
  for J := 34 to 63 do
    AssertEquals('UT-GgufQuant-015 el' + IntToStr(J) + ' = 0',
                 QWord($00000000), QWord(Bits(Y[J])));
end;

procedure TGgufQuantTest.TestUT_GgufQuant_016;
var
  B: TBytes;
  Y: array[0..63] of Single;
  E: Exception;
begin
  SetLength(B, 64);
  try
    DequantizeRow(2, @B[0], 32, @Y[0]);
    AssertFalse('UT-GgufQuant-016 type 2 must raise', True);
  except
    on E: Exception do
    begin
      if not (E is EGgufError) then
        AssertTrue('UT-GgufQuant-016 expected EGgufError, got ' + E.ClassName + ': ' + E.Message, False);
      if (Pos('unsupported tensor type 2', E.Message) = 0) then
        AssertTrue('UT-GgufQuant-016 message must name the type: ' + E.Message, False);
    end;
  end;
end;

procedure TGgufQuantTest.TestUT_GgufQuant_017;
var
  B: TBytes;
  Y: array[0..63] of Single;
  E: Exception;
begin
  SetLength(B, 64);
  try
    DequantizeRow(8, @B[0], 33, @Y[0]);
    AssertFalse('UT-GgufQuant-017 count 33 must raise', True);
  except
    on E: Exception do
    begin
      if not (E is EGgufError) then
        AssertTrue('UT-GgufQuant-017 expected EGgufError, got ' + E.ClassName + ': ' + E.Message, False);
      if (Pos('q8_0', E.Message) = 0) then
        AssertTrue('UT-GgufQuant-017 message must name the type: ' + E.Message, False);
      if (Pos('33', E.Message) = 0) then
        AssertTrue('UT-GgufQuant-017 message must name the count: ' + E.Message, False);
    end;
  end;
end;

procedure TGgufQuantTest.TestUT_GgufQuant_018;
var
  B, Short: TBytes;
  Y: TSingleArray;
  E: Exception;
begin
  { A valid 34-byte q8_0 block: d = $3C00, qs[0] = 127, rest 0. }
  SetLength(B, 34);
  FillChar(B[0], Length(B), 0);
  B[0] := $00;
  B[1] := $3C;
  B[2] := $7F;
  Y := DequantizeBytes(8, B, 32);
  AssertEquals('UT-GgufQuant-018 OK: 32 elements', Int64(32), Int64(Length(Y)));
  AssertEquals('UT-GgufQuant-018 OK: el0 = 127', QWord($42FE0000), QWord(Bits(Y[0])));
  AssertEquals('UT-GgufQuant-018 OK: el31 = 0', QWord($00000000), QWord(Bits(Y[31])));
  SetLength(Short, 33);
  Move(B[0], Short[0], 33);
  try
    Y := DequantizeBytes(8, Short, 32);
    AssertFalse('UT-GgufQuant-018 33 bytes must raise', True);
  except
    on E: Exception do
    begin
      if not (E is EGgufError) then
        AssertTrue('UT-GgufQuant-018 expected EGgufError, got ' + E.ClassName + ': ' + E.Message, False);
      if (Pos('33', E.Message) = 0) then
        AssertTrue('UT-GgufQuant-018 message must name the actual length: ' + E.Message, False);
      if (Pos('34', E.Message) = 0) then
        AssertTrue('UT-GgufQuant-018 message must name the expected length: ' + E.Message, False);
    end;
  end;
end;

procedure TGgufQuantTest.TestUT_GgufQuant_019;
var
  X: array[0..31] of Single;
  Buf: TBytes;
  I: Integer;
begin
  for I := 0 to 31 do
    X[I] := 0.0;
  X[0] := 127.0;
  X[1] := -3.0;
  X[2] := 2.0;
  { Amax = 127, D = 1.0, d = $3C00, and with Id = 1 every q is x itself. }
  SetLength(Buf, 34);
  QuantizeRowQ8_0(@X[0], 32, @Buf[0]);
  AssertEquals('UT-GgufQuant-019 d lo byte', Int64($00), Int64(Buf[0]));
  AssertEquals('UT-GgufQuant-019 d = $3C00 (D = 1.0)', Int64($3C), Int64(Buf[1]));
  AssertEquals('UT-GgufQuant-019 q[0] = 127', Int64($7F), Int64(Buf[2]));
  AssertEquals('UT-GgufQuant-019 q[1] = -3', Int64($FD), Int64(Buf[3]));
  AssertEquals('UT-GgufQuant-019 q[2] = 2', Int64($02), Int64(Buf[4]));
  for I := 5 to 31 do
    AssertEquals('UT-GgufQuant-019 q[' + IntToStr(I) + '] = 0', Int64(0), Int64(Buf[2 + I]));
end;

procedure TGgufQuantTest.TestUT_GgufQuant_020;
var
  X: array[0..31] of Single;
  Buf: TBytes;
  I: Integer;
begin
  for I := 0 to 31 do
    X[I] := 0.0;
  { Amax = 0: D = 0, and in D = 0 Id := 0, d := $0000, every q := 0. }
  SetLength(Buf, 34);
  QuantizeRowQ8_0(@X[0], 32, @Buf[0]);
  AssertEquals('UT-GgufQuant-020 d lo byte', Int64($00), Int64(Buf[0]));
  AssertEquals('UT-GgufQuant-020 d hi byte', Int64($00), Int64(Buf[1]));
  for I := 0 to 31 do
    AssertEquals('UT-GgufQuant-020 q[' + IntToStr(I) + '] = 0', Int64(0), Int64(Buf[2 + I]));
end;

procedure TGgufQuantTest.TestUT_GgufQuant_021;
var
  X: array[0..31] of Single;
  Buf: TBytes;
  I: Integer;
begin
  for I := 0 to 31 do
    X[I] := 0.0;
  X[0] := 0.5;
  X[1] := -0.5;
  { Amax = 127 -> D = 1.0 -> d = $3C00. }
  X[2] := 127.0;
  SetLength(Buf, 34);
  QuantizeRowQ8_0(@X[0], 32, @Buf[0]);
  AssertEquals('UT-GgufQuant-021 d lo byte', Int64($00), Int64(Buf[0]));
  AssertEquals('UT-GgufQuant-021 d = $3C00', Int64($3C), Int64(Buf[1]));
  AssertEquals('UT-GgufQuant-021 q[0] =  1 (0.5 rounds away)', Int64($01), Int64(Buf[2]));
  AssertEquals('UT-GgufQuant-021 q[1] = -1 (-0.5 rounds away)', Int64($FF), Int64(Buf[3]));
  AssertEquals('UT-GgufQuant-021 q[2] = 127', Int64($7F), Int64(Buf[4]));
  for I := 5 to 31 do
    AssertEquals('UT-GgufQuant-021 q[' + IntToStr(I) + '] = 0', Int64(0), Int64(Buf[2 + I]));
end;

procedure TGgufQuantTest.TestUT_GgufQuant_022;
var
  X: array[0..31] of Single;
  Buf: TBytes;
  I: Integer;
begin
  for I := 0 to 31 do
    X[I] := 1.0E-40;
  { Amax = 1e-40 -> D is a positive subnormal <= 2^-128: d := 0, every
    q := 0, and 1 / D must not be computed (it would overflow). }
  SetLength(Buf, 34);
  QuantizeRowQ8_0(@X[0], 32, @Buf[0]);
  AssertEquals('UT-GgufQuant-022 d lo byte', Int64($00), Int64(Buf[0]));
  AssertEquals('UT-GgufQuant-022 d hi byte', Int64($00), Int64(Buf[1]));
  for I := 0 to 31 do
    AssertEquals('UT-GgufQuant-022 q[' + IntToStr(I) + '] = 0', Int64(0), Int64(Buf[2 + I]));
end;

procedure TGgufQuantTest.TestUT_GgufQuant_023;
var
  X: array[0..31] of Single;
  Buf: TBytes;
  E: Exception;
begin
  { One element is +inf: EGgufError. }
  FillChar(X, SizeOf(X), 0);
  X[0] := FromBits($7F800000);
  SetLength(Buf, 34);
  try
    QuantizeRowQ8_0(@X[0], 32, @Buf[0]);
    AssertTrue('UT-GgufQuant-023 a +inf input must raise', False);
  except
    on E: Exception do
      if not (E is EGgufError) then
        AssertTrue('UT-GgufQuant-023 expected EGgufError, got ' + E.ClassName + ': ' +
                   E.Message, False);
  end;
  { One element is NaN: EGgufError. }
  FillChar(X, SizeOf(X), 0);
  X[3] := FromBits($7FC00000);
  try
    QuantizeRowQ8_0(@X[0], 32, @Buf[0]);
    AssertTrue('UT-GgufQuant-023 a NaN input must raise', False);
  except
    on E: Exception do
      if not (E is EGgufError) then
        AssertTrue('UT-GgufQuant-023 expected EGgufError, got ' + E.ClassName + ': ' +
                   E.Message, False);
  end;
end;

procedure TGgufQuantTest.TestUT_GgufQuant_024;
var
  X, Y: array[0..31] of Single;
  Buf: TBytes;
  D: Single;
  I: Integer;
begin
  { A block of varied finite x with Amax = 127, so D = 127 / 127 = 1.0 and
    every y = qs · 1.0 is exact. }
  for I := 0 to 31 do
    X[I] := 0.0;
  X[0] := 0.25;
  X[1] := -0.75;
  X[2] := 127.0;
  X[3] := 5.0;
  X[4] := -5.0;
  X[5] := 0.125;
  SetLength(Buf, 34);
  QuantizeRowQ8_0(@X[0], 32, @Buf[0]);
  DequantizeRow(8, @Buf[0], 32, @Y[0]);
  D := 127.0 / 127.0;
  for I := 0 to 31 do
    AssertEquals('UT-GgufQuant-024 el' + IntToStr(I) + ' within D/2 + 1 ulp',
                 Int64(True),
                 Int64(Abs(Y[I] - X[I]) <= D / 2.0 + 2.0 / 16777216.0));
  AssertEquals('UT-GgufQuant-024 el0 = +0.0', QWord($00000000), QWord(Bits(Y[0])));
  AssertEquals('UT-GgufQuant-024 el1 = -1.0', QWord($BF800000), QWord(Bits(Y[1])));
  AssertEquals('UT-GgufQuant-024 el2 = 127', QWord($42FE0000), QWord(Bits(Y[2])));
  AssertEquals('UT-GgufQuant-024 el3 = 5', QWord($40A00000), QWord(Bits(Y[3])));
  AssertEquals('UT-GgufQuant-024 el4 = -5', QWord($C0A00000), QWord(Bits(Y[4])));
  AssertEquals('UT-GgufQuant-024 el5 = +0.0', QWord($00000000), QWord(Bits(Y[5])));
end;

procedure TGgufQuantTest.TestUT_GgufQuant_025;
var
  V: TSingleArray;
  B: TBytes;
  I: Integer;
  E: Exception;
begin
  { The valid path: 32 elements of 127.0 (Amax = 127, D = 1.0). }
  SetLength(V, 32);
  for I := 0 to 31 do
    V[I] := 127.0;
  B := QuantizeQ8_0(V);
  AssertEquals('UT-GgufQuant-025 OK: length', Int64(34), Int64(Length(B)));
  AssertEquals('UT-GgufQuant-025 OK: d lo byte', Int64($00), Int64(B[0]));
  AssertEquals('UT-GgufQuant-025 OK: d = $3C00', Int64($3C), Int64(B[1]));
  for I := 0 to 31 do
    AssertEquals('UT-GgufQuant-025 OK: q[' + IntToStr(I) + '] = 127',
                 Int64(127), Int64(B[2 + I]));
  { A length of 33 is not a multiple of 32: EGgufError naming it. }
  SetLength(V, 33);
  try
    B := QuantizeQ8_0(V);
    AssertTrue('UT-GgufQuant-025 length 33 must raise', False);
  except
    on E: Exception do
    begin
      if not (E is EGgufError) then
        AssertTrue('UT-GgufQuant-025 expected EGgufError, got ' + E.ClassName + ': ' +
                   E.Message, False);
      if (Pos('33', E.Message) = 0) then
        AssertTrue('UT-GgufQuant-025 message must name the length: ' + E.Message, False);
    end;
  end;
end;

procedure TGgufQuantTest.TestUT_GgufQuant_026;
begin
  { SingleToFp16: |X| >= 65536 gives (in)finity. }
  AssertEquals('UT-GgufQuant-026  65536.0 -> $7C00',
               Int64(Word($7C00)), Int64(Word(SingleToFp16(65536.0))));
  AssertEquals('UT-GgufQuant-026   1.0e10 -> $7C00',
               Int64(Word($7C00)), Int64(Word(SingleToFp16(1.0E10))));
  AssertEquals('UT-GgufQuant-026  -1.0e10 -> $FC00',
               Int64(Word($FC00)), Int64(Word(SingleToFp16(-1.0E10))));
end;

procedure TGgufQuantTest.TestUT_GgufQuant_027;
var
  X: array[0..31] of Single;
  Buf: TBytes;
  I: Integer;
begin
  { A block whose max |x| lies in [1, 2): Amax = 1.5, D = 1.5 / 127.
    Q8_0NonFinite must classify these finite values as finite (BUG-006). }
  for I := 0 to 31 do
    X[I] := 0.0;
  X[0] := 1.5;
  X[1] := -1.25;
  SetLength(Buf, 34);
  QuantizeRowQ8_0(@X[0], 32, @Buf[0]);
  AssertEquals('UT-GgufQuant-027 d low byte = $0C', Int64($0C), Int64(Buf[0]));
  AssertEquals('UT-GgufQuant-027 d high byte = $22 (d = $220C)',
               Int64($22), Int64(Buf[1]));
  AssertEquals('UT-GgufQuant-027 q[0] = 127', Int64(127), Int64(Buf[2]));
  AssertEquals('UT-GgufQuant-027 q[1] = -106', Int64(-106),
               Int64(ShortInt(Buf[3])));
  for I := 2 to 31 do
    AssertEquals('UT-GgufQuant-027 q[' + IntToStr(I) + '] = 0',
                 Int64(0), Int64(Buf[2 + I]));
end;

var
  App: TTestRunner;
begin
  RegisterTest(TGgufQuantTest);
  App := TTestRunner.Create(nil);
  App.Initialize;
  App.Title := 'GgufQuant UT (stage 1)';
  App.Run;
  App.Free;
end.
