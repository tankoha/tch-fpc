{ UT program for src/GgufQuant.pas, stage 2 (T02b2): the rows
  UT-GgufQuant-101.. of tests/ut/cases/GgufQuant.cases.md (q4_K, q5_K, q6_K).
  One published method per case row, named TestUT_GgufQuant_NNN. Singles are
  compared by their bits (Bits below); all expected values of the case table
  are exact binary fractions, written as literals and asserted bit-exact.
  Invoked with `--all --format=plain` (see .sdlc/ut-coverage.sh). }
program GgufQuantK_test;

{$mode objfpc}{$H+}

uses
  SysUtils, Classes, Math, GgufFile, GgufQuant, fpcunit, testregistry, consoletestrunner;

type
  TGgufQuantKTest = class(TTestCase)
  public
    { The bits of a Single (no tolerance anywhere in this program). }
    class function Bits(AX: Single): Cardinal; static;
    { A Single from its bits. }
    class function FromBits(AU: Cardinal): Single; static;
  published
    { Test cases 101.. for q4_K, q5_K, q6_K. }
    procedure TestUT_GgufQuant_101;
    procedure TestUT_GgufQuant_102;
    procedure TestUT_GgufQuant_103;
  end;

class function TGgufQuantKTest.Bits(AX: Single): Cardinal;
begin
  Move(AX, Result, 4);
end;

class function TGgufQuantKTest.FromBits(AU: Cardinal): Single;
begin
  Move(AU, Result, 4);
end;

{ Test methods for q4_K, q5_K, q6_K (rows 101..). }

procedure TGgufQuantKTest.TestUT_GgufQuant_101;
var
  Src: array[0..143] of Byte;
  Dst: array[0..255] of Single;
  G, L: Integer;
begin
  { q4_K block, 144 bytes: d = $3C00 (D = 1.0), dmin = $3800 (Mn = 0.5),
    scales = 82 C1 03 C5 40 40 00 02 24 35 FF 27 (gives Sc/M = 2/0, 1/0, 3/0,
    5/2, 36/18, 53/19, 15/15, 55/2), and in each group's 32 qs bytes L0..3 = $21
    (low = 1, high = 2), L4..31 = $00. }
  FillChar(Src, SizeOf(Src), 0);
  Src[0] := $00; Src[1] := $3C;
  Src[2] := $00; Src[3] := $38;
  Src[4] := $82; Src[5] := $C1; Src[6] := $03; Src[7] := $C5;
  Src[8] := $40; Src[9] := $40; {Src[10] := $00; Src[11] := $02;}
  Src[10] := $00; Src[11] := $02;
  Src[12] := $24; Src[13] := $35; Src[14] := $FF; Src[15] := $27;
  for G := 0 to 3 do
    for L := 0 to 3 do
      Src[16 + G * 32 + L] := $21;
  DequantizeRow(12, @Src, 256, @Dst);
  { Group 0 (Is = 0, 1): D1 = 2, M1 = 0, D2 = 1, M2 = 0 ->
    el0..3 = 2*1 = 2; el4..31 = 0; el32..35 = 1*2 = 2; el36..63 = 0. }
  for L := 0 to 3 do
    AssertEquals('UT-GgufQuant-101 el' + IntToStr(L) + ' = 2',
                 QWord(Bits(2.0)), QWord(Bits(Dst[L])));
  for L := 4 to 31 do
    AssertEquals('UT-GgufQuant-101 el' + IntToStr(L) + ' = 0',
                 0, QWord(Bits(Dst[L])));
  for L := 32 to 35 do
    AssertEquals('UT-GgufQuant-101 el' + IntToStr(L) + ' = 2',
                 QWord(Bits(2.0)), QWord(Bits(Dst[L])));
  for L := 36 to 63 do
    AssertEquals('UT-GgufQuant-101 el' + IntToStr(L) + ' = 0',
                 0, QWord(Bits(Dst[L])));
  { Group 1 (Is = 2, 3): D1 = 3, M1 = 0, D2 = 5, M2 = 1 ->
    el64..67 = 3; el68..95 = 0; el96..99 = 5*2 - 1 = 9; el100..127 = -1. }
  for L := 64 to 67 do
    AssertEquals('UT-GgufQuant-101 el' + IntToStr(L) + ' = 3',
                 QWord(Bits(3.0)), QWord(Bits(Dst[L])));
  for L := 68 to 95 do
    AssertEquals('UT-GgufQuant-101 el' + IntToStr(L) + ' = 0',
                 0, QWord(Bits(Dst[L])));
  for L := 96 to 99 do
    AssertEquals('UT-GgufQuant-101 el' + IntToStr(L) + ' = 9',
                 QWord(Bits(9.0)), QWord(Bits(Dst[L])));
  for L := 100 to 127 do
    AssertEquals('UT-GgufQuant-101 el' + IntToStr(L) + ' = -1',
                 QWord(Bits(-1.0)), QWord(Bits(Dst[L])));
  { Group 2 (Is = 4, 5): D1 = 36, M1 = 9, D2 = 53, M2 = 9.5 ->
    el128..131 = 36*1 - 9 = 27; el132..159 = -9; el160..163 = 53*2 - 9.5 = 96.5;
    el164..191 = -9.5. }
  for L := 128 to 131 do
    AssertEquals('UT-GgufQuant-101 el' + IntToStr(L) + ' = 27',
                 QWord(Bits(27.0)), QWord(Bits(Dst[L])));
  for L := 132 to 159 do
    AssertEquals('UT-GgufQuant-101 el' + IntToStr(L) + ' = -9',
                 QWord(Bits(-9.0)), QWord(Bits(Dst[L])));
  for L := 160 to 163 do
    AssertEquals('UT-GgufQuant-101 el' + IntToStr(L) + ' = 96.5',
                 QWord(Bits(96.5)), QWord(Bits(Dst[L])));
  for L := 164 to 191 do
    AssertEquals('UT-GgufQuant-101 el' + IntToStr(L) + ' = -9.5',
                 QWord(Bits(-9.5)), QWord(Bits(Dst[L])));
  { Group 3 (Is = 6, 7): D1 = 15, M1 = 7.5, D2 = 55, M2 = 1 ->
    el192..195 = 15*1 - 7.5 = 7.5; el196..223 = -7.5; el224..227 = 55*2 - 1 = 109;
    el228..255 = -1. }
  for L := 192 to 195 do
    AssertEquals('UT-GgufQuant-101 el' + IntToStr(L) + ' = 7.5',
                 QWord(Bits(7.5)), QWord(Bits(Dst[L])));
  for L := 196 to 223 do
    AssertEquals('UT-GgufQuant-101 el' + IntToStr(L) + ' = -7.5',
                 QWord(Bits(-7.5)), QWord(Bits(Dst[L])));
  for L := 224 to 227 do
    AssertEquals('UT-GgufQuant-101 el' + IntToStr(L) + ' = 109',
                 QWord(Bits(109.0)), QWord(Bits(Dst[L])));
  for L := 228 to 255 do
    AssertEquals('UT-GgufQuant-101 el' + IntToStr(L) + ' = -1',
                 QWord(Bits(-1.0)), QWord(Bits(Dst[L])));
end;

procedure TGgufQuantKTest.TestUT_GgufQuant_102;
var
  Src: array[0..175] of Byte;
  Dst: array[0..255] of Single;
  G, L: Integer;
  E: array[0..3] of Single;
begin
  { q5_K block, 176 bytes: d = $3C00 (D = 1.0), dmin = $3800 (Mn = 0.5), the
    same 12 scale bytes as the q4_K block, qh = FF 55 AA 00 then 28 x $00
    (at L0 the H bit is set in all groups, at L1 only U1, at L2 only U2, at L3
    clear), and the same 128 qs bytes as the q4_K block (group L0..3 = $21). }
  FillChar(Src, SizeOf(Src), 0);
  Src[0] := $00; Src[1] := $3C;
  Src[2] := $00; Src[3] := $38;
  Src[4] := $82; Src[5] := $C1; Src[6] := $03; Src[7] := $C5;
  Src[8] := $40; Src[9] := $40; Src[10] := $00; Src[11] := $02;
  Src[12] := $24; Src[13] := $35; Src[14] := $FF; Src[15] := $27;
  Src[16] := $FF; Src[17] := $55; Src[18] := $AA; {Src[19] := $00;}
  for G := 0 to 3 do
    for L := 0 to 3 do
      Src[48 + G * 32 + L] := $21;
  DequantizeRow(13, @Src, 256, @Dst);
  { Group 0 (U1 = 1, U2 = 2): H1 = 16, 16, 0, 0; H2 = 16, 0, 16, 0;
    D1 = 2, M1 = 0, D2 = 1, M2 = 0. }
  E[0] := 34.0; E[1] := 34.0; E[2] := 2.0; E[3] := 2.0; { 2*(1 + 16) etc. }
  for L := 0 to 3 do
    AssertEquals('UT-GgufQuant-102 el' + IntToStr(L) + ' = ' + FloatToStr(E[L]),
                 QWord(Bits(E[L])), QWord(Bits(Dst[L])));
  for L := 4 to 31 do
    AssertEquals('UT-GgufQuant-102 el' + IntToStr(L) + ' = 0',
                 0, QWord(Bits(Dst[L])));
  E[0] := 18.0; E[1] := 2.0; E[2] := 18.0; E[3] := 2.0; { 1*(2 + 16) etc. }
  for L := 32 to 35 do
    AssertEquals('UT-GgufQuant-102 el' + IntToStr(L) + ' = ' + FloatToStr(E[L - 32]),
                 QWord(Bits(E[L - 32])), QWord(Bits(Dst[L])));
  for L := 36 to 63 do
    AssertEquals('UT-GgufQuant-102 el' + IntToStr(L) + ' = 0',
                 0, QWord(Bits(Dst[L])));
  { Group 1 (U1 = 4, U2 = 8): H1 = 16, 16, 0, 0; H2 = 16, 0, 16, 0;
    D1 = 3, M1 = 0, D2 = 5, M2 = 1. }
  E[0] := 51.0; E[1] := 51.0; E[2] := 3.0; E[3] := 3.0;
  for L := 64 to 67 do
    AssertEquals('UT-GgufQuant-102 el' + IntToStr(L) + ' = ' + FloatToStr(E[L - 64]),
                 QWord(Bits(E[L - 64])), QWord(Bits(Dst[L])));
  for L := 68 to 95 do
    AssertEquals('UT-GgufQuant-102 el' + IntToStr(L) + ' = 0',
                 0, QWord(Bits(Dst[L])));
  E[0] := 89.0; E[1] := 9.0; E[2] := 89.0; E[3] := 9.0;
  for L := 96 to 99 do
    AssertEquals('UT-GgufQuant-102 el' + IntToStr(L) + ' = ' + FloatToStr(E[L - 96]),
                 QWord(Bits(E[L - 96])), QWord(Bits(Dst[L])));
  for L := 100 to 127 do
    AssertEquals('UT-GgufQuant-102 el' + IntToStr(L) + ' = -1',
                 QWord(Bits(-1.0)), QWord(Bits(Dst[L])));
  { Group 2 (U1 = 16, U2 = 32): H1 = 16, 16, 0, 0; H2 = 16, 0, 16, 0;
    D1 = 36, M1 = 9, D2 = 53, M2 = 9.5. }
  E[0] := 603.0; E[1] := 603.0; E[2] := 27.0; E[3] := 27.0;
  for L := 128 to 131 do
    AssertEquals('UT-GgufQuant-102 el' + IntToStr(L) + ' = ' + FloatToStr(E[L - 128]),
                 QWord(Bits(E[L - 128])), QWord(Bits(Dst[L])));
  for L := 132 to 159 do
    AssertEquals('UT-GgufQuant-102 el' + IntToStr(L) + ' = -9',
                 QWord(Bits(-9.0)), QWord(Bits(Dst[L])));
  E[0] := 944.5; E[1] := 96.5; E[2] := 944.5; E[3] := 96.5;
  for L := 160 to 163 do
    AssertEquals('UT-GgufQuant-102 el' + IntToStr(L) + ' = ' + FloatToStr(E[L - 160]),
                 QWord(Bits(E[L - 160])), QWord(Bits(Dst[L])));
  for L := 164 to 191 do
    AssertEquals('UT-GgufQuant-102 el' + IntToStr(L) + ' = -9.5',
                 QWord(Bits(-9.5)), QWord(Bits(Dst[L])));
  { Group 3 (U1 = 64, U2 = 128): H1 = 16, 0, 16, 0; H2 = 16, 0, 16, 0;
    D1 = 15, M1 = 7.5, D2 = 55, M2 = 1. }
  E[0] := 247.5; E[1] := 247.5; E[2] := 7.5; E[3] := 7.5;
  for L := 192 to 195 do
    AssertEquals('UT-GgufQuant-102 el' + IntToStr(L) + ' = ' + FloatToStr(E[L - 192]),
                 QWord(Bits(E[L - 192])), QWord(Bits(Dst[L])));
  for L := 196 to 223 do
    AssertEquals('UT-GgufQuant-102 el' + IntToStr(L) + ' = -7.5',
                 QWord(Bits(-7.5)), QWord(Bits(Dst[L])));
  E[0] := 989.0; E[1] := 109.0; E[2] := 989.0; E[3] := 109.0;
  for L := 224 to 227 do
    AssertEquals('UT-GgufQuant-102 el' + IntToStr(L) + ' = ' + FloatToStr(E[L - 224]),
                 QWord(Bits(E[L - 224])), QWord(Bits(Dst[L])));
  for L := 228 to 255 do
    AssertEquals('UT-GgufQuant-102 el' + IntToStr(L) + ' = -1',
                 QWord(Bits(-1.0)), QWord(Bits(Dst[L])));
end;

procedure TGgufQuantKTest.TestUT_GgufQuant_103;
var
  Src: array[0..209] of Byte;
  Dst: array[0..255] of Single;
  L: Integer;
begin
  { q6_K block, 210 bytes: ql (128) with ql[0] = $12, ql[32] = $34, rest 0;
    qh (64) with qh[0] = $0F, rest 0; scales (16 int8) =
    FE 03 FF 05 07 FD 04 FA 01 FC 02 08 FB 06 FF 0A
    (= -2, 3, -1, 5, 7, -3, 4, -6, 1, -4, 2, 8, -5, 6, -1, 10); d = $3C00. }
  FillChar(Src, SizeOf(Src), 0);
  Src[0] := $12;
  Src[32] := $34;
  Src[128] := $0F;
  Src[192] := $FE; Src[193] := $03; Src[194] := $FF; Src[195] := $05;
  Src[196] := $07; Src[197] := $FD; Src[198] := $04; Src[199] := $FA;
  Src[200] := $01; Src[201] := $FC; Src[202] := $02; Src[203] := $08;
  Src[204] := $FB; Src[205] := $06; Src[206] := $FF; Src[207] := $0A;
  Src[208] := $00; Src[209] := $3C;
  DequantizeRow(14, @Src, 256, @Dst);
  { D = 1.0. Half 0: at L0 (ql[0] = $12, ql[32] = $34, qh[0] = $0F)
    Q1 = 18, Q2 = 20, Q3 = -31, Q4 = -29; every other L all-zero quants ->
    Q = -32. So the special elements are el0, el32, el64, el96. }
  AssertEquals('UT-GgufQuant-103 el0 = -36', QWord(Bits(-36.0)), QWord(Bits(Dst[0])));
  for L := 1 to 15 do
    AssertEquals('UT-GgufQuant-103 el' + IntToStr(L) + ' = 64',
                 QWord(Bits(64.0)), QWord(Bits(Dst[L])));
  for L := 16 to 31 do
    AssertEquals('UT-GgufQuant-103 el' + IntToStr(L) + ' = -96',
                 QWord(Bits(-96.0)), QWord(Bits(Dst[L])));
  AssertEquals('UT-GgufQuant-103 el32 = -20', QWord(Bits(-20.0)), QWord(Bits(Dst[32])));
  for L := 33 to 47 do
    AssertEquals('UT-GgufQuant-103 el' + IntToStr(L) + ' = 32',
                 QWord(Bits(32.0)), QWord(Bits(Dst[L])));
  for L := 48 to 63 do
    AssertEquals('UT-GgufQuant-103 el' + IntToStr(L) + ' = -160',
                 QWord(Bits(-160.0)), QWord(Bits(Dst[L])));
  AssertEquals('UT-GgufQuant-103 el64 = -217', QWord(Bits(-217.0)), QWord(Bits(Dst[64])));
  for L := 65 to 79 do
    AssertEquals('UT-GgufQuant-103 el' + IntToStr(L) + ' = -224',
                 QWord(Bits(-224.0)), QWord(Bits(Dst[L])));
  for L := 80 to 95 do
    AssertEquals('UT-GgufQuant-103 el' + IntToStr(L) + ' = 96',
                 QWord(Bits(96.0)), QWord(Bits(Dst[L])));
  AssertEquals('UT-GgufQuant-103 el96 = -116', QWord(Bits(-116.0)), QWord(Bits(Dst[96])));
  for L := 97 to 111 do
    AssertEquals('UT-GgufQuant-103 el' + IntToStr(L) + ' = -128',
                 QWord(Bits(-128.0)), QWord(Bits(Dst[L])));
  for L := 112 to 127 do
    AssertEquals('UT-GgufQuant-103 el' + IntToStr(L) + ' = 192',
                 QWord(Bits(192.0)), QWord(Bits(Dst[L])));
  { Half 1 (scales idx 8..15 = 1, -4, 2, 8, -5, 6, -1, 10, all zero quants,
    so every Q = -32): the ranges are SC[k] * -32. }
  for L := 128 to 143 do
    AssertEquals('UT-GgufQuant-103 el' + IntToStr(L) + ' = -32',
                 QWord(Bits(-32.0)), QWord(Bits(Dst[L])));
  for L := 144 to 159 do
    AssertEquals('UT-GgufQuant-103 el' + IntToStr(L) + ' = 128',
                 QWord(Bits(128.0)), QWord(Bits(Dst[L])));
  for L := 160 to 175 do
    AssertEquals('UT-GgufQuant-103 el' + IntToStr(L) + ' = -64',
                 QWord(Bits(-64.0)), QWord(Bits(Dst[L])));
  for L := 176 to 191 do
    AssertEquals('UT-GgufQuant-103 el' + IntToStr(L) + ' = -256',
                 QWord(Bits(-256.0)), QWord(Bits(Dst[L])));
  for L := 192 to 207 do
    AssertEquals('UT-GgufQuant-103 el' + IntToStr(L) + ' = 160',
                 QWord(Bits(160.0)), QWord(Bits(Dst[L])));
  for L := 208 to 223 do
    AssertEquals('UT-GgufQuant-103 el' + IntToStr(L) + ' = -192',
                 QWord(Bits(-192.0)), QWord(Bits(Dst[L])));
  for L := 224 to 239 do
    AssertEquals('UT-GgufQuant-103 el' + IntToStr(L) + ' = 32',
                 QWord(Bits(32.0)), QWord(Bits(Dst[L])));
  for L := 240 to 255 do
    AssertEquals('UT-GgufQuant-103 el' + IntToStr(L) + ' = -320',
                 QWord(Bits(-320.0)), QWord(Bits(Dst[L])));
end;

var
  App: TTestRunner;
begin
  RegisterTest(TGgufQuantKTest);
  App := TTestRunner.Create(nil);
  App.Initialize;
  App.Title := 'GgufQuant UT (stage 2)';
  App.Run;
  App.Free;
end.
