{$mode objfpc}{$H+}
unit GgufQuant;
{ Scalar dequantisers of the GGUF tensor types and the Q8_0 quantiser (T02).
  Must agree with upstream llama.cpp bit for bit: see tasks/T02-facts.md and
  docs/detailed-design/T02-detailed.md.

  Skeleton by the harness owner (the interface of the facts); the routines
  are written in T02b1a, T02b1b and T02b2. }

interface

uses
  SysUtils, Math, GgufFile;   { EGgufError, GgufTensorBytes, GgufTypeName }

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

implementation

function Fp16ToSingle(AH: Word): Single;
var
  E, M, S, U: Cardinal;
begin
  E := (AH shr 10) and $1F;
  M := AH and $3FF;
  S := AH and $8000;
  if E = 0 then
  begin
    Result := M * (1.0 / 16777216.0);
    if S <> 0 then
      Result := -Result;
  end
  else
  begin
    if E = 31 then
      if M = 0 then
        U := (S shl 16) or $7F800000
      else
        U := (S shl 16) or $7FC00000 or (M shl 13)
    else
      U := (S shl 16) or ((E + 112) shl 23) or (M shl 13);
    Move(U, Result, SizeOf(U));
  end;
end;

function SingleToFp16(AX: Single): Word;
var
  U, Sign, E, R, Rem, Half: Cardinal;
  M: Cardinal;
begin
  Move(AX, U, SizeOf(U));
  Sign := (U shr 16) and $8000;
  E := (U shr 23) and $FF;
  M := U and $7FFFFF;
  if E = 255 then
    if M <> 0 then
      R := $7E00 or (M shr 13)
    else
      R := $7C00
  else if E >= 143 then
    R := $7C00
  else if E >= 113 then
  begin
    R := ((E - 112) shl 10) or (M shr 13);
    Rem := M and $1FFF;
    if (Rem > $1000) or ((Rem = $1000) and ((R and 1) <> 0)) then
      R := R + 1;
  end
  else if E < 102 then
    R := 0
  else
  begin
    M := M or $800000;
    R := M shr (126 - E);
    Rem := M and ((1 shl (126 - E)) - 1);
    Half := 1 shl (125 - E);
    if (Rem > Half) or ((Rem = Half) and ((R and 1) <> 0)) then
      R := R + 1;
  end;
  Result := Sign or R;
end;

function Bf16ToSingle(AH: Word): Single;
var
  U: Cardinal;
begin
  U := Cardinal(AH) shl 16;
  Move(U, Result, SizeOf(U));
end;

function RoundHalfAway(AX: Single): Single;
var
  T: Single;
begin
  T := Int(AX);
  if Abs(AX - T) >= 0.5 then
    if AX > 0 then
      T := T + 1
    else if AX < 0 then
      T := T - 1;
  Result := T;
end;

{ K-quant dequantisers (T02b2). One block per call: 256 elements out, the
  block bytes at ASrc^, in the order fixed by the facts (one operation at a
  time, left to right, so the single-precision results are exact). }

procedure ScaleMin(AIs: Integer; AScales: PByte; var ASc, AM: Cardinal);
begin
  { 2 6-bit fields packed into the 12 scale bytes (J = 0..7). }
  if AIs < 4 then
  begin
    ASc := AScales[AIs] and 63;
    AM := AScales[AIs + 4] and 63;
  end
  else
  begin
    ASc := (AScales[AIs + 4] and 15) or ((AScales[AIs - 4] shr 6) shl 4);
    AM := (AScales[AIs + 4] shr 4) or ((AScales[AIs] shr 6) shl 4);
  end;
end;

procedure DequantQ4_K(ASrc: PByte; ADst: PSingle);
var
  D, Mn, D1, M1, D2, M2: Single;
  Q: PByte;
  Iv, L: Integer;
  Sc, Mc: Cardinal;
begin
  D := Fp16ToSingle(PWord(ASrc)^);
  Mn := Fp16ToSingle(PWord(ASrc + 2)^);
  Q := ASrc + 16;
  Iv := 0;
  while Iv < 8 do
  begin
    ScaleMin(Iv, ASrc + 4, Sc, Mc);
    D1 := D * Sc;
    M1 := Mn * Mc;
    ScaleMin(Iv + 1, ASrc + 4, Sc, Mc);
    D2 := D * Sc;
    M2 := Mn * Mc;
    for L := 0 to 31 do
      ADst[L] := D1 * (Q[L] and $F) - M1;
    for L := 0 to 31 do
      ADst[32 + L] := D2 * (Q[L] shr 4) - M2;
    ADst := ADst + 64;
    Q := Q + 32;
    Iv := Iv + 2;
  end;
end;

procedure DequantQ5_K(ASrc: PByte; ADst: PSingle);
var
  D, Mn, D1, M1, D2, M2: Single;
  Q, QH: PByte;
  Iv, L, U1, U2, H1, H2: Integer;
  Sc, Mc: Cardinal;
begin
  D := Fp16ToSingle(PWord(ASrc)^);
  Mn := Fp16ToSingle(PWord(ASrc + 2)^);
  QH := ASrc + 16;
  Q := ASrc + 48;
  Iv := 0;
  U1 := 1;
  U2 := 2;
  while Iv < 8 do
  begin
    ScaleMin(Iv, ASrc + 4, Sc, Mc);
    D1 := D * Sc;
    M1 := Mn * Mc;
    ScaleMin(Iv + 1, ASrc + 4, Sc, Mc);
    D2 := D * Sc;
    M2 := Mn * Mc;
    for L := 0 to 31 do
    begin
      if (QH[L] and U1) <> 0 then
        H1 := 16
      else
        H1 := 0;
      ADst[L] := D1 * ((Q[L] and $F) + H1) - M1;
    end;
    for L := 0 to 31 do
    begin
      if (QH[L] and U2) <> 0 then
        H2 := 16
      else
        H2 := 0;
      ADst[32 + L] := D2 * ((Q[L] shr 4) + H2) - M2;
    end;
    ADst := ADst + 64;
    Q := Q + 32;
    Iv := Iv + 2;
    U1 := U1 shl 2;
    U2 := U2 shl 2;
  end;
end;

procedure DequantQ6_K(ASrc: PByte; ADst: PSingle);
var
  D: Single;
  AOut: PSingle;
  QL, QH: PByte;
  ASc: PShortInt;
  H, L, Iv, Q1, Q2, Q3, Q4: Integer;
begin
  D := Fp16ToSingle(PWord(ASrc + 208)^);
  QL := ASrc;
  QH := ASrc + 128;
  ASc := PShortInt(ASrc + 192);
  AOut := ADst;
  for H := 0 to 1 do   { two halves of 128 elements }
  begin
    for L := 0 to 31 do
    begin
      Iv := L div 16;
      Q1 := ((QL[L] and $F) or ((QH[L] and 3) shl 4)) - 32;
      Q2 := ((QL[L + 32] and $F) or (((QH[L] shr 2) and 3) shl 4)) - 32;
      Q3 := ((QL[L] shr 4) or (((QH[L] shr 4) and 3) shl 4)) - 32;
      Q4 := ((QL[L + 32] shr 4) or (((QH[L] shr 6) and 3) shl 4)) - 32;
      AOut[L] := D * ASc[Iv] * Q1;
      AOut[L + 32] := D * ASc[Iv + 2] * Q2;
      AOut[L + 64] := D * ASc[Iv + 4] * Q3;
      AOut[L + 96] := D * ASc[Iv + 6] * Q4;
    end;
    AOut := AOut + 128;
    QL := QL + 64;
    QH := QH + 32;
    ASc := ASc + 8;
  end;
end;

procedure DequantizeRow(ATType: Cardinal; ASrc: PByte; ACount: Int64; ADst: PSingle);
var
  I, B: Int64;
  D: Single;
  ADims: array of QWord;
begin
  { The T01 checks first, with the T01 messages: an unknown type
    ("unsupported tensor type N") and a count that is not a multiple of the
    block size (type and count named). }
  ADims := [QWord(ACount)];
  GgufTensorBytes(ATType, ADims);
  case ATType of
    0:
      for I := 0 to ACount - 1 do
        Move(ASrc[I * 4], ADst[I], 4); { f32 (0): the 4 bytes are the Single }
    1:
      for I := 0 to ACount - 1 do
        ADst[I] := Fp16ToSingle(PWord(ASrc + I * 2)^);
    8:
      for B := 0 to (ACount div 32) - 1 do
      begin
        { q8_0 (8), 34 bytes: d: fp16 (bytes 0..1), qs: 32 x int8 (2..33). }
        D := Fp16ToSingle(PWord(ASrc + B * 34)^);
        for I := 0 to 31 do
          ADst[B * 32 + I] := PShortInt(ASrc + B * 34 + 2 + I)^ * D;
      end;
    12:
      for B := 0 to (ACount div 256) - 1 do
        DequantQ4_K(ASrc + B * 144, ADst + B * 256); { q4_K (12), 144 bytes }
    13:
      for B := 0 to (ACount div 256) - 1 do
        DequantQ5_K(ASrc + B * 176, ADst + B * 256); { q5_K (13), 176 bytes }
    14:
      for B := 0 to (ACount div 256) - 1 do
        DequantQ6_K(ASrc + B * 210, ADst + B * 256); { q6_K (14), 210 bytes }
    30:
      for I := 0 to ACount - 1 do
        ADst[I] := Bf16ToSingle(PWord(ASrc + I * 2)^);
  end;
end;

function DequantizeBytes(ATType: Cardinal; const ABytes: TBytes; ACount: Int64): TSingleArray;
var
  Expected: QWord;
  ADims: array of QWord;
begin
  ADims := [QWord(ACount)];
  Expected := GgufTensorBytes(ATType, ADims);
  if QWord(Length(ABytes)) <> Expected then
    raise EGgufError.CreateFmt(
      'gguf: tensor type %s: the byte length %d does not equal the expected tensor length %d',
      [GgufTypeName(ATType), Length(ABytes), Expected]);
  SetLength(Result, ACount);
  DequantizeRow(ATType, Pointer(ABytes), ACount, @Result[0]);
end;

const
  { Single 2^-128 (= 2.93873588e-39); 340282366920938463463374607431768211456
    = 2^128. The largest D such that 1 / D is infinite: for
    0 < D <= GgufTinyD do not compute 1 / D, store d = 0 and every q = 0
    (facts "Quantizing Q8_0"). }
  GgufTinyD: Single = 1.0 / 340282366920938463463374607431768211456.0;

function Q8_0NonFinite(AX: Single): Boolean;
var
  C: Cardinal;
begin
  { inf and NaN are exactly the values whose eight exponent bits are all 1. }
  Move(AX, C, SizeOf(C));
  Result := ((C shr 23) and $FF) = $FF;
end;

procedure QuantizeRowQ8_0(ASrc: PSingle; ACount: Int64; ADst: PByte);
var
  B, J: Int64;
  Amax, D, X, X0, Id: Single;
  W: Word;
begin
  for B := 0 to (ACount div 32) - 1 do
  begin
    Amax := 0.0;
    for J := 0 to 31 do
    begin
      X := ASrc[B * 32 + J];
      if Q8_0NonFinite(X) then
        raise EGgufError.Create('gguf: q8_0 quantise: a non-finite input');
      Amax := Max(Amax, Abs(X));
    end;
    D := Amax / 127;
    if (D > 0.0) and (D <= GgufTinyD) then
    begin
      { 0 < D <= 2^-128: 1 / D would be infinite, so do not compute it;
        d := 0 and every q := 0. }
      PWord(ADst + B * 34)^ := 0;
      for J := 0 to 31 do
        PShortInt(ADst + B * 34 + 2 + J)^ := 0;
    end
    else
    begin
      if D <> 0.0 then
        Id := 1 / D
      else
        Id := 0;
      W := SingleToFp16(D);
      PWord(ADst + B * 34)^ := W;
      for J := 0 to 31 do
      begin
        X0 := ASrc[B * 32 + J] * Id;
        PShortInt(ADst + B * 34 + 2 + J)^ := Int8(Trunc(RoundHalfAway(X0)));
      end;
    end;
  end;
end;

function QuantizeQ8_0(const AValues: TSingleArray): TBytes;
begin
  if (Length(AValues) mod 32) <> 0 then
    raise EGgufError.CreateFmt('gguf: q8_0 quantise: the length %d is not a multiple of 32',
                               [Length(AValues)]);
  SetLength(Result, (Length(AValues) div 32) * 34);
  QuantizeRowQ8_0(@AValues[0], Length(AValues), @Result[0]);
end;

end.
