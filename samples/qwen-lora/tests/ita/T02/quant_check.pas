{$mode objfpc}{$H+}
program quant_check;
{ ITa driver of T02 (harness owner; run by tests/ita/T02-run.sh).

  Usage
    quant_check MODEL

  Opens MODEL once and, for every tensor in file order, prints
    deq<TAB><name><TAB><type><TAB><elements><TAB><sha256 of the dequantised fp32 bytes>
  and, when the element count is a multiple of 32,
    q8<TAB><name><TAB><type><TAB><elements><TAB><sha256 of the Q8_0 bytes of those values>
  Uses only the final interfaces of GgufFile (T01) and GgufQuant (T02); the
  SHA-256 below is the harness owner's and is checked against sha256sum by
  the script. Errors go to stderr with exit code 1. }

uses
  SysUtils, GgufFile, GgufQuant;

{ ---- SHA-256 (FIPS 180-4) ---- }

const
  K: array[0..63] of Cardinal = (
    $428a2f98, $71374491, $b5c0fbcf, $e9b5dba5, $3956c25b, $59f111f1, $923f82a4, $ab1c5ed5,
    $d807aa98, $12835b01, $243185be, $550c7dc3, $72be5d74, $80deb1fe, $9bdc06a7, $c19bf174,
    $e49b69c1, $efbe4786, $0fc19dc6, $240ca1cc, $2de92c6f, $4a7484aa, $5cb0a9dc, $76f988da,
    $983e5152, $a831c66d, $b00327c8, $bf597fc7, $c6e00bf3, $d5a79147, $06ca6351, $14292967,
    $27b70a85, $2e1b2138, $4d2c6dfc, $53380d13, $650a7354, $766a0abb, $81c2c92e, $92722c85,
    $a2bfe8a1, $a81a664b, $c24b8b70, $c76c51a3, $d192e819, $d6990624, $f40e3585, $106aa070,
    $19a4c116, $1e376c08, $2748774c, $34b0bcb5, $391c0cb3, $4ed8aa4a, $5b9cca4f, $682e6ff3,
    $748f82ee, $78a5636f, $84c87814, $8cc70208, $90befffa, $a4506ceb, $bef9a3f7, $c67178f2);

function Ror(X: Cardinal; N: Integer): Cardinal; inline;
begin
  Result := (X shr N) or (X shl (32 - N));
end;

procedure Compress(var H: array of Cardinal; P: PByte);
var
  W: array[0..63] of Cardinal;
  A, B, C, D, E, F, G, HH, T1, T2, S0, S1: Cardinal;
  I: Integer;
begin
  for I := 0 to 15 do
    W[I] := (Cardinal(P[I * 4]) shl 24) or (Cardinal(P[I * 4 + 1]) shl 16) or
            (Cardinal(P[I * 4 + 2]) shl 8) or Cardinal(P[I * 4 + 3]);
  for I := 16 to 63 do
  begin
    S0 := Ror(W[I - 15], 7) xor Ror(W[I - 15], 18) xor (W[I - 15] shr 3);
    S1 := Ror(W[I - 2], 17) xor Ror(W[I - 2], 19) xor (W[I - 2] shr 10);
    W[I] := W[I - 16] + S0 + W[I - 7] + S1;
  end;
  A := H[0]; B := H[1]; C := H[2]; D := H[3]; E := H[4]; F := H[5]; G := H[6]; HH := H[7];
  for I := 0 to 63 do
  begin
    S1 := Ror(E, 6) xor Ror(E, 11) xor Ror(E, 25);
    T1 := HH + S1 + ((E and F) xor ((not E) and G)) + K[I] + W[I];
    S0 := Ror(A, 2) xor Ror(A, 13) xor Ror(A, 22);
    T2 := S0 + ((A and B) xor (A and C) xor (B and C));
    HH := G; G := F; F := E; E := D + T1; D := C; C := B; B := A; A := T1 + T2;
  end;
  H[0] += A; H[1] += B; H[2] += C; H[3] += D; H[4] += E; H[5] += F; H[6] += G; H[7] += HH;
end;

function Sha256Hex(P: PByte; ALen: Int64): string;
var
  H: array[0..7] of Cardinal = ($6a09e667, $bb67ae85, $3c6ef372, $a54ff53a,
                                $510e527f, $9b05688c, $1f83d9ab, $5be0cd19);
  Tail: array[0..127] of Byte;
  Full, Rest, I, TailLen: Int64;
  Bits: QWord;
begin
  Full := ALen div 64;
  for I := 0 to Full - 1 do
    Compress(H, P + I * 64);
  Rest := ALen - Full * 64;
  FillChar(Tail, SizeOf(Tail), 0);
  if Rest > 0 then
    Move((P + Full * 64)^, Tail[0], Rest);
  Tail[Rest] := $80;
  if Rest < 56 then TailLen := 64 else TailLen := 128;
  Bits := QWord(ALen) * 8;
  for I := 0 to 7 do
    Tail[TailLen - 1 - I] := Byte(Bits shr (8 * I));
  Compress(H, @Tail[0]);
  if TailLen = 128 then
    Compress(H, @Tail[64]);
  Result := '';
  for I := 0 to 7 do
    Result := Result + LowerCase(IntToHex(H[I], 8));
end;

{ ---- main ---- }

var
  F: TGgufFile;
  I, J: Integer;
  T: TGgufTensorInfo;
  N: Int64;
  Raw, Q: TBytes;
  Y: TSingleArray;
  Line: string;
begin
  if ParamCount <> 1 then
  begin
    WriteLn(ErrOutput, 'usage: quant_check MODEL');
    Halt(2);
  end;
  if ParamStr(1) = '--selftest' then
  begin
    { sha256("abc") }
    SetLength(Raw, 3); Raw[0] := Ord('a'); Raw[1] := Ord('b'); Raw[2] := Ord('c');
    WriteLn(Sha256Hex(@Raw[0], 3));
    Halt(0);
  end;
  try
    F := TGgufFile.Open(ParamStr(1));
    try
      for I := 0 to F.TensorCount - 1 do
      begin
        T := F.Tensor[I];
        N := 1;
        for J := 0 to T.NDims - 1 do
          N := N * Int64(T.Dims[J]);
        F.ReadTensorBytes(I, Raw);
        Y := DequantizeBytes(T.TType, Raw, N);
        Line := T.Name + #9 + GgufTypeName(T.TType) + #9 + IntToStr(N) + #9;
        WriteLn('deq'#9, Line, Sha256Hex(@Y[0], N * 4));
        if N mod 32 = 0 then
        begin
          Q := QuantizeQ8_0(Y);
          WriteLn('q8'#9, Line, Sha256Hex(@Q[0], Length(Q)));
        end;
      end;
    finally
      F.Free;
    end;
  except
    on E: Exception do
    begin
      WriteLn(ErrOutput, 'error: ', E.Message);
      Halt(1);
    end;
  end;
end.
