program MatMul;
{$mode objfpc}{$H+}
type
  TMatrix = array of array of Integer;

function Multiply(const A, B: TMatrix): TMatrix;
var
  I, J, K: Integer;
begin
  Result := nil;
  SetLength(Result, Length(A), Length(B[0]));
  for I := 0 to High(A) do
    for J := 0 to High(B[0]) do
    begin
      Result[I][J] := 0;
      for K := 0 to High(B) do
        Result[I][J] := Result[I][J] + A[I][K] * B[K][J];
    end;
end;

var
  A, B, C: TMatrix;
  I, J: Integer;
begin
  A := nil;
  B := nil;
  SetLength(A, 2, 3);
  SetLength(B, 3, 2);
  A[0][0] := 1; A[0][1] := 2; A[0][2] := 3;
  A[1][0] := 4; A[1][1] := 5; A[1][2] := 6;
  B[0][0] := 7; B[0][1] := 8;
  B[1][0] := 9; B[1][1] := 10;
  B[2][0] := 11; B[2][1] := 12;
  C := Multiply(A, B);
  for I := 0 to High(C) do
  begin
    for J := 0 to High(C[I]) do
    begin
      if J > 0 then
        Write(' ');
      Write(C[I][J]);
    end;
    WriteLn;
  end;
end.
