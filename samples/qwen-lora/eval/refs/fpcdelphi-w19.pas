program MatrixFib;
{$mode delphi}
type
  TMatrix2 = record
    A, B, C, D: Int64;
    class function Make(AA, AB, AC, AD: Int64): TMatrix2; static;
    function Mul(const M: TMatrix2): TMatrix2;
    function Power(N: Integer): TMatrix2;
  end;

class function TMatrix2.Make(AA, AB, AC, AD: Int64): TMatrix2;
begin
  Result.A := AA;
  Result.B := AB;
  Result.C := AC;
  Result.D := AD;
end;

function TMatrix2.Mul(const M: TMatrix2): TMatrix2;
begin
  Result := Make(A * M.A + B * M.C, A * M.B + B * M.D, C * M.A + D * M.C, C * M.B + D * M.D);
end;

function TMatrix2.Power(N: Integer): TMatrix2;
var
  Base: TMatrix2;
begin
  Result := Make(1, 0, 0, 1);
  Base := Self;
  while N > 0 do
  begin
    if Odd(N) then
      Result := Result.Mul(Base);
    Base := Base.Mul(Base);
    N := N div 2;
  end;
end;

begin
  WriteLn('fib(30) = ', TMatrix2.Make(1, 1, 1, 0).Power(30).B);
end.
