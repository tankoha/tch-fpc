program Fractions;
{$mode delphi}
uses
  SysUtils;
type
  TFraction = record
    Num, Den: Integer;
    class function Make(N, D: Integer): TFraction; static;
    class operator Add(const A, B: TFraction): TFraction;
    class operator Multiply(const A, B: TFraction): TFraction;
    function ToString: string;
  end;

function Gcd(A, B: Integer): Integer;
begin
  A := Abs(A);
  B := Abs(B);
  while B <> 0 do
  begin
    Result := A mod B;
    A := B;
    B := Result;
  end;
  Result := A;
end;

class function TFraction.Make(N, D: Integer): TFraction;
var
  G: Integer;
begin
  if D < 0 then
  begin
    N := -N;
    D := -D;
  end;
  G := Gcd(N, D);
  if G = 0 then
    G := 1;
  Result.Num := N div G;
  Result.Den := D div G;
end;

class operator TFraction.Add(const A, B: TFraction): TFraction;
begin
  Result := Make(A.Num * B.Den + B.Num * A.Den, A.Den * B.Den);
end;

class operator TFraction.Multiply(const A, B: TFraction): TFraction;
begin
  Result := Make(A.Num * B.Num, A.Den * B.Den);
end;

function TFraction.ToString: string;
begin
  Result := IntToStr(Num) + '/' + IntToStr(Den);
end;

begin
  WriteLn((TFraction.Make(1, 2) + TFraction.Make(1, 3)).ToString);
  WriteLn((TFraction.Make(3, 4) * TFraction.Make(2, 9)).ToString);
  WriteLn((TFraction.Make(-2, 4) + TFraction.Make(1, 4)).ToString);
end.
