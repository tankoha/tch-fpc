program GcdLcm;
{$mode objfpc}{$H+}

function Gcd(A, B: Integer): Integer;
begin
  if B = 0 then
    Result := A
  else
    Result := Gcd(B, A mod B);
end;

procedure Show(A, B: Integer);
begin
  WriteLn(A, ' ', B, ' ', Gcd(A, B), ' ', A div Gcd(A, B) * B);
end;

begin
  Show(12, 18);
  Show(17, 5);
  Show(100, 75);
end.
