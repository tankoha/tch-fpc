program Fib15;
{$mode objfpc}{$H+}
var
  A, B, T: Int64;
  I: Integer;
begin
  A := 0;
  B := 1;
  for I := 1 to 15 do
  begin
    if I > 1 then
      Write(' ');
    Write(A);
    T := A + B;
    A := B;
    B := T;
  end;
  WriteLn;
end.
