program Sieve;
{$mode objfpc}{$H+}
const
  Limit = 59;
var
  Composite: array[2..Limit] of Boolean;
  I, J, N: Integer;
begin
  for I := 2 to Limit do
    Composite[I] := False;
  for I := 2 to Limit do
    if not Composite[I] then
    begin
      J := I * I;
      while J <= Limit do
      begin
        Composite[J] := True;
        Inc(J, I);
      end;
    end;
  N := 0;
  for I := 2 to Limit do
    if not Composite[I] then
    begin
      WriteLn(I);
      Inc(N);
    end;
  WriteLn('count: ', N);
end.
