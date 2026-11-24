program Sum;
{$mode objfpc}{$H+}
var
  I, S: Integer;
begin
  S := 0;
  for I := 1 to 100 do
    S := S + I;
  WriteLn(S);
end.
