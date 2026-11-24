program Temps;
{$mode objfpc}{$H+}
var
  C: Integer;
begin
  C := -10;
  while C <= 40 do
  begin
    WriteLn(C, ' C = ', C * 9 div 5 + 32, ' F');
    C := C + 25;
  end;
end.
