program Squares;
{$mode objfpc}{$H+}
var
  I: Integer;
begin
  for I := 1 to 5 do
  begin
    if I > 1 then
      Write(' ');
    Write(I * I);
  end;
  WriteLn;
end.
