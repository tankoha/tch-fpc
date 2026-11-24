program Stack;
{$mode objfpc}{$H+}
var
  Items: array[0..9] of Integer;
  Top, V: Integer;
begin
  Top := 0;
  for V := 1 to 5 do
  begin
    Items[Top] := V * 10;
    Inc(Top);
  end;
  while Top > 0 do
  begin
    Dec(Top);
    Write(Items[Top], ' ');
  end;
  WriteLn;
end.
