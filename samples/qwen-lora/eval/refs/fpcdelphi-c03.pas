program Squares;
{$mode delphi}
uses
  SysUtils;
var
  Total, I, Sq: Integer;
begin
  Total := 0;
  for I := 1 to 4 do
  begin
    Sq := I * I;
    WriteLn(I, '^2 = ', Sq);
    Total := Total + Sq;
  end;
  WriteLn('total ', Total);
end.
