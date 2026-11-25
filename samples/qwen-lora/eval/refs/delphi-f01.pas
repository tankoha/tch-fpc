program Totals;
{$APPTYPE CONSOLE}
uses
  System.SysUtils;
var
  Total: Integer;
begin
  Total := 0;
  for var I := 1 to 4 do
    Total := Total + I;
  Writeln('total ', Total);
end.
