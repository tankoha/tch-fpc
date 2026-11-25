program PriceTotal;
{$mode objfpc}{$H+}
uses
  SysUtils;
const
  Prices: array[0..2] of Double = (1.50, 2.75, 3.20);
var
  Total: Double;
  I: Integer;
  FS: TFormatSettings;
begin
  FS := DefaultFormatSettings;
  FS.DecimalSeparator := '.';
  Total := 0.0;
  for I := 0 to High(Prices) do
    Total := Total + Prices[I];
  WriteLn('total ', FormatFloat('0.00', Total, FS));
end.
