program Stats;
{$mode objfpc}{$H+}
uses
  SysUtils, Math;
const
  Values: array[0..5] of Double = (4, 8, 15, 16, 23, 42);
var
  Mean, SD, Med: Double;
  I: Integer;
  FS: TFormatSettings;
begin
  FS := DefaultFormatSettings;
  FS.DecimalSeparator := '.';
  Mean := Math.Mean(Values);
  SD := 0;
  for I := 0 to High(Values) do
    SD := SD + Sqr(Values[I] - Mean);
  SD := Sqrt(SD / Length(Values));
  Med := (Values[2] + Values[3]) / 2;
  WriteLn('mean ', FormatFloat('0.00', Mean, FS));
  WriteLn('stddev ', FormatFloat('0.00', SD, FS));
  WriteLn('median ', FormatFloat('0.00', Med, FS));
end.
