program LeapYears;
{$mode objfpc}{$H+}
uses
  SysUtils, DateUtils;
const
  Years: array[0..3] of Word = (1900, 2000, 2024, 2023);
var
  I: Integer;
begin
  for I := 0 to High(Years) do
    if IsLeapYear(Years[I]) then
      WriteLn(Years[I], ': leap')
    else
      WriteLn(Years[I], ': common');
  WriteLn(DaysBetween(EncodeDate(2024, 1, 1), EncodeDate(2024, 3, 1)));
end.
