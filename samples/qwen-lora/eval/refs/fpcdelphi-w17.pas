program Weekdays;
{$mode delphi}
uses
  SysUtils, DateUtils;
const
  Months: array[1..12] of string = ('January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December');
var
  M, D, N: Integer;
  T: TDateTime;
begin
  for M := 1 to 12 do
  begin
    N := 0;
    for D := 1 to DaysInAMonth(2025, M) do
    begin
      T := EncodeDate(2025, M, D);
      if DayOfTheWeek(T) <= 5 then
        Inc(N);
    end;
    WriteLn(Months[M], ': ', N);
  end;
end.
