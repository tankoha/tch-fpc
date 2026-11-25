program Dates;
{$APPTYPE CONSOLE}
uses
  SysUtils, DateUtils;
const
  Names: array[1..7] of string = ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',
    'Saturday', 'Sunday');
var
  A, B: TDateTime;
begin
  A := EncodeDate(2024, 1, 15);
  B := EncodeDate(2024, 12, 25);
  Writeln('from ', Names[DayOfTheWeek(A)]);
  Writeln('to ', Names[DayOfTheWeek(B)]);
  Writeln('days ', DaysBetween(A, B));
end.
