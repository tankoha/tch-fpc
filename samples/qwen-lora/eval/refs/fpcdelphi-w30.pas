program Calendar;
{$mode delphi}
uses
  SysUtils, DateUtils;
var
  First, D, Col: Integer;
begin
  WriteLn('Mo Tu We Th Fr Sa Su');
  First := DayOfTheWeek(EncodeDate(2024, 2, 1));
  Col := 1;
  while Col < First do
  begin
    Write('   ');
    Inc(Col);
  end;
  for D := 1 to DaysInAMonth(2024, 2) do
  begin
    Write(Format('%2d', [D]));
    if Col = 7 then
    begin
      WriteLn;
      Col := 1;
    end
    else
    begin
      if D < DaysInAMonth(2024, 2) then
        Write(' ');
      Inc(Col);
    end;
  end;
  if Col <> 1 then
    WriteLn;
end.
