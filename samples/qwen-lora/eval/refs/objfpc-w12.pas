program DaySets;
{$mode objfpc}{$H+}
type
  TDay = (Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday);
  TDays = set of TDay;
const
  Names: array[TDay] of string = ('Monday', 'Tuesday', 'Wednesday', 'Thursday',
    'Friday', 'Saturday', 'Sunday');

procedure PrintDays(const ADays: TDays);
var
  D: TDay;
  First: Boolean;
begin
  First := True;
  for D in ADays do
  begin
    if not First then
      Write(' ');
    Write(Names[D]);
    First := False;
  end;
  WriteLn;
end;

var
  Work, Meetings: TDays;
begin
  Work := [Monday..Friday];
  Meetings := [Tuesday, Thursday, Saturday];
  PrintDays(Work * Meetings);
  PrintDays(Meetings - Work);
end.
