program Longest;
{$mode objfpc}{$H+}
uses
  SysUtils;
var
  Words: TStringArray;
  W: string;
  Best: Integer;
begin
  Words := 'a quick brown fox jumped'.Split(' ');
  Best := 0;
  for W in Words do
    if Length(W) > Best then
      Best := Length(W);
  WriteLn(Best);
end.
