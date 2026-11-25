program Grades;
{$mode objfpc}{$H+}
uses
  SysUtils;
const
  Scores: array[0..2] of Integer = (55, 72, 91);
var
  S: Integer;
  R: string;
begin
  for S in Scores do
  begin
    if S >= 60 then
      R := 'pass'
    else
      R := 'fail';
    WriteLn(IntToStr(S) + ' -> ' + R);
  end;
end.
