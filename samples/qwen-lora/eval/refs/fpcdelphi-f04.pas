program Totals;
{$mode delphi}
uses
  SysUtils;
function Total(const A: array of Integer): Integer;
var
  X: Integer;
begin
  Result := 0;
  for X in A do
    Result := Result + X;
end;
begin
  WriteLn('Total: ' + IntToStr(Total([1, 2, 3])));
end.
