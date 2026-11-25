program BinSearch;
{$mode objfpc}{$H+}
const
  Data: array[0..9] of Integer = (2, 5, 8, 12, 16, 23, 38, 56, 72, 91);

function Find(AValue: Integer): Integer;
var
  Lo, Hi, Mid: Integer;
begin
  Lo := 0;
  Hi := High(Data);
  while Lo <= Hi do
  begin
    Mid := (Lo + Hi) div 2;
    if Data[Mid] = AValue then
      Exit(Mid)
    else if Data[Mid] < AValue then
      Lo := Mid + 1
    else
      Hi := Mid - 1;
  end;
  Result := -1;
end;

const
  Wanted: array[0..3] of Integer = (23, 2, 91, 50);
var
  I: Integer;
begin
  for I := 0 to High(Wanted) do
    WriteLn(Wanted[I], ': ', Find(Wanted[I]));
end.
