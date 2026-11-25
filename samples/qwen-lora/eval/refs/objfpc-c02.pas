program Doubler;
{$mode objfpc}{$H+}
type
  TIntFunc = function(X: Integer): Integer;

function Twice(X: Integer): Integer;
begin
  Result := X * 2;
end;

var
  F: TIntFunc;
  I: Integer;
begin
  F := @Twice;
  for I := 1 to 3 do
    WriteLn(F(I));
end.
