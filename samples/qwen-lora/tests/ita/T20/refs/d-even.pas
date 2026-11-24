program Even;
{$mode delphi}
var
  I: Integer;
begin
  I := 2;
  Write(I);
  while I < 10 do
  begin
    I := I + 2;
    Write(' ', I);
  end;
  WriteLn;
end.
