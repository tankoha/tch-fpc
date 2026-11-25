program PascalTriangle;
{$mode objfpc}{$H+}
var
  Row: array of Int64;
  I, J: Integer;
begin
  Row := nil;
  for I := 0 to 6 do
  begin
    SetLength(Row, I + 1);
    Row[I] := 1;
    for J := I - 1 downto 1 do
      Row[J] := Row[J] + Row[J - 1];
    Row[0] := 1;
    Write(StringOfChar(' ', 6 - I));
    for J := 0 to I do
    begin
      if J > 0 then
        Write(' ');
      Write(Row[J]);
    end;
    WriteLn;
  end;
end.
