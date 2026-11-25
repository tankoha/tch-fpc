program Table;
{$mode delphi}
uses
  SysUtils;
var
  R, C: Integer;
begin
  for R := 1 to 6 do
  begin
    for C := 1 to 6 do
      Write(Format('%4d', [R * C]));
    WriteLn;
  end;
end.
