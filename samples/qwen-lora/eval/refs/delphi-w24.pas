program PowersOfTwo;
{$APPTYPE CONSOLE}
uses
  System.SysUtils;
begin
  for var N := 0 to 12 do
    Writeln(Format('%-6s%6d', ['2^' + IntToStr(N), 1 shl N]));
end.
