program Countdown;
{$APPTYPE CONSOLE}
uses
  System.SysUtils;
begin
  for var I := 3 downto 1 do
    Writeln(I, '...');
  Writeln('liftoff');
end.
