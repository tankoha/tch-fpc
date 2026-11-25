program FizzBuzz;
{$APPTYPE CONSOLE}
uses
  SysUtils;
begin
  for var I := 1 to 20 do
    if I mod 15 = 0 then
      Writeln('FizzBuzz')
    else if I mod 3 = 0 then
      Writeln('Fizz')
    else if I mod 5 = 0 then
      Writeln('Buzz')
    else
      Writeln(I);
end.
