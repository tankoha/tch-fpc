program Sizes;
{$APPTYPE CONSOLE}
uses
  SysUtils;
const
  Numbers: array[0..2] of Integer = (3, 30, 300);
var
  S: string;
begin
  for var N in Numbers do
  begin
    if N >= 100 then
      S := 'large'
    else if N >= 10 then
      S := 'medium'
    else
      S := 'small';
    Writeln(N, ': ', S);
  end;
end.
