program Roman;
{$APPTYPE CONSOLE}
uses
  System.SysUtils;
const
  Vals: array[0..12] of Integer = (1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1);
  Syms: array[0..12] of string = ('M', 'CM', 'D', 'CD', 'C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I');

function ToRoman(N: Integer): string;
begin
  Result := '';
  for var I := 0 to High(Vals) do
    while N >= Vals[I] do
    begin
      Result := Result + Syms[I];
      Dec(N, Vals[I]);
    end;
end;

const
  Numbers: array[0..3] of Integer = (1994, 2024, 3999, 4);
begin
  for var N in Numbers do
    Writeln(N, ' = ', ToRoman(N));
end.
