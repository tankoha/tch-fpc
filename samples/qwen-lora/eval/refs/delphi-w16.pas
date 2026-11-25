program Factors;
{$APPTYPE CONSOLE}
uses
  System.SysUtils;

function Factorise(N: Integer): string;
var
  P, E: Integer;
begin
  Result := '';
  P := 2;
  while N > 1 do
  begin
    E := 0;
    while N mod P = 0 do
    begin
      N := N div P;
      Inc(E);
    end;
    if E > 0 then
    begin
      if Result <> '' then
        Result := Result + ' * ';
      Result := Result + IntToStr(P);
      if E > 1 then
        Result := Result + '^' + IntToStr(E);
    end;
    Inc(P);
  end;
end;

const
  Numbers: array[0..3] of Integer = (360, 97, 1001, 1024);
begin
  for var N in Numbers do
    Writeln(N, ' = ', Factorise(N));
end.
