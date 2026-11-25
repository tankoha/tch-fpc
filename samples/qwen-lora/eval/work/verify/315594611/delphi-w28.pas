program Prefix;
{$APPTYPE CONSOLE}
uses
  SysUtils;

function CommonPrefix(const A: array of string): string;
begin
  Result := A[0];
  for var I := 1 to High(A) do
    while not A[I].StartsWith(Result) do
      Result := Result.Substring(0, Result.Length - 1);
end;

procedure Show(const A: array of string);
var
  P: string;
begin
  P := CommonPrefix(A);
  if P = '' then
    Writeln('(none)')
  else
    Writeln(P);
end;

begin
  Show(['interstellar', 'internet', 'interval', 'internal']);
  Show(['dog', 'racecar', 'car']);
end.
