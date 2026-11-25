program LookAndSay;
{$APPTYPE CONSOLE}
uses
  SysUtils;

function NextTerm(const S: string): string;
var
  I, N: Integer;
begin
  Result := '';
  I := 1;
  while I <= S.Length do
  begin
    N := 1;
    while (I + N <= S.Length) and (S[I + N] = S[I]) do
      Inc(N);
    Result := Result + IntToStr(N) + S[I];
    Inc(I, N);
  end;
end;

var
  S: string;
begin
  S := '1';
  for var I := 1 to 10 do
  begin
    Writeln(S);
    S := NextTerm(S);
  end;
end.
