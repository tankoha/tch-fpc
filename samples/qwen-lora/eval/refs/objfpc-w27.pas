program Caesar;
{$mode objfpc}{$H+}

function Shift(const S: string; K: Integer): string;
var
  I: Integer;
  C: Char;
begin
  Result := S;
  for I := 1 to Length(S) do
  begin
    C := S[I];
    if C in ['a'..'z'] then
      Result[I] := Chr(Ord('a') + (Ord(C) - Ord('a') + K + 26) mod 26)
    else if C in ['A'..'Z'] then
      Result[I] := Chr(Ord('A') + (Ord(C) - Ord('A') + K + 26) mod 26);
  end;
end;

var
  E: string;
begin
  E := Shift('Hello, World', 3);
  WriteLn(E);
  WriteLn(Shift(E, -3));
end.
