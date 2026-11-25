program Indexed;
{$APPTYPE CONSOLE}
uses
  SysUtils;
type
  TIndexedProc = reference to procedure(Index: Integer; const S: string);

procedure ForEachIndexed(const A: array of string; const P: TIndexedProc);
var
  I: Integer;
begin
  for I := 0 to High(A) do
    P(I, A[I]);
end;

var
  Chars: Integer;
begin
  Chars := 0;
  ForEachIndexed(['red', 'green', 'blue'],
    procedure(Index: Integer; const S: string)
    begin
      Writeln(Index, ': ', S);
      Chars := Chars + S.Length;
    end);
  Writeln('chars ', Chars);
end.
