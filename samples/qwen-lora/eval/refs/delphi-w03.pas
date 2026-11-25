program Helpers;
{$APPTYPE CONSOLE}
uses
  System.SysUtils;
var
  S, T: string;
  W: string;
begin
  S := '  Delphi Programming Language  ';
  T := S.Trim;
  Writeln(T);
  Writeln(T.ToUpper);
  Writeln(T.Length);
  Writeln(BoolToStr(T.StartsWith('Delphi'), True));
  for W in T.Split([' ']) do
    Writeln(W);
end.
