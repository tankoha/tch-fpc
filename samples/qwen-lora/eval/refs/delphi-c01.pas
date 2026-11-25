program Pairs;
{$APPTYPE CONSOLE}
uses
  System.SysUtils, System.Generics.Collections;
var
  M: TDictionary<string, Integer>;
begin
  M := TDictionary<string, Integer>.Create;
  try
    M.Add('a', 1);
    M.Add('b', 2);
    Writeln(M['a'] + M['b']);
  finally
    M.Free;
  end;
end.
