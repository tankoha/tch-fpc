program Sheet;
{$APPTYPE CONSOLE}
uses
  SysUtils, Generics.Collections;
var
  Cells: TDictionary<string, string>;

function Eval(const Name: string): Integer;
var
  F: string;
  P: Integer;
begin
  F := Cells[Name];
  P := F.IndexOf('+');
  if P >= 0 then
    Exit(Eval(F.Substring(0, P)) + Eval(F.Substring(P + 1)));
  P := F.IndexOf('*');
  if P >= 0 then
  begin
    if Cells.ContainsKey(F.Substring(P + 1)) then
      Exit(Eval(F.Substring(0, P)) * Eval(F.Substring(P + 1)))
    else
      Exit(Eval(F.Substring(0, P)) * StrToInt(F.Substring(P + 1)));
  end;
  Result := StrToInt(F);
end;

begin
  Cells := TDictionary<string, string>.Create;
  try
    Cells.Add('A1', '5');
    Cells.Add('A2', '7');
    Cells.Add('A3', 'A1+A2');
    Cells.Add('A4', 'A3*2');
    for var N in ['A1', 'A2', 'A3', 'A4'] do
      Writeln(N, ' = ', Eval(N));
  finally
    Cells.Free;
  end;
end.
