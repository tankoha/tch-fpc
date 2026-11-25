program Letters;
{$mode delphi}
uses
  SysUtils, Generics.Collections;
var
  D: TDictionary<Char, Integer>;
  Keys: TList<Char>;
  C: Char;
  N: Integer;
begin
  D := TDictionary<Char, Integer>.Create;
  Keys := TList<Char>.Create;
  try
    for C in 'mississippi' do
      if D.TryGetValue(C, N) then
        D[C] := N + 1
      else
        D.Add(C, 1);
    for C in D.Keys do
      Keys.Add(C);
    Keys.Sort;
    for C in Keys do
      WriteLn(C, ' ', D[C]);
  finally
    D.Free;
    Keys.Free;
  end;
end.
