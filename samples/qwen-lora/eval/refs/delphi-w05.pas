program Groups;
{$APPTYPE CONSOLE}
uses
  System.SysUtils, System.Generics.Collections;
var
  D: TObjectDictionary<Char, TList<string>>;
  Keys: TList<Char>;
  W: string;
  C: Char;
  L: TList<string>;
begin
  D := TObjectDictionary<Char, TList<string>>.Create([doOwnsValues]);
  Keys := TList<Char>.Create;
  try
    for W in ['apple', 'avocado', 'banana', 'blueberry', 'cherry', 'apricot'] do
    begin
      if not D.TryGetValue(W[1], L) then
      begin
        L := TList<string>.Create;
        D.Add(W[1], L);
      end;
      L.Add(W);
    end;
    for C in D.Keys do
      Keys.Add(C);
    Keys.Sort;
    for C in Keys do
      Writeln(C, ': ', string.Join(', ', D[C].ToArray));
  finally
    Keys.Free;
    D.Free;
  end;
end.
