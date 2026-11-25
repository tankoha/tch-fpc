program TopWords;
{$APPTYPE CONSOLE}
uses
  SysUtils, Generics.Collections, Generics.Defaults;
type
  TWordCount = TPair<string, Integer>;
  TWordList = TList<TWordCount>;
var
  D: TDictionary<string, Integer>;
  L: TWordList;
  W: string;
  N, I: Integer;
begin
  D := TDictionary<string, Integer>.Create;
  L := TWordList.Create(TComparer<TWordCount>.Construct(
    function(const A, B: TWordCount): Integer
    begin
      Result := B.Value - A.Value;
      if Result = 0 then
        Result := CompareStr(A.Key, B.Key);
    end));
  try
    for W in 'the quick brown fox jumps over the lazy dog the end'.Split([' ']) do
      if D.TryGetValue(W, N) then
        D[W] := N + 1
      else
        D.Add(W, 1);
    for var P in D do
      L.Add(P);
    L.Sort;
    for I := 0 to 2 do
      Writeln(L[I].Key, ' ', L[I].Value);
  finally
    L.Free;
    D.Free;
  end;
end.
