program SumList;
{$mode delphi}
uses
  Generics.Collections;
type
  TIntList = TList<Integer>;
var
  L: TIntList;
  I, S: Integer;
begin
  L := TIntList.Create;
  for I := 1 to 5 do
    L.Add(I);
  S := 0;
  for I in L do
    S := S + I;
  WriteLn(S);
  L.Free;
end.
