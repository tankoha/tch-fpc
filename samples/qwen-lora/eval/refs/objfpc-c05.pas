program Totals;
{$mode objfpc}{$H+}
uses
  Generics.Collections;
type
  TIntList = specialize TList<Integer>;
var
  L: TIntList;
  I, Sum, V: Integer;
begin
  L := TIntList.Create;
  try
    for I := 1 to 4 do
      L.Add(I * 10);
    Sum := 0;
    for V in L do
      Sum := Sum + V;
    WriteLn('items ', L.Count);
    WriteLn('sum ', Sum);
  finally
    L.Free;
  end;
end.
