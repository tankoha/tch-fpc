program ListCount;
{$mode objfpc}{$H+}
uses
  Generics.Collections;
type
  TIntList = specialize TList<Integer>;
var
  L: TIntList;
  I: Integer;
begin
  L := TIntList.Create;
  try
    L.Add(10);
    L.Add(20);
    L.Add(30);
    for I in L do
      WriteLn(I);
    WriteLn('count ', L.Count);
  finally
    L.Free;
  end;
end.
