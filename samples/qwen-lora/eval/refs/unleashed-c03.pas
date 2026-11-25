program Squares;
{$mode unleashed}
uses
  SysUtils, Generics.Collections;
begin
  var L := autofree TList<Integer>.Create;
  for var I := 1 to 5 do
    L.Add(I * I);
  for var V in L do
    Write(V, ' ');
  WriteLn;
end.
