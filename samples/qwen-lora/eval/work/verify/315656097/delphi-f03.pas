program MaxGeneric;
{$APPTYPE CONSOLE}
uses
  SysUtils, Generics.Collections;
type
  TIntList = TList<Integer>;
var
  L: TIntList;
  M: Integer;
begin
  L := TIntList.Create;
  try
    L.AddRange([8, 3, 5]);
    M := L[0];
    for var X in L do
      if X > M then
        M := X;
    Writeln('max ', M);
  finally
    L.Free;
  end;
end.
