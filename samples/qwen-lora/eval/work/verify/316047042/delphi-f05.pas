program Names;
{$APPTYPE CONSOLE}
uses
  SysUtils, Generics.Collections;
var
  L: TList<string>;
begin
  L := TList<string>.Create;
  try
    L.Add('Ann');
    L.Add('Bea');
    L.Add('Cal');
    for var S in L do
      Writeln(S);
  finally
    L.Free;
  end;
end.
