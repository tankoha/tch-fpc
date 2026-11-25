program Names;
{$mode unleashed}
uses
  Classes, SysUtils;
begin
  var L := autofree TStringList.Create;
  L.Add('Ada');
  L.Add('Linus');
  L.Add('Grace');
  L.Sort;
  for var I := 0 to L.Count - 1 do
    WriteLn($'{I + 1}: {L[I]}');
end.
