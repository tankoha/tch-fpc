program Cities;
{$mode objfpc}{$H+}
uses
  fgl;
type
  TPopMap = specialize TFPGMap<string, Integer>;
var
  M: TPopMap;
  I, Sum: Integer;
begin
  M := TPopMap.Create;
  try
    M.Sorted := True;
    M.Add('Alpha', 120);
    M.Add('Beta', 45);
    M.Add('Gamma', 300);
    M.Add('Delta', 75);
    M.Add('Epsilon', 210);
    Sum := 0;
    for I := 0 to M.Count - 1 do
    begin
      if M.Data[I] > 100 then
        WriteLn(M.Keys[I], ' ', M.Data[I]);
      Inc(Sum, M.Data[I]);
    end;
    WriteLn('sum ', Sum);
  finally
    M.Free;
  end;
end.
