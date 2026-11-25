program KeyValues;
{$mode objfpc}{$H+}
uses
  Classes, SysUtils;
var
  L: TStringList;
  I: Integer;
  FS: TFormatSettings;
  Price: Double;
  Qty: Integer;
begin
  FS := DefaultFormatSettings;
  FS.DecimalSeparator := '.';
  L := TStringList.Create;
  try
    L.Delimiter := ';';
    L.StrictDelimiter := True;
    L.DelimitedText := 'name=Widget;price=12.50;qty=4';
    for I := 0 to L.Count - 1 do
      WriteLn(L.Names[I], ' -> ', L.ValueFromIndex[I]);
    Price := StrToFloat(L.Values['price'], FS);
    Qty := StrToInt(L.Values['qty']);
    WriteLn('total ', FormatFloat('0.00', Price * Qty, FS));
  finally
    L.Free;
  end;
end.
