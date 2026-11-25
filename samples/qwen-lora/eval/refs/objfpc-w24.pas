program Report;
{$mode objfpc}{$H+}
uses
  SysUtils;
type
  TItem = record
    Name: string;
    Qty: Integer;
    Price: Double;
  end;
const
  Items: array[0..2] of TItem = (
    (Name: 'Pen'; Qty: 3; Price: 1.20),
    (Name: 'Notebook'; Qty: 2; Price: 3.50),
    (Name: 'Ruler'; Qty: 1; Price: 0.99));
var
  I: Integer;
  Sum, T: Double;
  FS: TFormatSettings;
begin
  FS := DefaultFormatSettings;
  FS.DecimalSeparator := '.';
  WriteLn('Item       Qty   Total');
  Sum := 0;
  for I := 0 to High(Items) do
  begin
    T := Items[I].Qty * Items[I].Price;
    Sum := Sum + T;
    WriteLn(Format('%-10s%4d%8.2f', [Items[I].Name, Items[I].Qty, T], FS));
  end;
  WriteLn(Format('%-10s%4s%8.2f', ['Sum', '', Sum], FS));
end.
