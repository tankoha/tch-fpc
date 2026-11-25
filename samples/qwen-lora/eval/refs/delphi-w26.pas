program Polygon;
{$APPTYPE CONSOLE}
uses
  System.SysUtils;
const
  X: array[0..4] of Double = (0, 4, 4, 2, 0);
  Y: array[0..4] of Double = (0, 0, 3, 5, 3);
var
  Area, Perim: Double;
  J: Integer;
  FS: TFormatSettings;
begin
  FS := TFormatSettings.Create;
  FS.DecimalSeparator := '.';
  Area := 0;
  Perim := 0;
  for var I := 0 to 4 do
  begin
    J := (I + 1) mod 5;
    Area := Area + X[I] * Y[J] - X[J] * Y[I];
    Perim := Perim + Sqrt(Sqr(X[J] - X[I]) + Sqr(Y[J] - Y[I]));
  end;
  Writeln('area ', FormatFloat('0.0', Abs(Area) / 2, FS));
  Writeln('perimeter ', FormatFloat('0.00', Perim, FS));
end.
