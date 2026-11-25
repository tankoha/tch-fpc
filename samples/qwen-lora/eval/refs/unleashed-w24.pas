program Distances;
{$mode unleashed}
uses
  SysUtils;

function Delta(X1, Y1, X2, Y2: Integer): (dx, dy: Integer);
begin
  Result := (X2 - X1, Y2 - Y1);
end;

procedure Show(X1, Y1, X2, Y2: Integer);
begin
  var (dx, dy) := Delta(X1, Y1, X2, Y2);
  var FS := DefaultFormatSettings;
  FS.DecimalSeparator := '.';
  WriteLn('d = ', FormatFloat('0.00', Sqrt(dx * dx + dy * dy), FS));
end;

begin
  Show(0, 0, 3, 4);
  Show(1, 1, 2, 3);
  Show(-2, 3, 4, -5);
end.
