program Newton;
{$mode delphi}
uses
  SysUtils;

procedure Show(N: Double);
var
  X, Prev: Double;
  It: Integer;
  FS: TFormatSettings;
begin
  FS := DefaultFormatSettings;
  FS.DecimalSeparator := '.';
  X := N;
  It := 0;
  repeat
    Prev := X;
    X := (X + N / X) / 2;
    Inc(It);
  until Abs(X - Prev) < 1e-12;
  WriteLn('sqrt(', FormatFloat('0', N, FS), ') = ', FormatFloat('0.000000', X, FS), ' [', It, ']');
end;

begin
  Show(2);
  Show(10);
  Show(144);
end.
