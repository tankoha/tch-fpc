program Counter;
{$APPTYPE CONSOLE}
uses
  System.SysUtils, System.Generics.Collections;
const
  Arrivals: array[0..4] of Integer = (0, 1, 2, 6, 7);
var
  Q: TQueue<Integer>;
  Free, Arrive, Wait, N, Total: Integer;
  FS: TFormatSettings;
begin
  FS := TFormatSettings.Create;
  FS.DecimalSeparator := '.';
  Q := TQueue<Integer>.Create;
  try
    for var T in Arrivals do
      Q.Enqueue(T);
    Free := 0;
    N := 0;
    Total := 0;
    while Q.Count > 0 do
    begin
      Arrive := Q.Dequeue;
      Inc(N);
      if Free < Arrive then
        Free := Arrive;
      Wait := Free - Arrive;
      Inc(Total, Wait);
      Writeln('customer ', N, ' waits ', Wait);
      Free := Free + 3;
    end;
    Writeln('average ', FormatFloat('0.00', Total / N, FS));
  finally
    Q.Free;
  end;
end.
