program TrafficLight;
{$mode objfpc}{$H+}
type
  TLight = (Red, RedYellow, Green, Yellow);
const
  Duration: array[TLight] of Integer = (3, 1, 4, 2);
  Names: array[TLight] of string = ('Red', 'RedYellow', 'Green', 'Yellow');
var
  State: TLight;
  Left, Tick: Integer;
begin
  State := Red;
  Left := Duration[State];
  for Tick := 0 to 11 do
  begin
    WriteLn(Tick, ': ', Names[State]);
    Dec(Left);
    if Left = 0 then
    begin
      if State = High(TLight) then
        State := Low(TLight)
      else
        State := Succ(State);
      Left := Duration[State];
    end;
  end;
end.
