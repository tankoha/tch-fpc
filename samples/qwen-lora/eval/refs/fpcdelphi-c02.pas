program Alarm;
{$mode delphi}
type
  TAlarmEvent = procedure(const Msg: string) of object;

  TSiren = class
    procedure Sound(const Msg: string);
  end;

  TSensor = class
    OnAlarm: TAlarmEvent;
    procedure Detect(Level: Integer);
  end;

procedure TSiren.Sound(const Msg: string);
begin
  WriteLn('ALARM: ', Msg);
end;

procedure TSensor.Detect(Level: Integer);
begin
  if (Level > 5) and Assigned(OnAlarm) then
    OnAlarm('level ' + Chr(Ord('0') + Level));
end;

var
  Siren: TSiren;
  Sensor: TSensor;
  I: Integer;
begin
  Siren := TSiren.Create;
  Sensor := TSensor.Create;
  Sensor.OnAlarm := Siren.Sound;
  for I := 4 to 7 do
    Sensor.Detect(I);
  Sensor.Free;
  Siren.Free;
end.
