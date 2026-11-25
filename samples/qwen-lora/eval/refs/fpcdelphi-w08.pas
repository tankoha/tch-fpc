program Temperature;
{$mode delphi}
uses
  SysUtils;
type
  TTemperature = class
  private
    FCelsius: Double;
    function GetFahrenheit: Double;
    function GetKelvin: Double;
  public
    property Celsius: Double read FCelsius write FCelsius;
    property Fahrenheit: Double read GetFahrenheit;
    property Kelvin: Double read GetKelvin;
  end;

function TTemperature.GetFahrenheit: Double;
begin
  Result := FCelsius * 9 / 5 + 32;
end;

function TTemperature.GetKelvin: Double;
begin
  Result := FCelsius + 273.15;
end;

const
  Values: array[0..3] of Double = (-40, 0, 37, 100);
var
  T: TTemperature;
  I: Integer;
  FS: TFormatSettings;
begin
  FS := DefaultFormatSettings;
  FS.DecimalSeparator := '.';
  T := TTemperature.Create;
  try
    for I := 0 to High(Values) do
    begin
      T.Celsius := Values[I];
      WriteLn(FormatFloat('0.0', T.Celsius, FS), ': ', FormatFloat('0.0', T.Fahrenheit, FS),
        ' / ', FormatFloat('0.00', T.Kelvin, FS));
    end;
  finally
    T.Free;
  end;
end.
