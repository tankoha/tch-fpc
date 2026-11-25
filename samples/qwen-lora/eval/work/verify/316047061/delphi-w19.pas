program Thermostat;
{$APPTYPE CONSOLE}
uses
  SysUtils;
type
  TTargetEvent = procedure(Sender: TObject; NewTarget: Integer) of object;

  TThermostat = class
  private
    FTarget: Integer;
    FOnChange: TTargetEvent;
    procedure SetTarget(AValue: Integer);
  public
    property Target: Integer read FTarget write SetTarget;
    property OnChange: TTargetEvent read FOnChange write FOnChange;
  end;

  TDisplay = class
  public
    procedure Show(Sender: TObject; NewTarget: Integer);
  end;

procedure TThermostat.SetTarget(AValue: Integer);
begin
  if AValue < 5 then
    AValue := 5;
  if AValue > 30 then
    AValue := 30;
  FTarget := AValue;
  if Assigned(FOnChange) then
    FOnChange(Self, FTarget);
end;

procedure TDisplay.Show(Sender: TObject; NewTarget: Integer);
begin
  Writeln('target now ', NewTarget);
end;

var
  T: TThermostat;
  D: TDisplay;
begin
  T := TThermostat.Create;
  D := TDisplay.Create;
  try
    T.OnChange := D.Show;
    T.Target := 21;
    T.Target := 40;
    T.Target := 2;
    Writeln('final ', T.Target);
  finally
    T.Free;
    D.Free;
  end;
end.
