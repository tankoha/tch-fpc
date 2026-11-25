program CounterEvent;
{$mode objfpc}{$H+}
type
  TChangeEvent = procedure(Sender: TObject; NewValue: Integer) of object;

  TCounter = class
  private
    FValue: Integer;
    FOnChange: TChangeEvent;
  public
    procedure Increment;
    property Value: Integer read FValue;
    property OnChange: TChangeEvent read FOnChange write FOnChange;
  end;

  TLogger = class
  public
    procedure Changed(Sender: TObject; NewValue: Integer);
  end;

procedure TCounter.Increment;
begin
  Inc(FValue);
  if Assigned(FOnChange) then
    FOnChange(Self, FValue);
end;

procedure TLogger.Changed(Sender: TObject; NewValue: Integer);
begin
  WriteLn('changed to ', NewValue);
end;

var
  C: TCounter;
  L: TLogger;
  I: Integer;
begin
  C := TCounter.Create;
  L := TLogger.Create;
  try
    C.OnChange := @L.Changed;
    for I := 1 to 3 do
      C.Increment;
    WriteLn('final ', C.Value);
  finally
    C.Free;
    L.Free;
  end;
end.
