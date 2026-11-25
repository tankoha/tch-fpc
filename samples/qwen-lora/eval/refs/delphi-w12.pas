program Counters;
{$APPTYPE CONSOLE}
uses
  System.SysUtils;
type
  ICounter = interface
    ['{9E4A2B71-5C3D-4E8F-A1B2-C3D4E5F60718}']
    procedure Increment;
    function Value: Integer;
  end;

  TCounter = class(TInterfacedObject, ICounter)
  private
    FValue: Integer;
  public
    procedure Increment;
    function Value: Integer;
  end;

procedure TCounter.Increment;
begin
  Inc(FValue);
end;

function TCounter.Value: Integer;
begin
  Result := FValue;
end;

procedure Bump(const C: ICounter; N: Integer);
begin
  for var I := 1 to N do
    C.Increment;
end;

var
  C: ICounter;
begin
  C := TCounter.Create;
  Bump(C, 5);
  C.Increment;
  C.Increment;
  Writeln('value ', C.Value);
end.
