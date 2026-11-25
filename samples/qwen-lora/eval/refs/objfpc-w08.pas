program ShapeAreas;
{$mode objfpc}{$H+}
uses
  SysUtils;
type
  TShape = class
  public
    function Area: Double; virtual; abstract;
    function Name: string; virtual; abstract;
  end;

  TRect = class(TShape)
  private
    FW, FH: Double;
  public
    constructor Create(AW, AH: Double);
    function Area: Double; override;
    function Name: string; override;
  end;

  TCircle = class(TShape)
  private
    FR: Double;
  public
    constructor Create(AR: Double);
    function Area: Double; override;
    function Name: string; override;
  end;

constructor TRect.Create(AW, AH: Double);
begin
  FW := AW;
  FH := AH;
end;

function TRect.Area: Double;
begin
  Result := FW * FH;
end;

function TRect.Name: string;
begin
  Result := 'Rect';
end;

constructor TCircle.Create(AR: Double);
begin
  FR := AR;
end;

function TCircle.Area: Double;
begin
  Result := Pi * FR * FR;
end;

function TCircle.Name: string;
begin
  Result := 'Circle';
end;

var
  Shapes: array[0..2] of TShape;
  I: Integer;
  Total: Double;
begin
  Shapes[0] := TRect.Create(3, 4);
  Shapes[1] := TCircle.Create(2);
  Shapes[2] := TRect.Create(5, 5);
  Total := 0;
  try
    for I := 0 to 2 do
    begin
      WriteLn(Shapes[I].Name, ': ', FormatFloat('0.00', Shapes[I].Area));
      Total := Total + Shapes[I].Area;
    end;
    WriteLn('Total: ', FormatFloat('0.00', Total));
  finally
    for I := 0 to 2 do
      Shapes[I].Free;
  end;
end.
