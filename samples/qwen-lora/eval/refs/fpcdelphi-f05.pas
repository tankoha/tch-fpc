program Rect;
{$mode delphi}
type
  TRectangle = class
  private
    FW, FH: Integer;
  public
    constructor Create(AW, AH: Integer);
    function Area: Integer;
  end;

constructor TRectangle.Create(AW, AH: Integer);
begin
  FW := AW;
  FH := AH;
end;

function TRectangle.Area: Integer;
begin
  Result := FW * FH;
end;

var
  R: TRectangle;
begin
  R := TRectangle.Create(3, 4);
  WriteLn('Area: ', R.Area);
  R.Free;
end.
