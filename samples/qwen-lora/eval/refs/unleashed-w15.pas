program IShapes;
{$mode unleashed}
uses
  SysUtils;
type
  IShape = interface
    ['{7C2D1E9A-3B4F-4A6C-8D2E-5F1A0B9C8D7E}']
    function Area: Double;
    function Name: string;
  end;

  TSquare = class(TInterfacedObject, IShape)
  private
    FSide: Double;
  public
    constructor Create(ASide: Double);
    function Area: Double;
    function Name: string;
  end;

  TTriangle = class(TInterfacedObject, IShape)
  private
    FBase, FHeight: Double;
  public
    constructor Create(ABase, AHeight: Double);
    function Area: Double;
    function Name: string;
  end;

constructor TSquare.Create(ASide: Double);
begin
  inherited Create;
  FSide := ASide;
end;

function TSquare.Area: Double;
begin
  Result := FSide * FSide;
end;

function TSquare.Name: string;
begin
  Result := 'Square';
end;

constructor TTriangle.Create(ABase, AHeight: Double);
begin
  inherited Create;
  FBase := ABase;
  FHeight := AHeight;
end;

function TTriangle.Area: Double;
begin
  Result := FBase * FHeight / 2;
end;

function TTriangle.Name: string;
begin
  Result := 'Triangle';
end;

var
  Shapes: array[0..1] of IShape;
begin
  Shapes[0] := TSquare.Create(3);
  Shapes[1] := TTriangle.Create(4, 5);
  var FS := DefaultFormatSettings;
  FS.DecimalSeparator := '.';
  for var S in Shapes do
    WriteLn(S.Name, ': ', FormatFloat('0.0', S.Area, FS));
  var Larger := if Shapes[0].Area > Shapes[1].Area then Shapes[0].Name else Shapes[1].Name;
  WriteLn($'larger: {Larger}');
end.
