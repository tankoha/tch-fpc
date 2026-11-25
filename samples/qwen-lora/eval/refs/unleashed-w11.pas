program Cart;
{$mode unleashed}
uses
  SysUtils, Generics.Collections;
type
  TShoppingCart = class
  private
    FPrices: TList<Integer>;
  public
    property Owner: string; readonly;
    property Discount: Integer;
    constructor Create(const AOwner: string);
    destructor Destroy; override;
    procedure Add(ACents: Integer);
    function TotalCents: Integer;
  end;

constructor TShoppingCart.Create(const AOwner: string);
begin
  inherited Create;
  FOwner := AOwner;
  FPrices := TList<Integer>.Create;
end;

destructor TShoppingCart.Destroy;
begin
  FPrices.Free;
  inherited Destroy;
end;

procedure TShoppingCart.Add(ACents: Integer);
begin
  FPrices.Add(ACents);
end;

function TShoppingCart.TotalCents: Integer;
begin
  var Sum := 0;
  for var P in FPrices do
    Sum += P;
  Result := Sum * (100 - FDiscount) div 100;
end;

begin
  var C := autofree TShoppingCart.Create('Kim');
  C.Discount := 10;
  C.Add(1999);
  C.Add(550);
  C.Add(1250);
  var T := C.TotalCents;
  WriteLn($'{C.Owner} pays {T div 100}.{Format('%.2d', [T mod 100])}');
end.
