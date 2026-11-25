program Inventory;
{$mode delphi}
uses
  SysUtils, Generics.Collections;
type
  TInventory = class
  private
    FItems: TDictionary<string, Integer>;
  public
    constructor Create;
    destructor Destroy; override;
    procedure Add(const AName: string; AQty: Integer);
    procedure Remove(const AName: string; AQty: Integer);
    procedure Print;
  end;

constructor TInventory.Create;
begin
  FItems := TDictionary<string, Integer>.Create;
end;

destructor TInventory.Destroy;
begin
  FItems.Free;
  inherited;
end;

procedure TInventory.Add(const AName: string; AQty: Integer);
var
  N: Integer;
begin
  if FItems.TryGetValue(AName, N) then
    FItems[AName] := N + AQty
  else
    FItems.Add(AName, AQty);
end;

procedure TInventory.Remove(const AName: string; AQty: Integer);
var
  N: Integer;
begin
  if (not FItems.TryGetValue(AName, N)) or (N < AQty) then
    WriteLn('not enough ', AName)
  else
    FItems[AName] := N - AQty;
end;

procedure TInventory.Print;
var
  Names: TList<string>;
  S: string;
begin
  Names := TList<string>.Create;
  try
    for S in FItems.Keys do
      Names.Add(S);
    Names.Sort;
    for S in Names do
      WriteLn(S, ': ', FItems[S]);
  finally
    Names.Free;
  end;
end;

var
  Inv: TInventory;
begin
  Inv := TInventory.Create;
  try
    Inv.Add('apples', 5);
    Inv.Add('pears', 3);
    Inv.Add('apples', 2);
    Inv.Remove('pears', 1);
    Inv.Remove('apples', 10);
    Inv.Print;
  finally
    Inv.Free;
  end;
end.
