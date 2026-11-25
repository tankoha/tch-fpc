program Todo;
{$mode unleashed}
uses
  SysUtils, Classes;
type
  TTodoList = class
  private
    FItems: TStringList;
  public
    constructor Create;
    destructor Destroy; override;
    procedure Add(const S: string);
    procedure Done(const S: string);
    procedure Print;
  end;

constructor TTodoList.Create;
begin
  inherited Create;
  FItems := TStringList.Create;
end;

destructor TTodoList.Destroy;
begin
  FItems.Free;
  inherited Destroy;
end;

procedure TTodoList.Add(const S: string);
begin
  FItems.Add(S);
end;

procedure TTodoList.Done(const S: string);
begin
  var I := FItems.IndexOf(S);
  if I >= 0 then
    FItems.Delete(I);
end;

procedure TTodoList.Print;
begin
  for var I := 0 to FItems.Count - 1 do
    WriteLn($'{I + 1}. {FItems[I]}');
end;

begin
  var L := autofree TTodoList.Create;
  L.Add('write');
  L.Add('test');
  L.Add('ship');
  L.Done('test');
  L.Print;
end.
