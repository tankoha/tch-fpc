program Vehicles;
{$mode unleashed}
uses
  SysUtils, Generics.Collections;
type
  TVehicle = class
  public
    function Wheels: Integer; virtual; abstract;
    function Describe: string;
  end;

  TCar = class(TVehicle)
    function Wheels: Integer; override;
  end;

  TBike = class(TVehicle)
    function Wheels: Integer; override;
  end;

  TTruck = class(TVehicle)
    function Wheels: Integer; override;
  end;

function TVehicle.Describe: string;
begin
  var N := ClassName.Substring(1);
  Result := $'{N} has {Wheels} wheels';
end;

function TCar.Wheels: Integer;
begin
  Result := 4;
end;

function TBike.Wheels: Integer;
begin
  Result := 2;
end;

function TTruck.Wheels: Integer;
begin
  Result := 6;
end;

begin
  var L := TObjectList<TVehicle>.Create(True);
  defer L.Free;
  L.Add(TCar.Create);
  L.Add(TBike.Create);
  L.Add(TTruck.Create);
  for var V in L do
    WriteLn(V.Describe);
end.
