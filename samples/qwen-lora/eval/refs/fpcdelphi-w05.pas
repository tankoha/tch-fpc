program Animals;
{$mode delphi}
uses
  SysUtils, Generics.Collections;
type
  TAnimal = class
  private
    FName: string;
  public
    constructor Create(const AName: string);
    function Sound: string; virtual; abstract;
    property Name: string read FName;
  end;

  TDog = class(TAnimal)
    function Sound: string; override;
  end;

  TCat = class(TAnimal)
    function Sound: string; override;
  end;

  TCow = class(TAnimal)
    function Sound: string; override;
  end;

constructor TAnimal.Create(const AName: string);
begin
  FName := AName;
end;

function TDog.Sound: string;
begin
  Result := 'woof';
end;

function TCat.Sound: string;
begin
  Result := 'meow';
end;

function TCow.Sound: string;
begin
  Result := 'moo';
end;

var
  L: TObjectList<TAnimal>;
  A: TAnimal;
begin
  L := TObjectList<TAnimal>.Create(True);
  try
    L.Add(TDog.Create('Rex'));
    L.Add(TCat.Create('Tom'));
    L.Add(TCow.Create('Bella'));
    for A in L do
      WriteLn(A.Name, ' says ', A.Sound);
  finally
    L.Free;
  end;
end.
