program Greeters;
{$mode objfpc}{$H+}
type
  IGreeter = interface
    ['{5B0C6C0E-6E3B-4C8A-9A4E-2E0F4B9A7C11}']
    function Greet(const AName: string): string;
  end;

  TEnglishGreeter = class(TInterfacedObject, IGreeter)
    function Greet(const AName: string): string;
  end;

  TGermanGreeter = class(TInterfacedObject, IGreeter)
    function Greet(const AName: string): string;
  end;

function TEnglishGreeter.Greet(const AName: string): string;
begin
  Result := 'Hello, ' + AName + '!';
end;

function TGermanGreeter.Greet(const AName: string): string;
begin
  Result := 'Hallo, ' + AName + '!';
end;

var
  G: array[0..1] of IGreeter;
  I: Integer;
begin
  G[0] := TEnglishGreeter.Create;
  G[1] := TGermanGreeter.Create;
  for I := 0 to 1 do
    WriteLn(G[I].Greet('Ada'));
end.
