program Greeting;
{$mode delphi}
type
  TPerson = class
  private
    FName: string;
  public
    constructor Create(const AName: string);
    function Greet: string;
    property Name: string read FName;
  end;

constructor TPerson.Create(const AName: string);
begin
  FName := AName;
end;

function TPerson.Greet: string;
begin
  Result := 'Hello, ' + FName;
end;

var
  P: TPerson;
begin
  P := TPerson.Create('Grace');
  try
    WriteLn(P.Greet);
  finally
    P.Free;
  end;
end.
