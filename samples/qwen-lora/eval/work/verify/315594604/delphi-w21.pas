program Compose;
{$APPTYPE CONSOLE}
uses
  SysUtils;
type
  TIntFunc = reference to function(X: Integer): Integer;

function ComposeF(const F, G: TIntFunc): TIntFunc;
begin
  Result :=
    function(X: Integer): Integer
    begin
      Result := G(F(X));
    end;
end;

var
  Add1, Square: TIntFunc;
begin
  Add1 :=
    function(X: Integer): Integer
    begin
      Result := X + 1;
    end;
  Square :=
    function(X: Integer): Integer
    begin
      Result := X * X;
    end;
  Writeln(ComposeF(Add1, Square)(3));
  Writeln(ComposeF(Square, Add1)(3));
end.
