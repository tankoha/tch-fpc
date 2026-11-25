program Squarer;
{$APPTYPE CONSOLE}
uses
  SysUtils;
type
  TIntFunc = reference to function(X: Integer): Integer;
var
  Sq: TIntFunc;
begin
  Sq := function(X: Integer): Integer
    begin
      Result := X * X
    end;
  Writeln(Sq(5));
end.
