program Lambdas;
{$APPTYPE CONSOLE}
uses
  System.SysUtils;
type
  TAdd = reference to function(A, B: Integer): Integer;
var
  Add: TAdd;
begin
  Add :=
    function(A, B: Integer): Integer
    begin
      Result := A + B;
    end;
  Writeln(Add(2, 3));
  Writeln(Add(10, -4));
end.
