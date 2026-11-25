program Hooks;
{$mode unleashed}
type
  THook = procedure(N: Integer) of object;

  TPrinter = class
    procedure Show(N: Integer);
  end;

procedure TPrinter.Show(N: Integer);
begin
  WriteLn('value ', N);
end;

begin
  var P := autofree TPrinter.Create;
  var H: THook := @P.Show;
  for var I := 1 to 3 do
    H(I * 10);
end.
