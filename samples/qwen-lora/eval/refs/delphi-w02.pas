program Multipliers;
{$APPTYPE CONSOLE}
uses
  System.SysUtils;
type
  TIntFunc = reference to function(X: Integer): Integer;

function MakeMultiplier(N: Integer): TIntFunc;
begin
  Result :=
    function(X: Integer): Integer
    begin
      Result := X * N;
    end;
end;

var
  Doubler, Tripler: TIntFunc;
begin
  Doubler := MakeMultiplier(2);
  Tripler := MakeMultiplier(3);
  Writeln(Doubler(7));
  Writeln(Tripler(7));
  Writeln(Doubler(Tripler(2)));
end.
