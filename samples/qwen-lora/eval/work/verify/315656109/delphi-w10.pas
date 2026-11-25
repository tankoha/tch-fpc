program Money;
{$APPTYPE CONSOLE}
uses
  SysUtils;
type
  TMoney = record
    Cents: Int64;
    class function Make(ACents: Int64): TMoney; static;
    class operator Add(const A, B: TMoney): TMoney;
    class operator Subtract(const A, B: TMoney): TMoney;
    class operator Multiply(const A: TMoney; N: Integer): TMoney;
    function ToString: string;
  end;

class function TMoney.Make(ACents: Int64): TMoney;
begin
  Result.Cents := ACents;
end;

class operator TMoney.Add(const A, B: TMoney): TMoney;
begin
  Result.Cents := A.Cents + B.Cents;
end;

class operator TMoney.Subtract(const A, B: TMoney): TMoney;
begin
  Result.Cents := A.Cents - B.Cents;
end;

class operator TMoney.Multiply(const A: TMoney; N: Integer): TMoney;
begin
  Result.Cents := A.Cents * N;
end;

function TMoney.ToString: string;
begin
  Result := Format('%d.%.2d', [Cents div 100, Cents mod 100]);
end;

begin
  Writeln((TMoney.Make(1999) * 3 - TMoney.Make(500) + TMoney.Make(3)).ToString);
end.
