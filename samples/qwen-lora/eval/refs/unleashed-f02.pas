program MaxIt;
{$mode unleashed}
uses
  SysUtils;

function MaxOf<T>(const A, B: T): T;
begin
  Result := if A > B then A else B;
end;

begin
  WriteLn($'max {MaxOf<Integer>(4, 9)}');
end.
