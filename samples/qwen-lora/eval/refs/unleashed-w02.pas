program Classify;
{$mode unleashed}

function Sign(N: Integer): string;
begin
  Result := if N < 0 then 'negative' else if N = 0 then 'zero' else 'positive';
end;

function ModName(N: Integer): string;
begin
  Result := match ((N mod 3) + 3) mod 3 of
    0: 'fizz-ish';
    1: 'one';
    _: 'two';
  end;
end;

begin
  for var N in [-4, 0, 7, 9] do
    WriteLn($'{N}: {Sign(N)}, {ModName(N)}');
end.
