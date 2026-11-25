program AsyncWork;
{$mode unleashed}
uses
  {$ifdef UNIX}cthreads,{$endif} SysUtils;

function CountPrimes(N: Integer): Integer;
begin
  Result := 0;
  for var I := 2 to N - 1 do
  begin
    var Prime := True;
    var D := 2;
    while D * D <= I do
    begin
      if I mod D = 0 then
      begin
        Prime := False;
        Break;
      end;
      Inc(D);
    end;
    if Prime then
      Inc(Result);
  end;
end;

function Fib(N: Integer): Int64;
begin
  Result := if N < 2 then N else Fib(N - 1) + Fib(N - 2);
end;

begin
  var P := async CountPrimes(20000);
  var F := async Fib(30);
  WriteLn($'primes {await P}');
  WriteLn($'fib {await F}');
end.
