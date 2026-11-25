program Tuples;
{$mode unleashed}

function MinMaxAvg(const A: array of Integer): (lo, hi, avg: Integer);
begin
  var Sum := 0;
  Result := (A[0], A[0], 0);
  for var V in A do
  begin
    if V < Result.lo then Result.lo := V;
    if V > Result.hi then Result.hi := V;
    Sum += V;
  end;
  Result.avg := Sum div Length(A);
end;

begin
  var (lo, hi, avg) := MinMaxAvg([12, 5, 33, 8, 20]);
  WriteLn($'lo={lo} hi={hi} avg={avg}');
end.
