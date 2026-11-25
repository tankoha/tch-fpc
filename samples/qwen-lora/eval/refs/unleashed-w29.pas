program Elevator;
{$mode unleashed}
const
  Requests: array[0..3] of Integer = (5, 2, 8, 3);

function Move(AFrom, ATo: Integer): (dest, dist: Integer);
begin
  Result := (ATo, Abs(ATo - AFrom));
end;

begin
  var Floor := 0;
  var Total := 0;
  for var Req in Requests do
  begin
    var (dest, dist) := Move(Floor, Req);
    WriteLn($'{Floor} -> {dest} ({dist} floors)');
    Floor := dest;
    Total += dist;
  end;
  WriteLn($'total {Total}');
end.
