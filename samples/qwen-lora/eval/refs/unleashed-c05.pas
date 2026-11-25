program Evens;
{$mode unleashed}
begin
  var Total := 0;
  var I := 2;
  while I <= 10 do
  begin
    WriteLn($'even {I}');
    Total += I;
    I += 2;
  end;
  WriteLn($'total {Total}');
end.
