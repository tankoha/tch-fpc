program SumUp;
{$mode unleashed}
begin
  var Sum := 0;
  for var I := 1 to 5 do
    Sum += I;
  WriteLn($'sum {Sum}');
end.
