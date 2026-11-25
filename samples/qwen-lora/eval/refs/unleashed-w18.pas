program Perfect;
{$mode unleashed}

begin
  var Found := 0;
  for var N := 2 to 9999 do
  begin
    var Sum := 1;
    var D := 2;
    while D * D <= N do
    begin
      if N mod D = 0 then
      begin
        Sum += D;
        if D <> N div D then
          Sum += N div D;
      end;
      Inc(D);
    end;
    if Sum = N then
    begin
      WriteLn(N);
      Inc(Found);
    end;
  end;
  WriteLn($'found {Found}');
end.
