program Seasons;
{$mode unleashed}
begin
  for var M := 1 to 12 do
  begin
    var S := match M of
      12, 1, 2: 'winter';
      3..5: 'spring';
      6..8: 'summer';
      _: 'autumn';
    end;
    WriteLn(M, ' ', S);
  end;
end.
