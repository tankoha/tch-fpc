program PairSums;
{$mode delphi}
const
  A: array[0..6] of Integer = (1, 4, 6, 9, 11, 15, 20);
var
  I, J, N: Integer;
begin
  N := 0;
  for I := 0 to High(A) - 1 do
    for J := I + 1 to High(A) do
      if A[I] + A[J] = 21 then
      begin
        WriteLn(A[I], ' + ', A[J]);
        Inc(N);
      end;
  WriteLn('pairs: ', N);
end.
