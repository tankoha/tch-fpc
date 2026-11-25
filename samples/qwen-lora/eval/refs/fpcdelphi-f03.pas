program MaxOf;
{$mode delphi}
const
  A: array[0..4] of Integer = (3, 9, 2, 7, 5);
var
  I, M: Integer;
begin
  M := A[0];
  for I := 1 to High(A) do
    if A[I] > M then
      M := A[I];
  WriteLn(M);
  WriteLn('end');
end.
