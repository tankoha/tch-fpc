program InsertSort;
{$mode objfpc}{$H+}
var
  A: array of Integer;
  I, J, Key: Integer;
begin
  A := [42, 7, 19, 3, 25, 11, 8];
  for I := 1 to High(A) do
  begin
    Key := A[I];
    J := I - 1;
    while (J >= 0) and (A[J] > Key) do
    begin
      A[J + 1] := A[J];
      Dec(J);
    end;
    A[J + 1] := Key;
  end;
  for I := 0 to High(A) do
  begin
    if I > 0 then
      Write(',');
    Write(A[I]);
  end;
  WriteLn;
end.
