program QuickSort;
{$APPTYPE CONSOLE}
uses
  SysUtils;
type
  TSort = class
  public
    class procedure Sort(var A: TArray<Integer>; Lo, Hi: Integer); static;
  end;

class procedure TSort.Sort(var A: TArray<Integer>; Lo, Hi: Integer);
var
  I, J, Pivot, T: Integer;
begin
  if Lo >= Hi then
    Exit;
  Pivot := A[Hi];
  I := Lo;
  for J := Lo to Hi - 1 do
    if A[J] <= Pivot then
    begin
      T := A[I];
      A[I] := A[J];
      A[J] := T;
      Inc(I);
    end;
  T := A[I];
  A[I] := A[Hi];
  A[Hi] := T;
  Sort(A, Lo, I - 1);
  Sort(A, I + 1, Hi);
end;

var
  A: TArray<Integer>;
  I: Integer;
begin
  A := [29, 3, 72, 15, 8, 41, 3, 56];
  TSort.Sort(A, 0, High(A));
  for I := 0 to High(A) do
  begin
    if I > 0 then
      Write(' ');
    Write(A[I]);
  end;
  Writeln;
end.
