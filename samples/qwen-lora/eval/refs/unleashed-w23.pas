program MergeSort;
{$mode unleashed}
uses
  SysUtils;
type
  TInts = array of Integer;

function Sort(const A: TInts): TInts;
begin
  if Length(A) <= 1 then
    Exit(A);
  var Mid := Length(A) div 2;
  var L := Sort(Copy(A, 0, Mid));
  var R := Sort(Copy(A, Mid, Length(A) - Mid));
  SetLength(Result, Length(A));
  var I := 0;
  var J := 0;
  for var K := 0 to High(Result) do
    if (J > High(R)) or ((I <= High(L)) and (L[I] <= R[J])) then
    begin
      Result[K] := L[I];
      Inc(I);
    end
    else
    begin
      Result[K] := R[J];
      Inc(J);
    end;
end;

begin
  var S := Sort([38, 27, 43, 3, 9, 82, 10]);
  var Line := '';
  for var V in S do
    Line += $'{V} ';
  WriteLn(Line.TrimRight);
end.
