program LargestDemo;
{$mode unleashed}

function Largest<T>(const A: array of T): T;
begin
  Result := A[0];
  for var X in A do
    if X > Result then
      Result := X;
end;

begin
  WriteLn(Largest<Integer>([4, 19, 7]));
  WriteLn(Largest<string>(['pear', 'apple', 'plum']));
end.
