program Stairs;
{$mode unleashed}
uses
  SysUtils, Generics.Collections;

var
  Memo: TDictionary<Integer, Int64>;

function Ways(N: Integer): Int64;
begin
  if N < 0 then
    Exit(0);
  if N = 0 then
    Exit(1);
  if Memo.TryGetValue(N, Result) then
    Exit;
  Result := Ways(N - 1) + Ways(N - 2) + Ways(N - 3);
  Memo.Add(N, Result);
end;

begin
  Memo := TDictionary<Integer, Int64>.Create;
  defer Memo.Free;
  for var N in [1, 5, 10, 30] do
    WriteLn($'{N}: {Ways(N)}');
end.
