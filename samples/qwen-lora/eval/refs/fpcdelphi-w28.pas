program Isbn;
{$mode delphi}

function Valid(const S: string): Boolean;
var
  I, Sum: Integer;
begin
  Sum := 0;
  for I := 1 to 13 do
    if Odd(I) then
      Inc(Sum, Ord(S[I]) - Ord('0'))
    else
      Inc(Sum, 3 * (Ord(S[I]) - Ord('0')));
  Result := Sum mod 10 = 0;
end;

procedure Check(const S: string);
begin
  if Valid(S) then
    WriteLn(S, ' valid')
  else
    WriteLn(S, ' invalid');
end;

begin
  Check('9780306406157');
  Check('9780306406158');
  Check('9781861972712');
end.
