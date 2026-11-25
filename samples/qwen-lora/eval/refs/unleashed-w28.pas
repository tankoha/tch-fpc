program Anagrams;
{$mode unleashed}
uses
  SysUtils;

function SortedLetters(const S: string): string;
begin
  Result := S;
  for var I := 1 to Length(Result) - 1 do
    for var J := I + 1 to Length(Result) do
      if Result[J] < Result[I] then
      begin
        var T := Result[I];
        Result[I] := Result[J];
        Result[J] := T;
      end;
end;

procedure Check(const A, B: string);
begin
  WriteLn($'{A} {B}: {if SortedLetters(A) = SortedLetters(B) then 'yes' else 'no'}');
end;

begin
  Check('listen', 'silent');
  Check('triangle', 'integral');
  Check('apple', 'paple');
  Check('rat', 'car');
end.
