program Palindromes;
{$mode objfpc}{$H+}
uses
  SysUtils;

function Reverse(const S: string): string;
var
  I: Integer;
begin
  Result := '';
  for I := Length(S) downto 1 do
    Result := Result + S[I];
end;

function IsPalindrome(const S: string): Boolean;
var
  T: string;
begin
  T := LowerCase(StringReplace(S, ' ', '', [rfReplaceAll]));
  Result := T = Reverse(T);
end;

procedure Show(const S: string);
begin
  if IsPalindrome(S) then
    WriteLn(S, ' -> yes')
  else
    WriteLn(S, ' -> no');
end;

begin
  WriteLn(Reverse('Free Pascal'));
  Show('Never odd or even');
  Show('Lazarus');
  Show('Step on no pets');
end.
