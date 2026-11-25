program RunLength;
{$mode delphi}
uses
  SysUtils;

function Encode(const S: string): string;
var
  I, N: Integer;
begin
  Result := '';
  I := 1;
  while I <= Length(S) do
  begin
    N := 1;
    while (I + N <= Length(S)) and (S[I + N] = S[I]) do
      Inc(N);
    Result := Result + IntToStr(N) + S[I];
    Inc(I, N);
  end;
end;

function Decode(const S: string): string;
var
  I, N: Integer;
begin
  Result := '';
  N := 0;
  for I := 1 to Length(S) do
    if S[I] in ['0'..'9'] then
      N := N * 10 + Ord(S[I]) - Ord('0')
    else
    begin
      Result := Result + StringOfChar(S[I], N);
      N := 0;
    end;
end;

var
  E: string;
begin
  E := Encode('aaabccddddde');
  WriteLn(E);
  WriteLn(Decode(E));
end.
