program HexCodec;
{$mode unleashed}
uses
  SysUtils;

function ToHex(const S: string): string;
begin
  Result := '';
  for var C in S do
    Result += IntToHex(Ord(C), 2);
end;

function FromHex(const S: string): string;
begin
  Result := '';
  var I := 1;
  while I < Length(S) do
  begin
    Result += Chr(StrToInt('$' + Copy(S, I, 2)));
    Inc(I, 2);
  end;
end;

begin
  var H := ToHex('Unleashed');
  WriteLn(H);
  WriteLn(FromHex(H));
end.
