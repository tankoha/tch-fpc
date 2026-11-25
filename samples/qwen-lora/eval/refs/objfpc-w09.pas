program Binary;
{$mode objfpc}{$H+}
uses
  SysUtils;

function ToBinary(N: Integer): string;
begin
  if N = 0 then
    Exit('0');
  Result := '';
  while N > 0 do
  begin
    Result := Chr(Ord('0') + N mod 2) + Result;
    N := N div 2;
  end;
end;

function FromBinary(const S: string): Integer;
var
  I: Integer;
begin
  Result := 0;
  for I := 1 to Length(S) do
    Result := Result * 2 + Ord(S[I]) - Ord('0');
end;

const
  Values: array[0..3] of Integer = (0, 5, 255, 1024);
var
  I: Integer;
begin
  for I := 0 to High(Values) do
    WriteLn(Values[I], ' = ', ToBinary(Values[I]));
  WriteLn(FromBinary('101101'));
end.
