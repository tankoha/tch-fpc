program Digits;
{$mode delphi}

function DigitSum(N: Int64): Int64;
begin
  Result := 0;
  while N > 0 do
  begin
    Inc(Result, N mod 10);
    N := N div 10;
  end;
end;

function DigitalRoot(N: Int64): Int64;
begin
  Result := N;
  while Result >= 10 do
    Result := DigitSum(Result);
end;

procedure Show(N: Int64);
begin
  WriteLn(N, ': sum ', DigitSum(N), ', root ', DigitalRoot(N));
end;

begin
  Show(987654321);
  Show(1000);
  Show(49);
end.
