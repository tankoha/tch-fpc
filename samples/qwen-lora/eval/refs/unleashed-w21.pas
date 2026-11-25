program LeibnizPi;
{$mode unleashed}
uses
  {$ifdef UNIX}cthreads,{$endif} SysUtils;

function Chunk(AFrom, ATo: Integer): Double;
begin
  Result := 0;
  for var K := AFrom to ATo do
    Result += (if Odd(K) then -1.0 else 1.0) / (2 * K + 1);
end;

begin
  var A := async Chunk(0, 249999);
  var B := async Chunk(250000, 499999);
  var C := async Chunk(500000, 749999);
  var D := async Chunk(750000, 999999);
  var Sum := await A;
  Sum += await B;
  Sum += await C;
  Sum += await D;
  var FS := DefaultFormatSettings;
  FS.DecimalSeparator := '.';
  WriteLn('pi ~ ', FormatFloat('0.00000', 4 * Sum, FS));
end.
