{ Pitfall: FromArray has no Int32 overload (it would make integer literals
  ambiguous). Right: TTensor.FromInt32Array(Data).
  expect-error: Incompatible type for arg no. 1: Got "{Dynamic} Array Of LongInt" }
program p03_int32_fromarray;
{$mode objfpc}{$H+}
uses Tch;
var
  Data: array of Int32;
  A: TTensor;
begin
  SetLength(Data, 2);
  A := TTensor.FromArray(Data);
end.
