{ Data in and out: FromBytes, FromInt32Array, CopyData, the To*Array
  functions, and totype. }
program data_in_out;

{$mode objfpc}{$H+}

uses
  SysUtils, Tch;

var
  Buf: array[0..5] of Single = (1.5, 2.5, 3.5, 4.5, 5.5, 6.5);
  Back: array[0..5] of Single;
  T, I: TTensor;
  D: TDoubleArray;
  K: Integer;
begin
  T := TTensor.FromBytes(Buf, [2, 3], kFloat);   // copies the 6 Singles
  WriteLn('shape ', T.Shape[0], 'x', T.Shape[1]);
  T.CopyData(Back, T.Numel);                     // in T's own kind
  WriteLn('back[5] = ', Back[5]:0:3);
  D := T.ToDoubleArray;                          // converted to Double
  WriteLn('D[3] = ', D[3]:0:3, ', length ', Length(D));

  I := TTensor.FromInt32Array([3, 1, 2]);
  WriteLn('int32: ', I.Kind = kInt);
  I := I.totype(kFloat);
  Write('as float:');
  for K := 0 to I.Numel - 1 do
    Write(' ', I.DoubleValue([K]):0:3);
  WriteLn;
end.
