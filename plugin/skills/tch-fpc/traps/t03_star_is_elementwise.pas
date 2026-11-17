{ Trap: * multiplies element by element (with broadcasting). For the
  matrix product use matmul. }
program t03_star_is_elementwise;

{$mode objfpc}{$H+}

uses
  SysUtils, Tch;

var
  A, B, C: TTensor;
begin
  A := TTensor.FromArray([1.0, 2.0, 3.0, 4.0]).reshape([2, 2]);
  B := TTensor.FromArray([5.0, 6.0, 7.0, 8.0]).reshape([2, 2]);
  C := A * B;
  WriteLn('wrong: (A * B)[0,0] = ', C.DoubleValue([0, 0]):0:3);
  C := A.matmul(B);
  WriteLn('right: A.matmul(B)[0,0] = ', C.DoubleValue([0, 0]):0:3);
end.
