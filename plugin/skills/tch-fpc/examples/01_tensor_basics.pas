{ Create tensors, read shape and kind, use the operators, get scalars out. }
program tensor_basics;

{$mode objfpc}{$H+}

uses
  SysUtils, Tch;

var
  A, B, C: TTensor;
  S: TInt64Array;
begin
  A := TTensor.FromArray([1.0, 2.0, 3.0, 4.0]).reshape([2, 2]);
  B := TTensor.ones([2, 2]);
  S := A.Shape;
  WriteLn('shape ', S[0], 'x', S[1], ', dim ', A.Dim, ', numel ', A.Numel);
  WriteLn('kind float: ', A.Kind = kFloat);
  C := A + B * 2.0;                     // * is elementwise
  WriteLn('C[1,0] = ', C.DoubleValue([1, 0]):0:3);
  WriteLn('sum = ', C.sum(kNone).DoubleValue([]):0:3);
  C := A.matmul(B);                     // the matrix product
  WriteLn('matmul[0,1] = ', C.DoubleValue([0, 1]):0:3);
  WriteLn('int kind: ', TTensor.FromArray([1, 2]).Kind = kInt64);
end.
