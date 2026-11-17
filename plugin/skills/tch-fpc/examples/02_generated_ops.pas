{ Generated ops: PyTorch names in snake_case, '&' before Pascal reserved
  words, static factories, and the optional-argument helpers. }
program generated_ops;

{$mode objfpc}{$H+}

uses
  SysUtils, Tch;

var
  A, B, R: TTensor;
begin
  { an op without a self tensor is a static (class) function }
  A := TTensor.arange(6, kFloat, TDevice.Cpu).reshape([2, 3]);
  B := TTensor.full([2, 3], 2.0, kFloat, TDevice.Cpu);

  { reserved words take '&': div, to, end, out, ... }
  R := A.&div(B);
  WriteLn('div[1,2] = ', R.DoubleValue([1, 2]):0:3);
  R := A.&to(TDevice.Cpu);
  WriteLn('device ', R.Device.ToString);

  { optional arguments: a plain value, or NoInt64 / NoDouble / NoIntList }
  WriteLn('argmax all = ', A.argmax(NoInt64).Int64Value([]));
  WriteLn('argmax dim 1 = ', A.argmax(1).Int64Value([0]));
  R := A.sum_dim_intlist(IntList([1]));
  WriteLn('row sums = ', R.DoubleValue([0]):0:3, ' ', R.DoubleValue([1]):0:3);

  { trailing-underscore ops work in place and return the same tensor }
  A.add_(B);
  WriteLn('after add_ = ', A.DoubleValue([0, 0]):0:3);
  WriteLn('clamp = ', A.clamp(3, 5).DoubleValue([1, 2]):0:3);
end.
