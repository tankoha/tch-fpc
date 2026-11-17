{ Trap: FromArray([1, 2]) makes an Int64 tensor, and mean on it raises.
  Write real literals, or convert with totype. }
program t04_int_literal_mean;

{$mode objfpc}{$H+}

uses
  SysUtils, Tch;

var
  T: TTensor;
begin
  T := TTensor.FromArray([1, 2]);
  try
    WriteLn(T.mean.DoubleValue([]):0:3);
  except
    on E: ETorchNativeError do
      WriteLn('wrong: mean of Int64 raises ETorchNativeError');
  end;
  WriteLn('right: ', TTensor.FromArray([1.0, 2.0]).mean.DoubleValue([]):0:3);
  WriteLn('right: ', T.totype(kFloat).mean.DoubleValue([]):0:3);
end.
