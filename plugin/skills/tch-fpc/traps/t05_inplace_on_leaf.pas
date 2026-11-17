{ Trap: an in-place op on a leaf tensor that requires grad raises. Update
  parameters by hand only under a NoGradGuard kept in a local. }
program t05_inplace_on_leaf;

{$mode objfpc}{$H+}

uses
  SysUtils, Tch;

procedure Update(const W: TTensor);
var
  G: IInterface;
begin
  G := NoGradGuard;
  W.sub_(TTensor.FromArray([0.5]));
end;

var
  W: TTensor;
begin
  W := TTensor.ones([2]).set_requires_grad(True);
  try
    W.sub_(TTensor.FromArray([0.5]));
  except
    on E: ETorchNativeError do
      WriteLn('wrong: in-place op on a leaf raises ETorchNativeError');
  end;
  Update(W);
  WriteLn('right: W[0] = ', W.DoubleValue([0]):0:3);
end.
