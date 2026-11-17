{ Autograd: requires_grad, Backward, grad, ZeroGrad, and NoGradGuard kept
  in a local. }
program autograd;

{$mode objfpc}{$H+}

uses
  SysUtils, Tch;

procedure UpdateWithoutGrad(const X: TTensor);
var
  G: IInterface;
begin
  G := NoGradGuard;             // keep it: grad mode is off until G goes
  X.sub_(X.grad * 0.1);
end;

var
  X, Y: TTensor;
begin
  X := TTensor.FromArray([1.0, 2.0, 3.0]).set_requires_grad(True);
  Y := (X * X).sum(kNone);      // y = sum(x^2), dy/dx = 2x
  Y.Backward;
  WriteLn('grad = ', X.grad.DoubleValue([0]):0:3, ' ',
    X.grad.DoubleValue([1]):0:3, ' ', X.grad.DoubleValue([2]):0:3);
  UpdateWithoutGrad(X);
  WriteLn('x[2] after step = ', X.DoubleValue([2]):0:3);
  X.ZeroGrad;
  WriteLn('grad after ZeroGrad = ', X.grad.DoubleValue([2]):0:3);
  WriteLn('requires grad: ', X.RequiresGrad);
end.
