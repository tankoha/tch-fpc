{ Trap: a bare "NoGradGuard;" statement releases the guard at once, so grad
  mode stays on. Keep the guard in a local for as long as it must hold. }
program t01_bare_nogradguard;

{$mode objfpc}{$H+}

uses
  SysUtils, Tch;

procedure Wrong(const W: TTensor);
begin
  NoGradGuard;                          // released at the end of the statement
  WriteLn('wrong: result requires grad = ', (W * 2.0).RequiresGrad);
end;

procedure Right(const W: TTensor);
var
  G: IInterface;
begin
  G := NoGradGuard;                     // holds until G goes out of scope
  WriteLn('right: result requires grad = ', (W * 2.0).RequiresGrad);
end;

var
  W: TTensor;
begin
  W := TTensor.ones([2]).set_requires_grad(True);
  Wrong(W);
  Right(W);
end.
