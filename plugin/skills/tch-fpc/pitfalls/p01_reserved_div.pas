{ Pitfall: an op whose PyTorch name is a Pascal reserved word needs '&'.
  Right: A.&div(B).
  expect-error: Syntax error, "identifier" expected but "div" found }
program p01_reserved_div;
{$mode objfpc}{$H+}
uses Tch;
var
  A, B: TTensor;
begin
  A := TTensor.ones([2]);
  B := TTensor.ones([2]);
  A := A.div(B);
end.
