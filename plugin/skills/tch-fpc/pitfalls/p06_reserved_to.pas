{ Pitfall: "to" is a reserved word too. Right: A.&to(TDevice.Cpu).
  expect-error: Syntax error, "identifier" expected but "TO" found }
program p06_reserved_to;
{$mode objfpc}{$H+}
uses Tch;
var
  A: TTensor;
begin
  A := TTensor.ones([2]);
  A := A.to(TDevice.Cpu);
end.
