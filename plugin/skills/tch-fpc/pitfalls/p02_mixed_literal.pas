{ Pitfall: an open-array literal is typed by its first element, so an
  integer followed by a real does not compile. Right: [1.0, 2.5].
  expect-error: Incompatible types: got "Single" expected "Int64" }
program p02_mixed_literal;
{$mode objfpc}{$H+}
uses Tch;
var
  A: TTensor;
begin
  A := TTensor.FromArray([1, 2.5]);
end.
