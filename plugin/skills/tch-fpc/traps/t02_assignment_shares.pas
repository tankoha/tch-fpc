{ Trap: B := A does not copy; an in-place op through either name changes
  both. Use Copy for independent data. }
program t02_assignment_shares;

{$mode objfpc}{$H+}

uses
  SysUtils, Tch;

var
  A, B: TTensor;
begin
  A := TTensor.FromArray([1.0, 2.0]);
  B := A;                               // wrong if B should be a backup
  B.mul_(TTensor.FromArray([0.0]));
  WriteLn('wrong: A[1] = ', A.DoubleValue([1]):0:3);

  A := TTensor.FromArray([1.0, 2.0]);
  B := A.Copy;                          // right: new storage
  B.mul_(TTensor.FromArray([0.0]));
  WriteLn('right: A[1] = ', A.DoubleValue([1]):0:3);
end.
