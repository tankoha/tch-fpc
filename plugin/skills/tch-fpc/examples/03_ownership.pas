{ TTensor is a managed record: never Free it. Assignment shares the
  tensor, ShallowClone is a new handle on the same storage, Copy is new
  storage. }
program ownership;

{$mode objfpc}{$H+}

uses
  SysUtils, Tch;

procedure Scale(const T: TTensor);
begin
  { const stops reassigning T, not changing its data }
  T.mul_(TTensor.FromArray([10.0]));
end;

var
  A, B, S, C, N: TTensor;
begin
  A := TTensor.FromArray([1.0, 2.0]);
  B := A;                       // the same tensor
  S := A.ShallowClone;          // new handle, same storage
  C := A.Copy;                  // new storage
  A.add_(TTensor.FromArray([1.0]));
  WriteLn('A=', A.DoubleValue([0]):0:1, ' B=', B.DoubleValue([0]):0:1,
    ' S=', S.DoubleValue([0]):0:1, ' C=', C.DoubleValue([0]):0:1);
  WriteLn('B same handle: ', B.SameHandle(A), ', S same handle: ', S.SameHandle(A));
  Scale(A);
  WriteLn('after Scale: ', A.DoubleValue([1]):0:1);
  { an unassigned TTensor holds no tensor at all }
  WriteLn('N.IsNil = ', N.IsNil);
end.
