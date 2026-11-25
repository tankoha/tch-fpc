program Boxes;
{$mode objfpc}{$H+}
type
  generic TBox<T> = record
    Value: T;
  end;
  TIntBox = specialize TBox<Integer>;
  TStrBox = specialize TBox<string>;
var
  A: TIntBox;
  B: TStrBox;
begin
  A.Value := 7;
  B.Value := 'seven';
  WriteLn(A.Value, ' ', B.Value);
end.
