{ Pitfall: in {$mode objfpc} a function parameter cannot be called
  "result" (it is the function result). tch-fpc's generated ops therefore
  name it result_; do the same in your own code. Right: (const Result_: TTensor).
  expect-error: Error: Duplicate identifier "result" }
program p04_param_result;
{$mode objfpc}{$H+}
uses Tch;

function Scaled(const result: TTensor): TTensor;
begin
  Scaled := result * 2.0;
end;

begin
  WriteLn(Scaled(TTensor.ones([1])).DoubleValue([0]):0:1);
end.
