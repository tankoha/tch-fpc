{ Pitfall: FPC 3.2.2 has no anonymous functions; TFuncModule and
  TSequential.AddFn take a named function or method. Right: declare
  function Relu(const X: TTensor): TTensor and pass @Relu.
  expect-error: Error: Illegal expression }
program p05_anonymous_function;
{$mode objfpc}{$H+}
uses Tch, TchNN;
var
  F: TFuncModule;
begin
  F := TFuncModule.Create(function(const X: TTensor): TTensor
    begin
      Result := X.relu;
    end);
  F.Free;
end.
