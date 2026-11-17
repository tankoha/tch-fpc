{ Run TorchScript modules: Forward for a tensor result, ForwardIV and
  TIValue for anything else. Needs a tch-fpc checkout: the models are
  tests/jit/models/foo*.pt (from tch-rs), read relative to the repository
  root. }
program jit_run;

{$mode objfpc}{$H+}

uses
  SysUtils, Tch, TchJit;

const
  Models = 'tests/jit/models/';

var
  M: TJitModule;
  R: TTensor;
  V: TIValue;
begin
  { foo.pt: 2x + y + a stored value }
  M := TJitModule.Create(Models + 'foo.pt', TDevice.Cpu);
  try
    R := M.Forward([TTensor.FromArray([3.0, 1, 4]), TTensor.FromArray([7.0])]);
    WriteLn('foo: ', R.DoubleValue([0]):0:3, ' ', R.DoubleValue([1]):0:3, ' ',
      R.DoubleValue([2]):0:3);
  finally
    M.Free;
  end;

  { foo2.pt returns a tuple: Forward would raise, ForwardIV does not }
  M := TJitModule.Create(Models + 'foo2.pt');
  try
    V := M.ForwardIV([TTensor.scalar_tensor(42, kInt64, TDevice.Cpu),
                      TTensor.scalar_tensor(1337, kInt64, TDevice.Cpu)]);
    WriteLn('foo2: ', V.TagName, ' of ', V.Count, ': ',
      V.Item(0).AsTensor.Int64Value([]), ' ', V.Item(1).AsTensor.Int64Value([]));
  finally
    M.Free;
  end;
end.
