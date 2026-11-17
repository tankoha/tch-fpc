{ Train a linear layer: a TVarStore, TLinear, Adam, BackwardStep, and the
  free order (optimizer, layer, store). }
program train_linear;

{$mode objfpc}{$H+}

uses
  SysUtils, Tch, TchNN;

var
  VS: TVarStore;
  Lin: TLinear;
  Opt: TOptimizer;
  X, Y, Loss: TTensor;
  First, Last: Double;
  Step: Integer;
begin
  ManualSeed(42);
  VS := TVarStore.Create(TDevice.Cpu);
  try
    Lin := TLinear.Create(VS.Root / 'lin', 3, 1);
    try
      Opt := TOptimizer.Adam(VS, 0.05);
      try
        { target: y = x0 + 2 x1 - x2 }
        X := TTensor.randn([64, 3]);
        Y := X.matmul(TTensor.FromArray([1.0, 2.0, -1.0]).reshape([3, 1]));
        First := 0;
        for Step := 1 to 300 do
        begin
          Loss := Lin.Forward(X, True).mse_loss(Y);
          if Step = 1 then
            First := Loss.DoubleValue([]);
          Opt.BackwardStep(Loss);
        end;
        Last := Loss.DoubleValue([]);
        WriteLn('variables: ', VS.Count);
        WriteLn('loss fell below 1% of start: ', Last < First * 0.01);
      finally
        Opt.Free;
      end;
    finally
      Lin.Free;
    end;
  finally
    VS.Free;
  end;
end.
