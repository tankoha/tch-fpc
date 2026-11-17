{ Trap: Forward's Train defaults to False. Batch norm then uses and keeps
  its running statistics; pass True while training. }
program t06_batchnorm_train_flag;

{$mode objfpc}{$H+}

uses
  SysUtils, Tch, TchNN;

var
  VS: TVarStore;
  BN: TBatchNorm;
  X, Mean: TTensor;
begin
  VS := TVarStore.Create(TDevice.Cpu);
  try
    BN := TBatchNorm.Create(VS.Root / 'bn', 1, 2);
    try
      X := TTensor.FromArray([1.0, 3.0, 5.0, 7.0]).reshape([2, 2]);
      VS.TryGet('bn.running_mean', Mean);
      BN.Forward(X);                    // Train = False
      WriteLn('wrong: running mean after Forward(X) = ', Mean.DoubleValue([0]):0:3);
      BN.Forward(X, True);
      WriteLn('right: running mean after Forward(X, True) = ', Mean.DoubleValue([0]):0:3);
    finally
      BN.Free;
    end;
  finally
    VS.Free;
  end;
end.
