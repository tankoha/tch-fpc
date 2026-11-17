{ Fine-tune a TorchScript module: TTrainableJitModule registers the
  module's parameters in a TVarStore, so an ordinary TOptimizer trains
  them. The module here is scripted from source and saved first. }
program jit_finetune;

{$mode objfpc}{$H+}

uses
  SysUtils, Tch, TchNN, TchJit;

const
  Source =
    'def forward(self, x):'#10 +
    '    return torch.matmul(x, torch.t(self.weight)) + self.bias'#10;

var
  FileName: String;
  M: TJitModule;
  VS: TVarStore;
  J: TTrainableJitModule;
  Opt: TOptimizer;
  X, Y: TTensor;
  Step: Integer;
begin
  FileName := GetTempDir + 'tch_fpc_jit_finetune.pt';
  M := TJitModule.CreateEmpty('Lin');
  try
    M.RegisterParameter('weight', TTensor.zeros([1, 3]).requires_grad_);
    M.RegisterParameter('bias', TTensor.zeros([1]).requires_grad_);
    M.Define(Source);
    M.Save(FileName);
  finally
    M.Free;
  end;

  X := TTensor.FromArray([1.0, 2, 3]).view([1, 3]);
  Y := TTensor.FromArray([1.0]).view([1, 1]);
  VS := TVarStore.Create(TDevice.Cpu);
  J := nil;
  Opt := nil;
  try
    J := TTrainableJitModule.Create(FileName, VS.Root / 'jit');
    WriteLn('store variables: ', VS.Count);
    Opt := TOptimizer.Sgd(VS, 0.02);
    for Step := 1 to 50 do
      Opt.BackwardStep(J.Forward(X, True).mse_loss(Y));
    WriteLn('prediction near 1: ', Abs(J.Forward(X).DoubleValue([0, 0]) - 1) < 1e-3);
  finally
    Opt.Free;          // optimizer first, then the module, then the store
    J.Free;
    VS.Free;
    DeleteFile(FileName);
  end;
end.
