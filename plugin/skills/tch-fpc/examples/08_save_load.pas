{ Variable names in a store, Save and Load, and Freeze. }
program save_load;

{$mode objfpc}{$H+}

uses
  SysUtils, Tch, TchNN;

var
  FileName: String;
  A, B: TVarStore;
  LA, LB: TLinear;
  Vars: TNamedTensorArray;
  X: TTensor;
  I: Integer;
begin
  FileName := GetTempDir + 'tch_fpc_save_load.ot';
  A := TVarStore.Create(TDevice.Cpu);
  B := TVarStore.Create(TDevice.Cpu);
  try
    LA := TLinear.Create(A.Root / 'enc' / 'lin', 2, 2);
    LB := TLinear.Create(B.Root / 'enc' / 'lin', 2, 2);
    try
      Vars := A.Variables;
      for I := 0 to High(Vars) do
        WriteLn('variable ', Vars[I].Name);
      A.Save(FileName);
      B.Load(FileName);                 // names must match
      X := TTensor.ones([1, 2]);
      WriteLn('same output after load: ',
        LA.Forward(X).sub(LB.Forward(X)).abs.max.DoubleValue([]) < 1e-6);
      { Freeze turns requires_grad off; the variables stay trainable ones }
      B.Freeze;
      WriteLn('requires grad after Freeze: ', LB.Forward(X).RequiresGrad);
      B.Unfreeze;
      WriteLn('requires grad after Unfreeze: ', LB.Forward(X).RequiresGrad);
    finally
      LB.Free;
      LA.Free;
    end;
  finally
    B.Free;
    A.Free;
    DeleteFile(FileName);
  end;
end.
