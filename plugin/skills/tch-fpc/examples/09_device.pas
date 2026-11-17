{ Device-agnostic code: pick the device once, create everything on it.
  On a ROCm build "Cuda" means the AMD GPU. Prints cpu in CI. }
program device;

{$mode objfpc}{$H+}

uses
  SysUtils, Tch;

var
  Dev: TDevice;
  T: TTensor;
begin
  Dev := TDevice.CudaIfAvailable;
  T := TTensor.ones([2, 2]).&to(Dev);
  WriteLn('device ', T.Device.ToString);
  WriteLn('on chosen device: ', T.Device.ToString = Dev.ToString);
  { bring results back to the CPU before reading them element-wise }
  WriteLn('sum = ', T.sum(kNone).&to(TDevice.Cpu).DoubleValue([]):0:3);
end.
