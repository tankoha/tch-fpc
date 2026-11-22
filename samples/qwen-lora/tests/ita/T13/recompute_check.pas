{ ITa driver of T13 (harness owner's): Recompute against the
  ordinary whole-graph backward of tests/ut/RecomputeFixture.pas.
  Writes one line per case to stdout: id, value, limit, note (tab
  separated; OK when value <= limit). With the argument "gpu" it runs the
  two-GPU cases instead (harness owner only; agents run with the GPUs
  hidden). }
program recompute_check;

{$mode objfpc}{$H+}

uses
  SysUtils, Math, DateUtils, Tch, Recompute, RecomputeFixture;

procedure Emit(const Id: string; Value, Limit: Double; const Note: string);
begin
  WriteLn(Id, #9, Format('%.2E', [Value]), #9, Format('%.2E', [Limit]), #9, Note);
end;

{ Recompute on A (OnHost as given) and the whole-graph reference on a
  clone B in RefKind on the CPU; returns max of parameter and input
  gradient differences. }
function Parity(Seed: Int64; N: Integer; D, Rows: Int64; AKind: TKind;
  const Devices: array of TDevice; OnHost: Boolean; RefKind: TKind;
  out Ms: Int64): Double;
var
  A, B: TStageArray;
  Saved: TSavedInputs;
  X0, GOut, GIn, RIn: TTensor;
  T0: TDateTime;
begin
  A := MakeStages(Seed, N, D, AKind, Devices);
  B := CloneStages(A, RefKind, [TDevice.Cpu]);
  ManualSeed(Seed + 1000);
  X0 := TTensor.randn([Rows, D], AKind);
  GOut := TTensor.randn([Rows, D], AKind);
  Saved := TSavedInputs.Create(OnHost);
  try
    T0 := Now;
    RecomputeForward(A, X0.&to(A[0].Device), Saved);
    GIn := RecomputeBackward(A, Saved, GOut.&to(A[High(A)].Device));
    Ms := MilliSecondsBetween(Now, T0);
    RIn := WholeGraphBackward(B, X0.to_device(TDevice.Cpu, RefKind), GOut.to_device(TDevice.Cpu, RefKind));
    Result := Max(MaxParamGradDiff(A, B), MaxAbsDiff(GIn, RIn));
  finally
    Saved.Free;
    FreeStages(A);
    FreeStages(B);
  end;
end;

procedure CpuCases;
var
  A, B: TStageArray;
  Saved: TSavedInputs;
  X0, GOut, Y: TTensor;
  Ms: Int64;
  Bad, I: Integer;
  V: Double;
begin
  V := Parity(31, 8, 16, 4, kDouble, [TDevice.Cpu], False, kDouble, Ms);
  Emit('ITa-T13-001', V, 1e-10, 'fp64, 8 stages, width 16, 4 rows, inputs kept on the device');
  V := Parity(32, 64, 64, 32, kDouble, [TDevice.Cpu], True, kDouble, Ms);
  Emit('ITa-T13-002', V, 1e-10, Format('fp64, 64 stages, width 64, 32 rows, inputs on the host, %d ms', [Ms]));
  V := Parity(33, 64, 256, 128, kFloat, [TDevice.Cpu], True, kFloat, Ms);
  Emit('ITa-T13-003', V, 1e-6, Format('fp32 against fp32, 64 stages, width 256, 128 rows, %d ms', [Ms]));

  { 004: the forward builds no graph and calls each stage once; the
    backward calls each stage once more, with a graph }
  A := MakeStages(34, 16, 8, kDouble, [TDevice.Cpu]);
  Saved := TSavedInputs.Create(True);
  try
    ManualSeed(35);
    X0 := TTensor.randn([3, 8], kDouble);
    Y := RecomputeForward(A, X0, Saved);
    Bad := 0;
    if Y.RequiresGrad then Inc(Bad);
    if Saved.Count <> 16 then Inc(Bad);
    for I := 0 to High(A) do
    begin
      if (A[I] as TAffineStage).Calls <> 1 then Inc(Bad);
      if (A[I] as TAffineStage).LastRequiresGrad then Inc(Bad);
      if Saved.Item(I).RequiresGrad then Inc(Bad);
    end;
    RecomputeBackward(A, Saved, TTensor.ones([3, 8], kDouble));
    for I := 0 to High(A) do
    begin
      if (A[I] as TAffineStage).Calls <> 2 then Inc(Bad);
      if not (A[I] as TAffineStage).LastRequiresGrad then Inc(Bad);
    end;
    Emit('ITa-T13-004', Bad, 0, 'count of violations: forward without graph, one call per stage, 16 saved inputs; backward one more call with graph');
  finally
    Saved.Free;
    FreeStages(A);
  end;

  { 005: parameter gradients accumulate: two backwards give twice one
    whole-graph backward }
  A := MakeStages(36, 8, 16, kDouble, [TDevice.Cpu]);
  B := CloneStages(A, kDouble, [TDevice.Cpu]);
  Saved := TSavedInputs.Create(False);
  try
    ManualSeed(37);
    X0 := TTensor.randn([4, 16], kDouble);
    GOut := TTensor.randn([4, 16], kDouble);
    RecomputeForward(A, X0, Saved);
    RecomputeBackward(A, Saved, GOut);
    RecomputeBackward(A, Saved, GOut);
    WholeGraphBackward(B, X0, GOut);
    V := 0;
    for I := 0 to High(A) do
      V := Max(V, MaxAbsDiff((A[I] as TAffineStage).W.grad, (B[I] as TAffineStage).W.grad * 2.0));
    Emit('ITa-T13-005', V, 1e-10, 'two backwards from the same saved inputs: W.grad equals 2 x the reference');
  finally
    Saved.Free;
    FreeStages(A);
    FreeStages(B);
  end;
end;

procedure GpuCases;
var
  Ms: Int64;
  V: Double;
  A: TStageArray;
  Saved: TSavedInputs;
  Bad, I: Integer;
begin
  if CudaDeviceCount < 2 then
  begin
    Emit('ITa-T13-101', 1, 0, Format('needs two GPUs, found %d', [CudaDeviceCount]));
    Exit;
  end;
  V := Parity(41, 8, 64, 32, kFloat, [TDevice.Cuda(0), TDevice.Cuda(1)], True, kDouble, Ms);
  Emit('ITa-T13-101', V, 1e-4, Format('fp32 on cuda:0 (stages 0-3) and cuda:1 (4-7), inputs on the host, against fp64 on the CPU, %d ms', [Ms]));
  V := Parity(42, 64, 512, 256, kFloat, [TDevice.Cuda(0), TDevice.Cuda(1)], False, kDouble, Ms);
  Emit('ITa-T13-102', V, 1e-4, Format('fp32, 64 stages split 32/32, width 512, 256 rows, inputs on the devices, against fp64 on the CPU, %d ms', [Ms]));
  { 103: where the saved inputs live }
  A := MakeStages(43, 4, 8, kFloat, [TDevice.Cuda(0), TDevice.Cuda(1)]);
  Bad := 0;
  Saved := TSavedInputs.Create(True);
  try
    RecomputeForward(A, TTensor.ones([2, 8]).&to(TDevice.Cuda(0)), Saved);
    for I := 0 to 3 do
      if Saved.Item(I).Device.IsCuda then Inc(Bad);
  finally
    Saved.Free;
  end;
  Saved := TSavedInputs.Create(False);
  try
    RecomputeForward(A, TTensor.ones([2, 8]).&to(TDevice.Cuda(0)), Saved);
    for I := 0 to 3 do
      if Saved.Item(I).Device.ToString <> A[I].Device.ToString then Inc(Bad);
  finally
    Saved.Free;
    FreeStages(A);
  end;
  Emit('ITa-T13-103', Bad, 0, 'count of saved inputs on the wrong device (host: CPU; else the stage''s device)');
end;

begin
  if (ParamCount >= 1) and (ParamStr(1) = 'gpu') then
    GpuCases
  else
    CpuCases;
end.
