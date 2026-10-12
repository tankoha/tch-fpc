{ GPU (DESIGN.md 4.8) against a CUDA or ROCm build of libtorch 2.13.0:
  device moves, autograd, H13 (real device move of a var store), loading
  onto the GPU, JIT, the 4.6.10 RNN numbers and H30, layers through the
  GPU kernels, the mixed-device error. Numbers are the CPU ones of earlier
  groups. On a CPU build it prints one SKIP line and exits 0.

  Build and run (from the repo root, after building build/libtch-<variant>
  as in DESIGN.md 4.8): tests/gpu/run.sh [rocm|cuda|cpu]. run.sh passes the
  variant as the program's argument. }
program gpu_test;

uses
  SysUtils, TchSys, Tch, TchNN, TchJit;

var
  Failures: Integer = 0;
  Gpu: TDevice;

procedure Check(Cond: Boolean; const What: String);
begin
  if Cond then
    WriteLn('  ok    ', What)
  else
  begin
    WriteLn('  FAIL  ', What);
    Inc(Failures);
  end;
end;

function Near(A, B, Tol: Double): Boolean;
begin
  Result := Abs(A - B) <= Tol;
end;

function Values(const T: TTensor): String;
var
  D: TDoubleArray;
  K: Integer;
begin
  D := T.to_device(TDevice.Cpu, T.Kind).ToDoubleArray;
  Result := '';
  for K := 0 to High(D) do
  begin
    if K > 0 then
      Result := Result + ' ';
    Result := Result + FormatFloat('0.####', D[K]);
  end;
end;

function OnGpu(const T: TTensor): TTensor;
begin
  Result := T.to_device(Gpu, T.Kind);
end;

function ToCpu(const T: TTensor): TTensor;
begin
  Result := T.to_device(TDevice.Cpu, T.Kind);
end;

function Scalar0(const T: TTensor): Double;
begin
  Result := ToCpu(T).reshape([-1]).DoubleValue([0]);
end;

{ 1. Capability print, recorded in HANDOFF, not checked }
procedure PrintCapabilities;
const
  Names: array[TGpuBackend] of String = ('none', 'cuda', 'rocm');
var
  S: String;
begin
  WriteLn('capabilities');
  WriteLn('  GpuBackend          ', Names[GpuBackend]);
  WriteLn('  CudaDeviceCount     ', CudaDeviceCount);
  WriteLn('  CudnnIsAvailable    ', CudnnIsAvailable);
  WriteLn('  CudnnUserEnabled    ', CudnnUserEnabled);
  WriteLn('  has_cuda / has_hip  ', at_context_has_cuda, ' / ', at_context_has_hip);
  WriteLn('  has_cudart          ', at_context_has_cudart);
  try
    S := IntToStr(GpuRuntimeVersion);
  except
    on E: ETorchNativeError do S := 'raises: ' + E.Message.Split([#10])[0];
  end;
  WriteLn('  GpuRuntimeVersion   ', S);
  try
    S := IntToStr(CudnnVersion);
  except
    on E: ETorchNativeError do S := 'raises: ' + E.Message.Split([#10])[0];
  end;
  WriteLn('  CudnnVersion        ', S);
end;

{ 1b. The backend detection (H35) and the device facts of the variant
  named by run.sh's first argument }
procedure TestBackend(const Variant: String);
begin
  WriteLn('backend: variant ', Variant);
  Check(CudaDeviceCount = 1, '1b. CudaDeviceCount = 1 with only the discrete GPU visible (H33)');
  if Variant = 'rocm' then
  begin
    Check(GpuBackend = gbRocm, '1b. GpuBackend = gbRocm: CUDA without CUDART (H35)');
    Check(not CudnnIsAvailable, '1b. CudnnIsAvailable False on ROCm (H34 false)');
  end
  else if Variant = 'cuda' then
    Check(GpuBackend = gbCuda, '1b. GpuBackend = gbCuda')
  else
    Check(False, '1b. unknown variant ''' + Variant + '''');
end;

{ 2.-4. Tensors, synchronize, seeds, autograd }
procedure TestTensors;
var
  T, A, B, X: TTensor;
  V: Double;
  Raised: Boolean;
begin
  WriteLn('tensors: device moves, sync, seeds, autograd');
  T := OnGpu(TTensor.FromArray([1.0, 2.0, 3.0]));
  Check(T.Device.IsCuda and (T.Device.ToC = 0) and (T.Device.ToString = 'cuda:0'),
    '2. FromArray(...).to_device(Cuda(0)) is on cuda:0');
  Check(Values(T * 2 + 1) = '3 5 7', '2. (T * 2 + 1) read back: 3 5 7');
  Check(Values(T.sum) = '6', '2. T.sum read back: 6');
  { DESIGN 4.8 expected a refusal; measured: libtorch copies to the host }
  Raised := False;
  V := 0;
  try
    V := T.sum.DoubleValue([]);
  except
    on E: ETorchNativeError do
    begin
      Raised := True;
      WriteLn('  note  DoubleValue on a GPU scalar raises: ', E.Message.Split([#10])[0]);
    end;
  end;
  if not Raised then
    WriteLn('  note  DoubleValue on a GPU scalar returns ', V:0:1);
  Check(not Raised and (V = 6), '2. DoubleValue on a GPU scalar returns 6 (copied to the host)');

  A := TTensor.ones([64, 64], kFloat, Gpu);
  B := A.matmul(A);
  CudaSynchronize;
  Check((Values(B.min) = '64') and (Values(B.max) = '64'), '3. ones(64,64) @ ones: all 64 after CudaSynchronize');
  CudaManualSeedAll(7);
  A := TTensor.randn([4], kFloat, Gpu);
  CudaManualSeedAll(7);
  B := TTensor.randn([4], kFloat, Gpu);
  CudaManualSeed(8);
  Check(A.equal(B) and B.Device.IsCuda, '3. same seed, same randn on the GPU');

  X := OnGpu(TTensor.FromArray([3.0])).set_requires_grad(True);
  (X * X).sum.Backward;
  Check(Values(X.grad) = '6', '4. d(x*x)/dx at 3 on the GPU: 6');
end;

{ 5. H13: TLinear's store moved to the GPU, optimizer created after }
procedure TestH13;
var
  VS: TVarStore;
  Lin: TLinear;
  Opt: TOptimizer;
  Cfg: TLinearConfig;
  NT: TNamedTensorArray;
  AllGpu: Boolean;
  K, Before: Integer;
begin
  WriteLn('H13: var store moved to the GPU');
  VS := TVarStore.Create(TDevice.Cpu);
  Lin := nil;
  Opt := nil;
  try
    Cfg := DefaultLinearConfig;
    Cfg.WsInit := TInit.Constant(1);
    Cfg.HasBsInit := True;
    Cfg.BsInit := TInit.Constant(0);
    Lin := TLinear.Create(VS.Root / 'l', 2, 1, Cfg);
    Before := TchLiveHandles;
    VS.SetDevice(Gpu);
    NT := VS.Variables;
    AllGpu := True;
    for K := 0 to High(NT) do
      AllGpu := AllGpu and NT[K].Tensor.Device.IsCuda;
    Check(VS.Device.IsCuda and (Lin.Weight.Device.ToC = 0) and AllGpu,
      '5. SetDevice(Cuda(0)): store, layer weight and every variable on cuda:0 (H13)');
    Check(TchLiveHandles = Before, '5. the move creates no handles (set_data)');
    Opt := TOptimizer.Sgd(VS, 0.1);
    Opt.BackwardStep(Lin.Forward(OnGpu(TTensor.FromArray([1.0, 2.0]).view([1, 2]))).sum);
    Check(Values(Lin.Weight) = '0.9 0.8', '5. SGD on the GPU: w = [0.9, 0.8] (H13)');
    VS.SetDevice(TDevice.Cpu);
    Check((Lin.Weight.Device.DeviceType = dtCpu) and (Values(Lin.Weight) = '0.9 0.8'),
      '5. SetDevice(Cpu): back with the values intact');
  finally
    Opt.Free;
    Lin.Free;
    VS.Free;
  end;
end;

{ 6. Load onto the GPU }
procedure TestLoad;
var
  VS1, VS2: TVarStore;
  U2: TTensor;
  FileName: String;
  G: IInterface;
begin
  WriteLn('load: a CPU-saved file into a store on the GPU');
  FileName := GetTempDir + Format('tch-fpc-gpu-%d.ot', [GetProcessID]);
  VS1 := TVarStore.Create(TDevice.Cpu);
  VS2 := TVarStore.Create(Gpu);
  try
    G := NoGradGuard;
    VS1.Root.Zeros('u', [3]).add_scalar_(42);
    G := nil;
    VS1.Save(FileName);
    U2 := VS2.Root.Zeros('u', [3]);
    Check(U2.Device.IsCuda, '6. variable created on cuda:0');
    VS2.Load(FileName);
    Check(U2.Device.IsCuda and (Values(U2) = '42 42 42'), '6. Load: values 42 on cuda:0');
  finally
    VS2.Free;
    VS1.Free;
    DeleteFile(FileName);
  end;
end;

{ 7. JIT on the GPU }
procedure TestJit;
var
  M: TJitModule;
  X, Y, R, Ref: TTensor;
  Raised: Boolean;
begin
  WriteLn('jit: foo.pt on the GPU');
  M := TJitModule.Create('tests/jit/models/foo.pt');
  try
    X := TTensor.FromArray([3.0, 1, 4, 1, 5]);
    Y := TTensor.FromArray([7.0]);
    Ref := M.Forward([X, Y]);
    M.ToDevice(Gpu, kFloat);
    R := M.Forward([OnGpu(X), OnGpu(Y)]);
    Check(R.Device.IsCuda and (Values(R) = Values(Ref)),
      '7. foo.pt forward on cuda:0 equals the CPU result: ' + Values(R));
  finally
    M.Free;
  end;
  Raised := False;
  try
    SetJitFuserCudaEnabled(True);
  except
    on ETorchNativeError do Raised := True;
  end;
  Check(Raised and not JitFuserCudaEnabled, '7. SetJitFuserCudaEnabled(True) raises (nvfuser removed)');
end;

{ 8. H30 and layers through the GPU kernels }
procedure TestLayers;
var
  VS: TVarStore;
  G: TGRU;
  L: TLSTM;
  W: TTensor;
  S: TRNNState;
  Conv: TConv2D;
  CCfg: TConvConfig;
  Guard: IInterface;
  Mode: TPaddingMode;
  Got: String;
const
  ModeNames: array[TPaddingMode] of String = ('zeros', 'reflect', 'replicate', 'circular');
  Expected: array[TPaddingMode] of String = ('10 10 10 10', '27 24 21 18', '18 21 24 27',
    '27 24 21 18');
begin
  WriteLn('layers: RNN hand numbers (H30), Conv2D padding modes');
  WriteLn('  note  flatten condition (IsCuda and CudnnIsAvailable): ', CudnnIsAvailable);
  VS := TVarStore.Create(Gpu);
  G := nil;
  L := nil;
  try
    G := TGRU.Create(VS.Root / 'g', 1, 1);
    L := TLSTM.Create(VS.Root / 'l', 1, 1);
    Guard := NoGradGuard;
    for W in G.FlatWeights do
      W.CopyFrom(W.zeros_like);
    for W in L.FlatWeights do
      W.CopyFrom(W.zeros_like);
    G.FlatWeights[2].CopyFrom(OnGpu(TTensor.FromArray([0.0, 0, 1])));
    L.FlatWeights[2].CopyFrom(OnGpu(TTensor.FromArray([0.0, 0, 1, 0])));
    Guard := nil;
    S := G.Step(TTensor.zeros([1, 1], kFloat, Gpu), G.ZeroState(1));
    Check(S.H.Device.IsCuda and Near(Scalar0(S.H), 0.380797, 1e-5),
      Format('8. GRU on the GPU: H = %.6f (0.380797) (H30)', [Scalar0(S.H)]));
    S := L.Step(TTensor.zeros([1, 1], kFloat, Gpu), L.ZeroState(1));
    Check(Near(Scalar0(S.C), 0.380797, 1e-5) and Near(Scalar0(S.H), 0.181700, 1e-5),
      Format('8. LSTM on the GPU: C = %.6f, H = %.6f (0.380797, 0.181700) (H30)',
      [Scalar0(S.C), Scalar0(S.H)]));
  finally
    L.Free;
    G.Free;
    VS.Free;
  end;

  for Mode := Low(TPaddingMode) to High(TPaddingMode) do
  begin
    VS := TVarStore.Create(Gpu);
    Conv := nil;
    try
      CCfg := DefaultConvConfig;
      CCfg.Padding := 1;
      CCfg.Bias := False;
      CCfg.PaddingMode := Mode;
      Conv := TConv2D.Create(VS.Root, 1, 1, 3, CCfg);
      TInit.Constant(1).Apply(Conv.Weight);
      Got := Values(Conv.Forward(OnGpu(TTensor.FromArray([1.0, 2, 3, 4]).view([1, 1, 2, 2]))));
      Check(Got = Expected[Mode], Format('8. conv2d %s on the GPU: %s', [ModeNames[Mode], Got]));
    finally
      Conv.Free;
      VS.Free;
    end;
  end;
end;

{ BatchNorm in training mode: the GPU kernels (MIOpen on ROCm) against
  the CPU ones on the same input and weights }
procedure TestBatchNorm;
var
  VC, VG: TVarStore;
  BC, BG: TBatchNorm;
  X, YC, YG: TTensor;
  D: Double;
  G: IInterface;
begin
  WriteLn('batch norm: training mode, GPU against CPU');
  ManualSeed(5);
  X := TTensor.randn([8, 3, 4, 4]);
  VC := TVarStore.Create(TDevice.Cpu);
  VG := TVarStore.Create(Gpu);
  BC := nil;
  BG := nil;
  try
    BC := TBatchNorm.Create(VC.Root, 2, 3);
    BG := TBatchNorm.Create(VG.Root, 2, 3);
    G := NoGradGuard;
    BG.Weight.CopyFrom(BC.Weight);
    G := nil;
    YC := BC.Forward(X, True);
    YG := BG.Forward(OnGpu(X), True);
    D := (ToCpu(YG) - YC).abs.max.DoubleValue([]);
    Check(YG.Device.IsCuda and (D < 1e-5), Format('8. BatchNorm2d train output, max |GPU - CPU| = %.2e', [D]));
    D := (ToCpu(BG.RunningVar) - BC.RunningVar).abs.max.DoubleValue([]);
    Check(D < 1e-5, Format('8. running_var updated alike, max diff %.2e', [D]));
  finally
    BG.Free;
    BC.Free;
    VG.Free;
    VC.Free;
  end;
end;

{ 9. Mixed devices }
procedure TestMixed;
var
  Msg: String;
  Z: TTensor;
begin
  WriteLn('mixed devices');
  Msg := '';
  try
    Z := TTensor.FromArray([1.0]) + OnGpu(TTensor.FromArray([1.0]));
  except
    on E: ETorchNativeError do Msg := E.Message;
  end;
  Check(Pos('Expected all tensors to be on the same device', Msg) > 0,
    '9. CPU + GPU raises: ' + Msg.Split([#10])[0]);
end;

begin
  if not CudaIsAvailable then
  begin
    WriteLn('SKIP: no GPU in this libtorch');
    Halt(0);
  end;
  Gpu := TDevice.Cuda(0);
  PrintCapabilities;
  TestBackend(ParamStr(1));
  TestTensors;
  TestH13;
  TestLoad;
  TestJit;
  TestLayers;
  TestBatchNorm;
  TestMixed;
  Check(TchLiveHandles = 0, '10. no live handles at the end');
  Check(JitLiveObjects = 0, '10. no live objects at the end');
  if Failures = 0 then
    WriteLn('OK')
  else
  begin
    WriteLn(Failures, ' FAILED');
    Halt(1);
  end;
end.
