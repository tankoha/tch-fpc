{ ITa driver of T10 (harness owner's): the trunk blocks against the slow
  scalar version of tests/ut/QwenRefScalar.pas, on the rehearsal model's
  real weights (every block) and on the tiny model of
  tests/ut/QwenBlocksFixture.pas (different key and value head counts).
  Usage: blocks_check <rehearsal GGUF> [gpu]
  Writes one line per case: id, value, limit, note (tab separated; OK when
  value <= limit). With "gpu" it runs the GPU case instead (harness owner
  only; agents run with the GPUs hidden). }
program blocks_check;

{$mode objfpc}{$H+}

uses
  SysUtils, Math, DateUtils, Tch, GgufFile, PackedLinear, QwenBlocks, Recompute,
  QwenRefScalar, QwenBlocksFixture;

procedure Emit(const Id: string; Value, Limit: Double; const Note: string);
begin
  WriteLn(Id, #9, Format('%.2E', [Value]), #9, Format('%.2E', [Limit]), #9, Note);
end;

function Arr(const T: TTensor): TDoubleArray;
var
  F: TTensor;
  I: Integer;
begin
  F := T.to_device(TDevice.Cpu, kDouble).reshape([-1]);
  Result := nil;
  SetLength(Result, F.Shape[0]);
  for I := 0 to High(Result) do
    Result[I] := F.DoubleValue([I]);
end;


function MaxAbsDiff(const A, B: TTensor): Double;
begin
  Result := (A.to_device(TDevice.Cpu, kDouble) - B.to_device(TDevice.Cpu, kDouble)).abs.max.DoubleValue([]);
end;

function DiffArr(const A, B: TDoubleArray): Double;
var
  I: Integer;
begin
  Result := 0;
  for I := 0 to High(A) do
    Result := Max(Result, Abs(A[I] - B[I]));
end;

{ block ABlock on random input of T rows against the scalar version }
function VsRef(G: TGgufFile; const D: TQwenDims; const RD: TRefDims; ABlock, T: Integer): Double;
var
  B: TQwenBlock;
  H: TTensor;
begin
  ManualSeed(1000 + ABlock * 7 + T);
  H := TTensor.randn([T, D.Hidden]);
  B := LoadBlock(G, ABlock, D, TDevice.Cpu);
  try
    Result := DiffArr(Arr(B.Forward(H)), RefBlock(G, ABlock, RD, Arr(H), T));
  finally
    B.Free;
  end;
end;

procedure CpuCases(const Model: string);
var
  G, GT: TGgufFile;
  D: TQwenDims;
  RD: TRefDims;
  I, J: Integer;
  V: Double;
  Tiny: string;
  Stages, Plain: TStageArray;
  Saved: TSavedInputs;
  H, Y, HR, GIn, Leaf, Ref, GOut: TTensor;
  HA: TDoubleArray;

begin
  G := TGgufFile.Open(Model);
  try
    D := ReadQwenDims(G);
    RD := RefDims(G);
    V := 0;
    for I := 0 to D.Blocks - 1 do
      V := Max(V, VsRef(G, D, RD, I, 8));
    Emit('ITa-T10-001', V, 1e-4, Format('every block of the rehearsal model (%d), T 8, against the scalar version', [D.Blocks]));
    Emit('ITa-T10-002', Max(VsRef(G, D, RD, 0, 70), VsRef(G, D, RD, 3, 70)), 1e-4,
      'rehearsal blocks 0 (linear) and 3 (full), T 70 (two chunks of the delta rule)');
    { 004: four blocks chained }
    SetLength(Stages, 4);
    for I := 0 to 3 do
      Stages[I] := LoadBlock(G, I, D, TDevice.Cpu);
    ManualSeed(77);
    H := TTensor.randn([16, D.Hidden]);
    Y := H;
    for I := 0 to 3 do
      Y := Stages[I].Forward(Y);
    HA := Arr(H);
    for I := 0 to 3 do
      HA := RefBlock(G, I, RD, HA, 16);
    { 003 needs the tiny model; keep the IDs in order }
    Tiny := GetTempDir + 'tiny-qwen-ita-' + IntToStr(GetProcessID) + '.gguf';
    WriteTinyQwen(Tiny, 5);
    GT := TGgufFile.Open(Tiny);
    try
      Emit('ITa-T10-003', Max(VsRef(GT, ReadQwenDims(GT), RefDims(GT), 0, 70),
        VsRef(GT, ReadQwenDims(GT), RefDims(GT), 1, 70)), 1e-4,
        'tiny model, T 70: 2 key / 4 value SSM heads (tiling), 4 query / 2 KV heads (GQA)');
    finally
      GT.Free;
      DeleteFile(Tiny);
    end;
    Emit('ITa-T10-004', DiffArr(Arr(Y), HA), 1e-4, 'rehearsal blocks 0-3 chained, T 16, against the scalar chain');
    { 005: the blocks as Recompute stages: dL/dX0 against one ordinary backward }
    ManualSeed(78);
    GOut := TTensor.randn([16, D.Hidden]);
    Saved := TSavedInputs.Create(True);
    try
      RecomputeForward(Stages, H, Saved);
      GIn := RecomputeBackward(Stages, Saved, GOut);
    finally
      Saved.Free;
    end;
    Leaf := H.detach.Copy.requires_grad_(True);
    Ref := Leaf;
    for I := 0 to 3 do
      Ref := Stages[I].Forward(Ref);
    (Ref * GOut).sum(kNone).Backward;
    Emit('ITa-T10-005', MaxAbsDiff(GIn, Leaf.grad), 1e-5,
      Format('rehearsal blocks 0-3 as Recompute stages: dL/dX0 against an ordinary backward (max |grad| %.2f)',
        [Leaf.grad.abs.max.DoubleValue([])]));
    for I := 0 to 3 do
      Stages[I].Free;
  finally
    G.Free;
  end;
end;

procedure GpuCase(const Model: string);
var
  G: TGgufFile;
  D: TQwenDims;
  I: Integer;
  B: TQwenBlock;
  H, Y, YG: TTensor;
  V: Double;

  Ms: Int64;
  T0: TDateTime;
begin
  if CudaDeviceCount < 1 then
  begin
    Emit('ITa-T10-101', 1, 0, 'needs a GPU');
    Exit;
  end;
  G := TGgufFile.Open(Model);
  try
    D := ReadQwenDims(G);
    V := 0;
    Ms := 0;
    for I := 0 to D.Blocks - 1 do
    begin
      ManualSeed(2000 + I);
      H := TTensor.randn([256, D.Hidden]);
      B := LoadBlock(G, I, D, TDevice.Cpu);
      Y := B.Forward(H);
      B.Free;
      B := LoadBlock(G, I, D, TDevice.Cuda(0));
      T0 := Now;
      YG := B.Forward(H.&to(TDevice.Cuda(0)));
      YG.abs.max.DoubleValue([]);
      Ms := Ms + MilliSecondsBetween(Now, T0);
      B.Free;
      V := Max(V, MaxAbsDiff(Y, YG));
    end;
    Emit('ITa-T10-101', V, 1e-3, Format('every block on cuda:0 against the CPU, T 256 (%d ms in all)', [Ms]));
  finally
    G.Free;
  end;
end;

begin
  if (ParamCount >= 2) and (ParamStr(2) = 'gpu') then
    GpuCase(ParamStr(1))
  else
    CpuCases(ParamStr(1));
end.
