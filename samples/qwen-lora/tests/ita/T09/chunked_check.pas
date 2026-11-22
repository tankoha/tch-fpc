{ ITa driver of T09 (harness owner's): DeltaRuleChunked against
  DeltaRuleTokens (T08, itself checked against the scalar reference) and
  against the fixture's scalar Double reference RefTokens, at the 27B
  model's sizes. Writes one line per case to stdout: id, value, limit,
  note (tab separated; OK when value <= limit). With the argument "gpu" it
  runs the GPU case instead (harness owner only; agents run with the GPUs
  hidden). }
program chunked_check;

{$mode objfpc}{$H+}

uses
  SysUtils, Math, DateUtils, Tch, DeltaRule, DeltaRuleFixture;

procedure Emit(const Id: string; Value, Limit: Double; const Note: string);
begin
  WriteLn(Id, #9, Format('%.2E', [Value]), #9, Format('%.2E', [Limit]), #9, Note);
end;

{ max of the O and State differences between the chunked and the token form }
function VsTokens(Seed, H, T, D: Int64; GScale: Double; C: Integer; const Dev: TDevice;
  out MsChunked: Int64): Double;
var
  Q, K, V, B, G, O1, S1, O2, S2: TTensor;
  T0: TDateTime;
begin
  RandomInputs(Seed, H, T, D, D, GScale, Q, K, V, B, G);
  O1 := DeltaRuleTokens(Q, K, V, B, G, S1);
  Q := Q.&to(Dev); K := K.&to(Dev); V := V.&to(Dev); B := B.&to(Dev); G := G.&to(Dev);
  T0 := Now;
  O2 := DeltaRuleChunked(Q, K, V, B, G, S2, C);
  O2.abs.max.DoubleValue([]);
  MsChunked := MilliSecondsBetween(Now, T0);
  Result := Max(MaxAbsDiff(O1, O2), MaxAbsDiff(S1, S2));
end;

function VsRef(Seed, H, T, D: Int64; GScale: Double): Double;
var
  Q, K, V, B, G, O, S, RO, RS: TTensor;
begin
  RandomInputs(Seed, H, T, D, D, GScale, Q, K, V, B, G);
  O := DeltaRuleChunked(Q, K, V, B, G, S);
  RefTokens(Q, K, V, B, G, RO, RS);
  Result := Max(MaxAbsDiff(O, RO), MaxAbsDiff(S, RS));
end;

function Grads: Double;
var
  Q, K, V, B, G, Q2, K2, V2, B2, G2, S, W: TTensor;
begin
  RandomInputs(22, 4, 70, 16, 16, 1.0, Q, K, V, B, G);
  ManualSeed(23);
  W := TTensor.randn([4, 70, 16]);
  Q2 := Q.detach.Copy.requires_grad_(True); K2 := K.detach.Copy.requires_grad_(True);
  V2 := V.detach.Copy.requires_grad_(True); B2 := B.detach.Copy.requires_grad_(True);
  G2 := G.detach.Copy.requires_grad_(True);
  Q := Q.detach.Copy.requires_grad_(True); K := K.detach.Copy.requires_grad_(True);
  V := V.detach.Copy.requires_grad_(True); B := B.detach.Copy.requires_grad_(True);
  G := G.detach.Copy.requires_grad_(True);
  (DeltaRuleTokens(Q, K, V, B, G, S) * W).sum(kNone).Backward;
  (DeltaRuleChunked(Q2, K2, V2, B2, G2, S, 16) * W).sum(kNone).Backward;
  Result := Max(Max(MaxAbsDiff(Q.grad, Q2.grad), MaxAbsDiff(K.grad, K2.grad)),
    Max(Max(MaxAbsDiff(V.grad, V2.grad), MaxAbsDiff(B.grad, B2.grad)), MaxAbsDiff(G.grad, G2.grad)));
end;

procedure CpuCases;
var
  T, C: Integer;
  Ms: Int64;
  V: Double;
begin
  V := 0;
  for T in [1, 63, 64, 65, 200] do
    V := Max(V, VsTokens(100 + T, 48, T, 128, 0.1, 64, TDevice.Cpu, Ms));
  Emit('ITa-T09-001', V, 1e-5, 'O and State against DeltaRuleTokens, H 48, 128 x 128, T = 1, 63, 64, 65, 200 (seeds 100 + T)');
  Emit('ITa-T09-002', VsRef(13, 48, 200, 128, 0.1), 1e-5, 'O and State against the scalar Double reference RefTokens, seed 13, H 48, T 200');
  Emit('ITa-T09-003', VsTokens(14, 16, 200, 128, 5.0, 64, TDevice.Cpu, Ms), 1e-5, 'strong decay: g in (-5, 0], seed 14, H 16, T 200');
  V := VsTokens(18, 48, 2048, 128, 0.1, 64, TDevice.Cpu, Ms);
  Emit('ITa-T09-004', V, 1e-5, Format('training length T 2048, H 48: %d ms on the CPU', [Ms]));
  Emit('ITa-T09-005', Grads, 1e-5, 'gradients of all five inputs against the token form, H 4, T 70, chunk 16');
  V := 0;
  for C in [1, 2, 3, 5, 7, 10, 49, 50, 51, 64] do
    V := Max(V, VsTokens(200 + C, 4, 50, 16, 1.0, C, TDevice.Cpu, Ms));
  Emit('ITa-T09-006', V, 1e-5, 'chunk sizes 1, 2, 3, 5, 7, 10, 49, 50, 51, 64 at T 50 (padding and a single chunk)');
end;

procedure GpuCases;
var
  Ms: Int64;
  V: Double;
begin
  if CudaDeviceCount < 1 then
  begin
    Emit('ITa-T09-101', 1, 0, 'needs a GPU');
    Exit;
  end;
  VsTokens(19, 48, 2048, 128, 0.1, 64, TDevice.Cuda(0), Ms);   { warm-up }
  V := VsTokens(19, 48, 2048, 128, 0.1, 64, TDevice.Cuda(0), Ms);
  Emit('ITa-T09-101', V, 1e-5, Format('cuda:0, T 2048, H 48, against the token form on the CPU: %d ms', [Ms]));
end;

begin
  if (ParamCount >= 1) and (ParamStr(1) = 'gpu') then
    GpuCases
  else
    CpuCases;
end.
