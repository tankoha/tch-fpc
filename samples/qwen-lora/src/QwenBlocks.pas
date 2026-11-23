{ T10: the trunk blocks of qwen35 in fp32 (DESIGN 8.10 "Block skeleton",
  "Linear block", "Full block"; 8.12 "QwenBlocks").

  A block is a TStage (unit Recompute): h1 = h + Attn(rms(h, attn_norm)),
  h2 = h1 + ffn_down(silu(ffn_gate(n)) * ffn_up(n)), n = rms(h1,
  post_attention_norm). TLinearBlock is the Gated DeltaNet block (causal
  depthwise convolution, L2-normalised q and k, key heads tiled to the
  value heads, DeltaRule.DeltaRuleChunked); TFullBlock the gated attention
  block (q/k RMS norms, RoPE on the first Rot dimensions of a head, causal
  SDPA with GQA). Every size comes from the GGUF metadata (TQwenDims), so
  the same code runs the rehearsal 0.8B and the 27B model. Target gives a
  block's matrices by their GGUF name, LoraTargets the names DESIGN 8.13
  adapts (T12); bf16 in M2.

  Interface by the harness owner (tasks/T10-facts.md, T12-facts.md); the
  bodies are written in T10b and T12b. }
unit QwenBlocks;

{$mode objfpc}{$H+}

interface

uses
  SysUtils, Math, Tch, GgufFile, PackedLinear, DeltaRule, Recompute;

type
  EQwenBlock = class(Exception);

  TQwenDims = record
    Blocks, Hidden, Ffn: Int64;
    HeadsQ, HeadsKV, HeadDim, Rot: Int64;
    RopeBase, Eps: Double;
    SsmHeadsK, SsmHeadsV, SsmDim, ConvKernel, FullInterval: Int64;
  end;

  { One trunk block: h1 = h + Attn(rms(h)); h2 = h1 + FFN(rms(h1)). }
  TQwenBlock = class(TStage)
  protected
    FDims: TQwenDims;
    FDevice: TDevice;
    FIndex: Integer;
    FAttnNorm, FPostNorm: TTensor;
    FFfnGate, FFfnUp, FFfnDown: TPackedLinear;
    function Attn(const X: TTensor): TTensor; virtual; abstract;
  public
    constructor Load(G: TGgufFile; ABlock: Integer; const ADims: TQwenDims; const ADevice: TDevice); virtual;
    destructor Destroy; override;
    function Device: TDevice; override;
    function Forward(const H: TTensor): TTensor; override;
    { The matrix AName ('ffn_gate', 'ffn_up', 'ffn_down' or one of the
      kind's own, without '.weight'); EQwenBlock for any other name. }
    function Target(const AName: string): TPackedLinear; virtual;
    { The names DESIGN 8.13 adapts in this kind of block. }
    function LoraTargets: TStringArray; virtual; abstract;
    property Index: Integer read FIndex;
  end;

  TLinearBlock = class(TQwenBlock)
  protected
    FQkv, FGate, FBeta, FAlpha, FSsmOut: TPackedLinear;
    FDtBias, FSsmA, FConv, FSsmNorm: TTensor;
    function Attn(const X: TTensor): TTensor; override;
  public
    constructor Load(G: TGgufFile; ABlock: Integer; const ADims: TQwenDims; const ADevice: TDevice); override;
    destructor Destroy; override;
    function Target(const AName: string): TPackedLinear; override;
    function LoraTargets: TStringArray; override;
  end;

  TFullBlock = class(TQwenBlock)
  protected
    FQ, FK, FV, FOut: TPackedLinear;
    FQNorm, FKNorm: TTensor;
    function Attn(const X: TTensor): TTensor; override;
  public
    constructor Load(G: TGgufFile; ABlock: Integer; const ADims: TQwenDims; const ADevice: TDevice); override;
    destructor Destroy; override;
    function Target(const AName: string): TPackedLinear; override;
    function LoraTargets: TStringArray; override;
  end;

function ReadQwenDims(G: TGgufFile): TQwenDims;
function IsFullBlock(const D: TQwenDims; ABlock: Integer): Boolean;
function LoadBlock(G: TGgufFile; ABlock: Integer; const ADims: TQwenDims; const ADevice: TDevice): TQwenBlock;
function RmsNorm(const X, W: TTensor; Eps: Double): TTensor;
function RopeText(const X: TTensor; Rot: Int64; Base: Double; Pos0: Int64 = 0): TTensor;

implementation

function KeyInt(G: TGgufFile; const AKey: RawByteString): Int64;
var
  I: Integer;
begin
  I := G.FindKey(AKey);
  if I < 0 then
    raise EQwenBlock.Create('metadata key missing: ' + AKey);
  Result := G.KV[I].IntVal;
end;

function KeyFloat(G: TGgufFile; const AKey: RawByteString): Double;
var
  I: Integer;
begin
  I := G.FindKey(AKey);
  if I < 0 then
    raise EQwenBlock.Create('metadata key missing: ' + AKey);
  Result := G.KV[I].FloatVal;
end;

function ReadQwenDims(G: TGgufFile): TQwenDims;
begin
  Result.Blocks := KeyInt(G, 'qwen35.block_count');
  Result.Hidden := KeyInt(G, 'qwen35.embedding_length');
  Result.Ffn := KeyInt(G, 'qwen35.feed_forward_length');
  Result.HeadsQ := KeyInt(G, 'qwen35.attention.head_count');
  Result.HeadsKV := KeyInt(G, 'qwen35.attention.head_count_kv');
  Result.HeadDim := KeyInt(G, 'qwen35.attention.key_length');
  Result.Rot := KeyInt(G, 'qwen35.rope.dimension_count');
  Result.RopeBase := KeyFloat(G, 'qwen35.rope.freq_base');
  Result.Eps := KeyFloat(G, 'qwen35.attention.layer_norm_rms_epsilon');
  Result.SsmHeadsK := KeyInt(G, 'qwen35.ssm.group_count');
  Result.SsmHeadsV := KeyInt(G, 'qwen35.ssm.time_step_rank');
  Result.SsmDim := KeyInt(G, 'qwen35.ssm.state_size');
  Result.ConvKernel := KeyInt(G, 'qwen35.ssm.conv_kernel');
  Result.FullInterval := KeyInt(G, 'qwen35.full_attention_interval');
end;

function IsFullBlock(const D: TQwenDims; ABlock: Integer): Boolean;
begin
  Result := (ABlock + 1) mod D.FullInterval = 0;
end;

function LoadBlock(G: TGgufFile; ABlock: Integer; const ADims: TQwenDims; const ADevice: TDevice): TQwenBlock;
begin
  if IsFullBlock(ADims, ABlock) then
    Result := TFullBlock.Load(G, ABlock, ADims, ADevice)
  else
    Result := TLinearBlock.Load(G, ABlock, ADims, ADevice);
end;

function RmsNorm(const X, W: TTensor; Eps: Double): TTensor;
begin
  Result := X * (X * X).mean_dim(IntList([-1]), True, kNone).add_scalar(Eps).rsqrt * W;
end;

function RopeText(const X: TTensor; Rot: Int64; Base: Double; Pos0: Int64): TTensor;
var
  T, D, Half: Int64;
  Pos, Freq, Ang, C, S, X0, X1: TTensor;
begin
  T := X.Shape[0];
  D := X.Shape[2];
  Half := Rot div 2;
  Pos := TTensor.arange_start(Pos0, Pos0 + T, kDouble, X.Device);
  Freq := (TTensor.arange(Half, kDouble, X.Device) * (-2.0 / Rot) * Ln(Base)).exp;
  Ang := Pos.unsqueeze(1) * Freq.unsqueeze(0);
  C := Ang.cos.to_device(X.Device, kFloat).unsqueeze(1);
  S := Ang.sin.to_device(X.Device, kFloat).unsqueeze(1);
  X0 := X.narrow(2, 0, Half);
  X1 := X.narrow(2, Half, Half);
  Result := TTensor.cat([X0 * C - X1 * S, X0 * S + X1 * C, X.narrow(2, Rot, D - Rot)], 2);
end;

constructor TQwenBlock.Load(G: TGgufFile; ABlock: Integer; const ADims: TQwenDims; const ADevice: TDevice);
var
  P: RawByteString;
begin
  inherited Create;
  FDims := ADims;
  FDevice := ADevice;
  P := 'blk.' + IntToStr(ABlock) + '.';
  FAttnNorm := LoadDense(G, P + 'attn_norm.weight', ADevice);
  FPostNorm := LoadDense(G, P + 'post_attention_norm.weight', ADevice);
  FFfnGate := LoadPacked(G, P + 'ffn_gate.weight', ADevice);
  FFfnUp := LoadPacked(G, P + 'ffn_up.weight', ADevice);
  FFfnDown := LoadPacked(G, P + 'ffn_down.weight', ADevice);
end;

function TQwenBlock.Target(const AName: string): TPackedLinear;
begin
  Result := nil;
  raise EQwenBlock.Create('TQwenBlock.Target: not implemented yet');
end;

destructor TQwenBlock.Destroy;
begin
  FFfnGate.Free;
  FFfnUp.Free;
  FFfnDown.Free;
  inherited Destroy;
end;

function TQwenBlock.Device: TDevice;
begin
  Result := FDevice;
end;

function TQwenBlock.Forward(const H: TTensor): TTensor;
var
  H1, N: TTensor;
begin
  H1 := H + Attn(RmsNorm(H, FAttnNorm, FDims.Eps));
  N := RmsNorm(H1, FPostNorm, FDims.Eps);
  Result := H1 + FFfnDown.Forward(FFfnGate.Forward(N).silu * FFfnUp.Forward(N));
end;

constructor TLinearBlock.Load(G: TGgufFile; ABlock: Integer; const ADims: TQwenDims; const ADevice: TDevice);
var
  P: RawByteString;
begin
  inherited Load(G, ABlock, ADims, ADevice);
  P := 'blk.' + IntToStr(ABlock) + '.';
  FQkv := LoadPacked(G, P + 'attn_qkv.weight', ADevice);
  FGate := LoadPacked(G, P + 'attn_gate.weight', ADevice);
  FBeta := LoadPacked(G, P + 'ssm_beta.weight', ADevice);
  FAlpha := LoadPacked(G, P + 'ssm_alpha.weight', ADevice);
  FSsmOut := LoadPacked(G, P + 'ssm_out.weight', ADevice);
  FDtBias := LoadDense(G, P + 'ssm_dt.bias', ADevice);
  FSsmA := LoadDense(G, P + 'ssm_a', ADevice);
  FConv := LoadDense(G, P + 'ssm_conv1d.weight', ADevice);
  FSsmNorm := LoadDense(G, P + 'ssm_norm.weight', ADevice);
end;

destructor TLinearBlock.Destroy;
begin
  FQkv.Free;
  FGate.Free;
  FBeta.Free;
  FAlpha.Free;
  FSsmOut.Free;
  inherited Destroy;
end;

function TLinearBlock.Target(const AName: string): TPackedLinear;
begin
  Result := nil;
  raise EQwenBlock.Create('TLinearBlock.Target: not implemented yet');
end;

function TLinearBlock.LoraTargets: TStringArray;
begin
  Result := nil;
  raise EQwenBlock.Create('TLinearBlock.LoraTargets: not implemented yet');
end;

function TLinearBlock.Attn(const X: TTensor): TTensor;
var
  T, Hk, Hv, Sd, K, C, I: Int64;
  Qkv, Z, Beta, G, Pad, U, Q, Kt, V, O, State: TTensor;
begin
  T := X.Shape[0];
  Hk := FDims.SsmHeadsK;
  Hv := FDims.SsmHeadsV;
  Sd := FDims.SsmDim;
  K := FDims.ConvKernel;
  Qkv := FQkv.Forward(X);
  Z := FGate.Forward(X);
  Beta := FBeta.Forward(X).sigmoid;
  G := (FAlpha.Forward(X) + FDtBias).softplus * FSsmA;
  C := Qkv.Shape[1];
  Pad := TTensor.cat([TTensor.zeros([K - 1, C], kFloat, X.Device), Qkv], 0);
  U := Pad.narrow(0, 0, T) * FConv.select(1, 0);
  for I := 1 to K - 1 do
    U := U + Pad.narrow(0, I, T) * FConv.select(1, I);
  U := U.silu;
  Q := U.narrow(1, 0, Hk * Sd).view([T, Hk, Sd]);
  Kt := U.narrow(1, Hk * Sd, Hk * Sd).view([T, Hk, Sd]);
  V := U.narrow(1, 2 * Hk * Sd, Hv * Sd).view([T, Hv, Sd]);
  Q := Q * (Q * Q).sum_dim_intlist(IntList([-1]), True, kNone).add_scalar(FDims.Eps).rsqrt;
  Kt := Kt * (Kt * Kt).sum_dim_intlist(IntList([-1]), True, kNone).add_scalar(FDims.Eps).rsqrt;
  if Hv <> Hk then
  begin
    Q := Q.&repeat([1, Hv div Hk, 1]);
    Kt := Kt.&repeat([1, Hv div Hk, 1]);
  end;
  Q := Q * (1.0 / Sqrt(Sd));
  O := DeltaRuleChunked(Q.permute([1, 0, 2]), Kt.permute([1, 0, 2]), V.permute([1, 0, 2]),
    Beta.transpose(0, 1).contiguous, G.transpose(0, 1).contiguous, State);
  O := RmsNorm(O.permute([1, 0, 2]), FSsmNorm, FDims.Eps) * Z.view([T, Hv, Sd]).silu;
  Result := FSsmOut.Forward(O.reshape([T, Hv * Sd]));
end;

constructor TFullBlock.Load(G: TGgufFile; ABlock: Integer; const ADims: TQwenDims; const ADevice: TDevice);
var
  P: RawByteString;
begin
  inherited Load(G, ABlock, ADims, ADevice);
  P := 'blk.' + IntToStr(ABlock) + '.';
  FQ := LoadPacked(G, P + 'attn_q.weight', ADevice);
  FK := LoadPacked(G, P + 'attn_k.weight', ADevice);
  FV := LoadPacked(G, P + 'attn_v.weight', ADevice);
  FOut := LoadPacked(G, P + 'attn_output.weight', ADevice);
  FQNorm := LoadDense(G, P + 'attn_q_norm.weight', ADevice);
  FKNorm := LoadDense(G, P + 'attn_k_norm.weight', ADevice);
end;

destructor TFullBlock.Destroy;
begin
  FQ.Free;
  FK.Free;
  FV.Free;
  FOut.Free;
  inherited Destroy;
end;

function TFullBlock.Target(const AName: string): TPackedLinear;
begin
  Result := nil;
  raise EQwenBlock.Create('TFullBlock.Target: not implemented yet');
end;

function TFullBlock.LoraTargets: TStringArray;
begin
  Result := nil;
  raise EQwenBlock.Create('TFullBlock.LoraTargets: not implemented yet');
end;

function TFullBlock.Attn(const X: TTensor): TTensor;
var
  T, Hq, Hkv, Dh: Int64;
  QG, Q, Gate, K, V, A: TTensor;
begin
  T := X.Shape[0];
  Hq := FDims.HeadsQ;
  Hkv := FDims.HeadsKV;
  Dh := FDims.HeadDim;
  QG := FQ.Forward(X).view([T, Hq, 2 * Dh]);
  Q := RmsNorm(QG.narrow(2, 0, Dh), FQNorm, FDims.Eps);
  Gate := QG.narrow(2, Dh, Dh).reshape([T, Hq * Dh]);
  K := RmsNorm(FK.Forward(X).view([T, Hkv, Dh]), FKNorm, FDims.Eps);
  V := FV.Forward(X).view([T, Hkv, Dh]);
  Q := RopeText(Q, FDims.Rot, FDims.RopeBase);
  K := RopeText(K, FDims.Rot, FDims.RopeBase);
  A := TTensor.scaled_dot_product_attention(
         Q.permute([1, 0, 2]).unsqueeze(0), K.permute([1, 0, 2]).unsqueeze(0),
         V.permute([1, 0, 2]).unsqueeze(0), Default(TTensor), 0.0, True,
         1.0 / Sqrt(Dh), True);
  A := A.squeeze_dim(0).permute([1, 0, 2]).reshape([T, Hq * Dh]);
  Result := FOut.Forward(A * Gate.sigmoid);
end;

end.
