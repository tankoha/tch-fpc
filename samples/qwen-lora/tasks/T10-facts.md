# T10 facts (checked by the harness owner)

T10 is two units: `PackedLinear` (a matrix kept as the GGUF's own bytes on
a device, dequantised to fp32 per call by T03's `DequantizeT`; DESIGN
8.12) and `QwenBlocks` (the trunk blocks of `qwen35` in fp32: RMS norm,
RoPE, the FFN, the linear (Gated DeltaNet) block and the full (gated
attention) block; DESIGN 8.10 "Block skeleton", "Linear block", "Full
block"). A block is a `TStage` of T13's `Recompute`, so T11 and T17 run
the trunk as stages. Every size comes from the GGUF metadata, so the same
code runs the rehearsal 0.8B model (24 blocks, hidden 1024; SSM 16 key =
16 value heads; 8 query / 2 KV heads) and the 27B. LoRA comes with T12,
bf16 with M2.

Measured by the harness owner with a prototype written exactly as below,
on the CPU (GPUs hidden), against a deliberately slow scalar version in
Double (`tests/ut/QwenRefScalar.pas`, written from DESIGN 8.10 with T02's
scalar dequantiser and plain loops): every block of the rehearsal model
within 8.1e-7 (outputs about 4); four blocks chained 8.9e-7; the tiny
model of `tests/ut/QwenBlocksFixture.pas` (2 key / 4 value SSM heads, 4
query / 2 KV heads) within 2.4e-6 at T 70; as `Recompute` stages the input
gradient equals an ordinary backward exactly; on cuda:0 all 24 blocks
within 4.0e-6 of the CPU. Every row of "Unit tests" gives the value in its
row; the messages are the prototype's. These facts are final; do not
investigate further or write probes.

## Unit `PackedLinear` (interface, final; in the skeleton)

```pascal
type
  EPackedLinear = class(Exception);

  TPackedLinear = class
  private
    FRaw: TTensor;
    FTType: Cardinal;
    FOutDim, FInDim: Int64;
  public
    constructor Create(const ARaw: TTensor; ATType: Cardinal; AOutDim, AInDim: Int64);
    function Weight: TTensor;
    function Forward(const X: TTensor): TTensor;
    function Device: TDevice;
    property TType: Cardinal read FTType;
    property OutDim: Int64 read FOutDim;
    property InDim: Int64 read FInDim;
  end;

function TensorRaw(G: TGgufFile; const AName: RawByteString; const ADevice: TDevice;
  out ATType: Cardinal; out ADims: TInt64Array): TTensor;
function LoadPacked(G: TGgufFile; const AName: RawByteString; const ADevice: TDevice): TPackedLinear;
function LoadDense(G: TGgufFile; const AName: RawByteString; const ADevice: TDevice): TTensor;

```

The skeleton's private `ShapeStr(T)` gives `'[2, 34]'`.

## Implementation of `PackedLinear` (write it exactly so)

```pascal
constructor TPackedLinear.Create(const ARaw: TTensor; ATType: Cardinal; AOutDim, AInDim: Int64);
var
  Want: QWord;
begin
  inherited Create;
  if (ARaw.Kind <> kUInt8) or (ARaw.Dim <> 1) then
    raise EPackedLinear.Create('the packed bytes must be a 1-D kUInt8 tensor, got ' + ShapeStr(ARaw));
  if (AOutDim < 1) or (AInDim < 1) then
    raise EPackedLinear.CreateFmt('dimensions must be positive, got %d x %d', [AOutDim, AInDim]);
  Want := GgufTensorBytes(ATType, [QWord(AOutDim * AInDim)]);
  if QWord(ARaw.Shape[0]) <> Want then
    raise EPackedLinear.CreateFmt('%s holds %d bytes, %d x %d of type %s need %d',
      [ShapeStr(ARaw), ARaw.Shape[0], AOutDim, AInDim, GgufTypeName(ATType), Want]);
  FRaw := ARaw;
  FTType := ATType;
  FOutDim := AOutDim;
  FInDim := AInDim;
end;

function TPackedLinear.Weight: TTensor;
begin
  Result := DequantizeT(FRaw, FTType, FOutDim * FInDim).view([FOutDim, FInDim]);
end;

function TPackedLinear.Forward(const X: TTensor): TTensor;
var
  S: TInt64Array;
begin
  S := X.Shape;
  if X.Kind <> kFloat then
    raise EPackedLinear.Create('X must be kFloat (fp32)');
  if (Length(S) = 0) or (S[High(S)] <> FInDim) then
    raise EPackedLinear.CreateFmt('X must end in %d, got %s', [FInDim, ShapeStr(X)]);
  Result := X.matmul(Weight.transpose(0, 1));
end;

function TPackedLinear.Device: TDevice;
begin
  Result := FRaw.Device;
end;

function TensorRaw(G: TGgufFile; const AName: RawByteString; const ADevice: TDevice;
  out ATType: Cardinal; out ADims: TInt64Array): TTensor;
function LoadPacked(G: TGgufFile; const AName: RawByteString; const ADevice: TDevice): TPackedLinear;
function LoadDense(G: TGgufFile; const AName: RawByteString; const ADevice: TDevice): TTensor;

implementation

function ShapeStr(const T: TTensor): string;
var
  S: TInt64Array;
  I: Integer;
begin
  S := T.Shape;
  Result := '[';
  for I := 0 to High(S) do
  begin
    if I > 0 then
      Result := Result + ', ';
    Result := Result + IntToStr(S[I]);
  end;
  Result := Result + ']';
end;
function TensorRaw(G: TGgufFile; const AName: RawByteString; const ADevice: TDevice;
  out ATType: Cardinal; out ADims: TInt64Array): TTensor;
var
  I, J: Integer;
  Info: TGgufTensorInfo;
  Bytes: TBytes;
begin
  I := G.FindTensor(AName);
  if I < 0 then
    raise EPackedLinear.Create('tensor not in the file: ' + AName);
  Info := G.Tensor[I];
  G.ReadTensorBytes(I, Bytes);
  ATType := Info.TType;
  ADims := nil;
  SetLength(ADims, Info.NDims);
  for J := 0 to Info.NDims - 1 do
    ADims[J] := Info.Dims[Info.NDims - 1 - J];
  Result := TTensor.FromBytes(Bytes[0], [Length(Bytes)], kUInt8).&to(ADevice);
end;

function LoadPacked(G: TGgufFile; const AName: RawByteString; const ADevice: TDevice): TPackedLinear;
function LoadDense(G: TGgufFile; const AName: RawByteString; const ADevice: TDevice): TTensor;

implementation

function ShapeStr(const T: TTensor): string;
var
  S: TInt64Array;
  I: Integer;
begin
  S := T.Shape;
  Result := '[';
  for I := 0 to High(S) do
  begin
    if I > 0 then
      Result := Result + ', ';
    Result := Result + IntToStr(S[I]);
  end;
  Result := Result + ']';
end;
function LoadPacked(G: TGgufFile; const AName: RawByteString; const ADevice: TDevice): TPackedLinear;
var
  T: Cardinal;
  D: TInt64Array;
  Raw: TTensor;
begin
  Raw := TensorRaw(G, AName, ADevice, T, D);
  if Length(D) <> 2 then
    raise EPackedLinear.CreateFmt('%s is not a matrix (%d dimensions)', [AName, Length(D)]);
  Result := TPackedLinear.Create(Raw, T, D[0], D[1]);
end;

function LoadDense(G: TGgufFile; const AName: RawByteString; const ADevice: TDevice): TTensor;

implementation

function ShapeStr(const T: TTensor): string;
var
  S: TInt64Array;
  I: Integer;
begin
  S := T.Shape;
  Result := '[';
  for I := 0 to High(S) do
  begin
    if I > 0 then
      Result := Result + ', ';
    Result := Result + IntToStr(S[I]);
  end;
  Result := Result + ']';
end;
function LoadDense(G: TGgufFile; const AName: RawByteString; const ADevice: TDevice): TTensor;
var
  T: Cardinal;
  D: TInt64Array;
  Raw: TTensor;
  N: Int64;
  J: Integer;
begin
  Raw := TensorRaw(G, AName, ADevice, T, D);
  N := 1;
  for J := 0 to High(D) do
    N := N * D[J];
  Result := DequantizeT(Raw, T, N).view(D);
end;

```

`TensorRaw` reverses the GGUF dims (a GGUF matrix lists (In, Out); the
result's `ADims` is [Out, In]); `GgufTensorBytes`, `GgufTypeName`,
`TGgufTensorInfo` are `GgufFile`'s.

## Unit `QwenBlocks` (interface, final; in the skeleton)

```pascal
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
    FAttnNorm, FPostNorm: TTensor;
    FFfnGate, FFfnUp, FFfnDown: TPackedLinear;
    function Attn(const X: TTensor): TTensor; virtual; abstract;
  public
    constructor Load(G: TGgufFile; ABlock: Integer; const ADims: TQwenDims; const ADevice: TDevice); virtual;
    destructor Destroy; override;
    function Device: TDevice; override;
    function Forward(const H: TTensor): TTensor; override;
  end;

  TLinearBlock = class(TQwenBlock)
  protected
    FQkv, FGate, FBeta, FAlpha, FSsmOut: TPackedLinear;
    FDtBias, FSsmA, FConv, FSsmNorm: TTensor;
    function Attn(const X: TTensor): TTensor; override;
  public
    constructor Load(G: TGgufFile; ABlock: Integer; const ADims: TQwenDims; const ADevice: TDevice); override;
    destructor Destroy; override;
  end;

  TFullBlock = class(TQwenBlock)
  protected
    FQ, FK, FV, FOut: TPackedLinear;
    FQNorm, FKNorm: TTensor;
    function Attn(const X: TTensor): TTensor; override;
  public
    constructor Load(G: TGgufFile; ABlock: Integer; const ADims: TQwenDims; const ADevice: TDevice); override;
    destructor Destroy; override;
  end;

function ReadQwenDims(G: TGgufFile): TQwenDims;
function IsFullBlock(const D: TQwenDims; ABlock: Integer): Boolean;
function LoadBlock(G: TGgufFile; ABlock: Integer; const ADims: TQwenDims; const ADevice: TDevice): TQwenBlock;
function RmsNorm(const X, W: TTensor; Eps: Double): TTensor;
function RopeText(const X: TTensor; Rot: Int64; Base: Double; Pos0: Int64 = 0): TTensor;

```

## Implementation of `QwenBlocks` (write it exactly so)

Private helpers and the routines:

```pascal
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
function IsFullBlock(const D: TQwenDims; ABlock: Integer): Boolean;
begin
  Result := (ABlock + 1) mod D.FullInterval = 0;
end;

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
function LoadBlock(G: TGgufFile; ABlock: Integer; const ADims: TQwenDims; const ADevice: TDevice): TQwenBlock;
begin
  if IsFullBlock(ADims, ABlock) then
    Result := TFullBlock.Load(G, ABlock, ADims, ADevice)
  else
    Result := TLinearBlock.Load(G, ABlock, ADims, ADevice);
end;

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
function RmsNorm(const X, W: TTensor; Eps: Double): TTensor;
begin
  Result := X * (X * X).mean_dim(IntList([-1]), True, kNone).add_scalar(Eps).rsqrt * W;
end;

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

```

`TQwenBlock` (base: the norms, the FFN, the residuals):

```pascal
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

```

`TFullBlock` (gated attention):

```pascal
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

```

`TLinearBlock` (Gated DeltaNet):

```pascal
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

```

Notes: `Default(TTensor)` is "no attention mask" for SDPA; `1.0 /
Sqrt(Dh)` converts to its `TOptDouble` scale; `enable_gqa` True maps query
head h to KV head h div (Hq / Hkv). In the linear block, value head j uses
key head j mod Hk (`&repeat` copies whole blocks of heads); the
convolution's index ConvKernel − 1 multiplies the current token; RoPE
angles are computed in kDouble and cast.

## Checks and messages (in this order)

`EPackedLinear`:
- `TPackedLinear.Create`: 1. not a 1-D kUInt8 tensor:
  `'the packed bytes must be a 1-D kUInt8 tensor, got ' + ShapeStr(ARaw)`;
  2. a dimension < 1: `CreateFmt('dimensions must be positive, got %d x %d', [AOutDim, AInDim])`;
  3. byte count: `CreateFmt('%s holds %d bytes, %d x %d of type %s need %d', ...)`,
  e.g. `[34] holds 34 bytes, 2 x 32 of type q8_0 need 68`.
- `Forward`: 1. `X must be kFloat (fp32)`; 2. `CreateFmt('X must end in %d, got %s', ...)`,
  e.g. `X must end in 32, got [3, 31]`.
- `TensorRaw`: `'tensor not in the file: ' + AName`.
- `LoadPacked`: `CreateFmt('%s is not a matrix (%d dimensions)', ...)`,
  e.g. `blk.0.attn_norm.weight is not a matrix (1 dimensions)`.

`EQwenBlock`: `KeyInt` / `KeyFloat`: `'metadata key missing: ' + AKey`
(`ReadQwenDims` reads `qwen35.block_count` first).

## UT fixture and oracle (harness owner's; nobody changes them)

- `tests/ut/QwenBlocksFixture.pas`: `WriteTinyQwen(Path, Seed)` writes the
  tiny model: hidden 64, FFN 96, 4 query / 2 KV heads of 32, Rot 16, base
  10000, SSM 2 key / 4 value heads of 16, kernel 4, `full_attention_interval`
  2: blk.0 linear, blk.1 full; matrices Q8_0 or F16, vectors F32.
- `tests/ut/QwenRefScalar.pas`: `RefDims(G)`, `RefBlock(G, Block, Dims, H, T)`
  (H and the result: T rows of Hidden Doubles), `RefRope(X, T, NHeads,
  HeadDim, Rot, Base, Pos0)`, `RefRms(X, W, Eps)`; `TDoubleArray`.
- `tests/ut/DeltaRuleFixture.pas` (T08's): `MaxAbsDiff`, `ShapeText`,
  `ValuesText`. `tests/ut/TokenizerFixture.pas` (T05's): `WriteTinyGguf(Path)`
  (a GGUF without any `qwen35` key).
- The test skeletons `tests/ut/PackedLinear_test.pas` and
  `tests/ut/QwenBlocks_test.pas` write the tiny model once at start
  (`TinyPath`) and have `Arr(T)` (a tensor's values as `TDoubleArray`) and
  `DiffArr(A, B)` (max |a − b|). Each test frees what it creates
  (`TGgufFile`, blocks, `TPackedLinear`) in `try ... finally`.

## Unit tests (rows; every row has one method, every method belongs to a row)

Error rows catch the unit's exception class and compare the whole message
(`AssertEquals`); a row fails if no exception comes. A tolerance row
passes when the difference is ≤ the limit; "exact" means the exact strings
or a difference of 0. "zeros" is `TTensor.zeros` (kFloat unless a kind is
given). G is `TGgufFile.Open(TinyPath)`; D is `ReadQwenDims(G)`, RD
`RefDims(G)`.

`PackedLinear`, stage 1 (task T10b1), 001–015. "The Q8_0 matrix": V :=
64 Singles `I / 10 - 3` (I = 0..63), Bytes := `QuantizeQ8_0(V)` (unit
`GgufQuant`, 68 bytes), Raw := `TTensor.FromBytes(Bytes[0], [68], kUInt8)`,
L := `TPackedLinear.Create(Raw, 8, 2, 32)`.

| Row | Input | Expected |
|---|---|---|
| 001 | the Q8_0 matrix | OutDim 2, InDim 32, TType 8, `Device.ToString` `'cpu'` |
| 002 | `L.Weight` | shape `[2, 32]`; exact: equal to `DequantizeBytes(8, Bytes, 64)` |
| 003 | `L.Forward(X)`, X := `TTensor.FromArray([1.0, -2.0, 0.5]).view([3, 1]) * TTensor.ones([1, 32])` | shape `[3, 2]`; equal (≤ 1e-5, measured 0) to the loop Σ_k W[j, k] X[i, k] over the `DequantizeBytes` values |
| 004 | `L.Forward(TTensor.ones([2, 5, 32]))` | shape `[2, 5, 2]` |
| 005 | F32: 6 Singles `I - 2.5` as 24 bytes (`Move`), `TPackedLinear.Create(bytes tensor, 0, 2, 3).Weight` | `[2, 3]` `-2.5 -1.5 -0.5 0.5 1.5 2.5` |
| 006 | `Create(TTensor.zeros([34]), 8, 1, 32)` (kFloat) | `the packed bytes must be a 1-D kUInt8 tensor, got [34]` |
| 007 | `Create(TTensor.zeros([2, 34], kUInt8), 8, 1, 32)` | `the packed bytes must be a 1-D kUInt8 tensor, got [2, 34]` |
| 008 | `Create(TTensor.zeros([34], kUInt8), 8, 0, 32)` | `dimensions must be positive, got 0 x 32` |
| 009 | `Create(TTensor.zeros([34], kUInt8), 8, 2, 32)` | `[34] holds 34 bytes, 2 x 32 of type q8_0 need 68` |
| 010 | `Create(zeros([34], kUInt8), 8, 1, 32).Forward(TTensor.zeros([3, 32], kDouble))` | `X must be kFloat (fp32)` |
| 011 | the same `.Forward(TTensor.zeros([3, 31]))` | `X must end in 32, got [3, 31]` |
| 012 | `LoadPacked(G, 'blk.0.ffn_down.weight', TDevice.Cpu)` | OutDim 64, InDim 96, TType 8 |
| 013 | `LoadDense(G, 'blk.0.ssm_conv1d.weight', TDevice.Cpu)` | shape `[128, 4]`, Kind kFloat |
| 014 | `LoadPacked(G, 'blk.9.x', TDevice.Cpu)` | `tensor not in the file: blk.9.x` |
| 015 | `LoadPacked(G, 'blk.0.attn_norm.weight', TDevice.Cpu)` | `blk.0.attn_norm.weight is not a matrix (1 dimensions)` |

`QwenBlocks`, stage 2 (task T10b2), the helpers, 101–107:

| Row | Input | Expected |
|---|---|---|
| 101 | `ReadQwenDims(G)` | Blocks 2, Hidden 64, Ffn 96, HeadsQ 4, HeadsKV 2, HeadDim 32, Rot 16, RopeBase 10000, Eps 1e-6 (delta 1e-12), SsmHeadsK 2, SsmHeadsV 4, SsmDim 16, ConvKernel 4, FullInterval 2 |
| 102 | `ReadQwenDims` of `WriteTinyGguf(GetTempDir + 'tiny-tok.gguf')` opened | `metadata key missing: qwen35.block_count` |
| 103 | D with FullInterval 4; `IsFullBlock(D, I)` for I = 0..7 | F F F T F F F T |
| 104 | `RmsNorm(TTensor.FromArray([3.0, 4.0]).view([1, 2]), TTensor.FromArray([1.0, 2.0]), 1e-6)` | `Arr` vs `RefRms([3.0, 4.0], [1.0, 2.0], 1e-6)` ≤ 1e-6 (measured 4.4e-8; values about 0.8485 2.2627) |
| 105 | `ManualSeed(31)`; X := `TTensor.randn([5, 3, 32])`; `RopeText(X, 16, 10000)` | vs `RefRope(Arr(X), 5, 3, 32, 16, 10000, 0)` ≤ 1e-6 (measured 2.0e-7); exact: `narrow(2, 16, 16)` equals X's |
| 106 | the X of 105; `RopeText(X, 16, 10000, 7)` | vs `RefRope(.., 7)` ≤ 1e-6 (measured 1.7e-7) |
| 107 | `RopeText(X.narrow(0, 0, 1), 16, 10000)` (position 0) | exact: equals `X.narrow(0, 0, 1)` |

`QwenBlocks`, stage 3 (task T10b3), `TQwenBlock` and `TFullBlock`,
201–204. B := `TFullBlock.Load(G, 1, D, TDevice.Cpu)`; H for T rows :=
`ManualSeed(40 + T); TTensor.randn([T, 64])`.

| Row | Input | Expected |
|---|---|---|
| 201 | B; T 8 | `B.Device.ToString` `'cpu'`; `Arr(B.Forward(H))` vs `RefBlock(G, 1, RD, Arr(H), 8)` ≤ 1e-4 (measured 1.0e-6) |
| 202 | B; T 70 | ≤ 1e-4 (measured 1.3e-6) |
| 203 | B; H of T 8 (`ManualSeed(48)`); `B.Forward(H)` and `B.Forward(H.narrow(0, 0, 5))` | the first 5 rows agree ≤ 1e-5 (measured 8.3e-7: causal) |
| 204 | B; H of T 8 as a leaf (`detach.Copy.requires_grad_(True)`); `B.Forward(H).sum(kNone).Backward` | `ShapeText(H.grad)` `[8, 64]`; `H.grad.isfinite.all` true |

`QwenBlocks`, stage 4 (task T10b4), `TLinearBlock` and `LoadBlock`,
211–215. B := `TLinearBlock.Load(G, 0, D, TDevice.Cpu)`; H as above.

| Row | Input | Expected |
|---|---|---|
| 211 | B; T 8 | vs `RefBlock(G, 0, RD, Arr(H), 8)` ≤ 1e-4 (measured 1.2e-6) |
| 212 | B; T 70 (two chunks of the delta rule) | ≤ 1e-4 (measured 6.2e-6) |
| 213 | B; as 203 | ≤ 1e-5 (measured 7.2e-7) |
| 214 | B; as 204 | `[8, 64]`, finite |
| 215 | `LoadBlock(G, 0, D, TDevice.Cpu)` and `LoadBlock(G, 1, ..)` | `ClassName` `'TLinearBlock'` and `'TFullBlock'` |

Build and run (the GPUs hidden as in the gate), for `<P>` = `PackedLinear`
or `QwenBlocks`:
`fpc -Mobjfpc -Sh -vew -l- -Fu{proj}/src -Fu{proj}/../../src -Fu{proj}/tests/ut -FU{proj}/work/bin -Fl{proj}/../../build/libtch-rocm -k-rpath={proj}/../../build/libtch-rocm {proj}/tests/ut/<P>_test.pas -o{proj}/work/bin/<P>_test`
then `HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 {proj}/work/bin/<P>_test --all --format=plain`.
Warnings from other units (`GgufQuant`, `GgufQuantT`, `QwenTokenizer`,
`generics.dictionaries.inc`) and libtorch's GPU warnings on stderr are
expected.

## ITa (prepared by the harness owner)

`QLORA_MODEL=<rehearsal GGUF> {proj}/tests/ita/T10-run.sh` builds
`tests/ita/T10/blocks_check.pas` against `src/`, runs it on the CPU from a
fresh directory outside the project and writes
`{proj}/work/ita/T10/summary.tsv` (case, OK/NG, value, limit, note; then
the driver's exit code and the count of stderr lines other than the GPU
warnings). OK when value ≤ limit. About a minute.

| Case | Compared | Limit |
|---|---|---|
| ITa-T10-001 | every block of the rehearsal model (24), T 8, against the scalar version | 1e-4 |
| ITa-T10-002 | rehearsal blocks 0 (linear) and 3 (full), T 70 | 1e-4 |
| ITa-T10-003 | the tiny model's two blocks, T 70 (tiling, GQA) | 1e-4 |
| ITa-T10-004 | rehearsal blocks 0–3 chained, T 16, against the scalar chain | 1e-4 |
| ITa-T10-005 | rehearsal blocks 0–3 as `Recompute` stages: dL/dX0 against an ordinary backward | 1e-5 |

Measured with the prototype: 8.1e-7, 5.4e-7, 2.4e-6, 8.9e-7, 0. With the
skeleton every case is NG; mutants are caught (key tiling interleaved
instead of repeated; convolution taps reversed; RoPE on adjacent pairs).
Case 101 (every block on cuda:0 against the CPU, 4.0e-6) is the harness
owner's (`T10-run.sh gpu`); agents do not run it.

## Stages and case IDs

- Stage 1 (task T10b1): `PackedLinear` (all of it); rows
  UT-PackedLinear-001 to 015.
- Stage 2 (task T10b2): `KeyInt`, `KeyFloat`, `ReadQwenDims`,
  `IsFullBlock`, `RmsNorm`, `RopeText`; rows UT-QwenBlocks-101 to 107.
- Stage 3 (task T10b3): `TQwenBlock`, `TFullBlock`; rows UT-QwenBlocks-201
  to 204.
- Stage 4 (task T10b4): `TLinearBlock`, `LoadBlock`; rows
  UT-QwenBlocks-211 to 215.
- ITa: ITa-T10-001 to 005.
