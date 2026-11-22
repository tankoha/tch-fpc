# Detailed design — units `PackedLinear` and `QwenBlocks` (T10)

Based on the external design `docs/external-design/T10-external.md`, the system
flow `docs/system-flow/T10-system.md` and the facts `tasks/T10-facts.md`
(final, checked by the harness owner). T10 is two units: `PackedLinear`
(a matrix kept as the GGUF's own bytes on a device, dequantised to fp32 per
call by T03's `DequantizeT`; DESIGN 8.12) and `QwenBlocks` (the trunk blocks
of `qwen35` in fp32: RMS norm, RoPE, the FFN, the linear (Gated DeltaNet)
block and the full (gated attention) block; DESIGN 8.10). A block is a
`TStage` of T13's `Recompute`, so T11 and T17 run the trunk as stages. Every
size comes from the GGUF metadata, so the same code runs the rehearsal 0.8B
model (24 blocks, hidden 1024; SSM 16 key = 16 value heads; 8 query / 2 KV
heads) and the 27B. LoRA comes with T12; bf16 with M2.

The complete implementation code below is final, from the facts ("write it
exactly so").

## 1. Module structure

| Source file | Contents |
|-------------|----------|
| `src/PackedLinear.pas` | Unit `PackedLinear`: `TPackedLinear` (a packed matrix), exception `EPackedLinear`, loaders `TensorRaw`, `LoadPacked`, `LoadDense`, private `ShapeStr` |
| `src/QwenBlocks.pas` | Unit `QwenBlocks`: `TQwenDims`, `TQwenBlock` (a trunk block, a `TStage`), `TFullBlock`, `TLinearBlock`, exception `EQwenBlock`, helpers `ReadQwenDims`, `IsFullBlock`, `LoadBlock`, `RmsNorm`, `RopeText`, private `KeyInt` / `KeyFloat` |

| Function | Unit | Purpose |
|----------|------|---------|
| `TPackedLinear.Create` | PackedLinear | Keep the packed bytes on a device with the type and dims; 3 checks (§5) |
| `TPackedLinear.Weight` | PackedLinear | Dequantise the matrix to fp32 [OutDim, InDim] |
| `TPackedLinear.Forward` | PackedLinear | `X · Wᵀ` on a dequantised weight; 2 checks (§5) |
| `TPackedLinear.Device` | PackedLinear | Device of the packed bytes |
| `TensorRaw` / `LoadPacked` / `LoadDense` | PackedLinear | Read a named tensor's bytes off a `TGgufFile` onto a device |
| `KeyInt` / `KeyFloat` (private) | QwenBlocks | Read one GGUF metadata value; raise on missing key |
| `ReadQwenDims` | QwenBlocks | Read the 14 `qwen35.*` keys into a `TQwenDims` |
| `IsFullBlock` / `LoadBlock` | QwenBlocks | Which trunk block is full; load the right class |
| `RmsNorm` / `RopeText` | QwenBlocks | RMS norm over the last dim; RoPE on the first `Rot` dims |
| `TQwenBlock.Load/Destroy/Device/Forward` | QwenBlocks | The block skeleton: norms, FFN, the residuals |
| `TFullBlock.Load/Destroy/Attn` | QwenBlocks | The gated-attention block |
| `TLinearBlock.Load/Destroy/Attn` | QwenBlocks | The Gated DeltaNet (linear) block |

Dependencies: `PackedLinear` needs `GgufFile` (T01: `FindTensor`,
`Tensor`, `ReadTensorBytes`, `GgufTensorBytes`, `GgufTypeName`,
`TGgufTensorInfo`) and `GgufQuantT` (T03: `DequantizeT`). `QwenBlocks` needs
`PackedLinear` and `DeltaRuleChunked` (T09).

## 2. Interface (final, from the facts; in the skeleton)

### Unit `PackedLinear`

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

The unit's private `ShapeStr(T)` gives `'[2, 34]'` for shape [2, 34].

### Unit `QwenBlocks`

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

## 3. Unit `PackedLinear`

A `TPackedLinear` never keeps the dequantised matrix: `FRaw` is the 1-D
kUInt8 byte tensor on the device, dequantised on every `Weight` / `Forward`
call. Tensors are managed records (no `Free`); the class fields holding
`TPackedLinear` are freed in the destructors of the blocks (§4.8, §4.9).

### 3.1 `TPackedLinear.Create`

Arguments: `ARaw` (the packed byte tensor), `ATType` (GGUF tensor type),
`AOutDim` × `AInDim` (the matrix, OutDim rows). Returns a `TPackedLinear`;
the byte tensor is referenced (shared), not copied.

Check order (§5.1): 1-D kUInt8, positive dims, byte count against
`GgufTensorBytes(ATType, [QWord(AOutDim * AInDim)])`.

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
```

### 3.2 `TPackedLinear.Weight`

`DequantizeT(FRaw, FTType, FOutDim * FInDim)` (T03) dequantises the whole
matrix to a 1-D kFloat [OutDim × InDim]; `.view([OutDim, InDim])` reshapes it
to the matrix:

```pascal
function TPackedLinear.Weight: TTensor;
begin
  Result := DequantizeT(FRaw, FTType, FOutDim * FInDim).view([FOutDim, FInDim]);
end;
```

### 3.3 `TPackedLinear.Forward`

Check order (§5.2): `X` kFloat; `X` ends in `InDim` (rank ≥ 1). Then
`X · Wᵀ`: `W` is [OutDim, InDim], transposed to [InDim, OutDim].

```pascal
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
```

### 3.4 `TPackedLinear.Device`

```pascal
function TPackedLinear.Device: TDevice;
begin
  Result := FRaw.Device;
end;
```

### 3.5 `TensorRaw`

`TensorRaw(G, AName, ADevice)`: find the named tensor, read its raw bytes,
report type and the GGUF dims **reversed** (a GGUF matrix lists (In, Out);
the result's `ADims` is [Out, In]), and build the 1-D kUInt8 byte tensor on
the target device.

```pascal
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
```

**Tensor ops:**
1. `G.FindTensor(AName)` — the tensor's index; `< 0` raises
   `EPackedLinear` (§5.3)
2. `G.Tensor[I]` — `TGgufTensorInfo` (the type `TType` and the GGUF dims)
3. `G.ReadTensorBytes(I, Bytes)` — the tensor's bytes exactly as stored
   (still quantised)
4. `ATType := Info.TType`
5. `ADims := nil; SetLength(ADims, Info.NDims); ADims[J] := Info.Dims[Info.NDims - 1 - J]`
   — dims reversed: a GGUF matrix lists (In, Out); `ADims` is [Out, In]
6. `TTensor.FromBytes(Bytes[0], [Length(Bytes)], kUInt8)` — 1-D kUInt8 byte
   tensor; `.&to(ADevice)` — move it onto the target device → Result

### 3.6 `LoadPacked`

Load a matrix tensor as a `TPackedLinear`: bytes stay packed; `D[0]` × `D[1]`
are [OutDim, InDim].

```pascal
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
```

**Tensor ops:**
1. `TensorRaw(G, AName, ADevice, T, D)` — the 1-D kUInt8 byte tensor, the
   type `T`, the reversed GGUF dims `D`
2. `Length(D) <> 2` — not a matrix: raise `EPackedLinear` (§5.4)
3. `TPackedLinear.Create(Raw, T, D[0], D[1])` — runs the three create
   checks (§5.1): kind/dims, positive dims, byte count

### 3.7 `LoadDense`

Load a non-matrix tensor (norm weights, biases) dequantised to fp32, with
its GGUF shape (reversed dims):

```pascal
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

**Tensor ops:**
1. `TensorRaw(G, AName, ADevice, T, D)` — bytes, type, reversed dims
2. `N := 1; for J := 0 to High(D) do N := N * D[J];` — element count
3. `DequantizeT(Raw, T, N)` (T03) — the whole tensor dequantised to 1-D
   kFloat [N] on `ADevice`
4. `.view(D)` — reshape to the GGUF shape (reversed dims) → Result

### 3.8 `ShapeStr` (private)

```pascal
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
```

## 4. Unit `QwenBlocks`

### 4.0 `KeyInt` and `KeyFloat` (private)

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
```

A missing key raises `EQwenBlock` (§5.5); `ReadQwenDims` reads
`qwen35.block_count` first, so that is the key a keyless file reports.

### 4.1 `ReadQwenDims`

Reads the 14 `qwen35.*` metadata keys in this order:

| # | Key | Field |
|---|-----|-------|
| 1 | `qwen35.block_count` | `Blocks` |
| 2 | `qwen35.embedding_length` | `Hidden` |
| 3 | `qwen35.feed_forward_length` | `Ffn` |
| 4 | `qwen35.attention.head_count` | `HeadsQ` |
| 5 | `qwen35.attention.head_count_kv` | `HeadsKV` |
| 6 | `qwen35.attention.key_length` | `HeadDim` |
| 7 | `qwen35.rope.dimension_count` | `Rot` |
| 8 | `qwen35.rope.freq_base` | `RopeBase` (float) |
| 9 | `qwen35.attention.layer_norm_rms_epsilon` | `Eps` (float) |
| 10 | `qwen35.ssm.group_count` | `SsmHeadsK` |
| 11 | `qwen35.ssm.time_step_rank` | `SsmHeadsV` |
| 12 | `qwen35.ssm.state_size` | `SsmDim` |
| 13 | `qwen35.ssm.conv_kernel` | `ConvKernel` |
| 14 | `qwen35.full_attention_interval` | `FullInterval` |

```pascal
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
```

### 4.2 `IsFullBlock`

Block `ABlock` is a full (gated attention) block when
`(ABlock + 1) mod FullInterval = 0`:

```pascal
function IsFullBlock(const D: TQwenDims; ABlock: Integer): Boolean;
begin
  Result := (ABlock + 1) mod D.FullInterval = 0;
end;
```

### 4.3 `LoadBlock`

```pascal
function LoadBlock(G: TGgufFile; ABlock: Integer; const ADims: TQwenDims; const ADevice: TDevice): TQwenBlock;
begin
  if IsFullBlock(ADims, ABlock) then
    Result := TFullBlock.Load(G, ABlock, ADims, ADevice)
  else
    Result := TLinearBlock.Load(G, ABlock, ADims, ADevice);
end;
```

### 4.4 `RmsNorm`

RMS normalisation over the last dim with a per-unit weight `W`:
`X / sqrt(mean(X²) + Eps) * W`, with the mean kept for the last dim only
(keepdim) and no type promotion (`kNone`):

```pascal
function RmsNorm(const X, W: TTensor; Eps: Double): TTensor;
begin
  Result := X * (X * X).mean_dim(IntList([-1]), True, kNone).add_scalar(Eps).rsqrt * W;
end;
```

**Tensor ops:**
1. `X * X` — element-wise squares
2. `.mean_dim(IntList([-1]), True, kNone)` — mean over the last dim,
   keepdim, no type promotion
3. `.add_scalar(Eps)` — the epsilon inside the mean
4. `.rsqrt` — 1 / sqrt(·)
5. `X * (…)` — scale `X`; left-associative, then
6. `* W` — the per-unit weight → Result

### 4.5 `RopeText`

Rotary embedding on the first `Rot` dims of the last axis (the head dim),
for a [T, heads, head-dim] tensor; positions start at `Pos0` (default 0).
Angles in kDouble, cast to kFloat:

```pascal
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

**Tensor ops:**
1. `T := X.Shape[0]`, `D := X.Shape[2]`, `Half := Rot div 2`
2. `TTensor.arange_start(Pos0, Pos0 + T, kDouble, X.Device)` — positions
   `[Pos0, Pos0+T)`, shape [T], kDouble
3. `TTensor.arange(Half, kDouble, X.Device) * (-2.0 / Rot) * Ln(Base)`,
   `.exp` — `freq[i] = Base^(-2i / Rot)`, shape [Half], kDouble
4. `Pos.unsqueeze(1) * Freq.unsqueeze(0)` — angles [T, Half], kDouble
5. `Ang.cos` / `Ang.sin` — [T, Half], kDouble; `.to_device(X.Device, kFloat)`
   casts to fp32; `.unsqueeze(1)` → [T, 1, Half] to broadcast over heads
6. `X.narrow(2, 0, Half)` → `X0`; `X.narrow(2, Half, Half)` → `X1`
7. `X0 * C - X1 * S`; `X0 * S + X1 * C` — the rotation of the first `Rot`
   dims; `X.narrow(2, Rot, D - Rot)` — the remaining `D − Rot` dims, passed
   through unchanged
8. `TTensor.cat([...], 2)` — join along the head dim → Result

At `Pos0 = 0` the first token's angles are 0 (C = 1, S = 0), so the first
row of `X` comes back exactly unchanged (UT-QwenBlocks-107).

### 4.6 `TQwenBlock` — the skeleton (norms, FFN, residuals)

Fields: `FDims`, `FDevice`, the two norms `FAttnNorm` / `FPostNorm` (dense
tensors, `LoadDense`), the three FFN packed linears `FFfnGate` / `FFfnUp` /
`FFfnDown`, and the abstract `Attn`.

**Load** — name is `'blk.' + ABlock + '.'`; norms first (dense), then the
FFN (packed):

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
```

**Destroy** — the three FFN `TPackedLinear`s, then `inherited`:

```pascal
destructor TQwenBlock.Destroy;
begin
  FFfnGate.Free;
  FFfnUp.Free;
  FFfnDown.Free;
  inherited Destroy;
end;
```

**Device:**

```pascal
function TQwenBlock.Device: TDevice;
begin
  Result := FDevice;
end;
```

**Forward** — `h1 = h + Attn(rms(h)); h2 = h1 + FFN(rms(h1))`:

```pascal
function TQwenBlock.Forward(const H: TTensor): TTensor;
var
  H1, N: TTensor;
begin
  H1 := H + Attn(RmsNorm(H, FAttnNorm, FDims.Eps));
  N := RmsNorm(H1, FPostNorm, FDims.Eps);
  Result := H1 + FFfnDown.Forward(FFfnGate.Forward(N).silu * FFfnUp.Forward(N));
end;
```

**Tensor ops:**
1. `RmsNorm(H, FAttnNorm, FDims.Eps)` — the attention input norm (§4.4)
2. `Attn(…)` — virtual: `TFullBlock.Attn` (§4.8) or `TLinearBlock.Attn`
   (§4.9)
3. `H1 := H + Attn(…)` — the first residual
4. `N := RmsNorm(H1, FPostNorm, FDims.Eps)` — the FFN input norm
5. `FFfnGate.Forward(N).silu * FFfnUp.Forward(N)` — the gated linear
6. `FFfnDown.Forward(…)` — down-projection
7. `Result := H1 + …` — the second residual

### 4.7 `TFullBlock` — the full (gated attention) block

Fields: `FQ`, `FK`, `FV`, `FOut` (packed), `FQNorm`, `FKNorm` (dense).

**Load** — inherits the skeleton's load (norms + FFN), then the attention
weights:

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
```

**Destroy** — the four packed linears, then `inherited` (the three FFN):

```pascal
destructor TFullBlock.Destroy;
begin
  FQ.Free;
  FK.Free;
  FV.Free;
  FOut.Free;
  inherited Destroy;
end;
```

**Attn** — the gate is the second half of the Q projection; Q and K get
per-head RMS norm and RoPE; scaled dot-product attention with GQA and a
causal mask; the output gate, then the output linear:

```pascal
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

**Tensor ops:**
1. `T := X.Shape[0]`; `Hq`/`Hkv`/`Dh` from `FDims`
2. `FQ.Forward(X)` — [T, HeadsQ × 2·HeadDim]; `.view([T, Hq, 2 * Dh])` —
   query half and gate half fused in one projection → `QG`
3. `QG.narrow(2, 0, Dh)` → the query half; `RmsNorm(…, FQNorm, FDims.Eps)` —
   per-head RMS norm → `Q`
4. `QG.narrow(2, Dh, Dh).reshape([T, Hq * Dh])` — the gate half → `Gate`
5. `FK.Forward(X).view([T, Hkv, Dh])`; `RmsNorm(…, FKNorm, FDims.Eps)` → `K`
6. `FV.Forward(X).view([T, Hkv, Dh])` → `V`
7. `RopeText(Q, Rot, RopeBase)` / `RopeText(K, Rot, RopeBase)` — positions
   start at 0 (§4.5)
8. `Q.permute([1, 0, 2]).unsqueeze(0)` (and K, V) — [1, H, T, Dh] for SDPA;
   `TTensor.scaled_dot_product_attention(…, Default(TTensor), 0.0, True,
   1.0 / Sqrt(Dh), True)` — no attention mask (`Default(TTensor)`), dropout
   0.0, `is_causal` True, scale `1.0 / Sqrt(Dh)` (its `TOptDouble`),
   `enable_gqa` True (query head h → KV head h div (Hq / Hkv));
   result [1, Hq, T, Dh]
9. `A.squeeze_dim(0).permute([1, 0, 2]).reshape([T, Hq * Dh])` — back to
   [T, Hq, Dh] and the flat head axis
10. `A * Gate.sigmoid` — the output gate
11. `FOut.Forward(…)` — the output projection → Result

### 4.8 `TLinearBlock` — the linear (Gated DeltaNet) block

Fields: the five packed linears `FQkv`, `FGate`, `FBeta`, `FAlpha`, `FSsmOut`
and four dense tensors `FDtBias`, `FSsmA`, `FConv`, `FSsmNorm`.

**Load** — inherits the skeleton's load, then the SSM weights (five packed,
four dense):

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
```

**Destroy** — the five packed linears, then `inherited`:

```pascal
destructor TLinearBlock.Destroy;
begin
  FQkv.Free;
  FGate.Free;
  FBeta.Free;
  FAlpha.Free;
  FSsmOut.Free;
  inherited Destroy;
end;
```

**Attn** — projections, causal 1-D convolution, split into Q/K/V, unit RMS
norm on Q and K, head tiling (value head j uses key head j mod Hk), then
T09's chunked delta rule with a fresh local state, the SSM norm, the gate,
and the output projection:

```pascal
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

**Tensor ops:**
1. `T`, `Hk` (key heads), `Hv` (value heads), `Sd` (SSM state dim), `K`
   (convolution kernel), `C := Qkv.Shape[1]` — the channel count
2. `Qkv := FQkv.Forward(X)` — [T, C]; `Z := FGate.Forward(X)` — [T, Hv·Sd]
   (the output gate); `Beta := FBeta.Forward(X).sigmoid`;
   `G := (FAlpha.Forward(X) + FDtBias).softplus * FSsmA` — the per-head
   decay (finite)
3. `TTensor.zeros([K - 1, C], kFloat, X.Device)` — `K − 1` zero rows on the
   device; `TTensor.cat([zeros, Qkv], 0)` — causal left-padding → `Pad`
4. `Pad.narrow(0, 0, T) * FConv.select(1, 0)` — the first convolution tap
   (the oldest token); then for `I := 1 to K - 1`:
   `U := U + Pad.narrow(0, I, T) * FConv.select(1, I)` — tap `I` uses the
   token `I` steps back, so the index `K − 1` multiplies the current token
5. `U := U.silu`
6. Split `U`: `Q := U.narrow(1, 0, Hk * Sd).view([T, Hk, Sd])`;
   `Kt := U.narrow(1, Hk * Sd, Hk * Sd).view([T, Hk, Sd])`;
   `V := U.narrow(1, 2 * Hk * Sd, Hv * Sd).view([T, Hv, Sd])`
7. Unit RMS norm per head on `Q` and `Kt`:
   `X * (X * X).sum_dim_intlist(IntList([-1]), True, kNone).add_scalar(Eps).rsqrt`
8. Head tiling, only when `Hv <> Hk`: `Q.&repeat([1, Hv div Hk, 1])` and
   the same for `Kt` — value head j uses key head j mod Hk; `&repeat`
   copies whole blocks of heads
9. `Q := Q * (1.0 / Sqrt(Sd))` — the query scale
10. `DeltaRuleChunked(Q.permute([1, 0, 2]), Kt.permute([1, 0, 2]),
    V.permute([1, 0, 2]), Beta.transpose(0, 1).contiguous,
    G.transpose(0, 1).contiguous, State)` — T09's chunked delta rule (chunk
    64, zero-padded), the recurrence's state a fresh local per call; result
    O [Hv, T, Sd]
11. `O.permute([1, 0, 2])` — [T, Hv, Sd]; `RmsNorm(…, FSsmNorm, FDims.Eps)`;
    `* Z.view([T, Hv, Sd]).silu` — the SSM output gate
12. `FSsmOut.Forward(O.reshape([T, Hv * Sd]))` → Result

## 5. Checks and messages (in order)

All the checks of both units, in the order the facts list them. Each raises
the unit's exception class with the whole message (UT compares the message
exactly; `CreateFmt` formats on the literal parts).

### 5.1 `EPackedLinear` — `TPackedLinear.Create`

| # | Failing condition | Message (exact) |
|---|-------------------|-----------------|
| 1 | `ARaw` not a 1-D kUInt8 tensor | `'the packed bytes must be a 1-D kUInt8 tensor, got ' + ShapeStr(ARaw)` |
| 2 | `AOutDim < 1` or `AInDim < 1` | `CreateFmt('dimensions must be positive, got %d x %d', [AOutDim, AInDim])` |
| 3 | byte count ≠ `GgufTensorBytes` | `CreateFmt('%s holds %d bytes, %d x %d of type %s need %d', …)`; example: `[34] holds 34 bytes, 2 x 32 of type q8_0 need 68` |

### 5.2 `EPackedLinear` — `TPackedLinear.Forward`

| # | Failing condition | Message (exact) |
|---|-------------------|-----------------|
| 1 | `X` not kFloat | `X must be kFloat (fp32)` |
| 2 | `X` 0-d or not ending in `InDim` | `CreateFmt('X must end in %d, got %s', …)`; example: `X must end in 32, got [3, 31]` |

### 5.3 `EPackedLinear` — `TensorRaw`

| # | Failing condition | Message (exact) |
|---|-------------------|-----------------|
| 3 | the tensor is not in the file | `'tensor not in the file: ' + AName` |

### 5.4 `EPackedLinear` — `LoadPacked`

| # | Failing condition | Message (exact) |
|---|-------------------|-----------------|
| 4 | `ADims` not 2-D (not a matrix) | `CreateFmt('%s is not a matrix (%d dimensions)', …)`; example: `blk.0.attn_norm.weight is not a matrix (1 dimensions)` |

### 5.5 `EQwenBlock` — `KeyInt` / `KeyFloat`

| # | Failing condition | Message (exact) |
|---|-------------------|-----------------|
| 5 | the metadata key is missing | `'metadata key missing: ' + AKey` |

`ReadQwenDims` reads `qwen35.block_count` first, so a file without the
`qwen35` keys reports that key (UT-QwenBlocks-102). A file carrying the
integer keys but lacking `qwen35.rope.freq_base` reports that key
instead (UT-QwenBlocks-216; the file is written by `WriteQwenMetaNoRopeBase`
of `tests/ut/QwenBlocksMetaFixture.pas`).

## 6. Notes (from the facts)

- `Default(TTensor)` is "no attention mask" for `scaled_dot_product_attention`.
- `1.0 / Sqrt(Dh)` converts to SDPA's `TOptDouble` scale.
- `enable_gqa` True maps query head h to KV head h div (Hq / Hkv).
- In the linear block, value head j uses key head j mod Hk (`&repeat` copies
  whole blocks of heads).
- The convolution's index ConvKernel − 1 multiplies the current token.
- RoPE angles are computed in kDouble and cast to kFloat.
- `GgufTensorBytes`, `GgufTypeName`, `TGgufTensorInfo` are `GgufFile`'s
  (T01); `DequantizeT` is `GgufQuantT`'s (T03); `DeltaRuleChunked` is T09's.
- Measured by the harness owner (prototype, CPU, GPUs hidden, against the
  deliberately slow scalar `Double` version of DESIGN 8.10): every block of
  the rehearsal model within 8.1e-7 (outputs about 4); four blocks chained
  8.9e-7; the tiny model within 2.4e-6 at T 70; as `Recompute` stages the
  input gradient equals an ordinary backward exactly; on cuda:0 all 24
  blocks within 4.0e-6 of the CPU. Mutants are caught (key tiling
  interleaved instead of repeated; convolution taps reversed; RoPE on
  adjacent pairs).

## 7. Change history

| Date | Change | Reason |
|------|--------|--------|
| 2026-11-22 | Initial creation | T10a delegation 1 of 2: the T10 documents |
| 2026-11-22 | §5.5: the `KeyFloat` missing-key case (UT-QwenBlocks-216) noted | T10b4 stage 4: 100 % line coverage of `src/QwenBlocks.pas` |
