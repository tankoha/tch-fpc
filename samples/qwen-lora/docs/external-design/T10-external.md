# External design — T10 (PackedLinear, QwenBlocks)

T10 comprises two units: `PackedLinear` (a GGUF matrix kept packed on a device, dequantised per call) and `QwenBlocks` (the trunk blocks of `qwen35` in fp32: RMS norm, RoPE, the FFN, the linear (Gated DeltaNet) block and the full (gated attention) block).

## Purpose

### PackedLinear

A `TPackedLinear` holds a quantised GGUF matrix as raw bytes on a device, dequantising to fp32 on each `Forward` call via T03's `DequantizeT`. This allows the rehearsal model's Q8_0 and F16 weights to stay in their compact GGUF form until needed, saving memory for the 27B model.

### QwenBlocks

`TQwenBlock` is a trunk block: h1 = h + Attn(rms(h)); h2 = h1 + FFN(rms(h1)). Each block is a `TStage` of T13's `Recompute`, enabling T11 and T17 to run the trunk as checkpointed stages. Every dimension comes from the GGUF metadata, so the same code runs both the rehearsal 0.8B model (24 blocks, hidden 1024, SSM 16 key / 16 value heads, 8 query / 2 KV heads) and the 27B.

## Block skeleton

`TQwenBlock` (base class) holds:
- The norms: `FAttnNorm`, `FPostNorm` (dense fp32 tensors)
- The FFN: `FFfnGate`, `FFfnUp`, `FFfnDown` (TPackedLinear)
- The abstract `Attn` method overridden by `TLinearBlock` and `TFullBlock`
- `Forward`: H1 := H + Attn(RmsNorm(H, FAttnNorm, Eps)); Result := H1 + FFN

Derived classes:
- `TLinearBlock`: Gated DeltaNet (SSM with delta rule, chunked)
- `TFullBlock`: Gated attention (query/key/value with RoPE and scaled dot-product attention)

`LoadBlock(G, Block, Dims, Device)` returns `TFullBlock` if `(Block + 1) mod FullInterval = 0`, else `TLinearBlock`.

## Linear block (Gated DeltaNet)

The linear block's `Attn`:
1. Qkv := QKV linear on X (packed)
2. Z := gate linear on X (packed)
3. Beta := sigmoid(beta linear on X)
4. G := softplus(alpha linear + dt_bias) * ssm_a
5. Convolution: pad Qkv with K-1 zeros at start, convolve with the 1-D kernel (stored as [K, C], tap K-1 is current)
6. U := silu of the convolution result
7. Split U into Q, K, V by SSM dimensions
8. Normalise Q and K (unit vectors per head)
9. Repeat Q and K if value heads > key heads (GQA)
10. O := DeltaRuleChunked(Q, K, V, Beta, G, State) — T09's chunked delta rule
11. Result := SSM output linear on (RmsNorm(O) * silu(Z))

The stateful recurrence runs chunked (64 rows per chunk, zero-padded) for T > 64.

## Full block (gated attention)

The full block's `Attn`:
1. QG := Q linear on X, reshaped to [T, HeadsQ, 2 * HeadDim]
2. Q := RmsNorm(first half of QG), Gate := reshape(second half to [T, HeadsQ * HeadDim])
3. K := RmsNorm(K linear on X), V := V linear on X
4. Apply RoPE to Q and K (text positions starting at 0)
5. A := scaled_dot_product_attention(Q, K, V) with causal mask, GQA, scale 1/sqrt(Dh)
6. Result := output linear on (A * sigmoid(Gate))

Both use TPackedLinear for the packed weights, dequantising per call.

## Interfaces

### PackedLinear

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

### QwenBlocks

```pascal
type
  EQwenBlock = class(Exception);

  TQwenDims = record
    Blocks, Hidden, Ffn: Int64;
    HeadsQ, HeadsKV, HeadDim, Rot: Int64;
    RopeBase, Eps: Double;
    SsmHeadsK, SsmHeadsV, SsmDim, ConvKernel, FullInterval: Int64;
  end;

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

  TLinearBlock = class(TQwenBlock) ... end;
  TFullBlock = class(TQwenBlock) ... end;

function ReadQwenDims(G: TGgufFile): TQwenDims;
function IsFullBlock(const D: TQwenDims; ABlock: Integer): Boolean;
function LoadBlock(G: TGgufFile; ABlock: Integer; const ADims: TQwenDims; const ADevice: TDevice): TQwenBlock;
function RmsNorm(const X, W: TTensor; Eps: Double): TTensor;
function RopeText(const X: TTensor; Rot: Int64; Base: Double; Pos0: Int64 = 0): TTensor;
```

## Error messages

`EPackedLinear`:
1. Create: not a 1-D kUInt8 tensor: `the packed bytes must be a 1-D kUInt8 tensor, got ` + ShapeStr
2. Create: dimension < 1: `dimensions must be positive, got %d x %d`
3. Create: byte count mismatch: `%s holds %d bytes, %d x %d of type %s need %d`
4. Forward: X not kFloat: `X must be kFloat (fp32)`
5. Forward: X last dim != InDim: `X must end in %d, got %s`
6. TensorRaw: not found: `tensor not in the file: ` + AName
7. LoadPacked: not a matrix: `%s is not a matrix (%d dimensions)`

`EQwenBlock`:
- KeyInt / KeyFloat: `metadata key missing: ` + AKey (`ReadQwenDims` reads `qwen35.block_count` first)
