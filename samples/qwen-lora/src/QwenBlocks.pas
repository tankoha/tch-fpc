{ T10: the trunk blocks of qwen35 in fp32 (DESIGN 8.10 "Block skeleton",
  "Linear block", "Full block"; 8.12 "QwenBlocks").

  A block is a TStage (unit Recompute): h1 = h + Attn(rms(h, attn_norm)),
  h2 = h1 + ffn_down(silu(ffn_gate(n)) * ffn_up(n)), n = rms(h1,
  post_attention_norm). TLinearBlock is the Gated DeltaNet block (causal
  depthwise convolution, L2-normalised q and k, key heads tiled to the
  value heads, DeltaRule.DeltaRuleChunked); TFullBlock the gated attention
  block (q/k RMS norms, RoPE on the first Rot dimensions of a head, causal
  SDPA with GQA). Every size comes from the GGUF metadata (TQwenDims), so
  the same code runs the rehearsal 0.8B and the 27B model. LoRA is added
  in T12, bf16 in M2.

  Interface by the harness owner (tasks/T10-facts.md); the bodies are
  written in T10b. }
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

implementation

function ReadQwenDims(G: TGgufFile): TQwenDims;
begin
  Result := Default(TQwenDims);
  raise EQwenBlock.Create('ReadQwenDims: not implemented yet');
end;

function IsFullBlock(const D: TQwenDims; ABlock: Integer): Boolean;
begin
  Result := False;
  raise EQwenBlock.Create('IsFullBlock: not implemented yet');
end;

function LoadBlock(G: TGgufFile; ABlock: Integer; const ADims: TQwenDims; const ADevice: TDevice): TQwenBlock;
begin
  Result := nil;
  raise EQwenBlock.Create('LoadBlock: not implemented yet');
end;

function RmsNorm(const X, W: TTensor; Eps: Double): TTensor;
begin
  Result := Default(TTensor);
  raise EQwenBlock.Create('RmsNorm: not implemented yet');
end;

function RopeText(const X: TTensor; Rot: Int64; Base: Double; Pos0: Int64): TTensor;
begin
  Result := Default(TTensor);
  raise EQwenBlock.Create('RopeText: not implemented yet');
end;

constructor TQwenBlock.Load(G: TGgufFile; ABlock: Integer; const ADims: TQwenDims; const ADevice: TDevice);
begin
  inherited Create;
  raise EQwenBlock.Create('TQwenBlock.Load: not implemented yet');
end;

destructor TQwenBlock.Destroy;
begin
  inherited Destroy;
end;

function TQwenBlock.Device: TDevice;
begin
  Result := Default(TDevice);
  raise EQwenBlock.Create('TQwenBlock.Device: not implemented yet');
end;

function TQwenBlock.Forward(const H: TTensor): TTensor;
begin
  Result := Default(TTensor);
  raise EQwenBlock.Create('TQwenBlock.Forward: not implemented yet');
end;

constructor TLinearBlock.Load(G: TGgufFile; ABlock: Integer; const ADims: TQwenDims; const ADevice: TDevice);
begin
  inherited Load(G, ABlock, ADims, ADevice);
  raise EQwenBlock.Create('TLinearBlock.Load: not implemented yet');
end;

destructor TLinearBlock.Destroy;
begin
  inherited Destroy;
end;

function TLinearBlock.Attn(const X: TTensor): TTensor;
begin
  Result := Default(TTensor);
  raise EQwenBlock.Create('TLinearBlock.Attn: not implemented yet');
end;

constructor TFullBlock.Load(G: TGgufFile; ABlock: Integer; const ADims: TQwenDims; const ADevice: TDevice);
begin
  inherited Load(G, ABlock, ADims, ADevice);
  raise EQwenBlock.Create('TFullBlock.Load: not implemented yet');
end;

destructor TFullBlock.Destroy;
begin
  inherited Destroy;
end;

function TFullBlock.Attn(const X: TTensor): TTensor;
begin
  Result := Default(TTensor);
  raise EQwenBlock.Create('TFullBlock.Attn: not implemented yet');
end;

end.
