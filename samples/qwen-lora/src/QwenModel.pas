{ QwenModel (T11): the whole qwen35 model for logits — the token embedding
  (rows dequantised for the ids only), the trunk blocks of QwenBlocks as
  Recompute stages split over two devices, the final RMS norm and the head
  (output.weight, or token_embd when the file has none: tied), dequantised
  HeadChunk rows at a time. fp32; with Compute kBFloat16 (T21) every
  matrix product of the trunk blocks runs in bf16 (the head stays fp32:
  bf16 logits would keep three significant digits). DESIGN 8.10,
  8.12, 8.13, 8.14 "Head"; tasks/T11-facts.md, T21-facts.md.

  Skeleton by the harness owner (interface final); the routines are
  written in T11b, AttachLora in T12b (tasks/T12-facts.md), Compute in
  T21b (tasks/T21-facts.md). }
unit QwenModel;

{$mode objfpc}{$H+}

interface

uses
  SysUtils, Tch, GgufFile, GgufQuantT, PackedLinear, QwenBlocks, Recompute, Lora;

const
  HeadChunkRows = 8192;

type
  EQwenModel = class(Exception);

  TQwenModel = class
  private
    FDims: TQwenDims;
    FTrunk: Integer;
    FSplit: Integer;
    FDevice0, FDevice1: TDevice;
    FBlocks: array of TQwenBlock;
    FEmbRaw, FHeadRaw, FOutNorm: TTensor;
    FEmbType, FHeadType: Cardinal;
    FVocab, FRowBytes, FHeadRowBytes: Int64;
    FTied: Boolean;
    FHeadChunk: Int64;
    FCompute: TKind;
    procedure SetHeadChunk(AValue: Int64);
    procedure SetCompute(AValue: TKind);
  public
    constructor Load(G: TGgufFile; const ADevice0, ADevice1: TDevice; ASplit: Integer);
    destructor Destroy; override;
    function Embed(const AIds: array of Integer): TTensor;
    function Stages: TStageArray;
    function Trunk(const X: TTensor): TTensor;
    function Logits(const H: TTensor): TTensor;
    function LastLogProbs(const AIds: array of Integer): TTensor;
    { Adds to ASet one pair per LoRA target of every trunk block
      ('blk.<i>.<target>', on the block's device) and attaches it to the
      matrix; returns the number of pairs added (T12, DESIGN 8.13). }
    function AttachLora(ASet: TLoraSet): Integer;
    property Dims: TQwenDims read FDims;
    property TrunkBlocks: Integer read FTrunk;
    property Split: Integer read FSplit;
    property Vocab: Int64 read FVocab;
    property Tied: Boolean read FTied;
    property HeadChunk: Int64 read FHeadChunk write SetHeadChunk;
    { kFloat (the default) or kBFloat16: sets every matrix of the trunk
      blocks (TPackedLinear.Compute); the head stays fp32 (T21). }
    property Compute: TKind read FCompute write SetCompute;
  end;

function TrunkBlockCount(G: TGgufFile; const D: TQwenDims): Integer;

implementation

function TrunkBlockCount(G: TGgufFile; const D: TQwenDims): Integer;
var
  I: Integer;
begin
  Result := D.Blocks;
  I := G.FindKey('qwen35.nextn_predict_layers');
  if I >= 0 then
    Result := D.Blocks - G.KV[I].IntVal;
end;

constructor TQwenModel.Load(G: TGgufFile; const ADevice0, ADevice1: TDevice; ASplit: Integer);
var
  D: TInt64Array;
  I: Integer;
begin
  inherited Create;
  FDims := ReadQwenDims(G);
  FTrunk := TrunkBlockCount(G, FDims);
  if (ASplit < 0) or (ASplit > FTrunk) then
    raise EQwenModel.CreateFmt('split %d outside 0..%d', [ASplit, FTrunk]);
  FSplit := ASplit;
  FDevice0 := ADevice0;
  FDevice1 := ADevice1;
  FEmbRaw := TensorRaw(G, 'token_embd.weight', ADevice0, FEmbType, D);
  if (Length(D) <> 2) or (D[1] <> FDims.Hidden) then
    raise EQwenModel.Create('token_embd.weight must be [vocab, hidden]');
  FVocab := D[0];
  FRowBytes := GgufTensorBytes(FEmbType, [QWord(FDims.Hidden)]);
  SetLength(FBlocks, FTrunk);
  for I := 0 to FTrunk - 1 do
    if I < ASplit then
      FBlocks[I] := LoadBlock(G, I, FDims, ADevice0)
    else
      FBlocks[I] := LoadBlock(G, I, FDims, ADevice1);
  FOutNorm := LoadDense(G, 'output_norm.weight', ADevice1);
  FTied := G.FindTensor('output.weight') < 0;
  if FTied then
  begin
    FHeadRaw := FEmbRaw.&to(ADevice1);
    FHeadType := FEmbType;
  end
  else
  begin
    FHeadRaw := TensorRaw(G, 'output.weight', ADevice1, FHeadType, D);
    if (Length(D) <> 2) or (D[0] <> FVocab) or (D[1] <> FDims.Hidden) then
      raise EQwenModel.Create('output.weight must be [vocab, hidden]');
  end;
  FHeadRowBytes := GgufTensorBytes(FHeadType, [QWord(FDims.Hidden)]);
  FHeadChunk := HeadChunkRows;
  FCompute := kFloat;
end;

procedure TQwenModel.SetHeadChunk(AValue: Int64);
begin
  if AValue < 1 then
    raise EQwenModel.CreateFmt('head chunk must be positive, got %d', [AValue]);
  FHeadChunk := AValue;
end;

destructor TQwenModel.Destroy;
var
  I: Integer;
begin
  for I := 0 to High(FBlocks) do
    FBlocks[I].Free;
  inherited Destroy;
end;

function TQwenModel.Embed(const AIds: array of Integer): TTensor;
var
  Ids: array of Int64;
  I: Integer;
  Rows: TTensor;
begin
  if Length(AIds) = 0 then
    raise EQwenModel.Create('no token ids');
  Ids := nil;
  SetLength(Ids, Length(AIds));
  for I := 0 to High(AIds) do
  begin
    if (AIds[I] < 0) or (AIds[I] >= FVocab) then
      raise EQwenModel.CreateFmt('token id %d outside 0..%d', [AIds[I], FVocab - 1]);
    Ids[I] := AIds[I];
  end;
  Rows := FEmbRaw.view([FVocab, FRowBytes])
    .index_select(0, TTensor.FromArray(Ids).&to(FDevice0)).reshape([-1]);
  Result := DequantizeT(Rows, FEmbType, Length(AIds) * FDims.Hidden)
    .view([Length(AIds), FDims.Hidden]);
end;

function TQwenModel.Stages: TStageArray;
var
  I: Integer;
begin
  Result := nil;
  SetLength(Result, FTrunk);
  for I := 0 to FTrunk - 1 do
    Result[I] := FBlocks[I];
end;

function TQwenModel.Trunk(const X: TTensor): TTensor;
var
  I: Integer;
begin
  Result := X;
  for I := 0 to FTrunk - 1 do
    Result := FBlocks[I].Forward(Result.&to(FBlocks[I].Device));
  Result := Result.&to(FDevice1);
end;

function TQwenModel.Logits(const H: TTensor): TTensor;
var
  N, W: TTensor;
  Parts: array of TTensor;
  R0, Rows: Int64;
  K: Integer;
begin
  if (H.Dim <> 2) or (H.Shape[1] <> FDims.Hidden) then
    raise EQwenModel.CreateFmt('H must be [T, %d]', [FDims.Hidden]);
  N := RmsNorm(H.&to(FDevice1), FOutNorm, FDims.Eps);
  Parts := nil;
  SetLength(Parts, (FVocab + FHeadChunk - 1) div FHeadChunk);
  R0 := 0;
  K := 0;
  while R0 < FVocab do
  begin
    Rows := FVocab - R0;
    if Rows > FHeadChunk then
      Rows := FHeadChunk;
    W := DequantizeT(FHeadRaw.narrow(0, R0 * FHeadRowBytes, Rows * FHeadRowBytes),
      FHeadType, Rows * FDims.Hidden).view([Rows, FDims.Hidden]);
    Parts[K] := N.mm(W.t);
    Inc(K);
    Inc(R0, Rows);
  end;
  Result := TTensor.cat(Parts, 1);
end;

procedure TQwenModel.SetCompute(AValue: TKind);
begin
  raise EQwenModel.Create('TQwenModel.SetCompute: not implemented yet');
end;

function TQwenModel.LastLogProbs(const AIds: array of Integer): TTensor;
var
  G: IInterface;
  H: TTensor;
begin
  G := NoGradGuard;
  H := Trunk(Embed(AIds));
  Result := Logits(H.narrow(0, Length(AIds) - 1, 1)).log_softmax(-1).view([FVocab]);
end;

function TQwenModel.AttachLora(ASet: TLoraSet): Integer;
var
  I, J: Integer;
  Names: TStringArray;
  L: TPackedLinear;
begin
  Result := 0;
  for I := 0 to FTrunk - 1 do
  begin
    Names := FBlocks[I].LoraTargets;
    for J := 0 to High(Names) do
    begin
      L := FBlocks[I].Target(Names[J]);
      L.Lora := ASet.Add('blk.' + IntToStr(I) + '.' + Names[J], L.InDim, L.OutDim, L.Device);
      Inc(Result);
    end;
  end;
end;

end.
