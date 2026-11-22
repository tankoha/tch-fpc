{ QwenModel (T11): the whole qwen35 model for logits — the token embedding
  (rows dequantised for the ids only), the trunk blocks of QwenBlocks as
  Recompute stages split over two devices, the final RMS norm and the head
  (output.weight, or token_embd when the file has none: tied), dequantised
  HeadChunk rows at a time. fp32. DESIGN 8.10, 8.12, 8.14 "Head";
  tasks/T11-facts.md.

  Skeleton by the harness owner (interface final); the routines are
  written in T11b. }
unit QwenModel;

{$mode objfpc}{$H+}

interface

uses
  SysUtils, Tch, GgufFile, GgufQuantT, PackedLinear, QwenBlocks, Recompute;

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
    procedure SetHeadChunk(AValue: Int64);
  public
    constructor Load(G: TGgufFile; const ADevice0, ADevice1: TDevice; ASplit: Integer);
    destructor Destroy; override;
    function Embed(const AIds: array of Integer): TTensor;
    function Stages: TStageArray;
    function Trunk(const X: TTensor): TTensor;
    function Logits(const H: TTensor): TTensor;
    function LastLogProbs(const AIds: array of Integer): TTensor;
    property Dims: TQwenDims read FDims;
    property TrunkBlocks: Integer read FTrunk;
    property Split: Integer read FSplit;
    property Vocab: Int64 read FVocab;
    property Tied: Boolean read FTied;
    property HeadChunk: Int64 read FHeadChunk write SetHeadChunk;
  end;

function TrunkBlockCount(G: TGgufFile; const D: TQwenDims): Integer;

implementation

function TrunkBlockCount(G: TGgufFile; const D: TQwenDims): Integer;
begin
  Result := 0;
  raise EQwenModel.Create('TrunkBlockCount: not implemented yet');
end;

constructor TQwenModel.Load(G: TGgufFile; const ADevice0, ADevice1: TDevice; ASplit: Integer);
begin
  inherited Create;
  raise EQwenModel.Create('TQwenModel.Load: not implemented yet');
end;

destructor TQwenModel.Destroy;
begin
  inherited Destroy;
end;

procedure TQwenModel.SetHeadChunk(AValue: Int64);
begin
  raise EQwenModel.Create('TQwenModel.SetHeadChunk: not implemented yet');
end;

function TQwenModel.Embed(const AIds: array of Integer): TTensor;
begin
  Result := Default(TTensor);
  raise EQwenModel.Create('TQwenModel.Embed: not implemented yet');
end;

function TQwenModel.Stages: TStageArray;
begin
  Result := nil;
  raise EQwenModel.Create('TQwenModel.Stages: not implemented yet');
end;

function TQwenModel.Trunk(const X: TTensor): TTensor;
begin
  Result := Default(TTensor);
  raise EQwenModel.Create('TQwenModel.Trunk: not implemented yet');
end;

function TQwenModel.Logits(const H: TTensor): TTensor;
begin
  Result := Default(TTensor);
  raise EQwenModel.Create('TQwenModel.Logits: not implemented yet');
end;

function TQwenModel.LastLogProbs(const AIds: array of Integer): TTensor;
begin
  Result := Default(TTensor);
  raise EQwenModel.Create('TQwenModel.LastLogProbs: not implemented yet');
end;

end.
