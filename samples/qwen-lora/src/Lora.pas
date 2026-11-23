{ Lora (T12): LoRA adapters (DESIGN 8.13). A TLoraPair is A [r, in] and
  B [out, r] on one device, B zero at the start, so the adapted model starts
  identical to the base; Delta(x) = s·((x·Aᵀ)·Bᵀ), MergedDelta = s·B·A for
  the merge (8.15). A TLoraSet holds the pairs and one TVarStore per device
  (a store has one device); names follow the GGUF, 'blk.12.attn_qkv', the
  variables 'blk.12.attn_qkv.lora_a' and '.lora_b'. Save/Load write and
  read all pairs as one tch archive on the CPU. Written without reference
  to Qwen.

  Interface by the harness owner (tasks/T12-facts.md); the bodies are
  written in T12b. }
unit Lora;

{$mode objfpc}{$H+}

interface

uses
  SysUtils, Tch, TchNN;

type
  ELora = class(Exception);

  TLoraPair = class
  private
    FName: string;
    FA, FB: TTensor;
    FInDim, FOutDim, FRank: Int64;
    FScale: Double;
  public
    { Registers lora_a (uniform in ±1/sqrt(AInDim)) and lora_b (zeros) under P. }
    constructor Create(const P: TVarPath; const AName: string; AInDim, AOutDim, ARank: Int64; AScale: Double);
    { s·((X·Aᵀ)·Bᵀ); X [.., InDim]; A and B are cast to X's kind. }
    function Delta(const X: TTensor): TTensor;
    { s·B·A, [OutDim, InDim], fp32, without a graph. }
    function MergedDelta: TTensor;
    property Name: string read FName;
    property A: TTensor read FA;
    property B: TTensor read FB;
    property InDim: Int64 read FInDim;
    property OutDim: Int64 read FOutDim;
    property Rank: Int64 read FRank;
    property Scale: Double read FScale;
  end;

  TVarStoreArray = array of TVarStore;

  TLoraSet = class
  private
    FRank: Int64;
    FScale: Double;
    FStores: TVarStoreArray;
    FPairs: array of TLoraPair;
    function StoreFor(const ADevice: TDevice): TVarStore;
  public
    constructor Create(ARank: Int64; AScale: Double);
    destructor Destroy; override;
    { A new pair AName (dot-separated, e.g. 'blk.3.attn_q') on ADevice. }
    function Add(const AName: string; AInDim, AOutDim: Int64; const ADevice: TDevice): TLoraPair;
    function Count: Integer;
    function Pair(I: Integer): TLoraPair;
    { The pair named AName, or nil. }
    function Find(const AName: string): TLoraPair;
    { The stores, in the order their devices first appeared. }
    function Stores: TVarStoreArray;
    { Number of trainable parameters: Σ Rank·(InDim + OutDim). }
    function ParameterCount: Int64;
    procedure Save(const AFileName: string);
    { Every pair must be in the file with its shape (TVarStore.Load: a
      missing name raises ETorchError, another shape ETorchNativeError);
      A and B keep their handles. }
    procedure Load(const AFileName: string);
    property Rank: Int64 read FRank;
    property Scale: Double read FScale;
  end;

{ P / 'blk' / '12' / 'attn_qkv' for 'blk.12.attn_qkv'. }
function LoraPath(const P: TVarPath; const AName: string): TVarPath;

implementation

function LoraPath(const P: TVarPath; const AName: string): TVarPath;
begin
  Result := P;
  raise ELora.Create('LoraPath: not implemented yet');
end;

constructor TLoraPair.Create(const P: TVarPath; const AName: string; AInDim, AOutDim, ARank: Int64; AScale: Double);
begin
  inherited Create;
  raise ELora.Create('TLoraPair.Create: not implemented yet');
end;

function TLoraPair.Delta(const X: TTensor): TTensor;
begin
  Result := Default(TTensor);
  raise ELora.Create('TLoraPair.Delta: not implemented yet');
end;

function TLoraPair.MergedDelta: TTensor;
begin
  Result := Default(TTensor);
  raise ELora.Create('TLoraPair.MergedDelta: not implemented yet');
end;

constructor TLoraSet.Create(ARank: Int64; AScale: Double);
begin
  inherited Create;
  raise ELora.Create('TLoraSet.Create: not implemented yet');
end;

destructor TLoraSet.Destroy;
begin
  inherited Destroy;
end;

function TLoraSet.StoreFor(const ADevice: TDevice): TVarStore;
begin
  Result := nil;
  raise ELora.Create('TLoraSet.StoreFor: not implemented yet');
end;

function TLoraSet.Add(const AName: string; AInDim, AOutDim: Int64; const ADevice: TDevice): TLoraPair;
begin
  Result := nil;
  raise ELora.Create('TLoraSet.Add: not implemented yet');
end;

function TLoraSet.Count: Integer;
begin
  Result := 0;
  raise ELora.Create('TLoraSet.Count: not implemented yet');
end;

function TLoraSet.Pair(I: Integer): TLoraPair;
begin
  Result := nil;
  raise ELora.Create('TLoraSet.Pair: not implemented yet');
end;

function TLoraSet.Find(const AName: string): TLoraPair;
begin
  Result := nil;
  raise ELora.Create('TLoraSet.Find: not implemented yet');
end;

function TLoraSet.Stores: TVarStoreArray;
begin
  Result := nil;
  raise ELora.Create('TLoraSet.Stores: not implemented yet');
end;

function TLoraSet.ParameterCount: Int64;
begin
  Result := 0;
  raise ELora.Create('TLoraSet.ParameterCount: not implemented yet');
end;

procedure TLoraSet.Save(const AFileName: string);
begin
  raise ELora.Create('TLoraSet.Save: not implemented yet');
end;

procedure TLoraSet.Load(const AFileName: string);
begin
  raise ELora.Create('TLoraSet.Load: not implemented yet');
end;

end.
