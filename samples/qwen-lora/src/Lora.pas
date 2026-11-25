{ Lora (T12): LoRA adapters (DESIGN 8.13). A TLoraPair is A [r, in] and
  B [out, r] on one device, B zero at the start, so the adapted model starts
  identical to the base; Delta(x) = s·((x·Aᵀ)·Bᵀ), MergedDelta = s·B·A for
  the merge (8.15). A TLoraSet holds the pairs and one TVarStore per device
  (a store has one device); names follow the GGUF, 'blk.12.attn_qkv', the
  variables 'blk.12.attn_qkv.lora_a' and '.lora_b'. Save/Load write and
  read all pairs as one tch archive on the CPU. Written without reference
  to Qwen.

  Interface by the harness owner (tasks/T12-facts.md, T21-facts.md); the
  bodies are written in T12b and T21b. }
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
    { Registers lora_a (uniform in ±1/sqrt(AInDim), drawn on the CPU and
      copied to P's device, so the values do not depend on the device; T21)
      and lora_b (zeros) under P. }
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
var
  Parts: TStringArray;
  I: Integer;
begin
  Parts := AName.Split(['.']);
  Result := P;
  for I := 0 to High(Parts) do
    Result := Result / Parts[I];
end;

constructor TLoraPair.Create(const P: TVarPath; const AName: string; AInDim, AOutDim, ARank: Int64; AScale: Double);
var
  Bound: Double;
begin
  inherited Create;
  if (AInDim < 1) or (AOutDim < 1) then
    raise ELora.CreateFmt('%s: dimensions must be positive, got %d -> %d', [AName, AInDim, AOutDim]);
  FName := AName;
  FInDim := AInDim;
  FOutDim := AOutDim;
  FRank := ARank;
  FScale := AScale;
  Bound := 1 / Sqrt(AInDim);
  FA := P.Uniform('lora_a', [ARank, AInDim], -Bound, Bound);
  FB := P.Zeros('lora_b', [AOutDim, ARank]);
end;

function TLoraPair.Delta(const X: TTensor): TTensor;
var
  S: TInt64Array;
  Ac, Bc: TTensor;
begin
  S := X.Shape;
  if (Length(S) = 0) or (S[High(S)] <> FInDim) then
    raise ELora.CreateFmt('%s: X must end in %d', [FName, FInDim]);
  Ac := FA;
  Bc := FB;
  if X.Kind <> Ac.Kind then
  begin
    Ac := Ac.totype(X.Kind);
    Bc := Bc.totype(X.Kind);
  end;
  Result := X.matmul(Ac.t).matmul(Bc.t) * FScale;
end;

function TLoraPair.MergedDelta: TTensor;
var
  G: IInterface;
begin
  G := NoGradGuard;
  Result := FB.mm(FA) * FScale;
end;

constructor TLoraSet.Create(ARank: Int64; AScale: Double);
begin
  inherited Create;
  if ARank < 1 then
    raise ELora.CreateFmt('rank must be positive, got %d', [ARank]);
  FRank := ARank;
  FScale := AScale;
end;

destructor TLoraSet.Destroy;
var
  I: Integer;
begin
  for I := 0 to High(FPairs) do
    FPairs[I].Free;
  for I := 0 to High(FStores) do
    FStores[I].Free;
  inherited Destroy;
end;

function TLoraSet.StoreFor(const ADevice: TDevice): TVarStore;
var
  I: Integer;
begin
  for I := 0 to High(FStores) do
    if FStores[I].Device.ToString = ADevice.ToString then
      Exit(FStores[I]);
  Result := TVarStore.Create(ADevice);
  SetLength(FStores, Length(FStores) + 1);
  FStores[High(FStores)] := Result;
end;

function TLoraSet.Add(const AName: string; AInDim, AOutDim: Int64; const ADevice: TDevice): TLoraPair;
begin
  if Find(AName) <> nil then
    raise ELora.Create('pair already present: ' + AName);
  Result := TLoraPair.Create(LoraPath(StoreFor(ADevice).Root, AName), AName, AInDim, AOutDim, FRank, FScale);
  SetLength(FPairs, Length(FPairs) + 1);
  FPairs[High(FPairs)] := Result;
end;

function TLoraSet.Count: Integer;
begin
  Result := Length(FPairs);
end;

function TLoraSet.Pair(I: Integer): TLoraPair;
begin
  if (I < 0) or (I >= Length(FPairs)) then
    raise ELora.CreateFmt('pair %d outside 0..%d', [I, High(FPairs)]);
  Result := FPairs[I];
end;

function TLoraSet.Find(const AName: string): TLoraPair;
var
  I: Integer;
begin
  for I := 0 to High(FPairs) do
    if FPairs[I].Name = AName then
      Exit(FPairs[I]);
  Result := nil;
end;

function TLoraSet.Stores: TVarStoreArray;
begin
  Result := Copy(FStores);
end;

function TLoraSet.ParameterCount: Int64;
var
  I: Integer;
begin
  Result := 0;
  for I := 0 to High(FPairs) do
    Inc(Result, FRank * (FPairs[I].InDim + FPairs[I].OutDim));
end;

procedure TLoraSet.Save(const AFileName: string);
var
  S: TVarStore;
  I: Integer;
  P: TVarPath;
  G: IInterface;
begin
  G := NoGradGuard;
  S := TVarStore.Create(TDevice.Cpu);
  try
    for I := 0 to High(FPairs) do
    begin
      P := LoraPath(S.Root, FPairs[I].Name);
      P.Add('lora_a', FPairs[I].A.&to(TDevice.Cpu).detach.Copy, False);
      P.Add('lora_b', FPairs[I].B.&to(TDevice.Cpu).detach.Copy, False);
    end;
    S.Save(AFileName);
  finally
    S.Free;
  end;
end;

procedure TLoraSet.Load(const AFileName: string);
var
  S: TVarStore;
  I: Integer;
  P: TVarPath;
  Fa, Fb: TTensor;
  G: IInterface;
begin
  S := TVarStore.Create(TDevice.Cpu);
  try
    for I := 0 to High(FPairs) do
    begin
      P := LoraPath(S.Root, FPairs[I].Name);
      P.Zeros('lora_a', [FRank, FPairs[I].InDim]);
      P.Zeros('lora_b', [FPairs[I].OutDim, FRank]);
    end;
    S.Load(AFileName);
    G := NoGradGuard;
    for I := 0 to High(FPairs) do
    begin
      S.TryGet(FPairs[I].Name + '.lora_a', Fa);
      S.TryGet(FPairs[I].Name + '.lora_b', Fb);
      FPairs[I].A.CopyFrom(Fa);
      FPairs[I].B.CopyFrom(Fb);
    end;
  finally
    S.Free;
  end;
end;

end.
