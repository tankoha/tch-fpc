# T12 facts (checked by the harness owner)

T12 is the unit `Lora` (LoRA adapters: `TLoraPair`, `TLoraSet`; DESIGN
8.13, 8.12 "Lora") and the places that use it: `TPackedLinear.Lora` (unit
`PackedLinear`), `TQwenBlock.Target` / `LoraTargets` / `Index` (unit
`QwenBlocks`) and `TQwenModel.AttachLora` (unit `QwenModel`). fp32 on the
CPU; a `TLoraSet` keeps one `TVarStore` per device, so the same code runs
the two-GPU split of the 27B.

DESIGN 8.18's acceptance for T12: "zero `B` leaves the output unchanged;
`MergedDelta` equals the two-product form". Decisions taken by the harness
owner for this task (DESIGN 8.13 leaves them open):
- `A` is initialised uniform in ±1/√in (PyTorch's `kaiming_uniform_` with
  a = √5, as common LoRA code does), `B` zero. Rank 16, scale 2 (alpha 32)
  are the run's settings; every routine takes them as parameters.
- A pair does not own its tensors' store; the set owns the pairs and the
  stores. A matrix (`TPackedLinear`) holds its pair without owning it.
  Free the model (or stop calling it) before the set.
- Names follow the GGUF: pair `blk.12.attn_qkv`, variables
  `blk.12.attn_qkv.lora_a` and `.lora_b` (the store's path is split at the
  dots).
- `Save` / `Load` write and read every pair as one tch archive (`.ot`,
  T07's `TVarStore.Save` format) through a temporary store on the CPU, so a
  file does not depend on the device map. Rank and scale are not in the
  file (T16's checkpoint keeps the run's settings).
- Targets (DESIGN 8.13): a linear block adapts `attn_qkv`, `attn_gate`,
  `ssm_out`; a full block `attn_q`, `attn_k`, `attn_v`, `attn_output`;
  the trunk blocks only (not the MTP block). On the rehearsal 0.8B that is
  78 pairs and 4 915 200 parameters at rank 16 (measured, ITa-T12-001).

Measured by the harness owner with a prototype written exactly as below:
- UT: every row of the tables below passes; the four units then have
  100 % line coverage; mutants are caught (`Delta` without the scale:
  Lora 009, 010, 014–016, PackedLinear 103; `AttachLora` skipping the last
  block: QwenModel 201, 205–207; `Load` not copying `B`: Lora 108;
  `Forward` without the delta: PackedLinear 103, 107, QwenModel 203, 204).
- ITa on the rehearsal model: 0, 0, 4.8e-5, 0, 0 (cases below); with the
  same mutants 003, 001, 005 and 003 are NG.

These facts are final; do not investigate further or write probes.

## Unit `Lora` (interface, final; in the skeleton)

```pascal
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
```

## Implementation of `Lora` (write it exactly so)

The implementation section below the interface, as a whole:

```pascal
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
```

## Additions to `PackedLinear` (interface in the skeleton)

Interface (already in `src/PackedLinear.pas`): `uses` gains `Lora`; the
class gains

```pascal
  private
    FLora: TLoraPair;
    procedure SetLora(AValue: TLoraPair);
  public
    { Not owned; nil (the default) for none. }
    property Lora: TLoraPair read FLora write SetLora;
```

Implementation: `SetLora` (replaces the skeleton's body) and two lines at
the end of `Forward` (the rest of `Forward` stays as it is):

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
  if FLora <> nil then
    Result := Result + FLora.Delta(X);
end;

procedure TPackedLinear.SetLora(AValue: TLoraPair);
begin
  if (AValue <> nil) and ((AValue.InDim <> FInDim) or (AValue.OutDim <> FOutDim)) then
    raise EPackedLinear.CreateFmt('LoRA pair %s is %d -> %d, the matrix %d -> %d',
      [AValue.Name, AValue.InDim, AValue.OutDim, FInDim, FOutDim]);
  FLora := AValue;
end;
```

## Additions to `QwenBlocks` (interface in the skeleton)

Interface (already in `src/QwenBlocks.pas`): `TQwenBlock` gains the field
`FIndex: Integer` and

```pascal
    { The matrix AName ('ffn_gate', 'ffn_up', 'ffn_down' or one of the
      kind's own, without '.weight'); EQwenBlock for any other name. }
    function Target(const AName: string): TPackedLinear; virtual;
    { The names DESIGN 8.13 adapts in this kind of block. }
    function LoraTargets: TStringArray; virtual; abstract;
    property Index: Integer read FIndex;
```

`TLinearBlock` and `TFullBlock` each gain `function Target(const AName:
string): TPackedLinear; override;` and `function LoraTargets: TStringArray;
override;`.

Implementation: in `TQwenBlock.Load` the line `FIndex := ABlock;` right
after `FDevice := ADevice;`, and these five routines (they replace the
skeleton's bodies):

```pascal
function TQwenBlock.Target(const AName: string): TPackedLinear;
begin
  if AName = 'ffn_gate' then
    Result := FFfnGate
  else if AName = 'ffn_up' then
    Result := FFfnUp
  else if AName = 'ffn_down' then
    Result := FFfnDown
  else
    raise EQwenBlock.CreateFmt('blk.%d has no matrix %s', [FIndex, AName]);
end;

function TLinearBlock.Target(const AName: string): TPackedLinear;
begin
  if AName = 'attn_qkv' then
    Result := FQkv
  else if AName = 'attn_gate' then
    Result := FGate
  else if AName = 'ssm_beta' then
    Result := FBeta
  else if AName = 'ssm_alpha' then
    Result := FAlpha
  else if AName = 'ssm_out' then
    Result := FSsmOut
  else
    Result := inherited Target(AName);
end;

function TLinearBlock.LoraTargets: TStringArray;
begin
  Result := nil;
  SetLength(Result, 3);
  Result[0] := 'attn_qkv';
  Result[1] := 'attn_gate';
  Result[2] := 'ssm_out';
end;

function TFullBlock.Target(const AName: string): TPackedLinear;
begin
  if AName = 'attn_q' then
    Result := FQ
  else if AName = 'attn_k' then
    Result := FK
  else if AName = 'attn_v' then
    Result := FV
  else if AName = 'attn_output' then
    Result := FOut
  else
    Result := inherited Target(AName);
end;

function TFullBlock.LoraTargets: TStringArray;
begin
  Result := nil;
  SetLength(Result, 4);
  Result[0] := 'attn_q';
  Result[1] := 'attn_k';
  Result[2] := 'attn_v';
  Result[3] := 'attn_output';
end;
```

## Addition to `QwenModel` (interface in the skeleton)

Interface (already in `src/QwenModel.pas`): `uses` gains `Lora`; the class
gains

```pascal
    { Adds to ASet one pair per LoRA target of every trunk block
      ('blk.<i>.<target>', on the block's device) and attaches it to the
      matrix; returns the number of pairs added (T12, DESIGN 8.13). }
    function AttachLora(ASet: TLoraSet): Integer;
```

Implementation (replaces the skeleton's body):

```pascal
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
```

## Checks and messages (in this order)

`ELora` (unit `Lora`):
- `TLoraPair.Create`: `CreateFmt('%s: dimensions must be positive, got %d
  -> %d', [AName, AInDim, AOutDim])` when either is below 1, before any
  variable is registered.
- `Delta`: `CreateFmt('%s: X must end in %d', [FName, FInDim])` when X has
  no dimension or its last one is not InDim.
- `TLoraSet.Create`: `CreateFmt('rank must be positive, got %d', [ARank])`.
- `Add`: `'pair already present: ' + AName` (names compared literally).
- `Pair`: `CreateFmt('pair %d outside 0..%d', [I, High(FPairs)])`.
- From tch-fpc, passed on unchanged: an empty segment in a name
  (`TVarPath./`, `ETorchError` `sub name cannot be empty or contain "." or
  "|": ""`); `Load` of a file without one of the set's variables
  (`ETorchError` `variable <name> not found in <file>`), of another shape
  (`ETorchNativeError` from the copy, e.g. `The size of tensor a (3) must
  match the size of tensor b (4) at non-singleton dimension 0`), of a
  missing file (`ETorchNativeError` `open file failed because of errno 2 on
  fopen: No such file or directory, file path: <file>`).

`EPackedLinear` (`SetLora`): `CreateFmt('LoRA pair %s is %d -> %d, the
matrix %d -> %d', [AValue.Name, AValue.InDim, AValue.OutDim, FInDim,
FOutDim])` when a non-nil pair's dimensions differ; `Lora` is unchanged.

`EQwenBlock` (`Target`): `CreateFmt('blk.%d has no matrix %s', [FIndex,
AName])` (names compared literally).

`AttachLora` raises what `Add` raises (a second call on the same set:
`pair already present: blk.0.attn_qkv`).

## UT fixture and helpers (harness owner's; nobody changes them)

- The tiny model of T10/T11 (`tests/ut/QwenBlocksFixture.pas`): block 0
  linear, block 1 full; Hidden 64. Matrices (in -> out): block 0
  `attn_qkv` 64 -> 128, `attn_gate` 64 -> 64, `ssm_beta` and `ssm_alpha`
  64 -> 4, `ssm_out` 64 -> 64; block 1 `attn_q` 64 -> 256, `attn_k` and
  `attn_v` 64 -> 64, `attn_output` 128 -> 64; both `ffn_gate`, `ffn_up`
  64 -> 96, `ffn_down` 96 -> 64.
- `tests/ut/Lora_test.pas` (skeleton: helpers, runner, no test methods):
  `Arr(T)`, `DiffArr(A, B)`, `SetT(T, Values)` (copies values into a leaf
  without a graph), `FixPair(P)` (A[k, c] = 0.1 (k + 1) − 0.05 c, B[o, k]
  = 0.03 (o − k)), `FixX(Rows, Cols)` (X[t, c] = sin(1 + t Cols + c)),
  `RefDelta(P, X)` (s·((X·Aᵀ)·Bᵀ) in Double), `NewPair(S)` (a store S on
  the CPU and the pair `p`, 8 -> 6, rank 3, scale 2; the caller frees the
  pair, then S), `CheckCreate(Self, In, Out, Message)`, `CheckDelta(Self,
  X)` (expects `p: X must end in 8`), `NewSet` (rank 4, scale 2, pairs
  `blk.0.attn_qkv` 64 -> 128 and `blk.1.attn_q` 64 -> 256),
  `CheckPair(Self, S, Index, Message)`, and `TmpDir` (a fresh directory
  ending in the path delimiter, for the files of rows 108 to 113).
- `tests/ut/PackedLinear_test.pas` gains: `NewLinear` (an F32 matrix 6 x
  8, W[o, c] = 0.01 (8 o + c) − 0.2), `FixX` (X [5, 8], sin(1 + i)),
  `NewPair(S, In, Out)` (the pair `q`, rank 3, scale 2), `SetB(P)` (B[o, k]
  = 0.03 (o − k)), `CheckSetLora(Self, In, Out, Message)` (expects the
  message and `L.Lora = nil` afterwards).
- `tests/ut/QwenBlocks_test.pas` gains: `TargetText(B, Name)` (`'<name>
  <in> -> <out>'`), `TinyBlock(Index, G)` (block Index of `TinyPath` on the
  CPU; the caller frees the block, then G), `CheckTarget(Self, Index, Name,
  Message)`.
- `tests/ut/QwenModel_test.pas` gains: `PairNames(S)` (the names joined by
  commas), `RandomB(S)` (`ManualSeed(6)`, every B := 0.1 N(0, 1)),
  `NoGradLogits(M)` (`Logits(Trunk(Embed(Ids8)))` without a graph), the
  constant `TinyPairs` = `blk.0.attn_qkv,blk.0.attn_gate,blk.0.ssm_out,blk.1.attn_q,blk.1.attn_k,blk.1.attn_v,blk.1.attn_output`;
  and from T11 `LoadCpu`, `Arr`, `DiffArr`, `Ids8`, `TiedPath`, `NextPath`.
- Each test frees what it creates in `try ... finally`. A routine that
  sets a guard (`G := NoGradGuard`) does so in its own procedure or
  function: two guards in one routine are released in the wrong order.

## Unit tests (rows; every row has one method, every method belongs to a row)

Error rows catch the named exception class and compare the whole message
(`AssertEquals`); a row fails if no exception comes. A tolerance row
passes when the difference is ≤ the limit; "exact" means equal strings or a
difference of 0.

`Lora`, stage 1 (task T12b1): `LoraPath`, `TLoraPair`; 001–016.

| Row | Input | Expected |
|---|---|---|
| 001 | `LoraPath(S.Root, 'blk.3.attn_q').FullName('lora_a')` (S a CPU store) | `blk.3.attn_q.lora_a` |
| 002 | `LoraPath(S.Root, 'x').FullName('y')` | `x.y` |
| 003 | `LoraPath(S.Root, 'blk..q')` | ETorchError `sub name cannot be empty or contain "." or "|": ""` |
| 004 | `P := NewPair(S)` | Name `p`, InDim 8, OutDim 6, Rank 3, Scale 2; `ShapeText(A)` `[3, 8]`, `ShapeText(B)` `[6, 3]`; both `RequiresGrad`; `S.Variables` 2: `p.lora_a`, `p.lora_b`; 2 trainable |
| 005 | `ManualSeed(1)`; `NewPair(S)` | max \|B\| 0 (exact); max \|A\| ≤ 1/√8 and > 0.3 (measured 0.33284819) |
| 006 | `TLoraPair.Create(S.Root / 'p', 'p', 0, 6, 3, 2.0)` | ELora `p: dimensions must be positive, got 0 -> 6`; `S.Count` 0 |
| 007 | the same with 8, -1 | ELora `p: dimensions must be positive, got 8 -> -1`; `S.Count` 0 |
| 008 | `P.Delta(FixX(5, 8))`, B zero | `ShapeText` `[5, 6]`; max \|·\| 0 (exact) |
| 009 | `FixPair(P)`; X := `FixX(5, 8)`; `P.Delta(X)` | vs `RefDelta(P, X)` ≤ 1e-6 (measured 6.0e-9) |
| 010 | `FixPair(P)`; X := `FixX(6, 8)`; `P.Delta(X.view([2, 3, 8]))` | `ShapeText` `[2, 3, 6]`; vs `RefDelta(P, X)` ≤ 1e-6 (measured 6.0e-9) |
| 011 | `P.Delta(TTensor.zeros([5, 7]))` | ELora `p: X must end in 8` |
| 012 | `P.Delta(TTensor.zeros([5, 7]).sum(kNone))` (0-dim) | ELora `p: X must end in 8` |
| 013 | `FixPair(P)`; M := `P.MergedDelta` | `ShapeText(M)` `[6, 8]`; `RequiresGrad` False; Kind kFloat; vs 2·B·A in Double ≤ 1e-6 (measured 6.3e-9) |
| 014 | `FixPair(P)`; X := `FixX(5, 8)` | `X.matmul(P.MergedDelta.t)` vs `P.Delta(X)` ≤ 1e-5 (measured 2.2e-8) |
| 015 | `ManualSeed(1)`; `NewPair(S)` (B zero); `P.Delta(FixX(5, 8)).sum(kNone).Backward` | max \|A.grad\| 0 (exact); B.grad[o, k] vs 2·Σₜ (X·Aᵀ)[t, k] in Double ≤ 1e-5 (measured 6.6e-8) |
| 016 | `FixPair(P)`; X := `FixX(5, 8).totype(kDouble)`; `P.Delta(X)` | Kind kDouble; `P.A.Kind` still kFloat; vs `RefDelta(P, X)` ≤ 1e-12 (measured 6.9e-18) |

`Lora`, stage 2 (task T12b2): `TLoraSet`; 101–114.

| Row | Input | Expected |
|---|---|---|
| 101 | `TLoraSet.Create(4, 2.0)` | Rank 4, Scale 2, Count 0, `Length(Stores)` 0, ParameterCount 0 |
| 102 | `TLoraSet.Create(0, 2.0)` | ELora `rank must be positive, got 0` |
| 103 | `S := NewSet` | Count 2; `Pair(0).Name` `blk.0.attn_qkv`, `Pair(1).Name` `blk.1.attn_q`; `Pair(1)`: Rank 4, Scale 2, OutDim 256; `Length(Stores)` 1, `Stores[0].Device.ToString` `cpu`; `Stores[0].Variables` 4: `blk.0.attn_qkv.lora_a`, `blk.0.attn_qkv.lora_b`, `blk.1.attn_q.lora_a`, `blk.1.attn_q.lora_b`; ParameterCount 2048 |
| 104 | `S.Add('blk.0.attn_qkv', 64, 128, TDevice.Cpu)` on `NewSet` | ELora `pair already present: blk.0.attn_qkv`; Count 2 |
| 105 | `NewSet`: `Find('blk.1.attn_q')`, `Find('BLK.1.ATTN_Q')`, `Find('')`; then `Add('BLK.1.ATTN_Q', 64, 256, Cpu)` | `Pair(1)`, nil, nil; then Count 3 and `Find('BLK.1.ATTN_Q')` = `Pair(2)` |
| 106 | `NewSet`: `Pair(2)`, `Pair(-1)` | ELora `pair 2 outside 0..1`, `pair -1 outside 0..1` |
| 107 | `TLoraSet.Create(4, 2.0).Pair(0)` | ELora `pair 0 outside 0..-1` |
| 108 | `ManualSeed(3)`; S, S2 := `NewSet`; S's B := randn; `S.Save(TmpDir + 'a.ot')`; `S2.Load(..)` | the file exists; every A and B of S2 equals S's (exact) |
| 109 | S := `NewSet`, S2 the same names at rank 3; `S.Save(TmpDir + 'b.ot')`; `S2.Load(..)` | ETorchNativeError `The size of tensor a (3) must match the size of tensor b (4) at non-singleton dimension 0` |
| 110 | `NewSet.Load(TmpDir + 'missing.ot')` | ETorchNativeError `'open file failed because of errno 2 on fopen: No such file or directory, file path: ' + TmpDir + 'missing.ot'` |
| 111 | S2 := `NewSet` plus `blk.2.attn_qkv` (64 -> 128); `S.Save(TmpDir + 'c.ot')` from `NewSet`; `S2.Load(..)` | ETorchError `'variable blk.2.attn_qkv.lora_a not found in ' + TmpDir + 'c.ot'` |
| 112 | `ManualSeed(4)`; S, S2 := `NewSet`; `SetCurrentDir(TmpDir)`; `S.Save('rel.ot')`; `S2.Load('rel.ot')`; the old directory restored in `finally` | `TmpDir + 'rel.ot'` exists; `Pair(1).A` equal (exact) |
| 113 | S, S2 := `NewSet`; A0 := `S2.Pair(0).A`; `S.Save(TmpDir + 'd.ot')`; `S2.Load(..)` | `S2.Pair(0).A.SameHandle(A0)`; A and B still `RequiresGrad`; A0 equals S's A (exact) |
| 114 | `TLoraSet.Create(4, 2.0).Add('', 64, 128, Cpu)` | ETorchError `sub name cannot be empty or contain "." or "|": ""` |

`PackedLinear`, stage T12 (task T12b3): `Lora`, `SetLora`, `Forward`;
101–107. L := `NewLinear`; S a CPU store; P := `NewPair(S, 8, 6)`.

| Row | Input | Expected |
|---|---|---|
| 101 | `NewLinear` | `L.Lora = nil` |
| 102 | Y0 := `L.Forward(FixX)`; `L.Lora := P` (B zero) | `L.Lora = P`; `L.Forward(FixX)` equals Y0 (exact) |
| 103 | `ManualSeed(2)`; `SetB(P)`; `L.Lora := P`; without a graph | Y := `L.Forward(FixX)`: vs `FixX.matmul(L.Weight.t) + P.Delta(FixX)` ≤ 1e-6 (measured 0); vs `FixX.matmul((L.Weight + P.MergedDelta).t)` ≤ 1e-5 (measured 6.0e-8); vs `FixX.matmul(L.Weight.t)` > 0.1 (measured 0.147) |
| 104 | `L.Lora := NewPair(S, 7, 6)` | EPackedLinear `LoRA pair q is 7 -> 6, the matrix 8 -> 6`; `L.Lora = nil` |
| 105 | `L.Lora := NewPair(S, 8, 5)` | EPackedLinear `LoRA pair q is 8 -> 5, the matrix 8 -> 6`; `L.Lora = nil` |
| 106 | Y0 := `L.Forward(FixX)`; `SetB(P)`; `L.Lora := P`; `L.Lora := nil` | `L.Lora = nil`; `L.Forward(FixX)` equals Y0 (exact) |
| 107 | `ManualSeed(2)`; `L.Lora := P` (B zero); `L.Forward(FixX).sum(kNone).Backward` | `ShapeText(P.B.grad)` `[6, 3]`; max \|B.grad\| > 0.1; max \|A.grad\| 0 (exact) |

`QwenBlocks`, stage T12 (task T12b3): `Index`, `Target`, `LoraTargets`;
301–308. B := `TinyBlock(<index>, G)`.

| Row | Input | Expected |
|---|---|---|
| 301 | block 0 | Index 0; `TargetText`: `attn_qkv 64 -> 128`, `attn_gate 64 -> 64`, `ssm_beta 64 -> 4`, `ssm_alpha 64 -> 4`, `ssm_out 64 -> 64`, `ffn_gate 64 -> 96`, `ffn_up 64 -> 96`, `ffn_down 96 -> 64` |
| 302 | block 1 | Index 1; `attn_q 64 -> 256`, `attn_k 64 -> 64`, `attn_v 64 -> 64`, `attn_output 128 -> 64`, `ffn_gate 64 -> 96`, `ffn_up 64 -> 96`, `ffn_down 96 -> 64` |
| 303 | block 0, `Target('attn_q')` | EQwenBlock `blk.0 has no matrix attn_q` |
| 304 | block 1, `Target('ATTN_Q')` | EQwenBlock `blk.1 has no matrix ATTN_Q` |
| 305 | block 1, `Target('attn_q.weight')` | EQwenBlock `blk.1 has no matrix attn_q.weight` |
| 306 | block 1, `Target('')` | EQwenBlock `blk.1 has no matrix ` (the name empty) |
| 307 | `string.Join(',', B.LoraTargets)` for blocks 0 and 1 | `attn_qkv,attn_gate,ssm_out`; `attn_q,attn_k,attn_v,attn_output` |
| 308 | block 1 | `Target('attn_v') = Target('attn_v')`; `Target('attn_v') <> Target('attn_k')` |

`QwenModel`, stage T12 (task T12b4): `AttachLora`; 201–207. M :=
`LoadCpu(TiedPath, G)` unless said otherwise; S := `TLoraSet.Create(4,
2.0)`; free M, then S, then G.

| Row | Input | Expected |
|---|---|---|
| 201 | `M.AttachLora(S)` | 7; Count 7; `PairNames(S)` = `TinyPairs`; ParameterCount 4864; block 0 (`M.Stages[0] as TQwenBlock`): `Target('attn_qkv').Lora = S.Pair(0)`, `Target('ssm_out').Lora = S.Pair(2)`, `Target('ssm_beta').Lora = nil`, `Target('ffn_gate').Lora = nil`; block 1 `Target('attn_output').Lora = S.Pair(6)` |
| 202 | M0 := a second `TQwenModel.Load(G, Cpu, Cpu, 0)`; `M.AttachLora(S)` | `NoGradLogits(M)` equals `NoGradLogits(M0)` (exact) |
| 203 | as 202, then `RandomB(S)`; then under `NoGradGuard` H := `RecomputeForward(M.Stages, M.Embed(Ids8), Saved)` | `NoGradLogits(M)` vs `NoGradLogits(M0)` > 0.1 (measured 18.1); H equals `M.Trunk(M.Embed(Ids8))` (exact) |
| 204 | `M.AttachLora(S)`; `RandomB(S)`; `M.Logits(M.Trunk(M.Embed(Ids8))).log_softmax(-1).sum(kNone).Backward`, keep every A.grad, B.grad, `ZeroGrad`; then the same loss block-wise (`RecomputeForward`, `Logits` on H detached with `requires_grad_`, `RecomputeBackward` with `H.grad`) | every pair's B.grad max \|·\| > 0; max \|grad block-wise − whole graph\| over all A and B ≤ 1e-6 (measured 0) |
| 205 | `M.AttachLora(S)` twice | ELora `pair already present: blk.0.attn_qkv`; Count 7 |
| 206 | M := `LoadCpu(NextPath, G)`; `M.AttachLora(S)` | 7; `PairNames(S)` = `TinyPairs` (no MTP block) |
| 207 | `TQwenModel.Load(G, Cpu, Cpu, 1)` on `TiedPath`; `M.AttachLora(S)` | 7; `Length(S.Stores)` 1; `S.Stores[0].Count` 14 |

Build and run (the GPUs hidden as in the gate), for `<P>` = `Lora`,
`PackedLinear`, `QwenBlocks` or `QwenModel`:
`fpc -Mobjfpc -Sh -vew -l- -Fu{proj}/src -Fu{proj}/../../src -Fu{proj}/tests/ut -FU{proj}/work/bin -Fl{proj}/../../build/libtch-rocm -k-rpath={proj}/../../build/libtch-rocm {proj}/tests/ut/<P>_test.pas -o{proj}/work/bin/<P>_test`
then `HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 {proj}/work/bin/<P>_test --all --format=plain`.
Warnings from other units (`GgufQuant`, `GgufQuantT`, `QwenTokenizer`,
`generics.dictionaries.inc`) and libtorch's GPU warnings on stderr are
expected. Until a stage is written, a test program without test methods
reports `No valid tests found` (`Lora_test` before T12b1); that is
expected.

## ITa (prepared by the harness owner)

`QLORA_MODEL=<rehearsal GGUF> {proj}/tests/ita/T12-run.sh` builds
`tests/ita/T12/lora_check.pas` against `src/` and runs it once on the CPU
from a fresh directory outside the project: rank 16, scale 2, seed 12;
writes `{proj}/work/ita/T12/summary.tsv` (case, OK/NG, value, limit, note;
then the exit code and the count of stderr lines other than the GPU
warnings). OK when value ≤ limit. About a minute; the merged GGUF (about
1 GB) is written to `work/ita/T12` and deleted.

| Case | Compared | Limit |
|---|---|---|
| ITa-T12-001 | `AttachLora` on the rehearsal model: pairs, their dimensions and `ParameterCount` against the tensor directory (count of mismatches) | 0 |
| ITa-T12-002 | B = 0: `LastLogProbs` with the adapters against a model without them, prompts pas-001, ja-065, chat-130 of `work/oracle/T11` | 0 (exact) |
| ITa-T12-003 | B random (0.02 N(0, 1)): `LastLogProbs` with the adapters against the base forward on a GGUF whose target matrices are W + s·B·A, computed in Double from T02's scalar dequantisation, same prompts | 1e-3 |
| ITa-T12-004 | chat-065, B random: gradients of every A and B, block-wise (`Recompute`) against one backward through the whole graph, relative to the largest gradient of each tensor | 1e-4 |
| ITa-T12-005 | `Save`, then `Load` into a fresh set on a fresh model: every A and B | 0 (exact) |

Measured with the prototype: 0, 0, 4.8e-5 (the adapters move the
log-probabilities by up to 9.8), 0, 0. With the skeleton every case is NG;
mutants are caught (`Delta` without the scale: 003 NG; `AttachLora`
skipping the last block: 001 NG and the driver stops; `Load` not copying
`B`: 005 NG; `Forward` without the delta: 003 NG).

## Stages and case IDs

- Stage 1 (task T12b1): `LoraPath`, `TLoraPair` (`Create`, `Delta`,
  `MergedDelta`); rows UT-Lora-001 to 016.
- Stage 2 (task T12b2): `TLoraSet` (all of it); rows UT-Lora-101 to 114.
- Stage 3 (task T12b3): `TPackedLinear.SetLora` and the two lines of
  `Forward`; `TQwenBlock` `FIndex`, the three `Target` and two
  `LoraTargets`; rows UT-PackedLinear-101 to 107, UT-QwenBlocks-301 to 308.
- Stage 4 (task T12b4): `TQwenModel.AttachLora`; rows UT-QwenModel-201 to
  207.
- ITa: ITa-T12-001 to 005.
