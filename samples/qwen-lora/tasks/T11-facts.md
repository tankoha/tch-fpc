# T11 facts (checked by the harness owner)

T11 is one unit, one CLI unit and one program: `QwenModel` (the whole
`qwen35` model for logits: the token embedding, the trunk blocks of T10 as
T13 `Recompute` stages split over two devices, the final RMS norm and the
head; DESIGN 8.10 "Block skeleton", 8.12, 8.14 "Head"), `LogprobsCli` (the
behaviour of `qlora_logprobs`) and `src/qlora_logprobs.pas` (a thin
wrapper, written by the harness owner; complete). fp32. The same code runs
the rehearsal 0.8B model (24 trunk blocks, vocabulary 248 320, no
`output.weight`: the head is `token_embd`, "tied") and the 27B (65 blocks
of which the last is the MTP block, `qwen35.nextn_predict_layers` 1, its
own `output.weight`).

**H43** (DESIGN 8.11 "H43 in detail") is this task's acceptance and was
measured by the harness owner with a prototype written exactly as below:
- Reference: the bundled `llama-server`, `/completion` with `n_predict` 1
  and `n_probs` 20, prompts given as token ids, 24 prompts (Pascal sources,
  Japanese prose, a chat rendering with special tokens, each cut to 1, 2,
  5, 63, 64, 65, 130 and 600 tokens) and 3 text prompts. Kept in
  `{proj}/work/oracle/T11/` (`prompts.jsonl`, `expected.jsonl`,
  `make-oracle.bb`).
- **The reference runs on an all-F32 copy of the rehearsal GGUF** (every
  tensor dequantised by T02's `DequantizeBytes`) with `-ctk f32 -ctv f32
  -fa off`. On the Q4_K_M file itself `llama-server` quantises the
  activations to Q8_K before every K-quant product and keeps the KV cache
  in fp16; against that the Pascal forward differs by up to 1.8 in a
  log-probability and in the top-1 token of 2 of 24 prompts (margins
  0.36 and 0.16). With the F32 copy and an fp16 KV cache: at most 2.4e-2,
  top-1 24/24. With the F32 copy and an f32 KV cache: **at most 1.3e-3,
  top-1 24/24** (text prompts 1.4e-3). The Pascal forward reads the same
  Q4_K_M file in both cases (its dequantisation is exact, T03).
- DESIGN's threshold (same top-1 everywhere, at most 0.02 over the
  reference's top 10) therefore holds on the CPU in fp32. **H43 holds on
  the rehearsal model.**
- As `Recompute` stages the trunk gives bit for bit the same output as
  `Trunk`; on cuda:0 / cuda:1 (`--split 12`) within 1.1e-4 of the CPU.

The UT rows were run against the prototype with the tests of the
harness owner (every row gives its measured value); with them both units
have 100 % line coverage. These facts are final; do not investigate
further or write probes.

## Unit `QwenModel` (interface, final; in the skeleton)

```pascal
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
```

## Implementation of `QwenModel` (write it exactly so)

The implementation section below the interface, as a whole:

```pascal
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

function TQwenModel.LastLogProbs(const AIds: array of Integer): TTensor;
var
  G: IInterface;
  H: TTensor;
begin
  G := NoGradGuard;
  H := Trunk(Embed(AIds));
  Result := Logits(H.narrow(0, Length(AIds) - 1, 1)).log_softmax(-1).view([FVocab]);
end;

end.
```

Notes: `TensorRaw`, `LoadDense` are T10's (`PackedLinear`); `LoadBlock`,
`ReadQwenDims`, `RmsNorm` T10's (`QwenBlocks`); `DequantizeT` T03's
(`GgufQuantT`); `GgufTensorBytes` T01's (`GgufFile`). The embedding keeps
the packed bytes and dequantises only the rows of the ids (a row of
`Hidden` values is `FRowBytes` bytes, whole blocks for every type used).
The head dequantises `HeadChunk` rows at a time (8192 by default; the UT
sets a small value to cover more than one chunk). When the constructor
raises, FPC calls `Destroy`, which frees the blocks loaded so far.

## Unit `LogprobsCli` (interface, final; in the skeleton)

```pascal
uses
  Classes, SysUtils;

{ AOut / AErr are filled (caller-created, caller-owned); the result is the
  exit code. }
function RunLogprobsCLI(const AArgs: array of string; AOut, AErr: TStrings): Integer;

{ One output line (the format above) from ids and log-probabilities. }
function TopLine(const AId: string; ACount: Integer; const AIds: array of Int64;
  const ALogProbs: array of Double): string;

implementation
uses
  fpjson, jsonparser, Tch, GgufFile, QwenTokenizer, QwenModel;

const
  K_USAGE = 'Usage: qlora_logprobs [--top N] [--split S] MODEL FILE';

type
  EUsage = class(Exception);

function TopLine(const AId: string; ACount: Integer; const AIds: array of Int64;
  const ALogProbs: array of Double): string;
var
  I: Integer;
  FS: TFormatSettings;
begin
  FS := DefaultFormatSettings;
  FS.DecimalSeparator := '.';
  Result := '{"id":"' + StringToJSONString(AId) + '","n":' + IntToStr(ACount) + ',"top":[';
  for I := 0 to High(AIds) do
  begin
    if I > 0 then
      Result := Result + ',';
    Result := Result + Format('{"id":%d,"logprob":%.7f}', [AIds[I], ALogProbs[I]], FS);
  end;
  Result := Result + ']}';
end;

function ParseCount(const AText, AFlag: string; ALow, AHigh: Integer): Integer;
begin
  if not TryStrToInt(AText, Result) or (Result < ALow) or (Result > AHigh) then
    raise EUsage.CreateFmt('%s must be %d..%d, got %s', [AFlag, ALow, AHigh, AText]);
end;

function RunLogprobsCLI(const AArgs: array of string; AOut, AErr: TStrings): Integer;
var
  Top, Split, I, J, LineNo: Integer;
  Pos: array of string;
  G: TGgufFile;
  Tok: TQwenTokenizer;
  M: TQwenModel;
  Lines: TStringList;
  Js: TJSONData;
  O: TJSONObject;
  Arr: TJSONArray;
  Ids: TTokenArray;
  Id: string;
  LP, Vals, Idx: TTensor;
  TopIds: array of Int64;
  TopLp: array of Double;
begin
  Top := 20;
  Split := -1;
  Pos := nil;
  try
    I := 0;
    while I <= High(AArgs) do
    begin
      if (AArgs[I] = '--top') or (AArgs[I] = '--split') then
      begin
        if I = High(AArgs) then
          raise EUsage.Create(AArgs[I] + ' needs a value');
        if AArgs[I] = '--top' then
          Top := ParseCount(AArgs[I + 1], '--top', 1, 100)
        else
          Split := ParseCount(AArgs[I + 1], '--split', 0, 1000);
        Inc(I, 2);
      end
      else if (AArgs[I] <> '') and (AArgs[I][1] = '-') then
        raise EUsage.Create('unknown option ' + AArgs[I])
      else
      begin
        SetLength(Pos, Length(Pos) + 1);
        Pos[High(Pos)] := AArgs[I];
        Inc(I);
      end;
    end;
    if Length(Pos) <> 2 then
      raise EUsage.Create('MODEL and FILE expected');
  except
    on E: EUsage do
    begin
      AErr.Add('qlora_logprobs: ' + E.Message);
      AErr.Add(K_USAGE);
      Exit(2);
    end;
  end;
  G := nil;
  Tok := nil;
  M := nil;
  Lines := TStringList.Create;
  try
    try
      G := TGgufFile.Open(Pos[0]);
      Lines.LoadFromFile(Pos[1]);
      if Split < 0 then
        M := TQwenModel.Load(G, TDevice.Cpu, TDevice.Cpu, 0)
      else
        M := TQwenModel.Load(G, TDevice.Cuda(0), TDevice.Cuda(1), Split);
      for LineNo := 1 to Lines.Count do
      begin
        if Trim(Lines[LineNo - 1]) = '' then
          Continue;
        try
          Js := GetJSON(Lines[LineNo - 1], False);
        except
          on E: Exception do
            raise EQwenModel.CreateFmt('line %d: not JSON (%s)', [LineNo, E.Message]);
        end;
        try
          if not (Js is TJSONObject) then
            raise EQwenModel.CreateFmt('line %d: not an object', [LineNo]);
          O := TJSONObject(Js);
          Id := O.Get('id', IntToStr(LineNo));
          if (O.IndexOfName('tokens') >= 0) and (O.Elements['tokens'] is TJSONArray) then
          begin
            Arr := O.Arrays['tokens'];
            Ids := nil;
            SetLength(Ids, Arr.Count);
            for J := 0 to Arr.Count - 1 do
              Ids[J] := Arr.Integers[J];
          end
          else if (O.IndexOfName('text') >= 0) and (O.Elements['text'] is TJSONString) then
          begin
            if Tok = nil then
              Tok := TQwenTokenizer.Create(G);
            Ids := Tok.Encode(RawByteString(UTF8String(O.Strings['text'])));
          end
          else
            raise EQwenModel.CreateFmt('line %d: needs "tokens" (array) or "text" (string)', [LineNo]);
          LP := M.LastLogProbs(Ids);
          LP.topk(Top, 0, True, True, Vals, Idx);
          TopIds := nil;
          TopLp := nil;
          SetLength(TopIds, Top);
          SetLength(TopLp, Top);
          for J := 0 to Top - 1 do
          begin
            TopIds[J] := Idx.Int64Value([J]);
            TopLp[J] := Vals.DoubleValue([J]);
          end;
          AOut.Add(TopLine(Id, Length(Ids), TopIds, TopLp));
        finally
          Js.Free;
        end;
      end;
      Result := 0;
    except
      on E: Exception do
      begin
        AErr.Add('qlora_logprobs: ' + E.Message);
        Result := 1;
      end;
    end;
  finally
    M.Free;
    Tok.Free;
    G.Free;
    Lines.Free;
  end;
end;

```

## Implementation of `LogprobsCli` (write it exactly so)

The implementation section below the interface, as a whole:

```pascal
implementation

uses
  fpjson, jsonparser, Tch, GgufFile, QwenTokenizer, QwenModel;

const
  K_USAGE = 'Usage: qlora_logprobs [--top N] [--split S] MODEL FILE';

type
  EUsage = class(Exception);

function TopLine(const AId: string; ACount: Integer; const AIds: array of Int64;
  const ALogProbs: array of Double): string;
var
  I: Integer;
  FS: TFormatSettings;
begin
  FS := DefaultFormatSettings;
  FS.DecimalSeparator := '.';
  Result := '{"id":"' + StringToJSONString(AId) + '","n":' + IntToStr(ACount) + ',"top":[';
  for I := 0 to High(AIds) do
  begin
    if I > 0 then
      Result := Result + ',';
    Result := Result + Format('{"id":%d,"logprob":%.7f}', [AIds[I], ALogProbs[I]], FS);
  end;
  Result := Result + ']}';
end;

function ParseCount(const AText, AFlag: string; ALow, AHigh: Integer): Integer;
begin
  if not TryStrToInt(AText, Result) or (Result < ALow) or (Result > AHigh) then
    raise EUsage.CreateFmt('%s must be %d..%d, got %s', [AFlag, ALow, AHigh, AText]);
end;

function RunLogprobsCLI(const AArgs: array of string; AOut, AErr: TStrings): Integer;
var
  Top, Split, I, J, LineNo: Integer;
  Pos: array of string;
  G: TGgufFile;
  Tok: TQwenTokenizer;
  M: TQwenModel;
  Lines: TStringList;
  Js: TJSONData;
  O: TJSONObject;
  Arr: TJSONArray;
  Ids: TTokenArray;
  Id: string;
  LP, Vals, Idx: TTensor;
  TopIds: array of Int64;
  TopLp: array of Double;
begin
  Top := 20;
  Split := -1;
  Pos := nil;
  try
    I := 0;
    while I <= High(AArgs) do
    begin
      if (AArgs[I] = '--top') or (AArgs[I] = '--split') then
      begin
        if I = High(AArgs) then
          raise EUsage.Create(AArgs[I] + ' needs a value');
        if AArgs[I] = '--top' then
          Top := ParseCount(AArgs[I + 1], '--top', 1, 100)
        else
          Split := ParseCount(AArgs[I + 1], '--split', 0, 1000);
        Inc(I, 2);
      end
      else if (AArgs[I] <> '') and (AArgs[I][1] = '-') then
        raise EUsage.Create('unknown option ' + AArgs[I])
      else
      begin
        SetLength(Pos, Length(Pos) + 1);
        Pos[High(Pos)] := AArgs[I];
        Inc(I);
      end;
    end;
    if Length(Pos) <> 2 then
      raise EUsage.Create('MODEL and FILE expected');
  except
    on E: EUsage do
    begin
      AErr.Add('qlora_logprobs: ' + E.Message);
      AErr.Add(K_USAGE);
      Exit(2);
    end;
  end;
  G := nil;
  Tok := nil;
  M := nil;
  Lines := TStringList.Create;
  try
    try
      G := TGgufFile.Open(Pos[0]);
      Lines.LoadFromFile(Pos[1]);
      if Split < 0 then
        M := TQwenModel.Load(G, TDevice.Cpu, TDevice.Cpu, 0)
      else
        M := TQwenModel.Load(G, TDevice.Cuda(0), TDevice.Cuda(1), Split);
      for LineNo := 1 to Lines.Count do
      begin
        if Trim(Lines[LineNo - 1]) = '' then
          Continue;
        try
          Js := GetJSON(Lines[LineNo - 1], False);
        except
          on E: Exception do
            raise EQwenModel.CreateFmt('line %d: not JSON (%s)', [LineNo, E.Message]);
        end;
        try
          if not (Js is TJSONObject) then
            raise EQwenModel.CreateFmt('line %d: not an object', [LineNo]);
          O := TJSONObject(Js);
          Id := O.Get('id', IntToStr(LineNo));
          if (O.IndexOfName('tokens') >= 0) and (O.Elements['tokens'] is TJSONArray) then
          begin
            Arr := O.Arrays['tokens'];
            Ids := nil;
            SetLength(Ids, Arr.Count);
            for J := 0 to Arr.Count - 1 do
              Ids[J] := Arr.Integers[J];
          end
          else if (O.IndexOfName('text') >= 0) and (O.Elements['text'] is TJSONString) then
          begin
            if Tok = nil then
              Tok := TQwenTokenizer.Create(G);
            Ids := Tok.Encode(RawByteString(UTF8String(O.Strings['text'])));
          end
          else
            raise EQwenModel.CreateFmt('line %d: needs "tokens" (array) or "text" (string)', [LineNo]);
          LP := M.LastLogProbs(Ids);
          LP.topk(Top, 0, True, True, Vals, Idx);
          TopIds := nil;
          TopLp := nil;
          SetLength(TopIds, Top);
          SetLength(TopLp, Top);
          for J := 0 to Top - 1 do
          begin
            TopIds[J] := Idx.Int64Value([J]);
            TopLp[J] := Vals.DoubleValue([J]);
          end;
          AOut.Add(TopLine(Id, Length(Ids), TopIds, TopLp));
        finally
          Js.Free;
        end;
      end;
      Result := 0;
    except
      on E: Exception do
      begin
        AErr.Add('qlora_logprobs: ' + E.Message);
        Result := 1;
      end;
    end;
  finally
    M.Free;
    Tok.Free;
    G.Free;
    Lines.Free;
  end;
end;

end.
```

## Program `qlora_logprobs` (harness owner's; complete, in `src/`)

`src/qlora_logprobs.pas` only forwards `ParamStr` to `RunLogprobsCLI`,
writes `AOut` to stdout and `AErr` to stderr and halts with the exit code
(as `qlora_tokenize` does for T05). Nobody changes it.

## Checks and messages (in this order)

`EQwenModel` (unit `QwenModel`):
- `TQwenModel.Load`: 1. (`ReadQwenDims` raises T10's `EQwenBlock` for a
  missing key); 2. `CreateFmt('split %d outside 0..%d', [ASplit, FTrunk])`,
  e.g. `split 3 outside 0..2`; 3. (`TensorRaw` raises T10's
  `EPackedLinear` `tensor not in the file: token_embd.weight`);
  4. `token_embd.weight must be [vocab, hidden]`; 5. with `output.weight`:
  `output.weight must be [vocab, hidden]`.
- `HeadChunk := ` value < 1: `CreateFmt('head chunk must be positive, got %d', ...)`.
- `Embed`: 1. `no token ids`; 2. per id in order:
  `CreateFmt('token id %d outside 0..%d', [AIds[I], FVocab - 1])`.
- `Logits`: `CreateFmt('H must be [T, %d]', [FDims.Hidden])` when H is not
  2-D or its last dimension is not Hidden.

`RunLogprobsCLI` (unit `LogprobsCli`): usage errors (exit 2) put two lines
into `AErr`: `'qlora_logprobs: ' + reason` and the usage line
`Usage: qlora_logprobs [--top N] [--split S] MODEL FILE`. Reasons, in the
order the arguments are read: `--top needs a value` (`--split` the same),
`--top must be 1..100, got <text>`, `--split must be 0..1000, got <text>`,
`unknown option <arg>` (options are compared literally: `--TOP` is
unknown), `MODEL and FILE expected`. Every other error (exit 1) puts one
line `'qlora_logprobs: ' + E.Message` into `AErr`: per record `line %d: not
JSON (<parser message>)`, `line %d: not an object`, `line %d: needs
"tokens" (array) or "text" (string)`; and the messages of the units
(`TGgufFile.Open`, `TQwenModel`, libtorch). Blank lines are skipped; the
line number counts them. A record without `"id"` gets its line number.

## UT fixture and oracle (harness owner's; nobody changes them)

- `tests/ut/QwenBlocksFixture.pas` (T10's, extended for T11):
  `WriteTinyQwenModel(Path, Seed, Head, NextN)` writes T10's tiny model
  (the same two blocks, the same weights for the same seed) plus, by
  `Head` (`TTinyHead`): `thNone` nothing (T10's file), `thTied`
  `token_embd.weight` (Q8_0, 274 × 64) and `output_norm.weight`, `thOwn`
  also `output.weight` (F16, 274 × 64), `thBadEmbd` a `token_embd` of 32
  columns, `thBadOutput` an `output.weight` of 273 rows. With a head the
  file also carries T05's UT tokenizer (`TokenizerFixture.TinyVocab`: 274
  tokens, the six `tokenizer.ggml.*` keys). `NextN` adds
  `qwen35.nextn_predict_layers` = 1 and makes `block_count` 3 (blk.2 is
  not written). `TinyModelVocab` = 274. `WriteTinyQwen` (T10) writes the
  same bytes as before.
- `tests/ut/QwenRefScalar.pas` (T10's): `RefLogits(G, RefDims(G), Ids)`
  (T rows of 274 Doubles: embedding rows, the trunk with `RefBlock`,
  `RefRms` with `output_norm`, then `output.weight` or `token_embd`).
- `tests/ut/DeltaRuleFixture.pas` (T08's): `ShapeText`.
- The test skeletons have the helpers the rows use:
  `tests/ut/QwenModel_test.pas`: `Arr(T)`, `DiffArr(A, B)`,
  `LoadCpu(Path, G)` (opens the file and loads the model on the CPU,
  split 0; the caller frees M, then G), `LogitsDiff(Path, Ids, Chunk = 0)`
  (max |`Logits(Trunk(Embed(Ids)))` − `RefLogits`|, after checking the
  shape `[T, 274]`), the constant `Ids8` = `(5, 17, 2, 33, 8, 0, 273,
  121)` and the files `TiedPath`, `OwnPath` (thOwn), `NextPath` (thTied,
  NextN), `NonePath`, `BadEmbdPath`, `BadOutPath`, all seed 5, written at
  start. `tests/ut/LogprobsCli_test.pas`: `RunCli(Args, O, E)` (runs
  `RunLogprobsCLI`; creates O and E, the caller frees them),
  `Want(Id, Ids, Top)` (the expected line: `TopLine` of the top `Top` of
  `TQwenModel.LastLogProbs(Ids)` on `TinyPath`), `WriteText(Path, Text)`,
  `CheckUsage(Self, Args, Reason)` (exit 2, no output, the two usage
  lines), `CheckError(Self, File, Message)` (`[TinyPath, File]`: exit 1,
  one line equal to Message), the constant `Usage`, and the files
  `TinyPath` (thTied, seed 5), `PromptPath` (four lines:
  `{"id": "a", "tokens": [5, 17, 2]}`, `{"tokens": [8]}`, a blank line,
  `{"id": "t", "text": "hello world"}`) and `TmpPath` (free for a row).
- Each test frees what it creates (`TGgufFile`, `TQwenModel`,
  `TStringList`) in `try ... finally`.

## Unit tests (rows; every row has one method, every method belongs to a row)

Error rows catch the named exception class and compare the whole message
(`AssertEquals`); a row fails if no exception comes. A tolerance row
passes when the difference is ≤ the limit; "exact" means equal strings or
a difference of 0. M is `LoadCpu(<path>, G)`.

`QwenModel`, stage 1 (task T11b1): `TrunkBlockCount`, `Load`, `Destroy`,
`SetHeadChunk`, `Embed`, `Stages`; 001–016.

| Row | Input | Expected |
|---|---|---|
| 001 | M on `TiedPath` | TrunkBlocks 2, Split 0, Vocab 274, Tied True, HeadChunk 8192, `Dims.Hidden` 64 |
| 002 | M on `OwnPath` | Tied False, Vocab 274 |
| 003 | M on `NextPath` | `ReadQwenDims(G).Blocks` 3; `TrunkBlockCount(G, ReadQwenDims(G))` 2; `M.TrunkBlocks` 2 |
| 004 | `TrunkBlockCount` on `TiedPath` (opened, no model) | 2 |
| 005 | `TQwenModel.Load(G, Cpu, Cpu, 3)`, G on `TiedPath` | EQwenModel `split 3 outside 0..2` |
| 006 | the same with split -1 | EQwenModel `split -1 outside 0..2` |
| 007 | `Load(.., 0)` on `NonePath` | EPackedLinear `tensor not in the file: token_embd.weight` |
| 008 | `Load(.., 0)` on `BadEmbdPath` | EQwenModel `token_embd.weight must be [vocab, hidden]` |
| 009 | `Load(.., 0)` on `BadOutPath` | EQwenModel `output.weight must be [vocab, hidden]` |
| 010 | M on `TiedPath`; `E := M.Embed([3, 0, 273])` | `ShapeText(E)` `[3, 64]`; exact: equal to the rows 3, 0, 273 of `DequantizeBytes(8, <bytes of token_embd.weight>, 274 * 64)` (`G.ReadTensorBytes(G.FindTensor(..))`) |
| 011 | `M.Embed(<empty array>)` | EQwenModel `no token ids` |
| 012 | `M.Embed([274])` | EQwenModel `token id 274 outside 0..273` |
| 013 | `M.Embed([5, -1])` | EQwenModel `token id -1 outside 0..273` |
| 014 | `S := M.Stages` | Length 2; `S[0].ClassName` `TLinearBlock`, `S[1].ClassName` `TFullBlock`; `S[0].Device.ToString` `cpu` |
| 015 | `Load(G, Cpu, Cpu, 1)`, then `Load(G, Cpu, Cpu, 2)` on one G | Split 1, then 2 |
| 016 | `M.HeadChunk := 0` | EQwenModel `head chunk must be positive, got 0` |

`QwenModel`, stage 2 (task T11b2): `Trunk`, `Logits`, `LastLogProbs`;
101–110. Lp(R, t) is the log-softmax of row t of `RefLogits` computed in
the test in Double: `R[t, j] - max - Ln(Σ exp(R[t, k] - max))`.

| Row | Input | Expected |
|---|---|---|
| 101 | `LogitsDiff(TiedPath, Ids8)` | ≤ 1e-4 (measured 1.9e-5) |
| 102 | `LogitsDiff(OwnPath, Ids8)` | ≤ 1e-4 (measured 2.1e-6) |
| 103 | `LogitsDiff(NextPath, Ids8)` | ≤ 1e-4 (measured 1.9e-5) |
| 104 | `LogitsDiff(TiedPath, Ids)`, Ids[I] = `I * 37 mod 274`, I = 0..69 (T 70: two delta-rule chunks) | ≤ 1e-4 (measured 3.4e-5) |
| 105 | M on `TiedPath`; H := `M.Trunk(M.Embed(Ids8))`; A := `M.Logits(H)`; `M.HeadChunk := 100`; B := `M.Logits(H)` (chunks of 100, 100, 74 rows) | HeadChunk 100; `ShapeText(B)` `[8, 274]`; `DiffArr(Arr(A), Arr(B))` ≤ 1e-5 (measured 3.8e-6) |
| 106 | `LP := M.LastLogProbs(Ids8)` on `TiedPath` | `ShapeText(LP)` `[274]`; `LP.RequiresGrad` False; `Arr(LP)` vs Lp(RefLogits(G, RefDims(G), Ids8), 7) ≤ 1e-4 (measured 9.9e-6) |
| 107 | `M.Logits(TTensor.zeros([3, 32]))` | EQwenModel `H must be [T, 64]` |
| 108 | `M.Logits(TTensor.zeros([64]))` | EQwenModel `H must be [T, 64]` |
| 109 | X := `M.Embed(Ids8).detach.Copy.requires_grad_(True)`; `M.Logits(M.Trunk(X)).sum(kNone).Backward` | `ShapeText(X.grad)` `[8, 64]`; `X.grad.isfinite.all.DoubleValue([]) = 1.0` |
| 110 | `M.LastLogProbs([5])` (one token) | max over j of abs(LP[j] − Lp(RefLogits(G, RefDims(G), [5]), 0)[j]) ≤ 1e-4 (measured 1.2e-5) |

`LogprobsCli`, stage 3 (task T11b3): `TopLine`, `ParseCount`,
`RunLogprobsCLI`; 201–217.

| Row | Input | Expected |
|---|---|---|
| 201 | `TopLine('a"b', 3, [5, 7], [-0.5, -1.25])` | exact: `{"id":"a\"b","n":3,"top":[{"id":5,"logprob":-0.5000000},{"id":7,"logprob":-1.2500000}]}` |
| 202 | `CheckUsage(Self, [], ..)` | reason `MODEL and FILE expected` |
| 203 | `['--top']` | reason `--top needs a value` |
| 204 | `['--top', '0', TinyPath, PromptPath]` | reason `--top must be 1..100, got 0` |
| 205 | `['--top', '101', TinyPath, PromptPath]` | reason `--top must be 1..100, got 101` |
| 206 | `['--split', 'x', TinyPath, PromptPath]` | reason `--split must be 0..1000, got x` |
| 207 | `['--TOP', '3', TinyPath, PromptPath]` | reason `unknown option --TOP` |
| 208 | `[TinyPath]` | reason `MODEL and FILE expected` |
| 209 | `RunCli(['--top', '3', TinyPath, PromptPath], O, E)` | exit 0; E empty; 3 lines; `O[0]` = `Want('a', [5, 17, 2], 3)`; `O[1]` = `Want('2', [8], 3)` (no id: the line number) |
| 210 | the same run; Ids := `TQwenTokenizer.Create(G).Encode('hello world')` (G on `TinyPath`; free both) | `O[2]` = `Want('t', Ids, 3)` |
| 211 | `RunCli([TinyPath, PromptPath], O, E)` (default top) | exit 0; 3 lines; each has 20 occurrences of `"logprob"` |
| 212 | `WriteText(TmpPath, 'x')`; `CheckError(Self, TmpPath, ..)` | `qlora_logprobs: line 1: not JSON (Error at line 1, Pos 1: Unexpected token (x) encountered.)` |
| 213 | `WriteText(TmpPath, '[1, 2]')` | `qlora_logprobs: line 1: not an object` |
| 214 | `WriteText(TmpPath, '{"id": "x"}')` | `qlora_logprobs: line 1: needs "tokens" (array) or "text" (string)` |
| 215 | `RunCli([TinyPath + '.missing', PromptPath], O, E)` | exit 1; one line `'qlora_logprobs: Unable to open file "' + TinyPath + '.missing": No such file or directory'` |
| 216 | `WriteText(TmpPath, '{"tokens": [274]}')` | `qlora_logprobs: token id 274 outside 0..273` |
| 217 | `RunCli(['--split', '1', TinyPath, PromptPath], O, E)` (the GPUs are hidden) | exit 1; no output; one line `qlora_logprobs: No CUDA GPUs are available` |

Build and run (the GPUs hidden as in the gate), for `<P>` = `QwenModel` or
`LogprobsCli`:
`fpc -Mobjfpc -Sh -vew -l- -Fu{proj}/src -Fu{proj}/../../src -Fu{proj}/tests/ut -FU{proj}/work/bin -Fl{proj}/../../build/libtch-rocm -k-rpath={proj}/../../build/libtch-rocm {proj}/tests/ut/<P>_test.pas -o{proj}/work/bin/<P>_test`
then `HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 {proj}/work/bin/<P>_test --all --format=plain`.
Warnings from other units (`GgufQuant`, `GgufQuantT`, `QwenTokenizer`,
`generics.dictionaries.inc`) and libtorch's GPU warnings on stderr are
expected. `QwenModel_test` takes about 10 s (the scalar oracle).

## ITa (prepared by the harness owner)

`QLORA_MODEL=<rehearsal GGUF> {proj}/tests/ita/T11-run.sh` builds
`src/qlora_logprobs.pas` and `tests/ita/T11/logprobs_check.pas` against
`src/`, runs `qlora_logprobs --top 50` once on the CPU from a fresh
directory outside the project on `work/oracle/T11/prompts.jsonl` and
compares with `expected.jsonl`; writes `{proj}/work/ita/T11/summary.tsv`
(case, OK/NG, value, limit, note; then both exit codes and the count of
stderr lines other than the GPU warnings). OK when value ≤ limit. About a
minute.

| Case | Compared | Limit |
|---|---|---|
| ITa-T11-001 | top-1 token against the reference, 24 token prompts of 1–600 tokens (count of differences) | 0 |
| ITa-T11-002 | max \|logprob − ref\| over the reference's top 10, token prompts | 0.02 |
| ITa-T11-003 | 3 text prompts through `QwenTokenizer`: same length and top-1, max \|logprob − ref\| over the top 10 | 0.02 |
| ITa-T11-004 | the trunk through `RecomputeForward(M.Stages, ..)` against `M.Trunk`, chat-130 | 0 (exact) |

Measured with the prototype: 0, 1.3e-3, 1.4e-3, 0. With the skeleton every
case is NG; mutants are caught (no `output_norm`: 001–003 NG; the
embedding scaled by √Hidden: 001–003 NG; the last trunk block skipped: all
four NG). Case 101 (`--split 12` on cuda:0 / cuda:1 against the CPU run,
1.1e-4, limit 1e-3) is the harness owner's (`T11-run.sh gpu`); agents do
not run it.

## Stages and case IDs

- Stage 1 (task T11b1): `TrunkBlockCount`, `TQwenModel.Load`, `Destroy`,
  `SetHeadChunk`, `Embed`, `Stages`; rows UT-QwenModel-001 to 016.
- Stage 2 (task T11b2): `Trunk`, `Logits`, `LastLogProbs`; rows
  UT-QwenModel-101 to 110.
- Stage 3 (task T11b3): `LogprobsCli` (all of it); rows UT-LogprobsCli-201
  to 217.
- ITa: ITa-T11-001 to 004.
