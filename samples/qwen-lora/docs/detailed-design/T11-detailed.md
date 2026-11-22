# Detailed design — unit `QwenModel`, unit `LogprobsCli`, program `qlora_logprobs` (T11)

Based on the facts `tasks/T11-facts.md` (final, checked by the harness owner)
and DESIGN 8.10 ("Block skeleton"), 8.12, 8.14 ("Head"). T11 is the whole
`qwen35` model for logits in fp32: the token embedding, the trunk blocks of
T10 as T13 `Recompute` stages split over up to two devices, the final RMS norm
and the head. The tensor-op sequences below restate the facts' final
implementation; they describe the logic and operations, they are not a copy of
the code.

| Source file | Contents |
|-------------|----------|
| `src/QwenModel.pas` | unit `QwenModel`: constant `HeadChunkRows` (8192), `EQwenModel`, `TQwenModel` (`Load`, `Embed`, `Stages`, `Trunk`, `Logits`, `LastLogProbs`, `SetHeadChunk`), the function `TrunkBlockCount` |
| `src/LogprobsCli.pas` | unit `LogprobsCli`: constant `K_USAGE`, `EUsage`, `RunLogprobsCLI`, `TopLine`, private `ParseCount` |
| `src/qlora_logprobs.pas` | program `qlora_logprobs`: a thin wrapper, harness owner's, complete, nobody changes it |

`qlora_logprobs` only forwards `ParamStr` to `RunLogprobsCLI`, writes `AOut`
to stdout and `AErr` to stderr and halts with the returned exit code (as
`qlora_tokenize` does for T05).

The same code runs the rehearsal 0.8B model (24 trunk blocks, vocabulary
248 320, no `output.weight`: the head is `token_embd`, "tied") and the 27B
(65 blocks of which the last is the MTP block, `qwen35.nextn_predict_layers`
1, its own `output.weight`).

**H43** (DESIGN 8.11 "H43 in detail") is this task's acceptance. The
reference is the bundled `llama-server`, `/completion` with `n_predict` 1 and
`n_probs` 20, prompts given as token ids (24 token prompts cut to 1, 2, 5,
63, 64, 65, 130 and 600 tokens, plus 3 text prompts), kept in
`work/oracle/T11/`. **It runs on an all-F32 copy of the rehearsal GGUF**
(every tensor dequantised by T02's `DequantizeBytes`) with `-ctk f32 -ctv f32
-fa off` and an f32 KV cache: the Pascal forward then differs by at most
1.3e-3 with top-1 on 24/24 (text prompts 1.4e-3). On the Q4_K_M file itself
`llama-server` quantises the activations to Q8_K before every K-quant product
and keeps the KV cache in fp16; against that the forward differs by up to 1.8
in a log-probability and the top-1 token of 2 of 24 prompts — that is why the
reference uses the F32 copy. The Pascal forward reads the same Q4_K_M file in
both cases (its dequantisation is exact, T03). As `Recompute` stages the trunk
gives bit for bit the same output as `Trunk`; on cuda:0 / cuda:1
(`--split 12`) it is within 1.1e-4 of the CPU.

Dependencies: `ReadQwenDims`, `LoadBlock`, `RmsNorm` are T10's (`QwenBlocks`);
`TensorRaw`, `LoadDense` are T10's (`PackedLinear`); `DequantizeT` is T03's
(`GgufQuantT`); `GgufTensorBytes` is T01's (`GgufFile`). Tensors are managed
records (no `Free`); the `TQwenBlock` fields are freed in `Destroy`.

## 1. `TrunkBlockCount` and `TQwenModel.Load`

### 1.1 `TrunkBlockCount(G, D)`

The trunk is the block count **without** the MTP (next-token prediction)
block:

1. `Result := D.Blocks` — `qwen35.block_count`, from `ReadQwenDims`
2. `I := G.FindKey('qwen35.nextn_predict_layers')` — the MTP layer count, if
   the metadata key exists
3. If `I >= 0`: `Result := D.Blocks - G.KV[I].IntVal`

Rehearsal 0.8B: no key → 24 trunk blocks. 27B: 65 blocks − 1 MTP layer → 64.
(The tiny fixture with `NextN` set writes `block_count` 3 with the key = 1 and
never writes `blk.2`; the count is 3 − 1 = 2.)

### 1.2 `TQwenModel.Load(G, ADevice0, ADevice1, ASplit)`

Fields: `FDims`, `FTrunk`, `FSplit`, `FDevice0`, `FDevice1`, `FBlocks`
(array of `TQwenBlock`), `FEmbRaw`, `FHeadRaw`, `FOutNorm` (tensors),
`FEmbType`, `FHeadType` (GGUF types), `FVocab`, `FRowBytes`,
`FHeadRowBytes`, `FTied`, `FHeadChunk` (with setter `SetHeadChunk`).
Properties: `Dims`, `TrunkBlocks`, `Split`, `Vocab`, `Tied`, `HeadChunk`
(read + write).

The steps run in exactly this order (checks numbered L1–L5, messages in §6):

1. `inherited Create`
2. `FDims := ReadQwenDims(G)` — T10's; a missing `qwen35.*` key raises
   T10's `EQwenBlock` (**check L1**)
3. `FTrunk := TrunkBlockCount(G, FDims)` — the MTP block excluded (§1.1)
4. Split range check: `(ASplit < 0) or (ASplit > FTrunk)` raises
   `EQwenModel` (**check L2**); otherwise `FSplit := ASplit`
5. `FDevice0 := ADevice0; FDevice1 := ADevice1`
6. `FEmbRaw := TensorRaw(G, 'token_embd.weight', ADevice0, FEmbType, D)` —
   T10's `PackedLinear` loader: the embedding stays as the GGUF's own packed
   bytes (a 1-D kUInt8 byte tensor on `ADevice0`), with the GGUF type and the
   reversed GGUF dims in `D` (a GGUF matrix lists (In, Out); `D` is
   [vocab, hidden]); a tensor missing from the file raises T10's
   `EPackedLinear` (**check L3**)
7. Shape check: `(Length(D) <> 2) or (D[1] <> FDims.Hidden)` raises
   `EQwenModel` (**check L4**); then `FVocab := D[0]` and
   `FRowBytes := GgufTensorBytes(FEmbType, [QWord(FDims.Hidden)])` — the
   number of bytes of a row of `Hidden` values, whole blocks for every type
   used
8. Blocks: `SetLength(FBlocks, FTrunk)`; for `I := 0 to FTrunk - 1`,
   `FBlocks[I] := LoadBlock(G, I, FDims, ADevice0)` when `I < ASplit`, else
   `LoadBlock(G, I, FDims, ADevice1)` — T10's `TLinearBlock` / `TFullBlock`;
   blocks 0 to split−1 live on device 0, the rest on device 1
9. `FOutNorm := LoadDense(G, 'output_norm.weight', ADevice1)` — T10's dense
   loader: the final norm, dequantised to fp32, on device 1
10. Tied head detection: `FTied := G.FindTensor('output.weight') < 0`
    (the logic in §1.3)
11. Head, tied branch: `FHeadRaw := FEmbRaw.&to(ADevice1)` and
    `FHeadType := FEmbType`. Head, own branch:
    `FHeadRaw := TensorRaw(G, 'output.weight', ADevice1, FHeadType, D)`, then
    the shape check `(Length(D) <> 2) or (D[0] <> FVocab) or
    (D[1] <> FDims.Hidden)` raises `EQwenModel` (**check L5**, only
    reachable when `output.weight` is in the file)
12. `FHeadRowBytes := GgufTensorBytes(FHeadType, [QWord(FDims.Hidden)])`;
    `FHeadChunk := HeadChunkRows` (8192)

### 1.3 The tied head

With **tied** weights (rehearsal 0.8B: no `output.weight`), the head *is* the
token embedding: no second matrix is read from the file, and `FHeadRaw` is
the packed embedding bytes moved to `ADevice1` (where `output_norm` and the
head products run) via `.&to`; `FHeadType` stays `FEmbType`, so in the tied
case `FHeadRowBytes` equals `FRowBytes`. With **own** head weights (27B),
`output.weight` is its own tensor — possibly a different GGUF type — and its
dims must be [vocab, hidden] (check L5). Either way the head is applied by
the same chunked multiplication in §4.1.

### 1.4 `SetHeadChunk`, `Destroy`

- `SetHeadChunk(AValue)` (the `HeadChunk` setter): `AValue < 1` raises
  `EQwenModel` (message in §6); otherwise `FHeadChunk := AValue`. The default
  is `HeadChunkRows` = 8192; the UT sets a small value to force several
  chunks.
- `Destroy`: frees `FBlocks[0..High(FBlocks)]` in order, then `inherited`
  (tensor fields need no `Free`). When the constructor raises, FPC calls
  `Destroy`, which frees the blocks loaded so far.

## 2. `Embed(AIds)`

Turns token ids into [T, Hidden] fp32 rows. First the checks (in this order,
messages in §6): the array is empty → error; then each id in order,
`(AIds[I] < 0) or (AIds[I] >= FVocab)` → error. The ids are converted to an
Int64 array in the same validation loop (a raise aborts before any tensor
work):

1. `Length(AIds) = 0` → `EQwenModel`
2. per id in order, out of range → `EQwenModel`
3. `Ids` := `nil`; `SetLength(Ids, Length(AIds))`; each
   `Ids[I] := AIds[I]` as Int64

**Tensor ops** (the packing stays bytes until the selected rows are picked
out):

1. `FEmbRaw.view([FVocab, FRowBytes])` — the packed byte tensor viewed as a
   [vocab, row-bytes] matrix of bytes (a row = `Hidden` values in
   `FRowBytes` bytes)
2. `TTensor.FromArray(Ids)` — the ids as a 1-D Int64 index tensor [T]
3. `.&to(FDevice0)` — the indices moved onto the embedding's device
4. `.index_select(0, Ids)` — pick exactly the requested rows → [T, RowBytes]
5. `.reshape([-1])` — flatten to 1-D [T × RowBytes]
6. `DequantizeT(Rows, FEmbType, Length(AIds) * FDims.Hidden)` — T03's
   dequantiser: only the selected rows, to a 1-D kFloat [T × Hidden] on the
   embedding's device
7. `.view([Length(AIds), FDims.Hidden])` → Result [T, Hidden]

So `Embed` keeps the whole embedding packed and dequantises only the rows of
the ids (a row of `Hidden` values is `FRowBytes` bytes, whole blocks for
every type used).

## 3. `Stages` and `Trunk`

### 3.1 `Stages`

Returns the trunk blocks as `TStageArray` for T13's `Recompute`:

1. `Result := nil`; `SetLength(Result, FTrunk)`
2. `for I := 0 to FTrunk - 1`: `Result[I] := FBlocks[I]`

Each `TQwenBlock` inherits `TStage`, so no wrapper is needed; the array is a
fresh view over the same block objects (no copies, no tensor ops). `Recompute`
runs the same `Forward`, so stages give bit for bit the same output as
`Trunk` (H43 facts).

### 3.2 `Trunk(X)`

Runs the blocks sequentially, moving each input to the block's device:

**Tensor ops:**

1. `Result := X` — identity start [T, Hidden]
2. for `I := 0 to FTrunk - 1`:
   - `Result.&to(FBlocks[I].Device)` — device transfer: blocks below the
     split on `FDevice0`, the rest on `FDevice1` (one transfer per boundary
     crossing)
   - `FBlocks[I].Forward(…)` — T10's block forward
3. `Result := Result.&to(FDevice1)` — the trunk output ends on `FDevice1`,
   where `output_norm` and the head live → Result [T, Hidden]

## 4. `Logits(H)` and `LastLogProbs(AIds)`

### 4.1 `Logits(H)`

[H, T, Hidden] → [T, vocab] logits, with the head dequantised in chunks of
`FHeadChunk` rows (default 8192).

First the check (message in §6): `(H.Dim <> 2) or (H.Shape[1] <> FDims.Hidden)`
→ `EQwenModel`.

**Tensor ops:**

1. `H.&to(FDevice1)` — `H` to the head's device
2. `N := RmsNorm(..., FOutNorm, FDims.Eps)` — T10's RMS norm with the output
   norm: `X * (X * X).mean_dim([-1], True, kNone).add_scalar(Eps).rsqrt * W`
3. Chunk count `:= (FVocab + FHeadChunk - 1) div FHeadChunk`;
   `Parts` of that length; `R0 := 0; K := 0`
4. Loop `while R0 < FVocab`: `Rows := FVocab - R0`, capped at `FHeadChunk`;
   then per chunk:
   - `FHeadRaw.narrow(0, R0 * FHeadRowBytes, Rows * FHeadRowBytes)` — narrow
     the packed head to this chunk's rows (in bytes)
   - `DequantizeT(..., FHeadType, Rows * FDims.Hidden)` — the chunk in 1-D
     fp32 [Rows × Hidden]
   - `.view([Rows, FDims.Hidden])` — the weight matrix `W` [Rows, Hidden]
   - `Parts[K] := N.mm(W.t)` — matrix product of `N` [T, Hidden] with the
     transposed chunk [Hidden, Rows] → [T, Rows]
   - `Inc(K); Inc(R0, Rows)`
5. `Result := TTensor.cat(Parts, 1)` — the chunks concatenated along dim 1
   → [T, vocab]

The default 8192-row chunk keeps one dequantised slice in memory independent
of the vocabulary (248 320 → 31 chunks on the rehearsal model); with a small
`HeadChunk` the same code runs several chunks (UT-105: 100, 100, 74 for
vocab 274).

### 4.2 `LastLogProbs(AIds)`

The next-token log-probabilities at the last position:

1. `G := NoGradGuard` — assigned to a local `IInterface`, so autograd is
   actually disabled for the call (a bare `NoGradGuard;` would do nothing)
2. `H := Trunk(Embed(AIds))` — §2 then §3.2 → [T, Hidden] on `FDevice1`
3. `H.narrow(0, Length(AIds) - 1, 1)` — the last position, [1, Hidden]
4. `Logits(...)` — §4.1 → [1, vocab]
5. `.log_softmax(-1)` — log-probabilities over the vocab axis
6. `.view([FVocab])` → Result, shape [vocab]

## 5. `LogprobsCli` (the behaviour of `qlora_logprobs`)

Constant: `K_USAGE = 'Usage: qlora_logprobs [--top N] [--split S] MODEL
FILE'`. Exception: `EUsage`. `RunLogprobsCLI(AArgs, AOut, AErr)` fills the
caller-owned `AOut`/`AErr` and returns the exit code (0 success, 2 usage
error, 1 every other error).

### 5.1 `TopLine(AId, ACount, AIds, ALogProbs)`

Builds one JSON line `{"id":"…","n":…,"top":[{"id":…,"logprob":…},…]}`:

1. `FS := DefaultFormatSettings`; `FS.DecimalSeparator := '.'` — the number
   format is locale-proof: with a comma decimal separator the JSON would be
   broken
2. Head: `{"id":"` + `StringToJSONString(AId)` + `","n`: `IntToStr(ACount)`
   (the prompt's token count, `Length(Ids)`) + `,"top":[`
3. Loop `I := 0 to High(AIds)`: a `,` before every entry but the first, then
   `Format('{"id":%d,"logprob":%.7f}', [AIds[I], ALogProbs[I]], FS)` — the
   log-probability with 7 decimals
4. Tail: `]}`

Example (UT-201): `TopLine('a"b', 3, [5, 7], [-0.5, -1.25])` gives
`{"id":"a\"b","n":3,"top":[{"id":5,"logprob":-0.5000000},{"id":7,"logprob":-1.2500000}]}`.

### 5.2 Argument loop

Defaults: `Top := 20`, `Split := -1` (negative = CPU), positional args in
`Pos`. Then scan `AArgs` left to right:

1. `AArgs[I] = '--top'` or `'--split'`:
   - if it is the last argument: `EUsage` `<flag> needs a value`
   - otherwise `ParseCount(AArgs[I+1], flag, low, high)`: `TryStrToInt`,
     range `--top` 1..100, `--split` 0..1000; a non-integer or out-of-range
     value raises `EUsage` `<flag> must be low..high, got <text>`
   - `Inc(I, 2)`
2. else a non-empty argument whose first character is `-`:
   `EUsage` `unknown option <arg>` — options are compared literally, so
   `--TOP` is unknown
3. else a positional argument: append to `Pos`, `Inc(I)`

After the loop, `Length(Pos) <> 2` → `EUsage` `MODEL and FILE expected`.

Every `EUsage` is caught and reported as **two** lines in `AErr` — first
`'qlora_logprobs: ' + E.Message`, then `K_USAGE` — and the function returns
**exit 2** (nothing goes to `AOut`).

### 5.3 Setup

1. `G := TGgufFile.Open(Pos[0])` — the model (a failure keeps the unit's own
   message)
2. `Lines.LoadFromFile(Pos[1])` — the record file
3. Model: `Split < 0` → `TQwenModel.Load(G, TDevice.Cpu, TDevice.Cpu, 0)`;
   otherwise `TQwenModel.Load(G, TDevice.Cuda(0), TDevice.Cuda(1), Split)`
   (no CUDA GPUs available → the libtorch message surfaces as an exit-1
   error)

### 5.4 Record loop

For `LineNo := 1 to Lines.Count`:

1. `Trim(line) = ''` → skip the line; **blank lines are skipped but the line
   number counts them**
2. Parse: `GetJSON(line, False)`; on any exception re-raise
   `line <LineNo>: not JSON (<parser message>)`
3. Not a JSON object (`Js is not TJSONObject`) → `line <LineNo>: not an
   object`
4. `Id := O.Get('id', IntToStr(LineNo))` — a record without `"id"` gets its
   line number
5. The input ids, in this branch order:
   - `"tokens"` present and a JSON array → `Ids[J] := Arr.Integers[J]`
   - else `"text"` present and a JSON string → the tokenizer is created
     lazily (`if Tok = nil then Tok := TQwenTokenizer.Create(G)`), then
     `Ids := Tok.Encode(...)` (the text as UTF-8)
   - neither → `line <LineNo>: needs "tokens" (array) or "text" (string)`
6. `LP := M.LastLogProbs(Ids)` — §4.2, [vocab] log-probabilities
7. `LP.topk(Top, 0, True, True, Vals, Idx)` — the `Top` largest values on
   axis 0, sorted, with their indices
8. `for J := 0 to Top - 1`: `TopIds[J] := Idx.Int64Value([J])`,
   `TopLp[J] := Vals.DoubleValue([J])`
9. `AOut.Add(TopLine(Id, Length(Ids), TopIds, TopLp))` — one line per record
10. `finally Js.Free` (the per-line JSON data)

**Errors:** any non-`EUsage` exception (JSON problems above, `TGgufFile.Open`,
`TQwenModel`'s checks, libtorch) is caught, written as the **one** line
`'qlora_logprobs: ' + E.Message` into `AErr`, and the function returns
**exit 1**. Success returns 0. The `finally` frees `M`, then `Tok`, then `G`,
then `Lines` (tensors are managed records and are not freed).

## 6. Checks and messages (in order)

Every message as it runs, with the exact text from the facts. UT rows catch
the named exception class and compare the whole message.

### 6.1 `EQwenModel` — unit `QwenModel`

**`TQwenModel.Load`**, in the order of §1.2:

| # | Where | Exception | Message (exact) |
|---|-------|-----------|-----------------|
| 1 | `ReadQwenDims` (T10) | `EQwenBlock` | T10's missing-key message, e.g. `metadata key missing: qwen35.block_count` |
| 2 | split range | `EQwenModel` | `CreateFmt('split %d outside 0..%d', [ASplit, FTrunk])`, e.g. `split 3 outside 0..2` |
| 3 | `tensor_embd` missing — `TensorRaw` (T10) | `EPackedLinear` | `tensor not in the file: token_embd.weight` |
| 4 | embedding shape | `EQwenModel` | `token_embd.weight must be [vocab, hidden]` |
| 5 | with `output.weight`: its shape | `EQwenModel` | `output.weight must be [vocab, hidden]` |

**`HeadChunk :=` value < 1** (`SetHeadChunk`): `EQwenModel`
`CreateFmt('head chunk must be positive, got %d', ...)`, e.g. `head chunk
must be positive, got 0`.

**`Embed`**: 1. `no token ids`; 2. per id in order, out of range:
`CreateFmt('token id %d outside 0..%d', [AIds[I], FVocab - 1])`, e.g.
`token id 274 outside 0..273`.

**`Logits`**: `H` not 2-D, or last dim not `Hidden`:
`CreateFmt('H must be [T, %d]', [FDims.Hidden])`, e.g. `H must be [T, 64]`.

### 6.2 `RunLogprobsCLI` — unit `LogprobsCli`

**Usage errors (exit 2)** put two lines into `AErr`:
`'qlora_logprobs: ' + reason` and the usage line
`Usage: qlora_logprobs [--top N] [--split S] MODEL FILE`. The reasons, in
the order the arguments are read:

| # | Trigger | Reason (exact) |
|---|---------|----------------|
| 1 | `--top` / `--split` is the last argument | `<flag> needs a value`, e.g. `--top needs a value` |
| 2 | `--top` value not an integer or outside 1..100 | `--top must be 1..100, got <text>` |
| 3 | `--split` value not an integer or outside 0..1000 | `--split must be 0..1000, got <text>` |
| 4 | any other argument starting with `-` (compared literally) | `unknown option <arg>`, e.g. `unknown option --TOP` |
| 5 | not exactly two positional arguments | `MODEL and FILE expected` |

**Every other error (exit 1)** puts one line
`'qlora_logprobs: ' + E.Message` into `AErr`. Per record, in order:
`line %d: not JSON (<parser message>)`; `line %d: not an object`; `line %d:
needs "tokens" (array) or "text" (string)`; and the messages of the units
(`TGgufFile.Open`, `TQwenModel`, libtorch).

## 7. Testing notes (from the facts)

- The unit tests (rows in the facts, staged in T11b1–T11b3) run against the
  rehearsal-free tiny models of `tests/ut/QwenBlocksFixture.pas`
  (`WriteTinyQwenModel`, seed 5; `TinyModelVocab` = 274) with the head
  variants `thTied` / `thOwn` / `thBadEmbd` / `thBadOutput` / `thNone` /
  `NextN`. `QwenModel_test` compares against T10's scalar oracle
  `RefLogits(G, RefDims(G), Ids)` (embedding rows, the trunk with
  `RefBlock`, `RefRms` with `output_norm`, then `output.weight` or
  `token_embd`) in `Double`. Every row has one method; error rows catch the
  named exception class and compare the whole message.
- The CLI tests (`LogprobsCli_test`) use `TinyPath` (thTied, seed 5) and the
  four-line `PromptPath` (`{"id": "a", "tokens": [5, 17, 2]}`,
  `{"tokens": [8]}`, a blank line, `{"id": "t", "text": "hello world"}`).
- ITa cases ITa-T11-001 to 004: top-1 on the 24 token prompts (limit 0),
  max |logprob − ref| over the reference's top 10 (0.02), the same for the 3
  text prompts (0.02), and the trunk through `RecomputeForward(M.Stages, ..)`
  against `M.Trunk` (0, exact). Measured with the prototype: 0, 1.3e-3,
  1.4e-3, 0. The `--split 12` GPU row (1.1e-4, limit 1e-3) is the harness
  owner's and agents do not run it.

## 8. Change history

| Date | Change | Reason |
|------|--------|--------|
| 2026-11-23 | Initial creation | T11a delegation 1 of 2: the T11 documents |
