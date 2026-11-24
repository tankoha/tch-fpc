# Detailed design — unit `GgufMerge` and program `qlora_merge` (T18)

Source of truth: `../external-design/T18-external.md`,
`../system-flow/T18-system.md` and the T18 facts
(`tasks/T18-facts.md`). This document restates and organises them; it adds
no new requirements.

## 1. Module composition

- Source files (all `{$mode objfpc}{$H+}`):
  - `src/GgufMerge.pas`, unit `GgufMerge` — the adapted matrices of a base
    GGUF (`MergeTargets`, with the local helper `AddTarget`), the
    checkpoint's adapters (`LoadAdapters`), one merged tensor
    (`MergeTensor`), the merged file (`WriteMerged`) and the command line
    (`RunMergeCLI`).
  - `src/qlora_merge.pas`, the program — the harness owner's thin wrapper,
    complete: it forwards the command line to `RunMergeCLI`, prints the
    output lines to stdout and the error lines to stderr, and exits with
    the returned code. Nobody changes it.
- Units used (as in the facts' code):
  - `GgufFile` (T01): `TGgufFile` — `Open`, `TensorCount`, `FindTensor`,
    `FindKey`, `KV`, `Tensor[i]` (`Name`, `TType`, `NDims`, `Dims`),
    `ReadTensorBytes`.
  - `GgufQuant` (T02): `DequantizeBytes`, `QuantizeQ8_0`, `EGgufError`.
  - `GgufWriter` (T04): `TGgufWriter` — `Create(AGguf)`, `SetTensorType`,
    `BeginWrite`, `WriteTensor`, `Finish`.
  - `QwenBlocks` (T10): `ReadQwenDims`, `TQwenDims` (`Blocks`),
    `IsFullBlock`, `EQwenBlock`.
  - `Lora` (T12): `TLoraSet` — `Create(rank, scale)`, `Add`, `Load`,
    `Find`; one `TLoraPair` per target (its `MergedDelta` is `s·B·A`).
  - `Checkpoint` (T16): `TRunState` (`Rank`, `Scale`, `BaseSha256`),
    `ReadRunState`.
  - `TrainCli` (T17): `ShaFromName` (the implementation's `uses`).
  - Support units: `Classes` (`TStrings`), `SysUtils` (`Format`,
    `IntToStr`, `FileExists`, `ExpandFileName`,
    `IncludeTrailingPathDelimiter`, `RenameFile`, `DeleteFile`), `Tch`
    (`TTensor`, `NoGradGuard`, `TDevice`).
- Declared in `GgufMerge`:
  - Constant `GgufTypeQ8_0 = 8` — the GGUF tensor type of Q8_0.
  - `EGgufMerge = class(Exception)` — the unit's error class (3).
  - The record `TMergeTarget` and the open array `TMergeTargets` (3).
  - Functions (interface section): `MergeTargets`, `LoadAdapters`,
    `MergeTensor`, `WriteMerged`, `RunMergeCLI`.
  - Local procedure (implementation section only): `AddTarget`.
- The targets are taken from the GGUF **alone** (no `TQwenModel` is
  loaded — the 27B base is 16.8 GB; the dimensions come from the tensor
  directory, the facts).
- The program `src/qlora_merge.pas` uses `GgufMerge` only.

## 2. Function specifications

| Function | Arguments | Returns | Raises | Preconditions |
|---|---|---|---|---|
| `AddTarget` (local) | `AGguf: TGgufFile; const AName: string; var R: TMergeTargets` | — | `EGgufMerge` (4.6, #1–#2) | — |
| `MergeTargets` | `AGguf: TGgufFile` | the targets, one per target, in block order | `EGgufMerge` (through `AddTarget`, 4.6, #1–#2); the errors of `ReadQwenDims` (`EQwenBlock`) pass through (4.6, #3) | the base GGUF is open |
| `LoadAdapters` | `const ATargets: TMergeTargets; const ADir: string; const AState: TRunState` | a `TLoraSet` of `AState`'s rank and scale, one CPU pair per target — **the caller frees it** | the errors of `TLoraSet.Load` pass through (4.6, #4) | `ATargets` valid (4.1) |
| `MergeTensor` | `ATType: Cardinal; const ABytes: TBytes; AIn, AOut: Int64; AP: TLoraPair` | the merged tensor as Q8_0 bytes (`AIn × AOut` elements dequantised, delta added, quantised) | `EGgufMerge` (first, 4.6, #5); the errors of `DequantizeBytes` and `QuantizeQ8_0` (`EGgufError`) pass through (4.6, #6) | `AP.InDim = AIn` and `AP.OutDim = AOut` (else #5) |
| `WriteMerged` | `AGguf: TGgufFile; const ATargets: TMergeTargets; ASet: TLoraSet; const AOutPath: string` | the number of merged tensors (`0` when `ASet` is nil) | `EGgufMerge` (4.6, #7 and #9); the writer's errors (`EGgufError`) pass through (4.6, #8) | `ASet = nil` is allowed (a byte-identical copy) |
| `RunMergeCLI` | `const AArgs: array of string; AOut, AErr: TStrings` (caller-owned) | the exit code (0 done, 1 error, 2 usage) | — (all errors caught, 4.5 step 6) | — |

`TTensor` is a managed record: never freed by this unit. The `TLoraSet`
and the `TGgufFile` objects are created and freed in `try … finally`
(4.5 step 7); `TLoraSet` is freed before `TGgufFile`.

## 3. Data structures

- Constant (interface section): `GgufTypeQ8_0 = 8` — used by
  `WriteMerged` in `SetTensorType`.
- `TMergeTarget` record (interface section):

  | Field | Type | Meaning |
  |---|---|---|
  | `Name` | `string` | the adapter name, as `TQwenModel.AttachLora` gives it (e.g. `blk.0.attn_qkv`) |
  | `Tensor` | `Integer` | the index of `<Name>.weight` in the base's tensor directory |
  | `InDim` | `Int64` | GGUF `Dims[0]` = in |
  | `OutDim` | `Int64` | GGUF `Dims[1]` = out |

- `TMergeTargets = array of TMergeTarget` — the list, in block order.
- `WriteMerged` local: `Pairs: array of TLoraPair` — sized to
  `AGguf.TensorCount` and **indexed by tensor-directory index**:
  `Pairs[i] = nil` → tensor `i` is copied byte for byte; not nil → it is
  merged (3, the interface comment of `WriteMerged`).
- `WriteMerged` local: `Part: string := AOutPath + '.part'` — the
  temporary name; `rename(2)` replaces an existing destination file
  atomically (5, external design "Part file").

## 4. Processing detail

### 4.1 `MergeTargets` (with `AddTarget`)

```pascal
{ The adapted matrices of AGguf: per trunk block 0..trunk-1 its kind's
  LoraTargets (linear: attn_qkv, attn_gate, ssm_out; full: attn_q,
  attn_k, attn_v, attn_output). Raises EGgufMerge when '<name>.weight'
  is missing or not two-dimensional. }
function MergeTargets(AGguf: TGgufFile): TMergeTargets;
```

#### 4.1.1 `AddTarget` (local)

```pascal
procedure AddTarget(AGguf: TGgufFile; const AName: string; var R: TMergeTargets);
var
  T, N: Integer;
  Info: TGgufTensorInfo;
begin
  T := AGguf.FindTensor(AName + '.weight');
  if T < 0 then
    raise EGgufMerge.CreateFmt('no tensor %s.weight', [AName]);
  Info := AGguf.Tensor[T];
  if Info.NDims <> 2 then
    raise EGgufMerge.CreateFmt('%s.weight has %d dimensions, expected 2', [AName, Info.NDims]);
  N := Length(R);
  SetLength(R, N + 1);
  R[N].Name := AName;
  R[N].Tensor := T;
  R[N].InDim := Info.Dims[0];
  R[N].OutDim := Info.Dims[1];
end;
```

Steps, in exactly this order:

1. `T := AGguf.FindTensor(AName + '.weight')`; when `T < 0` (the tensor
   is missing from the base's directory) raise
   `no tensor <AName>.weight` (4.6, #1).
2. `Info := AGguf.Tensor[T]`; when `Info.NDims <> 2` raise
   `<AName>.weight has <NDims> dimensions, expected 2` (4.6, #2).
3. Append one record to `R` (the array grows by one): `Name := AName`,
   `Tensor := T`, `InDim := Info.Dims[0]`, `OutDim := Info.Dims[1]`
   (GGUF `Dims[0]` = in, `Dims[1]` = out).

#### 4.1.2 `MergeTargets`

```pascal
function MergeTargets(AGguf: TGgufFile): TMergeTargets;
var
  D: TQwenDims;
  Trunk, B, K: Integer;
  P: string;
begin
  Result := nil;
  D := ReadQwenDims(AGguf);
  Trunk := D.Blocks;
  K := AGguf.FindKey('qwen35.nextn_predict_layers');
  if K >= 0 then
    Trunk := D.Blocks - AGguf.KV[K].IntVal;
  for B := 0 to Trunk - 1 do
  begin
    P := 'blk.' + IntToStr(B) + '.';
    if IsFullBlock(D, B) then
    begin
      AddTarget(AGguf, P + 'attn_q', Result);
      AddTarget(AGguf, P + 'attn_k', Result);
      AddTarget(AGguf, P + 'attn_v', Result);
      AddTarget(AGguf, P + 'attn_output', Result);
    end
    else
    begin
      AddTarget(AGguf, P + 'attn_qkv', Result);
      AddTarget(AGguf, P + 'attn_gate', Result);
      AddTarget(AGguf, P + 'ssm_out', Result);
    end;
  end;
end;
```

Steps, in exactly this order:

1. `Result := nil`; `D := ReadQwenDims(AGguf)` (T10; its `EQwenBlock`
   errors — a missing key — pass through, 4.6, #3).
2. `Trunk := D.Blocks` — the trunk is all the blocks, **minus the MTP
   block**: when the key `qwen35.nextn_predict_layers` is present
   (`K >= 0`), `Trunk := D.Blocks - AGguf.KV[K].IntVal` (the blocks
   `Trunk..Blocks-1` are the MTP blocks and are **not** targets; the
   harness owner's decision, the facts).
3. For each trunk block `B := 0 .. Trunk - 1`, the name prefix is
   `P := 'blk.' + B + '.'` and the targets are added, in this order:
   - `IsFullBlock(D, B)` (T10): `attn_q`, `attn_k`, `attn_v`,
     `attn_output`;
   - a linear block (the else branch): `attn_qkv`, `attn_gate`,
     `ssm_out`.

   Each goes through `AddTarget` (4.1.1), so the result is one
   `TMergeTarget` per target, in this block/target order — the kinds of
   `TQwenModel.AttachLora` (T12), taken from the GGUF alone.

### 4.2 `LoadAdapters`

```pascal
{ A TLoraSet of AState's rank and scale with one CPU pair per target,
  loaded from ADir's lora.ot (the caller frees it). }
function LoadAdapters(const ATargets: TMergeTargets; const ADir: string;
  const AState: TRunState): TLoraSet;
var
  I: Integer;
begin
  Result := TLoraSet.Create(AState.Rank, AState.Scale);
  try
    for I := 0 to High(ATargets) do
      Result.Add(ATargets[I].Name, ATargets[I].InDim, ATargets[I].OutDim, TDevice.Cpu);
    Result.Load(IncludeTrailingPathDelimiter(ADir) + 'lora.ot');
  except
    Result.Free;
    raise;
  end;
end;
```

Steps, in exactly this order:

1. `Result := TLoraSet.Create(AState.Rank, AState.Scale)` — the rank and
   the scale come from the checkpoint's `state.json` (`TRunState`, T16),
   not from the command line.
2. One CPU pair per target, in the targets' order:
   `Result.Add(ATargets[I].Name, InDim, OutDim, TDevice.Cpu)` (the names
   and the dimensions are exactly `MergeTargets`' — the pairs' A and B
   match the base's matrices, 4.1).
3. `Result.Load(IncludeTrailingPathDelimiter(ADir) + 'lora.ot')` (T12;
   the directory's trailing delimiter is forced, so `ADir` with or
   without one both work; the file-missing error of `Load` passes
   through, 4.6, #4).
4. On any exception the set is freed and the exception re-raised — the
   caller never receives a half-loaded set (the `try … except … raise`).

### 4.3 `MergeTensor`

```pascal
{ The tensor of type ATType (AIn × AOut elements in ABytes) dequantised
  to fp32, plus AP.MergedDelta, quantised to Q8_0. Raises EGgufMerge when
  AP's dimensions differ from AIn, AOut. }
function MergeTensor(ATType: Cardinal; const ABytes: TBytes; AIn, AOut: Int64;
  AP: TLoraPair): TBytes;
var
  W: TSingleArray;
  NG: IInterface;
begin
  if (AP.InDim <> AIn) or (AP.OutDim <> AOut) then
    raise EGgufMerge.CreateFmt('%s: adapter %d -> %d, tensor %d -> %d',
      [AP.Name, AP.InDim, AP.OutDim, AIn, AOut]);
  NG := NoGradGuard;
  W := DequantizeBytes(ATType, ABytes, AIn * AOut);
  W := (TTensor.FromSingleArray(W).reshape([AOut, AIn]) + AP.MergedDelta).ToSingleArray;
  Result := QuantizeQ8_0(W);
end;
```

Steps, in exactly this order:

1. The dimension check **before anything else** (4.6, #5):
   `(AP.InDim <> AIn) or (AP.OutDim <> AOut)` raises
   `<AP.Name>: adapter <AP.InDim> -> <AP.OutDim>, tensor <AIn> -> <AOut>`.
2. `NG := NoGradGuard` — assigned to a local `IInterface` so the guard
   lives for the whole call (a bare `NoGradGuard;` statement has no
   effect, the tch-fpc rule; the run expects the compiler note
   "Local variable NG is assigned but never used").
3. `W := DequantizeBytes(ATType, ABytes, AIn * AOut)` (T02) — the base's
   bytes (whatever their GGUF type: Q8_0, F16, …) become the fp32
   `AIn × AOut` elements; its `EGgufError` passes through (4.6, #6).
4. `W := (TTensor.FromSingleArray(W).reshape([AOut, AIn]) +
   AP.MergedDelta).ToSingleArray` — the base (fp32) plus the pair's
   `MergedDelta` (T12: `s·B·A`, the checkpoint's scale already applied),
   elementwise; the reshape `[AOut, AIn]` matches the pair's layout.
5. `Result := QuantizeQ8_0(W)` (T02: `roundf`; its `EGgufError` passes
   through, 4.6, #6) — the adapted matrix is stored in the merged file
   as Q8_0 (external design, "Adapted Tensors").

### 4.4 `WriteMerged`

```pascal
{ Writes AGguf to AOutPath with the pairs of ASet merged (nil: no pair,
  a byte-identical copy). Written to AOutPath + '.part' and renamed; the
  part file is removed on failure. Returns the number of merged tensors. }
function WriteMerged(AGguf: TGgufFile; const ATargets: TMergeTargets; ASet: TLoraSet;
  const AOutPath: string): Integer;
var
  W: TGgufWriter;
  Pairs: array of TLoraPair;
  I: Integer;
  Part: string;
  B: TBytes;
  Info: TGgufTensorInfo;
begin
  Result := 0;
  Pairs := nil;
  SetLength(Pairs, AGguf.TensorCount);
  if ASet <> nil then
    for I := 0 to High(ATargets) do
    begin
      Pairs[ATargets[I].Tensor] := ASet.Find(ATargets[I].Name);
      if Pairs[ATargets[I].Tensor] = nil then
        raise EGgufMerge.CreateFmt('no adapter %s', [ATargets[I].Name]);
    end;
  Part := AOutPath + '.part';
  W := TGgufWriter.Create(AGguf);
  try
    for I := 0 to High(Pairs) do
      if Pairs[I] <> nil then
        W.SetTensorType(AGguf.Tensor[I].Name, GgufTypeQ8_0);
    W.BeginWrite(Part);
    for I := 0 to AGguf.TensorCount - 1 do
    begin
      AGguf.ReadTensorBytes(I, B);
      if Pairs[I] <> nil then
      begin
        Info := AGguf.Tensor[I];
        B := MergeTensor(Info.TType, B, Info.Dims[0], Info.Dims[1], Pairs[I]);
        Inc(Result);
      end;
      W.WriteTensor(B);
    end;
    W.Finish;
  finally
    W.Free;   { removes the part file when not finished }
  end;
  if not RenameFile(Part, AOutPath) then   { rename(2): replaces a file }
  begin
    DeleteFile(Part);
    raise EGgufMerge.CreateFmt('cannot rename %s to %s', [Part, AOutPath]);
  end;
end;
```

Steps, in exactly this order:

1. **The pairs by tensor index** — `Pairs` is sized to
   `AGguf.TensorCount` and indexed by the tensor's directory index; when
   `ASet <> nil`, for each target in order:
   `Pairs[ATargets[I].Tensor] := ASet.Find(ATargets[I].Name)`; when the
   pair is missing, `no adapter <name>` — the **first** target, in
   order, without a pair in the set, and the check is done **before
   anything is written** (4.6, #7). When `ASet = nil` the loop is
   skipped: every `Pairs[I]` stays nil (the byte-identical copy).
2. `Part := AOutPath + '.part'`; `W := TGgufWriter.Create(AGguf)` (T04:
   the writer copies the metadata as bytes and rebuilds the tensor
   directory in the base's order — offsets recomputed for the new sizes,
   the base's alignment preserved; the external design's
   "Merged File Contents").
3. **The retyping** — before `BeginWrite`, for every index with a pair:
   `W.SetTensorType(AGguf.Tensor[I].Name, GgufTypeQ8_0)` — the adapted
   entries of the directory are type 8 (Q8_0) in the output, the rest
   keep the base's types.
4. `W.BeginWrite(Part)` — the output goes to the **part file** first
   (4.6, #8 for its `EGgufError`, e.g.
   `TGgufWriter.BeginWrite: cannot create file <out>.part`).
5. **The tensor loop** — for `I := 0 .. AGguf.TensorCount - 1`:
   - `AGguf.ReadTensorBytes(I, B)`;
   - when `Pairs[I] <> nil`:
     `B := MergeTensor(Info.TType, B, Info.Dims[0], Info.Dims[1],
     Pairs[I])` (the base's own type and dimensions, 4.3) and
     `Inc(Result)` (the count of merged tensors);
   - `W.WriteTensor(B)` — the non-target tensors go out byte for byte.
6. `W.Finish` — the file is complete; then the `finally` frees `W` (the
   writer's destructor **removes the part file when not finished**).
7. **The rename** — `RenameFile(Part, AOutPath)` is `rename(2)`: it
   **replaces** an existing destination file (the output may already
   exist). When it fails: `DeleteFile(Part)` (the part file is removed)
   and `cannot rename <Part> to <AOutPath>` (4.6, #9). On success
   `Result` — the number of merged tensors — is returned and nothing
   else remains but the merged file.

### 4.5 `RunMergeCLI`

```pascal
{ qlora_merge [--base-sha256 H] BASE CKPT OUT; AOut / AErr caller-owned;
  the result is the exit code (0 done, 1 error, 2 usage). }
function RunMergeCLI(const AArgs: array of string; AOut, AErr: TStrings): Integer;
var
  Args: array of string;
  Sha, A: string;
  I, N: Integer;
  G: TGgufFile;
  St: TRunState;
  T: TMergeTargets;
  S: TLoraSet;
begin
  Args := nil;
  Sha := '';
  I := 0;
  while I <= High(AArgs) do
  begin
    A := AArgs[I];
    if A = '--base-sha256' then
    begin
      if I = High(AArgs) then
      begin
        AErr.Add('--base-sha256: value missing');
        Exit(2);
      end;
      Sha := AArgs[I + 1];
      Inc(I, 2);
      Continue;
    end;
    if (Length(A) > 1) and (A[1] = '-') then
    begin
      AErr.Add('unknown option ' + A);
      Exit(2);
    end;
    SetLength(Args, Length(Args) + 1);
    Args[High(Args)] := A;
    Inc(I);
  end;
  if Length(Args) <> 3 then
  begin
    AErr.Add(Format('expected BASE CKPT OUT, got %d arguments', [Length(Args)]));
    Exit(2);
  end;
  Result := 1;
  if not FileExists(Args[0]) then
  begin
    AErr.Add('no model file ' + Args[0]);
    Exit;
  end;
  if ExpandFileName(Args[2]) = ExpandFileName(Args[0]) then
  begin
    AErr.Add('the output must not be the base model');
    Exit;
  end;
  if Sha = '' then
    Sha := ShaFromName(Args[0]);
  G := nil;
  S := nil;
  try
    try
      St := ReadRunState(Args[1]);
      if (St.BaseSha256 <> '') and (Sha <> '') and (St.BaseSha256 <> Sha) then
      begin
        AErr.Add('the base model differs from the checkpoint''s: ' + Args[1]);
        Exit;
      end;
      G := TGgufFile.Open(Args[0]);
      T := MergeTargets(G);
      S := LoadAdapters(T, Args[1], St);
      N := WriteMerged(G, T, S, Args[2]);
      AOut.Add(Format('merged %d tensors (rank %d, scale %g) into %s',
        [N, St.Rank, St.Scale, Args[2]]));
      Result := 0;
    except
      on E: Exception do
        AErr.Add(E.Message);
    end;
  finally
    S.Free;
    G.Free;
  end;
end;
```

Steps, in exactly this order:

**Step 1 — the command line (usage, exit 2)**

1a. The argument loop, over `AArgs` (with an empty array it does not run):
for each argument `A`:
- `A = '--base-sha256'`: when it is the last argument
  (`I = High(AArgs)`, the value is missing) → `AErr.Add`
  `--base-sha256: value missing`, `Exit(2)`; else `Sha :=
  AArgs[I + 1]` and two arguments are consumed (4.6, #10).
- else when `(Length(A) > 1) and (A[1] = '-')` → `unknown option <A>` —
  any argument longer than one character that starts with `-`, compared
  literally (no case folding: `--BASE-SHA256` is an unknown option, but
  a one-character `-` is a positional), `Exit(2)` (4.6, #11).
- else `A` is a positional, appended to `Args` (the three of them are
  `BASE CKPT OUT`).
1b. When the positionals are not exactly three →
  `expected BASE CKPT OUT, got <n> arguments`, `Exit(2)` (4.6, #12).
   The usage error is **the message alone on `AErr`** — unlike
   `qlora_train` there is no second `Usage` line (4.6).

**Step 2 — the base and the output (exit 1)**

2a. `Result := 1` — every later `Exit` without an explicit code returns
   1; only the success path below sets 0.
2b. `not FileExists(Args[0])` → `no model file <BASE>` (4.6, #13).
2c. `ExpandFileName(Args[2]) = ExpandFileName(Args[0])` (the relative
   names are expanded, so `out` and `./out` are the same) →
   `the output must not be the base model` (4.6, #14).
2d. The base's sha256: when `--base-sha256` gave nothing
   (`Sha = ''`), `Sha := ShaFromName(Args[0])` (T17: the file name
   `sha256-<64 lower-case hex>` of an Ollama blob, else `''`; **not
   computed** — stock FPC 3.2.2 has no sha256, the facts).

**Step 3 — the checkpoint and the merge (any raise: the message alone on
`AErr`, exit 1; success: exit 0)**

3a. `St := ReadRunState(Args[1])` (T16, from `Args[1]`'s `state.json`;
   its error `not a checkpoint: <CKPT>` passes through, 4.6, #15).
3b. When **both** `St.BaseSha256` and `Sha` are known (non-empty) and
   differ, **compared literally** (an upper-case digit on one side is a
   difference) →
   `the base model differs from the checkpoint's: <CKPT>` (4.6, #16).
   When either side is unknown there is no check.
3c. `G := TGgufFile.Open(Args[0])` (T01; its errors pass through, 4.6,
   #17).
3d. `T := MergeTargets(G)` (4.1; its `EGgufMerge` and the pass-through
   `EQwenBlock` of 4.6, #1–#3).
3e. `S := LoadAdapters(T, Args[1], St)` (4.2; the `TLoraSet.Load`
   pass-through of 4.6, #4).
3f. `N := WriteMerged(G, T, S, Args[2])` (4.4; the `EGgufMerge` of
   4.6, #7 and #9 and the writer's `EGgufError` of #8).
3g. Success: `AOut.Add(Format('merged %d tensors (rank %d, scale %g)
   into %s', [N, St.Rank, St.Scale, Args[2]]))` — the rank and the scale
   from the checkpoint's state (`%g` for the scale) — and `Result := 0`
   (4.6, #18).

**Step 4 — the error catch and the release**

4a. Every raise of step 3 (the checkpoint's, the GGUF's, the targets',
   the adapters', the writer's) is caught by the single
   `except on E: Exception`: `AErr.Add(E.Message)` — the message alone,
   one line — and `Result` stays 1.
4b. The `finally` frees, in this order: `S.Free` (the `TLoraSet`) then
   `G.Free` (the `TGgufFile`) — on every path, also the exits 2 of
   steps that came after the `try` and the `Exit`s of 2b/2c/3b (those
   two come before the `try` and leave `G`/`S` nil).

### 4.6 Checks and messages (in this order)

All raised or written exactly as in the T18 facts ("Checks and messages
(in this order)"). `EGgufMerge` for the unit's own checks; the
pass-throughs keep their own classes and messages; the `RunMergeCLI`
rows come out as exit codes. A UT row that expects an error compares the
**whole** message.

`MergeTargets` (`EGgufMerge`), per target in order:

| # | Where | Condition | Message |
|---|---|---|---|
| 1 | `AddTarget`, `FindTensor` | `<name>.weight` is missing from the base's directory | `CreateFmt('no tensor %s.weight', [AName])` |
| 2 | `AddTarget`, `Tensor[T]` (after #1) | `Info.NDims <> 2` | `CreateFmt('%s.weight has %d dimensions, expected 2', [AName, Info.NDims])` |
| 3 | `ReadQwenDims` (T10, passes through) | a missing key | `EQwenBlock`, its own message — before #1–#2 |

`LoadAdapters` (pass-through):

| # | Where | Condition | Message |
|---|---|---|---|
| 4 | `TLoraSet.Load` (T12, passes through) | `lora.ot` missing | `ETorchNativeError`: `open file failed because of errno 2 on fopen: No such file or directory, file path: <dir>lora.ot` |

`MergeTensor` (`EGgufMerge`), before anything else:

| # | Where | Condition | Message |
|---|---|---|---|
| 5 | first statement, always first | `(AP.InDim <> AIn) or (AP.OutDim <> AOut)` | `CreateFmt('%s: adapter %d -> %d, tensor %d -> %d', [AP.Name, AP.InDim, AP.OutDim, AIn, AOut])` |
| 6 | then `DequantizeBytes` / `QuantizeQ8_0` (T02, pass through) | their own conditions | `EGgufError`, their own messages — after #5 |

`WriteMerged`:

| # | Where | Condition | Message |
|---|---|---|---|
| 7 | the pair assignment (4.4 step 1, before anything is written) | the first target, in order, without a pair in the set | `CreateFmt('no adapter %s', [name])` |
| 8 | `SetTensorType` / `BeginWrite` / … (T04, pass through) | the writer's own conditions (e.g. the part file cannot be created) | `EGgufError`, e.g. `TGgufWriter.BeginWrite: cannot create file <out>.part` |
| 9 | after the write (`RenameFile` failed) | the rename(2) failed (the part file is removed first) | `CreateFmt('cannot rename %s to %s', [Part, AOutPath])` |

`RunMergeCLI`, usage errors (**exit 2**, the message alone on `AErr`),
in this order:

| # | Where | Condition | Message |
|---|---|---|---|
| 10 | step 1a | `--base-sha256` is the last argument | `--base-sha256: value missing` |
| 11 | step 1a | any argument longer than one character that starts with `-`, compared literally | `unknown option <arg>` |
| 12 | step 1b | the positionals are not exactly three | `expected BASE CKPT OUT, got <n> arguments` |

`RunMergeCLI`, runtime errors (**exit 1**, the message alone on `AErr`),
in this order:

| # | Where | Condition | Message |
|---|---|---|---|
| 13 | step 2b | `not FileExists(Args[0])` | `no model file <base>` |
| 14 | step 2c | the expanded output path equals the expanded base path | `the output must not be the base model` |
| 15 | step 3a, from `ReadRunState` (T16, pass through) | `state.json` missing or invalid | `not a checkpoint: <ckpt>` |
| 16 | step 3b | both `St.BaseSha256` and `Sha` known and different (compared literally) | `the base model differs from the checkpoint's: <ckpt>` |
| 17 | steps 3c–3f | any exception's message — the GGUF's, the targets' (#1–#3), the adapters' (#4), the writing's (#7–#9) | the exception's `Message`, as is |

The success line (`AOut`, **exit 0**):

| # | Where | Message |
|---|---|---|
| 18 | step 3g | `merged <n> tensors (rank <r>, scale <s>) into <out>` (`%g` for the scale) |

Notes:

- The exit codes: 0 done (step 3g), 1 error (steps 2a/2b/2c/3b or step
  4a), 2 usage (steps 1a/1b) — external design "Exit codes".
- The sha256 is **never computed** (2d, 3b): it is the given
  `--base-sha256`, else the base's file name via `ShaFromName` (T17),
  else unknown — and the comparison runs only when both sides are known.
- The unit's own errors are `EGgufMerge`; the pass-throughs keep their
  own classes (`EQwenBlock` T10, `EGgufError` T02/T04,
  `ETorchNativeError` T12) and are caught by step 4a like the others.

### 4.7 Business rules of the external design and where they are handled

| Business rule (`T18-external.md`) | Handled by |
|---|---|
| the metadata copied as bytes from the base; the tensor directory rebuilt in the base's order, offsets recomputed for the new sizes, the base's alignment preserved | `TGgufWriter` (T04), driven from `WriteMerged` (4.4 steps 2–6) |
| each adapted tensor: the base dequantised to fp32 plus `s·B·A`, quantised to Q8_0; the adapted entries of the directory are type 8 | `MergeTensor` (4.3) and the retyping of 4.4 step 3 (`GgufTypeQ8_0`, 3) |
| every other tensor copied byte for byte | `WriteMerged` (4.4 step 5, the `Pairs[I] = nil` branch) |
| the targets per trunk block (the blocks minus the MTP count of `qwen35.nextn_predict_layers`): linear `attn_qkv`, `attn_gate`, `ssm_out`; full `attn_q`, `attn_k`, `attn_v`, `attn_output` — the kinds of `TQwenModel.AttachLora`, from the GGUF alone | `MergeTargets` (4.1.2) |
| each target's dimensions from the GGUF: `Dims[0]` = in, `Dims[1]` = out; a missing tensor or one that is not two-dimensional is an error | `AddTarget` (4.1.1), 4.6 #1–#2 |
| the adapters are the checkpoint's rank and scale, one CPU pair per target, loaded from `<ckpt>/lora.ot`; the caller frees the set | `LoadAdapters` (4.2), 4.6 #4 |
| the base's sha256: `--base-sha256 H`, else the file name `sha256-<64 hex>` (not computed); when both it and the checkpoint's `base_sha256` are known they must be equal (compared literally) | `RunMergeCLI` steps 2d, 3b; `ShaFromName` (T17); 4.6 #16 |
| the output written to `<out>.part` and renamed (rename(2) replaces an existing file); on a failure the part file is removed | `WriteMerged` (4.4 steps 4–7), 4.6 #8–#9 |
| the exit codes 0 (done), 1 (error), 2 (usage) | `RunMergeCLI` steps 1a/1b, 2a, 3g, 4a |
| the error list (external design, "Error Conditions") | 4.6 (#1–#18) |
| the success line `merged <n> tensors (rank <r>, scale <s>) into <out>` | `RunMergeCLI` step 3g, 4.6 #18 |

The fidelity criterion (external design "Fidelity Criterion": the
retyping floor at most 1.5×, at least 90 % of the loss gain kept, H47 at
most 0.02) is measured by the ITa cases ITa-T18-001 to 006 (system flow,
"ITa Cases"); it is not a check of the unit.

## 5. Change history

| Date | Change | Reason |
|---|---|---|
| T18a (initial) | First edition, from the external design, the system flow and the T18 facts. | new unit and program |
