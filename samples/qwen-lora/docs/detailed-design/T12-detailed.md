# Detailed design — unit `Lora`, and its hooks in `PackedLinear`, `QwenBlocks`, `QwenModel` (T12)

Based on the facts `tasks/T12-facts.md` (final, checked by the harness owner)
and DESIGN 8.12 ("Lora"), 8.13. T12 is the unit `Lora` (LoRA adapters:
`TLoraPair`, `TLoraSet`) and the places that use them: `TPackedLinear.Lora`
(unit `PackedLinear`), `TQwenBlock.Target` / `LoraTargets` / `Index` (unit
`QwenBlocks`) and `TQwenModel.AttachLora` (unit `QwenModel`). fp32 on the
CPU; a `TLoraSet` keeps one `TVarStore` per device, so the same code runs the
two-GPU split of the 27B. The tensor-op sequences below restate the facts'
final implementation; they describe the logic and operations, they are not a
copy of the code.

DESIGN 8.18's acceptance for T12: "zero `B` leaves the output unchanged;
`MergedDelta` equals the two-product form". Decisions fixed by the facts
(DESIGN 8.13 leaves them open):

- `A` is initialised uniform in ±1/√inDim (PyTorch's `kaiming_uniform_` with
  a = √5, as common LoRA code does), `B` is zero. Rank 16, scale 2
  (alpha 32) are the run's settings; **every routine takes them as
  parameters**.
- **Ownership**: a pair does not own its tensors' store; the set owns the
  pairs and the stores; a matrix (`TPackedLinear`) holds its pair without
  owning it. Free the model (or stop calling it) before the set.
- **Names** follow the GGUF: pair `blk.12.attn_qkv`, variables
  `blk.12.attn_qkv.lora_a` and `blk.12.attn_qkv.lora_b` (the store's path
  is split at the dots).
- `Save` / `Load` write and read every pair through a **temporary store on
  the CPU**, so a file does not depend on the device map. One tch archive
  (`.ot`, T07's `TVarStore.Save` format). Rank and scale are **not** in the
  file (T16's checkpoint keeps the run's settings).
- **Targets** (DESIGN 8.13): a linear block adapts `attn_qkv`, `attn_gate`,
  `ssm_out`; a full block `attn_q`, `attn_k`, `attn_v`, `attn_output`; the
  trunk blocks only (not the MTP block). On the rehearsal 0.8B that is 78
  pairs and 4 915 200 parameters at rank 16 (measured, ITa-T12-001).

## 1. Module structure

| Source file | Contents |
|-------------|----------|
| `src/Lora.pas` (new) | Unit `Lora`: `ELora`, `TLoraPair` (`Create`, `Delta`, `MergedDelta`), `TLoraSet` (`Create`, `Destroy`, `StoreFor` (private), `Add`, `Count`, `Pair`, `Find`, `Stores`, `ParameterCount`, `Save`, `Load`), the function `LoraPath`, `TVarStoreArray` |
| `src/PackedLinear.pas` | `uses` gains `Lora`; `TPackedLinear` gains the field `FLora`, the setter `SetLora` and the property `Lora`; `Forward` gains two lines at its end (the rest stays as it is) |
| `src/QwenBlocks.pas` | `TQwenBlock` gains the field `FIndex`, the virtual `Target` (base body), the abstract `LoraTargets` and the property `Index`; `TLinearBlock` and `TFullBlock` each gain `Target` and `LoraTargets` overrides; `TQwenBlock.Load` sets `FIndex := ABlock;` right after `FDevice := ADevice;` |
| `src/QwenModel.pas` | `uses` gains `Lora`; `TQwenModel` gains `AttachLora` |

| Function | Unit | Purpose |
|----------|------|---------|
| `LoraPath(P, AName)` | Lora | Append the dot-separated name's parts to a `TVarPath` |
| `TLoraPair.Create` | Lora | Register `lora_a` (uniform ±1/√in) and `lora_b` (zeros) under `P`; 1 check (§8) |
| `TLoraPair.Delta(X)` | Lora | `s·((X·Aᵀ)·Bᵀ)`; A/B cast to X's kind; 1 check (§8) |
| `TLoraPair.MergedDelta()` | Lora | `s·B·A`, [OutDim, InDim], fp32, without a graph |
| `TLoraSet.Create` | Lora | Rank and scale of the set; 1 check (§8) |
| `TLoraSet.Destroy` | Lora | Free the pairs, then the stores |
| `TLoraSet.StoreFor` (private) | Lora | The set's one store per device, created on first use |
| `TLoraSet.Add` | Lora | A new pair under the set's store for the device; 1 check (§8) |
| `TLoraSet.Count` / `Pair` / `Find` / `Stores` | Lora | Pair list access (1 check in `Pair`, §8); the stores in first-appearance order |
| `TLoraSet.ParameterCount` | Lora | Σ Rank·(InDim + OutDim) over the pairs |
| `TLoraSet.Save` / `Load` | Lora | The whole set to / from one `.ot` file on the CPU |
| `TPackedLinear.SetLora` / `Lora` | PackedLinear | Attach (or clear) a pair, dims checked; not owned |
| `TPackedLinear.Forward` (2 new lines) | PackedLinear | `y = x·Wᵀ + s·((x·Aᵀ)·Bᵀ)` when a pair is attached |
| `TQwenBlock.Target` (base + 2 overrides) | QwenBlocks | The matrix named `AName`, or `EQwenBlock` (§8) |
| `TQwenBlock.LoraTargets` (2 overrides) | QwenBlocks | The names DESIGN 8.13 adapts in this kind of block |
| `TQwenBlock.Index` | QwenBlocks | The block number (set in `Load`) |
| `TQwenModel.AttachLora` | QwenModel | One pair per target of every trunk block, attached to its matrix |

Dependencies: `Lora` needs `Tch` (tensors) and `TchNN` (`TVarStore`,
`TVarPath`, `NoGradGuard`, `TDevice`). `PackedLinear` gains `Lora` in its
`uses`; `QwenModel` gains `Lora`. `QwenBlocks` needs no new unit. Tensors
are managed records (no `Free`); the classes are freed by hand.

## 2. `LoraPath`

`LoraPath(const P: TVarPath; const AName: string): TVarPath` — the path `P /
'blk' / '12' / 'attn_qkv'` for the name `'blk.12.attn_qkv'`:

1. `Parts := AName.Split(['.'])` — the name split on `.`
2. `Result := P`; for `I := 0 to High(Parts)`: `Result := Result / Parts[I]`
   — each part appended as a path segment

No tensor ops. An empty part (e.g. `'blk..q'`) raises tch-fpc's
`ETorchError` when the segment is appended: `sub name cannot be empty or
contain "." or "|": ""` (facts, passed on unchanged; UT-Lora-003, 114).

## 3. `TLoraPair`

Fields: `FName: string`, `FA`, `FB: TTensor`, `FInDim`, `FOutDim`,
`FRank: Int64`, `FScale: Double`. Properties: `Name`, `A`, `B`, `InDim`,
`OutDim`, `Rank`, `Scale`. The pair registers its two variables under the
path `P` it is created on (that path is `LoraPath(store.Root, Name)`, built
by the creator); it does not own the store.

### 3.1 `Create(P, AName, AInDim, AOutDim, ARank, AScale)`

The steps run in exactly this order (check in §8):

1. `inherited Create`
2. Check: `(AInDim < 1) or (AOutDim < 1)` raises `ELora` `CreateFmt('%s:
   dimensions must be positive, got %d -> %d', [AName, AInDim, AOutDim])` —
   **before any variable is registered**, so the store stays empty
3. `FName := AName`; `FInDim := AInDim`; `FOutDim := AOutDim`;
   `FRank := ARank`; `FScale := AScale`
4. **Tensor ops:**
   - `Bound := 1 / Sqrt(AInDim)`
   - `FA := P.Uniform('lora_a', [ARank, AInDim], -Bound, Bound)` — `A`
     [Rank, InDim] fp32 on the store's device, uniform in ±1/√inDim,
     registered trainable
   - `FB := P.Zeros('lora_b', [AOutDim, ARank])` — `B` [OutDim, Rank] fp32,
     zeros, registered trainable

**Why B starts at zero:** with `B = 0` the delta is exactly zero, so an
attached model starts in exactly the pretrained state — the acceptance
"zero `B` leaves the output unchanged" (ITa-T12-002 is exact against a
model without adapters). `B` is the weight that is trained to move the
output; `A` is a fixed random fan-in normalised start.

### 3.2 `Delta(X)`

`s·((X·Aᵀ)·Bᵀ)`; `X` [.., InDim]; `A` and `B` are cast to X's kind.

1. Check (message in §8): `if (Length(S) = 0) or (S[High(S)] <> FInDim)`
   raises, where `S := X.Shape` — X with no dimension, or whose last
   dimension is not `InDim`
2. `Ac := FA; Bc := FB;` — the original handles by default
3. If `X.Kind <> Ac.Kind`: `Ac := Ac.totype(X.Kind)` and
   `Bc := Bc.totype(X.Kind)` — casts only of the locals, `FA`/`FB` keep
   their kind (UT-Lora-016: X kDouble, `P.A.Kind` still kFloat)
4. **Tensor ops:** `Result := X.matmul(Ac.t).matmul(Bc.t) * FScale`
   - `X.matmul(Ac.t)` — [.., InDim] · [InDim, Rank] → [.., Rank]
   - `.matmul(Bc.t)` — [.., Rank] · [Rank, OutDim] → [.., OutDim]
   - `* FScale` — the elementwise scale

### 3.3 `MergedDelta()`

`s·B·A`, [OutDim, InDim], fp32, without a graph:

1. `G := NoGradGuard` — autograd off for the call (assigned to a local
   `IInterface`)
2. **Tensor ops:** `Result := FB.mm(FA) * FScale` — [OutDim, Rank] ·
   [Rank, InDim] → [OutDim, InDim], then the scale

This is the matrix a merged GGUF stores in place of `W`: `W + MergedDelta`
(the `Forward` without the delta equals `x·(Wᵀ + (s·B·A)ᵀ)`, UT-Lora-014,
ITa-T12-003).

## 4. `TLoraSet`

Fields: `FRank: Int64`, `FScale: Double`, `FStores: TVarStoreArray`,
`FPairs: array of TLoraPair`. Properties: `Rank`, `Scale` (read). `TVarStoreArray
= array of TVarStore`. The set **owns** the pairs and the stores.

### 4.1 `Create(ARank, AScale)`

1. `inherited Create`
2. Check: `ARank < 1` raises `ELora` `CreateFmt('rank must be positive, got %d', [ARank])`
3. `FRank := ARank; FScale := AScale`

No tensor ops. The set starts empty.

### 4.2 `Destroy`

1. Free `FPairs[0..High(FPairs)]` in order
2. Free `FStores[0..High(FStores)]` in order
3. `inherited Destroy`

Pairs first, then stores (a `TVarPath` must not outlive its store; a pair
does not own its store, so the set must free it).

### 4.3 `StoreFor(ADevice)` (private)

One store per device, in the order their devices first appeared:

1. `for I := 0 to High(FStores)`: if `FStores[I].Device.ToString =
   ADevice.ToString` → `Exit(FStores[I])` — the existing store
2. `Result := TVarStore.Create(ADevice)`; `SetLength(FStores, Length(FStores) +
   1)`; `FStores[High(FStores)] := Result`

No tensor ops of its own (the store creation is one per device).

### 4.4 `Add(AName, AInDim, AOutDim, ADevice)`

1. Check (message in §8): `if Find(AName) <> nil` raises `ELora`
   `'pair already present: ' + AName` — names compared literally
2. `StoreFor(ADevice)` — the store for the device, created on first use
3. Path: `LoraPath(StoreFor(ADevice).Root, AName)` (§2)
4. `Result := TLoraPair.Create(<that path>, AName, AInDim, AOutDim, FRank,
   FScale)` — registers `lora_a` / `lora_b` under the store (§3.1)
5. `SetLength(FPairs, Length(FPairs) + 1)`; `FPairs[High(FPairs)] := Result`

### 4.5 `Count`, `Pair(I)`, `Find(AName)`, `Stores`

- `Count`: `Result := Length(FPairs)`. No tensor ops.
- `Pair(I)`: check (message in §8): `(I < 0) or (I >= Length(FPairs))`
  raises `ELora` `CreateFmt('pair %d outside 0..%d', [I, High(FPairs)])`;
  otherwise `Result := FPairs[I]`. No tensor ops.
- `Find(AName)`: linear scan `for I := 0 to High(FPairs)`:
  `if FPairs[I].Name = AName then Exit(FPairs[I])`; otherwise
  `Result := nil`. Names compared literally (case-sensitive; UT-Lora-105).
  No tensor ops.
- `Stores`: `Result := Copy(FStores)` — a copy of the store array, in the
  order their devices first appeared. No tensor ops.

### 4.6 `ParameterCount`

`Result := 0;` then `for I := 0 to High(FPairs)`:
`Inc(Result, FRank * (FPairs[I].InDim + FPairs[I].OutDim))` —
Σ Rank·(InDim + OutDim). No tensor ops.

### 4.7 `Save(AFileName)`

The whole set into one file, device-map independent (temporary CPU store,
`.ot`, T07's `TVarStore.Save` format). Rank and scale are not in the file.

1. `G := NoGradGuard` — the copies run without a graph
2. `S := TVarStore.Create(TDevice.Cpu)` — the temporary store
3. `for I := 0 to High(FPairs)`, per pair:
   - `P := LoraPath(S.Root, FPairs[I].Name)`
   - `P.Add('lora_a', FPairs[I].A.&to(TDevice.Cpu).detach.Copy, False)`
     — **Tensor ops:** move `A` to the CPU, detach (no graph), copy to
     fresh storage, register under `<name>/lora_a`
   - `P.Add('lora_b', FPairs[I].B.&to(TDevice.Cpu).detach.Copy, False)` —
     the same for `B` under `<name>/lora_b`
4. `S.Save(AFileName)` — the tch archive
5. `finally S.Free`

### 4.8 `Load(AFileName)`

Every pair must be in the file with its shape; `A` and `B` keep their
handles (the values are copied in place).

1. `S := TVarStore.Create(TDevice.Cpu)` — the temporary store
2. `for I := 0 to High(FPairs)`, per pair — **Tensor ops** (placeholder
   zeros): `P := LoraPath(S.Root, FPairs[I].Name)`;
   `P.Zeros('lora_a', [FRank, FPairs[I].InDim])`;
   `P.Zeros('lora_b', [FPairs[I].OutDim, FRank])`. Pre-creating the
   variables at the pair's shapes is what makes `TVarStore.Load` report a
   missing name (`ETorchError`) or a different shape
   (`ETorchNativeError`) instead of silently skipping (§8)
3. `S.Load(AFileName)` — the file's values into the temporary store
4. `G := NoGradGuard` — the back-copies run without a graph
5. `for I := 0 to High(FPairs)`, per pair — **Tensor ops:**
   - `S.TryGet(FPairs[I].Name + '.lora_a', Fa)`;
     `S.TryGet(FPairs[I].Name + '.lora_b', Fb)` — from the store by full
     name
   - `FPairs[I].A.CopyFrom(Fa)`; `FPairs[I].B.CopyFrom(Fb)` — the values
     copied into the pair's own tensors (same handles, `RequiresGrad`
     kept); a shape mismatch here raises the copy's `ETorchNativeError`
6. `finally S.Free`

## 5. `TPackedLinear.SetLora` and `Forward`

The class gains the field `FLora: TLoraPair` (nil by default; **not
owned**), the private setter `SetLora` and the property
`Lora: TLoraPair read FLora write SetLora`.

### 5.1 `SetLora(AValue)`

1. Check (message in §8): `if (AValue <> nil) and ((AValue.InDim <>
   FInDim) or (AValue.OutDim <> FOutDim))` raises `EPackedLinear`
   `CreateFmt('LoRA pair %s is %d -> %d, the matrix %d -> %d', [AValue.Name,
   AValue.InDim, AValue.OutDim, FInDim, FOutDim])`; **`Lora` is unchanged**
   on the raise
2. `FLora := AValue` — the attach; `nil` clears again

No tensor ops.

### 5.2 `Forward(X)` — the adapted matrix

`y = x·Wᵀ + s·((x·Aᵀ)·Bᵀ)`. The first part stays as it is; two lines are
added at the end:

1. Checks (messages in §8), in this order: `X.Kind <> kFloat` raises
   `EPackedLinear` `X must be kFloat (fp32)`; then `(Length(S) = 0) or
   (S[High(S)] <> FInDim)` raises `EPackedLinear`
   `CreateFmt('X must end in %d, got %s', [FInDim, ShapeStr(X)])`
2. **Tensor ops:**
   - `Result := X.matmul(Weight.transpose(0, 1))` — `x·Wᵀ`, [.., InDim] →
     [.., OutDim] (the unchanged base product)
   - New: `if FLora <> nil then Result := Result + FLora.Delta(X)` — the
     `s·((x·Aᵀ)·Bᵀ)` of §3.2 added to the base product

With `B = 0` the delta is exactly zero, so `Forward` with an attached pair
equals the unadapted `Forward` bit for bit (UT-PackedLinear-102, ITa-T12-002
exact).

## 6. `TQwenBlock.Target`, `LoraTargets`, `Index`

`TQwenBlock` gains the field `FIndex: Integer`, the virtual function
`Target` (base body below), the abstract function `LoraTargets` and the
property `Index read FIndex`. `TLinearBlock` and `TFullBlock` each override
both functions. `TQwenBlock.Load` sets `FIndex := ABlock;` right after
`FDevice := ADevice;`. Names are compared **literally** (case-sensitive;
`'attn_q.weight'` and `''` fall through to the base raise). No tensor ops
in any of the five routines: `Target` returns a field and
`LoraTargets` builds a name array.

### 6.1 `TQwenBlock.Target` (base)

1. `AName = 'ffn_gate'` → `Result := FFfnGate`
2. else `AName = 'ffn_up'` → `Result := FFfnUp`
3. else `AName = 'ffn_down'` → `Result := FFfnDown`
4. else raise `EQwenBlock` `CreateFmt('blk.%d has no matrix %s', [FIndex, AName])`

### 6.2 `TLinearBlock.Target` (override)

In this order: `'attn_qkv'` → `FQkv`; `'attn_gate'` → `FGate`;
`'ssm_beta'` → `FBeta`; `'ssm_alpha'` → `FAlpha`; `'ssm_out'` → `FSsmOut`;
otherwise `inherited Target(AName)` (the base of §6.1 for the `ffn_*`
names, else the raise).

### 6.3 `TFullBlock.Target` (override)

In this order: `'attn_q'` → `FQ`; `'attn_k'` → `FK`; `'attn_v'` → `FV`;
`'attn_output'` → `FOut`; otherwise `inherited Target(AName)`.

### 6.4 `LoraTargets` (overrides)

- `TLinearBlock.LoraTargets`: `['attn_qkv', 'attn_gate', 'ssm_out']`
- `TFullBlock.LoraTargets`: `['attn_q', 'attn_k', 'attn_v', 'attn_output']`

These are exactly the names DESIGN 8.13 adapts in the kind of block (the
`ssm_beta` / `ssm_alpha` matrices of a linear block and the `ffn_*` matrices
are **not** adapted).

## 7. `TQwenModel.AttachLora`

`AttachLora(ASet: TLoraSet): Integer` — adds to `ASet` one pair per LoRA
target of every **trunk** block (`'blk.<i>.<target>'`, on the block's
device) and attaches it to the matrix; returns the number of pairs added.

1. `Result := 0`
2. `for I := 0 to FTrunk - 1` — trunk blocks only; the MTP block (if any)
   is never adapted
   1. `Names := FBlocks[I].LoraTargets`
   2. `for J := 0 to High(Names)`:
      - `L := FBlocks[I].Target(Names[J])` — the matrix (§6)
      - `L.Lora := ASet.Add('blk.' + IntToStr(I) + '.' + Names[J],
        L.InDim, L.OutDim, L.Device)` — the pair registered in the set's
        store for the matrix's device (§4.4, §3.1 — the only tensor ops of
        the routine: the `lora_a` / `lora_b` initialisations), attached to
        the matrix (§5.1)
      - `Inc(Result)`
3. Return `Result`

Raises what `Add` raises (a second call on the same set: `pair already
present: blk.0.attn_qkv`, facts/§8). On the rehearsal 0.8B: 78 pairs,
4 915 200 parameters at rank 16 (UT-QwenModel-201–207, ITa-T12-001).

## 8. Checks and messages (in order)

Every message as it runs, with the exact text from the facts. UT rows catch
the named exception class and compare the whole message (`AssertEquals`); a
row fails if no exception comes.

### 8.1 `ELora` — unit `Lora`

1. **`TLoraPair.Create`**, when `AInDim < 1` or `AOutDim < 1`, **before
   any variable is registered**: `CreateFmt('%s: dimensions must be
   positive, got %d -> %d', [AName, AInDim, AOutDim])`, e.g. `p: dimensions
   must be positive, got 0 -> 6` (UT-Lora-006), `8 -> -1` (007).
2. **`TLoraPair.Delta`**, when X has no dimension or its last one is not
   `InDim`: `CreateFmt('%s: X must end in %d', [FName, FInDim])`, e.g. `p: X
   must end in 8` (UT-Lora-011, 012).
3. **`TLoraSet.Create`**, when `ARank < 1`: `CreateFmt('rank must be
   positive, got %d', [ARank])`, e.g. `rank must be positive, got 0`
   (UT-Lora-102).
4. **`TLoraSet.Add`**, when the pair is already present (names compared
   literally): `'pair already present: ' + AName`, e.g. `pair already
   present: blk.0.attn_qkv` (UT-Lora-104, UT-QwenModel-205).
5. **`TLoraSet.Pair`**, for an index outside the list: `CreateFmt('pair %d
   outside 0..%d', [I, High(FPairs)])`, e.g. `pair 2 outside 0..1`, `pair
   -1 outside 0..1` (UT-Lora-106), `pair 0 outside 0..-1` on an empty set
   (107).
6. **From tch-fpc, passed on unchanged:**
   - an empty segment in a name (`TVarPath ./`, raised by `LoraPath`):
     `ETorchError` `sub name cannot be empty or contain "." or "|": ""`
     (UT-Lora-003, 114)
   - `Load` of a file without one of the set's variables: `ETorchError`
     `variable <name> not found in <file>`, e.g. `variable
     blk.2.attn_qkv.lora_a not found in <dir>c.ot` (UT-Lora-111)
   - `Load` of a variable of another shape: `ETorchNativeError` from the
     copy, e.g. `The size of tensor a (3) must match the size of tensor b
     (4) at non-singleton dimension 0` (UT-Lora-109)
   - `Load` of a missing file: `ETorchNativeError` `open file failed because
     of errno 2 on fopen: No such file or directory, file path: <file>`
     (UT-Lora-110)

### 8.2 `EPackedLinear` — `SetLora`

When a non-nil pair's dimensions differ: `CreateFmt('LoRA pair %s is %d ->
%d, the matrix %d -> %d', [AValue.Name, AValue.InDim, AValue.OutDim,
FInDim, FOutDim])`, e.g. `LoRA pair q is 7 -> 6, the matrix 8 -> 6`
(UT-PackedLinear-104), `LoRA pair q is 8 -> 5, the matrix 8 -> 6` (105);
`Lora` is unchanged on the raise.

### 8.3 `EQwenBlock` — `Target`

For a name the kind (and the base) does not know: `CreateFmt('blk.%d has no
matrix %s', [FIndex, AName])`, names compared literally: `blk.0 has no
matrix attn_q` (UT-QwenBlocks-303), `blk.1 has no matrix ATTN_Q` (304),
`blk.1 has no matrix attn_q.weight` (305), `blk.1 has no matrix ` (the name
empty, 306).

### 8.4 `AttachLora`

Raises what `Add` raises; a second call on the same set gives `pair already
present: blk.0.attn_qkv` (UT-QwenModel-205; the driver stops on it, facts).

## 9. Testing notes (from the facts)

- Stages and case-ID ranges: stage 1 (T12b1) `LoraPath`, `TLoraPair` — rows
  `UT-Lora-001` to 016; stage 2 (T12b2) `TLoraSet` — rows `UT-Lora-101` to
  114; stage 3 (T12b3) `TPackedLinear.SetLora` / `Forward` and
  `TQwenBlock` `Index` / `Target` / `LoraTargets` — rows
  `UT-PackedLinear-101` to 107, `UT-QwenBlocks-301` to 308; stage 4 (T12b4)
  `TQwenModel.AttachLora` — rows `UT-QwenModel-201` to 207.
- The unit tests run against the tiny models of
  `tests/ut/QwenBlocksFixture.pas` (block 0 linear, block 1 full; Hidden 64)
  with the harness owner's helpers and fixture routines in `Lora_test`,
  `PackedLinear_test`, `QwenBlocks_test` and `QwenModel_test` (every value,
  shape, message and limit is in the facts' "Unit tests" tables; e.g. the
  fixed pair `A[k,c] = 0.1(k+1) − 0.05c`, `B[o,k] = 0.03(o−k)`, X
  `sin(1+tC+c)`, `RefDelta` in Double). Every row has one method; error
  rows catch the named exception class and compare the whole message;
  tolerance rows pass when the difference is ≤ the limit, "exact" means
  equal strings or a difference of 0.
- Measured: every row passes; the four units have 100 % line coverage; the
  mutants are caught (no scale in `Delta`: Lora 009, 010, 014–016,
  PackedLinear 103; `AttachLora` skipping the last block: QwenModel 201,
  205–207; `Load` not copying `B`: Lora 108; `Forward` without the delta:
  PackedLinear 103, 107, QwenModel 203, 204).
- ITa on the rehearsal model (driver `tests/ita/T12/lora_check.pas`,
  `tests/ita/T12-run.sh`; rank 16, scale 2, seed 12; about a minute):
  ITa-T12-001 (pairs/dimensions/ParameterCount vs the tensor directory;
  limit 0), 002 (B = 0: `LastLogProbs` with adapters vs without, limit 0
  exact), 003 (B random 0.02 N(0,1): adapters vs a W + s·B·A GGUF base
  forward, limit 1e-3), 004 (chat-065, B random: A/B gradients block-wise
  vs one backward through the whole graph, limit 1e-4), 005 (Save, then
  Load into a fresh set on a fresh model, every A and B, limit 0 exact).
  Measured with the prototype: 0, 0, 4.8e-5, 0, 0 (the adapters move the
  log-probabilities by up to 9.8).

## 10. Change history

| Date | Change | Reason |
|------|--------|--------|
| 2026-11-23 | Initial creation | T12a delegation 1 of 2: the T12 documents |
