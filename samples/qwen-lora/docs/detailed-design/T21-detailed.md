# Detailed design — bf16 compute on the trunk matrix products, and the CPU adapter draw (T21)

Based on the external design `docs/external-design/T21-external.md`, the system flow
`docs/system-flow/T21-system.md` and the facts `tasks/T21-facts.md` (final, checked by the
harness owner). This document restates and organises them; it adds no new requirements.

T21 adds a compute kind to five existing units:

- the matrix products of the **trunk blocks** can run in bf16 — a `TPackedLinear` now remembers
  in what kind its products run (`Compute`, set per matrix by `SetCompute`) and `TQwenModel`
  fans the kind out to every trunk matrix (`Compute` / `SetCompute`) while the **head stays fp32**;
- the LoRA adapter matrix `A` is **drawn on the CPU** in `TLoraPair.Create`, so a run starts from
  the same adapter whatever the device map (copy it into the pair's store with `VarCopy`);
- both programs take the option **`--compute fp32|bf16`** — `qlora_train` (`TrainCli`) and
  `qlora_logprobs` (`LogprobsCli`) — which select the kind at run time.

The default is **fp32 and the fp32 path is unchanged bit for bit** (when `Compute = kFloat`, the
old `Forward` body runs), so every earlier UT row, ITa and oracle stays as it is. bf16 also runs on
the CPU, so the UT and the ITa can check bf16 without a GPU; the two-card GPU check (M2g-b) is run
by the harness owner.

The implementation below is final, from the facts ("write it exactly so"); the Pascal snippets are
copied verbatim from the facts.

## 1. Module structure

| Source file | T21 change |
|-------------|-----------|
| `src/PackedLinear.pas` | `TPackedLinear` gains the field `FCompute: TKind`, the setter `SetCompute` and the property `Compute` (all already in the skeleton); the constructor already ends with `FCompute := kFloat;`; `SetCompute` gets its body (replacing the skeleton) and `Forward` gets the new two-branch body (the checks stay first, unchanged) |
| `src/Lora.pas` | one line of `TLoraPair.Create` changes: `lora_a` is drawn on the CPU and `VarCopy`'d into the pair's store |
| `src/QwenModel.pas` | `TQwenModel` gains the field `FCompute: TKind`, the setter `SetCompute` and the property `Compute` (all already in the skeleton); `Load` already sets `FCompute := kFloat`; `Logits` is unchanged (fp32); `SetCompute` gets its body (replacing the skeleton) |
| `src/TrainCli.pas` | the `--compute` branch of `ParseTrainArgs`, and the line `M.Compute := C.Compute` in `RunTrainCLI`; the `--compute K` header line, the `TTrainConfig.Compute` field and `DefaultTrainConfig` setting `kFloat` are already in the skeleton; `ConfigHash` is unchanged (the kind is not in the hash) |
| `src/LogprobsCli.pas` | in `RunLogprobsCLI` a local `Compute: TKind`, the `--compute` branch of the option loop, and the line `M.Compute := Compute`; the `K_USAGE` line already reads `[--compute K]` |

| Function | Unit | Purpose |
|----------|------|---------|
| `TPackedLinear.Compute` | PackedLinear | read/write the kind of the matrix products (`kFloat` default, `kBFloat16`) |
| `TPackedLinear.SetCompute` | PackedLinear | check the kind is `kFloat` or `kBFloat16`; store it |
| `TPackedLinear.Forward` | PackedLinear | `X·Wᵀ` (+ the LoRA `Delta`) in the chosen kind; the fp32 branch is the old body, the bf16 branch casts, multiplies, adds the delta of the bf16 `X`, casts back |
| `TLoraPair.Create` (1 line) | Lora | `lora_a` drawn on the CPU, `VarCopy`'d into the store |
| `TQwenModel.Compute` | QwenModel | read/write the kind of the trunk-block matrix products |
| `TQwenModel.SetCompute` | QwenModel | check the kind; set `Compute` on every trunk-block matrix; the head is left untouched |
| `ParseTrainArgs` (`--compute` branch) | TrainCli | parse `--compute fp32|bf16` (compared literally) into `C.Compute` |
| `RunTrainCLI` (1 line) | TrainCli | `M.Compute := C.Compute` after the model is loaded, before `ManualSeed(C.Seed)` |
| `RunLogprobsCLI` (option + 1 line) | LogprobsCli | parse `--compute`, default `kFloat`; `M.Compute := Compute` after the model is loaded |

Dependencies: no new unit is added. `PackedLinear` and `QwenModel` use the tch-fpc kind
`kBFloat16` and the tensor op `totype`. `Lora` uses the tensor factory `TTensor.zeros` (the CPU
device) and `TVarPath.VarCopy`. `TrainCli` and `LogprobsCli` use the kind constants `kFloat` /
`kBFloat16` and the model's `Compute` property. `TTensor` is a managed record (never freed).
Nothing else changes.

## 2. Unit `PackedLinear` — `Compute`, `SetCompute`, `Forward`

A `TPackedLinear` still keeps the packed bytes and dequantises the weight to fp32 on every call
(`Weight`, T10). It now also remembers in what kind its matrix products must run. Two valid kinds:
`kFloat` (the default, the old behaviour) and `kBFloat16`.

### 2.1 The property `Compute` (interface, already in the skeleton)

```pascal
  private
    FCompute: TKind;
    procedure SetCompute(AValue: TKind);
  public
    { The kind of the matrix products: kFloat (the default) or kBFloat16.
      With kBFloat16, X (fp32) and the weight are cast to bf16, the LoRA
      Delta gets the bf16 X, and the sum is cast back to fp32. }
    property Compute: TKind read FCompute write SetCompute;
```

The constructor already ends with `FCompute := kFloat;`, so a matrix that is never told otherwise
stays fp32.

### 2.2 `SetCompute(AValue: TKind)`

Check first (message in §7.1): a kind that is neither `kFloat` nor `kBFloat16` raises
`EPackedLinear`. Because the assignment `FCompute := AValue` runs only after the check, **`Compute`
keeps its value** when the check raises.

```pascal
procedure TPackedLinear.SetCompute(AValue: TKind);
begin
  if (AValue <> kFloat) and (AValue <> kBFloat16) then
    raise EPackedLinear.Create('compute must be kFloat or kBFloat16');
  FCompute := AValue;
end;
```

No tensor ops.

### 2.3 `Forward(const X: TTensor): TTensor` — both branches

The two T10 checks stay first and unchanged (messages in §7.2); `X must be kFloat (fp32)` is enforced
**also when `Compute` is `kBFloat16`** — the input must arrive in fp32 and the branch does the
casting. Then the product runs in the stored kind.

**The fp32 branch (`FCompute = kFloat`) is the old body, unchanged bit for bit:**

1. `Result := X.matmul(Weight.transpose(0, 1))` — `X` [.., InDim] · [InDim, OutDim] → [.., OutDim],
   the base product on the dequantised fp32 weight
2. `if FLora <> nil then Result := Result + FLora.Delta(X)` — the LoRA `s·((X·Aᵀ)·Bᵀ)` (T12) added,
   where `Delta` casts `A` / `B` to `X`'s kind (`kFloat` here)

**The bf16 branch** (only when `FCompute = kBFloat16`):

1. `Xc := X.totype(FCompute)` — the fp32 `X` cast to bf16
2. `Result := Xc.matmul(Weight.totype(FCompute).transpose(0, 1))` — the bf16 product: the
   dequantised fp32 weight `Weight` is cast to bf16 and transposed to [InDim, OutDim]
3. `if FLora <> nil then Result := Result + FLora.Delta(Xc)` — the `Delta` fed the **bf16** `X`
   (`Delta` then casts `A` / `B` to bf16, T12)
4. `Result := Result.totype(kFloat)` — the sum cast back to fp32

The full body:

```pascal
function TPackedLinear.Forward(const X: TTensor): TTensor;
var
  S: TInt64Array;
  Xc: TTensor;
begin
  S := X.Shape;
  if X.Kind <> kFloat then
    raise EPackedLinear.Create('X must be kFloat (fp32)');
  if (Length(S) = 0) or (S[High(S)] <> FInDim) then
    raise EPackedLinear.CreateFmt('X must end in %d, got %s', [FInDim, ShapeStr(X)]);
  if FCompute = kFloat then
  begin
    Result := X.matmul(Weight.transpose(0, 1));
    if FLora <> nil then
      Result := Result + FLora.Delta(X);
  end
  else
  begin
    Xc := X.totype(FCompute);
    Result := Xc.matmul(Weight.totype(FCompute).transpose(0, 1));
    if FLora <> nil then
      Result := Result + FLora.Delta(Xc);
    Result := Result.totype(kFloat);
  end;
end;
```

With `FCompute = kFloat` this is exactly the T12 body (`X·Wᵀ` + the `Delta` of the fp32 `X`), so the
fp32 results — and the earlier UT/ITa/oracle — are unchanged.

## 3. Unit `Lora` — the CPU draw in `TLoraPair.Create`

`TLoraPair.Create` (T12) is unchanged except for one line. The check, the field stores and the
`lora_b` zero are as they are; only the `lora_a` initialisation changes — it is drawn **on the CPU**
and copied into the pair's store with `VarCopy`, instead of being drawn directly on the store's
device. `Bound` stays `1 / Sqrt(AInDim)`; `B` stays zero (so an attached model starts in exactly the
pretrained state, T12).

The one changed line (replaces `FA := P.Uniform('lora_a', [ARank, AInDim], -Bound, Bound);`):

```pascal
FA := P.VarCopy('lora_a', TTensor.zeros([ARank, AInDim], kFloat, TDevice.Cpu).uniform_(-Bound, Bound));
```

**Tensor ops** of that line:

1. `TTensor.zeros([ARank, AInDim], kFloat, TDevice.Cpu)` — a [Rank, InDim] fp32 zero tensor on the
   **CPU** (fresh storage, not in any store)
2. `.uniform_(-Bound, Bound)` — drawn in place into it, so the value is the same uniform draw a CPU
   store would make (the same operation as `TVarPath.Uniform` on a CPU store, T12's values)
3. `P.VarCopy('lora_a', …)` — the CPU tensor copied into the pair's store `P` (which may be on a GPU)
   and registered trainable

**Why on the CPU:** a run must start from the same adapter whatever the device map. Measured at
M2g-a: with `A` drawn on P's own (GPU) generator, the two-card run started from a *different* `A`
and its gradient norm at update 1 was 3.44 against 3.77 on the CPU run. Drawing on the CPU fixes the
start (the CPU generator consumes identically for a given seed); only the copy into the store's device
is left to `VarCopy`.

## 4. Unit `QwenModel` — `Compute`, `SetCompute`

`TQwenModel` gains the same pair as `TPackedLinear` (field, setter, property), plus `Load` already
sets `FCompute := kFloat`. `Logits` is **unchanged and stays fp32** (the head is dequantised
`HeadChunk` rows at a time, so its fp32 product costs little memory). `SetCompute` fans the kind out
to the trunk-block matrices by name, through T12's `TQwenBlock.Target`.

### 4.1 The property `Compute` (interface, already in the skeleton)

```pascal
    { kFloat (the default) or kBFloat16: sets every matrix of the trunk
      blocks (TPackedLinear.Compute); the head stays fp32 (T21). }
    property Compute: TKind read FCompute write SetCompute;
```

(with the private `FCompute: TKind` and `procedure SetCompute(AValue: TKind)`).

### 4.2 `SetCompute(AValue: TKind)` — the three name lists

Check first (message in §7.3): a kind that is neither `kFloat` nor `kBFloat16` raises `EQwenModel`,
**before any matrix is changed** (the raise precedes the loops). Otherwise, for every trunk block
`I` in `0 .. FTrunk - 1` (the MTP block, if any, is not a trunk block and is never touched), set
`Compute := AValue` on:

- the three FFN matrices of **every** block — `K_FFN`; and
- the attention matrices of that block, by its kind — `K_FULL` when `IsFullBlock(FDims, I)`, else
  `K_LINEAR` (T10's kind test).

The three name lists (local constants of the routine), compared literally through `Target`:

| List | Names | Where |
|------|-------|-------|
| `K_FFN` | `ffn_gate`, `ffn_up`, `ffn_down` | every trunk block (3) |
| `K_FULL` | `attn_q`, `attn_k`, `attn_v`, `attn_output` | a full (gated-attention) block (4) |
| `K_LINEAR` | `attn_qkv`, `attn_gate`, `ssm_beta`, `ssm_alpha`, `ssm_out` | a linear (Gated-DeltaNet) block (5) |

So a full block gets `3 + 4 = 7` matrices and a linear block `3 + 5 = 8`; on the tiny fixture
(block 0 linear, block 1 full) that is **15 matrices** (8 + 7). These cover **every** trunk-block
matrix (a wider set than T12's LoRA targets, which adapt only `attn_qkv` / `attn_gate` / `ssm_out`
and `attn_q`..`attn_output`).

The full body:

```pascal
procedure TQwenModel.SetCompute(AValue: TKind);
const
  K_FFN: array[0..2] of string = ('ffn_gate', 'ffn_up', 'ffn_down');
  K_LINEAR: array[0..4] of string = ('attn_qkv', 'attn_gate', 'ssm_beta', 'ssm_alpha', 'ssm_out');
  K_FULL: array[0..3] of string = ('attn_q', 'attn_k', 'attn_v', 'attn_output');
var
  I, J: Integer;
begin
  if (AValue <> kFloat) and (AValue <> kBFloat16) then
    raise EQwenModel.Create('compute must be kFloat or kBFloat16');
  for I := 0 to FTrunk - 1 do
  begin
    for J := 0 to High(K_FFN) do
      FBlocks[I].Target(K_FFN[J]).Compute := AValue;
    if IsFullBlock(FDims, I) then
      for J := 0 to High(K_FULL) do
        FBlocks[I].Target(K_FULL[J]).Compute := AValue
    else
      for J := 0 to High(K_LINEAR) do
        FBlocks[I].Target(K_LINEAR[J]).Compute := AValue;
  end;
  FCompute := AValue;
end;
```

`FCompute := AValue` runs **last**, after every matrix has been set, so the model's `Compute` read
back always agrees with the matrices it has just been given. The head and the embedding are never
touched, so `Logits` stays fp32. No tensor ops of its own: each assignment is a `TPackedLinear`
`SetCompute` call (§2.2).

## 5. Unit `TrainCli` — the `--compute` option of `qlora_train`

The `--compute K` header line, the `TTrainConfig.Compute` field (`kFloat` or `kBFloat16`) and
`DefaultTrainConfig` setting `Result.Compute := kFloat` are already in the skeleton. `ConfigHash` is
**unchanged** — the kind is not part of the configuration hash, so a checkpoint written in fp32
resumes in bf16 and the other way round (UT-TrainCli-301: the hashes are equal).

### 5.1 `ParseTrainArgs` — the `--compute` branch

Added **right after the `--split` branch**. As for every valued option, the value is `V` and a
missing value is already refused by the loop's generic check (`if I + 1 >= N` → `Exit(A + ': value
missing')`), which therefore gives the existing `--compute: value missing`. The value is then
compared **literally** — `fp32` / `bf16` only, case sensitive, so `BF16` is refused:

```pascal
    else if A = '--compute' then
    begin
      if V = 'fp32' then C.Compute := kFloat
      else if V = 'bf16' then C.Compute := kBFloat16
      else Exit(Format('--compute: fp32 or bf16, got "%s"', [V]));
    end
```

### 5.2 `RunTrainCLI` — set the kind

The kind is applied **right after the model is loaded** (the `if C.Split < 0 then … else …` that
calls `TQwenModel.Load`) and **before `ManualSeed(C.Seed)`**:

```pascal
      M.Compute := C.Compute;
```

`DefaultTrainConfig` leaves `Compute = kFloat`, so a run without `--compute` loads the model, sets
`M.Compute := kFloat` and behaves exactly as before.

## 6. Unit `LogprobsCli` — the `--compute` option of `qlora_logprobs`

The `K_USAGE` line already reads `Usage: qlora_logprobs [--top N] [--split S] [--compute K] MODEL
FILE` and `EUsage` already exits **2** with two lines (`'qlora_logprobs: ' + E.Message`, then
`K_USAGE`). T21 adds one local, one branch of the option loop, and one line that sets the kind.

### 6.1 `RunLogprobsCLI` — the option

1. A local `Compute: TKind;` set to `kFloat` **right after `Top := 20;`** (the defaults are
   `Top := 20`, `Split := -1`).
2. The option loop matches `--compute` in the same clause as `--top` and `--split` (one value;
   without it, `<flag> needs a value` as for the others). Inside, `--compute` is dispatched on the
   literal value:

```pascal
      if (AArgs[I] = '--top') or (AArgs[I] = '--split') or (AArgs[I] = '--compute') then
      begin
        if I = High(AArgs) then
          raise EUsage.Create(AArgs[I] + ' needs a value');
        if AArgs[I] = '--top' then
          Top := ParseCount(AArgs[I + 1], '--top', 1, 100)
        else if AArgs[I] = '--compute' then
        begin
          if AArgs[I + 1] = 'fp32' then
            Compute := kFloat
          else if AArgs[I + 1] = 'bf16' then
            Compute := kBFloat16
          else
            raise EUsage.Create('--compute must be fp32 or bf16, got ' + AArgs[I + 1]);
        end
        else
          Split := ParseCount(AArgs[I + 1], '--split', 0, 1000);
        Inc(I, 2);
      end
```

(`--top` and `--split` keep their `ParseCount` ranges, 1..100 and 0..1000; note this program's
`--compute` message has **no** quotes around the value, unlike `qlora_train`'s.)

### 6.2 `RunLogprobsCLI` — set the kind

Right after the model is loaded (after the `if Split < 0 then … else …` that calls
`TQwenModel.Load`):

```pascal
      M.Compute := Compute;
```

## 7. Checks and messages (in order)

Every message as it runs, in the order the facts list them, with the exact text. UT rows catch the
named exception class and compare the whole message (`AssertEquals`); a row fails if no exception
comes. The `ParseTrainArgs` / `RunLogprobsCLI` messages are reported strings (usage errors), not
raised exceptions.

### 7.1 `EPackedLinear` — `TPackedLinear.SetCompute`

1. `AValue` other than `kFloat` / `kBFloat16`: `compute must be kFloat or kBFloat16`. **`Compute`
   keeps its value** on the raise (the assignment runs after the check).

### 7.2 `EPackedLinear` — `TPackedLinear.Forward`

The T10 checks, **first and unchanged** (also when `Compute` is `kBFloat16`):

2. `X` not kFloat: `X must be kFloat (fp32)`.
3. `X` 0-d or not ending in `FInDim`: `CreateFmt('X must end in %d, got %s', [FInDim, ShapeStr(X)])`,
   e.g. `X must end in 8, got [5, 7]`.

The branch itself raises nothing.

### 7.3 `EQwenModel` — `TQwenModel.SetCompute`

4. `AValue` other than `kFloat` / `kBFloat16`: `compute must be kFloat or kBFloat16`, raised
   **before any matrix is changed** (and before `FCompute := AValue`, so the model still reports
   its previous kind and its matrices are untouched).

### 7.4 `ParseTrainArgs` (unit `TrainCli`) — `--compute`

5. value not literally `fp32` / `bf16`: `--compute: fp32 or bf16, got "<value>"`, e.g.
   `--compute: fp32 or bf16, got "fp16"` and `--compute: fp32 or bf16, got "BF16"` (case sensitive).
6. `--compute` is the last argument (no value): the existing generic check gives
   `--compute: value missing`.

### 7.5 `RunLogprobsCLI` (unit `LogprobsCli`) — `--compute` (usage errors, exit 2)

7. `--compute` is the last argument (no value): `--compute needs a value`.
8. value not literally `fp32` / `bf16`: `--compute must be fp32 or bf16, got <value>`, e.g.
   `--compute must be fp32 or bf16, got fp16`.

## 8. Notes (from the facts)

- **Only the trunk-block matrix products run in bf16.** The residual stream, the RMS norms, RoPE,
  the convolution, the delta rule, the attention around the projections, the head and the loss all
  stay fp32 (DESIGN 8.19 R2).
- **The head stays fp32.** bf16 logits keep three significant digits (steps of 0.0625 at the
  rehearsal model's logits, measured); and the head is dequantised `HeadChunk` rows at a time, so
  its fp32 product costs little memory.
- **Default fp32, unchanged bit for bit:** when `Compute = kFloat` the old `Forward` body runs, so
  every earlier UT row, ITa and oracle stays as it is.
- **`--compute` is not in the configuration hash** (a run setting, like `--split`): a checkpoint
  written in fp32 resumes in bf16 and the other way round.
- **bf16 runs on the CPU** too, so the UT and the ITa check bf16 without a GPU; the two-card
  split-in-bf16-against-the-CPU-curve check is M2g-b, run by the harness owner. There the
  `lora_a`-draw-on-P's-`device` mutant cannot be seen, so M2g-b checks it by the gradient norm at
  update 1 against the CPU run.
- **UT** (harness owner's fixture and helpers; unchanged): every row of the facts' tables passes
  with `-Cr` (the earlier rows of the five test programs too); the five units then have 100 % line
  coverage (84, 112, 119, 250, 100 lines). Mutants are caught — bf16 product without the casts
  (PackedLinear 302, 303; QwenModel 302; TrainCli 302; LogprobsCli 301); `Delta` of the fp32 `X`
  (PackedLinear 303); no cast back to fp32 (PackedLinear 302, 303; QwenModel 302; TrainCli 302;
  LogprobsCli 301); any kind accepted (PackedLinear 301; QwenModel 301); `FCompute` not set (PackedLinear 301); a
  matrix of the full block or of the FFN left out, or `FCompute` not stored (QwenModel 301);
  `M.Compute` not set (TrainCli 302; LogprobsCli 301); bf16 parsed as fp32 (TrainCli 301, 302;
  LogprobsCli 301); `A` drawn in `0..Bound` (Lora 301).
- **ITa** (harness owner's driver; the T11 prompts with the default kind and twice with
  `--compute bf16`, the T17 training records in the default kind, in bf16, and 2 updates fp32 then
  `--compute bf16 --resume`), compared against the oracle of the code before T21: 0, 0.1883, 0,
  5.33e-3, 0, 0; about 7 minutes. With the code before T21 (and the skeleton), ITa-T21-001 is OK and
  002–006 are NG (`--compute` is an unknown option there).
- **M2g-a** (the prototype, two cards): fp32 from its own start within 5.5e-3 of the CPU curve over
  16 updates (gradient norm at update 1: 3.76867 against 3.76875); bf16 within 6.3 % (2.2e-3 up to
  update 9), ending at 2.9 % of the first loss, 158 s against 219 s.

## 9. Change history

| Date | Change | Reason |
|------|--------|--------|
| 2026-11-25 | Initial creation | T21a, delegation 1 of 2: the T21 detailed design |
