# Detailed Design — Program `qlora_train` (T17)

Source of truth: `../external-design/T17-external.md`,
`../system-flow/T17-system.md` and the T17 facts
(`tasks/T17-facts.md`). This document restates and organises them; it adds
no new requirements.

## 1. Module composition

- Source files (all `{$mode objfpc}{$H+}`):
  - `src/TrainStep.pas`, unit `TrainStep` — one sample's loss and gradient
    (`SampleLossBackward`), the loss without gradients (`SampleLoss`), and
    the validation loss (`ValidationLoss`).
  - `src/TrainCli.pas`, unit `TrainCli` — the command line
    (`ParseTrainArgs`), the run (`RunTrainCLI`: the loop over epochs and
    updates, the log, checkpoints and resume). The unit header (kept from
    the skeleton) documents the command line, the run, the output and the
    exit codes.
  - `src/qlora_train.pas`, the program — the harness owner's thin wrapper,
    complete: it forwards the command line to `RunTrainCLI`, prints the
    output lines to stdout and the error lines to stderr, and exits with
    the returned code. Nobody changes it.
- Units used (as in the facts' code):
  - `TrainStep`: `Tch` (`TTensor`, `TTensor.FromArray`, `kNone`,
    `NoGradGuard`), `Math` (`Min`), `QwenModel` (T11: `TQwenModel`,
    `M.Stages`, `M.Embed`, `M.Logits`, `M.Trunk`), `Recompute` (T13:
    `RecomputeForward`, `RecomputeBackward`, `TSavedInputs`), `TrainData`
    (T15: `TTrainSample`, `TTrainSet`).
  - `TrainCli`: `Classes` (`TStringArray`, `TStrings`), `SysUtils`
    (`Format`, file and directory operations), `Math` (`NaN`, `IsNan`,
    `IsInfinite`), `DateUtils` (`Now`, `SecondSpan`), `md5` (`MD5Print`,
    `MD5File`, `MD5String`), `Tch` (`TDevice`, `ManualSeed`), `GgufFile`
    (T01: `TGgufFile`), `QwenTokenizer` (T05: `TQwenTokenizer`),
    `QwenModel` (T11: `TQwenModel`), `Lora` (T12: `TLoraSet`),
    `LoraOptim` (T14: `TLoraAdamW`, `DefaultAdamWConfig`, `ScheduleLR`),
    `Checkpoint` (T16: `TCheckpoints`, `TRunState`, `NewRunState`),
    `TrainData` (T15: `TTrainSet`, `EpochBatches`, `SkipBatches`),
    `TrainStep` (T17: `TSampleLoss`, `SampleLossBackward`,
    `ValidationLoss`).
- Declared in `TrainStep`:
  - `ETrainStep = class(Exception)` — the unit's error class.
  - `TSampleLoss` — the loss record (3, table).
  - Functions (interface section): `SampleLossBackward`, `SampleLoss`,
    `ValidationLoss`.
  - Local routines (implementation section only): `Targets`, `SliceLoss`,
    `CheckSample`.
- Declared in `TrainCli`:
  - `TTrainConfig` — the configuration record (3, table).
  - Functions (interface section): `DefaultTrainConfig`, `ParseTrainArgs`,
    `ConfigHash`, `TotalUpdates`, `ShaFromName`, `RunTrainCLI`.
  - Local routines (implementation section only): `AppendLine`, `G17`;
    `Int` and `Num`, nested in `ParseTrainArgs`.
  - Constant (implementation section): `Usage`.
- The program `src/qlora_train.pas` uses `TrainCli` only.

## 2. Function specifications

`TrainStep`:

| Function | Arguments | Returns | Raises | Preconditions |
|---|---|---|---|---|
| `Targets` (local) | `const ASample: TTrainSample; AFrom, ACount: Integer` | the targets of rows `AFrom .. AFrom + ACount - 1`, shape `[-1, 1]` | — | — |
| `SliceLoss` (local) | `M: TQwenModel; const H: TTensor; const ASample: TTrainSample; AFrom, ACount: Integer` | the summed cross-entropy of rows `[AFrom, AFrom + ACount)` of `H`, as a 0-d tensor | `ETrainStep` (through `Targets`, 4.1) | `ACount` rows inside `H` |
| `CheckSample` (local) | `const ASample: TTrainSample; ASlice: Integer` | — | `ETrainStep` (4.3, #1–#2) | — |
| `SampleLossBackward` | `M: TQwenModel; const ASample: TTrainSample; ADenom: Double; ASlice: Integer` | `TSampleLoss` (the summed cross-entropy of the loss tokens, `Count` = `LossTo - LossFrom`); the gradients of the adapters accumulated | `ETrainStep` (4.3, #1–#3) | `ADenom > 0`, `ASlice >= 1`, valid loss span |
| `SampleLoss` | `M: TQwenModel; const ASample: TTrainSample; ASlice: Integer` | the same `TSampleLoss`, without gradients (whole trunk under `NoGradGuard`) | `ETrainStep` (4.3, #1–#2) | `ASlice >= 1`, valid loss span |
| `ValidationLoss` | `M: TQwenModel; ASet: TTrainSet; ASlice: Integer` | the mean cross-entropy per loss token over every sample of `ASet`; `NaN` for an empty set | `ETrainStep` (through `SampleLoss`, 4.3, #1–#2) | `ASlice >= 1` |

`TrainCli`:

| Function | Arguments | Returns | Raises | Preconditions |
|---|---|---|---|---|
| `DefaultTrainConfig` | — | the default `TTrainConfig` (3) | — | — |
| `ParseTrainArgs` | `const AArgs: array of string; out C: TTrainConfig` | `''` when valid, else the usage error | — | — |
| `Int` (nested in `ParseTrainArgs`) | `const AName: string; ALo, AHi: Int64; out R: Int64` | `True` when `V` is an integer in `[ALo, AHi]`; on failure it sets `ParseTrainArgs` to the usage error | — | — |
| `Num` (nested in `ParseTrainArgs`) | `const AName: string; ALo: Double; out R: Double` | `True` when `V` is a finite number `>= ALo`; on failure it sets `ParseTrainArgs` to the usage error | — | — |
| `ShaFromName` | `const AFile: string` | the 64 hex digits when the file name is `sha256-` + 64 lower-case hex digits, else `''` | — | — |
| `ConfigHash` | `const C: TTrainConfig` | the MD5 (32 hex digits) of the settings that shape the run | tch-fpc / file errors from `MD5File` and `MD5String` (the files must be readable) | — |
| `TotalUpdates` | `ASet: TTrainSet; const C: TTrainConfig` | the number of updates of all epochs | `ETrainData` (through `EpochBatches`, T15) | `ASet` alive |
| `AppendLine` (local) | `const AFile, ALine: string` | — | file errors from `AssignFile` / `Append` / `Rewrite` | — |
| `G17` (local) | `X: Double` | `FloatToStrF(X, ffGeneral, 17, 0)`; `'nan'` for NaN | — | — |
| `RunTrainCLI` | `const AArgs: array of string; AOut, AErr: TStrings` | the exit code (0, 1, 2 or 3); `AOut` / `AErr` filled (caller-created, caller-owned) | — (all errors caught, 4.3) | — |

`TTensor` is a managed record: never freed by these units. `TQwenModel`,
`TGgufFile`, `TQwenTokenizer`, `TTrainSet`, `TLoraSet`, `TLoraAdamW`,
`TCheckpoints` and `TSavedInputs` objects are created and freed in
`try … finally` (4.2.8, step 7).

## 3. Data structures

- `TSampleLoss` record (interface section):

  | Field | Type | Meaning |
  |---|---|---|
  | `Sum` | `Double` | the summed cross-entropy of the loss tokens |
  | `Count` | `Integer` | the loss tokens (`LossTo - LossFrom`) |

- `TTrainConfig` record (interface section), with the defaults of
  `DefaultTrainConfig`:

  | Field | Type | Default | Meaning |
  |---|---|---|---|
  | `Model` | `string` | `''` | path to the base model (GGUF) |
  | `Train` | `string` | `''` | path to the training data (JSONL, T15 format) |
  | `OutDir` | `string` | `''` | the output directory (trailing delimiter set by `ParseTrainArgs`) |
  | `Valid` | `string` | `''` | the validation data (`''`: none) |
  | `Rank` | `Integer` | 16 | the LoRA rank |
  | `Alpha` | `Double` | 32 | the LoRA alpha (scale `Alpha / Rank`) |
  | `Lr` | `Double` | 1e-4 | the base learning rate |
  | `MinLrRatio` | `Double` | 0.1 | the minimum LR ratio |
  | `Clip` | `Double` | 1.0 | the gradient clipping norm |
  | `Warmup` | `Integer` | 50 | the warm-up updates |
  | `Epochs` | `Integer` | 3 | the number of epochs |
  | `Seed` | `Integer` | 1 | the seed (LoRA A matrices, epoch order) |
  | `Pool` | `Integer` | 64 | the samples per pool (T15) |
  | `Budget` | `Integer` | 2048 | the token budget per batch (T15) |
  | `BatchesPerUpdate` | `Integer` | 2 | the batches of one update |
  | `CkptEvery` | `Integer` | 50 | checkpoint every N updates |
  | `Keep` | `Integer` | 3 | the checkpoints kept (T16) |
  | `Classes` | `TStringArray` | `['compiled-run', 'fpc-approx']` | the admitted classes (T15) |
  | `Split` | `Integer` | −1 | the GPU split (−1: CPU; N ≥ 0: trunk blocks `0..N-1` on `cuda:0`, the rest and the head on `cuda:1`) |
  | `BaseSha256` | `string` | `''` (then from the model's name, 4.2.4) | the base's sha256 (not computed: stock FPC 3.2.2 has no sha256) |
  | `MaxUpdates` | `Int64` | 0 | stop after N updates (0: no limit) |
  | `Resume` | `Boolean` | `False` | continue from the latest checkpoint |

- Constant (implementation section):
  `Usage = 'usage: qlora_train [options] MODEL TRAIN OUTDIR (see the unit header)'`.
- The log `OutDir + 'train-log.tsv'`, appended line by line (a run takes
  hours; the output strings come at the end only):
  - the update line, 8 tab-separated fields:
    `update <tab> epoch <tab> position <tab> loss <tab> lr <tab> norm <tab> tokens <tab> seconds`
    (`Format('%d'#9'%d'#9'%d'#9'%s'#9'%s'#9'%s'#9'%d'#9'%.1f')`; the last
    is `SecondSpan(Now, T0)` with `%.1f`).
  - the checkpoint line, 4 tab-separated fields:
    `'ckpt' <tab> update <tab> validation loss <tab> directory`
    (`Format('ckpt'#9'%d'#9'%s'#9'%s')`).
  - the non-finite-loss line (exit 3), 4 fields:
    `<update (U + 1)> <tab> <epoch> <tab> <position> <tab> 'nan'`
    (`Format('%d'#9'%d'#9'%d'#9'%s')` with `G17(Loss)`).
  - the floats of the log are formatted by `G17` (4.2.7).
- The checkpoints in `OutDir + 'ckpt'`: the T16 layout
  (`ckpt-00000001/` … with `state.json`, `lora.ot`, `moments.ot`), created
  by `TCheckpoints` with `Keep`.
- `state.json` (the T16 `TRunState`): `Update`, `Epoch`, `Position`,
  `Seed`, `Rank`, `Scale`, `ConfigHash`, `BaseSha256`, `ValLoss` (T16,
  "Keys of `state.json`").

## 4. Processing detail

### 4.1 Unit `TrainStep`

The public three run the sample's loss through the head `M.Logits` and the
trunk; the trunk is one of two ways (facts, "One sample"):

- `SampleLossBackward` (with gradients): the trunk block-wise —
  `RecomputeForward` with the inputs on the host (T13), then
  `RecomputeBackward`; the head only on the loss rows (positions
  `LossFrom - 1 .. LossTo - 2` predict tokens `LossFrom .. LossTo - 1`),
  in slices of `Slice` rows (default 256), each slice's summed
  cross-entropy divided by the denominator and sent backward on its own,
  then the trunk.
- `SampleLoss` / `ValidationLoss` (without gradients): the whole trunk
  under `NoGradGuard`.

#### 4.1.1 `Targets` (local)

```pascal
{ The targets of rows AFrom .. AFrom + ACount - 1: tokens AFrom + 1 ... }
function Targets(const ASample: TTrainSample; AFrom, ACount: Integer): TTensor;
var
  T: array of Int64;
  I: Integer;
begin
  T := nil;
  SetLength(T, ACount);
  for I := 0 to ACount - 1 do
    T[I] := ASample.Tokens[AFrom + I + 1];
  Result := TTensor.FromArray(T).view([-1, 1]);
end;
```

- Row `AFrom + I` predicts the next token, `ASample.Tokens[AFrom + I + 1]`
  (the causal target); the slice of rows `P .. P + N - 1` therefore uses
  the targets of tokens `P + 1 .. P + N` (4.1.4).
- `FromArray` of an `Int64` array is `kInt64`; `view([-1, 1])` gives a
  column vector for the `gather` of `SliceLoss`.

#### 4.1.2 `SliceLoss` (local)

```pascal
{ The summed cross-entropy of rows [AFrom, AFrom + ACount) of H. }
function SliceLoss(M: TQwenModel; const H: TTensor; const ASample: TTrainSample;
  AFrom, ACount: Integer): TTensor;
var
  Lg: TTensor;
begin
  Lg := M.Logits(H.narrow(0, AFrom, ACount));
  Result := -Lg.log_softmax(-1).gather(1, Targets(ASample, AFrom, ACount).&to(Lg.Device)).sum(kNone);
end;
```

Notes:

- Only the `ACount` loss rows of the hidden states are passed to the head
  `M.Logits` (the head only on the loss rows, not on the whole sequence).
- `log_softmax(-1)` over the vocabulary, `gather(1, targets)` for the
  target log-probabilities, `sum(kNone)` of all of them → the summed
  cross-entropy of the slice, sign-flipped, as a 0-d tensor.
- `Targets` is moved to `Lg.Device` (`&to`) before the `gather`.
- The result is a tensor (not a `Double`) so that `SampleLossBackward`
  can send `(L / ADenom).Backward` on the same graph (4.1.4).

#### 4.1.3 `CheckSample` (local)

```pascal
procedure CheckSample(const ASample: TTrainSample; ASlice: Integer);
begin
  if ASlice < 1 then
    raise ETrainStep.CreateFmt('slice must be positive, got %d', [ASlice]);
  if (ASample.LossFrom < 1) or (ASample.LossTo <= ASample.LossFrom) or (ASample.LossTo > Length(ASample.Tokens)) then
    raise ETrainStep.CreateFmt('%s: loss span %d..%d outside 1..%d', [ASample.Id, ASample.LossFrom,
      ASample.LossTo, Length(ASample.Tokens)]);
end;
```

Notes:

- Checked, in this order (4.3, #1–#2): first the slice, then the loss
  span; the span condition is `LossFrom < 1` or `LossTo <= LossFrom` or
  `LossTo > Length(Tokens)` (the span is 1-based and inside the token
  sequence).
- Called first by both `SampleLossBackward` and `SampleLoss` (4.3 #1–#2
  apply to both; #3, the denominator, then `SampleLossBackward` only).

#### 4.1.4 `SampleLossBackward`

```pascal
function SampleLossBackward(M: TQwenModel; const ASample: TTrainSample; ADenom: Double;
  ASlice: Integer): TSampleLoss;
var
  Saved: TSavedInputs;
  H, Hl, L: TTensor;
  P, N: Integer;
begin
  CheckSample(ASample, ASlice);
  if not (ADenom > 0) then
    raise ETrainStep.CreateFmt('denominator must be positive, got %g', [ADenom]);
  Result.Sum := 0;
  Result.Count := ASample.LossTo - ASample.LossFrom;
  Saved := TSavedInputs.Create(True);
  try
    H := RecomputeForward(M.Stages, M.Embed(ASample.Tokens), Saved);
    Hl := H.detach.requires_grad_(True);
    P := ASample.LossFrom - 1;
    while P < ASample.LossTo - 1 do
    begin
      N := Min(ASlice, ASample.LossTo - 1 - P);
      L := SliceLoss(M, Hl, ASample, P, N);
      Result.Sum := Result.Sum + L.DoubleValue([]);
      (L / ADenom).Backward;
      Inc(P, N);
    end;
    RecomputeBackward(M.Stages, Saved, Hl.grad);
  finally
    Saved.Free;
  end;
end;
```

Steps, in exactly this order:

1. `CheckSample(ASample, ASlice)` — the slice and the span first (4.1.3,
   4.3 #1–#2).
2. `not (ADenom > 0)` raises (4.3, #3): `not (ADenom > 0)` covers zero,
   negative **and** NaN.
3. `Result.Count := ASample.LossTo - ASample.LossFrom`; `Result.Sum` is
   accumulated from the slices below.
4. `Saved := TSavedInputs.Create(True)`, freed in `finally`; the trunk
   runs block-wise through `RecomputeForward(M.Stages, M.Embed(ASample.
   Tokens), Saved)` (T13: the inputs stay on the host, the blocks
   checkpoint their inputs).
5. `Hl := H.detach.requires_grad_(True)` — the trunk's output becomes the
   only graph leaf; the loss graph is head-only plus this boundary.
6. The loss rows, positions `P := LossFrom - 1` up to `LossTo - 2`
   (loop `while P < LossTo - 1`), in slices of
   `N := Min(ASlice, LossTo - 1 - P)` rows, one step:
   - `L := SliceLoss(M, Hl, ASample, P, N)` — the summed cross-entropy of
     the slice (4.1.2);
   - `Result.Sum := Result.Sum + L.DoubleValue([])` — the numeric sum for
     the caller's loss;
   - `(L / ADenom).Backward` — each slice's sum is divided by the
     denominator and sent backward on its own; the gradients of the
     adapters **accumulate** (tch-fpc autograd; the caller zeroes them,
     see T14 `ZeroGrad`, UT row 004).
7. `RecomputeBackward(M.Stages, Saved, Hl.grad)` — the trunk's backward
   after the head's (T13), with the accumulated gradient at the boundary.

Notes:

- The loss rows `LossFrom - 1 .. LossTo - 2` predict the tokens
  `LossFrom .. LossTo - 1`; the number of loss rows is
  `LossTo - LossFrom` (the `Result.Count`), matching the `Targets` of
  4.1.1.
- `ADenom` is the loss-token count of the whole update (4.2.8, step 4a),
  so the gradient is that of the update's mean loss (facts, "One sample
  per forward", decided 2026-11-24).
- `TTensor`s are managed records and are never freed; only `Saved` is
  owned.

#### 4.1.5 `SampleLoss`

```pascal
function SampleLoss(M: TQwenModel; const ASample: TTrainSample; ASlice: Integer): TSampleLoss;
var
  G: IInterface;
  H: TTensor;
  P, N: Integer;
begin
  CheckSample(ASample, ASlice);
  G := NoGradGuard;
  Result.Sum := 0;
  Result.Count := ASample.LossTo - ASample.LossFrom;
  H := M.Trunk(M.Embed(ASample.Tokens));
  P := ASample.LossFrom - 1;
  while P < ASample.LossTo - 1 do
  begin
    N := Min(ASlice, ASample.LossTo - 1 - P);
    Result.Sum := Result.Sum + SliceLoss(M, H, ASample, P, N).DoubleValue([]);
    Inc(P, N);
  end;
end;
```

Notes:

- `G := NoGradGuard` assigns the guard to a local interface: a bare
  `NoGradGuard;` statement has no effect (tch-fpc rule; the local keeps
  it alive for the whole function, and the UT expects no A or B gradient
  defined afterwards — the guard is released when `G` goes out of scope).
- The whole trunk runs under the guard: `H := M.Trunk(M.Embed(ASample.
  Tokens))` (no `Recompute*`, no graph).
- The same slice loop as `SampleLossBackward` (4.1.4, step 6) but without
  the `/ ADenom` and without the `Backward`.
- The checks are the span and the slice through `CheckSample` (4.1.3);
  there is **no** denominator check here (4.3, #3 is `SampleLossBackward`
  only).

#### 4.1.6 `ValidationLoss`

```pascal
function ValidationLoss(M: TQwenModel; ASet: TTrainSet; ASlice: Integer): Double;
var
  I: Integer;
  R: TSampleLoss;
  Sum: Double;
  N: Int64;
begin
  if ASet.Count = 0 then
    Exit(NaN);
  Sum := 0;
  N := 0;
  for I := 0 to ASet.Count - 1 do
  begin
    R := SampleLoss(M, ASet[I], ASlice);
    Sum := Sum + R.Sum;
    Inc(N, R.Count);
  end;
  Result := Sum / N;
end;
```

Notes:

- An empty set returns `NaN` (interface comment), not 0 and not an error.
- `Sum` is the sum of the samples' summed cross-entropies and `N` the sum
  of their loss tokens, so the result is the mean cross-entropy **per
  loss token** over the set (interface comment).
- Every sample goes through `SampleLoss` (4.1.5), hence under the
  `NoGradGuard`; a bad slice or span raises `ETrainStep` of that sample
  (4.3, #1–#2).

### 4.2 Unit `TrainCli`

#### 4.2.1 `TTrainConfig` (interface section)

```pascal
type
  TTrainConfig = record
    Model, Train, OutDir, Valid: string;
    Rank: Integer;
    Alpha, Lr, MinLrRatio, Clip: Double;
    Warmup, Epochs, Seed, Pool, Budget, BatchesPerUpdate, CkptEvery, Keep, Slice: Integer;
    Classes: TStringArray;
    Split: Integer;            // -1: CPU
    BaseSha256: string;
    MaxUpdates: Int64;
    Resume: Boolean;
  end;
```

The fields and the defaults of `DefaultTrainConfig` are the table of 3.
The strings are `''` by default; the five string fields are `Model`,
`Train`, `OutDir`, `Valid`, `BaseSha256`; `Resume` is `False` by
default.

#### 4.2.2 `DefaultTrainConfig`

```pascal
function DefaultTrainConfig: TTrainConfig;
begin
  Result := Default(TTrainConfig);
  Result.Rank := 16;
  Result.Alpha := 32;
  Result.Lr := 1e-4;
  Result.MinLrRatio := 0.1;
  Result.Clip := 1.0;
  Result.Warmup := 50;
  Result.Epochs := 3;
  Result.Seed := 1;
  Result.Pool := 64;
  Result.Budget := 2048;
  Result.BatchesPerUpdate := 2;
  Result.CkptEvery := 50;
  Result.Keep := 3;
  Result.Slice := 256;
  Result.Classes := TStringArray.Create('compiled-run', 'fpc-approx');
  Result.Split := -1;
  Result.MaxUpdates := 0;
end;
```

Notes:

- The defaults of the table of 3 (external design, "Command line";
  `DESIGN` 8.14: the batches per update 2, the slice 256).
- `MaxUpdates` is `Int64` and defaults to 0 (no limit); `Resume` stays
  `False` from `Default(TTrainConfig)`.
- The scale of the adapters is `Alpha / Rank` (4.2.8, step 2k).

#### 4.2.3 `ParseTrainArgs` (with `Int` and `Num` nested)

```pascal
{ '' when AArgs are valid, else the usage error. }
function ParseTrainArgs(const AArgs: array of string; out C: TTrainConfig): string;
var
  I, N: Integer;
  Pos: array of string;
  A, V: string;
  X: Double;
  K: Int64;
  FS: TFormatSettings;

  function Int(const AName: string; ALo, AHi: Int64; out R: Int64): Boolean;
  begin
    Result := TryStrToInt64(V, R) and (R >= ALo) and (R <= AHi);
    if not Result then
      ParseTrainArgs := Format('%s: an integer in %d..%d, got "%s"', [AName, ALo, AHi, V]);
  end;

  function Num(const AName: string; ALo: Double; out R: Double): Boolean;
  begin
    Result := TryStrToFloat(V, R, FS) and not IsNan(R) and not IsInfinite(R) and (R >= ALo);
    if not Result then
      ParseTrainArgs := Format('%s: a number >= %g, got "%s"', [AName, ALo, V], FS);
  end;

begin
  FS := DefaultFormatSettings;
  FS.DecimalSeparator := '.';
  C := DefaultTrainConfig;
  Result := '';
  Pos := nil;
  I := 0;
  N := Length(AArgs);
  while I < N do
  begin
    A := AArgs[I];
    if A = '--resume' then
    begin
      C.Resume := True;
      Inc(I);
      Continue;
    end;
    if Copy(A, 1, 2) <> '--' then
    begin
      SetLength(Pos, Length(Pos) + 1);
      Pos[High(Pos)] := A;
      Inc(I);
      Continue;
    end;
    if I + 1 >= N then
      Exit(A + ': value missing');
    V := AArgs[I + 1];
    Inc(I, 2);
    if A = '--valid' then C.Valid := V
    else if A = '--base-sha256' then C.BaseSha256 := V
    else if A = '--classes' then C.Classes := V.Split([','])
    else if A = '--rank' then begin if not Int(A, 1, 4096, K) then Exit; C.Rank := K; end
    else if A = '--warmup' then begin if not Int(A, 0, MaxInt, K) then Exit; C.Warmup := K; end
    else if A = '--epochs' then begin if not Int(A, 1, 1000, K) then Exit; C.Epochs := K; end
    else if A = '--seed' then begin if not Int(A, -MaxInt, MaxInt, K) then Exit; C.Seed := K; end
    else if A = '--pool' then begin if not Int(A, 1, MaxInt, K) then Exit; C.Pool := K; end
    else if A = '--budget' then begin if not Int(A, 1, MaxInt, K) then Exit; C.Budget := K; end
    else if A = '--batches-per-update' then begin if not Int(A, 1, 1000, K) then Exit; C.BatchesPerUpdate := K; end
    else if A = '--ckpt-every' then begin if not Int(A, 1, MaxInt, K) then Exit; C.CkptEvery := K; end
    else if A = '--keep' then begin if not Int(A, 1, 1000, K) then Exit; C.Keep := K; end
    else if A = '--slice' then begin if not Int(A, 1, MaxInt, K) then Exit; C.Slice := K; end
    else if A = '--split' then begin if not Int(A, 0, 1000, K) then Exit; C.Split := K; end
    else if A = '--max-updates' then begin if not Int(A, 0, High(Int64), K) then Exit; C.MaxUpdates := K; end
    else if A = '--alpha' then begin if not Num(A, 0, X) then Exit; C.Alpha := X; end
    else if A = '--lr' then begin if not Num(A, 0, X) then Exit; C.Lr := X; end
    else if A = '--min-lr-ratio' then begin if not Num(A, 0, X) then Exit; C.MinLrRatio := X; end
    else if A = '--clip' then begin if not Num(A, 0, X) then Exit; C.Clip := X; end
    else
      Exit('unknown option ' + A);
  end;
  if Length(Pos) <> 3 then
    Exit(Format('expected MODEL TRAIN OUTDIR, got %d arguments', [Length(Pos)]));
  C.Model := Pos[0];
  C.Train := Pos[1];
  C.OutDir := IncludeTrailingPathDelimiter(Pos[2]);
  if C.BaseSha256 = '' then
    C.BaseSha256 := ShaFromName(C.Model);
end;
```

Notes:

- `FS` is `DefaultFormatSettings` with `DecimalSeparator := '.'` (the
  numbers are always parsed from and written in dot notation, independent
  of the locale); it is the `FS` the `Num` nested function uses.
- `C` starts as `DefaultTrainConfig` (4.2.2): every option is optional.
- The argument loop (each argument is `AArgs[I]`):
  - `--resume` is the flag: sets `C.Resume := True`, consumes no value.
  - An argument that does not start with `--` is a positional, appended
    to `Pos` (the three of them are `MODEL TRAIN OUTDIR`).
  - Every other option takes a value: when the value is missing
    (`I + 1 >= N`), the error is `A + ': value missing'` (4.3, #4).
  - The dispatch, in this order (4.3, #4–#8): `--valid`, `--base-sha256`,
    `--classes` (`V.Split([','])`); then the integer options `--rank`
    (1..4096), `--warmup` (0..MaxInt), `--epochs` (1..1000), `--seed`
    (−MaxInt..MaxInt), `--pool` (1..MaxInt), `--budget` (1..MaxInt),
    `--batches-per-update` (1..1000), `--ckpt-every` (1..MaxInt), `--keep`
    (1..1000), `--slice` (1..MaxInt), `--split` (0..1000),
    `--max-updates` (0..High(Int64)); then the numeric options
    `--alpha`, `--lr`, `--min-lr-ratio`, `--clip` (all `>= 0`); otherwise
    `unknown option A`.
- `Int` (nested): `TryStrToInt64` and the range `[ALo, AHi]`; on failure
  it sets `ParseTrainArgs` (the named result of the enclosing function)
  to `<AName>: an integer in <ALo>..<AHi>, got "<V>"` — `AName` is the
  option as given (e.g. `--rank`), `V` the text as given (4.3, #5).
- `Num` (nested): `TryStrToFloat` with `FS`, `not IsNan`, `not
  IsInfinite` and `>= ALo` (all four options use `ALo = 0`, so both `nan`
  and `±inf` are refused, 4.3, #6); on failure it sets
  `ParseTrainArgs` to `<AName>: a number >= <ALo>, got "<V>"`.
- After the loop: exactly three positionals, else
  `expected MODEL TRAIN OUTDIR, got <n> arguments` (4.3, #8); then
  `C.Model := Pos[0]`, `C.Train := Pos[1]`, `C.OutDir :=
  IncludeTrailingPathDelimiter(Pos[2])` (the `OutDir` always ends in the
  delimiter — `o/` stays `o/`, one delimiter).
- `C.BaseSha256`: given by `--base-sha256`, else empty at this point; the
  fallback to the model's name, `C.BaseSha256 := ShaFromName(C.Model)`,
  happens last (4.2.4) — an explicit `--base-sha256` wins.
- A valid command line returns `''` (interface comment); `C` is then fully
  filled.

#### 4.2.4 `ShaFromName`

```pascal
{ The base's sha256 from a file name sha256-<64 hex>, else ''. }
function ShaFromName(const AFile: string): string;
var
  N: string;
  I: Integer;
begin
  Result := '';
  N := ExtractFileName(AFile);
  if (Length(N) <> 71) or (Copy(N, 1, 7) <> 'sha256-') then
    Exit;
  for I := 8 to 71 do
    if not (N[I] in ['0'..'9', 'a'..'f']) then
      Exit;
  Result := Copy(N, 8, 64);
end;
```

Notes:

- Stock FPC 3.2.2 has no sha256 (`fpsha256` is in FPC 3.3.1 only,
  measured), so the base's sha256 is **not computed** (facts): it is
  `--base-sha256`, or the name of an Ollama blob `sha256-<64 hex>` (the
  27B is one), else empty.
- Only the file **name** is looked at (`ExtractFileName`); a path
  `.../sha256-<hex>/x` has the name `x` and gives `''`.
- The checks, in this order: the name is exactly 71 characters long
  (`sha256-` is 7 plus 64); it starts with `sha256-`; then every one of
  the 64 remaining characters is a lower-case hex digit (`'0'..'9',
  'a'..'f'`). Any violation returns `''` (the first one: 63 digits, upper
  case, another name, a longer name).
- On success it returns the 64 hex digits (`Copy(N, 8, 64)`).

#### 4.2.5 `ConfigHash`

```pascal
{ MD5 (hex) of the settings that must not change across a resume: the
  contents of TRAIN and VALID, rank, alpha, lr, warm-up, min ratio, clip,
  epochs, seed, pool, budget, batches per update, slice, classes. }
function ConfigHash(const C: TTrainConfig): string;
var
  T: string;
  FS: TFormatSettings;
begin
  FS := DefaultFormatSettings;
  FS.DecimalSeparator := '.';
  T := 'train=' + MD5Print(MD5File(C.Train));
  if C.Valid <> '' then
    T := T + ' valid=' + MD5Print(MD5File(C.Valid));
  T := T + Format(' rank=%d alpha=%g lr=%g warmup=%d min=%g clip=%g epochs=%d seed=%d pool=%d budget=%d bpu=%d slice=%d classes=%s',
    [C.Rank, C.Alpha, C.Lr, C.Warmup, C.MinLrRatio, C.Clip, C.Epochs, C.Seed, C.Pool,
     C.Budget, C.BatchesPerUpdate, C.Slice, string.Join(',', C.Classes)], FS);
  Result := MD5Print(MD5String(T));
end;
```

Notes:

- The hash covers, in this order: the **contents** of `C.Train` (MD5 of
  the file), the contents of `C.Valid` when given, and the settings as
  formatted above: `rank`, `alpha`, `lr`, `warmup`, `min`, `clip`,
  `epochs`, `seed`, `pool`, `budget`, `bpu` (batches per update), `slice`,
  `classes` (comma-joined).
- Excluded by design (facts, "Checkpoints"): `--max-updates`, `--resume`,
  `--split`, `--ckpt-every`, `--keep`, the output directory and the
  base's sha256 (the base is checked separately, 4.2.8 step 3).
- `md5`'s `MD5Print(MD5File(...))`/`MD5Print(MD5String(T))` give 32
  lower-case hex digits; the result changes with the train file's
  contents (even with one more line) and with every setting covered, and
  is equal for runs that differ only in the excluded options
  (external design, "Checkpoints and resume").
- The `%g` values are formatted with `FS` (dot decimal separator), so the
  hash does not depend on the locale.

#### 4.2.6 `TotalUpdates`

```pascal
{ The updates of all epochs: per epoch ceil(batches / batches per update). }
function TotalUpdates(ASet: TTrainSet; const C: TTrainConfig): Int64;
var
  E: Integer;
begin
  Result := 0;
  for E := 0 to C.Epochs - 1 do
    Inc(Result, (Length(EpochBatches(ASet, C.Seed, E, C.Pool, C.Budget)) + C.BatchesPerUpdate - 1)
      div C.BatchesPerUpdate);
end;
```

Notes:

- Per epoch `E` (0-based): `ceil(batches / BatchesPerUpdate)` as
  `(b + bpu - 1) div bpu` (no floating point, no off-by-one).
- The batch count of an epoch is `Length(EpochBatches(ASet, C.Seed, E,
  C.Pool, C.Budget))` (T15; the batches differ slightly between epochs,
  facts) — this is why the sum is computed over the epochs, not
  `epochs × ceil(batches/bpu)` of one epoch.
- It is the `Total` of `ScheduleLR(C.Lr, U, C.Warmup, Total,
  C.MinLrRatio)` (4.2.8), which is why `C.Warmup >= Total` is an error
  before the first update (4.2.8 step 2h, 4.3 #12).

#### 4.2.7 `AppendLine` (local) and `G17` (local)

```pascal
procedure AppendLine(const AFile, ALine: string);
var
  F: TextFile;
begin
  AssignFile(F, AFile);
  if FileExists(AFile) then
    Append(F)
  else
    Rewrite(F);
  WriteLn(F, ALine);
  CloseFile(F);
end;

function G17(X: Double): string;
var
  FS: TFormatSettings;
begin
  FS := DefaultFormatSettings;
  FS.DecimalSeparator := '.';
  if IsNan(X) then
    Result := 'nan'
  else
    Result := FloatToStrF(X, ffGeneral, 17, 0, FS);
end;
```

Notes:

- `AppendLine` writes exactly one line: it opens the log in `Append` mode
  when it exists and in `Rewrite` mode when it does not, writes the line
  and closes the file (the log is appended line by line **during** the
  run; a run takes hours, so the output strings come at the end only).
- `G17` formats a float for the log with `ffGeneral`, 17 significant
  digits (the exact round-trip of a `Double`), dot decimal separator;
  NaN is written as `nan` (the validation loss without `--valid`, the
  non-finite loss of the exit-3 line).

#### 4.2.8 `RunTrainCLI` (step by step)

```pascal
{ AOut / AErr are filled (caller-created, caller-owned); the result is the
  exit code. }
function RunTrainCLI(const AArgs: array of string; AOut, AErr: TStrings): Integer;
var
  C: TTrainConfig;
  Msg, Hash, LogFile, Dir: string;
  G: TGgufFile;
  Tok: TQwenTokenizer;
  M: TQwenModel;
  TrainSet, ValidSet: TTrainSet;
  S: TLoraSet;
  O: TLoraAdamW;
  Ck: TCheckpoints;
  St: TRunState;
  B: TBatches;
  Total, U: Int64;
  Epoch, I, J, K: Integer;
  Pos, Tokens: Int64;
  Denom, Sum, Loss, Norm, Lr, Val: Double;
  R: TSampleLoss;
  T0: TDateTime;
  EndOfEpoch: Boolean;
begin
  Msg := ParseTrainArgs(AArgs, C);
  if Msg <> '' then
  begin
    AErr.Add(Msg);
    AErr.Add(Usage);
    Exit(2);
  end;
  G := nil;
  Tok := nil;
  M := nil;
  TrainSet := nil;
  ValidSet := nil;
  S := nil;
  O := nil;
  Ck := nil;
  try
    try
      if not FileExists(C.Model) then
        raise Exception.Create('no model file ' + C.Model);
      G := TGgufFile.Open(C.Model);
      Tok := TQwenTokenizer.Create(G);
      TrainSet := TTrainSet.Create(Tok, C.Train, C.Classes, C.Budget);
      if TrainSet.Count = 0 then
        raise Exception.Create('no training sample in ' + C.Train);
      if C.Valid <> '' then
        ValidSet := TTrainSet.Create(Tok, C.Valid, C.Classes, C.Budget);
      Total := TotalUpdates(TrainSet, C);
      if C.Warmup >= Total then
        raise Exception.CreateFmt('warm-up %d must be below the %d updates', [C.Warmup, Total]);
      if C.Split < 0 then
        M := TQwenModel.Load(G, TDevice.Cpu, TDevice.Cpu, 0)
      else
        M := TQwenModel.Load(G, TDevice.Cuda(0), TDevice.Cuda(1), C.Split);
      ManualSeed(C.Seed);
      S := TLoraSet.Create(C.Rank, C.Alpha / C.Rank);
      M.AttachLora(S);
      O := TLoraAdamW.Create(S.Stores, DefaultAdamWConfig);
      ForceDirectories(C.OutDir);
      Ck := TCheckpoints.Create(C.OutDir + 'ckpt', C.Keep);
      Hash := ConfigHash(C);
      LogFile := C.OutDir + 'train-log.tsv';
      Epoch := 0;
      Pos := 0;
      U := 0;
      if C.Resume then
      begin
        Dir := Ck.Latest;
        if Dir = '' then
          raise Exception.Create('nothing to resume in ' + Ck.Root);
        St := Ck.Load(Dir, S, O);
        if St.ConfigHash <> Hash then
          raise Exception.Create('the configuration differs from the checkpoint''s: ' + Dir);
        if St.BaseSha256 <> C.BaseSha256 then
          raise Exception.Create('the base model differs from the checkpoint''s: ' + Dir);
        Epoch := St.Epoch;
        Pos := St.Position;
        U := St.Update;
        AOut.Add(Format('resumed from %s at update %d', [Dir, U]));
      end;
      while Epoch < C.Epochs do
      begin
        B := SkipBatches(EpochBatches(TrainSet, C.Seed, Epoch, C.Pool, C.Budget), Pos);
        I := 0;
        while I < Length(B) do
        begin
          T0 := Now;
          Denom := 0;
          for J := I to Min(I + C.BatchesPerUpdate, Length(B)) - 1 do
            for K := 0 to High(B[J]) do
              Denom := Denom + TrainSet[B[J][K]].LossTo - TrainSet[B[J][K]].LossFrom;
          Sum := 0;
          Tokens := 0;
          for J := I to Min(I + C.BatchesPerUpdate, Length(B)) - 1 do
            for K := 0 to High(B[J]) do
            begin
              R := SampleLossBackward(M, TrainSet[B[J][K]], Denom, C.Slice);
              Sum := Sum + R.Sum;
              Inc(Tokens, Length(TrainSet[B[J][K]].Tokens));
              Inc(Pos);
            end;
          Loss := Sum / Denom;
          if IsNan(Loss) or IsInfinite(Loss) then
          begin
            AErr.Add(Format('non-finite loss at update %d; stopped without updating', [U + 1]));
            AppendLine(LogFile, Format('%d'#9'%d'#9'%d'#9'%s', [U + 1, Epoch, Pos, G17(Loss)]));
            Exit(3);
          end;
          Inc(U);
          Norm := O.ClipGlobalNorm(C.Clip);
          Lr := ScheduleLR(C.Lr, U, C.Warmup, Total, C.MinLrRatio);
          O.Step(Lr);
          O.ZeroGrad;
          Inc(I, C.BatchesPerUpdate);
          EndOfEpoch := I >= Length(B);
          AppendLine(LogFile, Format('%d'#9'%d'#9'%d'#9'%s'#9'%s'#9'%s'#9'%d'#9'%.1f',
            [U, Epoch, Pos, G17(Loss), G17(Lr), G17(Norm), Tokens, SecondSpan(Now, T0)]));
          if EndOfEpoch or (U mod C.CkptEvery = 0) then
          begin
            Val := NaN;
            if ValidSet <> nil then
              Val := ValidationLoss(M, ValidSet, C.Slice);
            St := NewRunState;
            St.Update := U;
            { at an epoch's end the state is the next epoch's start }
            St.Epoch := Epoch + Ord(EndOfEpoch);
            if EndOfEpoch then
              St.Position := 0
            else
              St.Position := Pos;
            St.Seed := C.Seed;
            St.ConfigHash := Hash;
            St.BaseSha256 := C.BaseSha256;
            St.ValLoss := Val;
            Dir := Ck.Save(St, S, O);
            AppendLine(LogFile, Format('ckpt'#9'%d'#9'%s'#9'%s', [U, G17(Val), Dir]));
          end;
          if (C.MaxUpdates > 0) and (U >= C.MaxUpdates) then
          begin
            AOut.Add(Format('stopped after update %d of %d', [U, Total]));
            Exit(0);
          end;
        end;
        Inc(Epoch);
        Pos := 0;
      end;
      AOut.Add(Format('done: %d updates, best %s', [U, Ck.Best]));
      Result := 0;
    except
      on E: Exception do
      begin
        AErr.Add(E.Message);
        Result := 1;
      end;
    end;
  finally
    Ck.Free;
    O.Free;
    M.Free;
    S.Free;
    ValidSet.Free;
    TrainSet.Free;
    Tok.Free;
    G.Free;
  end;
end;
```

Steps, in exactly this order:

**Step 1 — the command line (usage, exit 2)**

1a. `Msg := ParseTrainArgs(AArgs, C)`.
1b. When `Msg <> ''`: `AErr.Add(Msg)`, `AErr.Add(Usage)`, `Exit(2)` —
   the usage error is the first line on `AErr` and the `Usage` line the
   second (4.3, #4–#8).

**Step 2 — the setup (any raise: the message alone on `AErr`, exit 1)**

2a. `FileExists(C.Model)`; `no model file <model>` otherwise (4.3, #9).
2b. `G := TGgufFile.Open(C.Model)` (T01's errors pass through).
2c. `Tok := TQwenTokenizer.Create(G)` (T05's errors pass through).
2d. `TrainSet := TTrainSet.Create(Tok, C.Train, C.Classes, C.Budget)`
   (T15's errors pass through, e.g. `no data file <train>` — 4.3, #10).
2e. `TrainSet.Count = 0` → `no training sample in <train>` (4.3, #11).
2f. `C.Valid <> ''` → `ValidSet := TTrainSet.Create(Tok, C.Valid,
   C.Classes, C.Budget)`; else `ValidSet` stays nil (the validation loss
   is then NaN, 4.2.8 step 4g).
2g. `Total := TotalUpdates(TrainSet, C)` (4.2.6).
2h. `C.Warmup >= Total` →
   `CreateFmt('warm-up %d must be below the %d updates', [C.Warmup,
   Total])` (4.3, #12) — before any update, so the schedule is valid for
   every update that will run.
2i. The model: `C.Split < 0` →
   `TQwenModel.Load(G, TDevice.Cpu, TDevice.Cpu, 0)` (all on the CPU);
   else `TQwenModel.Load(G, TDevice.Cuda(0), TDevice.Cuda(1), C.Split)`
   (blocks `0..Split-1` on `cuda:0`, the rest and the head on `cuda:1`
   — the unit header). `TQwenModel.Load`'s errors pass through, e.g.
   `No CUDA GPUs are available` with `--split` and the GPUs hidden,
   `split 3 outside 0..2` (4.3, #13).
2j. `ManualSeed(C.Seed)` (the `Tch` RNG seed, **before** `AttachLora`: it
   initialises the A matrices).
2k. `S := TLoraSet.Create(C.Rank, C.Alpha / C.Rank)` — the scale is
   `Alpha / Rank`; `M.AttachLora(S)` (T12).
2l. `O := TLoraAdamW.Create(S.Stores, DefaultAdamWConfig)` (T14).
2m. `ForceDirectories(C.OutDir)`; `Ck := TCheckpoints.Create(C.OutDir +
   'ckpt', C.Keep)` (T16; the checkpoints are in `OutDir/ckpt`);
   `Hash := ConfigHash(C)` (4.2.5); `LogFile := C.OutDir + 'train-log.tsv'`.
2n. `Epoch := 0; Pos := 0; U := 0` — the start of the run.

The eight objects (`G`, `Tok`, `M`, `TrainSet`, `ValidSet`, `S`, `O`,
`Ck`) are set to nil before the `try` and freed in `finally` in reverse
creation order: `Ck`, `O`, `M`, `S`, `ValidSet`, `TrainSet`, `Tok`, `G`
(tch-fpc ownership: optimiser, then modules, then the store's objects).

**Step 3 — the resume (only with `--resume`; the messages of 4.3 #14)**

3a. `Dir := Ck.Latest`; when `''` (no checkpoint) →
   `nothing to resume in <Ck.Root>` (the root with the trailing
   delimiter).
3b. `St := Ck.Load(Dir, S, O)` — restores the adapters, the moments and
   the optimiser's `StepCount` (T16 `Load`); its own errors pass through
   (e.g. the rank/scale refusal of T16, 4.7 #6).
3c. `St.ConfigHash <> Hash` →
   `the configuration differs from the checkpoint's: <Dir>`.
3d. `St.BaseSha256 <> C.BaseSha256` →
   `the base model differs from the checkpoint's: <Dir>`.
3e. The run continues from the checkpoint: `Epoch := St.Epoch`,
   `Pos := St.Position`, `U := St.Update` (4.2.8 step 4 restores the
   position and keeps the batches of the current epoch — `SkipBatches` of
   step 4a — so a resume inside an epoch is exact; without the position
   the resumed run diverges, the T17 facts' mutants catch 204).
3f. `AOut.Add(Format('resumed from %s at update %d', [Dir, U]))`.

**Step 4 — the epoch loop**

For `Epoch := 0` up to `C.Epochs - 1` (loop `while Epoch < C.Epochs`):

4a. `B := SkipBatches(EpochBatches(TrainSet, C.Seed, Epoch, C.Pool,
   C.Budget), Pos)` (T15; `Pos` is the number of samples of this epoch's
   order already consumed — 0 at the start and after each epoch), `I := 0`.

**The update loop** (`while I < Length(B)`; one iteration is one update):

4b. **The update** — `C.BatchesPerUpdate` consecutive batches of the
   epoch, starting at `I` (the last one may be shorter, `Min(I +
   C.BatchesPerUpdate, Length(B))`), their samples one after the other
   (facts, "One sample per forward"):
   - `T0 := Now` (the seconds of the update line).
   - `Denom` — the loss-token count of the whole update: over every batch
     `J` and every sample `K` of them, `Denom += TrainSet[B[J][K]].LossTo
     - TrainSet[B[J][K]].LossFrom`.
   - `Sum := 0; Tokens := 0;` then over the same batches and samples:
     `R := SampleLossBackward(M, TrainSet[B[J][K]], Denom, C.Slice)`
     (4.1.4 — each sample's loss divided by `Denom` goes backward, so the
     gradients accumulate to the update's mean loss);
     `Sum += R.Sum`; `Tokens += Length(TrainSet[B[J][K]].Tokens)` (all
     the tokens of the update, not only the loss tokens); `Inc(Pos)` (one
     consumed sample per sample — the position is per sample, 4.1.4 step
     6, 4.3 #14).
   - `Loss := Sum / Denom` — the mean cross-entropy per loss token of the
     update (external design, "Log").
4c. **The non-finite check (exit 3)** — `IsNan(Loss) or IsInfinite(Loss)`
   → `AErr.Add(Format('non-finite loss at update %d; stopped without
   updating', [U + 1]))` (the update that would have been run, 1-based),
   `AppendLine(LogFile, Format('%d'#9'%d'#9'%d'#9'%s', [U + 1, Epoch, Pos,
   G17(Loss)]))` — a 4-field line `<U + 1> <Epoch> <Pos> nan` — and
   `Exit(3)`. The run stops **before** its update: no `Inc(U)`, no step,
   no checkpoint (R7: "halve the rate and go on from the last good
   checkpoint" is the user's decision at M4, not the program's — the
   user can then `--resume`).
4d. **The update step** — in this order: `Inc(U)`; `Norm :=
   O.ClipGlobalNorm(C.Clip)` (the clipped global norm, T14); `Lr :=
   ScheduleLR(C.Lr, U, C.Warmup, Total, C.MinLrRatio)` (T14; `U` is
   1-based, `Total` the schedule's total of step 2g); `O.Step(Lr)`;
   `O.ZeroGrad` (the gradients are cleared **after** the step — without
   it the next update accumulates on top, the T17 facts' mutants catch
   204/205).
4e. `Inc(I, C.BatchesPerUpdate)`; `EndOfEpoch := I >= Length(B)`.
4f. **The update line of the log** (line by line, 4.2.7):
   `AppendLine(LogFile, Format('%d'#9'%d'#9'%d'#9'%s'#9'%s'#9'%s'#9'%d'#9'%.1f', [U, Epoch,
   Pos, G17(Loss), G17(Lr), G17(Norm), Tokens, SecondSpan(Now, T0)]))` —
   update, epoch, position, loss, lr, norm, tokens, seconds.
4g. **The checkpoint** — `EndOfEpoch or (U mod C.CkptEvery = 0)` (every
   `--ckpt-every` updates and at the end of each epoch):
   - `Val := NaN; if ValidSet <> nil then Val := ValidationLoss(M,
     ValidSet, C.Slice);` (4.1.6; NaN without `--valid`).
   - `St := NewRunState` (T16, zeros and `ValLoss` NaN), then:
     `St.Update := U`;
     `St.Epoch := Epoch + Ord(EndOfEpoch)` — **at an epoch's end the
     state is the next epoch's start**;
     `EndOfEpoch` → `St.Position := 0`, else `St.Position := Pos`;
     `St.Seed := C.Seed`; `St.ConfigHash := Hash`;
     `St.BaseSha256 := C.BaseSha256`; `St.ValLoss := Val`.
   - `Dir := Ck.Save(St, S, O)` (T16 `Save`: through the temporary
     directory and rename, then the retention of `Keep`);
     `AppendLine(LogFile, Format('ckpt'#9'%d'#9'%s'#9'%s', [U, G17(Val),
     Dir]))`.
4h. **The stop** — `(C.MaxUpdates > 0) and (U >= C.MaxUpdates)` →
   `AOut.Add(Format('stopped after update %d of %d', [U, Total]))` and
   `Exit(0)`.
4i. The next update (loop 4b), or — all the batches of the epoch done —
   `Inc(Epoch); Pos := 0` (the next epoch from the beginning) and loop 4a.

**Step 5 — the end (exit 0)**

5a. All epochs done:
   `AOut.Add(Format('done: %d updates, best %s', [U, Ck.Best]))` —
   `Ck.Best` is the checkpoint of the lowest finite validation loss (T16
   `Best`; `''` when there is no finite one, e.g. without `--valid`) —
   and `Result := 0`. The best path is the output for T18's merge
   (system flow, "Outputs").

**Step 6 — the error (exit 1)**

6a. Every raise of the setup (step 2), the resume (step 3) or the run
   (e.g. a file error of the log, a tch-fpc error) is caught by the
   single `except on E: Exception`: `AErr.Add(E.Message)` — the message
   alone, one line — and `Result := 1` (4.3, #9–#14).

**Step 7 — the release**

7a. The `finally` frees the eight objects in reverse creation order (step
   2 note): `Ck.Free; O.Free; M.Free; S.Free; ValidSet.Free;
   TrainSet.Free; Tok.Free; G.Free;` — also on the `Exit(2)`, `Exit(3)`
   and `Exit(0)` paths.

### 4.3 Checks and messages

All raised or written exactly as in the T17 facts ("Checks and messages
(in this order)"). `ETrainStep` for the rows of `TrainStep`; the rows of
`RunTrainCLI` come out as exit codes.

`ETrainStep` (unit `TrainStep`), raised in this order:

| # | Where | Condition (checked in this order) | Message |
|---|---|---|---|
| 1 | `CheckSample`, called first by both `SampleLossBackward` and `SampleLoss` | `ASlice < 1` | `CreateFmt('slice must be positive, got %d', [ASlice])` |
| 2 | `CheckSample` (after #1) | `ASample.LossFrom < 1` or `ASample.LossTo <= ASample.LossFrom` or `ASample.LossTo > Length(ASample.Tokens)` | `CreateFmt('%s: loss span %d..%d outside 1..%d', [ASample.Id, ASample.LossFrom, ASample.LossTo, Length(ASample.Tokens)])` |
| 3 | `SampleLossBackward` only (after #1–#2) | `not (ADenom > 0)` | `CreateFmt('denominator must be positive, got %g', [ADenom])` |

`ParseTrainArgs` (the usage error; `RunTrainCLI` prints it as the first
line of `AErr` and the `Usage` line as the second, **exit 2**), in this
order:

| # | Where | Condition | Message |
|---|---|---|---|
| 4 | `ParseTrainArgs` | an option without a value (it is the last argument) | `<option>: value missing` |
| 5 | `Int` (nested) | the value is not an integer in the option's range (UT row 104: `--rank 0`, `--epochs x`) | `<option>: an integer in <lo>..<hi>, got "<v>"`, with the ranges: `--rank` 1..4096, `--warmup` 0..MaxInt, `--epochs` 1..1000, `--seed` −MaxInt..MaxInt, `--pool` 1..MaxInt, `--budget` 1..MaxInt, `--batches-per-update` 1..1000, `--ckpt-every` 1..MaxInt, `--keep` 1..1000, `--slice` 1..MaxInt, `--split` 0..1000, `--max-updates` 0..High(Int64) |
| 6 | `Num` (nested) | the value is not a finite number `>= 0` for `--alpha`, `--lr`, `--min-lr-ratio`, `--clip` (0 itself is accepted; NaN and ±Inf refused — UT row 104: `--lr -1`, `--clip nan`) | `<option>: a number >= 0, got "<v>"` |
| 7 | `ParseTrainArgs` | an option not in the dispatch | `unknown option <option>` (UT row 104: `--frob 1`) |
| 8 | `ParseTrainArgs` | the positionals are not exactly three | `expected MODEL TRAIN OUTDIR, got <n> arguments` (UT row 104: `got 2 arguments`, `got 4 arguments`) |

`RunTrainCLI` (**exit 1**, the message alone on `AErr`), in this order:

| # | Where | Condition | Message |
|---|---|---|---|
| 9 | step 2a | `not FileExists(C.Model)` | `no model file <model>` (UT row 208: `no model file <tmp>none.gguf`) |
| 10 | step 2b–2d | from the GGUF, the tokenizer and TrainData (T01, T05, T15's own checks pass through) | e.g. `no data file <train>` (UT row 208: `no data file <tmp>none.jsonl`) |
| 11 | step 2e | `TrainSet.Count = 0` | `no training sample in <train>` (UT row 208: with `OtherFile` as train) |
| 12 | step 2h | `C.Warmup >= Total` | `CreateFmt('warm-up %d must be below the %d updates', [C.Warmup, Total])` (UT row 208: `warm-up 8 must be below the 8 updates`) |
| 13 | step 2i | from `TQwenModel.Load` (T11's own checks pass through) | e.g. `No CUDA GPUs are available` with `--split` and the GPUs hidden; `split 3 outside 0..2` (UT row 212) |
| 14 | step 3, in this order | `Ck.Latest = ''`; then `St.ConfigHash <> Hash`; then `St.BaseSha256 <> C.BaseSha256` | `nothing to resume in <outdir>ckpt/`; `the configuration differs from the checkpoint's: <dir>`; `the base model differs from the checkpoint's: <dir>` (UT row 206; `Ck.Load`'s own errors — e.g. the T16 rank/scale refusal — pass through before the two comparisons) |

The non-finite loss (**exit 3**, step 4c):

| Where | AErr | Log line written before the exit |
|---|---|---|
| step 4c | `non-finite loss at update <U + 1>; stopped without updating` (UT row 207: `non-finite loss at update 2; stopped without updating`) | `<U + 1> <Epoch> <Pos> nan` (4 fields, `G17(Loss)`) |

The success lines (stdout, `AOut`):

| Where | Message |
|---|---|
| step 3f (with `--resume`) | `resumed from <dir> at update <U>` (UT row 204: `resumed from <B>ckpt/ckpt-00000003 at update 3`) |
| step 4h (`--max-updates` reached) | `stopped after update <U> of <Total>` (UT row 204: `stopped after update 3 of 8`) |
| step 5a (all epochs done) | `done: <U> updates, best <Ck.Best>` (UT row 201: `done: 8 updates, best <O>ckpt/ckpt-00000008`; without `--valid` the `best` part is empty — UT row 211) |

The usage error (step 1b, **exit 2**) is printed as two lines on `AErr`:
first the usage error of 4.3 #4–#8 (e.g. UT row 209:
`--rank: an integer in 1..4096, got "0"`), then the `Usage` line
`usage: qlora_train [options] MODEL TRAIN OUTDIR (see the unit header)`.

Notes:

- A UT / ITa row that expects an error compares the **whole** message
  (T17 facts, "Unit tests"); a `RunTrainCLI` error message is the only
  line on `AErr` (exit 1) or, for exit 2, the line before the `Usage`.
- The pass-through errors of #10 and #13 keep their own messages and
  classes (T01 / T05 / T15 / T11), `ETrainStep`-style or tch-fpc; they
  are caught by the step 6 `except` like the others.
- The exit codes: 0 done (step 5a) or `--max-updates` reached (step 4h);
  1 error (step 6); 2 usage (step 1b); 3 non-finite loss (step 4c)
  (external design, "Exit codes"; the unit header).

### 4.4 Business rules of the external design and where they are handled

| Business rule (`T17-external.md`) | Handled by |
|---|---|
| one sample per forward; every sample's loss divided by the loss-token count of the whole update, so the gradient is the update's mean loss | `SampleLossBackward` with `ADenom` (4.1.4 steps 6–7), `RunTrainCLI` step 4b (the `Denom` over the update's samples) |
| the head only on the loss rows `LossFrom - 1 .. LossTo - 2`, in slices of `--slice` rows, each slice divided by the denominator and sent backward on its own, then the trunk block-wise | `SampleLossBackward` steps 4–7 (4.1.4), `SliceLoss` (4.1.2), `Targets` (4.1.1) |
| one update: `--batches-per-update` consecutive batches, then `ClipGlobalNorm(clip)`, `Step(ScheduleLR(...))`, `ZeroGrad` | `RunTrainCLI` step 4d (in this order) |
| the schedule's total = the sum over the epochs of `ceil(batches / batches per update)` (the batches differ between epochs) | `TotalUpdates` (4.2.6), `RunTrainCLI` steps 2g, 4d |
| a non-finite loss stops the run before its update, exit 3 (the log line is written; no step, no checkpoint) | `RunTrainCLI` step 4c, 4.3 (exit 3 row) |
| checkpoints in `OUTDIR/ckpt` after every `--ckpt-every` updates and at the end of each epoch | `RunTrainCLI` steps 2m, 4g (T16 `Save` for the layout and the retention) |
| at an epoch's end the state is the next epoch's start (epoch + 1, position 0) | `RunTrainCLI` step 4g (`St.Epoch := Epoch + Ord(EndOfEpoch)`, `St.Position := 0`) |
| the validation loss at a checkpoint when `--valid` is given, else NaN | `RunTrainCLI` step 4g, `ValidationLoss` (4.1.6) |
| `--resume`: the latest checkpoint; a different configuration (`ConfigHash`) or a different base sha256 is refused | `RunTrainCLI` step 3, `ConfigHash` (4.2.5), 4.3 #14 |
| the configuration hash: the contents of TRAIN and VALID and the settings that shape the run; excluding `--max-updates`, `--resume`, `--split`, `--ckpt-every`, `--keep`, the output directory and the base's sha256 | `ConfigHash` (4.2.5), step 2m / step 3c |
| the base's sha256: `--base-sha256`, or the name of an Ollama blob `sha256-<64 hex>`, else empty (not computed: stock FPC 3.2.2 has no sha256) | `ShaFromName` (4.2.4), `ParseTrainArgs` (4.2.3, last step), step 3d |
| the log `OUTDIR/train-log.tsv` appended line by line during the run: the update line `update epoch position loss lr norm tokens seconds`, the checkpoint line `ckpt update val-loss dir` | `AppendLine` and `G17` (4.2.7), `RunTrainCLI` step 4f and 4g |
| the exit codes 0 (done or `--max-updates`), 1 (error), 2 (usage), 3 (non-finite loss) | `RunTrainCLI` steps 1b, 4c, 4h, 5a, 6 |
| the error list (external design, "Errors") | 4.3 (#1–#14, the exit-3 row, the success and usage lines) |
| `ManualSeed(seed)` before `AttachLora` (the A matrices); the epoch order is TrainData's SplitMix64 from the same seed | `RunTrainCLI` steps 2j–2k, 4a (T15 `EpochBatches`), `SampleLossBackward`/`LoraSet` (T12) |
| the outputs for T18: `OUTDIR/ckpt` with the adapters, the optimizer state and the validation loss; the best checkpoint's path | `RunTrainCLI` step 4g (the T16 `state.json` + `lora.ot` + `moments.ot`) and step 5a (`Ck.Best` in the exit message) |
| the command line (every option and its default) and the usage error + `Usage` line | `ParseTrainArgs` (4.2.3), `DefaultTrainConfig` (4.2.2), `Usage` (3), step 1b |

## 5. Change history

| Date | Change | Reason |
|---|---|---|
| T17a (initial) | First edition, from the external design and the T17 facts. | new program |
