# T21 facts (checked by the harness owner)

T21 adds bf16 compute on the GPUs (DESIGN 8.13 "compute dtype": bf16 on
the GPUs, fp32 on the CPU) and draws the adapters' initial values on the
CPU. It changes five existing units: `PackedLinear` (`Compute`), `Lora`
(`TLoraPair.Create`), `QwenModel` (`Compute`), `TrainCli` and
`LogprobsCli` (`--compute fp32|bf16`). The programs `qlora_train` and
`qlora_logprobs` are unchanged.

Decisions of the harness owner for this task (recorded in DESIGN 8.13 as
"As written for T21"):
- **Only the matrix products of the trunk blocks run in bf16.** With
  `Compute = kBFloat16`, `TPackedLinear.Forward` casts X (fp32) and the
  dequantised weight to bf16, adds the LoRA `Delta` of the bf16 X (A and B
  are cast by `Delta`, T12), and casts the sum back to fp32. Everything
  else stays fp32: the residual stream, the RMS norms, RoPE, the
  convolution, the delta rule, the attention around the projections, the
  head and the loss (DESIGN 8.19 R2).
- **The head stays fp32.** bf16 logits keep three significant digits
  (steps of 0.0625 at the rehearsal model's logits, measured); the head is
  dequantised `HeadChunk` rows at a time, so its fp32 product costs little
  memory.
- **The default is fp32 and the fp32 path is unchanged bit for bit** (the
  old `Forward` body runs when `Compute = kFloat`): every earlier UT row,
  ITa and oracle stays as it is.
- **`lora_a` is drawn on the CPU** (`TTensor.zeros(.., kFloat,
  TDevice.Cpu).uniform_(-Bound, Bound)`, the same operation as
  `TVarPath.Uniform` on a CPU store) and copied into P's store with
  `VarCopy`: a run starts from the same adapter whatever the device map
  (measured at M2g-a: drawn on the GPU's own generator, a two-card run
  started from another A, gradient norm 3.44 against 3.77 at update 1). On
  the CPU the values are those of T12.
- `--compute` is a setting of the run like `--split`, **not part of the
  configuration hash**: a checkpoint written in fp32 resumes in bf16 and
  the other way round.
- bf16 runs on the CPU too, so the UT and the ITa check bf16 without a
  GPU. The GPU check (the two-card split in bf16 against the CPU curve) is
  M2g-b, run by the harness owner.

Measured by the harness owner with a prototype written exactly as below:
- UT: every row of the tables below passes with `-Cr` (the earlier rows
  of the five test programs too); the five units then have 100 % line
  coverage (84, 112, 119, 250, 100 lines); mutants are caught (the bf16 product without the casts: PackedLinear 302, 303, QwenModel 302, TrainCli 302, LogprobsCli 301; Delta of the fp32 X: PackedLinear 303; no cast back to fp32: PackedLinear 302, 303, QwenModel 302, TrainCli 302, LogprobsCli 301; any kind accepted: PackedLinear 301, QwenModel 301; FCompute not set in the constructor: PackedLinear 301 and many earlier rows; A drawn in 0..Bound: Lora 301; the full block's or the FFN matrices left out, or FCompute not stored: QwenModel 301; M.Compute not set: TrainCli 302, LogprobsCli 301; bf16 parsed as fp32: TrainCli 301, 302, LogprobsCli 301).
  Drawing `lora_a` on P's device again cannot be seen on the CPU; M2g-b
  checks it (the gradient norm at update 1 against the CPU run).
- ITa (cases below): 0, 0.1883, 0, 5.33e-3, 0, 0; about 7 minutes.
- On the two cards (M2g-a, the prototype): fp32 from its own start within
  5.5e-3 of the CPU curve over 16 updates (gradient norm at update 1:
  3.76867 against 3.76875); bf16 within 6.3 % (2.2e-3 up to update 9),
  ending at 2.9 % of the first loss, 158 s against 219 s.

These facts are final; do not investigate further or write probes.

## Additions to `PackedLinear` (interface in the skeleton)

Interface (already in `src/PackedLinear.pas`): the class gains

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

and the constructor already ends with `FCompute := kFloat;`.

Implementation: `SetCompute` (replaces the skeleton's body) and `Forward`
(the checks stay first, as they are; the fp32 branch is the old body):

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

procedure TPackedLinear.SetCompute(AValue: TKind);
begin
  if (AValue <> kFloat) and (AValue <> kBFloat16) then
    raise EPackedLinear.Create('compute must be kFloat or kBFloat16');
  FCompute := AValue;
end;
```

## Change to `Lora` (`TLoraPair.Create`)

One line of `TLoraPair.Create` changes (the comment of the interface is
already in the skeleton). Instead of `FA := P.Uniform('lora_a', [ARank,
AInDim], -Bound, Bound);` write:

```pascal
FA := P.VarCopy('lora_a', TTensor.zeros([ARank, AInDim], kFloat, TDevice.Cpu).uniform_(-Bound, Bound));
```

## Addition to `QwenModel` (interface in the skeleton)

Interface (already in `src/QwenModel.pas`): the class gains `FCompute:
TKind`, `procedure SetCompute(AValue: TKind)` and

```pascal
    { kFloat (the default) or kBFloat16: sets every matrix of the trunk
      blocks (TPackedLinear.Compute); the head stays fp32 (T21). }
    property Compute: TKind read FCompute write SetCompute;
```

`Load` already sets `FCompute := kFloat`; `Logits` is unchanged (fp32).
Implementation of `SetCompute` (replaces the skeleton's body):

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

## Additions to `TrainCli`

Already in the skeleton: the header's `--compute K` lines, the field
`Compute: TKind` of `TTrainConfig` (`kFloat` or `kBFloat16`) and
`Result.Compute := kFloat;` in `DefaultTrainConfig`. `ConfigHash` is
unchanged (the kind is not in the hash). Write:

1. In `ParseTrainArgs`, right after the `--split` branch (the option's
   value is `V`, as for the other options):

```pascal
    else if A = '--compute' then
    begin
      if V = 'fp32' then C.Compute := kFloat
      else if V = 'bf16' then C.Compute := kBFloat16
      else Exit(Format('--compute: fp32 or bf16, got "%s"', [V]));
    end
```

2. In `RunTrainCLI`, right after the model is loaded (the `if C.Split < 0
   then ... else ...` that calls `TQwenModel.Load`) and before
   `ManualSeed(C.Seed);`:

```pascal
      M.Compute := C.Compute;
```

## Additions to `LogprobsCli`

Already in the skeleton: the header and `K_USAGE` = `Usage: qlora_logprobs
[--top N] [--split S] [--compute K] MODEL FILE`. Write in
`RunLogprobsCLI`:

1. A local `Compute: TKind;`, set to `kFloat` right after `Top := 20;`.
2. The option loop takes `--compute` like `--top` and `--split` (one
   value; without it `'--compute needs a value'` as for the others):

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

3. Right after the model is loaded (after the `if Split < 0 then ... else
   ...` that calls `TQwenModel.Load`):

```pascal
      M.Compute := Compute;
```

## Checks and messages (in this order)

- `TPackedLinear.SetCompute`: `EPackedLinear` `compute must be kFloat or
  kBFloat16` for any other kind; `Compute` keeps its value.
- `TPackedLinear.Forward`: the checks of T10 first and unchanged (`X must
  be kFloat (fp32)` also when `Compute` is `kBFloat16`).
- `TQwenModel.SetCompute`: `EQwenModel` `compute must be kFloat or
  kBFloat16` for any other kind, before any matrix is changed.
- `ParseTrainArgs`: `--compute: fp32 or bf16, got "<value>"` (values
  compared literally: `BF16` is refused); a missing value gives the
  existing `--compute: value missing`.
- `RunLogprobsCLI` (usage errors, exit 2, as the other options):
  `--compute needs a value`; `--compute must be fp32 or bf16, got
  <value>`.

## UT fixture and helpers (harness owner's; nobody changes them)

The existing helpers of the five test programs (T10, T11, T12, T17):
- `PackedLinear_test`: `Arr(T)`, `DiffArr(A, B)`, `NewLinear` (an F32
  matrix 6 x 8, W[o, c] = 0.01 (8 o + c) − 0.2), `FixX` (X [5, 8] =
  sin(1 + i), **kDouble**: cast it with `.totype(kFloat)`), `NewPair(S,
  In, Out)` (the pair `q`, rank 3, scale 2), `SetB(P)`.
- `Lora_test`: `NewPair(S)` and the others of T12.
- `QwenModel_test`: `TiedPath` (the tiny model: block 0 linear, block 1
  full; **15 matrices**: 3 FFN + 5 linear-block, 3 FFN + 4 full-block),
  `Ids8`, `NoGradLogits(M)`.
- `TrainCli_test`: `RunArgs(Out, Extra)` (a run of 8 updates on the tiny
  model), `Run(Args, Out, Err)`, `NewOut`, `UpdateLines(Out)` (the update
  lines without the seconds column, joined by #10), `CheckParse(Self,
  Args, Message)`, `TrainFile`.
- `LogprobsCli_test`: `RunCli(Args, Out, Err)`, `Want(Id, Ids, Top)` (the
  expected line of the fp32 model), `CheckUsage(Self, Args, Reason)` (its
  usage line is already the new one), `TinyPath`, `PromptPath`.
A row may add a small helper of its own next to its method (e.g. a
`Want` with the model's `Compute` set to a kind).

## Unit tests (rows; every row has one method, every method belongs to a row)

Error rows catch the named exception class and compare the whole message;
a row fails if no exception comes. "exact" means a difference of 0. Use
fpcunit's checks (`AssertEquals`, `AssertTrue(msg, cond)`); plain
`Assert` is refused by the gate.

`PackedLinear` (stage 1):

| Row | Input | Expected |
|---|---|---|
| 301 | `NewLinear`: `Compute`; set `kBFloat16`; set `kHalf`; set `kDouble`; set `kFloat` | `kFloat`; `kBFloat16`; `EPackedLinear` `compute must be kFloat or kBFloat16` and still `kBFloat16`; the same message; `kFloat` |
| 302 | `NewLinear`, X := `FixX.totype(kFloat)`, Y32 := `Forward(X)`; `Compute := kBFloat16`, Y := `Forward(X)`; then `Compute := kFloat` | Y kind `kFloat`, shape [5, 6]; Y equals `X.totype(kBFloat16).matmul(Weight.totype(kBFloat16).transpose(0, 1)).totype(kFloat)` (exact); max \|Y − Y32\| > 0 and ≤ 0.02; `Forward(X)` equals Y32 again (exact) |
| 303 | `NewLinear`, pair `NewPair(S, 8, 6)`, `SetB(P)`, `L.Lora := P`, `Compute := kBFloat16`; Y := `Forward(X)`; `Y.sum(kNone).Backward` | Y kind `kFloat`; Y equals `(Xc.matmul(Weight.totype(kBFloat16).transpose(0, 1)) + P.Delta(Xc)).totype(kFloat)`, Xc = X in bf16 (exact); A.grad and B.grad kind `kFloat`, both with a sum of squares > 0 |
| 304 | `Compute := kBFloat16`; `Forward(FixX.totype(kBFloat16))` | `EPackedLinear` `X must be kFloat (fp32)` |

`Lora` (stage 1):

| Row | Input | Expected |
|---|---|---|
| 301 | `ManualSeed(7)`, a CPU store S1, P := `TLoraPair.Create(S1.Root / 'p', 'p', 8, 6, 3, 2.0)`, N1 := `TTensor.rand([4], kFloat, TDevice.Cpu)`; `ManualSeed(7)`, a CPU store S2, Old := `S2.Root.Uniform('x', [3, 8], -1 / Sqrt(8), 1 / Sqrt(8))`, N2 := `TTensor.rand([4], kFloat, TDevice.Cpu)` | A equals Old (exact); N1 equals N2 (exact: the generator is consumed alike); `P.A.RequiresGrad`; `S1.Count` 2; B all zero |

`QwenModel` (stage 2):

| Row | Input | Expected |
|---|---|---|
| 301 | `TQwenModel.Load` of `TiedPath` on the CPU, split 1: `Compute`; the matrices of every trunk block (`M.Stages[I] as TQwenBlock`, `Target` of `ffn_gate`, `ffn_up`, `ffn_down` and the kind's own: `attn_qkv`, `attn_gate`, `ssm_beta`, `ssm_alpha`, `ssm_out` or `attn_q`, `attn_k`, `attn_v`, `attn_output`) with `Compute = kFloat`; set `kBFloat16`; set `kHalf`; set `kFloat` | `kFloat`, 15 of 15; `kBFloat16`, 15 of 15 in bf16; `EQwenModel` `compute must be kFloat or kBFloat16`, still `kBFloat16` and 15 of 15 in bf16; 15 of 15 in fp32 |
| 302 | the same model: L32 := `NoGradLogits(M)`; `Compute := kBFloat16`, L16 := `NoGradLogits(M)`; under `NoGradGuard` H := `M.Trunk(M.Embed(Ids8))`, A := `M.Logits(H)`; `Compute := kFloat`, B := `M.Logits(H)`, then `NoGradLogits(M)` | L16 kind `kFloat`; max \|L16 − L32\| > 0 and, divided by max \|L32\|, ≤ 0.05; A equals B (exact: the head is fp32); the last logits equal L32 (exact) |

`TrainCli` (stage 2):

| Row | Input | Expected |
|---|---|---|
| 301 | `ParseTrainArgs` of `m t o`; of `--compute bf16 m t o`; of `--compute fp32 m t o`; `--compute fp16 m t o`; `--compute BF16 m t o`; `m t o --compute`; `ConfigHash` of the first config with `Train := TrainFile`, against the same with `Compute := kBFloat16` | `''` and `kFloat`; `''` and `kBFloat16`; `''` and `kFloat`; `--compute: fp32 or bf16, got "fp16"`; `--compute: fp32 or bf16, got "BF16"`; `--compute: value missing`; equal |
| 302 | `Run(RunArgs(O1, ['--max-updates', '2']))`; `Run(RunArgs(O2, ['--max-updates', '2', '--compute', 'bf16']))` | both exit 0; the second's err `''` and out `stopped after update 2 of 8`; as many update lines; the loss (fourth column) of the first update differs from the fp32 run's, relative difference ≤ 0.01 |

`LogprobsCli` (stage 2):

| Row | Input | Expected |
|---|---|---|
| 301 | `RunCli(['--compute', 'bf16', '--top', '3', TinyPath, PromptPath])`; `RunCli(['--top', '3', '--compute', 'fp32', TinyPath, PromptPath])` | exit 0, no stderr, 3 lines; line 1 equals the expected line of record `a` (ids 5, 17, 2) with the model's `Compute := kBFloat16` (as `Want`, top 3), and differs from `Want('a', [5, 17, 2], 3)`; the fp32 call: exit 0, line 1 = `Want('a', [5, 17, 2], 3)` |
| 302 | `--compute` alone; `--compute fp16 TinyPath PromptPath` | `CheckUsage`: `--compute needs a value`; `--compute must be fp32 or bf16, got fp16` |

Build and run (the GPUs hidden as in the gate), for `<P>` = `PackedLinear`,
`Lora`, `QwenModel`, `TrainCli` or `LogprobsCli`:
`fpc -Mobjfpc -Sh -vew -l- -Fu{proj}/src -Fu{proj}/../../src -Fu{proj}/tests/ut -FU{proj}/work/bin -Fl{proj}/../../build/libtch-rocm -k-rpath={proj}/../../build/libtch-rocm {proj}/tests/ut/<P>_test.pas -o{proj}/work/bin/<P>_test`
then `HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 {proj}/work/bin/<P>_test --all --format=plain`.
Warnings from other units (`GgufQuant`, `GgufQuantT`, `QwenTokenizer`,
`generics.dictionaries.inc`) and libtorch's GPU warnings on stderr are
expected.

## ITa (prepared by the harness owner)

`QLORA_MODEL=<rehearsal GGUF> {proj}/tests/ita/T21-run.sh` builds
`src/qlora_logprobs.pas` and `src/qlora_train.pas` against `src/` and runs
them on the CPU: the T11 prompts (`work/oracle/T11/prompts.jsonl`) with
the default kind (top 20) and twice with `--compute bf16` (top 100); the
T17 training records (`tests/ita/T17/train.jsonl`, 4 epochs of one update,
lr 1e-3, warm-up 1, a checkpoint each) with the default kind, with
`--compute bf16`, and 2 updates in fp32 then `--compute bf16 --resume`.
The driver `tests/ita/T21/check.bb` compares against the oracle of the
code before T21 (`work/oracle/T21/lp-fp32.jsonl`, `train-fp32.tsv`).
Writes `{proj}/work/ita/T21/summary.tsv` (case, OK/NG, value, limit, note;
then the exit codes and the count of stderr lines other than the GPU
warnings). OK when value ≤ limit. About 7 minutes.

| Case | Compared | Limit |
|---|---|---|
| ITa-T21-001 | default kind: the logprobs and the 4 update lines byte-identical to the code before T21 (count of differences) | 0 |
| ITa-T21-002 | `--compute bf16`: largest \|logprob difference\| over the fp32 top 10 of each of the 27 prompts | 0.25 |
| ITa-T21-003 | `--compute bf16`: prompts whose top-1 token differs where the fp32 margin exceeds 0.5 (count) | 0 |
| ITa-T21-004 | `--compute bf16` training: largest relative loss difference to fp32 over 4 updates | 0.02 |
| ITa-T21-005 | 2 updates fp32, then `--compute bf16 --resume`: 4 updates, the first 2 identical to fp32, the rest within 0.02 (count of violations) | 0 |
| ITa-T21-006 | `--compute bf16` twice: the logprobs byte-identical (1 if not) | 0 |

Measured with the prototype: 0, 0.1883, 0, 5.33e-3, 0, 0; about 7 minutes. With the code before T21 (and the
skeleton) 001 is OK and 002–006 are NG (`--compute` is an unknown option).

## Stages and case IDs

- Stage 1 (task T21b1): `TPackedLinear.SetCompute` and `Forward`;
  `TLoraPair.Create`'s line; rows UT-PackedLinear-301 to 304, UT-Lora-301.
- Stage 2 (task T21b2): `TQwenModel.SetCompute`; `TrainCli`'s branch and
  line; `LogprobsCli`'s local, option and line; rows UT-QwenModel-301 to
  302, UT-TrainCli-301 to 302, UT-LogprobsCli-301 to 302.
- ITa: ITa-T21-001 to 006.
