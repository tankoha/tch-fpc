# T18 facts (checked by the harness owner)

T18 is the D3 writer (DESIGN 8.15, 8.11 M2, 8.18): the unit `GgufMerge`
(the adapted matrices of a base GGUF, the checkpoint's adapters, one
merged tensor, the merged file, the command line) and the program
`src/qlora_merge.pas` (the harness owner's thin wrapper, complete; like
`qlora_train`). It uses `GgufFile` (T01), `GgufQuant` (T02),
`GgufWriter` (T04), `QwenBlocks` (T10), `Lora` (T12), `Checkpoint` (T16)
and `TrainCli.ShaFromName` (T17).

DESIGN 8.15: `qlora_merge <base.gguf> <checkpoint> <out.gguf>` copies the
metadata as bytes, rebuilds the tensor directory in the base's order with
the adapted tensors as Q8_0 (offsets recomputed, alignment of the base),
copies every other tensor byte for byte, and writes each adapted tensor as
the base dequantised to fp32 plus `s·B·A` (`TLoraPair.MergedDelta`),
quantised to Q8_0 (`QuantizeQ8_0`, T02: `roundf`). Decisions of the
harness owner for this task:
- **The targets** are those of `TQwenModel.AttachLora` (T12): per trunk
  block (the blocks without the `qwen35.nextn_predict_layers` MTP block)
  its kind's `LoraTargets` — linear `attn_qkv`, `attn_gate`, `ssm_out`;
  full (`IsFullBlock`) `attn_q`, `attn_k`, `attn_v`, `attn_output` — taken
  from the GGUF alone (no model is loaded: the 27B base is 16.8 GB), with
  the dimensions of `<name>.weight` (GGUF `Dims[0]` = in, `Dims[1]` =
  out).
- **The adapters** are read into a `TLoraSet` of the checkpoint's rank and
  scale (`ReadRunState`), one CPU pair per target, from `<ckpt>/lora.ot`.
- **The file** is written to `<out>.part` and renamed (rename(2) replaces
  an existing file); on a failure the part file is removed (the writer's
  destructor, or here when the rename fails).
- **The base's sha256**: as in T17, not computed. `--base-sha256 H`, else
  the base's file name `sha256-<64 hex>` (`ShaFromName`); when both that
  and the checkpoint's `base_sha256` are known they must be equal
  (compared literally).
- **The fidelity criterion** (decided by the user 2026-11-25, DESIGN 8.15
  "Decision"): on the rehearsal model the merged file is compared with the
  forward with the adapter against the retyping floor (the same targets
  merged with B = 0 against the base), at most 1.5 ×; at least 90 % of the
  loss gain kept; H47: the bundled `llama-server` on an all-F32 copy of the
  merged file against the Pascal forward on it, at most 0.02.

Measured by the harness owner with a prototype written exactly as below:
- UT: every row of the tables below passes with `-Cr`; the unit then has
  100 % line coverage (129 lines); mutants are caught (without
  `MergedDelta`: 004, 005, 014; the dimensions swapped in `WriteMerged`:
  011, 013, 014, 101, 105; no `SetTensorType`: 011, 013, 014, 101, 105;
  the MTP block not subtracted: 002; the sha256 not compared: 105; the
  output compared with the base by name only: 103; the part file kept
  after a failed rename: 013; the block kinds swapped: every row but 102,
  103, 104).
- On the rehearsal model with the M2 exit checkpoint (32 samples, update
  16): 78 tensors, 532 MB to 629 MB in a few seconds; round trip 0.63
  steps; gain kept 99.99 %; merged against the adapter 0.318, floor 0.243
  (1.31); H47 1.3e-3.
- ITa (cases below): 0, 0.643, 0.608, 3.75e-4, 1.74e-3, 0.

These facts are final; do not investigate further or write probes.

## Unit `GgufMerge` (interface, final; in the skeleton)

```pascal
const
  GgufTypeQ8_0 = 8;

type
  EGgufMerge = class(Exception);

  { One adapted matrix: the adapter name (as TQwenModel.AttachLora gives
    it), the index of '<Name>.weight' in the base's directory, its
    dimensions (GGUF Dims[0] = InDim, Dims[1] = OutDim). }
  TMergeTarget = record
    Name: string;
    Tensor: Integer;
    InDim, OutDim: Int64;
  end;
  TMergeTargets = array of TMergeTarget;

{ The adapted matrices of AGguf: per trunk block 0..trunk-1 its kind's
  LoraTargets (linear: attn_qkv, attn_gate, ssm_out; full: attn_q,
  attn_k, attn_v, attn_output). Raises EGgufMerge when '<name>.weight'
  is missing or not two-dimensional. }
function MergeTargets(AGguf: TGgufFile): TMergeTargets;

{ A TLoraSet of AState's rank and scale with one CPU pair per target,
  loaded from ADir's lora.ot (the caller frees it). }
function LoadAdapters(const ATargets: TMergeTargets; const ADir: string;
  const AState: TRunState): TLoraSet;

{ The tensor of type ATType (AIn × AOut elements in ABytes) dequantised
  to fp32, plus AP.MergedDelta, quantised to Q8_0. Raises EGgufMerge when
  AP's dimensions differ from AIn, AOut. }
function MergeTensor(ATType: Cardinal; const ABytes: TBytes; AIn, AOut: Int64;
  AP: TLoraPair): TBytes;

{ Writes AGguf to AOutPath with the pairs of ASet merged (nil: no pair,
  a byte-identical copy). Written to AOutPath + '.part' and renamed; the
  part file is removed on failure. Returns the number of merged tensors. }
function WriteMerged(AGguf: TGgufFile; const ATargets: TMergeTargets; ASet: TLoraSet;
  const AOutPath: string): Integer;

{ qlora_merge [--base-sha256 H] BASE CKPT OUT; AOut / AErr caller-owned;
  the result is the exit code (0 done, 1 error, 2 usage). }
function RunMergeCLI(const AArgs: array of string; AOut, AErr: TStrings): Integer;
```

## Implementation of `GgufMerge` (write it exactly so)

```pascal
implementation

uses
  TrainCli;   { ShaFromName }

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

## The program `src/qlora_merge.pas` (harness owner's; complete)

It forwards the command line to `RunMergeCLI`, prints the output lines to
stdout and the error lines to stderr, and exits with the code. Nobody
changes it.

## Checks and messages (in this order)

`MergeTargets` (EGgufMerge), per target in order: `no tensor
<name>.weight`; `<name>.weight has <n> dimensions, expected 2`. Errors of
`ReadQwenDims` (EQwenBlock, a missing key) pass through.

`LoadAdapters`: the errors of `TLoraSet.Load` pass through (a missing
file: ETorchNativeError `open file failed because of errno 2 on fopen: No
such file or directory, file path: <dir>lora.ot`).

`MergeTensor` (EGgufMerge), before anything else: `<pair name>: adapter
<in> -> <out>, tensor <in> -> <out>`; then the errors of `DequantizeBytes`
and `QuantizeQ8_0` (EGgufError).

`WriteMerged`: EGgufMerge `no adapter <name>` (the first target, in
order, without a pair in the set; checked before anything is written);
the writer's errors (EGgufError, e.g. `TGgufWriter.BeginWrite: cannot
create file <out>.part`); EGgufMerge `cannot rename <out>.part to <out>`
(the part file removed).

`RunMergeCLI`: usage errors, exit 2 (the message alone on AErr):
`--base-sha256: value missing`; `unknown option <arg>` (any argument
longer than one character that starts with `-`, compared literally);
`expected BASE CKPT OUT, got <n> arguments`. Then exit 1: `no model file
<base>`; `the output must not be the base model` (the expanded paths
equal); from `ReadRunState` `not a checkpoint: <ckpt>`; `the base model
differs from the checkpoint's: <ckpt>`; then any exception's message
(GGUF, targets, adapters, writing). Success, exit 0: AOut `merged <n>
tensors (rank <r>, scale <s>) into <out>` (`%g` for the scale).

## UT fixture and helpers (harness owner's; nobody changes them)

The tiny model of T10/T11 (`tests/ut/QwenBlocksFixture.pas`):
`WriteTinyQwenModel(TinyPath, 5, thTied, False)` and the same with the MTP
key (`NextNPath`, `ANextN` True). Two trunk blocks: blk.0 linear
(`attn_qkv` Q8_0 64 → 128, `attn_gate` F16 64 → 64, `ssm_out` Q8_0 64 →
64), blk.1 full (`attn_q` Q8_0 64 → 256, `attn_k` Q8_0 64 → 64, `attn_v`
F16 64 → 64, `attn_output` Q8_0 128 → 64); `blk.0.attn_norm.weight` is
F32 [64].

`tests/ut/GgufMerge_test.pas` (skeleton: helpers, runner, no test
methods): `NewDir` (a fresh directory, created, ending in the delimiter);
`FileBytes`, `SaveBytes`, `SameBytes`; `PatchedCopy(Src, Dst, Old, New)`
(every occurrence of each `Old[i]` replaced by `New[i]`, same lengths, in
order; with empty arrays a plain copy); `NewSet(T, Seed)` (rank 4, scale 2,
one CPU pair per target after `ManualSeed(3)`; every B := 0.1 N(0, 1)
after `ManualSeed(Seed)`, B zero for Seed < 0); `SaveCkpt(Dir, S, Sha)`
(`lora.ot` and `state.json`: update 8, the set's rank and scale,
BaseSha256 Sha); `RefMerged(Type, Bytes, In, Out, P)` (the merge in
Double, scalar); `StepErr(Q8, Ref)` (max |dequantised − Ref| / the
block's d); `TensorBytes(G, Name)`; `TopDiff(A, B)` (mean |Δ logprob|
over A's top 10 at the last position of tokens (7 i + 3) mod 270, i =
0..19); `RunCli(Args, Out, Err)` (the exit code; the lines joined by
#10). Variables `TinyPath`, `NextNPath`, `TmpDir`.

## Unit tests (rows; every row has one method, every method belongs to a row)

Error rows compare the whole message (`AssertEquals`). G := the tiny
model opened, T := `MergeTargets(G)`, S := `NewSet(T, 6)` unless said
otherwise; free S, then G.

`GgufMerge`, stage 1 (task T18b1): `MergeTargets`, `LoadAdapters`,
`MergeTensor`, `WriteMerged`; 001–014.

| Row | Input | Expected |
|---|---|---|
| 001 | `MergeTargets` of TinyPath | 7 targets, in order `blk.0.attn_qkv`, `blk.0.attn_gate`, `blk.0.ssm_out`, `blk.1.attn_q`, `blk.1.attn_k`, `blk.1.attn_v`, `blk.1.attn_output`; Tensor = `FindTensor(<name>.weight)`; in 64, 64, 64, 64, 64, 64, 128; out 128, 64, 64, 256, 64, 64, 64 |
| 002 | `MergeTargets` of NextNPath (block_count 3, one MTP block) | 7 targets; the first `blk.0.attn_qkv`, the last `blk.1.attn_output` |
| 003 | `PatchedCopy` with `blk.1.attn_k.weight` → `blk.1.attn_X.weight`; another with `blk.0.attn_gate.weight` → `blk.0.attn_XXXX.weight`, then `blk.0.attn_norm.weight` → `blk.0.attn_gate.weight` | `no tensor blk.1.attn_k.weight`; `blk.0.attn_gate.weight has 1 dimensions, expected 2` |
| 004 | `MergeTensor(8, <bytes of blk.0.attn_qkv.weight>, 64, 128, S.Find('blk.0.attn_qkv'))` (type 8 checked in the directory) | 8704 bytes; `StepErr` against `RefMerged` ≤ 0.6; not equal to the base's bytes |
| 005 | the same for `blk.0.attn_gate` (type 1, F16), 64, 64 | 4352 bytes; `StepErr` ≤ 0.6 |
| 006 | S := `NewSet(T, -1)`: `MergeTensor` of `blk.0.attn_qkv` (Q8_0); of `blk.0.attn_gate` (F16) | the Q8_0 bytes unchanged; the F16 one = `QuantizeQ8_0(DequantizeBytes(1, <bytes>, 4096))` |
| 007 | `MergeTensor(8, <bytes of blk.0.attn_qkv.weight>, 64, 128, S.Find('blk.0.attn_gate'))` | `blk.0.attn_gate: adapter 64 -> 64, tensor 64 -> 128` |
| 008 | `SaveCkpt(D, S, '')`; L := `LoadAdapters(T, D, ReadRunState(D))` | Count 7, Rank 4, Scale 2; pair I named `T[I].Name`; every A and B equal to S's (max difference 0) |
| 009 | the same with `lora.ot` deleted | ETorchNativeError `open file failed because of errno 2 on fopen: No such file or directory, file path: <D>lora.ot` |
| 010 | `WriteMerged(G, T, nil, D + 'id.gguf')` | 0; the file equals TinyPath byte for byte; no `id.gguf.part` |
| 011 | `WriteMerged(G, T, S, D + 'm.gguf')`, then the file opened | 7; no part file; tensor and key counts equal; the metadata bytes (KVStart..KVEnd) equal; per tensor the same name; targets type 8 with the bytes of `MergeTensor` of the base's (pair: the name without `.weight`); the others the base's type and bytes |
| 012 | a set with only `blk.0.attn_qkv` (64 → 128): `WriteMerged` | `no adapter blk.0.attn_gate`; neither `m.gguf` nor `m.gguf.part` exists |
| 013 | out `D + 'none/m.gguf'` (no such directory); out `D + 'dir'` (an existing directory); out a 3-byte file `D + 'm.gguf'`, then `D + 'n.gguf'` | EGgufError `TGgufWriter.BeginWrite: cannot create file <D>none/m.gguf.part`; EGgufMerge `cannot rename <D>dir.part to <D>dir`, no `dir.part`; 7, and `m.gguf` equals `n.gguf` |
| 014 | M0, MA (with the set saved by `SaveCkpt` attached through `AttachLora` and loaded), MM (the file of `WriteMerged` with S): Effect := `TopDiff(M0, MA)`, Fid := `TopDiff(MA, MM)` | Effect > 0.5; Fid ≤ 0.1 × Effect (measured 0.999 and 0.068) |

`GgufMerge`, stage 2 (task T18b2): `RunMergeCLI`; 101–106. D := `NewDir`.

| Row | Input | Expected |
|---|---|---|
| 101 | `SaveCkpt(D + 'c', S, '')`; `RunCli([TinyPath, D + 'c', D + 'm.gguf'])` | 0; out `merged 7 tensors (rank 4, scale 2) into <D>m.gguf`; err ''; the file equals `WriteMerged(G, T, S, ..)`'s |
| 102 | `[]`; `['a', 'b']`; `['a', 'b', 'c', 'd']`; `['--frob', 'a', 'b', 'c']`; `['a', 'b', 'c', '--base-sha256']`; `['--BASE-SHA256', 'x', 'a', 'b', 'c']` | exit 2 each: `expected BASE CKPT OUT, got 0 arguments`, `.. got 2 arguments`, `.. got 4 arguments`; `unknown option --frob`; `--base-sha256: value missing`; `unknown option --BASE-SHA256`; out '' |
| 103 | base `<TmpDir>none.gguf`; out = TinyPath; out `<TmpDir>d1/../tiny.gguf`; out `tiny.gguf` with TmpDir as the current directory | exit 1 each: `no model file <TmpDir>none.gguf`; `the output must not be the base model` (three times); out '' |
| 104 | an empty directory D as the checkpoint | exit 1; `not a checkpoint: <D>`; out ''; no output file |
| 105 | checkpoints `c` (BaseSha256 64 `a`) and `e` (''); copies of TinyPath named `sha256-<64 a>` and `sha256-<64 b>`: `--base-sha256 <64 b>` with c; `--base-sha256 <64 A>` (upper case) with c; base `sha256-<64 b>` with c; then `--base-sha256 <64 a>` with c; base `sha256-<64 a>` with c; TinyPath with c; `--base-sha256 <64 b>` with e | exit 1 three times, `the base model differs from the checkpoint's: <D>c`, no output; then exit 0 four times, err '' |
| 106 | `SaveCkpt(D + 'c', ..)` with `lora.ot` deleted | exit 1; err `open file failed because of errno 2 on fopen: No such file or directory, file path: <D>c/lora.ot`; out ''; no output file |

Build and run (the GPUs hidden as in the gate):
`fpc -Mobjfpc -Sh -vew -l- -Fu{proj}/src -Fu{proj}/../../src -Fu{proj}/tests/ut -FU{proj}/work/bin -Fl{proj}/../../build/libtch-rocm -k-rpath={proj}/../../build/libtch-rocm {proj}/tests/ut/GgufMerge_test.pas -o{proj}/work/bin/GgufMerge_test`
then `HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 {proj}/work/bin/GgufMerge_test --all --format=plain`.
Warnings from other units and libtorch's GPU warnings on stderr are
expected; so is the note "Local variable NG is assigned but never used"
(the `NoGradGuard` idiom). Until a stage is written, a test program
without test methods reports `No valid tests found`; that is expected.

## ITa (prepared by the harness owner)

`QLORA_MODEL=<rehearsal GGUF> {proj}/tests/ita/T18-run.sh` builds
`src/qlora_train.pas`, `src/qlora_merge.pas` and
`tests/ita/T18/merge_check.pas` against `src/` and runs on the CPU from a
fresh directory outside the project: `qlora_train` on
`tests/ita/T17/train.jsonl` (8 short records; 4 epochs of one update, lr
1e-3, warm-up 1), `qlora_merge` of its `ckpt-00000004`, then
`merge_check` (the H43 prompts of `work/oracle/T11`; the H47 reference
`work/oracle/T18/expected.jsonl`: the bundled `llama-server` on an
all-F32 copy of the prototype's merged file, `-ctk f32 -ctv f32 -fa off`,
taken once). Writes `{proj}/work/ita/T18/summary.tsv` (case, OK/NG,
value, limit, note; then the exit codes and the count of stderr lines
other than the GPU warnings). OK when value ≤ limit. About 6 minutes.

| Case | Compared | Limit |
|---|---|---|
| ITa-T18-001 | `WriteMerged` without adapters against the base (0 when byte-identical) | 0 |
| ITa-T18-002 | every merged tensor dequantised against the fp32 merge, max difference / the block's Q8_0 step (+ 99 per tensor missing or not Q8_0) | 1 |
| ITa-T18-003 | worst mean \|Δ logprob\| over the top 10 of the 27 prompts: the merged file against the adapter, divided by the B = 0 merge against the base | 1.5 |
| ITa-T18-004 | mean loss over the 8 records: 1 − (base − merged) / (base − adapter) | 0.1 |
| ITa-T18-005 | H47: max \|Pascal on the merged file − the reference\| over the reference's top 10 (+ 99 per top-1 differing) | 0.02 |
| ITa-T18-006 | the merged file against the prototype's (sha256 in `work/oracle/T18/merged.sha256`): 0 when byte-identical | 0 |

Measured with the prototype: 0, 0.643, 0.608 (0.148 / 0.243), 3.75e-4,
1.74e-3, 0 in 5 min 48 s. With the skeleton every case is NG. The CPU
runs are deterministic: the code of the facts gives these values exactly;
006 above 0 means the code differs from the facts.

## Stages and case IDs

- Stage 1 (task T18b1): `MergeTargets` (with `AddTarget`),
  `LoadAdapters`, `MergeTensor`, `WriteMerged`; rows UT-GgufMerge-001 to
  014.
- Stage 2 (task T18b2): `RunMergeCLI`; rows UT-GgufMerge-101 to 106.
- ITa: ITa-T18-001 to 006.
