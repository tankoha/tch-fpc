# T13 facts (checked by the harness owner)

T13 is the unit `Recompute`: the block-wise backward by recomputation
(DESIGN 8.14, "Recompute"). The forward runs every stage without a graph
and keeps each stage's input; the backward goes from the last stage to the
first, runs the stage again with a graph from its saved input, and
propagates the gradient. Parameter gradients accumulate in the stages' own
leaves. It is independent of Qwen: a stage is anything with a device and a
forward. T17 uses it for the 64 trunk blocks on the two GPUs.

Measured by the harness owner with a prototype written exactly as below,
on the CPU (GPUs hidden): on 8 and 64 stages in fp64 the parameter and
input gradients are **bit-for-bit equal** to one ordinary backward through
the whole stack (difference 0; the design's limit is 1e-10); every row of
"Unit tests" gives the value in its row; the messages are the
prototype's. The harness owner also ran it on the two GPUs (stages split
over cuda:0 and cuda:1, fp32, saved inputs on the host): within 3.8e-5 of
fp64 on the CPU. These facts are final; do not investigate further or
write probes.

## Unit `Recompute` (interface, final; in the skeleton)

```pascal
type
  ERecompute = class(Exception);

  TStage = class abstract
  public
    function Device: TDevice; virtual; abstract;
    function Forward(const X: TTensor): TTensor; virtual; abstract;
  end;
  TStageArray = array of TStage;

  TSavedInputs = class
  private
    FOnHost: Boolean;
    FItems: array of TTensor;
  public
    constructor Create(AOnHost: Boolean);
    procedure Clear;
    procedure Add(const X: TTensor);
    function Count: Integer;
    function Item(I: Integer): TTensor;
    property OnHost: Boolean read FOnHost;
  end;

function RecomputeForward(const Stages: TStageArray; const X0: TTensor;
  Saved: TSavedInputs): TTensor;
function RecomputeBackward(const Stages: TStageArray; Saved: TSavedInputs;
  const GradOut: TTensor): TTensor;
```

`RecomputeBackward` returns dL/dX0 (on the first stage's device); the
design's procedure is a function so that the input gradient can be
checked. The skeleton's `ShapeStr(T)` (private, implementation section)
gives `'[2, 3]'`.

## Implementation (write it exactly so)

`TSavedInputs`:
- `Create`: `inherited Create; FOnHost := AOnHost;`
- `Clear`: `SetLength(FItems, 0);`
- `Add`: append `X.detach.&to(TDevice.Cpu)` when `FOnHost`, else
  `X.detach` (`SetLength(FItems, N + 1)`, then `FItems[N] := ...`).
- `Count`: `Length(FItems)`.
- `Item(I)`: the check of "Checks and messages", then `FItems[I]`.

`RecomputeForward` (after its checks):

```pascal
  Saved.Clear;
  G := NoGradGuard;          // G: IInterface, a local variable (a bare
  X := X0;                   // "NoGradGuard;" is released at once, H11)
  for I := 0 to High(Stages) do
  begin
    X := X.&to(Stages[I].Device);
    Saved.Add(X);
    X := Stages[I].Forward(X);
  end;
  Result := X;
```

`RecomputeBackward` (after its checks):

```pascal
  G := GradOut;
  for I := High(Stages) downto 0 do
  begin
    X := Saved.Item(I).&to(Stages[I].Device).detach.requires_grad_(True);
    Y := Stages[I].Forward(X);
    G := G.&to(Stages[I].Device);
    if ShapeStr(Y) <> ShapeStr(G) then
      raise ERecompute.CreateFmt('stage %d: output %s, gradient %s',
        [I, ShapeStr(Y), ShapeStr(G)]);
    (Y * G).sum.Backward;
    G := X.grad;
  end;
  Result := G;
```

`X`, `Y`, `G` are `TTensor` locals (reference counted; the graph of a
stage is freed when they are overwritten).

## Checks and messages (ERecompute; in this order)

- `TSavedInputs.Item(I)` with `I < 0` or `I >= Count`:
  `CreateFmt('saved input %d out of range (count %d)', [I, Count])`,
  e.g. `saved input 2 out of range (count 2)`.
- `RecomputeForward`: 1. no stages: `no stages`; 2. `not X0.Defined`:
  `X0 is undefined`.
- `RecomputeBackward`: 1. `Saved.Count <> Length(Stages)`:
  `CreateFmt('saved inputs: %d, stages: %d', ...)`, e.g.
  `saved inputs: 1, stages: 2`; 2. `not GradOut.Defined`:
  `GradOut is undefined`; 3. in the loop (above): e.g.
  `stage 1: output [2], gradient [3]`.

## UT fixture (`tests/ut/RecomputeFixture.pas`, harness owner's)

Used by the UT and the ITa; nobody changes it.
- `TScaleStage.Create(W)`: y = x · w, `W` a kFloat leaf `[1]` that
  requires grad, on the CPU. Hand values: dL/dw = Σ x·g, dL/dx = w·g.
- `TAffineStage`: y = tanh(x W + b) with leaves `W` [D, D], `B` [D].
  Both stage classes count their forwards in `Calls` and keep
  `LastRequiresGrad` (whether the last forward's output requires grad,
  i.e. built a graph).
- `MakeStages(Seed, N, D, Kind, Devices)`: N affine stages;
  `CloneStages(S, Kind, Devices)`: the same parameter values in new leaves;
  `FreeStages(S)`.
- `WholeGraphBackward(S, X0, GradOut)`: the oracle, one ordinary forward
  with graph and one backward of Σ y·GradOut; returns dL/dX0.
- `MaxAbsDiff(A, B)`, `MaxParamGradDiff(A, B)` (over all stages' `W.grad`
  and `B.grad`), `ShapeText(T)` (`'[3, 8]'`), `ValuesText(T)` (`FloatToStr`
  of each value, single spaces: `'6 12'`).

The test skeleton `tests/ut/Recompute_test.pas` has `Hand(S0, S1)`: two
scale stages with w = 2 and w = 3 (S0 first), returned as a `TStageArray`.
Each test frees what it creates (`TSavedInputs.Free`, `FreeStages`) in
`try ... finally`.

## Unit tests (rows; every row has one method, every method belongs to a row)

"exact" means the exact strings or `MaxAbsDiff = 0`. Error rows catch
`ERecompute` and compare the whole message (`AssertEquals`); a row fails
if no exception comes (`Fail('no ERecompute')`).

Stage 1, `TSavedInputs` and `RecomputeForward` (001–013):

| Row | Input | Expected |
|---|---|---|
| 001 | `TSavedInputs.Create(False)` | `OnHost` False, `Count` 0 |
| 002 | `TSavedInputs.Create(True)` | `OnHost` True, `Count` 0 |
| 003 | `Create(False)`; `Add(X)` with X = `FromArray([1.0, 2]).requires_grad_(True)`; `Add(FromArray([3.0]))` | `Count` 2; `ValuesText(Item(0))` `'1 2'`, `Item(0).RequiresGrad` False, `Item(0).SameHandle(X)` False; `ValuesText(Item(1))` `'3'` |
| 004 | `Create(True)`; `Add(FromArray([1.0, 2]))` | `ValuesText(Item(0))` `'1 2'`; `Item(0).Device.ToString` `'cpu'` |
| 005 | `Create(True)`; two `Add`; `Clear` | `Count` 0 |
| 006 | two `Add`; `Item(2)` | `saved input 2 out of range (count 2)` |
| 007 | two `Add`; `Item(-1)` | `saved input -1 out of range (count 2)` |
| 008 | `Hand`; `RecomputeForward(St, FromArray([1.0, 2]), Saved)` (`Saved` = `Create(False)`) | result `'6 12'`, `RequiresGrad` False; `Count` 2, items `'1 2'` and `'2 4'`; `Calls` 1 and 1; `LastRequiresGrad` False and False |
| 009 | `Saved` with three `Add` of `FromArray([1.0])`; forward of 008 | `Count` 2 (the forward clears first) |
| 010 | `MakeStages(51, 4, 8, kDouble, [TDevice.Cpu])`; `ManualSeed(52)`; X = `TTensor.randn([3, 8], kDouble)`; forward | every stage `Calls` 1 and `LastRequiresGrad` False |
| 011 | as 010, and `CloneStages(A, kDouble, [TDevice.Cpu])` made before the forward; the plain chain `B[0..3].Forward` from X | result shape `'[3, 8]'`; exact: `MaxAbsDiff(result, chain)` = 0 |
| 012 | `RecomputeForward(nil, FromArray([1.0]), Saved)` | `no stages` |
| 013 | `Hand`; `RecomputeForward(St, Default(TTensor), Saved)` | `X0 is undefined` |

Stage 2, `RecomputeBackward` (101–108):

| Row | Input | Expected |
|---|---|---|
| 101 | forward of 008, then `RecomputeBackward(St, Saved, FromArray([1.0, 1]))` | result `'6 6'`; `S0.W.grad` `'9'`, `S1.W.grad` `'6'`; `Calls` 2 and 2; `LastRequiresGrad` True and True |
| 102 | `MakeStages(53, 8, 16, kDouble, [TDevice.Cpu])`, B := `CloneStages(A, kDouble, [TDevice.Cpu])`; `ManualSeed(54)`; X, GOut = `randn([4, 16], kDouble)` (in that order); forward and backward on A, `WholeGraphBackward(B, X, GOut)` | exact: input gradient difference 0 and `MaxParamGradDiff(A, B)` 0 |
| 103 | as 010, then `RecomputeBackward(A, Saved, TTensor.ones([3, 8], kDouble))` | every stage `Calls` 2 and `LastRequiresGrad` True |
| 104 | forward of 008, two backwards with `FromArray([1.0, 1])` | second result `'6 6'`; `S0.W.grad` `'18'`, `S1.W.grad` `'12'` (accumulated) |
| 105 | inputs of 102 on A and on a second clone C; A with `Create(False)`, C with `Create(True)` | exact: input gradients equal, `MaxParamGradDiff(A, C)` 0 |
| 106 | `Hand`; `Saved` with one `Add`; backward with `FromArray([1.0, 1])` | `saved inputs: 1, stages: 2` |
| 107 | forward of 008; backward with `Default(TTensor)` | `GradOut is undefined` |
| 108 | forward of 008; backward with `FromArray([1.0, 1, 1])` | `stage 1: output [2], gradient [3]` |

Build and run (the GPUs hidden as in the gate):
`fpc -Mobjfpc -Sh -vew -l- -Fu{proj}/src -Fu{proj}/../../src -Fu{proj}/tests/ut -FU{proj}/work/bin -Fl{proj}/../../build/libtch-rocm -k-rpath={proj}/../../build/libtch-rocm {proj}/tests/ut/Recompute_test.pas -o{proj}/work/bin/Recompute_test`
then `HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 {proj}/work/bin/Recompute_test --all --format=plain`.
libtorch prints warnings about the hidden GPUs on stderr; they are expected.

## ITa (prepared by the harness owner)

`{proj}/tests/ita/T13-run.sh` (no variable needed) builds
`tests/ita/T13/recompute_check.pas` against `src/`, runs it on the CPU from
a fresh directory outside the project and writes
`{proj}/work/ita/T13/summary.tsv` (case, OK/NG, value, limit, note; then
the driver's exit code and the count of stderr lines other than the GPU
warnings). OK when value ≤ limit. A few seconds. The oracle is the
fixture's `WholeGraphBackward`, outside the unit.

| Case | Compared | Limit |
|---|---|---|
| ITa-T13-001 | fp64, 8 stages, width 16, 4 rows, inputs on the device: max of parameter and input gradient differences | 1e-10 |
| ITa-T13-002 | fp64, 64 stages, width 64, 32 rows, inputs on the host | 1e-10 |
| ITa-T13-003 | fp32 against fp32, 64 stages, width 256, 128 rows | 1e-6 |
| ITa-T13-004 | count of violations: forward without graph, one call per stage, 16 saved inputs without grad; backward one more call with graph | 0 |
| ITa-T13-005 | two backwards: `W.grad` equals twice the reference | 1e-10 |

Measured with the prototype: 0 in every case. With the skeleton every case
is NG; mutants are caught (the gradient not chained: 001–003, 005;
forward without `NoGradGuard`: 004; stages in the wrong order: 001–003,
005). Cases 101–103 (two GPUs) are the harness owner's (`T13-run.sh gpu`
with the GPUs visible); agents do not run them.

## Stages and case IDs

- Stage 1 (task T13b1): `TSavedInputs`, `RecomputeForward`; rows
  UT-Recompute-001 to 013.
- Stage 2 (task T13b2): `RecomputeBackward`; rows UT-Recompute-101 to 108.
- ITa: ITa-T13-001 to 005.
