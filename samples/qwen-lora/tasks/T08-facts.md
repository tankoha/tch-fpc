# T08 facts (checked by the harness owner)

T08 is the unit `DeltaRule`: the gated delta rule of the Gated DeltaNet
(linear) blocks, **token by token** (DESIGN 8.10, "Linear block", step 7).
It is the reference form: T09 adds the chunked form and is checked against
this one; T10 calls it from the linear block. The query is used as given
(the block scales it by 1/sqrt(Dk) before, DESIGN step 6); key heads are
already tiled to the value heads (step 5). No autograd requirement.

Measured by the harness owner with a prototype written exactly as below, on
the CPU (GPUs hidden): the hand cases give exactly the values below; at the
27B model's size (48 value heads, Dk = Dv = 128, T = 200) the unit agrees
with a scalar reference in Double to 4.8e-7 (outputs up to 1.1) and takes
about 50 ms; every row of "Unit tests" gives the value in its row; the
messages are the prototype's. These facts are final; do not investigate
further or write probes.

## The rule

Per head h, a state M [Dv, Dk] (value × key), zero at the sequence start.
Per token t, with g_t ≤ 0 a log decay and 0 ≤ β_t ≤ 1:

```
M   := exp(g_t) · M
d   := β_t · (v_t − M k_t)        (d has Dv values)
M   := M + d k_tᵀ                  (outer product, Dv × Dk)
o_t := M q_t                       (Dv values)
```

Hand case (H = 1, Dk = Dv = 2, two tokens; all values exact in fp32):

| | q | k | v | β | g | M after (row-major) | o |
|---|---|---|---|---|---|---|---|
| token 0, from M = 0 | (1, 1) | (1, 0) | (2, 4) | 0.5 | 0 | 1 0 2 0 | (1, 2) |
| token 1 | (1, 2) | (1, 1) | (3, 3) | 0.5 | 0 | 2 1 2.5 0.5 | (4, 3.5) |
| token 1 with g = ln(1/2) | (1, 2) | (1, 1) | (3, 3) | 0.5 | ln 0.5 | 1.75 1.25 2 1 | (4.25, 4) |

(token 1: M k = (1, 2), d = 0.5 · ((3, 3) − (1, 2)) = (1, 0.5). With the
decay, M = (0.5 0; 1 0) first, M k = (0.5, 1), d = (1.25, 1).
`exp` of the fp32 value of ln(1/2) is exactly 0.5 in libtorch, measured.)

## Unit `DeltaRule` (interface, final; in the skeleton)

```pascal
unit DeltaRule;
{$mode objfpc}{$H+}
interface
uses SysUtils, Tch;
type
  EDeltaRule = class(Exception);
function DeltaStep(const M, Q, K, V, Beta, G: TTensor; out O: TTensor): TTensor;
function DeltaRuleTokens(const Q, K, V, Beta, G: TTensor; out State: TTensor): TTensor;
```

- `DeltaStep`: one token, all heads. M [H, Dv, Dk]; Q, K [H, Dk];
  V [H, Dv]; Beta, G [H]. Returns the new state [H, Dv, Dk]; O receives
  [H, Dv]. M is not changed (a new tensor is returned).
- `DeltaRuleTokens`: a whole sequence from a zero state. Q, K [H, T, Dk];
  V [H, T, Dv]; Beta, G [H, T]; T ≥ 1. Returns O [H, T, Dv]; State
  receives the final state [H, Dv, Dk].
- All tensors kFloat (fp32); results on the inputs' device.

## Implementation (write it exactly so; the bitwise rows depend on it)

`DeltaStep`, after the checks below (H = M.Shape[0], Dv = M.Shape[1],
Dk = M.Shape[2]):

```pascal
Md := M * G.exp.view([H, 1, 1]);
Pred := Md.matmul(K.unsqueeze(2)).squeeze_dim(2);          { [H, Dv] }
D := (V - Pred) * Beta.unsqueeze(1);                        { [H, Dv] }
Result := Md + D.unsqueeze(2).matmul(K.unsqueeze(1));       { [H, Dv, Dk] }
O := Result.matmul(Q.unsqueeze(2)).squeeze_dim(2);          { [H, Dv] }
```

`DeltaRuleTokens`, after its checks (H = Q.Shape[0], T = Q.Shape[1],
Dk = Q.Shape[2], Dv = V.Shape[2]): `M := TTensor.zeros([H, Dv, Dk], kFloat,
Q.Device)`; for I := 0 to T − 1, `M := DeltaStep(M, Q.select(1, I),
K.select(1, I), V.select(1, I), Beta.select(1, I), G.select(1, I), Ot)` and
keep `Ot` in an array; `Result := TTensor.stack(Outs, 1)`; `State := M`.

## Checks and messages (EDeltaRule; in this order)

Shapes are written `[2, 3, 4]` (`IntToStr` of each size, `', '` between,
in brackets); `<H>` etc. below are the numbers.

`DeltaStep`: first the kinds, in the order M, Q, K, V, Beta, G:
`<name> must be kFloat (fp32)` (e.g. `M must be kFloat (fp32)`). Then:
1. M not 3-D: `M must be [H, Dv, Dk], got <shape of M>`
2. Q not [H, Dk]: `Q must be [<H>, <Dk>], got <shape of Q>`
3. K not [H, Dk]: `K must be [<H>, <Dk>], got <shape of K>`
4. V not [H, Dv]: `V must be [<H>, <Dv>], got <shape of V>`
5. Beta not [H]: `Beta must be [<H>], got <shape of Beta>`
6. G not [H]: `G must be [<H>], got <shape of G>`

`DeltaRuleTokens`: first the kinds, in the order Q, K, V, Beta, G (same
message). Then:
1. Q not 3-D: `Q must be [H, T, Dk], got <shape of Q>`
2. T = 0: `no tokens (T = 0)`
3. K not the shape of Q: `K must be the shape of Q <shape of Q>, got <shape of K>`
4. V not [H, T, any]: `V must be [<H>, <T>, Dv], got <shape of V>`
5. Beta not [H, T]: `Beta must be [<H>, <T>], got <shape of Beta>`
6. G not [H, T]: `G must be [<H>, <T>], got <shape of G>`

## UT fixture (`tests/ut/DeltaRuleFixture.pas`, harness owner's)

Used by the UT and the ITa; nobody changes it.
- `RandomInputs(Seed, H, T, Dk, Dv, GScale, Q, K, V, Beta, G)`: seeded
  random inputs (Q scaled by 1/sqrt(Dk), K rows of length 1, Beta in
  [0, 1), G in (−GScale, 0]).
- `RefTokens(Q, K, V, Beta, G, O, S)`: the rule in Double, independent of
  the unit (the oracle of the random rows).
- `MaxAbsDiff(A, B)`, `ShapeText(T)` (`'[1, 2, 2]'`), `ValuesText(T)`
  (`FloatToStr` of each value, single spaces: `'1 2 4 3.5'`), `Values(T)`.

The test skeleton `tests/ut/DeltaRule_test.pas` has `Make(Values, Shape)`
(a kFloat tensor) and `CheckTensor(What, Shape, Values, T)` (fails unless
`ShapeText` and `ValuesText` are those strings).

## Unit tests (rows; every row has one method, every method belongs to a row)

`Make([..], [..])` is the skeleton's helper. "zeros" is `TTensor.zeros`.
A tolerance row passes when `MaxAbsDiff ≤` the limit; "exact" means
`MaxAbsDiff = 0` or the exact strings. Error rows catch `EDeltaRule` and
compare the whole message (`AssertEquals`); a row fails if no exception
is raised. In error rows every other argument is valid: for `DeltaStep`
M zeros([2, 5, 4]), Q and K zeros([2, 4]), V zeros([2, 5]), Beta and G
zeros([2]); for `DeltaRuleTokens` Q and K zeros([2, 3, 4]), V zeros([2, 3,
5]), Beta and G zeros([2, 3]).

**Stage 1, `DeltaStep`, rows UT-DeltaRule-001..**

| Row | Input | Expected |
|---|---|---|
| 001 | M zeros([1, 2, 2]), Q Make([1, 1], [1, 2]), K Make([1, 0], [1, 2]), V Make([2, 4], [1, 2]), Beta Make([0.5], [1]), G Make([0], [1]) | result `[1, 2, 2]` `1 0 2 0`; O `[1, 2]` `1 2` |
| 002 | M Make([1, 0, 2, 0], [1, 2, 2]), Q (1, 2), K (1, 1), V (3, 3), Beta 0.5, G 0 (shapes as 001) | result `2 1 2.5 0.5`; O `4 3.5` |
| 003 | as 002 with G Make([Ln(0.5)], [1]) | result `1.75 1.25 2 1`; O `4.25 4` |
| 004 | two heads: M Make([0, 0, 0, 0, 1, 0, 2, 0], [2, 2, 2]), Q Make([1, 1, 1, 2], [2, 2]), K Make([1, 0, 1, 1], [2, 2]), V Make([2, 4, 3, 3], [2, 2]), Beta Make([0.5, 0.5], [2]), G Make([0, 0], [2]) | result `[2, 2, 2]` `1 0 2 0 2 1 2.5 0.5`; O `[2, 2]` `1 2 4 3.5` |
| 005 | β = 0: `RandomInputs(23, 3, 1, 4, 5, 1.0, …)`, then M := `TTensor.randn([3, 5, 4])`; step with token 0 (`select(1, 0)`) of Q, K, V, G and Beta zeros([3]) | exact: result = `M * G.select(1, 0).exp.view([3, 1, 1])`; O = `result.matmul(Q.select(1, 0).unsqueeze(2)).squeeze_dim(2)` |
| 006 | as 005 with G zeros([3]) too | exact: result = M |
| 007 | `RandomInputs(21, 3, 1, 4, 5, 1.0, …)`; step from zeros([3, 5, 4]) with token 0; `RefTokens` of the same inputs | O vs `RO.select(1, 0)` and result vs RS ≤ 1e-6 (measured 1.5e-8, 3.0e-8) |
| 008 | `RandomInputs(22, 3, 2, 4, 5, 1.0, …)`; two steps from zeros([3, 5, 4]) (tokens 0, 1); `RefTokens` | second O vs `RO.select(1, 1)` and result vs RS ≤ 1e-6 (measured 6.0e-8, 1.2e-7) |
| 009 | M zeros([2, 5, 4], kDouble) | `M must be kFloat (fp32)` |
| 010 | G zeros([2], kBFloat16) | `G must be kFloat (fp32)` |
| 011 | M zeros([5, 4]) | `M must be [H, Dv, Dk], got [5, 4]` |
| 012 | Q zeros([2, 3]) | `Q must be [2, 4], got [2, 3]` |
| 013 | K zeros([1, 4]) | `K must be [2, 4], got [1, 4]` |
| 014 | V zeros([2, 4]) | `V must be [2, 5], got [2, 4]` |
| 015 | Beta zeros([3]) | `Beta must be [2], got [3]` |
| 016 | G zeros([2, 1]) | `G must be [2], got [2, 1]` |

**Stage 2, `DeltaRuleTokens`, rows UT-DeltaRule-101..**

| Row | Input | Expected |
|---|---|---|
| 101 | hand case: Q Make([1, 1, 1, 2], [1, 2, 2]), K Make([1, 0, 1, 1], [1, 2, 2]), V Make([2, 4, 3, 3], [1, 2, 2]), Beta Make([0.5, 0.5], [1, 2]), G Make([0, 0], [1, 2]) | O `[1, 2, 2]` `1 2 4 3.5`; State `[1, 2, 2]` `2 1 2.5 0.5` |
| 102 | as 101 with G Make([0, Ln(0.5)], [1, 2]) | O `1 2 4.25 4`; State `1.75 1.25 2 1` |
| 103 | β = 0: `RandomInputs(7, 3, 5, 4, 6, 1.0, …)`, Beta replaced by zeros([3, 5]) | O `[3, 5, 6]`, State `[3, 6, 4]`; `O.abs.max` and `State.abs.max` are 0 (the acceptance: the state stays zero) |
| 104 | g = −inf: `RandomInputs(8, 3, 5, 4, 6, 1.0, …)`, G replaced by `TTensor.zeros([3, 5]) + Double(NegInfinity)` (unit `Math`) | O vs `(V * Beta.unsqueeze(2)) * (K * Q).sum_dim_intlist(IntList([2]), True, kNone)` ≤ 1e-6 (measured 6.0e-8): each token sees only itself |
| 105 | T = 1 equals one step: `RandomInputs(9, 3, 5, 4, 6, 1.0, …)`; `DeltaRuleTokens` of `narrow(1, 0, 1)` of all five; `DeltaStep` from zeros([3, 6, 4]) with `select(1, 0)` | exact: O `select(1, 0)` = the step's O; State = the step's result |
| 106 | heads are independent: inputs of 105, whole; and `narrow(0, 1, 1)` of all five | exact: `O.narrow(0, 1, 1)` and `State.narrow(0, 1, 1)` equal the single head's O and State |
| 107 | `RandomInputs(11, 3, 20, 4, 5, 0.1, …)`; `RefTokens` | O vs RO and State vs RS ≤ 1e-5 (measured 4.8e-7, 1.2e-7) |
| 108 | `RandomInputs(12, 2, 65, 16, 16, 5.0, …)`; `RefTokens` | ≤ 1e-5 (measured 2.4e-7, 3.0e-8) |
| 109 | Q zeros([2, 3, 4], kDouble) | `Q must be kFloat (fp32)` |
| 110 | G zeros([2, 3], kBFloat16) | `G must be kFloat (fp32)` |
| 111 | Q zeros([2, 3]) | `Q must be [H, T, Dk], got [2, 3]` |
| 112 | Q, K zeros([2, 0, 4]), V zeros([2, 0, 5]), Beta, G zeros([2, 0]) | `no tokens (T = 0)` |
| 113 | K zeros([2, 3, 3]) | `K must be the shape of Q [2, 3, 4], got [2, 3, 3]` |
| 114 | V zeros([1, 3, 5]) | `V must be [2, 3, Dv], got [1, 3, 5]` |
| 115 | V zeros([2, 2, 5]) | `V must be [2, 3, Dv], got [2, 2, 5]` |
| 116 | Beta zeros([2, 4]) | `Beta must be [2, 3], got [2, 4]` |
| 117 | G zeros([2]) | `G must be [2, 3], got [2]` |

Build and run (the GPUs hidden as in the gate):
`fpc -Mobjfpc -Sh -vew -l- -Fu{proj}/src -Fu{proj}/tests/ut -FU{proj}/work/bin {proj}/tests/ut/DeltaRule_test.pas -o{proj}/work/bin/DeltaRule_test`
then `HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 {proj}/work/bin/DeltaRule_test --all --format=plain`.
libtorch prints warnings about the hidden GPUs on stderr; they are expected.

## ITa (prepared by the harness owner)

`{proj}/tests/ita/T08-run.sh` (no variable needed) builds
`tests/ita/T08/deltarule_check.pas` against `src/`, runs it on the CPU
from a fresh directory outside the project and writes
`{proj}/work/ita/T08/summary.tsv` (case, OK/NG, value, limit, note; then
the driver's exit code and the count of stderr lines other than the GPU
warnings). OK when value ≤ limit. About 10 s. The oracle is the fixture's
`RefTokens` and the hand values, both outside the unit.

| Case | Compared | Limit |
|---|---|---|
| ITa-T08-001 | hand case, two tokens (O and State) | 0 |
| ITa-T08-002 | hand case with g = ln(1/2) on token 1 | 0 |
| ITa-T08-003 | β = 0 at H 48, T 200, 128 × 128: max \|O\| and \|State\| | 0 |
| ITa-T08-004 | `RefTokens`, seed 13, H 48, T 200, g in (−0.1, 0] | 1e-5 |
| ITa-T08-005 | `RefTokens`, seed 14, H 16, T 200, g in (−5, 0] | 1e-5 |
| ITa-T08-006 | `RefTokens`, H 48, T = 1, 63, 64, 65, 200 (T09's lengths) | 1e-5 |
| ITa-T08-007 | g = −inf, H 48, T 20: o_t = β_t v_t (k_t · q_t) | 1e-6 |

Measured with the prototype: 0, 0, 0, 4.8e-7, 2.4e-7, 4.2e-7, 2.4e-7;
with the skeleton every case is NG; with the decay left out, 002 and 004
to 007 are NG.

## Stages and case IDs

- Stage 1 (task T08b1): `DeltaStep`; rows UT-DeltaRule-001 to 016.
- Stage 2 (task T08b2): `DeltaRuleTokens`; rows UT-DeltaRule-101 to 117.
