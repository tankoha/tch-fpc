# T09 facts (checked by the harness owner)

T09 adds `DeltaRuleChunked` to the unit `DeltaRule` (T08): the gated delta
rule of `DeltaRuleTokens` in the **chunked form** (DESIGN 8.10, "Linear
block", step 7: chunks of 64 tokens, the sequence padded to a multiple of
64, a unit-lower-triangular solve inside each chunk, one state update per
chunk). T10's linear block uses it for training and scoring; it must give
what the token form gives. It is the form of the gate measurement (design
session 9, `build/gate-2026-11-16/h42.pas`, `ChunkGDR`), adapted to T08's
conventions: state `[H, Dv, Dk]`, q used as given (no scaling inside), the
final state returned.

Measured by the harness owner with a prototype written exactly as below,
on the CPU (GPUs hidden): against `DeltaRuleTokens` at the 27B model's
sizes (48 heads, 128 × 128) for T = 1, 63, 64, 65, 200 and 2048 the
outputs and the final state differ by at most 7.2e-7 (the design's limit
is 1e-5); with strong decay (g down to −5 per token) 4.4e-6; the
gradients of all five inputs 2.1e-6; the two hand cases of T08 come out
exactly. T = 2048: 100 ms against 379 ms for the token form on the CPU,
13 ms on the RX 9070. **G must be finite:** g = −inf (allowed in the token
form) gives NaN here (−inf − (−inf) in the decay matrix); the real gates
are finite. Every row of "Unit tests" gives the value in its row; the
messages are the prototype's. These facts are final; do not investigate
further or write probes.

## The chunked form

For one head, chunk c of C tokens (positions i, j inside the chunk), with
`Gc` the cumulative sum of g inside the chunk:
- Decay[i, j] = exp(Gc[i] − Gc[j]) for i ≥ j, else 0;
- L = strictly lower part of (β k) kᵀ ⊙ Decay; T = (I + L)⁻¹ (triangular
  solve);
- U = T (β v); W = T (β k ⊙ exp(Gc));
- per chunk, with the state S [Dk, Dv] (the transpose of T08's M):
  v' = U − W S; o = (q ⊙ exp(Gc)) S + (strictly-and-diagonal lower part
  of q kᵀ ⊙ Decay) v'; S := exp(Gc[C−1]) S + (k ⊙ exp(Gc[C−1] − Gc))ᵀ v'.
Padding tokens have q = k = v = 0, β = 0 and g = 0: they leave the state
unchanged, and their outputs are cut off.

## Unit `DeltaRule`, the new function (interface, final; in the skeleton)

```pascal
function DeltaRuleChunked(const Q, K, V, Beta, G: TTensor; out State: TTensor;
  ChunkSize: Integer = 64): TTensor;
```

Inputs and outputs as `DeltaRuleTokens` (Q, K [H, T, Dk]; V [H, T, Dv];
Beta, G [H, T]; returns O [H, T, Dv]; State [H, Dv, Dk]). Do not change
`DeltaStep` or `DeltaRuleTokens`; the unit's private `ShapeStr` is used
for the messages.

## Implementation (write it exactly so)

Variables: `H, T, Dk, Dv, C, N, P, I: Int64; Qp, Kp, Vp, Bp, Gp, KB, VB,
Gc, Decay, L, Eye, Tm, U, W, S: TTensor; Qi, Ki, Ui, Wi, Gi, Di, Ai, VNew,
Last: TTensor; Outs: array of TTensor`. First `State := Default(TTensor);
Result := Default(TTensor);`, then the checks (below), then:

```pascal
  C := ChunkSize;
  N := (T + C - 1) div C;
  P := N * C - T;
  Qp := Q; Kp := K; Vp := V; Bp := Beta; Gp := G;
  if P > 0 then
  begin
    Qp := TTensor.cat([Q, TTensor.zeros([H, P, Dk], kFloat, Q.Device)], 1);
    Kp := TTensor.cat([K, TTensor.zeros([H, P, Dk], kFloat, Q.Device)], 1);
    Vp := TTensor.cat([V, TTensor.zeros([H, P, Dv], kFloat, Q.Device)], 1);
    Bp := TTensor.cat([Beta, TTensor.zeros([H, P], kFloat, Q.Device)], 1);
    Gp := TTensor.cat([G, TTensor.zeros([H, P], kFloat, Q.Device)], 1);
  end;
  KB := (Kp * Bp.unsqueeze(2)).reshape([H, N, C, Dk]);
  VB := (Vp * Bp.unsqueeze(2)).reshape([H, N, C, Dv]);
  Qp := Qp.reshape([H, N, C, Dk]);
  Kp := Kp.reshape([H, N, C, Dk]);
  Gc := Gp.reshape([H, N, C]).cumsum(2);
  Decay := (Gc.unsqueeze(3) - Gc.unsqueeze(2)).tril.exp.tril;
  L := (KB.matmul(Kp.transpose(2, 3)) * Decay).tril(-1);
  Eye := TTensor.eye(C, kFloat, Q.Device);
  Tm := (L + Eye).linalg_solve_triangular(Eye.expand([H, N, C, C]), False);
  U := Tm.matmul(VB);
  W := Tm.matmul(KB * Gc.exp.unsqueeze(3));
  S := TTensor.zeros([H, Dk, Dv], kFloat, Q.Device);
  SetLength(Outs, N);
  for I := 0 to N - 1 do
  begin
    Qi := Qp.select(1, I);
    Ki := Kp.select(1, I);
    Ui := U.select(1, I);
    Wi := W.select(1, I);
    Gi := Gc.select(1, I);
    Di := Decay.select(1, I);
    Ai := (Qi.matmul(Ki.transpose(1, 2)) * Di).tril;
    VNew := Ui - Wi.matmul(S);
    Outs[I] := (Qi * Gi.exp.unsqueeze(2)).matmul(S) + Ai.matmul(VNew);
    Last := Gi.narrow(1, C - 1, 1);
    S := S * Last.exp.unsqueeze(2)
         + (Ki * (Last - Gi).exp.unsqueeze(2)).transpose(1, 2).matmul(VNew);
  end;
  Result := TTensor.stack(Outs, 1).reshape([H, N * C, Dv]).narrow(1, 0, T);
  State := S.transpose(1, 2);
```

## Checks and messages (EDeltaRule; in this order)

The same as `DeltaRuleTokens` (T08), then the chunk size:
1. the kinds, in the order Q, K, V, Beta, G: `<name> must be kFloat (fp32)`;
2. Q not 3-D: `Q must be [H, T, Dk], got <shape of Q>`;
3. T = 0: `no tokens (T = 0)`;
4. K not the shape of Q: `K must be the shape of Q <shape of Q>, got <shape of K>`;
5. V not [H, T, any]: `V must be [<H>, <T>, Dv], got <shape of V>`
   (read `Dv := V.Shape[2]` only after this check);
6. Beta not [H, T]: `Beta must be [<H>, <T>], got <shape of Beta>`;
7. G not [H, T]: `G must be [<H>, <T>], got <shape of G>`;
8. `ChunkSize < 1`: `'chunk size must be at least 1, got ' + IntToStr(ChunkSize)`.

## UT fixture and helpers

`tests/ut/DeltaRuleFixture.pas` (T08's, harness owner's; nobody changes
it): `RandomInputs(Seed, H, T, Dk, Dv, GScale, Q, K, V, Beta, G)`,
`RefTokens`, `MaxAbsDiff`, `ShapeText`, `ValuesText`. The test program is
T08's `tests/ut/DeltaRule_test.pas` (helpers `Make(Values, Shape)` and
`CheckTensor(What, Shape, Values, T)`); the new methods are added to it.
"tokens(..)" below is `DeltaRuleTokens` on the same inputs (T08's, the
oracle of the value rows).

## Unit tests (rows; every row has one method, every method belongs to a row)

"zeros" is `TTensor.zeros` (kFloat unless a kind is given). Error rows
catch `EDeltaRule` and compare the whole message (`AssertEquals`); a row
fails if no exception comes (`Fail('no EDeltaRule')`). A tolerance row
passes when `MaxAbsDiff ≤` the limit for O and for State; "exact" means
the exact strings or a difference of 0.

Stage 1 (task T09b1), the function and its checks (201–212). Unless a row
says otherwise: Q, K zeros([2, 3, 4]), V zeros([2, 3, 5]), Beta, G
zeros([2, 3]), ChunkSize 64.

| Row | Input | Expected |
|---|---|---|
| 201 | Q zeros([2, 3, 4], kDouble) | `Q must be kFloat (fp32)` |
| 202 | K zeros([2, 3, 4], kDouble) | `K must be kFloat (fp32)` |
| 203 | V zeros([2, 3, 5], kDouble) | `V must be kFloat (fp32)` |
| 204 | Beta zeros([2, 3], kDouble) | `Beta must be kFloat (fp32)` |
| 205 | G zeros([2, 3], kBFloat16) | `G must be kFloat (fp32)` |
| 206 | Q zeros([2, 3]) | `Q must be [H, T, Dk], got [2, 3]` |
| 207 | Q, K zeros([2, 0, 4]), V zeros([2, 0, 5]), Beta, G zeros([2, 0]) | `no tokens (T = 0)` |
| 208 | K zeros([2, 3, 3]) | `K must be the shape of Q [2, 3, 4], got [2, 3, 3]` |
| 209 | V zeros([2, 2, 5]) | `V must be [2, 3, Dv], got [2, 2, 5]` |
| 210 | Beta zeros([2, 4]) | `Beta must be [2, 3], got [2, 4]` |
| 211 | G zeros([2]) | `G must be [2, 3], got [2]` |
| 212 | ChunkSize 0 | `chunk size must be at least 1, got 0` |

Stage 2 (task T09b2), the values (213–224):

| Row | Input | Expected |
|---|---|---|
| 213 | T08's hand case: Q Make([1, 1, 1, 2], [1, 2, 2]), K Make([1, 0, 1, 1], [1, 2, 2]), V Make([2, 4, 3, 3], [1, 2, 2]), Beta Make([0.5, 0.5], [1, 2]), G Make([0, 0], [1, 2]); default chunk | exact: O `[1, 2, 2]` `1 2 4 3.5`; State `[1, 2, 2]` `2 1 2.5 0.5` |
| 214 | as 213 with G Make([0, Ln(0.5)], [1, 2]) | exact: O `1 2 4.25 4`; State `1.75 1.25 2 1` |
| 215 | `RandomInputs(7, 3, 5, 4, 6, 1.0, …)`, Beta replaced by zeros([3, 5]); ChunkSize 2 | O `[3, 5, 6]`, State `[3, 6, 4]`; `O.abs.max` and `State.abs.max` 0 |
| 216 | `RandomInputs(20, 2, 5, 4, 6, 1.0, …)` (Dk 4, Dv 6); ChunkSize 2 | O `[2, 5, 6]`, State `[2, 6, 4]`; against tokens(..) ≤ 1e-6 (measured 1.8e-7, 1.2e-7) |
| 217 | `RandomInputs(16, 4, 50, 16, 16, 1.0, …)`; ChunkSize 1 | against tokens(..) ≤ 1e-6 (measured 1.8e-7) |
| 218 | `RandomInputs(17, 4, 50, 16, 16, 1.0, …)`; ChunkSize 7 (padding 6) | ≤ 1e-6 (measured 1.8e-7) |
| 219 | `RandomInputs(101, 48, 1, 128, 128, 0.1, …)`; default chunk (T = 1) | ≤ 1e-5 (measured 2.4e-7) |
| 220 | `RandomInputs(164, 48, 64, 128, 128, 0.1, …)`; default (T = 64, no padding) | ≤ 1e-5 (measured 7.2e-7) |
| 221 | `RandomInputs(165, 48, 65, 128, 128, 0.1, …)`; default (T = 65, padding 63) | ≤ 1e-5 (measured 5.7e-7) |
| 222 | `RandomInputs(14, 16, 200, 128, 128, 5.0, …)`; default (strong decay) | ≤ 1e-5 (measured 4.4e-6) |
| 223 | `RandomInputs(21, 3, 70, 8, 8, 1.0, …)`; once without ChunkSize, once with 64 | exact: equal O and State |
| 224 | `RandomInputs(22, 4, 70, 16, 16, 1.0, …)`; `ManualSeed(23)`; W := `TTensor.randn([4, 70, 16])`; two copies of the five inputs as new leaves (`detach.Copy.requires_grad_(True)`); backward of `(tokens(..) * W).sum(kNone)` on one and of `(DeltaRuleChunked(.., 16) * W).sum(kNone)` on the other | the five gradients agree ≤ 1e-5 (measured 2.1e-6) |

Build and run (the GPUs hidden as in the gate):
`fpc -Mobjfpc -Sh -vew -l- -Fu{proj}/src -Fu{proj}/../../src -Fu{proj}/tests/ut -FU{proj}/work/bin -Fl{proj}/../../build/libtch-rocm -k-rpath={proj}/../../build/libtch-rocm {proj}/tests/ut/DeltaRule_test.pas -o{proj}/work/bin/DeltaRule_test`
then `HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 {proj}/work/bin/DeltaRule_test --all --format=plain`.
libtorch prints warnings about the hidden GPUs on stderr; they are expected.

## ITa (prepared by the harness owner)

`{proj}/tests/ita/T09-run.sh` (no variable needed) builds
`tests/ita/T09/chunked_check.pas` against `src/`, runs it on the CPU from a
fresh directory outside the project and writes
`{proj}/work/ita/T09/summary.tsv` (case, OK/NG, value, limit, note; then
the driver's exit code and the count of stderr lines other than the GPU
warnings). OK when value ≤ limit. About 10 s. The oracles are
`DeltaRuleTokens` (T08, checked against the scalar reference) and the
fixture's scalar Double reference `RefTokens`.

| Case | Compared | Limit |
|---|---|---|
| ITa-T09-001 | O and State against `DeltaRuleTokens`, H 48, 128 × 128, T = 1, 63, 64, 65, 200 (the design's lengths) | 1e-5 |
| ITa-T09-002 | against the scalar Double reference `RefTokens`, H 48, T 200 | 1e-5 |
| ITa-T09-003 | strong decay, g in (−5, 0], H 16, T 200 | 1e-5 |
| ITa-T09-004 | training length T 2048, H 48 (time in the note) | 1e-5 |
| ITa-T09-005 | gradients of all five inputs against the token form | 1e-5 |
| ITa-T09-006 | chunk sizes 1, 2, 3, 5, 7, 10, 49, 50, 51, 64 at T 50 | 1e-5 |

Measured with the prototype: 7.2e-7, 4.8e-7, 4.4e-6, 7.2e-7, 2.1e-6,
9.8e-7. With the skeleton every case is NG; mutants are caught (State not
transposed; the decay factor left out of the output; the diagonal kept in
L; the padding g not zero). Case 101 (GPU, 8.3e-7, 13 ms at T 2048) is the
harness owner's (`T09-run.sh gpu`); agents do not run it.

## Stages and case IDs

- Stage 1 (task T09b1): `DeltaRuleChunked` (checks and body); rows
  UT-DeltaRule-201 to 212.
- Stage 2 (task T09b2): rows UT-DeltaRule-213 to 224 (no change to the
  unit unless a row fails because the code differs from the facts).
- ITa: ITa-T09-001 to 006.
