# Detailed design — unit `DeltaRule`, `DeltaRuleChunked` (T09)

Based on the external design `docs/external-design/T09-external.md` and the
facts `tasks/T09-facts.md` (final, checked by the harness owner). T09 adds
`DeltaRuleChunked` to the unit `DeltaRule` (T08): the gated delta rule in the
chunked form (DESIGN 8.10, "Linear block", step 7). T10's linear block uses it
for training and scoring; it must give what the token form gives.

## 1. Module structure

| Source file | Contents |
|-------------|----------|
| `src/DeltaRule.pas` | Unit implementing the gated delta rule: `DeltaStep`, `DeltaRuleTokens`, `DeltaRuleChunked`, exception `EDeltaRule`, private shape-text helper `ShapeStr` |

### Functions

| Function | Purpose |
|----------|---------|
| `DeltaStep` | One token step for all heads; returns new state and output (T08, unchanged) |
| `DeltaRuleTokens` | Process a whole sequence from zero state; returns outputs and final state (T08, unchanged) |
| `DeltaRuleChunked` | The gated delta rule in the chunked form: the sequence padded to a multiple of `ChunkSize` (64 by default), one state update per chunk, the final state returned |

Do not change `DeltaStep` or `DeltaRuleTokens`.

## 2. Function specification

### `DeltaRuleChunked`

```pascal
function DeltaRuleChunked(const Q, K, V, Beta, G: TTensor; out State: TTensor;
  ChunkSize: Integer = 64): TTensor;
```

**Arguments:**
- `Q`: query tensor [H, T, Dk] (kFloat)
- `K`: key tensor [H, T, Dk] (kFloat)
- `V`: value tensor [H, T, Dv] (kFloat)
- `Beta`: beta tensor [H, T] (kFloat)
- `G`: decay tensor [H, T] (kFloat); must be finite (§6)
- `ChunkSize`: number of tokens per chunk; default 64

Inputs and outputs are as `DeltaRuleTokens`: Q, K [H, T, Dk]; V [H, T, Dv];
Beta, G [H, T].

**Returns:**
- Result: output tensor O [H, T, Dv]
- `State`: final state tensor [H, Dv, Dk] (T08's state layout)

Both tensors are on the same device as the inputs.

**Preconditions:**
- All tensors are kFloat
- Q is 3-D [H, T, Dk] with T ≥ 1
- K has the same shape as Q
- V is [H, T, Dv]
- Beta and G are [H, T]
- ChunkSize ≥ 1
- G is finite (the caller's obligation; §6)

**Exceptions:**
- `EDeltaRule` with the exact message of the first failing check (§5)

**Values:** with finite G the outputs and the final state agree with
`DeltaRuleTokens` on the same inputs within 1e-5 (the design's limit).
Measured (prototype, CPU): at the 27B model's sizes (48 heads, 128 × 128),
T = 1, 63, 64, 65, 200, 2048: at most 7.2e-7; with strong decay (g down to
−5 per token) 4.4e-6; the gradients of all five inputs 2.1e-6; T08's two hand
cases come out exactly. At T = 2048: 100 ms against 379 ms for the token form
on the CPU; 13 ms on the RX 9070.

## 3. Data structures

### Tensors

All tensors use kFloat (fp32). Intermediate tensors are on the same device as
the inputs (`Q.Device`).

| Tensor | Shape | Purpose |
|--------|-------|---------|
| Qp, Kp, Vp | [H, N·C, Dk], [H, N·C, Dk], [H, N·C, Dv] | Padded inputs (or the inputs themselves when P = 0) |
| Bp, Gp | [H, N·C] | Padded beta and g |
| KB | [H, N, C, Dk] | (β k) reshaped per chunk |
| VB | [H, N, C, Dv] | (β v) reshaped per chunk |
| Gc | [H, N, C] | Cumulative sum of g inside each chunk (padding g = 0) |
| Decay | [H, N, C, C] | exp(Gc[i] − Gc[j]) for i ≥ j, else 0 |
| L | [H, N, C, C] | Strictly lower part of (β k) kᵀ ⊙ Decay |
| Eye | [C, C] | Identity for the triangular solve |
| Tm | [H, N, C, C] | (I + L)⁻¹ |
| U | [H, N, C, Dv] | Tm (β v) |
| W | [H, N, C, Dk] | Tm (β k ⊙ exp(Gc)) |
| S | [H, Dk, Dv] | State across chunks — the transpose of T08's M |
| Qi, Ki | [H, C, Dk] | Chunk I of Q, K |
| Ui | [H, C, Dv] | Chunk I of U |
| Wi | [H, C, Dk] | Chunk I of W |
| Gi | [H, C] | Chunk I of Gc |
| Di | [H, C, C] | Chunk I of Decay |
| Ai | [H, C, C] | Strictly-and-diagonal lower part of q kᵀ ⊙ Decay (chunk I) |
| VNew | [H, C, Dv] | v′ = U − W S (chunk I) |
| Last | [H, 1] | Gc[C−1] of chunk I |
| Outs | array of TTensor [0..N−1] | Per-chunk outputs |

### Local variables (final, from the facts)

```pascal
H, T, Dk, Dv, C, N, P, I: Int64;
Qp, Kp, Vp, Bp, Gp, KB, VB, Gc, Decay, L, Eye, Tm, U, W, S: TTensor;
Qi, Ki, Ui, Wi, Gi, Di, Ai, VNew, Last: TTensor;
Outs: array of TTensor;
```

Dimension values: `H` = Q.Shape[0], `T` = Q.Shape[1], `Dk` = Q.Shape[2]
(after the Q checks); `Dv := V.Shape[2]` is read only after the V check.

## 4. Processing details

First `State := Default(TTensor); Result := Default(TTensor);`, then the
checks (§5), then the body below.

### 4.1 Padding

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
```

- `N` = number of chunks = ceil(T / C) by integer arithmetic;
  `P` = N·C − T padding tokens; the only branch in the function is `if P > 0`.
- When P > 0, P zero tokens are appended at the end of the token axis (dim 1)
  of every input; when P = 0 the inputs are used as-is.
- Padding tokens have q = k = v = 0, β = 0 and g = 0: they leave the state
  unchanged in the loop (§4.3), and their outputs are cut off
  (`narrow(1, 0, T)`, §4.3).

**Tensor ops:**
1. `N := (T + C - 1) div C` — ceil(T / C), the number of chunks
2. `P := N * C - T` — number of padding tokens
3. `TTensor.zeros([H, P, Dk] / [H, P, Dv] / [H, P], kFloat, Q.Device)` — one
   zero block per input, on the inputs' device
4. `TTensor.cat([X, zeros], 1)` — append the block on the token axis, for
   Q, K, V, Beta, G

### 4.2 Per-chunk matrices (Decay, L, Tm, U, W)

```pascal
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
```

Mathematics (for one head, positions i, j inside a chunk, `Gc` the
cumulative sum of g inside the chunk):

- Decay[i, j] = exp(Gc[i] − Gc[j]) for i ≥ j, else 0
- L = the strictly lower part of (β k) kᵀ ⊙ Decay; Tm = (I + L)⁻¹ (triangular solve)
- U = Tm (β v); W = Tm (β k ⊙ exp(Gc))

**Tensor ops:**
1. `Bp.unsqueeze(2)` — [H, N·C] → [H, N·C, 1] for broadcasting
2. `Kp * Bp.unsqueeze(2)` → `reshape([H, N, C, Dk])` — KB = (β k) per chunk
3. `Vp * Bp.unsqueeze(2)` → `reshape([H, N, C, Dv])` — VB = (β v) per chunk
4. `Qp.reshape([H, N, C, Dk])`, `Kp.reshape([H, N, C, Dk])`
5. `Gp.reshape([H, N, C]).cumsum(2)` — Gc: cumulative sum of g inside each
   chunk (the padding g are 0)
6. `Gc.unsqueeze(3) - Gc.unsqueeze(2)` — Gc[i] − Gc[j] over i, j in a chunk;
   `.tril` zeroes the upper triangle; `.exp`; `.tril` again zeroes above the
   diagonal (the diagonal is exp(0) = 1) → Decay
7. `KB.matmul(Kp.transpose(2, 3))` — (β k) kᵀ per chunk; `* Decay`;
   `.tril(-1)` keeps the strictly lower part → L
8. `TTensor.eye(C, kFloat, Q.Device)` — C × C identity
9. `(L + Eye).linalg_solve_triangular(Eye.expand([H, N, C, C]), False)` —
   solves X · A = B with B the identity, so X = (I + L)⁻¹ → Tm
10. `Tm.matmul(VB)` → U
11. `Gc.exp.unsqueeze(3)` — exp(Gc) broadcast over Dk; `KB * ...`;
    `Tm.matmul(...)` → W

### 4.3 Chunk loop and final state

```pascal
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

Per-chunk state update, with the state S [Dk, Dv] (the transpose of T08's M):

- v′ = U − W S
- o = (q ⊙ exp(Gc)) S + (strictly-and-diagonal lower part of q kᵀ ⊙ Decay) v′
- S := exp(Gc[C−1]) S + (k ⊙ exp(Gc[C−1] − Gc))ᵀ v′

**Tensor ops (initialisation):**
1. `TTensor.zeros([H, Dk, Dv], kFloat, Q.Device)` — S starts at the zero state
2. `SetLength(Outs, N)` — room for one output per chunk

**Tensor ops (each iteration I):**
1. `select(1, I)` on Qp, Kp, U, W, Gc, Decay → Qi, Ki, Ui, Wi, Gi, Di
2. `Qi.matmul(Ki.transpose(1, 2))` — q kᵀ of the chunk; `* Di`; `.tril`
   (diagonal kept, in contrast to L) → Ai
3. `Ui - Wi.matmul(S)` → VNew = v′
4. `Gi.exp.unsqueeze(2)` — exp(Gc) broadcast over Dv;
   `(Qi * ...).matmul(S)` — the state part of the output;
   `Ai.matmul(VNew)` — the attention part; sum → `Outs[I]`
5. `Gi.narrow(1, C - 1, 1)` → Last = Gc[C−1] (shape [H, 1])
6. `Last.exp.unsqueeze(2)` — exp(Gc[C−1]); `S * ...` — decay the state
7. `Last - Gi` — Gc[C−1] − Gc[i]; `.exp.unsqueeze(2)`;
   `Ki * ...` — k ⊙ exp(Gc[C−1] − Gc); `.transpose(1, 2)`; `.matmul(VNew)`;
   add to the decayed S → the new state

**Tensor ops (after the loop):**
1. `TTensor.stack(Outs, 1)` — [H, N, C, Dv];
   `reshape([H, N * C, Dv])` — back to the token axis;
   `narrow(1, 0, T)` — cut off the P padding outputs → Result (O)
2. `S.transpose(1, 2)` — [H, Dk, Dv] → [H, Dv, Dk], T08's state layout → State

## 5. Checks and messages (in order)

All checks raise `EDeltaRule`. The same checks as `DeltaRuleTokens` (T08),
then the chunk size. `<shape…>` is built with the unit's private `ShapeStr`;
`<H>` = Q.Shape[0], `<T>` = Q.Shape[1].

| # | Failing condition | Message (exact) |
|---|-------------------|-----------------|
| 1 | Q not kFloat | `Q must be kFloat (fp32)` |
| 2 | K not kFloat | `K must be kFloat (fp32)` |
| 3 | V not kFloat | `V must be kFloat (fp32)` |
| 4 | Beta not kFloat | `Beta must be kFloat (fp32)` |
| 5 | G not kFloat | `G must be kFloat (fp32)` |
| 6 | Q not 3-D | `Q must be [H, T, Dk], got <shape of Q>` |
| 7 | T = 0 | `no tokens (T = 0)` |
| 8 | K not the shape of Q | `K must be the shape of Q <shape of Q>, got <shape of K>` |
| 9 | V not [H, T, any] | `V must be [<H>, <T>, Dv], got <shape of V>` |
| 10 | Beta not [H, T] | `Beta must be [<H>, <T>], got <shape of Beta>` |
| 11 | G not [H, T] | `G must be [<H>, <T>], got <shape of G>` |
| 12 | ChunkSize < 1 | `'chunk size must be at least 1, got ' + IntToStr(ChunkSize)` |

`Dv := V.Shape[2]` is read only after check 9.

## 6. G must be finite

g = −inf (allowed in the token form) gives NaN here (−inf − (−inf) in the
decay matrix). The real gates are finite. The function does not check this;
the caller must ensure G is finite.

## 7. Change history

| Date | Change | Reason |
|------|--------|--------|
| 2026-11-22 | Initial creation | T09a delegation: the T09 documents |
