# Detailed design — unit `DeltaRule` (T08)

## 1. Module structure

| Source file | Contents |
|-------------|----------|
| `src/DeltaRule.pas` | Unit implementing the gated delta rule: `DeltaStep`, `DeltaRuleTokens`, exception `EDeltaRule` |

### Functions

| Function | Purpose |
|----------|---------|
| `DeltaStep` | One token step for all heads; returns new state and output |
| `DeltaRuleTokens` | Process a whole sequence from zero state; returns outputs and final state |

## 2. Function specifications

### `DeltaStep`

```pascal
function DeltaStep(const M, Q, K, V, Beta, G: TTensor; out O: TTensor): TTensor;
```

**Arguments:**
- `M`: state tensor [H, Dv, Dk] (kFloat)
- `Q`: query tensor [H, Dk] (kFloat)
- `K`: key tensor [H, Dk] (kFloat)
- `V`: value tensor [H, Dv] (kFloat)
- `Beta`: beta tensor [H] (kFloat)
- `G`: decay tensor [H] (kFloat)

**Returns:**
- Result: new state tensor [H, Dv, Dk]
- `O`: output tensor [H, Dv]

**Preconditions:**
- All tensors are kFloat
- M is 3-D [H, Dv, Dk]
- Q and K are [H, Dk]
- V is [H, Dv]
- Beta and G are [H]

**Exceptions:**
- `EDeltaRule` with message if preconditions not met

### `DeltaRuleTokens`

```pascal
function DeltaRuleTokens(const Q, K, V, Beta, G: TTensor; out State: TTensor): TTensor;
```

**Arguments:**
- `Q`: query tensor [H, T, Dk] (kFloat)
- `K`: key tensor [H, T, Dk] (kFloat)
- `V`: value tensor [H, T, Dv] (kFloat)
- `Beta`: beta tensor [H, T] (kFloat)
- `G`: decay tensor [H, T] (kFloat)

**Returns:**
- Result: output tensor [H, T, Dv]
- `State`: final state tensor [H, Dv, Dk]

**Preconditions:**
- All tensors are kFloat
- Q is 3-D [H, T, Dk] with T ≥ 1
- K has the same shape as Q
- V is [H, T, Dv]
- Beta and G are [H, T]

**Exceptions:**
- `EDeltaRule` with message if preconditions not met

## 3. Data structures

### Tensors

All tensors use kFloat (fp32). Results are on the same device as inputs.

| Tensor | Shape | Purpose |
|--------|-------|---------|
| M | [H, Dv, Dk] | State (value × key matrix per head) |
| Q | [H, Dk] or [H, T, Dk] | Query |
| K | [H, Dk] or [H, T, Dk] | Key |
| V | [H, Dv] or [H, T, Dv] | Value |
| Beta | [H] or [H, T] | Beta (interpolation factor) |
| G | [H] or [H, T] | Log decay |
| O | [H, Dv] or [H, T, Dv] | Output |
| State | [H, Dv, Dk] | Final state |

### Local variables in `DeltaRuleTokens`

- `Outs`: array of TTensor [0..T-1] to collect outputs per token
- `M`: current state tensor [H, Dv, Dk]
- `Ot`: output tensor for one token

## 4. Processing details

### 4.1 `DeltaStep` algorithm

After validation (H = M.Shape[0], Dv = M.Shape[1], Dk = M.Shape[2]):

```pascal
Md := M * G.exp.view([H, 1, 1]);
Pred := Md.matmul(K.unsqueeze(2)).squeeze_dim(2);          { [H, Dv] }
D := (V - Pred) * Beta.unsqueeze(1);                        { [H, Dv] }
Result := Md + D.unsqueeze(2).matmul(K.unsqueeze(1));       { [H, Dv, Dk] }
O := Result.matmul(Q.unsqueeze(2)).squeeze_dim(2);          { [H, Dv] }
```

**Tensor operations sequence:**
1. `G.exp` — element-wise exponential of decay
2. `.view([H, 1, 1])` — reshape for broadcasting with M
3. `M * G.exp.view(...)` — decay the state (Md)
4. `K.unsqueeze(2)` — reshape K from [H, Dk] to [H, Dk, 1]
5. `Md.matmul(K.unsqueeze(2))` — matrix product yields [H, Dv, 1]
6. `.squeeze_dim(2)` — remove dimension 2 to get [H, Dv] (Pred = Md × K)
7. `V - Pred` — prediction error
8. `Beta.unsqueeze(1)` — reshape Beta from [H] to [H, 1]
9. `(V - Pred) * Beta.unsqueeze(1)` — scale error by beta (D)
10. `D.unsqueeze(2)` — reshape D from [H, Dv] to [H, Dv, 1]
11. `K.unsqueeze(1)` — reshape K from [H, Dk] to [H, 1, Dk]
12. `D.unsqueeze(2).matmul(K.unsqueeze(1))` — outer product [H, Dv, Dk]
13. `Md + ...` — update state (Result)
14. `Q.unsqueeze(2)` — reshape Q from [H, Dk] to [H, Dk, 1]
15. `Result.matmul(Q.unsqueeze(2))` — matrix product [H, Dv, 1]
16. `.squeeze_dim(2)` — remove dimension 2 to get [H, Dv] (O)

### 4.2 `DeltaRuleTokens` algorithm

After validation (H = Q.Shape[0], T = Q.Shape[1], Dk = Q.Shape[2], Dv = V.Shape[2]):

```pascal
M := TTensor.zeros([H, Dv, Dk], kFloat, Q.Device);
for I := 0 to T - 1 do
  M := DeltaStep(M, Q.select(1, I), K.select(1, I), V.select(1, I),
                 Beta.select(1, I), G.select(1, I), Ot);
  Outs[I] := Ot;
end;
Result := TTensor.stack(Outs, 1);
State := M;
```

**Loop structure:**
1. Initialize state M to zeros
2. Loop I from 0 to T-1:
   - Extract token I: `Q.select(1, I)` → [H, Dk]
   - Call `DeltaStep` with token I of all inputs
   - Store output in `Outs[I]`
   - Update state M
3. Stack outputs along dimension 1: `TTensor.stack(Outs, 1)` → [H, T, Dv]
4. Return final state

## 5. Error handling

### 5.1 `DeltaStep` checks (in order)

**Kind checks (M, Q, K, V, Beta, G):**

1. M not kFloat: `M must be kFloat (fp32)`
2. Q not kFloat: `Q must be kFloat (fp32)`
3. K not kFloat: `K must be kFloat (fp32)`
4. V not kFloat: `V must be kFloat (fp32)`
5. Beta not kFloat: `Beta must be kFloat (fp32)`
6. G not kFloat: `G must be kFloat (fp32)`

**Shape checks (after kind checks):**

1. M not 3-D: `M must be [H, Dv, Dk], got [x, y]`
2. Q not [H, Dk]: `Q must be [<H>, <Dk>], got [x, y]`
3. K not [H, Dk]: `K must be [<H>, <Dk>], got [x, y]`
4. V not [H, Dv]: `V must be [<H>, <Dv>], got [x, y]`
5. Beta not [H]: `Beta must be [<H>], got [x]`
6. G not [H]: `G must be [<H>], got [x]`

Where `<H>`, `<Dk>`, `<Dv>` are the actual values from M's shape.

### 5.2 `DeltaRuleTokens` checks (in order)

**Kind checks (Q, K, V, Beta, G):**

1. Q not kFloat: `Q must be kFloat (fp32)`
2. K not kFloat: `K must be kFloat (fp32)`
3. V not kFloat: `V must be kFloat (fp32)`
4. Beta not kFloat: `Beta must be kFloat (fp32)`
5. G not kFloat: `G must be kFloat (fp32)`

**Shape checks (after kind checks):**

1. Q not 3-D: `Q must be [H, T, Dk], got [x, y]`
2. Q.Shape[1] = 0: `no tokens (T = 0)`
3. K not same shape as Q: `K must be the shape of Q [x, y, z], got [a, b, c]`
4. V not [H, T, any]: `V must be [<H>, <T>, Dv], got [x, y, z]`
5. Beta not [H, T]: `Beta must be [<H>, <T>], got [x, y]`
6. G not [H, T]: `G must be [<H>, <T>], got [x, y]`

Where `<H>` = Q.Shape[0], `<T>` = Q.Shape[1].

## 6. Change history

| Date | Change | Reason |
|------|--------|--------|
| 2026-11-22 | Initial creation | T08a delegation from main agent |
