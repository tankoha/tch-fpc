# T09 External Design: DeltaRuleChunked

## Purpose

T09 adds `DeltaRuleChunked` to the unit `DeltaRule` (T08): the gated delta rule
in the chunked form. T10's linear block uses it for training and scoring. It
must give the same results as the token form (`DeltaRuleTokens`) but faster
for long sequences.

## The chunked form

For one head, chunk c of C tokens (positions i, j inside the chunk), with
`Gc` the cumulative sum of g inside the chunk:

- Decay[i, j] = exp(Gc[i] − Gc[j]) for i ≥ j, else 0
- L = strictly lower part of (β k) kᵀ ⊙ Decay; T = (I + L)⁻¹ (triangular solve)
- U = T (β v); W = T (β k ⊙ exp(Gc))
- Per chunk, with the state S [Dk, Dv] (the transpose of T08's M):
  - v′ = U − W S
  - o = (q ⊙ exp(Gc)) S + (strictly-and-diagonal lower part of q kᵀ ⊙ Decay) v′
  - S := exp(Gc[C−1]) S + (k ⊙ exp(Gc[C−1] − Gc))ᵀ v′

Padding tokens have q = k = v = 0, β = 0 and g = 0: they leave the state
unchanged, and their outputs are cut off.

## Interface

```pascal
function DeltaRuleChunked(const Q, K, V, Beta, G: TTensor; out State: TTensor;
  ChunkSize: Integer = 64): TTensor;
```

Inputs and outputs as `DeltaRuleTokens`:
- Q, K [H, T, Dk]
- V [H, T, Dv]
- Beta, G [H, T]
- Returns O [H, T, Dv]
- State [H, Dv, Dk]

The default chunk size is 64 tokens.

## Errors (EDeltaRule)

Checks in order:

1. Kinds: Q, K, V, Beta, G must be kFloat (fp32), in that order
2. Q not 3-D: `Q must be [H, T, Dk], got <shape of Q>`
3. T = 0: `no tokens (T = 0)`
4. K not the shape of Q: `K must be the shape of Q <shape of Q>, got <shape of K>`
5. V not [H, T, any]: `V must be [<H>, <T>, Dv], got <shape of V>`
6. Beta not [H, T]: `Beta must be [<H>, <T>], got <shape of Beta>`
7. G not [H, T]: `G must be [<H>, <T>], got <shape of G>`
8. ChunkSize < 1: `chunk size must be at least 1, got <ChunkSize>`

## G must be finite

g = −inf (allowed in the token form) gives NaN here (−inf − (−inf) in the
decay matrix). The real gates are finite. The function does not check this;
the caller must ensure G is finite.
