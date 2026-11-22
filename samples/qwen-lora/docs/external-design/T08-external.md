# External design — unit `DeltaRule` (T08)

## Purpose

The unit `DeltaRule` implements the gated delta rule of the Gated DeltaNet
(Linear) blocks, token by token (DESIGN 8.10, "Linear block", step 7).

It is the reference form: T09 adds the chunked form and is checked against
this one; T10 calls it from the linear block. The query is used as given
(the block scales it by 1/sqrt(Dk) before, DESIGN step 6); key heads are
already tiled to the value heads (step 5). No autograd requirement.

## The rule

Per head h, a state M [Dv, Dk] (value × key), zero at the sequence start.
Per token t, with g_t ≤ 0 a log decay and 0 ≤ β_t ≤ 1:

```
M   := exp(g_t) · M
d   := β_t · (v_t − M k_t)        (d has Dv values)
M   := M + d k_tᵀ                  (outer product, Dv × Dk)
o_t := M q_t                       (Dv values)
```

### Hand case (H = 1, Dk = Dv = 2, two tokens; all values exact in fp32)

|     | q     | k     | v     | β   | g     | M after (row-major) | o         |
|-----|-------|-------|-------|-----|-------|---------------------|-----------|
| token 0, from M = 0 | (1, 1) | (1, 0) | (2, 4) | 0.5 | 0     | 1 0 2 0             | (1, 2)    |
| token 1             | (1, 2) | (1, 1) | (3, 3) | 0.5 | 0     | 2 1 2.5 0.5         | (4, 3.5)  |
| token 1 with g = ln(1/2) | (1, 2) | (1, 1) | (3, 3) | 0.5 | ln 0.5 | 1.75 1.25 2 1     | (4.25, 4) |

## Interface

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

### `DeltaStep`

One token, all heads.

- M: state tensor [H, Dv, Dk]
- Q, K: query and key tensors [H, Dk]
- V: value tensor [H, Dv]
- Beta, G: beta and decay tensors [H]
- O: output tensor [H, Dv] (output parameter)
- Returns: new state tensor [H, Dv, Dk]

M is not changed; a new tensor is returned.

### `DeltaRuleTokens`

A whole sequence from a zero state.

- Q, K: query and key tensors [H, T, Dk]
- V: value tensor [H, T, Dv]
- Beta, G: beta and decay tensors [H, T]
- T ≥ 1 (at least one token)
- State: final state tensor [H, Dv, Dk] (output parameter)
- Returns: output tensor [H, T, Dv]

All tensors must be kFloat (fp32); results are on the inputs' device.

## Errors (EDeltaRule)

Shapes are written `[2, 3, 4]` (IntToStr of each size, `, ` between, in brackets).

### `DeltaStep`

First the kinds, in the order M, Q, K, V, Beta, G:
`<name> must be kFloat (fp32)` (e.g. `M must be kFloat (fp32)`).

Then the shape checks:

1. M not 3-D: `M must be [H, Dv, Dk], got <shape of M>`
2. Q not [H, Dk]: `Q must be [<H>, <Dk>], got <shape of Q>`
3. K not [H, Dk]: `K must be [<H>, <Dk>], got <shape of K>`
4. V not [H, Dv]: `V must be [<H>, <Dv>], got <shape of V>`
5. Beta not [H]: `Beta must be [<H>], got <shape of Beta>`
6. G not [H]: `G must be [<H>], got <shape of G>`

### `DeltaRuleTokens`

First the kinds, in the order Q, K, V, Beta, G (same message).

Then the shape checks:

1. Q not 3-D: `Q must be [H, T, Dk], got <shape of Q>`
2. T = 0: `no tokens (T = 0)`
3. K not the shape of Q: `K must be the shape of Q <shape of Q>, got <shape of K>`
4. V not [H, T, any]: `V must be [<H>, <T>, Dv], got <shape of V>`
5. Beta not [H, T]: `Beta must be [<H>, <T>], got <shape of Beta>`
6. G not [H, T]: `G must be [<H>, <T>], got <shape of G>`
