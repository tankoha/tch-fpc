# Program flow — unit `QwenBlocks` (T10)

Step-by-step tensor operations for the three routines; the full tensor-op
lists are in `docs/detailed-design/T10-detailed.md` (§4.6–4.8). Tensors are
managed records (no `Free`); the blocks free their `TPackedLinear` fields in
`Destroy`.

## `TQwenBlock.Forward` flow (the skeleton)

```mermaid
flowchart TD
    Start([Start TQwenBlock.Forward H]) --> N1[N1 := RmsNorm H, FAttnNorm, Eps]
    N1 --> Attn[Attn N1 — virtual method]
    Attn --> H1[H1 := H + Attn N1 — first residual]
    H1 --> N2[N := RmsNorm H1, FPostNorm, Eps]
    N2 --> Ffn[G1 := FFfnGate.Forward N, silu — G2 := FFfnUp.Forward N]
    Ffn --> Gate[Gated := G1 times G2]
    Gate --> Down[FFfnDown.Forward, Gated]
    Down --> Out([Return H1 + FFfnDown ...])
```

`Attn` dispatches on the class: `TFullBlock.Attn` (next) or
`TLinearBlock.Attn` (third). There is no other branch.

## `TFullBlock.Attn` flow (gated attention)

```mermaid
flowchart TD
    Start([Start TFullBlock.Attn X — T := X.Shape 0]) --> QG[QG := FQ.Forward X, view T Hq 2Dh]
    QG --> Qpart[Q := RmsNorm QG narrow 2 0 Dh, FQNorm, Eps]
    QG --> Gatep[Gate := QG narrow 2 Dh Dh, reshape T Hq*Dh]
    Qpart --> K[K := RmsNorm FK.Forward X, view T Hkv Dh, FKNorm, Eps]
    Gatep --> V[V := FV.Forward X, view T Hkv Dh]
    K --> RopeQ[Q := RopeText Q, Rot, RopeBase — positions from 0]
    V --> RopeK[K := RopeText K, Rot, RopeBase]
    RopeQ --> SDPA[A := scaled_dot_product_attention of Q K V permuted 1 0 2 unsqueezed 0]
    RopeK --> SDPA
    SDPA --> A2[A := A squeeze_dim 0, permute 1 0 2, reshape T Hq*Dh]
    A2 --> Gated[A times Gate.sigmoid]
    Gated --> Out([Return FOut.Forward ...])
```

SDPA arguments: `attn_mask = Default(TTensor)` (no mask), `dropout_p = 0.0`,
`is_causal = True`, `scale = 1.0 / Sqrt(Dh)` (`TOptDouble`),
`enable_gqa = True` — query head h uses KV head h div (Hq / Hkv).

## `TLinearBlock.Attn` flow (Gated DeltaNet)

```mermaid
flowchart TD
    Start([Start TLinearBlock.Attn X]) --> Qkv[Qkv := FQkv.Forward X]
    Qkv --> Z[Z := FGate.Forward X]
    Z --> B[Beta := FBeta.Forward X, sigmoid]
    B --> G[G := FAlpha.Forward X + FDtBias, softplus — times FSsmA]
    G --> C[C := Qkv.Shape 1]
    C --> Pad[Pad := cat zeros K-1 C and Qkv, dim 0]
    Pad --> U0[U := Pad narrow 0 0 T, times FConv select 1 0]
    U0 --> Loop{i from 1 to K minus 1}
    Loop -->|Yes| Tap[U := U + Pad narrow 0 i T, times FConv select 1 i]
    Tap --> Loop
    Loop -->|No| Silu[U := U, silu]
    Silu --> Split[Q := U narrow 1 0 Hk*Sd, view T Hk Sd; Kt := U narrow 1 Hk*Sd Hk*Sd, view T Hk Sd; V := U narrow 1 2Hk*Sd Hv*Sd, view T Hv Sd]
    Split --> NormQ[Q and Kt, each times its own unit RMS over the last dim]
    NormQ --> HvHk{Hv <> Hk?}
    HvHk -->|Yes| Rep[Q := Q repeat 1 Hv div Hk 1; Kt the same]
    HvHk -->|No| Scale[Q := Q times 1 over sqrt of Sd]
    Rep --> Scale
    Scale --> Delta[O := DeltaRuleChunked Q Kt V permuted 1 0 2, Beta and G transposed 0 1 contiguously, fresh State]
    Delta --> GateO[O := RmsNorm O permuted 1 0 2, FSsmNorm, Eps, times Z view T Hv Sd, silu]
    GateO --> Out([Return FSsmOut.Forward, O reshaped T Hv*Sd])
```

The convolution tap `I` multiplies the token `I` steps back, so the index
`K − 1` (ConvKernel − 1) multiplies the current token. `DeltaRuleChunked` is
T09's chunked delta rule (chunk 64, zero-padded), with the recurrence state
a fresh local per call; its inputs' `G` must be finite (the caller's
obligation, T09 §6).

## Branch coverage notes

### `TLinearBlock.Attn`

| Branch | Condition | UT / ITa case |
|--------|-----------|---------------|
| Head tiling taken (`Hv <> Hk`) | tiny model: 2 key / 4 value heads | UT-QwenBlocks-211, 212, 213, 214, 215; ITa-T10-003 |
| Head tiling not taken (`Hv = Hk`) | rehearsal model: 16 key / 16 value heads | ITa-T10-001, 002, 004, 005 |
| Convolution loop, all `K − 1` extra taps | tiny model: kernel 4 (taps 1, 2, 3) | UT-QwenBlocks-211, 212 (T 70 = two delta-rule chunks), 213, 214 |
| Convolution loop, rehearsal kernel | kernel from the rehearsal metadata | ITa-T10-001 to 005 |

### `TFullBlock.Attn`

| Path | Where | UT / ITa case |
|------|-------|---------------|
| Fused Q+gate projection, Q/K RMS norm, RoPE | every call | UT-QwenBlocks-201 to 204; ITa-T10-001 to 003 |
| GQA (`Hq > Hkv`), causal SDPA, scale, gate | tiny model 4 query / 2 KV heads, T 8 and T 70 | UT-QwenBlocks-201, 202 (T 70), 203 (causality: first 5 rows agree), 204 (gradients); ITa-T10-001 to 005 |

### `TQwenBlock.Forward`

No branch of its own; the only dispatch is the virtual `Attn`:

| Path | UT / ITa case |
|------|---------------|
| `Attn` → `TLinearBlock` (blk.0, blk.2 of the tiny model; rehearsal blocks 0, 1) | UT-QwenBlocks-211 to 214; ITa-T10-001 to 005 |
| `Attn` → `TFullBlock` (blk.1 of the tiny model; rehearsal blocks 3, …) | UT-QwenBlocks-201 to 204; ITa-T10-001 to 005 |
| `Recompute` stages: input gradient vs ordinary backward | ITa-T10-005 (exact, 0) |

The checks of `TPackedLinear.Create` / `Forward` / `TensorRaw` / `LoadPacked`
and of `KeyInt` / `KeyFloat` are driven by the rows of `tests/ut/cases/
PackedLinear.cases.md` (UT-PackedLinear-001 to 015) and
`QwenBlocks.cases.md` (UT-QwenBlocks-101 to 107, plus 216 for the
`KeyFloat` missing-key raise), per the detailed design §5.
