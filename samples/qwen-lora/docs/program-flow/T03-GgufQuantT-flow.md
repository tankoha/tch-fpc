# Program Flow: GgufQuantT Unit

## DequantizeT Function Flow

```mermaid
flowchart TD
    A[Start DequantizeT] --> B{Check ARaw tensor}
    B -->|Not kUInt8 or not 1-D| C[Raise EGgufError: tensor type/shape]
    B -->|Valid tensor| D{Check ATType}
    D -->|Unsupported type| E[Raise EGgufError: unsupported tensor type N]
    D -->|Supported type| F{Check ACount}
    F -->|Not multiple of block size| G[Raise EGgufError: block size mismatch]
    F -->|Valid block size| H{Check ARaw length}
    H -->|≠ GgufTensorBytes| I[Raise EGgufError: length mismatch]
    H -->|Valid length| J[Switch on ATType]
    J --> K{ATType value}
    K -->|0 (f32)| L[ARaw.contiguous.view_dtype<br/>(kFloat)]
    K -->|1 (f16)| M[ARaw.contiguous.view_dtype<br/>(kHalf).totype(kFloat)]
    K -->|30 (bf16)| N[ARaw.contiguous.view_dtype<br/>(kBFloat16).totype(kFloat)]
    K -->|8 (q8_0)| O[R := ARaw.reshape([N/32, 34])]
    O --> P[D := Cols(R,0,2).contiguous<br/>.view_dtype(kHalf).totype(kFloat)]
    P --> Q[Q := Cols(R,2,34).contiguous<br/>.view_dtype(kInt8).totype(kFloat)]
    Q --> R[(Q * D).reshape([N])]
    K -->|12 (q4_K) or 13 (q5_K)| S[R := ARaw.reshape([nb, 144 or 176])]
    S --> T[D, DMin from bytes 0-3]
    T --> U[S = Cols(R,4,16)]
    U --> V[Sc = cat([S[0-3] and 63,<br/>(S[8-11] and 15) or<br/>((S[0-3] shr 6) shl 4)], 1)]
    V --> W[Mn = cat([S[4-7] and 63,<br/>(S[8-11] shr 4) or<br/>((S[4-7] shr 6) shl 4)], 1)]
    W --> X[Qs = quant bytes reshaped [nb,4,32]]
    X --> Y[Q = stack([Qs and 15, Qs shr 4], 2)]
    Y --> Z{q5_K?}
    Z -->|Yes| AA[Apply Qh high bits<br/>to reconstruct full Q]
    Z -->|No| AB[Continue with Q]
    AA --> AC[Q reshaped [nb,8,32]]
    AB --> AC
    AC --> AD[((D * Sc.totype(kFloat)<br/>.unsqueeze(2)) * Q.totype(kFloat)<br/>- DMin * Mn.totype(kFloat)<br/>.unsqueeze(2)).reshape([N])]
    K -->|14 (q6_K)| AE[R := ARaw.reshape([nb,210])]
    AE --> AF[Extract Ql, Qh, Sc, D]
    AF --> AG[Reconstruct quantized values Q]
    AG --> AH[((D * Sc.totype(kFloat)<br/>.unsqueeze(2)) * (Q.totype(kFloat) - 32))<br/>.reshape([N])]
    L --> AI[Return result tensor]
    M --> AI
    N --> AI
    R --> AI
    AD --> AI
    AH --> AI
```

## QuantizeQ8_0T Function Flow

```mermaid
flowchart TD
    A[Start QuantizeQ8_0T] --> B{Check AX tensor}
    B -->|Not kFloat or not 1-D| C[Raise EGgufError: tensor type/shape]
    B -->|Valid tensor| D{Check length}
    D -->|Not multiple of 32| E[Raise EGgufError: length not multiple of 32]
    D -->|Valid length| F{Check finite values}
    F -->|Non-finite values present| G[Raise EGgufError: non-finite values]
    F -->|All values finite| H[X := AX.reshape([nb,32])]
    H --> I[Amax := X.abs.amax([1], True)]
    I --> J[D := Amax / 127.0]
    J --> K[Tiny := D.le(2.9387358770557188e-39)]
    K --> L[Id := D.reciprocal.where_scalarother<br/>(D.gt(0.0).logical_and(Tiny.logical_not), 0.0)]
    L --> M[X0 := X * Id]
    M --> N[T := X0.trunc]
    N --> O[Q := T + X0.sign * (X0 - T).abs.ge(0.5).totype(kFloat)]
    O --> P[Q := Q.where_scalarother(Tiny.logical_not, 0.0)]
    P --> Q[DH := D.where_scalarother(Tiny.logical_not, 0.0)<br/>.totype(kHalf).view_dtype(kUInt8)]
    Q --> R[cat([DH, Q.totype(kInt8).view_dtype(kUInt8)], 1)<br/>.reshape([nb * 34])]
    R --> S[Return result tensor]
```