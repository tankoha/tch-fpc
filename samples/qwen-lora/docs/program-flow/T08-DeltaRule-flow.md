# Program flow — unit `DeltaRule` (T08)

## `DeltaStep` function flow

```mermaid
flowchart TD
    Start([Start DeltaStep]) --> KindCheck{Check tensor kinds}
    
    KindCheck -->|M not kFloat| ErrM[E: M must be kFloat fp32]
    KindCheck -->|Q not kFloat| ErrQ[E: Q must be kFloat fp32]
    KindCheck -->|K not kFloat| ErrK[E: K must be kFloat fp32]
    KindCheck -->|V not kFloat| ErrV[E: V must be kFloat fp32]
    KindCheck -->|Beta not kFloat| ErrBeta[E: Beta must be kFloat fp32]
    KindCheck -->|G not kFloat| ErrG[E: G must be kFloat fp32]
    
    KindCheck -->|All kFloat| ShapeM{M is 3-D?}
    
    ShapeM -->|No| ErrMShape[E: M must be H Dv Dk got shape]
    ShapeM -->|Yes| SetDim[Set H Dv Dk from M shape]
    
    SetDim --> ShapeQ{Q is H Dk?}
    ShapeQ -->|No| ErrQShape[E: Q must be H Dk got shape]
    ShapeQ -->|Yes| ShapeK{K is H Dk?}
    
    ShapeK -->|No| ErrKShape[E: K must be H Dk got shape]
    ShapeK -->|Yes| ShapeV{V is H Dv?}
    
    ShapeV -->|No| ErrVShape[E: V must be H Dv got shape]
    ShapeV -->|Yes| ShapeBeta{Beta is H?}
    
    ShapeBeta -->|No| ErrBetaShape[E: Beta must be H got shape]
    ShapeBeta -->|Yes| ShapeG{G is H?}
    
    ShapeG -->|No| ErrGShape[E: G must be H got shape]
    ShapeG -->|Yes| Decay[Md := G.exp.view H 1 1 times M]
    
    Decay --> Pred[Pred := Md.matmul K.unsqueeze 2.squeeze_dim 2]
    Pred --> Delta[D := V minus Pred times Beta.unsqueeze 1]
    Delta --> Update[Result := Md + D.unsqueeze 2.matmul K.unsqueeze 1]
    Update --> Output[O := Result.matmul Q.unsqueeze 2.squeeze_dim 2]
    Output --> Return([Return Result O])
    
    ErrM --> End([End])
    ErrQ --> End
    ErrK --> End
    ErrV --> End
    ErrBeta --> End
    ErrG --> End
    ErrMShape --> End
    ErrQShape --> End
    ErrKShape --> End
    ErrVShape --> End
    ErrBetaShape --> End
    ErrGShape --> End
```

## `DeltaRuleTokens` function flow

```mermaid
flowchart TD
    Start([Start DeltaRuleTokens]) --> KindCheck{Check tensor kinds}
    
    KindCheck -->|Q not kFloat| ErrQ[E: Q must be kFloat fp32]
    KindCheck -->|K not kFloat| ErrK[E: K must be kFloat fp32]
    KindCheck -->|V not kFloat| ErrV[E: V must be kFloat fp32]
    KindCheck -->|Beta not kFloat| ErrBeta[E: Beta must be kFloat fp32]
    KindCheck -->|G not kFloat| ErrG[E: G must be kFloat fp32]
    
    KindCheck -->|All kFloat| ShapeQ{Q is 3-D?}
    
    ShapeQ -->|No| ErrQShape[E: Q must be H T Dk got shape]
    ShapeQ -->|Yes| SetDim[Set H T Dk from Q shape]
    
    SetDim --> CheckT{T = 0?}
    CheckT -->|Yes| ErrNoTokens[E: no tokens T = 0]
    CheckT -->|No| ShapeK{K same as Q?}
    
    ShapeK -->|No| ErrKShape[E: K must be shape of Q got shape]
    ShapeK -->|Yes| ShapeV{V is H T any?}
    
    ShapeV -->|No| ErrVShape[E: V must be H T Dv got shape]
    ShapeV -->|Yes| ShapeBeta{Beta is H T?}
    
    ShapeBeta -->|No| ErrBetaShape[E: Beta must be H T got shape]
    ShapeBeta -->|Yes| ShapeG{G is H T?}
    
    ShapeG -->|No| ErrGShape[E: G must be H T got shape]
    ShapeG -->|Yes| InitState[M := zeros H Dv Dk on Q.Device]
    
    InitState --> LoopInit[I := 0]
    LoopInit --> LoopCond{I < T?}
    
    LoopCond -->|Yes| SelectToken[Select token I from Q K V Beta G]
    SelectToken --> CallStep[M Ot := DeltaStep M Q K V Beta G]
    CallStep --> StoreOut[Outs I := Ot]
    StoreOut --> LoopInc[I := I + 1]
    LoopInc --> LoopCond
    
    LoopCond -->|No| StackResult[Result := stack Outs 1]
    StackResult --> SetState[State := M]
    SetState --> Return([Return Result State])
    
    ErrQ --> End([End])
    ErrK --> End
    ErrV --> End
    ErrBeta --> End
    ErrG --> End
    ErrQShape --> End
    ErrNoTokens --> End
    ErrKShape --> End
    ErrVShape --> End
    ErrBetaShape --> End
    ErrGShape --> End
```

## Branch coverage notes

### `DeltaStep` branches

| Branch | Condition | UT case |
|--------|-----------|---------|
| M not kFloat | M.Kind ≠ kFloat | UT-DeltaRule-009 |
| Q not kFloat | Q.Kind ≠ kFloat | UT-DeltaRule-017 |
| K not kFloat | K.Kind ≠ kFloat | UT-DeltaRule-018 |
| V not kFloat | V.Kind ≠ kFloat | UT-DeltaRule-019 |
| Beta not kFloat | Beta.Kind ≠ kFloat | UT-DeltaRule-020 |
| G not kFloat | G.Kind ≠ kFloat | UT-DeltaRule-010 |
| M not 3-D | len(M.Shape) ≠ 3 | UT-DeltaRule-011 |
| Q wrong shape | Q.Shape ≠ [H, Dk] | UT-DeltaRule-012 |
| K wrong shape | K.Shape ≠ [H, Dk] | UT-DeltaRule-013 |
| V wrong shape | V.Shape ≠ [H, Dv] | UT-DeltaRule-014 |
| Beta wrong shape | Beta.Shape ≠ [H] | UT-DeltaRule-015 |
| G wrong shape | G.Shape ≠ [H] | UT-DeltaRule-016 |

All path combinations with other arguments valid are covered.

### `DeltaRuleTokens` branches

| Branch | Condition | UT case |
|--------|-----------|---------|
| Q not kFloat | Q.Kind ≠ kFloat | UT-DeltaRule-109 |
| K not kFloat | K.Kind ≠ kFloat | UT-DeltaRule-118 |
| V not kFloat | V.Kind ≠ kFloat | UT-DeltaRule-119 |
| Beta not kFloat | Beta.Kind ≠ kFloat | UT-DeltaRule-120 |
| G not kFloat | G.Kind ≠ kFloat | UT-DeltaRule-110 |
| Q not 3-D | len(Q.Shape) ≠ 3 | UT-DeltaRule-111 |
| T = 0 | Q.Shape[1] = 0 | UT-DeltaRule-112 |
| K not same shape | K.Shape ≠ Q.Shape | UT-DeltaRule-113 |
| V wrong H | V.Shape[0] ≠ H | UT-DeltaRule-114 |
| V wrong T | V.Shape[1] ≠ T | UT-DeltaRule-115 |
| Beta wrong shape | Beta.Shape ≠ [H, T] | UT-DeltaRule-116 |
| G wrong shape | G.Shape ≠ [H, T] | UT-DeltaRule-117 |

Loop path (I from 0 to T-1) exercised by:
- T = 0: UT-DeltaRule-112 (error path)
- T = 1: UT-DeltaRule-105 (one iteration)
- T = 2: UT-DeltaRule-008 (two iterations via DeltaStep)
- T = 20, 65, 200: ITa cases (many iterations)
