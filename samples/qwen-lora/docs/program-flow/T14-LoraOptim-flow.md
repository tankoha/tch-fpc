# Program Flow — Unit `LoraOptim` (T14)

Flowcharts per function, from `../detailed-design/T14-detailed.md`. Every
branch is drawn out: they are the ground for the UT cases and the coverage
gate.

## `DefaultAdamWConfig`

```mermaid
flowchart TD
  A([enter]) --> B["Beta1 := 0.9<br/>Beta2 := 0.999<br/>Eps := 1e-8<br/>WeightDecay := 0.01"]
  B --> C([return the config])
```

## `ScheduleLR(Base, Update, Warmup, Total, MinRatio)`

```mermaid
flowchart TD
  A([enter]) --> B{"(Update < 1) or (Update > Total)"}
  B -- yes --> B1["raise ELoraOptim:<br/>update %d outside 1..%d<br/>(Update, Total)"]
  B -- no --> C{"(Warmup < 0) or (Warmup >= Total)"}
  C -- yes --> C1["raise ELoraOptim:<br/>warm-up %d outside 0..%d<br/>(Warmup, Total - 1)"]
  C -- no --> D{"Update <= Warmup"}
  D -- yes --> D1([return Base * Update / Warmup])
  D -- no --> E["P := (Update - Warmup) / (Total - Warmup)"]
  E --> F([return Base * (MinRatio + (1 - MinRatio) * 0.5 * (1 + Cos(Pi * P)))])
```

## `Create(Stores, AConfig)`

```mermaid
flowchart TD
  A([inherited Create]) --> B{"Length(Stores) = 0"}
  B -- yes --> B1["raise ELoraOptim:<br/>no stores"]
  B -- no --> C["FConfig := AConfig<br/>SetLength of FParams, FM, FV, FMoments to Length(Stores)"]
  C --> D["I := 0"]
  D --> E{"I <= High(Stores)"}
  E -- no --> N([return])
  E -- yes --> F["P := Stores[I].TrainableVariables"]
  F --> G{"P is empty (no trainable variables)"}
  G -- yes --> G1["raise ELoraOptim:<br/>store %d has no trainable variables (I)"]
  G -- no --> H["FParams[I] := P<br/>FMoments[I] := TVarStore.Create(Device, Kind)<br/>Root := FMoments[I].Root<br/>SetLength(FM[I], FV[I] to Length(P))"]
  H --> I1["J := 0"]
  I1 --> I2{"J <= High(P)"}
  I2 -- no --> K["Inc(I); I := I + 1"]
  I2 -- yes --> I3["FM[I][J] := Root.Sub('m').ZerosNoTrain(IntToStr(J), P[J].Shape)<br/>FV[I][J] := Root.Sub('v').ZerosNoTrain(IntToStr(J), P[J].Shape)"]
  I3 --> I4["J := J + 1"] --> I2
  K --> E
```

## `Destroy`

```mermaid
flowchart TD
  A([enter]) --> B["free every FMoments[I] (I = 0 to High(FMoments))"]
  B --> C([inherited Destroy])
```

## `ZeroGrad`

```mermaid
flowchart TD
  A([enter]) --> B["for I := 0 to High(FParams)<br/>for J := 0 to High(FParams[I])<br/>FParams[I][J].ZeroGrad"]
  B --> C([return])
```

## `StoreCount`

```mermaid
flowchart TD
  A([enter]) --> B([return Length(FMoments)])
```

## `Moments(I)`

```mermaid
flowchart TD
  A([enter]) --> B{"I < 0 or I > High(FMoments)"}
  B -- yes --> B1["raise ELoraOptim:<br/>store %d out of range (count %d)<br/>(I, Length(FMoments))"]
  B -- no --> C([return FMoments[I]])
```

## `ClipGlobalNorm(MaxNorm)`

```mermaid
flowchart TD
  A([enter]) --> B["G := NoGradGuard;  Sq := 0"]
  B --> C["for I := 0 to High(FParams)<br/>for J := 0 to High(FParams[I])"]
  C --> D{"Gr := FParams[I][J].grad<br/>Gr.Defined"}
  D -- yes --> E["Sq := Sq + Sqr(Gr.norm.DoubleValue([]))"]
  D -- no --> F["loop to next pair; after the last pair go to G"]
  E --> F
  F --> G["Result := Sqrt(Sq)<br/>Coef := MaxNorm / (Result + 1e-6)"]
  G --> H{"Coef < 1"}
  H -- no --> Z([return Result])
  H -- yes --> I["for I := 0 to High(FParams)<br/>for J := 0 to High(FParams[I])"]
  I --> J{"Gr := FParams[I][J].grad<br/>Gr.Defined"}
  J -- yes --> K["Gr.mul_scalar_(Coef)"]
  J -- no --> L["loop to next pair; after the last pair go to return"]
  K --> L
  L --> Z
```

## `Step(LR)`

```mermaid
flowchart TD
  A([enter]) --> B["G := NoGradGuard<br/>Inc(FStepCount)<br/>Bc1 := 1 - Power(FConfig.Beta1, FStepCount)<br/>Bc2 := 1 - Power(FConfig.Beta2, FStepCount)"]
  B --> C["for I := 0 to High(FParams)<br/>for J := 0 to High(FParams[I])"]
  C --> D["P := FParams[I][J]<br/>Gr := P.grad"]
  D --> E{"Gr.Defined"}
  E -- no --> N["loop to next pair; after the last pair go to return"]
  E -- yes --> F["M := FM[I][J];  V := FV[I][J]<br/>P.mul_scalar_(1 - LR * FConfig.WeightDecay)"]
  F --> G["M.mul_scalar_(FConfig.Beta1).add_(Gr * (1 - FConfig.Beta1))<br/>V.mul_scalar_(FConfig.Beta2).add_(Gr * Gr * (1 - FConfig.Beta2))"]
  G --> H["P.add_(M / (V.sqrt / Sqrt(Bc2) + FConfig.Eps) * (-LR / Bc1))"]
  H --> N
  N --> R([return])
```
