# プログラムフロー — unit `Recompute` (T13)

## TSavedInputs.Add

```mermaid
flowchart TD
    Start([Add 呼び出し]) --> N[\"N := Length(FItems)\"]
    N --> Extend[\"SetLength(FItems, N + 1)\"]
    Extend --> Check{FOnHost?}
    Check -->|True| ToCpu[\"FItems[N] := X.detach.&to(TDevice.Cpu)\"]
    Check -->|False| NoCpu[\"FItems[N] := X.detach\"]
    ToCpu --> End([終了])
    NoCpu --> End
```

## TSavedInputs.Item

```mermaid
flowchart TD
    Start([Item 呼び出し]) --> Check{I < 0 または<br>I >= Count?}
    Check -->|True| RaiseError[\"raise ERecompute.CreateFmt<br>('saved input %d out of range (count %d)',<br>[I, Count])\"]
    Check -->|False| Return[\"Result := FItems[I]\"]
    RaiseError --> End([終了])
    Return --> End
```

## RecomputeForward

```mermaid
flowchart TD
    Start([RecomputeForward 呼び出し]) --> CheckStages{Stages = nil または<br>Length = 0?}
    CheckStages -->|True| RaiseNoStages["raise ERecompute.Create('no stages')"]
    CheckStages -->|False| CheckX0{X0.Defined?}
    CheckX0 -->|False| RaiseX0Undef["raise ERecompute.Create('X0 is undefined')"]
    CheckX0 -->|True| Clear["Saved.Clear"]
    Clear --> Guard["G := NoGradGuard"]
    Guard --> InitX["X := X0"]
    InitX --> InitI["I := 0"]
    InitI --> LoopCheck{I <= High(Stages)?}
    LoopCheck -->|True| ToDevice["X := X.&to(Stages[I].Device)"]
    ToDevice --> SaveInput["Saved.Add(X)"]
    SaveInput --> ForwardCall["X := Stages[I].Forward(X)"]
    ForwardCall --> IncI["I := I + 1"]
    IncI --> LoopCheck
    LoopCheck -->|False| ReturnResult["Result := X"]
    ReturnResult --> End([終了])
    RaiseNoStages --> End
    RaiseX0Undef --> End
```

## RecomputeBackward

```mermaid
flowchart TD
    Start([RecomputeBackward 呼び出し]) --> CheckCount{"Saved.Count <> Length(Stages)?"}
    CheckCount -->|True| RaiseCount["raise ERecompute.CreateFmt<br>('saved inputs: %d, stages: %d', ...)"]
    CheckCount -->|False| CheckGrad{GradOut.Defined?}
    CheckGrad -->|False| RaiseGradUndef["raise ERecompute.Create('GradOut is undefined')"]
    CheckGrad -->|True| InitG["G := GradOut"]
    InitG --> InitI["I := High(Stages)"]
    InitI --> LoopCheck{I >= 0?}
    LoopCheck -->|True| LoadInput["X := Saved.Item(I).&to(Stages[I].Device)<br>.detach.requires_grad_(True)"]
    LoadInput --> ForwardCall["Y := Stages[I].Forward(X)"]
    ForwardCall --> MoveGrad["G := G.&to(Stages[I].Device)"]
    MoveGrad --> CheckShape{"ShapeStr(Y) <> ShapeStr(G)?"}
    CheckShape -->|True| RaiseShape["raise ERecompute.CreateFmt<br>('stage %d: output %s, gradient %s', ...)"]
    CheckShape -->|False| BackwardStep["(Y * G).sum.Backward"]
    BackwardStep --> GetGrad["G := X.grad"]
    GetGrad --> DecI["I := I - 1"]
    DecI --> LoopCheck
    LoopCheck -->|False| ReturnResult["Result := G"]
    ReturnResult --> End([終了])
    RaiseCount --> End
    RaiseGradUndef --> End
    RaiseShape --> End
```
