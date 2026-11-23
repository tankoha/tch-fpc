# Program Flow — Unit `TrainData` (T15)

Flowcharts per function, from `../detailed-design/T15-detailed.md`. Every
branch is drawn out: they are the ground for the UT cases and the coverage
gate. Message texts and conditions are the exact ones of the detailed
design (4.10).

## `SplitMix64` (local)

```mermaid
flowchart TD
  A([enter]) --> B["S := S + QWord($9E3779B97F4A7C15)<br/>Z := S<br/>Z := (Z xor (Z shr 30)) * QWord($BF58476D1CE4E5B9)<br/>Z := (Z xor (Z shr 27)) * QWord($94D049BB133111EB)"]
  B --> C([return Z xor (Z shr 31)])
```

## `StateOf` (local)

```mermaid
flowchart TD
  A([enter]) --> B([return QWord(ASeed) + QWord(AEpoch) * QWord($D1B54A32D192ED03) + QWord(APool) * QWord($9E3779B97F4A7C15)])
```

## `Shuffle` (local)

```mermaid
flowchart TD
  A([enter]) --> B["I := High(A)"]
  B --> C{"I >= 1"}
  C -- no --> D([return])
  C -- yes --> E["J := Integer(SplitMix64(AState) mod QWord(I + 1))<br/>swap A[I] and A[J]"]
  E --> F["I := I - 1"]
  F --> C
```

For an empty `A`, `High(A) = -1` and the loop does not run.

## `EncodeSample`

```mermaid
flowchart TD
  A([enter]) --> B["M := nil<br/>SetLength(M, 2)<br/>M[0] := ChatMsg(crUser, APrompt)<br/>M[1] := ChatMsg(crAssistant, AAnswer)<br/>O := DefaultChatOptions<br/>O.EnableThinking := False"]
  B --> C([return EncodeTraining(Tok, M, O, LossFrom, LossTo)<br/>(the loss span comes out of EncodeTraining, T07)])
```

## `HasWideEscape` (local)

```mermaid
flowchart TD
  A([enter]) --> B["Result := False<br/>I := 1"]
  B --> C{"I < Length(L)"}
  C -- no --> D([return Result])
  C -- yes --> E{"L[I] = '\\"}
  E -- yes --> F{"(L[I + 1] = 'u') and (I + 5 <= Length(L))<br/>and TryStrToInt('$' + Copy(L, I + 2, 4), V)<br/>and (V > 127)"}
  F -- yes --> G([return True])
  F -- no --> H["I := I + 2"]
  E -- no --> I1["I := I + 1"]
  H --> C
  I1 --> C
```

A backslash always skips two bytes: an escaped backslash is text and the
sequence `u00e9` after it does not re-trigger; only `\u` with four hex
digits of a value above 127 refuses.

## `GetStr` (local)

```mermaid
flowchart TD
  A([enter]) --> B["D := O.Find(AKey)"]
  B --> C{"(D = nil) or (D.JSONType <> jtString)"}
  C -- yes --> C1["raise ETrainData:<br/>line ALine: missing or bad AKey"]
  C -- no --> D([return D.AsString])
```

## `TTrainSet.Create`

```mermaid
flowchart TD
  A([inherited Create]) --> B{"AMaxLen < 1"}
  B -- yes --> B1["raise ETrainData:<br/>max length must be positive, got AMaxLen"]
  B -- no --> C{"not FileExists(AFile)"}
  C -- yes --> C1["raise ETrainData:<br/>no data file AFile"]
  C -- no --> D["L := TStringList.Create<br/>Ids := TStringList.Create<br/>(both freed in the final finally)"]
  D --> E["Ids.Sorted := True<br/>L.LoadFromFile(AFile)<br/>N := 0"]
  E --> F["I := 0"]
  F --> G{"I < L.Count"}
  G -- no --> Z["SetLength(FSamples, N)"]
  Z --> AA([finally: Ids.Free; L.Free<br/>return the created set])
  G -- yes --> H{"Trim(L[I]) = ''"}
  H -- yes --> N1["I := I + 1"]
  N1 --> F
  H -- no --> W{"HasWideEscape(L[I])"}
  W -- yes --> W1["raise ETrainData:<br/>line I + 1: a \\u escape of a non-ASCII character; write UTF-8"]
  W -- no --> P1["D := GetJSON(L[I], False)<br/>on E: EJSONParser -> raise ETrainData:<br/>line I + 1: not a JSON object"]
  P1 --> OB{"not (D is TJSONObject)"}
  OB -- yes --> OB1["raise ETrainData:<br/>line I + 1: not a JSON object"]
  OB -- no --> KEYS["O := TJSONObject(D)   (D freed in finally)<br/>S.Id := GetStr(O, 'id', I + 1)<br/>S.Dialect := GetStr(O, 'dialect', I + 1)<br/>Prompt := GetStr(O, 'prompt', I + 1)<br/>Answer := GetStr(O, 'answer', I + 1)<br/>Cls := GetStr(O, 'class', I + 1)<br/>(each GetStr raises on a missing or non-string key)"]
  KEYS --> DUP{"Ids.IndexOf(S.Id) >= 0"}
  DUP -- yes --> DUP1["raise ETrainData:<br/>line I + 1: duplicate id S.Id"]
  DUP -- no --> ADD["Ids.Add(S.Id)<br/>Admit := False<br/>for J := 0 to High(AClasses):<br/>if AClasses[J] = Cls then Admit := True"]
  ADD --> ADM{"not Admit"}
  ADM -- yes --> SKC["Inc(FSkippedClass)"]
  SKC --> N1
  ADM -- no --> ENC["S.Tokens := EncodeSample(Tok, Prompt, Answer, S.LossFrom, S.LossTo)"]
  ENC --> LEN{"Length(S.Tokens) > AMaxLen"}
  LEN -- yes --> SKL["Inc(FSkippedLong)"]
  SKL --> N1
  LEN -- no --> STORE["if N = Length(FSamples) then<br/>SetLength(FSamples, 2 * N + 16)<br/>FSamples[N] := S<br/>Inc(N)"]
  STORE --> N1
```

The `finally: D.Free` also covers the `Continue` taken after
`Inc(FSkippedClass)`; only `EJSONParser` out of `GetJSON` is converted to
`ETrainData`, other exceptions pass through.

## `TTrainSet.Count`

```mermaid
flowchart TD
  A([enter]) --> B([return Length(FSamples)])
```

## `TTrainSet.MaxLength`

```mermaid
flowchart TD
  A([enter]) --> B["Result := 0<br/>I := 0"]
  B --> C{"I <= High(FSamples)"}
  C -- no --> D([return Result])
  C -- yes --> E["Result := Max(Result, Length(FSamples[I].Tokens))<br/>I := I + 1"]
  E --> C
```

For an empty set `High(FSamples) = -1`, the loop does not run and the
result is 0.

## `TTrainSet.GetSample`

```mermaid
flowchart TD
  A([enter]) --> B{"(I < 0) or (I >= Length(FSamples))"}
  B -- yes --> B1["raise ETrainData:<br/>sample I outside 0..Length(FSamples) - 1"]
  B -- no --> C([return FSamples[I]])
```

## `EpochOrder`

```mermaid
flowchart TD
  A([enter]) --> B["Result := nil<br/>SetLength(Result, N)<br/>for I := 0 to N - 1: Result[I] := I"]
  B --> C["Shuffle(Result, StateOf(ASeed, AEpoch, 0))"]
  C --> D([return Result])
```

## `EpochBatches`

```mermaid
flowchart TD
  A([enter]) --> B{"APool < 1"}
  B -- yes --> B1["raise ETrainData:<br/>pool must be positive, got APool"]
  B -- no --> C{"ABudget < ASet.MaxLength"}
  C -- yes --> C1["raise ETrainData:<br/>budget ABudget is below the longest sample, ASet.MaxLength"]
  C -- no --> D["Result := nil<br/>NB := 0<br/>Order := EpochOrder(ASet.Count, ASeed, AEpoch)<br/>P := 0"]
  D --> E{"P * APool < Length(Order)"}
  E -- no --> Z([return Result])
  E -- yes --> POOL["Lo := P * APool<br/>Hi := Min(Lo + APool, Length(Order)) - 1<br/>Pool := Copy(Order, Lo, Hi - Lo + 1)<br/>Len[I] := Length(ASet[Pool[I]].Tokens) for all I<br/>insertion sort of Pool and Len together:<br/>while J > 0 and Len[J - 1] > Len[J]:<br/>swap Len and Pool, J := J - 1<br/>(stable: ties keep the epoch's order)"]
  POOL --> CUR["PoolB := nil<br/>Cur := nil<br/>I := 0"]
  CUR --> H{"I <= High(Pool)"}
  H -- no --> LAST["SetLength(PoolB, Length(PoolB) + 1)<br/>PoolB[High(PoolB)] := Cur   (the last batch)"]
  H -- yes --> FIT{"(Length(Cur) + 1) * Len[I] > ABudget"}
  FIT -- yes --> FLUSH["SetLength(PoolB, Length(PoolB) + 1)<br/>PoolB[High(PoolB)] := Cur<br/>Cur := nil"]
  FLUSH --> ADD["SetLength(Cur, Length(Cur) + 1)<br/>Cur[High(Cur)] := Pool[I]"]
  FIT -- no --> ADD
  ADD --> NXT["I := I + 1"]
  NXT --> H
  LAST --> SHUF["Perm[K] := K for all K<br/>Shuffle(Perm, StateOf(ASeed, AEpoch, P + 1))<br/>append PoolB[Perm[K]] to Result for all K<br/>NB := NB + Length(PoolB)"]
  SHUF --> ADV["P := P + 1"]
  ADV --> E
```

The budget test `>` is strict: a batch whose count × longest equals
`ABudget` is admitted. For an empty set the `while` does not run and no
batches are returned.

## `SkipBatches`

```mermaid
flowchart TD
  A([enter]) --> B{"APosition < 0"}
  B -- yes --> B1["raise ETrainData:<br/>position APosition is negative"]
  B -- no --> C["I := 0<br/>Done := 0"]
  C --> D{"(Done < APosition) and (I < Length(AB))"}
  D -- yes --> E["Done := Done + Length(AB[I])<br/>I := I + 1"]
  E --> D
  D -- no --> F{"Done <> APosition"}
  F -- yes --> F1["raise ETrainData:<br/>position APosition is not at the end of a batch<br/>(also for a position beyond the last batch)"]
  F -- no --> G([return Copy(AB, I, Length(AB) - I)<br/>(APosition = 0 gives AB)])
```

## `BatchTensors`

```mermaid
flowchart TD
  A([enter]) --> B["B := Length(ABatch)"]
  B --> C{"B = 0"}
  C -- yes --> C1["raise ETrainData:<br/>empty batch"]
  C -- no --> D["T := 0<br/>for I := 0 to B - 1:<br/>T := Max(T, Length(ASet[ABatch[I]].Tokens))<br/>(an out-of-range index raises in GetSample:<br/>sample I outside 0..N - 1)"]
  D --> AL["Ids := nil; Tg := nil; M := nil<br/>SetLength(Ids, B * T)<br/>SetLength(Tg, B * T)<br/>SetLength(M, B * T)"]
  AL --> FI["I := 0"]
  FI --> FB{"I < B"}
  FB -- no --> TENS["AIds := TTensor.FromArray(Ids).view([B, T])<br/>ATargets := TTensor.FromArray(Tg).view([B, T])<br/>AMask := TTensor.FromArray(M).view([B, T])"]
  TENS --> Z([return; the three tensors int64, int64, float])
  FB -- yes --> ROW["S := ASet[ABatch[I]]<br/>N := Length(S.Tokens)<br/>J := 0"]
  ROW --> FC{"J < T"}
  FC -- no --> NI["I := I + 1"]
  NI --> FB
  FC -- yes --> IDK{"J < N"}
  IDK -- yes --> ID1["Ids[I * T + J] := S.Tokens[J]"]
  IDK -- no --> ID2["Ids[I * T + J] := APad"]
  ID1 --> TGK{"J + 1 < N"}
  ID2 --> TGK
  TGK -- yes --> TG1["Tg[I * T + J] := S.Tokens[J + 1]"]
  TGK -- no --> TG2["Tg[I * T + J] := APad"]
  TG1 --> MSK{"(J + 1 >= S.LossFrom) and (J + 1 < S.LossTo)"}
  TG2 --> MSK
  MSK -- yes --> MS1["M[I * T + J] := 1"]
  MSK -- no --> MS2["M[I * T + J] := 0"]
  MS1 --> NJ["J := J + 1"]
  MS2 --> NJ
  NJ --> FC
```
