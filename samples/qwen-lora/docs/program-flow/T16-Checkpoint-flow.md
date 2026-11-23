# Program Flow — Unit `Checkpoint` (T16)

Flowcharts per function, from `../detailed-design/T16-detailed.md`. Every
branch is drawn out: they are the ground for the UT cases and the coverage
gate. Message texts and conditions are the exact ones of the detailed
design (4.7).

## `NewRunState`

```mermaid
flowchart TD
  A([enter]) --> B["Result.Update := 0<br/>Result.Epoch := 0<br/>Result.Position := 0<br/>Result.Seed := 0<br/>Result.Rank := 0<br/>Result.Scale := 0<br/>Result.ConfigHash := ''<br/>Result.BaseSha256 := ''<br/>Result.ValLoss := NaN"]
  B --> C([return Result])
```

## `CheckAscii(AKey, AValue)` (local)

```mermaid
flowchart TD
  A([enter]) --> B["I := 1"]
  B --> C{"I <= Length(AValue)"}
  C -- no --> E([return])
  C -- yes --> D{"(AValue[I] < #32) or (AValue[I] > #126)"}
  D -- yes --> D1["raise ECheckpoint:<br/>AKey + ' is not printable ASCII'"]
  D -- no --> F["I := I + 1"]
  F --> C
```

## `RunStateToJson(S)`

```mermaid
flowchart TD
  A([enter]) --> B["CheckAscii('config_hash', S.ConfigHash)"]
  B --> C["CheckAscii('base_sha256', S.BaseSha256)"]
  C --> D["O := TJSONObject.Create"]
  D --> E["O.Add('update', S.Update)<br/>O.Add('epoch', S.Epoch)<br/>O.Add('position', S.Position)<br/>O.Add('seed', S.Seed)<br/>O.Add('rank', S.Rank)<br/>O.Add('scale', S.Scale)<br/>O.Add('config_hash', S.ConfigHash)<br/>O.Add('base_sha256', S.BaseSha256)"]
  E --> F{"IsNan(S.ValLoss) or IsInfinite(S.ValLoss)"}
  F -- yes --> G["O.Add('val_loss', TJSONNull.Create)"]
  F -- no --> H["O.Add('val_loss', S.ValLoss)"]
  G --> I["Result := O.FormatJSON + LineEnding"]
  H --> I
  I --> J(["finally: O.Free<br/>return Result"])
```

## `GetInt(O, AKey)` (local)

```mermaid
flowchart TD
  A([enter]) --> B["D := O.Find(AKey)"]
  B --> C{"(D = nil) or not ((D is TJSONIntegerNumber) or (D is TJSONInt64Number))"}
  C -- yes --> C1["raise ECheckpoint:<br/>state.json: missing or bad <key>"]
  C -- no --> D([return D.AsInt64])
```

## `GetFloat(O, AKey, AllowNull)` (local)

```mermaid
flowchart TD
  A([enter]) --> B["D := O.Find(AKey)"]
  B --> C{"(D <> nil) and AllowNull and (D.JSONType = jtNull)"}
  C -- yes --> Z([return NaN])
  C -- no --> D{"(D = nil) or (D.JSONType <> jtNumber)"}
  D -- yes --> D1["raise ECheckpoint:<br/>state.json: missing or bad <key>"]
  D -- no --> E([return D.AsFloat])
```

## `GetStr(O, AKey)` (local)

```mermaid
flowchart TD
  A([enter]) --> B["D := O.Find(AKey)"]
  B --> C{"(D = nil) or (D.JSONType <> jtString)"}
  C -- yes --> C1["raise ECheckpoint:<br/>state.json: missing or bad <key>"]
  C -- no --> D([return D.AsString])
```

## `RunStateFromJson(AText)`

```mermaid
flowchart TD
  A([enter]) --> B["D := GetJSON(AText)"]
  B -. "on E: EJSONParser" .-> B1["raise ECheckpoint:<br/>state.json: not a JSON object"]
  B --> C{"not (D is TJSONObject)"}
  C -- yes --> C1["raise ECheckpoint:<br/>state.json: not a JSON object"]
  C -- no --> D["O := TJSONObject(D)<br/>Result.Update := GetInt(O, 'update')<br/>Result.Epoch := GetInt(O, 'epoch')<br/>Result.Position := GetInt(O, 'position')<br/>Result.Seed := GetInt(O, 'seed')<br/>Result.Rank := GetInt(O, 'rank')<br/>Result.Scale := GetFloat(O, 'scale', False)<br/>Result.ConfigHash := GetStr(O, 'config_hash')<br/>Result.BaseSha256 := GetStr(O, 'base_sha256')<br/>Result.ValLoss := GetFloat(O, 'val_loss', True)"]
  D --> E(["finally: D.Free<br/>return Result"])
```

## `ReadText(AFile)` (local)

```mermaid
flowchart TD
  A([enter]) --> B["L := TStringList.Create"]
  B --> C["L.LoadFromFile(AFile)"]
  C --> D["Result := L.Text"]
  D --> E(["finally: L.Free"])
```

## `WriteText(AFile, AText)` (local)

```mermaid
flowchart TD
  A([enter]) --> B["L := TStringList.Create"]
  B --> C["L.Text := AText"]
  C --> D["L.SaveToFile(AFile)"]
  D --> E(["finally: L.Free"])
```

## `ReadRunState(ADir)`

```mermaid
flowchart TD
  A([enter]) --> B["F := IncludeTrailingPathDelimiter(ADir) + StateFile"]
  B --> C{"not FileExists(F)"}
  C -- yes --> C1["raise ECheckpoint:<br/>not a checkpoint: ADir (as given)"]
  C -- no --> D([return RunStateFromJson(ReadText(F))])
```

## `CheckOptim(ASet, AOptim)` (local)

```mermaid
flowchart TD
  A([enter]) --> B["St := ASet.Stores"]
  B --> C{"Length(St) <> AOptim.StoreCount"}
  C -- yes --> C1["raise ECheckpoint:<br/>the optimiser has %d stores, the set %d<br/>(AOptim.StoreCount, Length(St))"]
  C -- no --> D["I := 0"]
  D --> E{"I <= High(St)"}
  E -- no --> F([return])
  E -- yes --> G{"AOptim.Moments(I).Count <> 2 * St[I].Count"}
  G -- yes --> G1["raise ECheckpoint:<br/>store I: the optimiser has %d moments, the set %d variables<br/>(AOptim.Moments(I).Count, St[I].Count)"]
  G -- no --> H["I := I + 1"]
  H --> E
```

## `MomentsOf(AOptim, I, J, M, V)` (local)

```mermaid
flowchart TD
  A([enter]) --> B["AOptim.Moments(I).TryGet('m.' + IntToStr(J), M)"]
  B --> C["AOptim.Moments(I).TryGet('v.' + IntToStr(J), V)"]
  C --> D([return; M, V are the out moments])
```

## `SaveMoments(AFile, ASet, AOptim)` (local)

```mermaid
flowchart TD
  A([enter]) --> B["G := NoGradGuard<br/>St := ASet.Stores<br/>S := TVarStore.Create(TDevice.Cpu)"]
  B --> C["I := 0"]
  C --> D{"I <= High(St)"}
  D -- no --> L1["S.Save(AFile)"]
  D -- yes --> E["Vars := St[I].Variables<br/>J := 0"]
  E --> F{"J <= High(Vars)"}
  F -- no --> G1["I := I + 1"]
  F -- yes --> H["MomentsOf(AOptim, I, J, M, V)<br/>P := LoraPath(S.Root, Vars[J].Name)<br/>P.Add('adam_m', M.&to(TDevice.Cpu).detach.Copy, False)<br/>P.Add('adam_v', V.&to(TDevice.Cpu).detach.Copy, False)"]
  H --> I1["J := J + 1"]
  I1 --> F
  G1 --> C
  L1 --> M1(["finally: S.Free"])
```

## `LoadMoments(AFile, ASet, AOptim)` (local)

```mermaid
flowchart TD
  A([enter]) --> B["St := ASet.Stores<br/>S := TVarStore.Create(TDevice.Cpu)"]
  B --> C["I := 0"]
  C --> D{"I <= High(St)"}
  D -- no --> L1["S.Load(AFile)<br/>G := NoGradGuard"]
  D -- yes --> E["Vars := St[I].Variables<br/>J := 0"]
  E --> F{"J <= High(Vars)"}
  F -- no --> G1["I := I + 1"]
  F -- yes --> H["P := LoraPath(S.Root, Vars[J].Name)<br/>P.ZerosNoTrain('adam_m', Vars[J].Tensor.Shape)<br/>P.ZerosNoTrain('adam_v', Vars[J].Tensor.Shape)"]
  H --> I1["J := J + 1"]
  I1 --> F
  G1 --> C
  L1 --> K["for I := 0 to High(St), for J := 0 to High(St[I].Variables):<br/>MomentsOf(AOptim, I, J, M, V)<br/>S.TryGet(Vars[J].Name + '.adam_m', Fm)<br/>S.TryGet(Vars[J].Name + '.adam_v', Fv)<br/>M.CopyFrom(Fm)<br/>V.CopyFrom(Fv)"]
  K --> M1(["finally: S.Free"])
```

## `TCheckpoints.Create(ARoot, AKeep)`

```mermaid
flowchart TD
  A([inherited Create]) --> B{"AKeep < 1"}
  B -- yes --> B1["raise ECheckpoint:<br/>keep must be at least 1, got AKeep"]
  B -- no --> C["FRoot := IncludeTrailingPathDelimiter(ARoot)<br/>FKeep := AKeep"]
  C --> D{"not ForceDirectories(FRoot)"}
  D -- yes --> D1["raise ECheckpoint:<br/>cannot create FRoot (with the delimiter)"]
  D -- no --> E([return])
```

## `TCheckpoints.DirOf(AUpdate)` (private)

```mermaid
flowchart TD
  A([enter]) --> B([return FRoot + Format('ckpt-%.8d', AUpdate)])
```

## `TCheckpoints.RemoveTree(ADir)` (private)

```mermaid
flowchart TD
  A([enter]) --> B["D := IncludeTrailingPathDelimiter(ADir)<br/>FindFirst(D + '*', faAnyFile, R)"]
  B --> C{"first entry found (= 0)"}
  C -- no --> H["RemoveDir(ADir)"]
  C -- yes --> D{"R.Name = '.' or R.Name = '..'"}
  D -- yes --> F["loop: FindNext(R)"]
  D -- no --> E{"(R.Attr and faDirectory) <> 0"}
  E -- yes --> F1["RemoveTree(D + R.Name)"]
  E -- no --> G["DeleteFile(D + R.Name)"]
  F1 --> F
  G --> F
  F --> C2{"FindNext(R) = 0 (more entries)"}
  C2 -- yes --> D
  C2 -- no --> H2(["finally: FindClose(R)"])
  H2 --> H
```

## `TCheckpoints.List`

```mermaid
flowchart TD
  A([enter]) --> B["L := TStringList.Create<br/>FindFirst(FRoot + 'ckpt-*', faDirectory, R)"]
  B --> C{"first entry found (= 0)"}
  C -- no --> N1["L.Sort"]
  C -- yes --> D{"(R.Attr and faDirectory) <> 0<br/>and Length(R.Name) = 13<br/>and TryStrToInt64(Copy(R.Name, 6, 8), N)<br/>and N >= 0<br/>and FileExists(FRoot + R.Name + PathDelim + StateFile)"}
  D -- yes --> E["L.Add(R.Name)"]
  D -- no --> F["loop: FindNext(R)"]
  E --> F
  F --> C2{"FindNext(R) = 0 (more entries)"}
  C2 -- yes --> D
  C2 -- no --> G(["finally: FindClose(R)"])
  G --> N1
  N1 --> N2["SetLength(Result, L.Count)<br/>for I := 0 to L.Count - 1: Result[I] := FRoot + L[I]"]
  N2 --> N3(["finally: L.Free<br/>return Result (full paths, ascending)"])
```

## `TCheckpoints.Latest`

```mermaid
flowchart TD
  A([enter]) --> B["A := List"]
  B --> C{"Length(A) = 0"}
  C -- yes --> Z([return ''])
  C -- no --> D([return A[High(A)]])
```

## `TCheckpoints.Best`

```mermaid
flowchart TD
  A([enter]) --> B["Result := ''<br/>BestV := Infinity<br/>A := List"]
  B --> C["I := 0"]
  C --> D{"I <= High(A)"}
  D -- no --> F([return Result; '' when no finite loss])
  D -- yes --> E["V := ReadRunState(A[I]).ValLoss"]
  E --> G{"(not IsNan(V)) and (V < BestV)"}
  G -- yes --> H["BestV := V<br/>Result := A[I]"]
  G -- no --> I1["I := I + 1 (on a tie the earlier checkpoint stays)"]
  H --> I1
  I1 --> C
```

## `TCheckpoints.Prune` (private)

```mermaid
flowchart TD
  A([enter]) --> B["A := List<br/>B := Best"]
  B --> C["I := 0"]
  C --> D{"I <= High(A) - FKeep"}
  D -- no --> F([return])
  D -- yes --> E{"A[I] <> B"}
  E -- yes --> G["RemoveTree(A[I])"]
  E -- no --> H["I := I + 1"]
  G --> H
  H --> C
```

When `High(A) - FKeep < 0` the loop bound is below the start and nothing is
deleted (Pascal `for` does not run).

## `TCheckpoints.Save(AState, ASet, AOptim)`

```mermaid
flowchart TD
  A([enter]) --> B["CheckOptim(ASet, AOptim)"]
  B --> C["Result := DirOf(AState.Update)"]
  C --> D{"DirectoryExists(Result)"}
  D -- yes --> D1["raise ECheckpoint:<br/>checkpoint exists: Result"]
  D -- no --> E["leftovers of an interrupted save:<br/>for each tmp-ckpt-* entry:<br/>directory -> RemoveTree(FRoot + R.Name)<br/>file -> DeleteFile(FRoot + R.Name)"]
  E --> F["T := FRoot + 'tmp-' + ExtractFileName(Result)"]
  F --> G{"not ForceDirectories(T)"}
  G -- yes --> G1["raise ECheckpoint:<br/>cannot create T"]
  G -- no --> H["S := AState<br/>S.Rank := ASet.Rank<br/>S.Scale := ASet.Scale"]
  H --> I["ASet.Save(T + PathDelim + LoraFile)"]
  I --> J["SaveMoments(T + PathDelim + MomentFile, ASet, AOptim)"]
  J --> K["WriteText(T + PathDelim + StateFile, RunStateToJson(S))"]
  K --> L{"not RenameFile(T, Result)"}
  L -- yes --> L1["raise ECheckpoint:<br/>cannot rename T<br/>(the temporary directory stays until the next save)"]
  L -- no --> M["Prune"]
  M --> N([return Result])
```

## `TCheckpoints.Load(ADir, ASet, AOptim)`

```mermaid
flowchart TD
  A([enter]) --> B["Result := ReadRunState(ADir)"]
  B --> C{"(Result.Rank <> ASet.Rank) or (Result.Scale <> ASet.Scale)"}
  C -- yes --> C1["raise ECheckpoint:<br/>ADir: rank Result.Rank scale Result.Scale, the set rank ASet.Rank scale ASet.Scale"]
  C -- no --> D["CheckOptim(ASet, AOptim)"]
  D --> E["D := IncludeTrailingPathDelimiter(ADir)"]
  E --> F["ASet.Load(D + LoraFile)"]
  F --> G["LoadMoments(D + MomentFile, ASet, AOptim)"]
  G --> H["AOptim.StepCount := Result.Update"]
  H --> I([return Result])
```
