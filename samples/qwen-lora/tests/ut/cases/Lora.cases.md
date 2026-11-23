# UT cases — unit `Lora` (T12)

Context (T12-facts): S is a `TVarStore` on the CPU. `NewPair(S)` is the pair
`p`, 8 -> 6, rank 3, scale 2 (the caller frees the pair, then S); `FixPair(P)`
sets A[k, c] = 0.1 (k + 1) − 0.05 c, B[o, k] = 0.03 (o − k); `FixX(Rows, Cols)`
sets X[t, c] = sin(1 + t Cols + c); `RefDelta(P, X)` is s·((X·Aᵀ)·Bᵀ) in Double.
`NewSet` is a set of rank 4, scale 2 with pairs `blk.0.attn_qkv` 64 -> 128 and
`blk.1.attn_q` 64 -> 256. TmpDir is a fresh directory ending in the path
delimiter (rows 108 to 113). Error rows catch the named exception class and
AssertEquals the whole message; a row fails if no exception comes. A tolerance
row passes when the difference is ≤ the limit; "exact" means equal strings or a
difference of 0.

Stage 1 (task T12b1): `LoraPath`, `TLoraPair` (Create, Delta, MergedDelta); rows
UT-Lora-001 to 016. Stage 2 (task T12b2): `TLoraSet` (all of it); rows
UT-Lora-101 to 114.

| Case ID | Title | Target function | Input (short) | Expected result | Notes |
|---|---|---|---|---|---|
| UT-Lora-001 | LoraPath, three name segments | LoraPath | `LoraPath(S.Root, 'blk.3.attn_q').FullName('lora_a')` (S a CPU store) | `blk.3.attn_q.lora_a` | |
| UT-Lora-002 | LoraPath, single name segment | LoraPath | `LoraPath(S.Root, 'x').FullName('y')` | `x.y` | |
| UT-Lora-003 | LoraPath, empty segment in the name | LoraPath | `LoraPath(S.Root, 'blk..q')` | ETorchError `sub name cannot be empty or contain "." or "|": ""` | |
| UT-Lora-004 | Create, registers lora_a and lora_b | TLoraPair.Create | `P := NewPair(S)` | Name `p`, InDim 8, OutDim 6, Rank 3, Scale 2; `ShapeText(A)` `[3, 8]`, `ShapeText(B)` `[6, 3]`; both `RequiresGrad`; `S.Variables` 2: `p.lora_a`, `p.lora_b`; 2 trainable | |
| UT-Lora-005 | Create, A's bounds under the seed | TLoraPair.Create | `ManualSeed(1)`; `NewPair(S)` | max \|B\| 0 (exact); max \|A\| ≤ 1/√8 and > 0.3 (measured 0.33284819) | |
| UT-Lora-006 | Create, InDim 0 | TLoraPair.Create | `TLoraPair.Create(S.Root / 'p', 'p', 0, 6, 3, 2.0)` | ELora `p: dimensions must be positive, got 0 -> 6`; `S.Count` 0 | |
| UT-Lora-007 | Create, OutDim −1 | TLoraPair.Create | the same with 8, -1 | ELora `p: dimensions must be positive, got 8 -> -1`; `S.Count` 0 | |
| UT-Lora-008 | Delta, B zero | TLoraPair.Delta | `P.Delta(FixX(5, 8))`, B zero | `ShapeText` `[5, 6]`; max \|·\| 0 (exact) | |
| UT-Lora-009 | Delta, 2-D X vs the reference | TLoraPair.Delta | `FixPair(P)`; X := `FixX(5, 8)`; `P.Delta(X)` | vs `RefDelta(P, X)` ≤ 1e-6 (measured 6.0e-9) | |
| UT-Lora-010 | Delta, 3-D X vs the reference | TLoraPair.Delta | `FixPair(P)`; X := `FixX(6, 8)`; `P.Delta(X.view([2, 3, 8]))` | `ShapeText` `[2, 3, 6]`; vs `RefDelta(P, X)` ≤ 1e-6 (measured 6.0e-9) | |
| UT-Lora-011 | Delta, last dimension wrong | TLoraPair.Delta | `P.Delta(TTensor.zeros([5, 7]))` | ELora `p: X must end in 8` | |
| UT-Lora-012 | Delta, 0-dim X | TLoraPair.Delta | `P.Delta(TTensor.zeros([5, 7]).sum(kNone))` (0-dim) | ELora `p: X must end in 8` | |
| UT-Lora-013 | MergedDelta | TLoraPair.MergedDelta | `FixPair(P)`; M := `P.MergedDelta` | `ShapeText(M)` `[6, 8]`; `RequiresGrad` False; Kind kFloat; vs 2·B·A in Double ≤ 1e-6 (measured 6.3e-9) | |
| UT-Lora-014 | Delta equals X·MergedDeltaᵀ | TLoraPair.Delta / MergedDelta | `FixPair(P)`; X := `FixX(5, 8)` | `X.matmul(P.MergedDelta.t)` vs `P.Delta(X)` ≤ 1e-5 (measured 2.2e-8) | |
| UT-Lora-015 | Delta, backward with B zero | TLoraPair.Delta | `ManualSeed(1)`; `NewPair(S)` (B zero); `P.Delta(FixX(5, 8)).sum(kNone).Backward` | max \|A.grad\| 0 (exact); B.grad[o, k] vs 2·Σₜ (X·Aᵀ)[t, k] in Double ≤ 1e-5 (measured 6.6e-8) | |
| UT-Lora-016 | Delta, X kind kDouble | TLoraPair.Delta | `FixPair(P)`; X := `FixX(5, 8).totype(kDouble)`; `P.Delta(X)` | Kind kDouble; `P.A.Kind` still kFloat; vs `RefDelta(P, X)` ≤ 1e-12 (measured 6.9e-18) | |
| UT-Lora-101 | Set, empty create | TLoraSet.Create | `TLoraSet.Create(4, 2.0)` | Rank 4, Scale 2, Count 0, `Length(Stores)` 0, ParameterCount 0 | |
| UT-Lora-102 | Set, rank 0 | TLoraSet.Create | `TLoraSet.Create(0, 2.0)` | ELora `rank must be positive, got 0` | |
| UT-Lora-103 | Set, two pairs on the CPU | TLoraSet.Create / Add | `S := NewSet` | Count 2; `Pair(0).Name` `blk.0.attn_qkv`, `Pair(1).Name` `blk.1.attn_q`; `Pair(1)`: Rank 4, Scale 2, OutDim 256; `Length(Stores)` 1, `Stores[0].Device.ToString` `cpu`; `Stores[0].Variables` 4: `blk.0.attn_qkv.lora_a`, `blk.0.attn_qkv.lora_b`, `blk.1.attn_q.lora_a`, `blk.1.attn_q.lora_b`; ParameterCount 2048 | |
| UT-Lora-104 | Add, name already present | TLoraSet.Add | `S.Add('blk.0.attn_qkv', 64, 128, TDevice.Cpu)` on `NewSet` | ELora `pair already present: blk.0.attn_qkv`; Count 2 | |
| UT-Lora-105 | Find is literal; Add upper-case name | TLoraSet.Find / Add | `NewSet`: `Find('blk.1.attn_q')`, `Find('BLK.1.ATTN_Q')`, `Find('')`; then `Add('BLK.1.ATTN_Q', 64, 256, Cpu)` | `Pair(1)`, nil, nil; then Count 3 and `Find('BLK.1.ATTN_Q')` = `Pair(2)` | |
| UT-Lora-106 | Pair, indices outside 0..1 | TLoraSet.Pair | `NewSet`: `Pair(2)`, `Pair(-1)` | ELora `pair 2 outside 0..1`, `pair -1 outside 0..1` | |
| UT-Lora-107 | Pair(0) on the empty set | TLoraSet.Pair | `TLoraSet.Create(4, 2.0).Pair(0)` | ELora `pair 0 outside 0..-1` | |
| UT-Lora-108 | Save, then Load into a fresh set | TLoraSet.Save / Load | `ManualSeed(3)`; S, S2 := `NewSet`; S's B := randn; `S.Save(TmpDir + 'a.ot')`; `S2.Load(..)` | the file exists; every A and B of S2 equals S's (exact) | |
| UT-Lora-109 | Load, the set has rank 3 | TLoraSet.Load | S := `NewSet`, S2 the same names at rank 3; `S.Save(TmpDir + 'b.ot')`; `S2.Load(..)` | ETorchNativeError `The size of tensor a (3) must match the size of tensor b (4) at non-singleton dimension 0` | |
| UT-Lora-110 | Load, missing file | TLoraSet.Load | `NewSet.Load(TmpDir + 'missing.ot')` | ETorchNativeError `'open file failed because of errno 2 on fopen: No such file or directory, file path: ' + TmpDir + 'missing.ot'` | |
| UT-Lora-111 | Load, a pair only in the loaded set | TLoraSet.Load | S2 := `NewSet` plus `blk.2.attn_qkv` (64 -> 128); `S.Save(TmpDir + 'c.ot')` from `NewSet`; `S2.Load(..)` | ETorchError `'variable blk.2.attn_qkv.lora_a not found in ' + TmpDir + 'c.ot'` | |
| UT-Lora-112 | Save and Load with a bare relative name | TLoraSet.Save / Load | `ManualSeed(4)`; S, S2 := `NewSet`; `SetCurrentDir(TmpDir)`; `S.Save('rel.ot')`; `S2.Load('rel.ot')`; the old directory restored in `finally` | `TmpDir + 'rel.ot'` exists; `Pair(1).A` equal (exact) | |
| UT-Lora-113 | Load keeps handles and grad flags | TLoraSet.Load | S, S2 := `NewSet`; A0 := `S2.Pair(0).A`; `S.Save(TmpDir + 'd.ot')`; `S2.Load(..)` | `S2.Pair(0).A.SameHandle(A0)`; A and B still `RequiresGrad`; A0 equals S's A (exact) | |
| UT-Lora-114 | Add, empty name | TLoraSet.Add | `TLoraSet.Create(4, 2.0).Add('', 64, 128, Cpu)` | ETorchError `sub name cannot be empty or contain "." or "|": ""` | |
