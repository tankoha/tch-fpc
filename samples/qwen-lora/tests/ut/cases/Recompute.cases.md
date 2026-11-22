# UT cases — unit `Recompute` (T13)

| Case ID | Title | Target function | Input (short) | Expected result | Notes |
|---|---|---|---|---|---|
| UT-Recompute-001 | Create SavedInputs OnHost False | TSavedInputs.Create | `TSavedInputs.Create(False)` | `OnHost` False, `Count` 0 | |
| UT-Recompute-002 | Create SavedInputs OnHost True | TSavedInputs.Create | `TSavedInputs.Create(True)` | `OnHost` True, `Count` 0 | |
| UT-Recompute-003 | Add tensors without OnHost | TSavedInputs.Add | `Create(False)`; `Add(X)` with X = `FromArray([1.0, 2]).requires_grad_(True)`; `Add(FromArray([3.0]))` | `Count` 2; `ValuesText(Item(0))` `'1 2'`, `Item(0).RequiresGrad` False, `Item(0).SameHandle(X)` False; `ValuesText(Item(1))` `'3'` | |
| UT-Recompute-004 | Add tensors with OnHost | TSavedInputs.Add | `Create(True)`; `Add(FromArray([1.0, 2]))` | `ValuesText(Item(0))` `'1 2'`; `Item(0).Device.ToString` `'cpu'` | |
| UT-Recompute-005 | Clear clears items | TSavedInputs.Clear | `Create(True)`; two `Add`; `Clear` | `Count` 0 | |
| UT-Recompute-006 | Item out of range (high) | TSavedInputs.Item | two `Add`; `Item(2)` | `saved input 2 out of range (count 2)` | |
| UT-Recompute-007 | Item out of range (negative) | TSavedInputs.Item | two `Add`; `Item(-1)` | `saved input -1 out of range (count 2)` | |
| UT-Recompute-008 | Forward two scale stages | RecomputeForward | `Hand`; `RecomputeForward(St, FromArray([1.0, 2]), Saved)` (`Saved` = `Create(False)`) | result `'6 12'`, `RequiresGrad` False; `Count` 2, items `'1 2'` and `'2 4'`; `Calls` 1 and 1; `LastRequiresGrad` False and False | |
| UT-Recompute-009 | Forward clears Saved | RecomputeForward | `Saved` with three `Add` of `FromArray([1.0])`; forward of 008 | `Count` 2 (the forward clears first) | |
| UT-Recompute-010 | Forward four affine stages | RecomputeForward | `MakeStages(51, 4, 8, kDouble, [TDevice.Cpu])`; `ManualSeed(52)`; X = `TTensor.randn([3, 8], kDouble)`; forward | every stage `Calls` 1 and `LastRequiresGrad` False | |
| UT-Recompute-011 | Forward matches plain chain | RecomputeForward | as 010, and `CloneStages(A, kDouble, [TDevice.Cpu])` made before the forward; the plain chain `B[0..3].Forward` from X | result shape `'[3, 8]'`; exact: `MaxAbsDiff(result, chain)` = 0 | |
| UT-Recompute-012 | No stages error | RecomputeForward | `RecomputeForward(nil, FromArray([1.0]), Saved)` | `no stages` | |
| UT-Recompute-013 | X0 undefined error | RecomputeForward | `Hand`; `RecomputeForward(St, Default(TTensor), Saved)` | `X0 is undefined` | |
| UT-Recompute-101 | Backward two scale stages | RecomputeBackward | forward of 008, then `RecomputeBackward(St, Saved, FromArray([1.0, 1]))` | result `'6 6'`; `S0.W.grad` `'9'`, `S1.W.grad` `'6'`; `Calls` 2 and 2; `LastRequiresGrad` True and True | |
| UT-Recompute-102 | Backward matches whole graph | RecomputeBackward | `MakeStages(53, 8, 16, kDouble, [TDevice.Cpu])`, B := `CloneStages(A, kDouble, [TDevice.Cpu])`; `ManualSeed(54)`; X, GOut = `randn([4, 16], kDouble)` (in that order); forward and backward on A, `WholeGraphBackward(B, X, GOut)` | exact: input gradient difference 0 and `MaxParamGradDiff(A, B)` 0 | |
| UT-Recompute-103 | Backward sets RequiresGrad | RecomputeBackward | as 010, then `RecomputeBackward(A, Saved, TTensor.ones([3, 8], kDouble))` | every stage `Calls` 2 and `LastRequiresGrad` True | |
| UT-Recompute-104 | Two backwards accumulate | RecomputeBackward | forward of 008, two backwards with `FromArray([1.0, 1])` | second result `'6 6'`; `S0.W.grad` `'18'`, `S1.W.grad` `'12'` (accumulated) | |
| UT-Recompute-105 | OnHost True and False equal | RecomputeBackward | inputs of 102 on A and on a second clone C; A with `Create(False)`, C with `Create(True)` | exact: input gradients equal, `MaxParamGradDiff(A, C)` 0 | |
| UT-Recompute-106 | Saved count mismatch | RecomputeBackward | `Hand`; `Saved` with one `Add`; backward with `FromArray([1.0, 1])` | `saved inputs: 1, stages: 2` | |
| UT-Recompute-107 | GradOut undefined error | RecomputeBackward | forward of 008; backward with `Default(TTensor)` | `GradOut is undefined` | |
| UT-Recompute-108 | Shape mismatch error | RecomputeBackward | forward of 008; backward with `FromArray([1.0, 1, 1])` | `stage 1: output [2], gradient [3]` | |
