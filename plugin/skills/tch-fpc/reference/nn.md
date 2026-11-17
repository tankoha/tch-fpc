# TchNN

## Store and paths
- `VS := TVarStore.Create(Device[, Kind])`; `VS.Root` is the root path.
- `P / 'name'` (or `P.Sub('name')`) builds a path; variables are named
  with dots: `VS.Root / 'enc' / 'lin'` gives `enc.lin.weight`, `enc.lin.bias`.
- `P.Variable`, `P.Zeros`, `P.Ones`, `P.ZerosNoTrain`, `P.KaimingUniform`,
  `P.Randn`, ... add tensors; `P.Add(Name, T, Trainable)` adds your own.
- `VS.Variables` (insertion order), `VS.TrainableVariables`,
  `VS.TryGet('enc.lin.weight', T)`, `VS.Count`.
- `VS.Save(File)` / `VS.Load(File)` (names must match), `LoadPartial`
  returns the missing names. Format: the `.ot` of tch-rs.
- `VS.Freeze` / `VS.Unfreeze` switch `requires_grad` of the trainable
  variables; they stay in `TrainableVariables`.
- `VS.SetKind(kHalf)`, `SetFloatKind`, `SetDevice` convert in place.

## Modules
- Base class `TModule` with
  `function Forward(const X: TTensor; Train: Boolean = False): TTensor; virtual; abstract;`
  An override must repeat `= False`.
- `Train` defaults to False. Pass True while training: batch norm and
  dropout behave differently (trap t06).
- Layers take the path first: `TLinear.Create(P, In, Out)`,
  `TConv2D.Create(P, In, Out, K)`, `TBatchNorm.Create(P, Nd, Features)`,
  `TLayerNorm.Create(P, [Dims])`, `TEmbedding.Create(P, N, Dim)`,
  `TGroupNorm`, `TConvTranspose2D`, `TLSTM`, `TGRU`, `TIdentity`.
  Each has an overload with a config record (`DefaultLinearConfig`, ...).
- `TSequential.Create.Add(M).AddFn(@F)`: owns its modules; `ForwardAll`
  returns every intermediate output.
- `TFuncModule.Create(@F)` with `function F(const X: TTensor): TTensor` or
  `(const X: TTensor; Train: Boolean)`, or a method `of object`.
  No anonymous functions in FPC 3.2.2 (pitfall p05).

## Optimizers
- Class functions: `TOptimizer.Sgd(VS, LR, Momentum, ...)`,
  `Adam(VS, LR, Beta1, Beta2, WD, Eps, AmsGrad)`, `AdamW`, `RmsProp`.
- `Opt.BackwardStep(Loss)` = ZeroGrad + Backward + Step.
  `BackwardStepClip`, `BackwardStepClipNorm`, `ClipGradNorm`;
  `SetLearningRate`, `SetMomentum`, `SetWeightDecay` (and `...Group` forms).
- Variables added to the store after the optimizer was built join at the
  next `ZeroGrad` / `Step`. Free the optimizer before the store.
- `TOptimizer` cannot save its state.

## Losses
Generated ops on the prediction: `Pred.mse_loss(Target)`,
`Logits.cross_entropy_loss(Labels, NoWeight)` with an unassigned
`NoWeight: TTensor` (reduction 1 = mean is the default), `LogProbs.nll_loss(...)`.

Source: DESIGN 4.6.1–4.6.7, 4.6.10; examples 06–08.
