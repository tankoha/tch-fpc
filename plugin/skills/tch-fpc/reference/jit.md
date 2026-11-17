# TchJit

## Running a TorchScript file
- `M := TJitModule.Create(FileName)` or `Create(FileName, Device)`;
  `CreateFromStream(S[, Device])` reads from a stream. Free it yourself.
- `M.Forward([X, Y])` returns a tensor; it raises `ETorchNativeError` when
  the result is something else (a tuple, a number).
- `M.ForwardIV([..])` takes and returns `TIValue`; `M.Method(Name, [..])`
  and `M.MethodIV(Name, [..])` call other methods.
- `M.SetEval` / `M.SetTrain`, `M.ToDevice(Dev, Kind)`, `M.Save(File)`,
  `M.NamedParameters` (parameters only, sharing the module's tensors).

## TIValue
- A record; `Default(TIValue)` is None. Implicit conversion from
  `TTensor`, `Int64`, `Double`, `Boolean`, `String`, `IJitObject`, so
  `M.ForwardIV([X, 2, 0.5])` works.
- Constructors: `TIValue.Tuple([..])`, `IntList`, `DoubleList`, `BoolList`,
  `StringList`, `TensorList`, `GenericList`, `GenericDict([K1, V1, K2, V2])`,
  `None`, `FromDevice`.
- Read: `V.Tag` (`ivNone`, `ivTensor`, `ivDouble`, `ivInt`, `ivBool`,
  `ivTuple`, `ivIntList`, `ivDoubleList`, `ivBoolList`, `ivString`,
  `ivTensorList`, `ivGenericList`, `ivGenericDict`, `ivObject`,
  `ivStringList`, `ivDevice`), `V.TagName`, `V.Count`, `V.Item(I)`,
  `V.Key(I)`, `V.Value(I)`, `V.AsTensor`, `AsInt`, `AsDouble`, `AsBool`,
  `AsString`, `AsIntList`. A wrong `As...` raises `ETorchError` naming
  both tags.

## TorchScript objects
`IJitObject` (tag `ivObject`): `O.MethodIV(Name, [..])`, `O.GetAttr(Name)`;
`M.CreateClass('__torch__.Name', [..])` runs `__init__`. Reference-counted
like a tensor.

## Building a module in Pascal
- `M := TJitModule.CreateEmpty('Name')`, `M.RegisterParameter(Name, T)`
  (`IsBuffer` for buffers), `M.Define(Source)` with TorchScript source
  (`def` at column 0, lines joined with `#10`), then `M.Save(File)`.
- `TJitModule.CreateByTracing(ModuleName, FnName, Inputs, @Fn)` records a
  named Pascal function.

## Fine-tuning
- `J := TTrainableJitModule.Create(FileName, VS.Root / 'jit')` loads on the
  path's device and registers the parameters in the store (`jit.weight`,
  ...); an ordinary `TOptimizer` on the store trains them.
- `J.Forward(X, Train)` switches train/eval mode when `Train` differs.
- Free order: optimizer, `J`, store. `J.Save(File)` writes the trained
  module.

## Test models
`tests/jit/models/foo*.pt` come from tch-rs (MIT OR Apache-2.0, licence
files next to them). A program that reads them needs a checkout and runs
from the repository root.

Source: DESIGN 4.6.8, 4.6.9; examples 10, 11.
