# Naming

## Generated ops
- Names are PyTorch's, verbatim, in snake_case: `T.max_pool2d(...)`,
  `T.add_(U)`, `T.sum_dim_intlist(IntList([1]), False, kNone)`.
  Search `src/TchTensorGenerated_intf.inc` for `function <name>(`.
- A trailing `_` means in place (`add_`, `mul_`, `zero_`); the result is the
  same tensor.
- Leading and double underscores are kept (`_cast_byte`, `__and__`).
- Overloads are separate names, as in tch-rs: `sum` (all), `sum_dim_intlist`
  (over dims); `arange`, `arange_start`, `arange_start_step`.
- The hand-written API is CamelCase: `Shape`, `Kind`, `Dim`, `Numel`,
  `DoubleValue`, `Int64Value`, `ToDoubleArray`, `Backward`, `ZeroGrad`,
  `Copy`, `ShallowClone`, `RequiresGrad`, `Device`.

## Receiver rule
- An op with a tensor argument named `self` is a method of that tensor.
- Otherwise, one with a tensor named `input` is a method of it:
  `X.conv2d(W, B, ...)`.
- Otherwise it is a class function: `TTensor.zeros([2, 3], kFloat, TDevice.Cpu)`,
  `TTensor.cat([A, B], 0)`, `TTensor.stack([A, B], 0)`.
- Short factories with CPU/kFloat defaults: `TTensor.zeros([2, 3])`,
  `ones`, `empty`, `rand`, `randn`.

## Reserved words
- Ops: `&div`, `&var`, `&to`, `&set`, `&repeat`, `&absolute`, `&alias`,
  `&index`. Call them with the `&`: `T.&div(U)`, `T.&to(Dev)`.
- Parameters: `&out`, `&end`, `&to`, `&index`, `&var`, `&forward`.
- A parameter named `result` (3 ops) is `result_`: in `{$mode objfpc}`
  `result` is the function result, `&result` does not help.
- Without the `&` the compiler reports
  `Syntax error, "identifier" expected but "div" found` (pitfalls p01, p06).

## Optional arguments
- `TOptInt64`, `TOptDouble`: pass a value, or `NoInt64` / `NoDouble`.
- `TOptIntList`: `IntList([1, 2])`, `IntList([])` (present and empty), or
  `NoIntList` (absent). They are different to libtorch.
- Optional tensors: pass an unassigned `TTensor` (it holds no tensor).
- Optional kind: `kNone`. Optional layout: `ltNone`.
- `TScalar` parameters accept `Int64` and `Double`: `T.clamp(0, 1.5)`.

Source: DESIGN 4.4.
