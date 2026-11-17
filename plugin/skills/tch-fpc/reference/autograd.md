# Autograd

- Mark leaves: `W := TTensor.randn([3, 1]).set_requires_grad(True);`
  (`requires_grad_` is the generated in-place form).
- `Loss.Backward;` then `W.grad` holds dLoss/dW. `Backward(True)` keeps the
  graph for a second backward.
- `W.ZeroGrad;` zeroes the gradient (detached); gradients accumulate
  otherwise. A `TOptimizer` does it in `ZeroGrad` / `BackwardStep`.
- `RequiresGrad` reads the flag.
- `Detach` (generated `detach`) cuts a tensor out of the graph.

## Grad mode
- `G := NoGradGuard;` with `G: IInterface` turns grad mode off until G is
  released (end of the routine, or `G := nil`). Results computed then do
  not require grad.
- A bare `NoGradGuard;` statement is released at once and does nothing
  (trap t01). The compiler does not warn.
- `SetGradEnabled(False)` returns the previous mode; restore it yourself.
- Grad mode is per thread in libtorch.

## In-place on parameters
- An in-place op on a leaf that requires grad raises `ETorchNativeError`
  (trap t05). Do manual updates under a guard:

```pascal
procedure SgdStep(const W: TTensor; LR: Double);
var
  G: IInterface;
begin
  G := NoGradGuard;
  W.sub_(W.grad * LR);
  W.ZeroGrad;
end;
```

- `CopyFrom` into a parameter also needs the guard.

Source: DESIGN 4.7, 4.1; example 05, traps t01 and t05.
