# Ownership

## TTensor
- A managed record around a reference-counted handle. Never `Free` it,
  never call a destructor; it is released when the last copy goes.
- `B := A` copies the handle: the same tensor. An in-place op through B is
  seen through A (trap t02).
- `A.ShallowClone`: a new handle (new `TTensor` object in libtorch) on the
  same storage. Data changes are shared; `SameHandle` is False.
- `A.Copy`: new storage with the same data.
- `A.CopyFrom(Src)`: copies Src's data into A, in place.
- A `const` parameter stops reassigning the variable, not changing the
  tensor's data: `T.mul_(...)` inside works.
- `IsNil`: the record holds no handle at all (an unassigned variable).
  `Defined`: libtorch's `defined()` (an undefined tensor can come back from
  an op, e.g. `grad` before any backward).
- Optional tensor arguments: pass an unassigned `TTensor`.

## Classes (free by hand)

| Class | Free | Notes |
|---|---|---|
| `TVarStore` | last | every `TVarPath`, layer and optimizer built on it must be gone first |
| `TModule` descendants (`TLinear`, ...) | before the store | a layer holds tensors registered in the store |
| `TSequential` | before the store | owns and frees the modules added to it |
| `TOptimizer` | first | built on a store's trainable variables |
| `TJitModule` | any time | owns the TorchScript module |
| `TTrainableJitModule` | before its store | owns its `TJitModule`, shares parameters with the store |

Pattern:

```pascal
VS := TVarStore.Create(TDevice.Cpu);
try
  Net := TLinear.Create(VS.Root / 'lin', 3, 1);
  try
    Opt := TOptimizer.Adam(VS, 1e-3);
    try
      ...
    finally
      Opt.Free;
    end;
  finally
    Net.Free;
  end;
finally
  VS.Free;
end;
```

- `TVarPath` is a record that points at its store: using one after the
  store is freed is an access violation, not an exception.
- `TIValue` is a record; tensors and objects inside it are managed too.

Source: DESIGN 4.1, 4.6.2, 4.6.4, 4.6.8.
