# Errors

## Exceptions
- `ETorchError` (from `Exception`): misuse caught on the Pascal side, for
  example passing a `TTensor` that holds no tensor, or a wrong
  `TIValue.As...`.
- `ETorchNativeError` (from `ETorchError`): an error reported by libtorch.
  `Message` is the text before the C++ backtrace; `Detail` is the full
  text. Catch it before `ETorchError` when both are handled.
- Typical native errors: shape mismatch in `matmul` ("cannot be
  multiplied"), an op not defined for a kind (`mean` on Int64), an in-place
  op on a leaf that requires grad, tensors on two devices.
- After an exception the tensors involved are still valid; nothing leaks
  (handles are reference-counted records).

## FPU exceptions
- libtorch expects masked floating-point exceptions (it produces NaN and
  infinities on purpose). `Tch` masks them for the main thread at start-up,
  and new `TThread`s inherit the mask.
- Do not unmask them (`SetExceptionMask`, `Set8087CW`) around tch-fpc
  calls: an op may then crash with an FPU exception inside libtorch.
- A thread not created by FPC (a callback from a C library) calls
  `TchMaskFpuExceptions` first.

## Not caught at all
- A `TVarPath` used after its store was freed: access violation.
- A class freed twice, or a store freed before its layers and optimizers:
  access violation or heap corruption. Keep the order of
  `reference/ownership.md`.

Source: DESIGN 4.2; example 12, traps t04 and t05.
