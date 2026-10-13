# tch-fpc

Free Pascal bindings for the C++ API of PyTorch (libtorch), inspired by
[tch-rs](https://github.com/LaurentMazare/tch-rs).

**Status:** early implementation (Linux x86-64, FPC 3.2.2, libtorch 2.13.0).
Runs on the CPU, and on the GPU through a ROCm build of libtorch (tested on
an AMD RX 9070 with `2.13.0+rocm7.1`); the CUDA build path exists but is
untested, as there is no NVIDIA hardware here. Tensors with the 2,657
generated ops and a hand-written core API, autograd, `TchNN` (initialisation,
var store with save/load, SGD/Adam/AdamW/RmsProp, Linear, Conv1D/2D/3D,
ConvTranspose1D/2D/3D, BatchNorm, LayerNorm, GroupNorm, Embedding, LSTM/GRU,
Identity, Sequential) and `TchJit` (loading, running, tracing and fine-tuning
TorchScript modules, TorchScript objects) work against the real libtorch. The
API may still change.

For a ROCm or CUDA libtorch, see section 4.8 of
[docs/DESIGN.md](docs/DESIGN.md): the libtorch directory is linked manually
(`-DTCH_LIBTORCH_DIR=...`), with one `build/libtch-<variant>` per variant.
The ROCm libtorch zip carries no symlinks and needs eight after unpacking;
they are listed under "Results" in 4.8.

## Naming, and how it differs from tch-rs

- Generated ops keep the PyTorch/tch-rs name verbatim in snake_case
  (`t.max_pool2d(...)`, `t.add_(u)`); the hand-written API uses Pascal
  CamelCase (`Shape`, `Kind`, `ToString`).
- Names that are Pascal reserved words or modifiers are escaped with `&`,
  ops and parameters alike: `t.&div(u)`, `t.&var(0)`, `add_out(&out, ...)`.
  The one exception is a parameter named `result` (3 ops), which becomes
  `result_` because `Result` is the function result in Pascal.
- Parameter names are otherwise verbatim too (`dim`, `indices`, `pow`),
  even where they equal a `TTensor` member such as `Dim` or the `indices`
  op. `Tch.pas` enables `{$modeswitch duplicatelocals}` for this, the part
  of Delphi mode that lets a parameter shadow a member, as Rust does
  implicitly (parameters are locals, methods are reached as `self.dim()`).
  Your own units are not affected; the switch lives in `Tch.pas` only.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option. This matches tch-rs, whose C wrapper and code generator
this project forks.

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in this project, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.

Japanese version: [README_jp.md](README_jp.md)
