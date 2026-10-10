# tch-fpc

Free Pascal bindings for the C++ API of PyTorch (libtorch), inspired by
[tch-rs](https://github.com/LaurentMazare/tch-rs).

**Status:** early implementation. The C wrapper builds against libtorch
2.13.0 and the core design has been verified on FPC 3.2.2; the generated
tensor API does not exist yet. Nothing usable yet.

## Naming, and how it differs from tch-rs

- Generated ops keep the PyTorch/tch-rs name verbatim in snake_case
  (`t.max_pool2d(...)`, `t.add_(u)`); the hand-written API uses Pascal
  CamelCase (`Shape`, `Kind`, `ToString`).
- Op names that are Pascal reserved words or modifiers are escaped with
  `&`: `t.&div(u)`, `t.&var(0)`.
- **Parameter names may carry trailing underscores** (`dim_`, `indices_`,
  `pow__`). In `{$mode objfpc}` a method parameter may not share its name
  with a member of the record, and every op is a member of `TTensor`, so
  a parameter named like a member (a hand-written method, or another op such
  as `indices`) gets `_` appended until it is unique. Rust has no such
  conflict: parameters are plain locals and methods are only reachable as
  `self.dim()`, so tch-rs keeps `dim: i64` next to `fn dim(&self)`.
  Parameter names never appear at call sites, so this only shows in
  signatures.
- For the same reason the hand-written API avoids common parameter names:
  the number of dimensions is `NumDims`, not `Dim` (`dim` is a parameter
  of about 250 ops).

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
