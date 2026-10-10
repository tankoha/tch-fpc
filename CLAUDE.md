# tch-fpc

Free Pascal bindings for libtorch, modeled after tch-rs.

## Conventions
- English docs are canonical; Japanese translations use the `_jp` suffix.
- Target compiler: FPC 3.2.2.
- Scripting: bash for one-liners; Babashka or instantfpc for anything larger.
  Do not use Python or Node.js for ad-hoc scripts.
- Measure before asserting: for ABI, calling conventions and memory layout,
  verify by building and running, not from memory.

## License caution
The license is undecided and the repo is private. tch-rs is MIT/Apache-2.0.
When porting code or generated declarations from tch-rs, note the source file
in the commit message so attribution can be added later.
