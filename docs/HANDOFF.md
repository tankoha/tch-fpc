# Handoff

Read this first when resuming. Decisions and rationale: [DESIGN.md](DESIGN.md).

## State (2026-10-10, end of cloud design session 1)

- Design for all three layers written in DESIGN.md. Nothing built yet.
- Reference: tch-rs `4227b89b72059b0651ff83a38637693574e0a2d6`,
  libtorch 2.13.0.
- Decided: fork tch-rs libtch (keep C names), shared `libtch.so`,
  `TTensor` = record wrapping a refcounted interface, exceptions via
  `ETorchError` after every call, snake_case generated ops +
  CamelCase hand-written API, Pascal emitter fed by a manifest from a forked
  gen.ml.

## Next steps (local implementation)

1. **Toolchain check.** FPC 3.2.2, g++, CMake, libtorch 2.13.0 CPU
   (cxx11 ABI) unpacked into `libtorch/` (already git-ignored).
2. **Vendor libtch.** Copy from tch-rs at the commit above into
   `csrc/libtch/` and `gen/`; add LICENSE-MIT / LICENSE-APACHE. Commit the
   pristine copy first (message names the source files), then a second
   commit with our patch: remove stream/python/image code, add `tchf_free`.
   Build `libtch.so` with CMake.
3. **Smoke test in C** (tiny C program, not Python): create a tensor,
   `atg_add`, print, trigger an error and read `get_and_reset_last_err`.
4. **ABI verification in Pascal** — measure, do not assume:
   - `bool` returns (e.g. `at_autocast_is_enabled`) read correctly as
     `Boolean`.
   - `c10::ScalarType` numeric values match `TKind` (print
     `at_scalar_type` for tensors of each kind).
   - Linking: `external 'tch'` + rpath so the program finds `libtch.so` and
     libtorch at run time.
5. **FPU mask test.** Without `SetExceptionMask`, does `1.0 / 0` on a CPU
   tensor crash an FPC program? Confirm the fix in `Tch` initialization.
6. **Ownership test.** Loop 10^6 times over `(A + B) * C` and check RSS is
   flat; check `B := A` then `A := Default(TTensor)` keeps B alive. This
   validates the record-with-interface design on 3.2.2.
7. **Generator.** Add `ops.tsv` output to gen.ml; write `gen/tchgen.pas`;
   generate `TchSysGenerated.pas` and the `.inc` files; compile everything.

## Open questions

- **OCaml dependency.** Regenerating needs OCaml + `yaml`/`base`/`stdio`
  opam packages. Acceptable locally? Alternative: regenerate `ops.tsv` only
  in cloud sessions and commit it.
- **`IntListOption` in Pascal.** Open arrays cannot express "none" vs
  "empty". Candidates: `TOptIntList` record with implicit operator from
  `TInt64Array`, or overloads with/without the argument.
- **Multi-output ops**: `out` parameters (current choice) vs small result
  records (`TTensorPair`). Revisit after writing real code with `max_dim`,
  `sort`, `lstm`.
- **Default arguments.** How many convenience overloads to generate vs write
  by hand.
- **Streams / images.** If needed later: replace tch-rs's Rust-implemented
  stream symbols with a struct of C function pointers passed in by Pascal
  (wrapping `TStream`). Images can be done in Pascal (FPImage) instead of stb.
- **Windows / macOS.** Not designed yet beyond `cdecl` and `tchf_free`.
- **License.** Still undecided; repo stays private. The fork brings
  MIT/Apache-2.0 code, so dual-licensing tch-fpc the same way is the
  path of least friction — user decision.
