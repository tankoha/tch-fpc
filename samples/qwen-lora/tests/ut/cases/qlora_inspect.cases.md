# UT cases — program `qlora_inspect` (T01c)

Program under test: `src/qlora_inspect.pas` — thin CLI wrapper (R20). It
copies `ParamStr(1..ParamCount)` into a zero-based array, calls
`InspectCli.RunInspectCLI` with fresh `TStringList`s, writes the AOut lines
to stdout and the AErr lines to stderr, and `Halt`s with the returned code
(0 inspected / 1 file error or data_end <> file_size / 2 usage).

The unit's own behavior is covered by UT-InspectCli-001..011 (in process);
these rows pin the wrapper's binary contract — the only untested lines in
the program body — through a real invocation.

The test compiles the program with the system `fpc` into a scratch dir
(`work/ut/qi-bin`, rebuilt once) and runs it with the project root as the
working directory. Test GGUF files are written under `work/ut/inspect/`
with unique names.

| Case ID | Title | Input (short) | Expected result | Notes |
|---|---|---|---|---|
| UT-qlora_inspect-001 | Usage error through the wrapper: no arguments | args (none) | rc = 2; stdout empty; stderr = `Usage: qlora_inspect [--kv] [--tensors] FILE` + LF (AErr loop of the program body) | forwards to `RunInspectCLI` check 1 (2.12); R18 |
| UT-qlora_inspect-002 | Valid file, relative path, no flags (default) | test file: 0 KV, tensor f32 [4] (16 B), data ends at file end; args [relative FILE, lowercase name] | rc = 0; stderr empty; stdout = the 7 summary lines (version 3, alignment 32, kv 0, tensors 1, data_offset 64, data_end 80, file_size 80) + `type f32 1` (AOut loop of the program body) | 2.12 steps 2–5; R13, R14 |
| UT-qlora_inspect-003 | --kv with absolute path, mixed-case name and keys | test file: 2 KV (uint8 key `KA`, uint32 key `kb`), 0 tensors, file size = data_offset; args ['--kv', absolute FILE with uppercase extension] | rc = 0; stderr empty; stdout = the 7 summary lines (version 3, alignment 32, kv 2, tensors 0, data_offset 64, data_end 64, file_size 64) + `kv KA uint8` + `kv kb uint32`, no type lines | 2.12 steps 3, 6; R15, R17 |
