# UT cases — program `qlora_verify` (T06c)

Program under test: `src/qlora_verify.pas` — thin CLI wrapper. It reads
`QLORA_FPC` / `QLORA_FPC_UNLEASHED` / `HOME`, builds the `TEnvRec` with
`ReadEnvRec`, passes its raw command line (zero-based `AArgs`) to
`DialectVerify.VerifyMain` with work root `work/verify`, prints the result
lines and exits with `VerifyMain`'s code (0 all good / 1 some not / 2 usage).

The test compiles the program with the system `fpc` into a scratch dir and
runs it with the project root as the working directory.

| Case ID | Target | Description | Input | Expected |
|---|---|---|---|---|
| UT-qlora_verify-001 | Program | `--help` forwards to `VerifyMain`, prints the usage line once, exit 0 | no env (default compilers); args `--help` | stdout = `usage: qlora_verify [--dialect D] [--timeout SECONDS] PATH...` + LF; exit code 0 |
| UT-qlora_verify-002 | Program | no arguments → usage error (the `if not VerifyMain` branch of the program body), exit 2 | no env; args (none) | stdout = same usage line + LF; exit code 2 |
| UT-qlora_verify-003 | Program | real run: one `objfpc` example with `.expected` is compiled by the real `fpc` into `work/verify/<n>` and run inside it; result line and exit code per the facts file | sample `s.pas` (`{$mode objfpc}`, prints `Hello`), `s.expected` (`Hello`); `QLORA_FPC` = absolute `fpc` path (via `sh -c` env form; the unit's `IsPathUsable` needs an existing file); args `--dialect objfpc` + sample path | one stdout line `<path>\tobjfpc\tcompiled-run\taccepted\t` (path = the argument given); exit code 0 |
