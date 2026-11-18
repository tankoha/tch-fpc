# Detailed Design — T06: `DialectVerify` unit and `qlora_verify` program

Based on: `docs/external-design/T06-external.md`, `docs/system-flow/T06-system.md`,
`tests/ita/cases/T06-ita.md`, `tasks/T06-facts.md` (authoritative where it differs),
`../../docs/DESIGN.md` §8.12/§8.16/§8.18.

Dialect names, outcomes, flags and the delphi-copy rewrite below are defined by
`tasks/T06-facts.md` (checked against DESIGN 8.16 and the `pascal-dialects`
oracle `tests/check-examples.sh`).

## 1. Module composition

This task defines no source code yet; the UT cases (step 2 of the SDLC flow)
are written from this table, and only then may `src/` files be created.
Everything is Free Pascal 3.2.2, `{$mode objfpc}{$H+}`, Pascal only + compilers.

| File | Kind | Responsibility |
|---|---|---|
| `src/DialectVerify.pas` | unit | All business logic: dialect detection, compiler choice, delphi-copy rewrite, expect-error extraction, compile/run with time limit, expected-output comparison, TSV line building, file collection, CLI core |
| `src/qlora_verify.pas` | program | Thin CLI entry: read `QLORA_FPC` / `QLORA_FPC_UNLEASHED` / `HOME` via `GetEnvironmentVariable`, build the `TEnvRec` with `DialectVerify.ReadEnvRec(fpc, fpcUnleashed, home)`, then call `DialectVerify.VerifyMain(params, 'work/verify', env, lines, ec)`, print each line, `Halt(ec)` |

The unit is fully covered by fpcunit (`tests/ut/DialectVerify_test.pas`); the
program body is intentionally out of the coverage path (it only forwards).
Fake compiler scripts used by the tests are written by the test setup under
`work/ut/` (git-ignored), never inside the repository tree.

## 2. Function specifications

Unit `DialectVerify`, interface section. Signatures without `const` string
parameters are implementation notes; all text parameters are `const string`.

### 2.1 `function ReadEnvRec(const AFpc, AFpcUnleashed, AHome: string): TEnvRec`
- Args: the env var values `QLORA_FPC` / `QLORA_FPC_UNLEASHED` / `HOME`,
  supplied by the caller — only the program `qlora_verify` calls
  `GetEnvironmentVariable` (T06c); an empty string means "not set".
  The UT never sets environment variables.
- Returns: `Fpc` = `AFpc` if non-empty else `'fpc'`;
  `FpcUnleashed` = `AFpcUnleashed` if non-empty else
  `AHome + '/unleashed/fpc/bin/fpc'`.
- No env of its own is read inside (in FPC 3.2.2 `GetEnvironmentVariable`
  reads a copy taken at program start, so a value set later with libc
  `setenv` is not even seen — the value-passing form removes the trap);
  no raise.

### 2.2 `function DetectDialect(AText: string; AFileName: string): string`
- Returns one of `objfpc`, `fpc-delphi`, `unleashed`, `delphi`, `oxygene`.
- Steps (first hit wins; facts order):
  1. earliest line whose first non-blank content is exactly
     `{$mode objfpc}` / `{$mode delphi}` / `{$mode unleashed}`
     (case-insensitive, blanks after `{` tolerated) → `objfpc` /
     `fpc-delphi` / `unleashed`;
  2. first word of the file is `namespace` (case-insensitive) → `oxygene`;
  3. `AFileName` ends with `.dpr` (case-insensitive) or a line contains
     `{$APPTYPE CONSOLE}` (case-insensitive) → `delphi`;
  4. else `objfpc`.
- No raise.

### 2.3 `function ChooseCompiler(ADialect: string; const AEnv: TEnvRec; out AClass: string): string`
- Returns the compiler path and sets `AClass`.
- `objfpc` → (`AEnv.Fpc`, `compiled-run`);
- `fpc-delphi` → (`AEnv.Fpc`, `compiled-run`);
- `unleashed` → (`AEnv.FpcUnleashed`, `compiled-run`);
- `delphi` → (`AEnv.FpcUnleashed`, `fpc-approx`);
- `oxygene` → (`''`, `static-only`);
- unknown dialect → (`''`, `''`) — caller produces a usage error
  (defensive; the CLI rejects it first).

### 2.4 `function BuildCompileCmdLine(ACompiler, ASrcPath, AOutDir: string; ADelphi: Boolean): string`
- Base flags (facts): `-vew -l- -FE<out> -FU<out>`;
  `ADelphi=True` prepends `-Mdelphiunicode -Manonymousfunctions
  -Mfunctionreferences -Minlinevars` (DESIGN 8.16 oracle order).
- Paths quoted. Pure string building; no raise.

### 2.5 `function RewriteDelphiUnits(AText: string): string`
- Whole-word rewrites (facts list), applied to the source text; order:
  `System.Generics.X` → `Generics.X` for `X ∈ {Collections, Defaults}`, then
  `System.X` → `X` for `X ∈ {SysUtils, Classes, Math, StrUtils, DateUtils,
  Types, Variants, TypInfo, Rtti}`.
- Whole word = the preceding char (if any) is not `[A-Za-z0-9_$.]` and the
  char after the unit name (if any) is not `[A-Za-z0-9_$]` — so
  `XSystem.SysUtils` or `System.SysUtilsX` are left untouched.
- Comments are rewritten too (oracle `sed` rewrites the whole file);
  rewrites inside string literals are tolerated (oracle does the same).
- No raise.

### 2.6 `function ExtractExpectErrorText(AText: string): string`
- A pitfall is `{ ... expect-error: <text> }` (or `(* ... *)` carrying
  `expect-error:`). `<text>` runs to the closing `}` (or end of line) with
  blanks trimmed (facts; oracle `sed` does the same).
- Returns the first such text, else `''`. No raise.

### 2.7 `function FirstCompilerErrorLine(ALog: string): string`
- Returns the first line containing `Error:` or `Fatal:` (case-insensitive as
  in FPC logs), else `''`. Used for the `<detail>` cell (facts).

### 2.8 `function NormalizeDetail(A: string): string`
- Takes A as one line; replaces every tab by `' '`; keeps at most 200 chars
  (facts: max 200 chars, tabs → blanks). No raise.

### 2.9 `function RunLimited(const AExe: string; const AArgs: array of string; const ADir: string; ATimeoutMs: Integer; out AOutput: string; out AExitCode: Integer; out ATimedOut: Boolean): Boolean`
- Adapted from the validated snippet `tasks/snippets/runlimited.pas`
  (FPC 3.2.2); the signature follows the snippet: exe and args are separate
  parameters (no `ACmdLine` string), args are a list in-parameter.
- Starts the exe with `TProcess` (`poUsePipes`, `poStderrToOutPut`); on
  exception (bad exe etc.) returns False and `AOutput` gets
  `process error: ...`.
- Polls until exit or `ATimeoutMs`; on timeout `Terminate(9)`,
  `ATimedOut:=True` (`AExitCode` = 255 sentinel, partial output kept).
- On success `AOutput` = merged stdout+stderr, `AExitCode` = real code
  (`P.ExitCode`, not the raw wait status).
- Returns True when the process started (even on timeout); False when it
  could not be started. Callers branch on `ATimedOut` / `AExitCode`.
- `AExe` and `ADir` are resolved against the program working directory
  before start: the child chdirs to `ADir` before `exec`, so a relative
  path would otherwise resolve against the new working directory and fail
  with ENOENT (exit 127, observed on FPC 3.2.2 / glibc).

### 2.10 `function ExpectedPathFor(ASrcName: string): string`
- `<dir>/<basename>.expected` — the `.expected` file next to the source,
  same base name regardless of the source's extension (facts).
- Pure string building.

### 2.11 `function ReadTextFile(APath: string; out AText: string): Boolean`
- Whole-file read (binary, no translation). Returns False when the file
  cannot be open/read. No raise.

### 2.12 `function IsPathUsable(APath: string): Boolean`
- File exists and is readable, or directory exists. Used for CLI validation
  and the compiler-existence check.

### 2.13 `function CompileStep(ACompiler, ASrcPath, AOutDir: string; ATimeoutMs: Integer; ADelphi: Boolean; out ALog: string; out AExePath: string): TCompileState`
- `TCompileState = (csNoCompiler, csStartFail, csTimeout, csFail, csOk)`.
- For `delphi` (ADelphi=True): write the rewritten copy to
  `<out>/<name>` (same file name, facts) and compile the copy;
  else compile `<src>` directly.
- `csOk` := exit 0 and the executable `<out>/<name>` exists.
- `AExePath` := `<out>/<name>` when `csOk`, else `''`.
- `CompileStep` never returns `csStartFail`: the compiler is pre-validated
  (missing/not executable → `csNoCompiler`), and FPC 3.2.2's `TProcess`
  reports post-start exec failures as exit 127 (classified `csFail`). The
  state stays in the enum for `VerifyFile`'s defensive classification.
- `AOutDir` is resolved to an absolute path up front (so `-FE`/`-FU`/src
  and `<out>/<name>` are absolute regardless of the caller's relative path).
  The compiler is run with an EMPTY `RunLimited` work dir: FPC 3.2.2's
  `TProcess` cannot exec a shebang script once `CurrentDirectory` is set
  (verified: exit 127 for scripts, fine for ELF binaries); the rehearsal
  fixtures are shell scripts, and the absolute flag table makes the cwd
  change unnecessary.

### 2.14 `function VerifyFile(ASrcName: string; AForcedDialect: string; ATimeoutSec: Integer; const AEnv: TEnvRec; AWorkRoot: string; out AOK: Boolean): string`
- Args: explicit file path; `''` = detect dialect (else forced, already
  validated to one of the five); run time limit in seconds; env record;
  work root (e.g. `work/verify`).
- Returns the finished TSV line (see 2.16): `<path>\t<dialect>\t<class>\t
  <outcome>\t<detail>`. `AOK=True` iff outcome ∈ {`accepted`, `static-ok`,
  `skipped`} (exit-code rule, facts).
- Pipeline in branch order:
  1. dialect := forced, else `DetectDialect(text, name)`;
  2. `oxygene` → static check: first word `namespace` → `static-ok` else
     `static-reject`; nothing is compiled (facts, ITa-008/009);
  3. `ChooseCompiler` → (path, class); path empty or not on disk →
     `skipped`, detail `compiler not found` (not a failure, facts/external);
  4. fresh out dir `<workroot>/<n>` (n per file, kept unique per call) —
     `<out>` of the flag table (facts);
  5. `CompileStep` under `ATimeoutSec*1000` ms:
     - `csNoCompiler`/`csStartFail` → `skipped`;
     - `csTimeout` → `skipped`, detail `timeout` (external: killed by
       `--timeout` → skipped; ITa-012);
     - `csFail` → pitfall (`ExtractExpectErrorText <> ''`) and log contains
       the text literally → `accepted`; pitfall, text absent → `wrong-error`,
       detail `FirstCompilerErrorLine(log)`; not a pitfall → `compile-error`,
       detail `FirstCompilerErrorLine(log)` (facts);
     - `csOk` + pitfall → `compiled-but-should-fail`, detail `` (facts);
  6. run `<out>/<name>` under `ATimeoutSec*1000` ms, stdout+stderr captured
     together (facts; ITa-011 for `timeout` + `--timeout 1`):
     - start failure → `skipped`;
     - timeout → `timeout`, detail = the limit in seconds used;
     - exit ≠ 0 → `run-error`, detail = the exit code (facts);
     - exit 0 → `.expected` next to source (2.10) missing → `no-expected`;
       byte-for-byte equal → `accepted`; else `wrong-output` (facts: no
       partial match).
- No raise on normal failure paths (all classified as outcomes).

### 2.15 `function CollectPascalFiles(APath: string; out AFiles: array of string): Boolean`
- `APath` a file → `[APath]`. A directory → all `*.pas` / `*.dpr` recursively
  (extension match case-insensitive; no `.pp`, facts), sorted by full path
  (facts). Non-existent path → False.

### 2.16 TSV line shape (used by 2.14)
`<path>\t<dialect>\t<class>\t<outcome>\t<detail>` — five cells, `<detail>`
`NormalizeDetail`-ed, possibly empty cell (facts). Outcome vocabulary:
`accepted`, `compiled-but-should-fail`, `wrong-error`, `compile-error`,
`run-error`, `timeout`, `wrong-output`, `no-expected`, `static-ok`,
`static-reject`, `skipped`.

### 2.17 `procedure SortStrings(A: TStringArray)`
- `TStringArray` (SysUtils dynamic array), not an open-array `out` parameter
  (lists are carried as `TStringArray`; an open array cannot be resized).
- In-place ascending, stable enough for equal keys (duplicates). n=0,1 no-op.

### 2.18 `function IsGoodOutcome(AOutcome: string): Boolean`
- `accepted` / `static-ok` / `skipped` → True (exit-code rule, facts).

### 2.19 `function VerifyMain(AArgs: array of string; AWorkRoot: string; const AEnv: TEnvRec; out ALines: array of string; out AExitCode: Integer): Boolean`
- `AEnv` is the `TEnvRec` **supplied by the caller**: the only reader of the
  environment is the program `qlora_verify` (it calls `GetEnvironmentVariable`
  for `QLORA_FPC` / `QLORA_FPC_UNLEASHED` / `HOME` and builds the record via
  `ReadEnvRec` — 2.1); **`VerifyMain` itself never reads the environment**
  (2.1 / T06b3). `AWorkRoot` e.g. `work/verify`.
- Parses argv (2.20), expands paths (2.15, `False` → usage error exit 2),
  runs `VerifyFile` per file in order (passing `AEnv` through to each).
- `AExitCode := 0` iff every outcome is `IsGoodOutcome`, else 1; usage
  error → `ALines := [usage text]`, `AExitCode := 2`, returns `False`;
  otherwise returns `True` (facts: 0 / 1 / 2).

### 2.20 CLI parsing (inside 2.19)
- `--dialect D`: D must be one of the five dialect names; repeated → last
  wins; missing/invalid value → usage error (`--dialect` with no value).
- `--timeout N`: N integer ≥ 1 else usage error; wins over the default 10 s
  (facts: default 10 s); repeated → last wins.
- `PATH...`: every remaining argument, file or directory (both allowed, in
  any mix and order). A missing path → usage error (nothing to verify;
  documented interpretation, facts silent).
- Unknown `--*` → usage error. No PATH at all → usage error, exit 2.
- `--` alone ends option parsing (remaining args are paths).
- `--help` → print usage, exit 0 (documented convenience, facts silent).

## 3. Data structures & constants

```pascal
const
  D_OBJFPC = 'objfpc';        D_FPC_DELPHI = 'fpc-delphi';
  D_UNLEASHED = 'unleashed';  D_DELPHI = 'delphi';   D_OXYGENE = 'oxygene';
  VS_COMPILED_RUN = 'compiled-run';  VS_FPC_APPROX = 'fpc-approx';
  VS_STATIC_ONLY = 'static-only';
  OUT_ACCEPTED='accepted'; OUT_PIT_OK='compiled-but-should-fail';
  OUT_WRONG_ERR='wrong-error'; OUT_COMPILE_ERR='compile-error';
  OUT_RUN_ERR='run-error'; OUT_TIMEOUT='timeout'; OUT_WRONG_OUT='wrong-output';
  OUT_NO_EXPECTED='no-expected'; OUT_STATIC_OK='static-ok';
  OUT_STATIC_REJECT='static-reject'; OUT_SKIPPED='skipped';
  K_DEFAULT_TIMEOUT_S = 10;   // facts: run limit default 10 s
  K_DETAIL_MAX = 200;         // facts: detail max 200 chars
  K_OUT_MAX = 10*1024;        // external struct: stdout truncation
  K_SIMPLE_UNITS: array[0..8] of string = ('SysUtils','Classes','Math','StrUtils','DateUtils','Types','Variants','TypInfo','Rtti');
  K_GENERIC_UNITS: array[0..1] of string = ('Collections','Defaults');
type
  TEnvRec = record Fpc, FpcUnleashed: string; end;
  TCompileState = (csNoCompiler, csStartFail, csTimeout, csFail, csOk);
```

No `QLORA_TIMEOUT` env, no `QLORA_DELPHI_PATH` (facts: `QLORA_DELPHI_PATH`
is not used). Env read by the program: `HOME`, `QLORA_FPC`,
`QLORA_FPC_UNLEASHED`. `QLORA_DIALECT_SKILLS` is an ITa-harness variable
(oracle run only), not read by `qlora_verify`.
`work/verify/` holds one fresh `<out>` directory per verified file;
`work/` is git-ignored (project rule).

## 4. Business-rule → implementation map

Every rule of the external design's normative sections is realized here:

| # | Rule (external design / facts) | Where |
|---|---|---|
| R1 | `{$mode objfpc/delphi/unleashed}` line → `objfpc`/`fpc-delphi`/`unleashed` (case-insensitive, first wins) | `DetectDialect` step 1 |
| R2 | first word `namespace` → `oxygene` | `DetectDialect` step 2 |
| R3 | `.dpr` file or `{$APPTYPE CONSOLE}` line → `delphi` | `DetectDialect` step 3 |
| R4 | no indicator → `objfpc` | `DetectDialect` step 4 |
| R5 | compiler per dialect: `QLORA_FPC` (objfpc, fpc-delphi), `QLORA_FPC_UNLEASHED` (unleashed, delphi), none (oxygene) | `ChooseCompiler` |
| R6 | classes `compiled-run` / `fpc-approx` / `static-only` | `ChooseCompiler` |
| R7 | flags `-vew -l- -FE<out> -FU<out>`; delphi adds the four `-M*` switches on the rewritten copy | `BuildCompileCmdLine`, `CompileStep` |
| R8 | delphi copy: `System.X`→`X` (9 units), `System.Generics.X`→`Generics.X` (2 units), whole words only, written to `<out>` with the same name | `RewriteDelphiUnits`, `CompileStep` |
| R9 | compiler absent → `skipped` (not a failure) | `VerifyFile` 3. |
| R10 | compiler run fails to start or is killed by the time limit → `skipped` | `VerifyFile` 5./6., `CompileStep` |
| R11 | pitfall `{ ... expect-error: <text> }`: compile must fail and log contain `<text>` → `accepted`; else `wrong-error` (detail: first `Error:`/`Fatal:` line); compile success → `compiled-but-should-fail` | `ExtractExpectErrorText`, `VerifyFile` 5. |
| R12 | example: compile, run under limit (stdout+stderr together), byte-for-byte against `<basename>.expected` → `accepted`; no partial match | `VerifyFile` 6., `ReadTextFile` |
| R13 | outcomes `compile-error` (detail: first error line), `run-error` (detail: exit code), `timeout`, `wrong-output`, `no-expected` | `VerifyFile` 5./6. |
| R14 | Oxygene: `static-ok` iff first word `namespace`, else `static-reject`; nothing compiled | `VerifyFile` 2. |
| R15 | TSV `<path> <dialect> <class> <outcome> <detail>`; detail ≤ 200 chars, tabs → blanks | `NormalizeDetail`, 2.16 |
| R16 | CLI: `PATH...` (file or dir, recursive `*.pas`/`*.dpr`, sorted); `--dialect D`; `--timeout N` (default 10 s); exit 0/1/2 | 2.15, 2.19, 2.20 |
| R17 | exit 0 iff every file `accepted` / `static-ok` / `skipped` | `IsGoodOutcome`, `VerifyMain` |
| R18 | `<out>` fresh per file under `work/verify/` | `VerifyFile` 4. |
| R19 | `QLORA_DIALECT_SKILLS` names the `pascal-dialects` `skills/` dir for the oracle run (ITa-013); each skill dir verified with its dialect | ITa only; `--dialect` in the CLI |
| R20 | `.expected` same base name next to the source, regardless of the source's extension | `ExpectedPathFor` |

## 5. Edge cases & error handling

| Situation | Behavior |
|---|---|
| Source file missing/unreadable | `VerifyFile` reports it; `VerifyMain` treats a non-existent PATH as a usage error (exit 2) — documented, facts silent |
| `TProcess.Execute` raises (bad exe) | `RunLimited` `AStarted=False` → outcome `skipped` (R10) |
| Compile timeout | `skipped`, detail `timeout` (ITa-012) |
| Run timeout | `timeout`, detail = limit seconds (ITa-011: `timeout 1` → detail `1`) |
| Run exit ≠ 0 | `run-error`, detail = exit code |
| `.expected` with trailing/leading whitespace | byte-for-byte compare (no trimming) — facts |
| CRLF vs LF | byte-for-byte; samples are generated LF — facts silent, oracle `diff` does the same |
| expect-error text spanning the closing `}` | cut at `}` or end of line, blanks trimmed (oracle `sed`) |
| `System.SysUtils` inside a longer identifier | no rewrite (whole-word rule R8) |
| Same file listed twice (arg + dir) | emitted twice, once per occurrence — caller's fault, documented |
| Empty directory argument | contributes no lines; with nothing else → usage error exit 2 (no target) |
| `--timeout 0` / negative / non-numeric | usage error exit 2 |

## 6. Interpretation notes (no external-design change requested)

- **D-1** The external design's "Outcome Struct (JSON)" contradicts its own
  "Output Format" (TSV outcome name + detail) and the facts. Facts win
  (stated in T06-facts.md); the JSON struct is not implemented. Recorded
  here for traceability; an external-design cleanup is recommended.
- **D-2** Compile step also runs under `--timeout`: the external design says
  a compiler run "killed by `--timeout` produces `skipped`", so the limit
  applies before the run step too.
- **D-3** `--help` exits 0 and prints usage (convenience; facts silent).
- **D-4** `pascal-delphi` examples use the `.dpr` extension in the
  `pascal-dialects` plugin — covered by R3 of `DetectDialect` and by the
  `*.pas`/`*.dpr` collection (R16), so a forced `--dialect delphi` run over
  `pascal-dialects/skills/pascal-delphi` verifies correctly (ITa-013).

## 7. Change history

| Date | Change | Reason |
|---|---|---|
| 2026-11-17 | Initial detailed design | first version for T06 |
| 2026-11-17 | Rewritten to the corrected external design | T06-facts.md corrections: dialect names (`fpc-delphi`, no `dialect-*` prefix), `--dialect`/`PATH...` CLI (no `--dir`/`--expect`), TSV outcomes (no JSON), `skipped` for missing compiler, `expected`/no-expected, delphi-copy rewrite (R8), `.dpr`/`{$APPTYPE CONSOLE}` → `delphi`, 10 s default timeout, `QLORA_DELPHI_PATH` dropped, `.pas`/`.dpr` only |
| 2026-11-18 | 2.1 `ReadEnvRec(const AFpc, AFpcUnleashed, AHome: string)` | env values are passed as arguments; only the program `qlora_verify` calls `GetEnvironmentVariable` (FPC 3.2.2 `setenv`-after-start trap); UT rows 052–054 retired, 121–123 added |
| 2026-11-18 | 2.17 `SortStrings(A: TStringArray)` | lists are carried as `TStringArray` (SysUtils), not open-array `out` parameters |
| 2026-11-18 | 2.19 `VerifyMain(…, const AEnv: TEnvRec, …)` | `VerifyMain` takes the `TEnvRec` from the caller and never reads the environment; only the program `qlora_verify` calls `GetEnvironmentVariable` + `ReadEnvRec` (2.1, T06b3). Module table (`src/qlora_verify.pas`) updated to match |
