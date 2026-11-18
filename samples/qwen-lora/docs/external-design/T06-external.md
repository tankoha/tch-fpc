# External Design for Task T06: `DialectVerify` unit and `qlora_verify` program

## Scope

### Unit: `DialectVerify`
Functions for detecting the dialect of a Pascal source file, compiling it with the correct compiler/flags, running the program, and classifying the verification outcome.

### Program: `qlora_verify`
A CLI tool that:
- Verifies one or more Pascal source files or directories (recursively).
- Outputs one TSV line per file: filename, dialect, verification class, and outcome.

## Dialect Detection
The unit shall scan the file for the following indicators:

| Indicate               | Dialect    |
|------------------------|------------|
| `{$mode objfpc}`       | `objfpc`   |
| `{$mode delphi}`       | `fpc-delphi` |
| `{$mode unleashed}`    | `unleashed` |
| `namespace` (first word) | `oxygene`  |

Detection logic:
1. The first line that starts with `{$mode ` (after optional whitespace) determines the dialect (`objfpc`, `fpc-delphi`, or `unleashed`).
2. If no mode directive is found, a first word `namespace` determines `oxygene`.
3. If the file extension is `.dpr` or the source contains `{$APPTYPE CONSOLE}`, the dialect is `delphi`.
4. If none of the above apply, the dialect is `objfpc`.

No real Delphi compiler is used. Without `--dialect`, the dialect is always detected as above.

## Verification Classes

| Class          | Description                                                                                     |
|----------------|-------------------------------------------------------------------------------------------------|
| `compiled-run` | Program compiles, runs, and matches its expected output exactly (or passes `expect-error` if specified). |
| `fpc-approx`   | Compile and run with FPC in Delphi mode (approximation; not a guarantee of real Delphi compatibility). |
| `static-only`  | No compiler available or static check only (Oxygene).                                           |

## Program `qlora_verify`

### CLI Synopsis
```bash
qlora_verify [--dialect D] [--timeout SECONDS] PATH...
```

- `PATH` is a file or directory. In a directory, all `*.pas` and `*.dpr` files are verified recursively.
- `--dialect D` forces the dialect for every file: `objfpc`, `fpc-delphi`, `unleashed`, `delphi`, or `oxygene`. Without it, the dialect is detected.
- `--timeout SECONDS` overrides the time limit per program (default 10 s).
- Exit codes: 0 if every file is `accepted`, `static-ok`, or `skipped`; 1 otherwise; 2 for usage errors.

### Compilers and Environment Variables

| Dialect      | Compiler Variable          | Default                     | Flags                                                                                     | Class         |
|--------------|----------------------------|-----------------------------|-------------------------------------------------------------------------------------------|---------------|
| `objfpc`     | `QLORA_FPC`                | `fpc`                       | `-vew -l- -FE<out> -FU<out>`                                                              | `compiled-run`|
| `fpc-delphi` | `QLORA_FPC`                | `fpc`                       | same                                                                                      | `compiled-run`|
| `unleashed`  | `QLORA_FPC_UNLEASHED`      | `$HOME/unleashed/fpc/bin/fpc` | same                                                                                      | `compiled-run`|
| `delphi`     | `QLORA_FPC_UNLEASHED`      | `$HOME/unleashed/fpc/bin/fpc` | same + `-Mdelphiunicode -Manonymousfunctions -Mfunctionreferences -Minlinevars`, on a copy with `System.X` → `X` and `System.Generics.X` → `Generics.X` (whole-word only) | `fpc-approx` |
| `oxygene`    | none                       | none                        | (static check only)                                                                       | `static-only` |

- Missing compilers produce `skipped` (not a failure).
- A compiler run that fails to start or is killed by `--timeout` produces `skipped`.

### Output Format
TSV on stdout, one file per line:
```
<file> <dialect> <class> <outcome> <detail>
```

- `<detail>` is empty or a single line (first error message or exit code), max 200 chars, tabs replaced by spaces.
- `<outcome>` is one of the outcomes above (e.g., `accepted`, `compile-error`).

## Expected-Output File

If a file with the same base name and extension `.expected` exists next to the source (regardless of the source's extension), its content is the **exact** expected output. Without it, `no-expected` is reported for programs that compile and run.

## System Flow
See file `T06-system.md`.