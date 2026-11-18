# T06 facts (checked against the oracle)

These facts were checked by the harness owner against DESIGN 8.16 and against
`tests/check-examples.sh` of the `pascal-dialects` plugin (the T06 oracle).
Where the current T06 documents differ, these facts win.

## Command line

```
qlora_verify [--dialect D] [--timeout SECONDS] PATH...
```

- `PATH` is a file or a directory. In a directory, every `*.pas` and `*.dpr`
  file is taken, recursively, sorted by path.
- `--dialect D` sets the dialect of every file: `objfpc`, `fpc-delphi`,
  `unleashed`, `delphi` or `oxygene`. Records of the data pipeline carry their
  dialect, and the oracle needs it: `pascal-unleashed/pitfalls/p06_missing_mode.pas`
  says `{$mode objfpc}` but is checked with the Unleashed compiler.
- Without `--dialect`, the dialect is detected:
  1. a line that starts (after blanks) with `{$mode objfpc}` gives `objfpc`,
     `{$mode delphi}` gives `fpc-delphi`, `{$mode unleashed}` gives
     `unleashed` (case-insensitive; the first such line wins);
  2. otherwise a first word `namespace` gives `oxygene`;
  3. otherwise a `.dpr` file or a `{$APPTYPE CONSOLE}` line gives `delphi`;
  4. otherwise `objfpc`.
- `--timeout` is the run limit per program, default 10 seconds.
- Exit code: 0 when every file is `accepted`, `static-ok` or `skipped`;
  1 otherwise; 2 for a usage error.

## Compilers and flags (DESIGN 8.16)

| Dialect | Compiler | Flags | Class |
|---|---|---|---|
| `objfpc` | `QLORA_FPC` (default `fpc`) | `-vew -l- -FE<out> -FU<out>` | `compiled-run` |
| `fpc-delphi` | `QLORA_FPC` | same | `compiled-run` |
| `unleashed` | `QLORA_FPC_UNLEASHED` (default `$HOME/unleashed/fpc/bin/fpc`) | same | `compiled-run` |
| `delphi` | `QLORA_FPC_UNLEASHED` | same plus `-Mdelphiunicode -Manonymousfunctions -Mfunctionreferences -Minlinevars`, on a copy (below) | `fpc-approx` |
| `oxygene` | none | static check only | `static-only` |

- `<out>` is a fresh directory per file under `work/verify/`.
- The `delphi` copy is written to `<out>` with the same file name. In it,
  `System.X` becomes `X` for X in SysUtils, Classes, Math, StrUtils,
  DateUtils, Types, Variants, TypInfo, Rtti, and `System.Generics.X` becomes
  `Generics.X` for X in Collections, Defaults (whole words only).
- There is no real Delphi compiler. `QLORA_DELPHI_PATH` is not used.
- A compiler that does not exist gives the outcome `skipped`; this is not a
  failure.

## Example or pitfall

A source that contains `expect-error: <text>` is a pitfall. `<text>` runs to
the closing `}` of that comment (or to the end of the line), with blanks
trimmed. Pitfalls are written as `{ ... expect-error: <text> }`.

- Pitfall: compiling must fail, and the compiler log must contain `<text>`
  literally. Outcomes: `accepted`, `compiled-but-should-fail`,
  `wrong-error` (the detail is the log's first `Error:` or `Fatal:` line).
- Example: compile, then run the program inside `<out>`, with stdout and
  stderr captured together, under the time limit. The captured text must
  equal the `.expected` file next to the source (same base name) byte for
  byte. Outcomes: `accepted`, `compile-error` (detail: first error line),
  `run-error` (exit code not 0), `timeout`, `wrong-output`, `no-expected`
  (no `.expected` file). There is no partial match.
- Oxygene: `static-ok` when the first word of the source is `namespace`,
  else `static-reject`. Nothing is compiled.

## Output

One line per file on stdout, tab-separated:

```
<path> <dialect> <class> <outcome> <detail>
```

`<detail>` is empty or one line (the first error line, or the exit code),
at most 200 characters, with tabs replaced by blanks.

## The oracle (ITa)

`QLORA_DIALECT_SKILLS` names the `skills/` directory of `pascal-dialects`.
For each skill directory, run `qlora_verify` on it with the dialect of
that skill:

| skill | `--dialect` |
|---|---|
| `pascal-fpc-objfpc` | `objfpc` |
| `pascal-fpc-delphi` | `fpc-delphi` |
| `pascal-unleashed` | `unleashed` |
| `pascal-delphi` | `delphi` |
| `pascal-oxygene` | `oxygene` |

Expected: every file of the first four is `accepted`, as the line `OK` of
`tests/check-examples.sh` (in the plugin directory, the parent of `skills/`)
says for it; every Oxygene file is `static-ok` (that script skips Oxygene).
The ITa run executes both and compares them file by file.
