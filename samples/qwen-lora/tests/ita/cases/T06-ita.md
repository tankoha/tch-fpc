# ITa Test Cases for T06: `qlora_verify`

## ITa-T06-001: Single objfpc file, example, exact output match

Source: `sample_001.pas`
```pascal
{$mode objfpc}
begin
  writeln('Hello');
end.
```

Expected output file: `sample_001.expected`
```text
Hello
```

Expected TSV:
```tsv
sample_001.pas	objfpc	compiled-run	accepted	
```

## ITa-T06-002: Single objfpc file, example, no expected file

Source: `sample_002.pas`
```pascal
{$mode objfpc}
begin
  // No output
end.
```

Expected TSV:
```tsv
sample_002.pas	objfpc	compiled-run	no-expected	
```

## ITa-T06-004: Single objfpc file, expect-error

Source: `sample_004.pas`
```pascal
{$mode objfpc}
// expect-error: Identifier not found "nonexistentproc"
begin
  nonexistentproc;
end.
```

Expected TSV:
```tsv
sample_004.pas	dialect-objfpc	compiled-run	{"compiled": false, "match": "expect-error"}
```

## ITa-T06-006: Single fpc-delphi file, example, exact match

Source: `sample_006.pas`
```pascal
{$mode delphi}
begin
  writeln('Hello from Delphi');
end.
```

Expected output file: `sample_006.expected`
```text
Hello from Delphi
```

Expected TSV:
```tsv
sample_006.pas	fpc-delphi	compiled-run	accepted	
```

## ITa-T06-007: Single unleashed file, example, exact match

Source: `sample_007.pas`
```pascal
{$mode unleashed}
begin
  WriteLn('Hello from Unleashed');
end.
```

Expected output file: `sample_007.expected`
```text
Hello from Unleashed
```

Expected TSV (with `QLORA_FPC_UNLEASHED` defined):
```tsv
sample_007.pas	unleashed	compiled-run	accepted	
```

Without `QLORA_FPC_UNLEASHED`: `unleashed	compiled-run	skipped	compiler not found`.

## ITa-T06-008: Single oxygene file, static-ok

Source: `sample_008.pas`
```pascal
namespace MyTest;
interface
  method Show;
implementation
  method Show;
  begin
    Console.WriteLine('Hello from Oxygene');
  end;
end.
```

Expected TSV:
```tsv
sample_008.pas	oxygene	static-only	static-ok	
```

## ITa-T06-009: Single oxygene file, static-reject

Source: `sample_009.pas`
```pascal
interface
  method Show;
implementation
  method Show;
  begin
    Console.WriteLine('Hello from Oxygene');
  end;
end.
```

Expected TSV:
```tsv
sample_009.pas	oxygene	static-only	static-reject	
```

## ITa-T06-010: Directory with mixed dialects (examples and pitfalls)

Directory contents:
```
001.pas (objfpc, example, exact)
002.pas (objfpc, example, no expected file)
003.pas (objfpc, example, compile-error)
004.pas (objfpc, pitfall, expect-error match)
005.pas (fpc-delphi, example, exact)
006.pas (unleashed, example, exact)
007.pas (oxygene, static-ok)
008.pas (oxygene, static-reject)
009.pas (objfpc, pitfall, compiled-but-should-fail)
```

Expected TSV (all files present, dialect and class per facts table):
```tsv
001.pas	objfpc	compiled-run	accepted	
002.pas	objfpc	compiled-run	no-expected	
003.pas	objfpc	compiled-run	compile-error	Identifier not found "nonexistentproc"
004.pas	objfpc	compiled-run	accepted	
005.pas	fpc-delphi	compiled-run	accepted	
006.pas	unleashed	compiled-run	accepted	
007.pas	oxygene	static-only	static-ok	
008.pas	oxygene	static-only	static-reject	
009.pas	objfpc	compiled-run	compiled-but-should-fail	
```

## ITa-T06-011: Timeout handling (example)

Source: `sample_011.pas`
```pascal
{$mode objfpc}
begin
  while true do ;
end.
```

Expected TSV (with `--timeout 1`):
```tsv
sample_011.pas	objfpc	compiled-run	timeout	1
```

## ITa-T06-012: Timeout handling (compile-time)

Source: `sample_012.pas`
```pascal
{$mode objfpc}
// Never finishes compiling
begin
  while true do ;
end.
```

Expected TSV (with `QLORA_FPC` set to a script that hangs):
```tsv
sample_012.pas	objfpc	compiled-run	skipped	timeout
```

## ITa-T06-013: Oracle run (each skill directory, one dialect)

Environment:
```bash
export QLORA_DIALECT_SKILLS=/path/to/pascal-dialects/skills
```

Command:
```bash
for skill in pascal-fpc-objfpc pascal-fpc-delphi pascal-unleashed pascal-delphi pascal-oxygene; do
  qlora_verify --dialect "${skill#pascal-}" "$QLORA_DIALECT_SKILLS/$skill"
done
```

Expected TSV (one line per file, per skill directory):
- `pascal-fpc-objfpc`: every file `objfpc	compiled-run	accepted` (per `tests/check-examples.sh` OK lines).
- `pascal-fpc-delphi`: every file `fpc-delphi	compiled-run	accepted`.
- `pascal-unleashed`: every file `unleashed	compiled-run	accepted`.
- `pascal-delphi`: every file `delphi	fpc-approx	accepted`.
- `pascal-oxygene`: every file `oxygene	static-only	static-ok` (not `accepted`).