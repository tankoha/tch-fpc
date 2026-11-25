# T20 External Design: qlora_eval

## Purpose

`qlora_eval` judges recorded model replies against evaluation tasks, reports
pass/fail counts by dialect and check type, computes flips between before/after
models, runs a sign test, and evaluates the adoption rule. It is the "Runs",
"Report", and "Adoption rule" step of DESIGN 8.17–8.18.

## Command Line

```
qlora_eval [options] TASKS BEFORE [AFTER] OUTDIR
```

Arguments:
- `TASKS`: path to the evaluation tasks file (JSONL, one task per line)
- `BEFORE`: path to the before-model replies file (JSONL, one reply per line)
- `AFTER`: path to the after-model replies file (optional; if omitted, only the before model is evaluated)
- `OUTDIR`: directory for output files (created if missing)

Options:
- `--timeout S`: verification timeout per reply in seconds, 1–3600 (default: 10)
- `--speed-ratio R`: after/before generation speed ratio, a number > 0 (default: not measured)

Defaults:
- Timeout: 10 seconds
- SpeedRatio: NaN (not measured)

## Task Format

Each line in `TASKS` is a JSON object:

A Pascal task:
- `id` (string): task identifier
- `dialect` (string): one of `objfpc`, `fpc-delphi`, `unleashed`, `delphi`, `oxygene`
- `prompt` (string): the instruction
- `expected_stdout` (string): the expected program output

A regression task:
- `id` (string): task identifier
- `dialect` (string): must be `regression`
- `prompt` (string): the instruction
- `check` (string): either `tool-call` or `contains`
- For `tool-call`: `tool` (string): expected tool name
- For `contains`: `text` (string): text that must appear in the reply

## Reply Format

Each line in `BEFORE` and `AFTER` is a JSON object:
- `task` (string): the task id
- `seed` (integer): random seed used
- `think` (boolean): thinking mode on or off
- `content` (string): the model's text reply
- `tool_calls` (optional array): array of `{name, arguments}` objects

## Judging a Pascal Reply

The first closed fenced block of `content` is extracted:
- Opening: a line starting with three backquotes
- Closing: a line equal to three backquotes (trailing spaces ignored)
- The block between becomes `main.pas`

Verification:
1. `main.pas` is written with `main.expected` containing the task's `expected_stdout`
2. Verified with `DialectVerify.VerifyFile` in the task's dialect
3. **Compiled**: true unless outcome is `compile-error`, `static-ok`, `static-reject`, or `skipped`
4. **Passed**: true when outcome is `accepted` AND the block has no `expect-error:` marker

When there is no fenced block:
- Outcome: `no-block`
- Compiled: false
- Passed: false

## Judging a Regression Reply

For `tool-call` check:
- Passed when any element of `tool_calls` has `name` equal to the task's `tool`
- Content is not examined

For `contains` check:
- Passed when `content` contains the task's `text` (literal string match, case-sensitive)
- Tool calls are not examined

## Columns

The report tracks these columns:
- Each dialect present in replies: `objfpc`, `fpc-delphi`, `unleashed`, `delphi`, `oxygene`
- `verified`: aggregate of `objfpc`, `fpc-delphi`, `unleashed`, `oxygene` (Delphi is marked an approximation, not verified)
- `tool-call`: regression tasks with check `tool-call`
- `contains`: regression tasks with check `contains`

## Flips

Flips pair replies by task and seed within a thinking mode:
- **B**: count of fail → pass (before failed, after passed)
- **C**: count of pass → fail (before passed, after failed)
- Replies without a partner in the other file are not counted

## Sign Test

Exact two-sided sign test of B against C:

```
p = min(1, 2 × Σ_{k=0}^{min(B,C)} C(B+C, k) / 2^(B+C))
```

Returns 1 when B + C = 0.

## Adoption Rule

Per thinking mode, one line per condition:

1. **Verified aggregate rises**: pass rate of `verified` column increased
2. **Flips significant**: B > C and p < 0.05 (more fail→pass than pass→fail, statistically significant)
3. **No verified dialect falls by more than 5 points**: for each verified dialect, the after rate ≥ before rate − 0.05
4. **Tool-call check loses nothing**: after passed count ≥ before passed count
5. **H49 speed at least 0.7 of before**: `--speed-ratio` must be ≥ 0.7 (not measured when NaN)

Each condition reports `yes` or `no` (or `not measured` for H49).

**Verdict**:
- `adopt: yes`: all conditions passed
- `adopt: no`: any condition failed
- `adopt: undecided`: no condition failed, but H49 was not measured

An empty series (NaN rate) neither rises nor falls.

## Output Files

Written to `OUTDIR`:

1. `results.tsv`: one header line, then per reply (before then after)
   - Columns: `model`, `task`, `seed`, `think` (on/off), `dialect`, `outcome`, `compiled` (1/0), `passed` (1/0)
   - `outcome`: verifier outcome, or `no-block`, `check-ok`, `check-failed`

2. `report.md`: evaluation report
   - Header: task counts, reply counts
   - Per thinking mode (on first, then off)
   - Table of columns with compiled/passed counts
   - With AFTER: flips and adoption lines

## Exit Codes

| Code | Meaning |
|---|---|
| 0 | Success: replies judged, files written |
| 1 | Runtime error: file not found, invalid JSON, missing task |
| 2 | Usage error: wrong arguments, unknown option, missing value |

## Error Conditions

### Usage Errors (exit 2)

From `ParseEvalArgs`:
- `<option>: value missing`
- `--timeout: an integer in 1..3600, got "<value>"`
- `--speed-ratio: a number > 0, got "<value>"` (NaN and ±Inf refused)
- `unknown option <arg>`
- `expected TASKS BEFORE [AFTER] OUTDIR, got <n> arguments`

### Runtime Errors (exit 1)

From `LoadTasks`:
- `no file <path>`
- `task line <n>: not a JSON object`
- `task line <n>: id` / `dialect` / `prompt` / `expected_stdout` / `tool` / `text` / `check`

From `RunEvalCLI`:
- `<file> line <n>: not a JSON object`
- `<file> line <n>: no task <id>`

### Success Output

On success (exit 0), stdout contains:
```
<n> replies judged into <outdir>
```

## Units Used

- `DialectVerify` (T06): `VerifyFile`, verifier outcomes, `ExtractExpectErrorText`
