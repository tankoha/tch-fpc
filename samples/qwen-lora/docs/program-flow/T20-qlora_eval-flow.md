# Program flow — `qlora_eval` (T20)

Source of truth: the T20 facts (`tasks/T20-facts.md`, the run and the
messages final, checked by the harness owner), the T20 detailed design
(`../detailed-design/T20-detailed.md`), and T20a. This document is the
flow of the evaluation run only; the function internals are in the
detailed design.

## 0. Program and entry

- `src/qlora_eval.pas` (unit `EvalCli`): the thin `main` — reads the
  verifier's environment (`QLORA_FPC`, `QLORA_FPC_UNLEASHED` and
  `HOME`, as with `qlora_verify`) into the `TEnvRec` (`DialectVerify`,
  T06) of the run, and calls
  `RunEvalCLI(argv, Env, WorkRoot, Out, Err)`, printing the `Out` lines
  to stdout and the `Err` lines to stderr and exiting with the returned
  code (the detailed design 1; nobody changes the wrapper).
- Command line (detailed design 3.3):
  `qlora_eval [--timeout S] [--speed-ratio R] TASKS BEFORE [AFTER]
  OUTDIR`:
  - `--timeout S`: the verification seconds per reply (default 10, 1…3600).
  - `--speed-ratio R`: the after/before generation ratio of H49 (> 0,
    not measured without it).
  - `TASKS` the JSONL task file, `BEFORE` the recorded replies of the
    before model, `AFTER` the replies of the after model (optional),
    `OUTDIR` the directory of `results.tsv` and `report.md`.
- No `QLORA_*` variable of the run itself: the evaluation needs no
  model file; the `QLORA_*` variables of `DialectVerify` (T06) apply
  only to the verification of the judged blocks.
- The tch-fpc units (`Tch`, `TchNN`, `TchJit`) are **not** used: none of
  the run has a tensor operation.

## 1. Parse the command line (3.3 of the detailed design)

```
parse AArgs into the config, default Timeout 10, SpeedRatio NaN
  --timeout / --speed-ratio: value missing, or an invalid value
    -> the message (4.1) + K_EVAL_USAGE, exit 2
  an unknown -option
    -> the message (4.1) + K_EVAL_USAGE, exit 2
  the positionals not 3..4
    -> the message (4.1) + K_EVAL_USAGE, exit 2
otherwise
  Tasks, Before, After (if 4th), OutDir (trailing delimiter), Timeout, SpeedRatio
```

## 2. Load the tasks (3.5)

```
LoadTasks(C.Tasks)
  the tasks file missing                         -> no file <path>          (4.2)
  per line (blank lines skipped):
    not an object                                 -> task line <n>: not a JSON object
    id / dialect / prompt missing or not a string -> task line <n>: <name>
    a regression task:
      check not tool-call / contains              -> task line <n>: check
      tool (resp. text) missing or not a string   -> task line <n>: tool (text)
    a Pascal task:
      expected_stdout missing or not a string     -> task line <n>: expected_stdout
  (a raise -> the message alone, exit 1)
```

## 3. Judge the replies (3.6–3.8)

```
results := the header model task seed think dialect outcome compiled passed
before := JudgeFile(C.Before, 'before')
            the file missing                        -> no file <path>        (4.3)
            a line not an object                    -> <file> line <n>: not a JSON object
            the task name not in the tasks          -> <file> line <n>: no task <id>
            per reply (in file order) JudgeReply:
              a regression task:
                tool-call: any tool_calls element object named ATask.Tool -> Passed
                contains:  Pos(ATask.Text, content) > 0                   -> Passed
                                              outcome check-ok | check-failed
              a Pascal task:
                no closed fenced block in content   -> outcome no-block
                else write A.Dir/main.pas (= the first block), A.Dir/main.expected (= the task's expected_stdout)
                VerifyFile (T06, dialect = the task's):
                  Compiled := outcome not compile-error,static-ok,static-reject,skipped
                  Passed   := outcome accepted and no expect-error: marker in the block
after (if C.After <> '') := JudgeFile(C.After, 'after')   (the same steps)
  (a raise -> the message alone, exit 1)
```

The work of each reply is in
`<workroot>/<model>/<line no., 1-based>/`; the verifier's own work dir
is `<...>/build` (the `DialectVerify` rule, T06).

## 4. Write the report (3.10, 3.11)

```
report.md := '# Evaluation report'
  ''
  'Tasks: <np> Pascal, <nr> regression. Replies: before <b>, after <a>.'
  'Compiled and passed as passed/replies; "verified" is objfpc,fpc-delphi,
   unleashed,oxygene together; delphi is an approximation, not in it.'
  for the thinking modes on -> off:
    '## Thinking on|off'
    the columns: the dialects present (ascending, no regression),
      verified, tool-call, contains
      (a delphi column: '(approximation)')
      a row: the cells of before (and of after, when present):
        compiled = compiled/replies, passed = passed/replies, '-' when none;
        a regression column: no compiled cell;
        when present: fail -> pass and pass -> fail of the column
      and, when present: the adoption of the mode (the detailed design 2.10):
        verified aggregate rises (…)
        flips significant (…)          (the exact sign test, p < 0.05, more fail -> pass)
        no verified dialect falls by more than 5 points (…)
        tool-call check loses nothing (…)
        H49 speed at least 0.7 of before (…)   | not measured (no --speed-ratio,
                                               marks no Failed)
        adopt: yes | no | undecided
results.tsv := the header, then one line per reply (before first, then after)
OUTDIR/results.tsv, OUTDIR/report.md  (OUTDIR created if absent)
```

`AOut`:

```
<n> replies judged into <outdir>
```

`n` = the before replies + the after replies. **exit 0**.

## 5. Errors and messages (4)

In the order of the detailed design 4 (the whole line, the UT row
compares it, `AssertEquals`):

1. usage (exit 2, the message **and** `K_EVAL_USAGE`): `<option>: value
   missing`, `--timeout: an integer in 1..3600, got "<v>"`,
   `--speed-ratio: a number > 0, got "<v>"`, `unknown option <arg>`,
   `expected TASKS BEFORE [AFTER] OUTDIR, got <n> arguments`;
2. tasks (exit 1): `no file <path>`, `task line <n>: not a JSON
   object`, `task line <n>: id|dialect|prompt|tool|text|check|
   expected_stdout`;
3. replies (exit 1): the task errors, `no file <path>` (BEFORE/AFTER),
   `<file> line <n>: not a JSON object`, `<file> line <n>: no task
   <id>`.

## 6. Change history

| Date | Change | Reason |
|---|---|---|
| T20a (initial) | First edition: entry, the parse, the tasks, the judging of the recorded replies, the report (tables and adoption), the messages in their order. | new program (T20; documents only in T20a — no source code, no tests) |
