# T20 System Flow: qlora_eval

## Components Used

| Component | Purpose |
|---|---|
| `DialectVerify` | Verify programs in Pascal dialects, `VerifyFile`, `ExtractExpectErrorText` |

## Who Runs It

### Harness Owner (M3, M5)
- Evaluates the before model (M3, after initial fine-tuning)
- Evaluates the after model (M5, after the merge)
- Compares results with the adoption rule

### Generator Pipeline
- The evaluation tasks are prepared separately (M3)
- Replies are recorded outside the program via Ollama's `/api/chat` endpoint using `tests/ita/T20/ask.bb`

### User
- Reads the report for the adoption decision

## Inputs

### Evaluation Tasks (M3)
- File: `tasks.jsonl`
- Eleven Pascal tasks in four dialects (objfpc, fpc-delphi, unleashed, oxygene)
- Three regression tasks (tool-call, contains)
- Reference solutions accepted by `qlora_verify`
- Made by `tests/ita/T20/make-tasks.bb`

### Recorded Replies

The replies are recorded once with `tests/ita/T20/ask.bb`:
- Ollama's `/api/chat` endpoint
- The model's own renderer and parser
- Deployed sampling parameters
- Seeds 1 and 2
- Thinking on and thinking off modes
- Tool `get_weather` for a tool-call task

**Before model** (`replies-before.jsonl`):
- Replies from `qlora-m2-base` (before M2 merge)
- 56 replies (14 tasks × 2 seeds × thinking on and off)

**After model** (`replies-after.jsonl`):
- Replies from `qlora-m2` (after M2 merge)
- 56 replies (14 tasks × 2 seeds × thinking on and off)

Each reply line:
- `task`, `seed`, `think`, `content`
- Optionally `tool_calls` (array of `{name, arguments}`)

## Outputs

### Results File
- `results.tsv`: header plus one line per reply
- Columns: model, task, seed, think, dialect, outcome, compiled, passed
- Used for detailed analysis and debugging

### Report File
- `report.md`: Markdown evaluation report
- Summary of tasks and replies
- Tables by thinking mode
- Flips and adoption verdict when both models evaluated

### Adoption Decision
- The user reads `report.md`
- Verdict `adopt: yes` / `adopt: no` / `adopt: undecided`
- Based on verified aggregate, flips significance, dialect stability, tool-call check, speed ratio

## Downstream Consumers

### User (Harness Owner)
- Reads `report.md` to decide whether to adopt the after model
- Inspects `results.tsv` for detailed reply outcomes

### Defect Recording
- Failures tracked in `docs/defects.md` when needed

## Unit Test Environment

### Test Fixtures
- `tests/ut/EvalReport_test.pas`: helpers for `EvalReport`
  - `J(...)`: construct a `TJudged`
  - `Arr([...])`: construct a `TJudgedArray`
  - `Series(Dialect, N, Passed)`: N replies with given pass count
  - `Cat(A, B)`: concatenate arrays
  - `Joined(A)`: join by `|`
- `tests/ut/EvalCli_test.pas`: helpers for `EvalCli`
  - `PSum`: constant that prints `10`
  - `NewDir`, `SaveLines`, `FileText`
  - `TaskLine(...)`, `ReplyLine(...)`: construct task and reply JSON
  - `Task(...)`: construct a `TEvalTask`
  - `Judge(...)`: call `JudgeReply` in a fresh directory
  - `RunCli(...)`: call `RunEvalCLI` with test environment

### Test Environment
- `TmpDir`: temporary directory for test files
- `Env`: `ReadEnvRec('', '', HOME)` for stock FPC
- GPUs hidden (not used by these units)

### Build and Run
```
fpc -Mobjfpc -Sh -vew -l- -Fu<src> -Fu<tests/ut> -FU<work/bin> -Fl<libtch> -k-rpath=<libtch> tests/ut/<Unit>_test.pas
work/bin/<Unit>_test --all --format=plain
```

### Test Cases
- `EvalReport`: rows UT-EvalReport-001 to 009 (T20b1)
- `EvalCli`: rows UT-EvalCli-101 to 107 (T20b2)

## ITa Environment

### Driver
- `tests/ita/T20-run.sh`
- Builds `src/qlora_eval.pas` and `src/qlora_verify.pas`
- Runs twice from fresh directories outside the project

### Test Data
- `tests/ita/T20/tasks.jsonl`: 11 Pascal tasks + 3 regression tasks
- `tests/ita/T20/replies-before.jsonl`: 56 replies from rehearsal model (before)
- `tests/ita/T20/replies-after.jsonl`: 56 replies from rehearsal model (after)
- Made by `tests/ita/T20/make-tasks.bb` and `ask.bb`

### Command
```
<project root>/tests/ita/T20-run.sh
```
(Runs from outside the project, no `QLORA_MODEL` needed)

### Validation
- `tests/ita/T20/check.bb` verifies every reply again with `qlora_verify`
- Recomputes the report from judgements
- Compares results and report against expected values

### Result File
- `<project root>/work/ita/T20/summary.tsv`
- Columns: case, OK/NG, value, limit, note
- OK when value ≤ limit
- About 5 seconds per run

## ITa Cases

| Case ID | Title | Expected (short) |
|---|---|---|
| ITa-T20-001 | Result lines match reply order (model, task, seed, think, dialect) | 0 mismatches |
| ITa-T20-002 | Outcome, compiled, passed match verifier recomputation | 0 mismatches |
| ITa-T20-003 | Report table rows recomputed correctly | 0 missing |
| ITa-T20-004 | Adoption lines recomputed, sign test included | 0 missing |
| ITa-T20-005 | Second run: results.tsv and report.md byte-identical | 0 files differ |

## Acceptance

From DESIGN 8.18: "against the rehearsal model: the report of 8.17 from a fixed transcript of replies"

Measured with prototype:
- All ITa cases: 0 (limit 0)
- 112 replies (56 per model, 2 thinking modes)
- Rehearsal model: passes none of the 11 Pascal tasks before or after M2 merge
- Two unleashed replies compile
- 14 passes from regression checks
