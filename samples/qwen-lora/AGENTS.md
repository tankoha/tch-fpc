# Development process rules (SDLC)

Japanese version: [AGENTS_jp.md](AGENTS_jp.md). English is canonical.

This project is developed with phases and owners kept apart. The main agent
(GLM 5 on Bedrock from T03; Mistral Large 3 until T02) and the sub-agent
(GLM 5 on Bedrock from T05b3, started with `delegate_task`; the local Qwen3.8
until T02, other models in between)
both follow these rules. Part of them is enforced by a shell hook
(`sdlc.bb`). When the hook blocks you, do what the reason says; do not work
around it.

Always give absolute paths: for `write_file` and `patch`, and for the
`workdir` of `terminal`. With a relative path the hook cannot tell the
project and stops the operation.

## Working with the tools

Each rule below avoids a refusal seen in T06; every refusal costs a long
re-think.

- Probe scripts go into a file under `work/` and run as `bash <file>`.
  Do not pass code inline (`bash -c`, `bb -e`, a heredoc into an
  interpreter): Hermes asks for approval, and this run cannot give it.
- Do not clean up with `rm -r` or `find -delete` (refused the same way).
  For a fresh directory use `mktemp -d <proj>/work/tmp.XXXXXX`; `work/` is
  git-ignored and cleaned by the user.
- Every `.pas` file starts with `{$mode objfpc}{$H+}`, probes and
  scratch files too; load the skills with `skill_view` before writing the
  first one (the Pascal guard blocks otherwise).
- Read large files by line range (`read_file` with offset and limit) after
  locating the place with `search_files`; do not read whole files you have
  read before. Context compression costs minutes and the details.
- Before `patch`, read the lines you change, and take `old_string` from
  that read. Read a file before overwriting it.
- When a hook blocks, do what its reason says. Do not read the hooks'
  sources (`~/.hermes/agent-hooks/`).
- Do only the step you were given. When a gate or check fails because a
  later step (another delegation or task file) is not done yet, report it and
  stop; do not do that step (T01a: a sub-agent asked for UT case tables ran
  gate ut, then wrote the units and tests for 1.5 hours).

## Owners

| Phase | Owner | Output |
|---|---|---|
| External design | main (GPT-OSS) | `docs/external-design/`, `docs/system-flow/` |
| ITa / ITb / ST test cases | main (GPT-OSS) | `tests/ita/cases/`, `tests/itb/cases/`, `tests/st/cases/` |
| Detailed design | sub (Qwen Coder) | `docs/detailed-design/`, `docs/program-flow/` |
| UT cases, implementation, UT run | sub (Qwen Coder) | `tests/ut/`, `src/`, `test-results/ut.md` |
| ITa / ITb run | sub (Qwen Coder) | `test-results/ita.md`, `test-results/itb.md` |
| Confirming ITa / ITb NG results, isolating the cause | main (GPT-OSS) | `docs/defects.md` |

- The main agent does not write anything from detailed design on (`src/`,
  `docs/detailed-design/`, `docs/program-flow/`, `tests/ut/`) itself. It
  always delegates with `delegate_task`.
- The sub-agent does not change the external design, the system flow or the
  ITa/ITb/ST cases. If it finds a change necessary, it returns an "external
  design change request" with the reason and the place in its final report.

## How to delegate (main agent)

The sub-agent knows nothing of the parent's conversation. `goal` and
`context` of `delegate_task` always contain:

- the skills to read with `skill_view`: `sdlc-detailed-design` for detailed
  design, `sdlc-ut-tdd` for UT and implementation, `sdlc-it-execution` for
  ITa/ITb; and always `pascal-fpc-objfpc` and `tch-fpc`
- the external design and system flow files concerned
- the sections of `../../docs/DESIGN.md` the task is built from
- the completion condition (which outputs, updated how)

One delegation covers one unit or one program. Keep each delegation to a step of the task file, not the whole task: small
steps finish in minutes, a whole task did not finish in hours.

When a task file is given (`tasks/<Txx>.md`), a hook puts "First read <task
file>" at the head of every delegated goal. Keep your own goal short and do
not restate the task file's rules in other words.

After each delegation returns, run `~/.hermes/agent-hooks/sdlc.bb review <project root>`
(terminal, absolute path) and compare its numbers and changed files with the
sub-agent's report. Until you do, a hook blocks the next `delegate_task` and
any write to `test-results/` or `docs/defects.md`. A run may use at most 4
delegations; plan the steps so they fit.

Before writing a result or defect row, read the output files it cites
(`.out`, `.err`, `.diff`) with `read_file` and copy from them. Never write a
message, a count or a cause you have not read (T01c recorded error messages
the program never printed).

## Updating the design documents

- When a fix or change alters the external design or the system flow, the
  main agent updates them in the same piece of work.
- When it alters the detailed design or the program flow, the sub-agent
  updates them in the same piece of work.
- If not updated, the final report says why no update was needed.

## Tests

- Test cases come before the implementation. A source file without the UT
  case table `tests/ut/cases/<unit>.cases.md` cannot be written (the hook
  stops it).
- Test cases are never deleted. Changes that reduce cases, test procedures
  or result rows are stopped by the hook. A case no longer needed stays,
  with the literal `[廃止]` (retired) followed by the reason in its notes:
  `[廃止] replaced by UT-X-012`. Not `[廃止: ...]`: the gate recognises only
  the literal and would count the row as not run.
- UT cases cover, for every text, name and path input: upper and lower case
  (and texts compared literally), the default when an environment variable
  or option is not set, and a bare name or relative path as well as an
  absolute one. Line coverage of 100 % does not show these.
- When a test you have just written fails, first compare its expected value
  with the facts and the case table: in T02 the test, not the code, was
  wrong (126.5 rounds to 127.0 = $42FE0000; the test said $42FD0000).
  Never record a cause you have not shown (environment, memory corruption,
  precision): record what the output shows and "切り分け待ち" (to be isolated).
- Expected values of numeric cases are written in the case table as hex
  bits (`$42FE0000`), and the test copies them; do not convert by hand in
  the test.
- A test that does not pass is reported. If possible, record symptom and
  cause in `docs/defects.md` and put the ID in the result table's Defect ID
  column. Format: skill `sdlc-defect-record`.

## UT completion

UT is complete only when both hold; `sdlc.bb gate ut . --run` decides.

1. Coverage: running the UT cases executes 100 % of the lines of `src/`
   (`:coverage-kind :c0`; kcov measures lines, not branches).
2. Execution: every case is "OK", or "NG" with a defect ID recorded: 100 %.

ITa / ITb / ST are complete at 100 % execution (`sdlc.bb gate ita` etc.).

- Run ITa as a user would: environment variables that override defaults
  (`QLORA_*`) unset unless the case sets them, and the program started from
  a working directory outside the project. Compare with the task's oracle.
- A gate prints OK also when NG rows carry a defect ID. Report the numbers
  (cases run, OK, NG, open defects), never "passed" while NG rows remain.

## This project: Pascal and tch-fpc

The sample fine-tunes Qwen3.8 27B with LoRA on two GPUs, in Pascal on top of
tch-fpc. Its design is `../../docs/DESIGN.md`, sections 8.9–8.19; the task
list with the acceptance oracle of each task is 8.18.

- Everything is Free Pascal 3.2.2, `{$mode objfpc}{$H+}` in every file.
  Units `Tch`, `TchNN` (tch-fpc, in `../../src`); HTTP and JSON with
  `fphttpclient`, `fpjson`; processes with `process`. No Python, no
  Node.js, also not through a script file (the guard hook blocks
  `python*`, `node`, `nodejs` in `terminal`: `:forbidden-commands` of
  `.sdlc/config.edn`); bash, grep, awk and sed for one-line work.
- One unit per file in `src/`, named as in DESIGN 8.12 (`GgufFile`,
  `GgufQuant`, ...). Programs are `src/qlora_*.pas`.
- Unit tests use fpcunit: `tests/ut/<Unit>_test.pas` is a program with a
  `TTestCase` class whose published methods are `procedure Test...`, run by
  `consoletestrunner`. Case IDs are `UT-<UNIT>-NNN`.
- Build line (what `.sdlc/ut-coverage.sh` uses):
  `fpc -Mobjfpc -Sh -gl -Fusrc -Fu../../src -FU<units> -Fl<libtch> -k-rpath=<libtch> <file.pas>`.
- Tests that need a model file read its path from an environment variable
  (`QLORA_MODEL` for the rehearsal GGUF) and are skipped without it.
- Never write generated data, model files or results into the repository:
  they go to `work/` (git-ignored).
- The GPUs are not used by tests. `.sdlc/ut-coverage.sh` hides them.
- Numeric code is checked against an oracle written first (DESIGN 8.18):
  a hand-computed case, a slow scalar version, or a reference from
  `llama-server`.
