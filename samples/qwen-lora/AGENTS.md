# Development process rules (SDLC)

Japanese version: [AGENTS_jp.md](AGENTS_jp.md). English is canonical.

This project is developed with phases and owners kept apart. The main agent
(Mistral Large 3) and the sub-agent (Qwen3.8, started with `delegate_task`)
both follow these rules. Part of them is enforced by a shell hook
(`sdlc.bb`). When the hook blocks you, do what the reason says; do not work
around it.

Always give absolute paths: for `write_file` and `patch`, and for the
`workdir` of `terminal`. With a relative path the hook cannot tell the
project and stops the operation.

## Owners

| Phase | Owner | Output |
|---|---|---|
| External design | main (Mistral) | `docs/external-design/`, `docs/system-flow/` |
| ITa / ITb / ST test cases | main (Mistral) | `tests/ita/cases/`, `tests/itb/cases/`, `tests/st/cases/` |
| Detailed design | sub (Qwen) | `docs/detailed-design/`, `docs/program-flow/` |
| UT cases, implementation, UT run | sub (Qwen) | `tests/ut/`, `src/`, `test-results/ut.md` |
| ITa / ITb run | sub (Qwen) | `test-results/ita.md`, `test-results/itb.md` |
| Confirming ITa / ITb NG results, isolating the cause | main (Mistral) | `docs/defects.md` |

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

One delegation covers one unit or one program. The sub-agent's context is
64k tokens.

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
  with "[廃止]" (retired) and the reason in its notes.
- A test that does not pass is reported. If possible, record symptom and
  cause in `docs/defects.md` and put the ID in the result table's Defect ID
  column. Format: skill `sdlc-defect-record`.

## UT completion

UT is complete only when both hold; `sdlc.bb gate ut . --run` decides.

1. Coverage: running the UT cases executes 100 % of the lines of `src/`
   (`:coverage-kind :c0`; kcov measures lines, not branches).
2. Execution: every case is "OK", or "NG" with a defect ID recorded: 100 %.

ITa / ITb / ST are complete at 100 % execution (`sdlc.bb gate ita` etc.).

## This project: Pascal and tch-fpc

The sample fine-tunes Qwen3.8 27B with LoRA on two GPUs, in Pascal on top of
tch-fpc. Its design is `../../docs/DESIGN.md`, sections 8.9–8.19; the task
list with the acceptance oracle of each task is 8.18.

- Everything is Free Pascal 3.2.2, `{$mode objfpc}{$H+}` in every file.
  Units `Tch`, `TchNN` (tch-fpc, in `../../src`); HTTP and JSON with
  `fphttpclient`, `fpjson`; processes with `process`. No Python, no Node.js;
  bash only for one-line wrappers.
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
