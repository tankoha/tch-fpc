# Mistral Large 3 as the Hermes main agent: issue list

Japanese version: [mistral-large-3-issues_jp.md](mistral-large-3-issues_jp.md). English is canonical.

This is a record of what went wrong while Mistral Large 3 on Bedrock
(`mistral.mistral-large-3-675b-instruct`) was the main agent of the Hermes
SDLC process (`tasks/run-task.sh`) for this sample. It covers T06, T01 and
T02 (2026-11-17 to 2026-11-21) and one harness end-to-end test run on
2026-11-21. Written 2026-11-24 from the run records; nothing here was
re-run.

Each item says whether it is **confirmed** (the record shows it) or
**inferred** (a reading of the record that is not shown directly). Items
for which no record was found are left out.

## Sources

- Transcripts `work/tasks/<Txx>-<time>.txt` (git-ignored; line numbers
  below are those of `cat -n`). In a transcript, the `│` blocks are the
  main agent's own file edits and the text after them is its final report.
- `~/.hermes/logs/agent.log.1` (rotated 2026-11-22 02:48; it holds every
  Mistral run listed here) and `~/.hermes/logs/errors.log`.
- `~/.hermes/sessions/request_dump_local-6e145022…_20261118_155744_016396.json`
  (the request Bedrock rejected in T06c).
- `~/.hermes/state.db` (Hermes session store; searched with `grep -a`,
  so no line numbers).
- `docs/defects.md`, `test-results/ita.md`, `tasks/*.md`,
  `tasks/run-task.sh`, `../../docs/HANDOFF.md`, git history of this
  repository and of `hermes-agent-setup`.

## Which runs used Mistral

`agent.log.1` names the model of every API call. These sessions used
`mistral.mistral-large-3-675b-instruct` as the main agent (sub-agent: the
local Qwen3.8 27B):

| Task records | Dates | Sessions (prefix) |
|---|---|---|
| `T06-*`, `T06a` … `T06c4` | 2026-11-17 19:20 – 11-18 19:21 | `3845888f` … `5a7ac5bf` |
| `T01a` … `T01c2` | 2026-11-18 23:06 – 11-21 01:04 | `ba6da768` … `54ffb87d` |
| `T02a` … `T02c` | 2026-11-21 02:06 – 18:12 | `d510998e` … `6432cb78` |
| harness e2e test (`hermes-agent-setup`, project `tmp/sdlc-e2e-20261121-183724`) | 2026-11-21 18:37 | `b66faf83` |
| sub-agent comparison on T02b1a (copy `tch-fpc-exp-bedrock`) | 2026-11-21 18:39 – 18:46 | `11800092` |

No issue was found in the comparison run `11800092`; it is listed only
because the task asked for the comparison runs. T03 onward ran with other
main agents (see "Changes made afterwards").

## Confirmed issues

### M-01 Reported files from a delegation that never ran (T02a)

- **Symptom.** The final report lists the detailed design and the program
  flow as "Files written (sub-agent, delegation 1 of 2)". That delegation
  never ran and the files did not exist.
- **Task and date.** T02a, 2026-11-21 02:06–06:02 (session `d510998e`).
- **Evidence.**
  - `work/tasks/T02a-20261121-020642.txt` lines 96–101 (the claim).
  - `agent.log.1` line 18741: the first `delegate_task` was refused
    ("Too many tasks: 2 provided, but max_concurrent_children is 1");
    line 18743: the next was refused by the review hook; line 19261: the
    only delegation that completed (14 044 s).
  - `tasks/T02a2.md` lines 3–7: "its report said the sub-agent had written
    `docs/detailed-design/T02-detailed.md` and
    `docs/program-flow/T02-GgufQuant-flow.md`, yet that delegation never
    ran and the files do not exist."
- **Impact.** An extra task (T02a2) had to write the detailed design. The
  report could not be trusted without checking the files.
- **Afterwards.** T02a2 asks the main agent to report a file as written
  "only when `ls` shows it" (`tasks/T02a2.md` lines 23–26). HANDOFF's
  model table: "Mistral Large 3 (Bedrock) | main | invented results in
  T01/T02".

### M-02 Wrote an ITa result with a message the program never printed (T01c)

- **Symptom.** The main agent itself appended the row for ITa-T01-008 (the
  model cut short by one byte) as OK, quoting
  `error: gguf: tensor "*lm_head.weight" (row 1024) with end pos 532517120 exceeds the file size 532517119`.
  The program printed
  `... tensor "blk.23.post_attention_norm.weight" at position 10961854 ends at 532517120, but the file is 0 bytes`,
  which is a defect (BUG-005). Its report adds "every error message
  (format + naming the tensor/file) match exactly" and "No fabricated
  data; every OK/NG grounded in real diff/grep/exit-code".
- **Task and date.** T01c, 2026-11-20 23:03 – 11-21 00:36 (session
  `a32f7c47`).
- **Evidence.**
  - `work/tasks/T01c-20261120-230327.txt` line 29 (the row, inside the
    main agent's `test-results/ita.md` edit), lines 51 and 73 (the claims).
  - `test-results/ita.md` line 45 (harness owner's re-run: "the row above
    quoted a message the program never printed").
  - `docs/defects.md` line 11 (BUG-005: "the ITa row written in T01c
    quoted a message the program never printed").
  - Commit 0d12114: "Records corrected where the main agent wrote results
    it had not read".
- **Impact.** A real defect was recorded as OK and would have passed the
  gate. It was found only by the harness owner's re-run.
- **Afterwards.** BUG-005 recorded and fixed in T01c2. Hook change
  hermes-agent-setup 550582b: after each delegation the parent must run
  `sdlc.bb review` before writing to `test-results/` or `docs/defects.md`.
  3fe5a86: records are copied from the evidence after reading it, causes
  are not guessed.

### M-03 Recorded a defect with an invented cause (T01c, BUG-004)

- **Symptom.** The main agent recorded BUG-004 for ITa-T01-005:
  `qlora_inspect` "reports `type\tbf16\t1`", caused by "`qwen2.block_count`
  (a scalar `uint32 1`) misread as the count for type bf16". The model's
  architecture is `qwen35`, there is no `qwen2.block_count`, and the
  output has no bf16 line. The NG came from an empty right-hand side of the
  comparison.
- **Task and date.** T01c, 2026-11-20/21 (session `a32f7c47`).
- **Evidence.**
  - `work/tasks/T01c-20261120-230327.txt` line 8 (the main agent's
    `docs/defects.md` edit), line 26 (the ITa row: "discrepancy is
    exclusively the newline produced by `qlora_inspect`"), lines 48–50
    (report).
  - `docs/defects.md` line 10, Status: "取り下げ (2026-11-21, harness
    owner): 不具合ではない … 記載の原因も事実と違う".
  - `test-results/ita.md` line 43 (re-run OK, BUG-004 withdrawn).
- **Impact.** A false defect with a plausible-looking but invented cause;
  a fix for it would have changed correct code.
- **Afterwards.** Withdrawn. AGENTS.md (commit b7eb75b): "Never record a
  cause you have not shown … record what the output shows and
  '切り分け待ち'". Main agent changed (see below).

### M-04 Ran the ITa case on a different input and broke the results table (T01c2)

- **Symptom.**
  1. The task asked to re-run ITa-T01-008 on the model cut short by one
     byte. The main agent ran it on the 79-byte UT fixture
     `work/ut/gguf/UT_GgufFile_049.gguf` and recorded that as the ITa
     result.
  2. Its patch to `test-results/ita.md` turned the table's separator and
     rows into `||…` lines and put the new row in the middle of the T06
     rows; a later patch failed.
  3. The run hit the turn limit (40/40), yet the report ends
     "Deliverables complete: task goal met, BUG-005 fixed and verified",
     followed by Hermes' warning that an edit had failed.
- **Task and date.** T01c2, 2026-11-21 00:37–01:04 (session `54ffb87d`).
- **Evidence.**
  - `work/tasks/T01c2-20261121-003755.txt` lines 1–5 (`run_008.sh` copies
    `UT_GgufFile_049.gguf`), lines 14–20 (`||` rows), lines 66–70
    (reported as the ITa result), lines 94–97 (claim and the failed-edit
    warning).
  - `agent.log.1` line 18590: `max_iterations_reached(40/40)`.
  - `test-results/ita.md` line 50: "The T01c2 run itself tested a UT file,
    not the model".
  - hermes-agent-setup 87d0ceb: "T01c で見出しが消え、行が壊れた" (the
    commit names T01c; the `||` edit is in the T01c2 transcript).
- **Impact.** The ITa result for the fix was not a test of the product;
  the harness owner re-ran it on the model and repaired the table.
- **Afterwards.** hermes-agent-setup 87d0ceb: the guard refuses
  `write_file` on existing `test-results/*.md` / `docs/defects.md` and any
  write whose lines start with `||`.

### M-05 Reported the ITa run as passing with seven NG rows (T06c3)

- **Symptom.** The report says "ITa gate: ✅ 判定: OK (11/11 cases
  executed)" and "Gate scripts (`ut`/`ita`) pass 100%", while the same
  report lists 7 NG cases. It calls BUG-003 "Bare compiler exec fails
  (environment). No SDLC impact", but BUG-003 was a code defect in
  `RunLimited`, fixed in T06c4.
- **Task and date.** T06c3, 2026-11-18 17:13–18:30 (session `cc063f41`).
- **Evidence.**
  - `work/tasks/T06c3-20261118-171343.txt` lines 50, 66 and 80.
  - `test-results/ita.md` lines 13–23 (7 NG rows with BUG-003).
  - `docs/defects.md` line 9 (BUG-003: cause in `src/DialectVerify.pas`,
    fixed in T06c4).
  - HANDOFF, "Local session 10, continued", "Check, do not trust": "gate
    ita prints OK when NG rows carry a defect ID, and Qwen reported 'pass'
    with seven NG".
- **Impact.** Taken at its word, T06 would have closed with a defect open
  and its cause filed as environmental.
- **Afterwards.** T06c4 fixed BUG-003. HANDOFF lesson: count NG rows
  instead of reading the verdict; the harness owner runs the oracle at the
  end of every task. Later, gate ut requires a recorded OK to have passed
  in the last run (HANDOFF, "Harness changes").

### M-06 Bundled several delegations into one call (T06c, T02a)

- **Symptom.** The task files give one delegation per step. Mistral sent
  two steps as a two-item `tasks` array (refused: `max_concurrent_children`
  is 1). In T06c it then sent two `delegate_task` calls in one message;
  Hermes ran one, dropped the other, and the next request to Bedrock was
  rejected because the dropped call had no tool result. The run ended
  with no report after the sub-agent had worked 8 592 s (250 tool calls,
  the limit).
- **Task and date.** T06c, 2026-11-18 13:33–15:57 (session `6e145022`);
  T02a, 2026-11-21 02:07 (session `d510998e`).
- **Evidence.**
  - `agent.log.1` line 10825 ("Too many tasks: 2 provided"), line 10836
    ("Truncated 1 excess delegate_task call(s) to enforce
    max_concurrent_children=1"), line 11864 (delegation completed after
    8591.52 s), lines 11865–11866 (`ValidationException`).
  - Request dump of session `6e145022`: message 1 holds one
    `delegate_task` with 2 tasks; message 3 holds two `delegate_task`
    calls (`tooluse_xQyB…`, `tooluse_Rle4…`); message 4 holds a result
    only for the first; the sub-agent's summary says it ran out of steps
    at 250 tool calls.
  - `work/tasks/T06c-20261118-133343.txt` line 3: "Expected toolResult
    blocks at messages.4.content for the following Ids:
    tooluse_Rle4gVIx6clj3g3UWBEFcH".
  - T02a: `agent.log.1` lines 18741 and 18743 (see M-01).
- **Impact.** T06c lost about 2.4 hours and had to be redone as T06c2–c4.
  In T02a the refused call left a pending review that blocked the next
  delegation, and the report then claimed it had run (M-01).
- **Afterwards.** hermes-agent-setup 87d0ceb: the `delegate` hook refuses
  a call with two or more tasks before Hermes sees it. Task files state
  "This task uses **1 delegation**" (`tasks/T02a2.md`, `tasks/T02c.md`).

### M-07 Printed a tool call as text instead of making it (e2e test)

- **Symptom.** In the harness end-to-end test the reply contained
  `Egyesültdelegate_task{"tasks": [{"goal": "外部設計書 …` as text (a
  stray Hungarian word followed by the call in JSON). No delegation
  happened and the turn ended normally after 3 API calls, so the run
  looked finished.
- **Task and date.** `hermes-agent-setup` e2e test, project
  `tmp/sdlc-e2e-20261121-183724`, 2026-11-21 18:37 (session `b66faf83`).
- **Evidence.**
  - `state.db`: assistant message of session `local-b66faf83…` beginning
    `Egyesültdelegate_task{"tasks": [{"goal": "外部設計書 …`.
  - `agent.log.1` line 22183: `Turn ended: reason=text_response(finish_reason=stop) … api_calls=3/30`.
  - Commit b7eb75b (2026-11-21 19:12): "exit 5 when a tool call was
    printed as text and the run stopped early (seen with GPT-OSS and
    Mistral on Bedrock)"; `tasks/run-task.sh` lines 7–9 ("GPT-OSS-120B and
    Mistral leaked tool calls as text").
  - In the T06/T01/T02 transcripts the pattern that `run-task.sh` checks
    for matches nothing (`grep -l` on `work/tasks/*.txt`: no file).
- **Impact.** A run can end early and look complete; the transcript has
  to be checked for it.
- **Afterwards.** `run-task.sh` exits 5 with a warning when the transcript
  contains a tool call printed as text (b7eb75b). The default main agent
  became GPT-OSS-120B for T03 (b7eb75b), which showed the same fault
  (HANDOFF model table), then GLM 5 (1cece1a, 2026-11-22 02:59).

### M-08 Shortened the sub-agent's instructions and dropped rules (T06)

- **Symptom.** When delegating, Mistral wrote its own short goal instead
  of passing on the rules of the task file.
- **Task and date.** T06 (2026-11-17/18). The record does not say which
  step.
- **Evidence.**
  - HANDOFF, "Local session 10, continued", "Delegation harness": "Mistral
    shortens Qwen's instructions and drops rules."
  - hermes-agent-setup 237865f (2026-11-18 22:02): "メインエージェントが
    課題の規則を要約して落としても、サブエージェントが原文を読むようにする".
  - `tasks/T06c.md` lines 3–6: "Begin the sub-agent's `goal` with this
    sentence, word for word … Do not summarise the rules of that file".
  - Not found: a delegation goal in the logs that shows a specific
    dropped rule. Only the records above, written at the time, say it.
- **Impact.** The sub-agent worked without some of the task's rules.
- **Afterwards.** The `delegate` hook (pre_tool_call on `delegate_task`)
  puts "First read <task file>" at the head of every goal; the task file
  comes from `~/.hermes/sdlc/current-task`, written by `run-task.sh`.

### M-09 First T06 documents did not match the task (T06, first attempt)

- **Symptom.** The first T06 run wrote an external design, system flow and
  ITa cases with a JSON outcome, "partial" matches, dialect detection by
  `{$LIBRARYNAME _}` and VCL units, a real-Delphi path through
  `QLORA_DELPHI_PATH`, and `// expect-error:` comments. DESIGN writes the
  marker as `{ expect-error: ... }` (DESIGN.md line 3418).
- **Task and date.** T06, 2026-11-17 19:30–23:11 (session `8d94f87e`,
  stopped by the user after 15 API calls).
- **Evidence.**
  - `work/tasks/T06-20261117-193044.txt` lines 23, 41–60, 100–103 (the
    edits).
  - `agent.log.1` line 5224: `reason=interrupted_by_user`.
  - `tasks/T06a.md` lines 6–9: "A previous attempt wrote these documents,
    but they differ from those facts (JSON outcome, partial match, dialect
    detection, real Delphi, compiler flags, `//` expect-error comments)".
- **Impact.** The facts file `tasks/T06-facts.md` was written (its file
  time is 23:13, two minutes after the run stopped) and T06a corrected the
  documents.
- **Afterwards.** Every later task starts from a facts file written by the
  harness owner and tells the main agent not to read DESIGN.md (for
  example `tasks/T01a.md` lines 7–10).

## Inferred (not shown directly)

- **M-09, the source of the differences.** `grep` finds no JSON outcome,
  "partial" match or `QLORA_DELPHI_PATH` in DESIGN.md, so these were
  probably the model's own additions rather than copied from the design.
  Not checked against the whole of DESIGN 8.16.
- **M-04, the attribution of the remaining NG.** The T01c2 report says
  the one NG left in gate ita is "BUG-001, unrelated". By then every T06
  case's last row was OK (`test-results/ita.md` lines 24–30) and the open
  NG was ITa-T01-005 with BUG-004 (line 35). The attribution was
  therefore probably wrong.
- **M-05, who wrote "pass".** HANDOFF attributes the "pass with seven NG"
  to Qwen; the T06c3 transcript is the main agent's final report. Either
  Mistral took over the sub-agent's verdict without checking it, or
  wrote it itself; the record does not tell which.
- **M-06, Hermes' share.** The rejected request comes from Hermes
  dropping the second call ("Truncated 1 excess delegate_task call(s)")
  while keeping it in the history. Mistral triggered it; another model
  that sends parallel delegations would probably trigger it too.
- **Why the false reports happened** (M-01 to M-05): not known.

## Not counted as Mistral issues

Failures in the same runs that do not depend on the main model:

- expired `aws login` sessions (T06 2026-11-17 19:20, T06b1 2026-11-18
  06:23, T01b1 2026-11-20 18:50) and Bedrock set-up errors on 2026-11-16;
- runs ended by a Hermes gateway shutdown (T06b1 2026-11-18 04:50 and
  06:04, T01a 2026-11-19 01:36);
- the local Qwen's long delegations (T02b1: 3.5 hours, stopped by the
  user; T06b: no code written) and its context compressions;
- the stale task file of the delegate hook in T06c3 (`delegations.log`
  names `T06c2.md` for session `cc063f41`).

## Changes made afterwards

| Date | Change | Where |
|---|---|---|
| 2026-11-17 23:13 | facts file before the documents; later tasks do not read DESIGN.md | `tasks/T06-facts.md`, `tasks/T06a.md` |
| 2026-11-18 22:02 | delegate hook: "First read <task file>" at the head of every goal | hermes-agent-setup 237865f |
| 2026-11-18 22:45 | AGENTS.md: NG counts in reports, ITa compared with the oracle | 529e65b … d74ab8c |
| 2026-11-21 01:09 | `sdlc.bb review` forced after each delegation; at most 4 delegations; records copied from evidence | hermes-agent-setup 550582b, 3fe5a86; commit 0d12114 |
| 2026-11-21 18:22 | guard: no rewriting of record tables, no `||` rows, no bundled delegations | hermes-agent-setup 87d0ceb |
| 2026-11-21 19:12 | main agent GPT-OSS-120B by default; `run-task.sh` exit 5 on a tool call printed as text; AGENTS: no unproven causes | b7eb75b |
| 2026-11-22 02:59 | main agent GLM 5 (`zai.glm-5`) by default ("from T03 on it reported honestly") | 1cece1a |

DESIGN 8.18 and the role table in DESIGN 8.9 name GLM 5 as the main agent
since 2026-11-24 (before, they named Mistral). Mistral Large 3 stays the generator of the
training samples (same table) and the last R4 fallback author (DESIGN
8.19, R4 row); this list is only about its role as the main agent.
