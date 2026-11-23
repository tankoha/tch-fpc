# Handoff

Japanese version: [HANDOFF_jp.md](HANDOFF_jp.md)

Read this first when resuming. Decisions and rationale: [DESIGN.md](DESIGN.md).
Status at a glance of the knowledge-package and fine-tuning work:
[progress.md](progress.md).

## Local session 14, continued (2026-11-23/24): T12 and T16 done, T15 prepared — resume here

**T12 (`Lora`) done** (T12a, T12b1–T12b4, T12c through
`tasks/run-task.sh`, about 4 h). Each step checked: rebuilt with `-Cr`;
the implementation sections of `src/Lora.pas`, `PackedLinear.pas`,
`QwenBlocks.pas` and `QwenModel.pas` equal the prototype in
`work/proto/T12/` (no difference but white space); the new UT rows run
against mutants (Delta or MergedDelta without the scale, A's bound
doubled, B not loaded, duplicate check removed, no LoRA in `Forward`,
`Target('attn_v')` returning K, last trunk block skipped): each caught.
UT gate 611/611, line coverage 100 % (2778/2778). ITa, rerun by the
harness owner: ITa-T12-001 0, 002 0, 003 4.8e-5, 004 0, 005 0 (limits 0,
0, 1e-3, 1e-4, 0), equal to the prototype's; ITa gate 94/94.

What happened on the way:
- T12a stayed in scope: the "no source code ... stop" sentence in each
  delegation's goal (after T11a) worked.
- T12b1's first start found the `aws login` session expired (the script
  refused, exit 4; nothing changed); rerun after `aws login`.
- T12b3 wrote result notes in `test-results/ut.md` that did not describe
  its rows (the main agent reported it); the tests and verdicts were
  right; the 15 notes were replaced with the case titles by hand.
- T12c's first start was refused (exit 3): the gateway auto-started by
  T12b4 had `QLORA_MODEL` empty. `hermes gateway stop`, then rerun. Before
  every run with `QLORA_MODEL`, stop the gateway first.

**T16 (`Checkpoint`) prepared** (next on the path to T17; DESIGN 8.14
"As written for T16"). Prototype in `samples/qwen-lora/work/proto/T16/`
(git-ignored; reference tests in `ref/`, mutants in `mut2/`). Decided on
the way: the moments are stored by variable name (`<variable>.adam_m`,
`.adam_v`) through a CPU store, because T14's moment stores are indexed by
position and a per-store file resumed with another device split would load
other matrices' moments, silently where shapes agree; `state.json` strings
are printable ASCII (fpjson writes UTF-8 encoded twice); a non-finite
validation loss is `null`. Measured: fpjson round-trips Double exactly (an
apparent mismatch with `0.1` was the Extended constant). Committed:
- `src/Checkpoint.pas` skeleton (interface final);
  `tests/ut/Checkpoint_test.pas` skeleton with helpers (a small set and its
  AdamW, no model);
- `tasks/T16-facts.md` (32 UT rows: 001–012 run state and `state.json`,
  101–120 `TCheckpoints`, each run against the prototype; 100 % line
  coverage; seven mutants caught);
- ITa `tests/ita/T16-run.sh`, driver `tests/ita/T16/ckpt_check.pas`, four
  processes (`full`, `part1`, `part2` resumed with another seed,
  `compare`): 001 every A, B and moment, one go vs resumed: 0; 002
  `state.json` fields: 0; 003 retention (last 3 and the best): 0; 004 the
  loss of every update: 0. About 2 min. Skeleton: all NG; three mutants
  NG where expected.
- tasks `T16a` (documents, 2 delegations, the "no source code ... stop"
  sentence in each goal), `T16b1` (run state), `T16b2` (checkpoints),
  `T16c`.
The gate with the skeleton: 611/611 earlier cases OK; only the skeleton's
lines are open.

**T16 done** (T16a, T16b1, T16b2, T16c through `tasks/run-task.sh`, about
3 h, no rework). Each step checked: rebuilt with `-Cr`; the
implementation of `src/Checkpoint.pas` equals the prototype; the agents'
UT run against six mutants catches each on the same rows as the
prototype's reference tests; the result notes match the case titles. UT
gate 643/643, line coverage 100 % (3024/3024). ITa, rerun by the harness
owner: 0 / 0 / 0 / 0, equal to the prototype's (10 updates in one go and
5 + checkpoint + resume in a new process + 5 bit-identical over
14 745 600 values; the loss 245.9 to 92.7); ITa gate 98/98. The gateway
was stopped before every run; no refusal this time.

**T15 (`TrainData`) prepared** (the last unit T17 needs; DESIGN 8.14 "As
written for T15"). Prototype in `samples/qwen-lora/work/proto/T15/`
(git-ignored; reference tests in `ref/`, mutants in `mut/`). Decided and
measured on the way: records are JSONL with `id`, `dialect`, `prompt`,
`answer`, `class` (T19 must write the class into every record and write
raw UTF-8: fpjson turns a `\u` escape of a non-ASCII character into `?`,
so `TrainData` refuses such escapes); **no token cache** (`EncodeTraining`
of a 2000-token sample takes 4 ms); a seeded SplitMix64 order, pools
sorted by length, greedy batches within the budget, resumable by
skipping batches. Committed:
- `src/TrainData.pas` skeleton (interface final);
  `tests/ut/TrainData_test.pas` skeleton with helpers (T05's tiny
  tokenizer);
- `tasks/T15-facts.md` (24 UT rows: 001–014 samples and the data file,
  101–110 order, batches, tensors, each run against the prototype; 100 %
  line coverage; nine mutants caught);
- ITa `tests/ita/T15-run.sh`, driver `tests/ita/T15/data_check.pas`, input
  `tests/ita/T15/records.jsonl` (42 records cut from this sample's sources
  by `make-records.bb`, committed): 001 tokens and loss spans against
  text rendered by the driver: 0; 002 counts: 0; 003 three epochs within
  the budget, every sample once: 0; 004 two processes give the same
  batches: 0; 005 masked targets of the batch tensors: 0. 22 s. Skeleton:
  all NG; three mutants NG where expected.
- tasks `T15a`, `T15b1` (samples), `T15b2` (batches), `T15c`.
The gate with the skeleton: 643/643 earlier cases OK.

**Open question for T17 (found while preparing T15):** `QwenModel` (T11)
runs one sequence `[T, H]`. DESIGN 8.14 batches short samples so that one
forward pays the dequantisation (about 1 s on the 27B) for many; with
the current model a batch costs B forwards. Either T17 runs the samples
of a batch one by one (simple; slower on the 27B by roughly the batch
size in dequantisation, 3.4 samples per batch on the ITa's records) or
the blocks gain a batch dimension (DeltaRule, attention, RoPE; a change
to T08–T11 with their tests). To be decided before T17 is prepared.

**Next:** `tasks/run-task.sh T15a`, then `T15b1`, `T15b2` (check each:
rebuild with `-Cr`, compare with `work/proto/T15/`, run the gate; stop the
gateway before every run), then `QLORA_MODEL=<rehearsal GGUF>
tasks/run-task.sh T15c`. Then the question above, T18 (`GgufMerge`,
H47), T19 (`qlora_dataset`), T20 (`qlora_eval`); T17 (`qlora_train`)
last. And the M4 question of local session 13.

## Local session 14 (2026-11-23): T11 done, H43 holds with the agents' code, M1 complete; T12 prepared

**T11 done** (T11a, T11b1–T11b3, T11c through `tasks/run-task.sh`, about
4 h including the stop below). Each step checked: rebuilt with `-Cr`; the
implementation of `src/QwenModel.pas` and `src/LogprobsCli.pas` equals the
prototype in `work/proto/T11/`; UT 26 + 17 rows OK; gate 559/559, line
coverage 100 % (2600/2600). ITa rerun by the harness owner: ITa-T11-001 0,
002 1.3e-3, 003 1.4e-3, 004 0 (limits 0, 0.02, 0.02, 0); `T11-run.sh gpu`
(ITa-T11-101, `--split 12` over cuda:0 / cuda:1 against the CPU) 1.1e-4
(limit 1e-3). **H43 holds on the rehearsal model with the agents' code.**

**T11a overran its scope.** In the second delegation (the UT case tables,
no code) the local Qwen sub-agent went on to implement both units and their
tests (the work of T11b1–T11b3) and spent about 1 h on a hang in its own
`LogprobsCli_test`, with probe programs in `work/`. Stopped after 2.5 h: the
code, the tests and `test-results/ut.md` were reverted, the documents and
case tables kept (checked against the facts), then T11b1–T11b3 ran as
planned; the hang did not recur (the agents' LogprobsCli UT runs in 5 s).
Also one rejection by the output contract validator in the first
delegation, and a resident gateway left by the stopped run had to be
stopped before T11c. Suggestion for later task files: a delegation with the
skill `sdlc-ut-tdd` that must not write code says so in the delegation text
itself ("no source code; stop after the tables"), not only in the task.

**M2 started: T12 (`Lora`) prepared** (the method of T08–T11; DESIGN 8.13
"As written for T12"). Prototype in `samples/qwen-lora/work/proto/T12/`
(git-ignored; reference tests in `ref/`, the mutants in `mut/`). Committed:
- `src/Lora.pas` skeleton (interface final); interface additions with
  "not implemented" bodies in `PackedLinear` (`Lora`, `SetLora`),
  `QwenBlocks` (`Target`, `LoraTargets`, `Index`) and `QwenModel`
  (`AttachLora`);
- `tests/ut/Lora_test.pas` skeleton and helpers in the three other test
  programs (no T12 test methods yet);
- `tasks/T12-facts.md` (52 UT rows: Lora 001–016 and 101–114,
  PackedLinear 101–107, QwenBlocks 301–308, QwenModel 201–207, each run
  against the prototype; 100 % line coverage; four mutants caught);
- ITa `tests/ita/T12-run.sh`, driver `tests/ita/T12/lora_check.pas`
  (001 pairs and parameters against the tensor directory: 78 pairs,
  4 915 200 parameters at rank 16; 002 B = 0 exact; 003 adapted forward
  against the base forward on a GGUF with W + s·B·A computed in Double:
  4.8e-5, limit 1e-3; 004 block-wise against whole-graph gradients: 0;
  005 save/load: 0). Skeleton: all NG; prototype: all OK; mutants caught.
- tasks `T12a` (documents, 2 delegations; each delegation's goal now
  carries "no source code ... stop" itself, after T11a), `T12b1` (pair),
  `T12b2` (set), `T12b3` (matrix and block), `T12b4` (model), `T12c`.
The gate with the skeletons: 559/559 earlier cases OK; only the new
routines and the empty `Lora_test` are open.

**Next:** `tasks/run-task.sh T12a`, then `T12b1` to `T12b4` (check each:
rebuild with `-Cr`, compare with `work/proto/T12/`, run the gate), then
`QLORA_MODEL=<rehearsal GGUF> tasks/run-task.sh T12c` (stop a resident
gateway first). Then the rest of M2: T15 (`TrainData`), T19
(`qlora_dataset`) and T20 (`qlora_eval`) depend on nothing open; T16
(`Checkpoint`) and T18 (`GgufMerge`, H47) need T12; T17 (`qlora_train`)
comes last. And the M4 question of local session 13.

## Local session 13 (2026-11-23): T11 prepared, H43 measured with the prototype

**H43 measured** (DESIGN 8.11 "H43 in detail"; prototype of `QwenModel`
in `samples/qwen-lora/work/proto/T11/`, git-ignored). Reference: the
bundled `llama-server` started by hand on the CPU (GPUs hidden),
`/completion` with `n_predict` 1, `n_probs` 20, prompts as token ids, no
prompt cache; 24 prompts (Pascal sources, Japanese prose, a chat rendering
with special tokens, each cut to 1, 2, 5, 63, 64, 65, 130 and 600 tokens)
and 3 text prompts; `work/oracle/T11/make-oracle.bb`. The
`top_logprobs` are the log-softmax over the whole vocabulary before any
sampler (`post_sampling_probs` false).

| Reference | max \|Δ logprob\| over the reference's top 10 | top-1 equal |
|---|---|---|
| `llama-server` on the Q4_K_M file | up to **1.8** (0.1–1.8 on every prompt, 0.76 already at T = 1) | 22/24 (misses at margins 0.36, 0.16) |
| on an all-F32 copy of it, KV cache fp16 (default) | 2.4e-2 | 24/24 |
| on the F32 copy, `-ctk f32 -ctv f32 -fa off` | **1.3e-3** (text prompts 1.4e-3) | **24/24** |

The Pascal forward reads the Q4_K_M file in every row (its dequantisation
is exact, T03), so the difference in the first row is `llama-server`'s:
on the CPU it quantises the activations to Q8_K before every K-quant
product, and it keeps the KV cache in fp16. The F32 copy is written by
`work/proto/T11/tof32.pas` (T02's `DequantizeBytes` per tensor, T04's
writer; `build/models/Qwen3.5-0.8B-F32.gguf`, 3.0 GB). The norm weights in
the GGUF already carry the `+1` (means 0.95–4.3), so there is no
`(1 + w)` question. **H43 holds on the rehearsal model** against the F32
reference with DESIGN's threshold (0.02, same top-1); `T11c` confirms it
with the agents' code. Also measured: the trunk as `Recompute` stages is
bit for bit `Trunk`; `--split 12` on cuda:0 / cuda:1 within 1.1e-4 of the
CPU; about 2 s per prompt on the CPU, most of it the head's dequantisation
(248 320 × 1024 Q6_K).

**Question for the user / a design session (M4, not blocking T11):** the
27B criterion of 8.11 (0.15 in bf16) was meant against the running model,
but `llama-server` on a quantised file is itself off by up to 1.8 on the
0.8B. An F32 (about 108 GB) or F16 (about 54 GB) copy of the 27B does not
fit the 30 GB of RAM. Options: compare the 27B in bf16 on the GPUs with
the Pascal forward in fp32 on the CPU (host memory 30 GB: blocks one at a
time), with `llama-server` on the Q4_K_M file only as a coarse check
(top-1 where the margin is large); or a reference from a Q8_0 copy (about
29 GB, still activation-quantised). Not decided here.

**T11 prepared** (the method of T08–T10):
- unit `QwenModel` (`TQwenModel.Load(G, Device0, Device1, Split)`:
  blocks below Split on Device0, the rest and the head on Device1;
  `Embed` dequantises only the rows of the ids; `Stages` for `Recompute`;
  `Trunk`; `Logits` with the head dequantised `HeadChunk` rows at a time
  (8192); `LastLogProbs`; `TrunkBlockCount` leaves out the MTP block
  through `qwen35.nextn_predict_layers`; the head is `output.weight` or,
  without it, `token_embd` (tied, as in the rehearsal model));
- unit `LogprobsCli` and the complete program `src/qlora_logprobs.pas`
  (`[--top N] [--split S] MODEL FILE`; JSONL `{"id", "tokens"}` or
  `{"id", "text"}` in, one JSON line with the top N out);
- `tasks/T11-facts.md` (both units' code from the prototype, 43 UT rows
  all run against it, 100 % line coverage), skeletons `src/QwenModel.pas`,
  `src/LogprobsCli.pas`, test skeletons with the helpers;
- fixtures: `tests/ut/QwenBlocksFixture.pas` gains `WriteTinyQwenModel`
  (T10's tiny model plus `token_embd` (274 × 64, Q8_0), `output_norm`,
  optionally `output.weight`, a bad embedding or head, the MTP key, and
  T05's UT tokenizer); `WriteTinyQwen` writes the same bytes as before
  (sha256 checked). `tests/ut/QwenRefScalar.pas` gains `RefLogits`;
- ITa `tests/ita/T11-run.sh` (driver `tests/ita/T11/logprobs_check.pas`,
  4 cases: top-1, top-10 difference, text prompts, `Recompute`;
  prototype 4/4, skeleton all NG; mutants without `output_norm`, with the
  embedding scaled by √Hidden, without the last block are caught; GPU case
  101 harness owner only);
- tasks `T11a` (documents, 2 delegations), `T11b1` (`QwenModel` stage 1:
  load, embed, stages, 16 rows), `T11b2` (trunk, logits, 10 rows),
  `T11b3` (`LogprobsCli`, 17 rows), `T11c` (ITa).
The gate with the skeletons: 516/516 earlier cases OK; only the two new
units and their empty test classes are open, as for T10 before T10b1.

**Next (done, local session 14 above):** `tasks/run-task.sh T11a`, then `T11b1`, `T11b2`, `T11b3` (check
each: rebuild with `-Cr`, compare the code with `work/proto/T11/`, run the
gate), then `QLORA_MODEL=<rehearsal GGUF> tasks/run-task.sh T11c` (stop a
resident Hermes gateway started without `QLORA_MODEL` first) and
`tests/ita/T11-run.sh gpu`. That ends M1; then M2 (T12, T15–T20) and the
M4 question above.

## Local session 12 (2026-11-22): local models compared, Oxygene compiler verified, T07–T10, T13 and T14 done

**Local models on the H39 tasks** (the ten checked tasks of
`tests/skill/tasks/`, one run per task and mode, GPU logging on).

Through Hermes (`tests/skill/run-h39.sh`, 40 turns;
`build/h39/run2-20261122-0553/`):

| Model | with the skill | without |
|---|---|---|
| magistral small 24B (64k) | 0/10 | 0/10 |
| devstral-small-2 24B (64k) | 2/10 (04, 06) | 0/10 |

Both models fail mostly at tool calling, not at Pascal: magistral returns
its tool calls in a form Ollama does not parse (a probe with one tool
gave text, not a `tool_calls` entry), and its context is capped at 40 000
tokens; devstral often ends a turn without a tool call. Hermes therefore
measures the agent loop, so a second harness without an agent was written.

**Direct harness** `tests/skill/run-direct.bb` (Babashka; usage in its
header): the model is asked for the whole program, the program is built
and run with the task's checker, and the compiler messages or the output
difference go back, up to 6 answers per task. "with" puts the tch-fpc
skill's SKILL.md and reference files into the system prompt; "without"
gives the task only (no access to the sources, unlike Hermes).
`build/h39/direct-20261122-0640/summary.tsv`:

| Model | with the skill | without |
|---|---|---|
| magistral small 24B | 0/10 (1,036 s) | 2/10: 04, 05 (855 s) |
| devstral-small-2 24B | 2/10: 07, 09 (725 s) | 1/10: 05 (692 s) |
| Qwen3.8 27B (128k) | **9/10** (2,858 s; 6 of 9 at the 2nd answer, 2 at the 1st; 05 failed on `Single` passed for a `TTensor`) | 1/10: 05 (7,017 s) |

Reading: without the skill and without the sources every model invents
the API (`tch_tensor_zeros_2d`, `JitLoad`, `T_ReLu`); with the skill only
Qwen turns it into working programs. The 24B models do not reach a
compiling, correct program in 6 answers even with the skill. **Conclusion:
neither 24B model replaces Qwen as the tch-fpc coder or as a sub-agent;
R4's local fallback (devstral) is weak.** The choice of a LoRA base
(magistral is a plain `llama` transformer; this sample's forward is
written for `qwen35`) is the user's and is not decided by these numbers.

**GPU temperatures** (`samples/qwen-lora/work/gpulog.tsv`, 03:16–10:20,
logger stopped): card0 (RX 9070) junction **up to 107 °C** (06:11; 30
samples at 100 °C or more between 06:07 and 10:17), earlier maximum 97 °C;
card1 (RX 9070 XT) up to 91 °C. No reset this time. Power limits and
fans are the user's to change.

**Oxygene compiler (Next 2 of session 11, done).** RemObjects Elements
14.0.0.3157 builds from the command line with EBuild inside the Earth
flatpak (`flatpak run --command=/app/dotnet/dotnet com.remobjects.earth
/app/earth/EBuild.dll X.elements`, then the same `dotnet Bin/X.dll`; a .NET
Core (Echoes) `.elements` project, no `Echoes` reference). The
`pascal-oxygene` skill's six examples build and print their `*.expected`
output (6/6); `tests/check-examples.sh` now builds and runs them (SKIP when
`flatpak info com.remobjects.earth` fails; work files under
`${XDG_CACHE_HOME:-~/.cache}`). Three traps found by the compiler and
written into SKILL.md (version 1.1.0): `=` on a generic `T` is not
allowed (use `Equals` / `EqualityComparer`); a `{ }` comment ends at the
first `}` (use `(* *)` around code with braces); `loop` is reserved
(rename or `&Loop`). Examples 02, 04 and 06 were fixed. Changes are in
`~/.claude/skills/polyglot-scripting` (repository
tankoha/polyglot-scripting), commit c45962e.
**DESIGN 8.16 / M6, question 2:** the compiler arrived before the data
set is frozen (M3 not started), so by the decision in 8.16 Oxygene is
compiled and run like the other dialects and is in the first run; the
filter row for Oxygene (EBuild build and run) is added at M3. Question 2
is closed unless the user wants otherwise.

**Model wording fixed (Next 4 of session 11, done; hermes-agent-setup 4cd71d0):**
hermes-agent-setup `sdlc/README.md` and `sdlc/project-template/AGENTS.md`
name GLM 5 as the sub-agent; `samples/qwen-lora/AGENTS.md` (and `_jp`)
state the models per task range (main: Mistral Large 3 until T02, GLM 5
from T03; sub: local Qwen3.8 until T02, others in between, GLM 5 from
T05b3).

**T08 prepared by the harness owner** (the method of session 11):
- prototype of the gated delta rule (per head `M := exp(g)·M`,
  `d := β(v − M k)`, `M := M + d kᵀ`, `o := M q`), equal to a hand case
  and to a Double scalar reference within 4.8e-7 at the 27B sizes
  (H = 48, 128 × 128, T = 200; about 50 ms);
- `tasks/T08-facts.md` (interface, the five implementation lines, checks
  and messages in order, 33 UT rows all run against the prototype),
  skeleton `src/DeltaRule.pas`, `tests/ut/DeltaRule_test.pas`, fixture
  `tests/ut/DeltaRuleFixture.pas` (random inputs, scalar reference);
- ITa `tests/ita/T08-run.sh` with driver `tests/ita/T08/deltarule_check.pas`,
  7 cases: prototype 7/7 OK, skeleton all NG, a mutant without the decay
  NG on 5;
- tasks `T08a` (documents), `T08b1` (`DeltaStep`), `T08b2`
  (`DeltaRuleTokens`), `T08c` (ITa).

**Next:**
1. ~~T08a → T08b1 → T08b2 → T08c~~ **done, below**; with `tasks/run-task.sh` (main and sub
   GLM 5), the harness owner checking after each (progress below this
   list when done).
2. M1 remainder: ~~T13, T14, T07, T09~~ (done, below), then T10, T11 (H43).
3. ~~Commit~~ **committed 2026-11-22** (tch-fpc, hermes-agent-setup 4cd71d0,
   polyglot-scripting c45962e).

**T08 done (2026-11-22 10:22–10:50, four runs of `run-task.sh`, main and
sub GLM 5).** Harness owner's checks after each stage:
- T08a (2 delegations): the case table has the 33 rows of the facts,
  inputs and expected values identical; detailed design with the five
  lines and the messages in order; ITa case file as specified.
- T08b1: rebuilt and rerun, 16/16; `DeltaStep` is the facts' code.
- T08b2: 40 tests, gate UT 377/377, line coverage 100 %. The sub-agent
  added 7 rows (017–020, 118–120: kind errors of Q, K, V, Beta) that the
  facts missed and coverage needs; they are right and recorded in the
  case table. **Harness owner's fix:** `DeltaRuleTokens` read
  `V.Shape[2]` before checking V's rank (out of range for a 2-D V, no row
  hits it); the line was moved after the V check, rerun with `-Cr`: 40/40,
  gate OK.
- T08c: ITa 7/7 OK, values equal to the prototype's (0, 0, 0, 4.77e-7,
  2.38e-7, 4.17e-7, 2.38e-7); rerun by the harness owner gives the same;
  `ita.md` rows equal `summary.tsv`; no defect.
- Lesson for the facts: list one error row per raise (each kind check),
  and write the UT build command with `-Fu../../src -Fl<libtch>
  -k-rpath=<libtch>` (Hermes' shell had the library path; a plain shell
  does not).

**T13 done (2026-11-22 12:18–12:41, four runs of `run-task.sh`, main and
sub GLM 5).** Unit `Recompute` (DESIGN 8.14): `TStage`, `TSavedInputs`,
`RecomputeForward`, `RecomputeBackward`. One change to the design:
`RecomputeBackward` is a function returning dL/dX0 (so the input gradient
is checked too). Prepared by the harness owner as for T08 (prototype,
`tasks/T13-facts.md` with 21 UT rows all run against it, skeleton, fixture
`tests/ut/RecomputeFixture.pas` with hand-computable scale stages, affine
stages and the oracle `WholeGraphBackward`, ITa `tests/ita/T13-run.sh`).
Checks after each stage:
- T13a: the case table equals the facts' 21 rows; documents as specified.
- T13b1: rebuilt with `-Cr`, 13/13; code as in the facts.
- T13b2: 21/21, gate UT 398/398, line coverage 100 %; code as in the
  facts; no harness-owner fix needed this time.
- T13c: ITa 5/5 OK, every value 0 (bit-for-bit equal to the whole-graph
  backward in fp64, 8 and 64 stages); rerun by the harness owner the same;
  `ita.md` equals `summary.tsv`. Mutants (gradient not chained, forward
  without `NoGradGuard`, stages in the wrong order) are caught.
- Two GPUs (`tests/ita/T13-run.sh gpu`, harness owner only; agents run
  with the GPUs hidden): stages split over cuda:0 and cuda:1, fp32, against
  fp64 on the CPU: 2.8e-6 (8 stages, inputs on the host) and 3.8e-5 (64
  stages, width 512); saved inputs on the right device. 3/3 OK.

**T14 done (2026-11-22 13:00–14:30, five runs of `run-task.sh`; main GLM
5, sub-agent the local `qwen3.8-27b-128k:latest` again, by the user's
choice).** Unit `LoraOptim` (DESIGN 8.14): `TLoraAdamW` (moments `m.<j>`,
`v.<j>` in one `TVarStore` per parameter store, `Step(LR)`,
`ClipGlobalNorm`, `ZeroGrad`, `Moments(I)`, `StepCount`),
`DefaultAdamWConfig`, `ScheduleLR`. Prepared as for T08/T13 (prototype,
`tasks/T14-facts.md` with 28 UT rows all run against it, skeleton, fixture
`tests/ut/LoraOptimFixture.pas` with a toy problem and the oracle
`ReferenceRun` = `TOptimizer.AdamW`, ITa `tests/ita/T14-run.sh`). For the
local sub-agent the code went in three stages (b1 config and schedule, b2
construction and moment stores, b3 clipping and update). Checks after each
stage:
- T14a: the case table equals the facts' 28 rows.
- T14b1 (17 min), T14b2 (30 min), T14b3 (23 min): rebuilt with `-Cr`,
  10, 16, 28 tests pass; the code equals the prototype; gate UT 426/426,
  line coverage 100 %. **Qwen needed no fix by the harness owner.**
- T14c: ITa 7/7 OK, values equal to the prototype's (1.2e-7, 1.1e-16,
  2.4e-7, 0, 6.0e-8, 0, 0); rerun by the harness owner the same;
  `ita.md` equals `summary.tsv`. Mutants (no bias correction; weight decay
  added to the gradient; a wrong clipping norm) are caught; the last one
  only by ITa-T14-007, which compares the clipped gradients directly
  (Adam is nearly scale-invariant, so a wrong clip barely moves the
  parameters).
- GPU (`tests/ita/T14-run.sh gpu`, harness owner only): fp32 on cuda:0
  against the CPU reference 1.5e-7; w on cuda:0 and b on cuda:1 with
  clipping over both against one CPU store 6.6e-7. 2/2 OK.
- Temperatures while Qwen ran (13:30–14:26, `gpulog.sh` now also logs the
  CPU's Tctl in column `cpuC`): card0 junction up to 106 °C at only
  about 100 W (84 of 1622 samples at 95 °C or more), card1 up to 91 °C,
  CPU Tctl up to 91 °C (mean 79 °C; Ollama runs the 128k model 5 % on the
  CPU). The user changed the surroundings of the PC; no clear drop yet.

**T07 done (2026-11-22 14:45–16:41, main GLM 5, sub-agent the local
Qwen).** Unit `QwenChat`: `RenderChat` (system, user and assistant text
turns; thinking on/off; effort xhigh/medium/low; `reasoning_content`;
`PreserveThinking`; generation prompt), `RenderTraining` (answer byte
span), `EncodeTraining` (token span `[LossFrom, LossTo)` = the last answer
and its end-of-turn token), `TrimChat`, `ReasoningInstruction`.
Measured while preparing it:
- **The rehearsal 0.8B GGUF's chat template is not the 27B's** (7,816
  against 8,952 bytes; only the 27B one has the reasoning-effort
  sentence). The oracle is therefore the bundled `llama-server` on the
  0.8B model with `--chat-template-file` of the template extracted from
  the `qwen3.8:27b` blob (`work/oracle/T07/tpl27.jinja`,
  `make-oracle.bb`): 14 conversations, 5 training samples with
  `/tokenize` ids, 4 rejected conversations.
- `llama-server` treats a final assistant message as a prefill and drops
  its `<|im_end|>\n`; training samples were rendered with one more user
  turn and cut.
- Trimming is ASCII white space only (space, \t, \n, \v, \f, \r);
  U+3000 and U+00A0 stay. (Ollama's own renderer may differ; the captured
  prompt of M3 settles it.)
- fpjson's `GetJSON(..., UseUTF8 = True)` corrupts non-ASCII in FPC 3.2.2
  (already seen in T05): the ITa driver reads with `False`.
- **Tags are never written literally** in facts, tasks, skeletons or
  tests (GLM 5 and Qwen read them as their own special tokens): facts use
  placeholders (`<s>`, `<e>`, `{think}`, ...), Pascal uses constants built
  from pieces. The local Qwen turned the placeholders into the constant
  expressions in the case table by itself; no run broke on a tag.
- Two prototype guards in `EncodeTraining` (token boundaries) could never
  fire and were dropped for 100 % coverage; ITa-T07-003 checks the spans.
Checks after each stage: T07a case table equal to the facts' 34 rows;
T07b1 10, T07b2 27, T07b3 34 tests pass with `-Cr`; the code equals the
prototype (b1 and `IsToolResponse` written slightly differently,
equivalent); gate UT 460/460, coverage 100 % (run by the harness owner).
T07c ITa 4/4 (every value 0), rerun by the harness owner the same; mutants
(narrow trim, think block only with PreserveThinking, wrong span end)
caught. Interruptions: T07b3's first run met an expired `aws login`
session (the user logged in again); T07c's first run stopped at
`run-task.sh`'s gateway check (exit 3: the resident gateway lacked
`QLORA_MODEL`; `hermes gateway stop`, then rerun).

**T09 done (2026-11-22 17:11–18:12, four runs of `run-task.sh`, main
GLM 5, sub-agent the local Qwen).** `DeltaRuleChunked` added to
`DeltaRule` (chunk 64 by default, the sequence padded with zero tokens,
one `linalg_solve_triangular` per chunk; T08's conventions: state
[H, Dv, Dk], q as given, the final state returned). Prepared from the
gate measurement's `ChunkGDR` (`build/gate-2026-11-16/h42.pas`). Against
`DeltaRuleTokens` at the 27B sizes: at most 7.2e-7 for T = 1, 63, 64, 65,
200 and 2048; 4.4e-6 with g down to −5 per token; gradients of all five
inputs 2.1e-6; T08's hand cases exact. T = 2048: 100–116 ms on the CPU
(token form 379 ms), 13–17 ms on the RX 9070. **G must be finite** (g =
−inf, allowed in the token form, gives NaN in the chunked form; real
gates are finite). Checks after each stage: T09a appended rows 201–224 to
`DeltaRule.cases.md` exactly as the facts; T09b1 52 and T09b2 64 tests
pass with `-Cr`; the function equals the prototype (the K check written
as in T08); gate UT 484/484, coverage 100 % (run by the harness owner);
T09c ITa 6/6 equal to the prototype's values, rerun the same; GPU case
101 8.3e-7. Mutants (State not transposed, decay factor dropped from the
output, diagonal kept in L, padding g not zero) are caught; padding β
with ones is an equivalent mutant (k and v are zero there). No fix by the
harness owner. Temperatures 14:45–18:13: card0 junction up to 106 °C,
card1 93 °C, CPU Tctl up to 93 °C (mean 73 °C).

**T10 prepared, T10a done (2026-11-22 evening; resume at T10b1 in a new
session).** Prototype by the harness owner in `samples/qwen-lora/work/proto/T10/`
(git-ignored): `PackedLinear.pas` (packed GGUF bytes on a device,
dequantised per call by T03's `DequantizeT`; `LoadPacked`, `LoadDense`)
and `QwenBlocks.pas` (`ReadQwenDims` from the metadata, `RmsNorm`,
`RopeText`, `TQwenBlock` = `TStage` with the FFN, `TLinearBlock`,
`TFullBlock`, `LoadBlock`; fp32; LoRA comes with T12). Oracle, committed
with the next commit: `tests/ut/QwenRefScalar.pas`, a slow scalar Double
version of both blocks from DESIGN 8.10 (T02's scalar dequantiser, plain
loops), and `tests/ut/QwenBlocksFixture.pas`, which writes a tiny `qwen35`
GGUF (blk.0 linear, blk.1 full; 2 key / 4 value SSM heads, 4 query / 2 KV
heads; Q8_0, F16, F32). Measured: rehearsal blocks 0 (linear) and 3
(full) at T 8 and 70 agree with the scalar version to 4.0e-7 / 5.6e-7
(outputs about 4); the tiny model to 3.4e-6; mutants caught (key tiling
interleaved instead of repeated, convolution taps reversed, RoPE on
adjacent pairs). The rehearsal model has equal key and value head counts,
so only the tiny model exercises the tiling. Next: facts with UT rows
(PackedLinear, helpers, blocks), skeletons, ITa (all 24 rehearsal blocks
against the scalar version), tasks T10a, b1–b4, c. **Since done:**
`tasks/T10-facts.md` (both units' code from the prototype, 31 UT rows all
run against it), skeletons `src/PackedLinear.pas`, `src/QwenBlocks.pas`,
test skeletons, ITa `tests/ita/T10-run.sh` (needs `QLORA_MODEL`; driver
`tests/ita/T10/blocks_check.pas`: every rehearsal block against the scalar
version, the tiny model, four blocks chained, the blocks as `Recompute`
stages; prototype 5/5, skeleton all NG; GPU case 101, 4.0e-6). T10a ran
18:27–18:58 (local Qwen as sub-agent): documents and the case tables
`PackedLinear.cases.md` (15 rows) and `QwenBlocks.cases.md` (16 rows)
equal to the facts. **Next: `tasks/run-task.sh T10b1`, then b2, b3, b4
(check each as before: rebuild with `-Cr`, compare the code with
`work/proto/T10/`, run the gate), then `QLORA_MODEL=<rehearsal GGUF>
tasks/run-task.sh T10c` (stop a resident Hermes gateway started without
`QLORA_MODEL` first), then T11 (H43).**

**T10 done (2026-11-22 20:48–23:04, five runs of `run-task.sh`; main
GLM 5, sub-agent the local Qwen).** Units `PackedLinear` and `QwenBlocks`.
Checks after each stage (rebuilt with `-Cr`, the implementation compared
with `work/proto/T10/`):
- T10b1 (28 min): 15/15; `PackedLinear` equal to the prototype.
- T10b2 (34 min): 7/7; the helpers equal to the prototype.
- T10b3 (22 min): 11/11; `TQwenBlock` and `TFullBlock` equal.
- T10b4 (40 min): QwenBlocks 17, PackedLinear 15 tests; both units equal
  to the prototype (only the separator comments differ); gate UT 516/516,
  line coverage 100 % (2412/2412, run by the harness owner). The sub-agent
  added row UT-QwenBlocks-216 (`KeyFloat`'s missing-key raise, the float
  counterpart of 102; the facts missed it, as in T08) with its own fixture
  `tests/ut/QwenBlocksMetaFixture.pas` (the tiny model's metadata without
  `qwen35.rope.freq_base`); right, recorded in the case table.
- T10c (6 min; the resident gateway stopped first, `QLORA_MODEL` = the
  rehearsal GGUF): ITa 5/5 OK, values equal to the prototype's (8.1e-7,
  5.4e-7, 2.4e-6, 8.9e-7, 0); rerun by the harness owner gives the same
  `summary.tsv`; `ita.md` equals it. GPU case 101 (every block on cuda:0
  against the CPU, T 256): 4.0e-6, OK.
No fix by the harness owner.

**Next (done, local session 13 above):** commit T10; then T11 (H43: the Pascal forward against the
running rehearsal model's log-probabilities), prepared as before (facts,
prototype, skeletons, ITa).

## Local session 11 (2026-11-19 to 22): M1 T01–T05 done, H46 holds

**Done** (Hermes SDLC, each checked against an oracle by the harness owner):

| Task | Unit / program | Oracle result | Commit |
|---|---|---|---|
| T01 | `GgufFile` reader, `qlora_inspect` | UT 166/166, ITa 12/12 | 0d12114 |
| T02 | `GgufQuant` (scalar dequantisers, Q8_0 quantiser) | bit-exact with upstream C on all 320 tensors | 38d9f3b |
| T03 | `GgufQuantT` (the same in libtorch ops) | bit-exact on all tensors, ITa 7/7 | 13fd344 |
| T04 | `GgufWriter` | identity copy sha256-equal; 78 tensors retyped to Q8_0 byte-equal to upstream ggml's own writer | d40d23b |
| T05 | `QwenTokenizer`, `TokenizeCli`, `qlora_tokenize` | **H46 holds**: 2978/2978 records (Pascal of five dialects, English and Japanese paragraphs, special tokens, boundary lines, 2000 random strings; 260 253 tokens) equal `llama-server /tokenize`; decode gives every text back; ITa 8/8 | this session |

**The method that works** (keep it for T08 onward):
- The harness owner measures first and writes `tasks/Txx-facts.md` from a
  prototype that already matches the oracle; the facts list every UT row
  with its expected value.
- The oracle comes from outside the code under test (upstream C, upstream
  ggml's `gguf_*` through Ollama's `libggml-base.so`, `llama-server` on the
  CPU) and lives in `work/oracle/Txx` (not in git). The ITa script and its
  driver are the harness owner's; agents run them and copy `summary.tsv`.
- The harness owner places skeletons (interface, stubs, test programs,
  fixtures). Tasks are small: one delegation per stage, at most four per
  run. After every task the harness owner rebuilds, reruns the tests and
  compares with the prototype or the oracle; reports are never taken as
  fact.

**Models** (`run-task.sh` defaults the main agent to `zai.glm-5`; the
sub-agent is `delegation.model` of `~/.hermes/config.yaml`, now
`zai.glm-5` on Bedrock):

| Model | Role | Observed |
|---|---|---|
| Mistral Large 3 (Bedrock) | main | invented results in T01/T02 |
| GPT-OSS-120B (Bedrock) | main | printed tool calls as text (run-task.sh exits 5) |
| GLM 5 (Bedrock) | main | honest reports from T03 on; once misattributed the harness owner's skeleton to the sub-agent |
| Qwen3 Coder 480B (Bedrock) | sub | fast, left stubs and claimed "done" |
| GPT-OSS-120B (Bedrock) | sub | code mostly right; broke the fpcunit layout twice; **recorded 19 OK rows for tests that did not exist** (T05b2) |
| devstral-small-2 24B (local) | sub | ended every delegation without a tool call |
| GPT-OSS 20B (local, 128k) | sub | wrote documents and case tables; could not write code |
| magistral small 24B (local) | sub | never called `patch`; once reported edits it had not made |
| GLM 5 (Bedrock) | sub | correct code and tests (T05b3b, T05b3c) |

**GLM 5 and Qwen's special tokens.** GLM 5 uses `<think>` and
`<tool_call>` itself. When it writes those strings in its own output,
Bedrock parses them as a tool call and rejects the request
(`ValidationException ... toolUse.name`); it also could not write
`<think>` into a test. Tasks therefore never contain the tags literally,
and tests build them from pieces (`'<' + 'think>x'`). Keep this rule for
T07 (chat template) and T15.

**Harness changes** (hermes-agent-setup 4881659 and this session's
uncommitted changes; project `.sdlc/`):
- guard: git subcommands that change the work tree, index or history are
  blocked for every agent, also as `git -C dir ...` (a main agent had
  restored a committed answer with `git -C <proj> checkout HEAD -- src/...`);
  test `sdlc/tests/guard-git.sh`;
- delegate hook: adds its whole sentence unless the goal already holds all
  of it (a main agent's own "First read <task>" had suppressed the skills
  and "only this step" sentences); test `sdlc/tests/delegate-sentence.sh`;
- gate ut: a case recorded OK must have **passed in the last run** under
  `Test{id_}` (`:run-logs`, `:passed-re`, `:test-name` in
  `.sdlc/config.edn`); the coverage run happens before the logs are read;
- `.sdlc/ut-coverage.sh`: `kcov --skip-solibs` (kcov's library wrapper
  made every libtorch test exit 1, so `GgufQuantT` was not measured) and
  `fpc -B` (a stray `GgufFile.ppu` without line info in `src/` had dropped
  `GgufFile` from the coverage);
- `.sdlc/cobertura2lcov.bb`: lines that are only a comment are left out
  (FPC maps code of a `specialize TDictionary<...>` to line 1).

**Incidents.** 2026-11-22 02:33 the PC reset without a shutdown while the
local devstral ran on both GPUs (power at the caps, 175 W and 215 W);
not reproduced. card0 is the RX 9070 (rev c3, junction up to 97 °C under
load), card1 the RX 9070 XT (rev c0). `samples/qwen-lora/work/gpulog.sh`
logs power and temperatures every 2 s to disk.

**Next:**
1. Compare magistral small 24B and devstral-small-2 24B **standalone**
   (no Hermes) on the 10 checked tasks of `tests/skill/tasks/` (the H39
   harness), with GPU logging. The user considers magistral instead of
   devstral as a LoRA/QLoRA base; note that this sample's forward is
   written for `qwen35`, magistral is a plain `llama` transformer.
2. RemObjects Elements 14.0.0.3157 is licensed; the Earth IDE flatpak is
   installed (`flatpak install --user`, runs with `flatpak run
   com.remobjects.earth`). Find the command-line compiler (EBuild; the
   "Mac & Linux Zip Distro" may suit scripts better), then verify the
   `pascal-oxygene` skill's examples and settle DESIGN M6 / Oxygene
   question 2.
3. M1 remainder: T08, T13, T14, then T07, T09, T10, T11 (H43). Same
   method; sub-agent GLM 5.
4. hermes-agent-setup: README and `config.yaml` template still name the
   local GPT-OSS 20B as the sub-agent; set them to GLM 5.

## Local session 10, continued (2026-11-18): M1 T06 done, lessons

**T06 done** (`qlora_verify`, the dialect verifier M3 needs first): unit
`DialectVerify` and program `src/qlora_verify.pas`, written by the local
Qwen through Hermes SDLC (main agent Mistral Large 3). Gate ut OK 104/104,
line coverage 100 % (516/516); gate ita OK 11/11, no NG. Checked by Claude
outside the agents: with `QLORA_FPC*` unset, `qlora_verify --dialect D` on
each `pascal-dialects` skill directory accepts every file the plugin's
oracle (`tests/check-examples.sh`) reports OK (44; Oxygene is static-only).
ITa found three defects that 100 % line coverage had hidden (BUG-001
expect-error text compared after lowercasing; BUG-002 bare compiler name
not looked up in `PATH`; BUG-003 bare name turned into `<cwd>/fpc`), all
fixed test-first. It took about 20 hours of wall time in seven runs
(`samples/qwen-lora/tasks/T06*.md`).

Lessons, applied to every remaining M1/M2 task:

- **Small tasks.** A whole task in one run made Qwen hit 250 tool calls,
  context compression and Bedrock errors. Write a facts file
  (`tasks/Txx-facts.md`) and steps of at most about 60 turns each (a,
  b1..., c...); split T10/T11 most.
- **Schedule.** Qwen thinks 20–30 minutes per reply; one verifier took
  about 20 hours. The other twelve M1 tasks may take one to several weeks.
  Measure after T05; if M4 is at risk, hand parts to the R4 alternates
  (devstral, GPT-OSS-120B, Mistral Large 3) for the schedule, not only for
  quality.
- **Coverage is not enough.** UT viewpoints now required (AGENTS.md):
  upper and lower case, defaults when a variable is unset, bare names and
  relative paths. ITa runs as a user would: `QLORA_*` unset, started
  outside the project, compared with the oracle.
- **Check, do not trust.** gate ita prints OK when NG rows carry a defect
  ID, and Qwen reported "pass" with seven NG. At the end of each task
  Claude runs the oracle itself (scratchpad) and diffs Qwen's outputs;
  count NG rows instead of reading the verdict.
- **Delegation harness** (`hermes-agent-setup` `sdlc.bb`): Mistral
  shortens Qwen's instructions and drops rules. The `delegate` hook
  (pre_tool_call on `delegate_task`) puts "First read <task file>" at the
  head of each goal; the task file comes from `~/.hermes/sdlc/current-task`
  written by `tasks/run-task.sh` (one run at a time). A resident Hermes
  gateway keeps the environment of the run that started it, so the hook
  does not use environment variables, and `run-task.sh` refuses to start
  (exit 3) when the gateway's GPU hiding or `QLORA_*` differ. Check
  `~/.hermes/sdlc/delegations.log` after each run.
- **Rules in AGENTS.md**, not repeated per task: the literal `[廃止]`
  (the gate counts `[廃止: ...]` as not run), no Python or Node.js (now
  blocked in `terminal` by `:forbidden-commands` in `.sdlc/config.edn`;
  Qwen had used python3 through a script file), the ITa conditions and the
  report rule.

- **Fewer triggers for long thinking.** Hermes logs of T06: about 100
  refused or failed tool calls (execute_code 21, dangerous-command
  approvals about 40, Pascal guard about 30, patch mismatches 14) and 68
  context compressions; the long silences (8-28 min) followed reading
  large files whole. Now: `run-task.sh` passes
  `-t terminal,file,skills,todo,delegation` (no `execute_code`; children
  never gain tools the parent lacks); the delegate hook also asks the
  sub-agent to load `:delegate-skills` of `.sdlc/config.edn` first;
  AGENTS.md "Working with the tools" (scripts in files run with `bash`,
  no `rm -r` but `mktemp -d`, `{$mode}` in probes too, read by line
  range, read before `patch`, do not read hook sources). Task design:
  one test program per stage or group of functions. Not known: whether the
  thinking loops (logs keep only its start), and whether the sampling
  settings (temperature 1, presence_penalty 0, no thinking budget) matter;
  an Ollama experiment needs the user's consent.

Next: M1 T01, T02, T03, T04, T05 (H46), T08, T13, T14, then T07, T09, T10,
T11 (H43), each as facts file plus small steps through `tasks/run-task.sh`.

## Local session 10 (2026-11-17): Theme A implemented, M0 prepared

Plan: C4 → M0 → M1 → M2 → M2g (M3 separately). Nothing committed yet in
this session. Done so far, all on the CPU (a Qwen benchmark held the GPUs):

- **Theme A implemented** (DESIGN 7): `plugin/` with the skill
  `plugin/skills/tch-fpc/` (`SKILL.md` 160 lines, eight `reference/*.md`,
  twelve examples, six pitfalls, six traps, `scripts/build-example.sh`),
  `.claude-plugin/marketplace.json`, `tests/skill/check-examples.sh` (24 OK
  against `build/libtch-rocm` with the GPUs hidden),
  `tests/skill/check-index.sh` (20 index rows OK), both in CI, README
  sections. Expected outputs avoid raw float printing.
- **H38 holds**: every pitfall fails with a stable message, now quoted
  exactly in `SKILL.md` and in each file's `expect-error:`.
- **H37, Hermes side holds**: `<checkout>/plugin/skills` added to
  `skills.external_dirs` in the local Hermes config; `hermes skills list`
  shows `tch-fpc` enabled. Claude Code side: needs the pushed repository
  (`/plugin marketplace add tankoha/tch-fpc`), for the user.
- **H39 prepared**: ten tasks in `tests/skill/tasks/NN_*/` (prompt and
  expected output; written by Claude, never training data),
  `tests/skill/check-task.sh` (checker, self-tested with reference answers
  kept outside the repository), `tests/skill/run-h39.sh <with|without>`
  (Hermes with the local Qwen as the main model, toolsets
  terminal/file/skills; "with" preloads `tch-fpc`). Not run yet: waits
  for the GPUs.
- **M0**: `samples/qwen-lora/` created from the SDLC template, translated
  to English (`AGENTS.md` canonical with the project's Pascal rules,
  `AGENTS_jp.md`); `.sdlc/config.edn` with `:coverage-kind :c0`,
  `.sdlc/ut-coverage.sh` (fpcunit tests built with `-gl`, run under kcov,
  merged) and `.sdlc/cobertura2lcov.bb`. **H50 holds**: on a throwaway
  probe unit `sdlc.bb gate ut . --run` reported the one untested line as
  NG, and OK at 100 % once a case covered it; the probe was removed.
  `samples/qwen-lora/work/` and `coverage/` are git-ignored.
- **llama-server on the rehearsal model, CPU (M0)**: the bundled build
  (`39febd2ad`) serves the 0.8B GGUF with the GPUs hidden; `/tokenize`
  (with `with_pieces`), `/apply-template` and `/completion` with
  `n_probs` answer. `/completion` returns
  `completion_probabilities[].top_logprobs[]` with `id`, `token`, `bytes`,
  `logprob` (natural log); `/apply-template` ends the generation prompt
  with `<think>\n` (thinking on). This settles the endpoint row of DESIGN
  8.10's last table.

- **H39 run** (DESIGN 7.5 "H39 result"): 10/10 with and 10/10 without
  the skill; total time 29 against 55 minutes. Results in the git-ignored
  `build/h39/run1/`. C4: the local Qwen writes small tch-fpc programs; the
  user decides the gate.

Next: the Pascal verify hook of Hermes (`hermes-agent-setup`,
`pascal/agent-hooks/pascal.bb`) compiles a written `.pas` with only the
file's directory as unit path, so every M1 unit that uses `Tch` would be
sent back with "Can't find unit Tch". Give it the tch-fpc paths (for
example an environment variable with extra fpc options) before M1; then M1
through Hermes in the order of DESIGN 8.18.

## Design session 10 (2026-11-17): Theme B designed in detail

Run as a Fable sub-agent from the local session. Result: DESIGN 8.9–8.19
and UML 10. Nothing is implemented, nothing was committed, the GPUs and
Ollama were not touched. The plan was **not** found infeasible; eight
points need the user (below).

**Decided in this session** (rationale in DESIGN):
- Milestones M0–M6 (8.11): everything is written and exercised on a small
  `qwen35` model on the CPU (M1, M2) and smoke-tested on the two GPUs with
  that small model (M2g) before the 27B run (M4), because Qwen is down
  while the 27B trains. H43 and H46 close M1.
- The sample is an SDLC project of its own at `samples/qwen-lora/`,
  all Pascal, 17 units and 8 programs (8.12).
- LoRA: rank 16, alpha 32, on the seven attention-side projections of all
  64 trunk blocks (208 tensors, 39.58 M parameters); one `TVarStore` per
  device; no dropout; `blk.64` untouched (8.13).
- The block-wise backward is a Qwen-independent unit `Recompute`; the
  sample has its own AdamW because `TOptimizer` cannot save its state;
  checkpoints every 50 updates with exact resume (8.14).
- Training text uses the no-thinking form for the first run; the loss
  covers the answer only (8.14).
- D3 writer: metadata copied as bytes, adapted tensors as Q8_0, file
  20.08 GB, about 24.3 GB in Ollama; checks in the order identity copy,
  round trip, fidelity, H47, H45 (8.15).
- **H44 is off the plan**; the intermediate check is the new H47 (the
  bundled `llama-server` reads the merged rehearsal file).
- Data: the specification's contents, the record schema, quotas (1000 per
  dialect), the filter per dialect, dedup and leak check, provenance
  (8.16). **Oxygene records are generated and stored as `static-only` and
  held back from training until a compiler is here**; Delphi via FPC is the
  class `fpc-approx`, trained on and reported separately.
- Evaluation: 200 held-out tasks, two seeds, thinking on and off, before
  and after; verified aggregate of three dialects as the headline; Delphi
  and Oxygene in separate columns; regression and tool-call checks; MTP
  speed (8.17).
- Twenty Hermes tasks T01–T20, each with its oracle (8.18); risks and stop
  criteria R1–R18 (8.19).

**Verified in upstream source** (llama.cpp at `39febd2ad`, the commit the
bundled `llama-server` reports) and in the GGUF itself: the block graph,
the gated delta rule (token-by-token form and the chunk size 64), the L2
norm, the convolution, RoPE (IMROPE; for text it rotates 64 of the 256
head dimensions like plain rotary), the Q/K norms, GQA, the MTP block, the
Q8_0 quantiser, the pre-tokenizer regex, the chat template. Two traps
found: key heads are **tiled** in the linear blocks (`j mod 16`) but
**divided** in the full blocks (`h div 6`); and **Ollama does not run the
GGUF's chat template** (the Modelfile has `RENDERER qwen3.8`), so one
rendered prompt must be captured. What is still from memory is tabled at
the end of DESIGN 8.10.

**Questions for the user** (answered 2026-11-17 except 2):
1. ~~Rehearsal model~~ **0.8B, downloaded** (Unsloth Q4_K_M + mmproj BF16
   in `build/models/`, sha256 verified; Q5_K and Q8_0 tensors, a tied
   output head, no MTP block; DESIGN 8.11).
2. **Open:** Oxygene held back from the first run unless the compiler
   arrives before the data set is frozen (8.16) — or train on the
   unverified class knowingly? (The design keeps assuming "held back".)
3. ~~Reasoning text~~ **none in the first run, as designed** (8.14). The
   user asked about the concerns; they are recorded in 8.14
   "Reasoning text: concerns". Second run: Qwen's own reasoning, kept
   only where the final code passes the filter (decided).
4. ~~Prompt languages~~ **English 90 % / Japanese 10 %** (8.16).
5. ~~Data set~~ **published**; the held-out tasks and their
   Opus/Sonnet-written reference solutions may sit in this public
   repository (never training data). The data set lives in
   `samples/qwen-lora/data/dataset-v1/` (decided; DESIGN 8.16
   "Publication").
6. ~~Coverage~~ **kcov 43, installed; works on FPC binaries, lcov via a Babashka converter (DESIGN 8.9 H50); the gate itself at M0.** FPC has no
   gcov-style instrumentation of its own; kcov reads the DWARF line
   tables of any native binary (build with `-gl`) and is in the Ubuntu
   archive. It measures lines only, so this project's
   `.sdlc/config.edn` uses `:coverage-kind :c0` and an lcov file
   converted from kcov's output (Babashka) if kcov does not write lcov
   directly.
7. ~~Fallback author (R4)~~ **in order: local
   `devstral-small-2:24b-instruct-2512-q4_K_M`, then GPT-OSS-120B (Bedrock),
   then Mistral Large 3 (Bedrock); if all three fail, stop and ask the
   user** (R4).
8. ~~SDLC document language~~ **English canonical, Japanese translations
   with the `_jp` suffix**, as in the rest of the repository.

**Next local steps, in order:**
1. Answer the questions above; download the rehearsal model (1).
2. C4 as before: Theme A, local list below, through H39.
3. M0 (DESIGN 8.11): create `samples/qwen-lora/` from the SDLC project
   template; settle H50; check that the bundled `llama-server` serves the
   rehearsal model on the CPU with the GPUs hidden
   (`HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1`) and that
   `/tokenize`, `/completion` with `n_probs` and `/apply-template` answer
   as DESIGN 8.10's last table assumes; add `samples/qwen-lora/work/` to
   `.gitignore`.
4. Hand T01–T06, T08, T13, T14 to Hermes (DESIGN 8.18), then the rest in
   the stated order. Record H46 and H43 on the rehearsal model here.
5. In parallel (M3): Opus/Sonnet write `data-spec/` and the evaluation
   tasks; Mistral generates; `qlora_verify` and `qlora_dataset` freeze
   data set v1; the user runs the baseline evaluation and captures one
   rendered prompt from Ollama.
6. M2g, M4, M5 by the user, as listed at the end of DESIGN 8.18. Stop
   criteria: DESIGN 8.19.

New hypotheses: H47 (merged file under `llama-server`), H48 (generator
yield), H49 (MTP speed after the merge), H50 (coverage tool). Throwaway
tools of this session: `build/design10/` (git-ignored).

## Local session 9 (2026-11-16): ROCm libtorch restored, GPU gate run

**Where things stand.** Nothing is committed: the working tree has the
design session 9 documents (DESIGN 7 and 8, UML 9, this file, the `_jp`
translations) plus the gate results below. Nothing of Theme A or B is
implemented.

Done in this session:
- ROCm libtorch back as `libtorch-rocm/` (`build-version`
  2.13.0+rocm7.1; zip kept in `~/Downloads`), the eight symlinks of
  DESIGN 4.8, `build/libtch-rocm` built (same four warnings). No CPU
  libtorch on this host: CI downloads its own; the ROCm build also runs
  on the CPU.
- The six CI tests (abi, api, nn, jit, generated, ownership) built against
  `build/libtch-rocm` pass on the CPU with
  `HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1`.
- `rocm-smi` and `libdrm-dev` installed (by the user). The two cards are
  an RX 9070 XT (PCI 03:00.0, HIP device 0, drives both monitors) and an
  RX 9070 (PCI 09:00.0, HIP device 1); the Raphael iGPU (10:00.0) is
  enabled and has a DP and an HDMI output.
- **GPU gate (C2) run: passed for T = 1024 with new conditions.** Numbers
  in DESIGN 8.8 "Gate results"; programs in `build/gate-2026-11-16/`
  (git-ignored). The second T = 2048 run hung the XT's gfx ring under
  labwc and the desktop session restarted.

Next, in order:
1. ~~**User:** move both monitors to the iGPU~~ **done 2026-11-17**:
   `boot_vga` on 10:00.0, both RX 9070 idle at 0.08 GiB. Only the DP
   monitor (`card2-DP-7`) was detected as connected at the time; check
   the HDMI one.
2. ~~Re-run H42 with a re-balanced split~~ **done 2026-11-17** (DESIGN 8.8
   "Gate re-run"): 36/28 gives 167 tokens/s at T = 1024 (13.06 / 13.88
   GiB) and, with saved inputs in host memory, 172 tokens/s at T = 2048
   (14.03 / 14.74 GiB); no ring timeouts. The gate is passed.
3. **User:** ~~C1~~ **D3, decided 2026-11-17**; C3: ~~Mistral output
   terms~~ **Apache-2.0, settled 2026-11-17**; ~~GGUF base~~ **GGUF for
   the first run, original after it**; ~~Oxygene excluded~~ **Oxygene in
   scope** (licence requested from RemObjects); ~~blob read access~~ **ACLs set,
   both blobs readable** (below, DESIGN 8.8 "Decisions"). C3 is done;
   then the next design
   session continues DESIGN 8 (milestones, module structure, LoRA layer,
   data specification, evaluation, hand-off to the Hermes process).
4. In parallel: Theme A, local list below (C4).

## Design session 9 (2026-11-16): knowledge package designed; 27B fine-tuning stopped at the feasibility gate

Run as a Fable sub-agent from a local session. Two themes; decisions and
numbers are in DESIGN 7 and 8. Nothing is implemented and nothing was
committed by the design session.

**Theme A — LLM-facing knowledge package (DESIGN 7): designed, ready to
implement locally.** A plugin under `plugin/` with one skill
`plugin/skills/tch-fpc/` (`SKILL.md`, `reference/`, `examples/`,
`pitfalls/`, `traps/`, `scripts/build-example.sh`), a root
`.claude-plugin/marketplace.json`, verification in `tests/skill/`, one
more CI step. English only; licence as the repository.

Local list for Theme A:
1. `plugin/.claude-plugin/plugin.json`, root `.claude-plugin/marketplace.json`.
2. `scripts/build-example.sh` and `tests/skill/check-examples.sh`; make
   one example pass before writing the rest.
3. The twelve examples, six pitfalls, six traps of DESIGN 7.4, each with
   its expected output or `expect-error:` text. **(H38)**: record the
   compiler message of each pitfall; drop or reword one whose message is
   not stable.
4. `SKILL.md` (at most 160 lines) and `reference/*.md` from DESIGN; the
   doc index; `tests/skill/check-index.sh`.
5. CI: `check-examples.sh` in `libtorch-cpu`, `check-index.sh` in
   `no-libtorch`.
6. **(H37)**: install through the marketplace in Claude Code and through
   `skills.external_dirs` in Hermes; note what had to change.
7. **(H39)**: ten tasks with checkers in `tests/skill/tasks/`; run them
   through Hermes with the local Qwen, with and without the skill; record
   both pass rates here. This is also condition C4 below.
8. README / README_jp: a short install section. Bump the plugin version
   with the release.

**Theme B — fine-tuning Qwen3.8 27B as a tch-fpc sample (DESIGN 8):
verdict GO with conditions; the design stops here on purpose.** No module
structure, LoRA classes, data specification or milestones exist yet.

What the verdict rests on:
- *Computed from the measured tensor list:* packed weights 8.10 GB and
  8.43 GB on the two GPUs; peak about 13.4–14.2 GB of 16 GB at 2048 tokens
  with block-wise recomputation and a chunked loss (activation and runtime
  parts are estimates).
- *Measured (Ollama log):* this model does 740–1000 tokens/s of prompt
  processing on these GPUs. *Estimate:* training at 60–250 tokens/s, so
  about 12–49 h for three epochs of 3.5 M tokens.
- *Measured (grep):* every tensor op the model needs exists in the
  generated API; the gaps are small.
- *Measured (installed Ollama 0.40.13):* new LoRA adapters cannot be
  created through a Modelfile any more; the bundled `llama-server` still
  has `--lora`, and `llama-quantize` is bundled.
- *Not measured:* anything about libtorch on the GPUs (no libtorch is
  installed; the GPUs were not touched). **Since measured** in local
  session 9 (above): DESIGN 8.8 "Gate results".

**Conditions — the user decides or enables each (trade-offs in DESIGN
8.3, 8.4, 8.8):**

- **C1, deployment path. Decided 2026-11-17: D3** (DESIGN 8.8
  "Decisions").
  - D2: a GGUF LoRA adapter served by `llama-server --lora` outside
    Ollama. Least code, exactly the trained configuration; Hermes needs a
    second endpoint and the server has to be started by something. (H44)
  - D3: a merged GGUF with the adapted tensors as Q8_0, imported with
    `ollama create`. Stays in Ollama; about +3.3 GB of VRAM; more code
    (GGUF writer, merge). (H45)
  - D4: merge into the original bf16 weights and requantise with
    upstream tools. 54 GB download and a Python conversion script.
  - The design session leans to D3 with D2 as the first experiment (D2's
    writer is a subset of D3's).
- **C2, GPU gate.** Restore the ROCm libtorch
  (`libtorch-shared-with-deps-2.13.0+rocm7.1.zip` from
  `https://download.pytorch.org/libtorch/rocm7.1/`, 12 GB unpacked as
  `libtorch-rocm/`, the eight symlinks of DESIGN 4.8, then
  `build/libtch-rocm`), unload the Ollama model, and run three throwaway
  measurements: H40 (bf16 and fp16 `matmul` at model size, forward and
  backward), H41 (Q4_K/Q6_K dequantisation rate), H42 (peak VRAM of one
  synthetic block at 1024 and 2048 tokens, two devices from one process).
  About an hour. Abort criteria: no working half-precision product, under
  30 tokens/s projected, or a peak above 15 GB at 1024 tokens.
  **Done since (2026-11-16, same day):** the ROCm libtorch is back
  (`build-version` 2.13.0+rocm7.1, eight symlinks), `build/libtch-rocm`
  builds (the same four warnings), and the six CI tests (abi, api, nn,
  jit, generated, ownership) built against it pass on the CPU with
  `HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1` (no GPU visible, so
  `tests/api`'s CPU-build assertions hold too). `rocm-smi` is installed.
  **New fact for H40/H42:** the two cards differ: GPU 0 is an RX 9070 XT,
  GPU 1 an RX 9070 (both gfx1201, 16 GB). The 32/32 block split of
  DESIGN 8.2 assumes equal speed; measure both cards in H40 and size the
  split by speed as well as memory.
  **Gate run (2026-11-16): passed for T = 1024, with new conditions** —
  details in DESIGN 8.8 "Gate results". H40 and H41 hold (bf16 100 / 81
  TFLOP/s; dequantisation exact, about 1 s per model forward); the
  synthetic 64-block step runs at 166 tokens/s at T = 1024 with peaks of
  12.79 / 14.89 GiB; T = 2048 runs out of memory on the 9070 (head phase).
  The second T = 2048 run hung the desktop's GPU ring (labwc, XT) and the
  session restarted. Before any further GPU run: move the display to the
  iGPU (or train without a session), re-balance the split (e.g. 36/28),
  keep T ≤ 1024 or offload saved inputs to host memory.
- **C3, inputs.** Read access to the model blob under
  `/usr/share/ollama/.ollama/models/blobs/` (16.8 GB; a copy or a
  group/ACL change, needs root; **done 2026-11-17 with read-only ACLs**; the
  text GGUF and the `mmproj` GGUF are both readable); ~~confirm GGUF as the base rather than the 54 GB original~~
  **2026-11-17: GGUF for the first run, the original only after it
  completes**; ~~check Mistral's terms on using its outputs as training
  data~~ **settled 2026-11-17: Mistral Large 3 is Apache-2.0**; ~~decide whether Oxygene is excluded~~ **2026-11-17: Oxygene in
  scope**, a licence is being requested from RemObjects; the next design
  session decides how Oxygene samples are gated until a compiler is here and whether the FPC approximation of Delphi is accepted as a
  separately reported class.
- **C4, readiness.** Theme A implemented and H39 run. If the local Qwen
  cannot write small tch-fpc programs with the skill, it will not write
  this sample.

If C2 fails, the alternatives to discuss: (a) train a smaller model of the
same family locally (the code is the same, the rehearsal becomes the
product); (b) rent one large GPU for the training run only and keep
everything else local; (c) drop fine-tuning and invest in the skills.

Also for the user to know:
- Training and the local LLM exclude each other on the GPUs. While a 27B
  run is on, the Qwen sub-agent is down, for a day or two per run.
- After any adapter the MTP draft head no longer matches; generation is
  expected to get slower (draft acceptance today 0.42–0.89), not wrong.
- The architecture details in DESIGN 8.1 come from a tool-made summary of
  upstream `qwen35.cpp`; the delta-rule kernel, RoPE and the chat template
  were not read. The next design session must read them in the source.

**Next design session — held as design session 10, above.** It was to cover (C1 = D3, C3 answered; C4 may run in parallel):
staged milestones (CPU rehearsal on a small `qwen35` model, then 27B),
module and unit structure of the sample, the LoRA layer in `TchNN` terms,
the block-wise backward as a reusable routine, data pipeline and data
specification, evaluation (per-dialect compile/run pass rate on a
held-out task set, before and after), and the boundary of what Mistral's
external design must cover. Hypotheses H43 (log-probability parity with
the running model) and H46 (tokenizer parity) belong to its first
milestone.

Open hypotheses from this session: H37–H39 (Theme A), H40–H46 (Theme B);
each is defined where it is first marked in DESIGN 7 and 8.

## Release v0.1.0 (2026-10-14)

The repository was made public on 2026-10-13 (secrets, personal paths,
history and license placement checked first; README status updated). The
first release, **v0.1.0**, is a GitHub pre-release on tag `v0.1.0`
(commit b7933d8): https://github.com/tankoha/tch-fpc/releases/tag/v0.1.0.
Next: choose the theme of the next milestone with the user, then a design
session (Fable sub-agent). (Done: design session 9, above.)

## Overall review (2026-10-12, after local session 8)

Three independent Opus reviewers (code vs design, en/jp documents, test
coverage; read-only, no context from the sessions) reviewed the whole
repository. Every finding was checked against the code before acting.
Fixed and committed:
- DESIGN 4.8 body and the section 6 status brought in line with the
  measured "Results" (GpuBackend rule, `at_context_has_cudart` in the L2
  list, which calls raise on a CPU build, GPU test items 1b/2/6/7/8, the
  mock bullet, the fuser flag); 4.6.10's flatten bullet points to 4.8;
  4.6.9's mock counts annotated; a status paragraph for the milestone 3
  layers; the real ROCm archive name; the autocast patch marked compiled.
- HANDOFF: forward pointers on stale older passages (H32 amended, design
  sessions via Fable settled, open questions done, ROCm done), design
  session headings 6–8 point to their local sessions; "(below)" → "(above)".
- UML status table and TchSys box; `CLAUDE.md` workflow; source comments
  in `Tch.pas`, `TchSys.pas`, `tests/jit`, `tests/gpu`.
- Tests: Embedding `PaddingIdx` checked through the gradient (the old check
  could not fail); `GpuBackend`, H33, H34 asserted in the GPU test (variant
  passed by `run.sh`); `DoubleValue` on a GPU scalar must return 6; the
  CPU-build GPU API asserted in `tests/api`; H26's dict case checks a fresh
  `Raw`; mock GPU stubs now fail where real libtorch does, plus a fuser
  flag test in `jit_mock_test`.

**Follow-up (same day): fixed and committed.**
- `TVarStore.AddEntry`: when `<name>__<Count>` is taken too, the next free
  number; all checks before the store changes (test fails on the old code
  with `EListError`). `TBatchNorm` raises unless `Nd` is 1..3, before
  registering.
- Compiler hypotheses asserted: `tests/compiler/h3`, `h10`, `h11` check
  their results and `run.sh` counts a non-zero exit; `tests/ownership`
  asserts the H1 counts; `tests/fpu/run.sh` checks every mode (masked modes
  +Inf twice, nomask-small EZeroDivide, nomask-large and pool-first must
  crash or hang).
- Untested API now tested: tracing a method (incl. a raising one),
  `CreateFromStream(S, Device)`, `TTrainableJitModule` `CreateFromStream`/
  `ForwardIV`/`Method`/`MethodIV`/`SetTrain`/`SetEval`/`Inner`, buffers;
  `NonLinearityGain`, `KaimingNormal`, the `TVarPath` helpers,
  `BackwardStepClipNorm`, transpose and layer-norm configs, `TRNN`
  properties; `SetCudnnUserEnabled`, `TchMaskFpuExceptions`.
- **Bug found by the new tests:** `NonLinearityGain(nlTanh)` was
  1.6666666269 because FPC folds `5.0 / 3.0` in Single; now `Double(5) / 3`
  (DESIGN 4.6.1).
- `tests/common/layers.inc`: the conv padding and RNN hand setups shared by
  `tests/nn` and `tests/gpu`.

**B7, 2026-10-13: handle types decided and implemented.** DESIGN 3 said
`type PTorchTensor = type Pointer;` (distinct pointer types); the code had
plain `Pointer`. *Measured* (FPC 3.2.2): `type Pointer` aliases are not
distinguished (value parameter, `var` parameter and assignment all compile),
but pointers to distinct empty records are ("Incompatible type"; `nil`, an
untyped `Pointer` and casts still compile). `TchSys.pas` now declares
`PTorchTensor`, `PTorchScalar`, `PTorchOptimizer`, `PTorchModule` and
`PTorchIValue` that way; the whole code base including `TchSysGenerated`
(about 4,700 uses) recompiled without error, so no mix-up was hidden, and a
deliberate `at_free(<a PTorchIValue>)` now fails to compile. C side and ABI
unchanged. `{$packrecords c}` is not needed and not added (no struct in
`torch_api.h`, no record crosses the C boundary); add it with the first
function that passes a record. DESIGN 3 updated.

**A2, 2026-10-13: store rollback.** The user chose a rollback over
pre-checks. `TVarStore.RollbackTo(ACount)` removes the variables registered
after the store held `ACount` of them; every constructor that registers
variables (and `TTrainableJitModule.RegisterInto`) calls it on any exception
and re-raises. Test: `TestRollback`. DESIGN 4.6.2 and 4.6.5 updated.

Nothing from the review is open any more.

## Local session 8 (2026-10-12): GPU (4.8) implemented, runs on the RX 9070

**Where things stand.** 4.8 is implemented and committed; the working
tree is clean. Steps 1–10 of design session 8's local list are done. Every
design item handed back at the end of local session 4 is now implemented.

- `csrc/libtch/CMakeLists.txt`: `-DTCH_LIBTORCH_DIR=<dir>` manual-link
  mode (a `CACHE PATH`, not `option()`); default path bit-identical.
  Builds: `build/libtch` (CPU, unchanged), `build/libtch-rocm`
  (`cmake -S csrc/libtch -B build/libtch-rocm
  -DTCH_LIBTORCH_DIR=$PWD/libtorch-rocm -DCMAKE_BUILD_TYPE=Release &&
  cmake --build build/libtch-rocm -j 12`, about a minute),
  `build/libtch-manual` (manual mode on the CPU archive, a check only).
- `TchSys`: the `atc_*`, `at_context_*` (plus `at_context_has_cudart`),
  `atm_fuser_cuda_*` externals. `Tch`: `CudnnIsAvailable`,
  `CudaManualSeed(All)`, `CudaSynchronize`, `CudnnUserEnabled`/`Set…`,
  `SetCudnnBenchmark`, `TGpuBackend`/`GpuBackend`, `GpuRuntimeVersion`,
  `CudnnVersion`, `TDevice.ToString`. `TchJit`:
  `JitFuserCudaEnabled`/`SetJitFuserCudaEnabled`. `TchNN`: the guarded
  cuDNN flatten call in `TRNN`.
- `tests/gpu/gpu_test.pas` + `run.sh [rocm|cuda|cpu]`: all items OK on the
  RX 9070; `cpu` prints `SKIP`. Mock: CPU-build stubs and
  `TestGpuBackend`. CPU jit test: the fuser flag. Regression on CPU (abi,
  api, nn, jit, mock): all OK.
- `libtorch-rocm/lib` needed **eight symlinks** the zip does not carry
  (seven SONAME links and `libnuma.so`); without them the process does not
  start. The command is in DESIGN 4.8 "Results". Redo it after
  re-unpacking.

**Measured, different from the design** (all in DESIGN 4.8 "Results"):
- H34 false: no cuDNN on ROCm (`CudnnIsAvailable` False), so the RNN
  flatten is never called there (H30 moot); RNN numbers reproduced.
- H35: `has_hip` False, `has_cuda` True, `has_cudart` False on ROCm.
  `GpuBackend` uses "HIP, or CUDA without CUDART" for ROCm.
  `GpuRuntimeVersion` raises on ROCm.
- nvfuser is removed in libtorch 2.13: `SetJitFuserCudaEnabled(True)`
  raises on every build.
- `libtorch_rocshmem.so` prints `(null): No such file or directory` four
  times on stderr at start-up (RDMA `dlopen`s); harmless.
- Haiku ran the mock stubs and both regressions; this time its work and
  reports were correct (diffs reviewed).

**Open / next (nothing designed is left to implement):**
- The CUDA path is unmeasured (no NVIDIA GPU here): the `gbCuda` branch,
  the cuDNN flatten call, `GpuRuntimeVersion`/`CudnnVersion` values.
- H29 (`.ot` interchange with tch-rs-written files) still needs a Rust
  toolchain.
- The next milestone has no design yet: pick a topic with the user and run
  a design session (Fable sub-agent, as sessions 6–8).

## Local session 7 (2026-10-12): milestone 3 layers (4.6.10) implemented

**Where things stand.** 4.6.10 is implemented and committed; the working
tree is clean. Steps 1–6, 8 and 9 of design session 7's local list are
done; step 7 (optional mock check) was skipped, since names and order are
checked against libtorch.

- `src/TchNN.pas`: `TConvConfigND`, `DefaultConvConfigND`,
  `ConvConfigToND`, `PadWithMode` (replaces the private `ModeNames`),
  `TConvND` + `TConv1D/2D/3D` (`TConv2D` rebased, API unchanged), the
  conv-transpose family, `TGroupNorm`, `TRNNConfig`/`TRNNState`/`TRNN`/
  `TLSTM`/`TGRU`, `TIdentity`.
- `tests/nn/nn_test.pas` group 11, items 1–11 of 4.6.10 (44 checks): all
  OK against libtorch 2.13.0 CPU, every measured number reproduced
  (conv1d modes, PyTorch-order ND padding `[[7, 10], [5, 8]]`, transpose
  1D/2D/3D, group norm ±0.99998, GRU `0.380797`, LSTM `0.380797 /
  0.181700`, the ten tch-rs RNN configs, `BatchFirst False`). Group 8's
  conv numbers are unchanged through the new base.
- Regression: `tests/abi`, `tests/api`, `tests/nn`, `tests/jit`, mock —
  all OK.
- `docs/UML.md`/`UML_jp.md`: 4.6.10 elements no longer planned.

**Deviations from the design, found while implementing** (recorded at the
end of DESIGN 4.6.10):
- The RNN constructors' parameter is `AHiddenDim`: `HiddenDim` is a
  duplicate identifier next to the `HiddenDim` property.
- The protected `TRNN.Create` draws warning 3018; silenced with
  `{$push}{$warn 3018 off}` as `TJitModule.CreateRaw`.

**Next:** **4.8 GPU** — the only design item left unimplemented. ROCm
prerequisites are met (`libtorch-rocm/`, H32/H33); start at step 1 of
design session 8's local list below. H31, H34, H35 open. The `TRNN`
cuDNN flatten call (H30) now has its `TRNN` to go into.

## Local session 6 (2026-10-12): TorchScript objects (4.6.9) implemented

**Where things stand.** 4.6.9 is implemented and committed; the working
tree is clean. Steps 1–8 of design session 6's local list are done.

- `src/TchSys.pas`: the four externals (`atm_create_class_`,
  `ati_object_method_`, `ati_object_getattr_`, `ati_clone`); "Not
  declared" is down to `atm_fuser_cuda_*`.
- `src/TchJit.pas`: `ivObject = 14`, `IJitObject` (forward-declared
  before `TIValue`), `TJitObjectHandle`, `FromObject`/`:=`/`AsObject`,
  `ToC` clones, `FromC` keeps the pointer (`Keep`),
  `TJitModule.CreateClass`, `JitLiveObjects`.
- `tests/jit/jit_test.pas`: `TestObjects`, items a–g of 4.6.9 (29
  checks), all OK against libtorch 2.13.0 CPU. **H26, H27, H28 hold**
  (results recorded in DESIGN 4.6.9). `[O, 42]` converts element-wise
  through `:=` (was unmeasured).
- Mock: `mock_tch.c` tag-14 objects, `jit_mock_test.pas` `TestObjects`;
  `tests/mock/run.sh` OK.
- Regression: `tests/abi`, `tests/api`, `tests/nn`, `tests/jit`, mock —
  all OK.
- `docs/UML.md`/`UML_jp.md`: 4.6.9 elements no longer planned.

**Deviations from the design, found while implementing:**
- `CreateClass(const ClassName: …)` does not compile: `ClassName` is
  `TObject.ClassName` (duplicate identifier). The parameter is
  `QualifiedName`; DESIGN 4.6.9 updated.
- Mock counts: a mock object is one cell **plus one per attribute**
  (attributes are cloned ivalues), so the design's
  `mock_live_ivalues = 1` is `Cells` (3 for `[1, 2.5]`) in the test. Item
  7 (failing Device sibling) is checked as a delta around the statement:
  temporaries from items 5–6 still hold handles inside the routine (H1).
  The delta is 0 — the converted sibling is released at once.

**Working with Haiku (new this session).** The user wants Claude Haiku
4.5 sub-agents for mechanical work and for running tests and reporting.
This session: Haiku wrote the mock stubs (from the 4.6.9 spec), synced
UML en/jp, ran the regression. Its mock had a leak (`ati_object_method_`
cloned the elements and `ati_tuple` cloned them again), and it reported
the resulting failures as "unrelated"; its UML edit missed one sentence.
Both fixed here. So: give Haiku fully specified tasks, tell it to report
failures verbatim without interpreting them, and review its diff.

**Next (ask which first if the user does not say):**
- **4.6.10 milestone 3 layers** (CPU only; local list in design session 7
  below). 4.8's `TRNN` flatten call depends on it.
- **4.8 GPU** (ROCm prerequisites met; design session 8's local list
  from step 1; H31, H34, H35 open).

## Local session 5 (2026-10-11/12): three designs in, ROCm libtorch obtained

**Where things stand.** Everything below is committed and pushed. Nothing
is half-done; the working tree is clean.

- **Design sessions 6–8 (sections below)** were run as Claude Fable 5.1
  sub-agents from this local session, one item at a time, each reviewed
  and committed here: TorchScript objects (DESIGN 4.6.9), milestone 3
  layers (4.6.10), CUDA/ROCm (4.8). Cost was about $5–6 per design on
  Anthropic credit; a focused prompt (named sections only, a tool-call
  budget, no downloads) kept the later ones cheaper. H36 was checked here
  against tch-rs 4227b89b `src/nn/rnn.rs` (resolved; stride0 2 without
  biases is a stated deviation).
- **`docs/UML.md` / `UML_jp.md`**: eight Mermaid diagrams of what is
  implemented and what is planned (units, class diagrams per unit, the
  ForwardIV ownership and fine-tuning sequences, build variants) plus a
  status table. Update it when a planned element is implemented.
- **ROCm libtorch obtained** (the user-side prerequisite 1 of design
  session 8): `libtorch-shared-with-deps-2.13.0+rocm7.1.zip` from
  `https://download.pytorch.org/libtorch/rocm7.1/`, unpacked as
  `libtorch-rocm/` (12 GB, `build-version` = `2.13.0+rocm7.1`; the zip was
  deleted afterwards). `.gitignore` now has `libtorch-*/`. The archive
  name has no `cxx11-abi` part any more; it is the cxx11 ABI. A
  2.14.1+rocm7.14 archive (1.6 GB) exists too but was not taken: our
  generated ops are from 2.13.0.
  - **(H33) resolved:** `libtorch_hip.so` contains `gfx1201` (RX 9070),
    also gfx1200, gfx11xx, gfx1030, gfx9xx; not `gfx1036` (the iGPU), so
    `HIP_VISIBLE_DEVICES=0` stays required.
  - **(H32) resolved:** `ldd libtorch_hip.so` finds everything;
    `libamdhip64`, `libMIOpen`, `librocblas`, `libhipblaslt`, `librccl`,
    `libhsa-runtime64` are bundled in `lib/`. No AMD ROCm packages needed.
    **Amended in local session 8:** the zip carries no symlinks; eight must
    be added before anything loads (DESIGN 4.8 "Results").
  - **Trap:** `LD_LIBRARY_PATH=libtorch-rocm/lib` makes system tools load
    the bundled `libtinfo.so.6` (bash warns "no version information
    available"). Use the per-build rpath of DESIGN 4.8, never a global
    `LD_LIBRARY_PATH`.
- **Working agreement with the user (also in auto-memory):** after a large
  download finishes, report and stop; unpack/verify only when asked.

**Next (the user starts it after their 18:00 token reset; ask which first
if they do not say):**
- **4.8 GPU** — the ROCm prerequisites are met, so design session 8's
  local list below can start at its step 1 (prerequisite steps 1–3 are
  done: archive present, H32 holds, device access works from the local
  login). H31, H34, H35 are the open hypotheses.
- **4.6.9 TorchScript objects** and **4.6.10 milestone 3 layers** — CPU
  only, local lists in design sessions 6 and 7 below.
- Keep DESIGN/HANDOFF/UML (en + jp) current at the end, as usual.

## Design session 8 (2026-10-11): CUDA/ROCm designed — implemented in local session 8 above

Answers item 3 of "Handing back (end of local session 4)" below — the last
open one. Design only: nothing committed, nothing downloaded or installed;
one throwaway cmake build in the scratchpad. Details and rationale in
DESIGN 4.8.

**Decisions** (DESIGN 4.8):
- One unpacked libtorch per variant (`libtorch/` CPU as today,
  `libtorch-rocm/`, `libtorch-cuda/`) and one build dir per variant
  (`build/libtch`, `build/libtch-rocm`, `build/libtch-cuda`). Pascal units,
  C wrapper and ABI are unchanged: HIP masquerades as CUDA (confirmed in
  `libtorch/include/ATen/Context.h:638-650`), so `TDevice.Cuda(0)`,
  `device_of_int`, `atc_*` address the AMD GPU as is.
- `csrc/libtch/CMakeLists.txt` gains `-DTCH_LIBTORCH_DIR=<dir>`: manual
  include/link lines (`-Wl,--no-as-needed libtorch.so`, then
  `libtorch_cpu.so libc10.so`, `-D_GLIBCXX_USE_CXX11_ABI=1`, `BUILD_RPATH
  <dir>/lib`) instead of `find_package(Torch)`, because a ROCm archive's
  `Caffe2Config.cmake` includes `LoadHIP.cmake`, which wants the ROCm SDK
  at `/opt/rocm`. Default path unchanged for the CPU archive.
- L2: `atc_cudnn_is_available`, `atc_manual_seed(_all)`, `atc_synchronize`,
  `atc_user_enabled_cudnn`, `atc_set_user_enabled_cudnn`,
  `atc_set_benchmark_cudnn`, `at_context_has_cuda/hip/cudnn`,
  `at_context_version_cudart/cudnn`, `atm_fuser_cuda_*`. L3 (4.7 block):
  `CudnnIsAvailable`, `CudaManualSeed(All)`, `CudaSynchronize(Index = -1)`,
  `CudnnUserEnabled`/`SetCudnnUserEnabled`, `SetCudnnBenchmark`,
  `TGpuBackend = (gbNone, gbCuda, gbRocm)` + `GpuBackend`,
  `GpuRuntimeVersion`, `CudnnVersion`, `TDevice.ToString`; `TchJit`:
  `JitFuserCudaEnabled`/`SetJitFuserCudaEnabled`. Only `CudaIsAvailable` /
  `CudaDeviceCount` are safe on a CPU build; the others raise
  `ETorchNativeError` there.
- `TRNN`: tch-rs's flatten call when `Device.IsCuda and CudnnIsAvailable`
  (`_cudnn_rnn_flatten_weight(FFlatWeights, 4 or 2, InputSize, Mode 2/3,
  HiddenSize, 0, NumLayers, BatchFirst, Bidirectional)`), result
  discarded, `ETorchNativeError` swallowed (hint only; MIOpen may refuse).
- `tests/gpu/gpu_test.pas` + `run.sh <variant>`: `SKIP` + exit 0 when
  `CudaIsAvailable` is False; otherwise ten items reusing the CPU numbers
  of 4.6.7/4.6.8/4.6.10 (`3 5 7`, `6`, `[0.9, 0.8]`, `0.380797`,
  `0.1817`), H13's real device move (item 5) and H30 (item 8). Run with
  `HIP_VISIBLE_DEVICES=0` so the Raphael iGPU (`gfx1036`, unsupported) is
  hidden and `CudaDeviceCount = 1`.
- Mock: new `atc_*`/`at_context_*` stubs return what a CPU build returns;
  `mock_test.pas` checks `GpuBackend = gbNone`, the fuser flag round trip.

**Inspected** (nothing changed): `rocm-smi --showproductname` → GPU[0] RX
9070 `gfx1201`, GPU[1] Raphael iGPU `gfx1036`; `/sys/class/kfd` lists both
(kernel 7.0.0 amdgpu/KFD work). Installed ROCm user space is **only**
`rocm-smi 7.1.1` + `librocm_smi64` (Ubuntu); no `/opt/rocm`, no
`libamdhip64`, no MIOpen, no `rocminfo`. User groups: `joerg adm cdrom sudo
dip plugdev users lxd ollama` — not in `render`/`video`, but `/dev/kfd` is
already read-write for the user through the seat ACL. Measured: the
manual-link CMakeLists builds libtch against the CPU archive in 28 s; the
result `NEEDED`s only `libtorch_cpu.so` + `libc10.so` (as-needed dropped
the umbrella — hence the explicit `--no-as-needed` for GPU builds, where
the backend is reached only through `libtorch.so`).

**New hypotheses** (need the ROCm libtorch; none measurable here):
- **H31** the manual-link libtch works unchanged against the ROCm archive
  (no extra defines such as `__HIP_PLATFORM_AMD__` needed by
  `torch_api.cpp`).
- **H32** the `libtorch-cxx11-abi-shared-with-deps-2.13.0+rocmX.Y` archive
  bundles the HIP runtime, MIOpen, rocBLAS, hipBLASLt, RCCL in `lib/`
  (check `ldd libtorch-rocm/lib/libtorch_hip.so | grep 'not found'`). If
  not: AMD's ROCm runtime packages of the same X.Y.
- **H33** that archive contains `gfx1201` code objects (`strings -n 7
  libtorch_hip.so | grep -oE 'gfx[0-9a-f]+' | sort -u`). Not visible
  locally; PyTorch's ROCm 7.x builds are expected to, ROCm 6.x ones are
  not.
- **H34** `atc_cudnn_is_available` is 1 on ROCm (MIOpen stands in for
  cuDNN through the hipified hooks).
- **H35** `at_context_has_hip` is True on ROCm; whether
  `at_context_has_cuda` is also True is open (`GpuBackend` checks HIP
  first, so it is right either way; the test prints both).
- **H36** tch-rs `rnn.rs` flatten call reconstructed from memory (repo not
  local): `is_cuda && cudnn_is_available`, stride0 4, mode 2 (LSTM) /
  3 (GRU), proj_size 0. **Resolved** by the orchestrating session against
  tch-rs 4227b89b `src/nn/rnn.rs` (lines 154–166, 239–251): matches;
  tch-rs hard-codes stride0 4 even without biases, DESIGN 4.8 uses 2
  then (PyTorch), a stated deviation.

**User-side prerequisites** (not done by the design session; described
only):
1. Obtain `libtorch-cxx11-abi-shared-with-deps-2.13.0+rocmX.Y.zip` from
   download.pytorch.org (ROCm variant of the same 2.13.0 used for CPU;
   pick the highest ROCm version offered for 2.13.0 — RDNA4/gfx1201 needs
   ROCm 7.x) and unpack it as `libtorch-rocm/` in the repo root
   (git-ignored). Verify H33 and H32 with the two one-liners above
   before building anything.
2. If H32 fails: install AMD's ROCm runtime packages matching X.Y
   (`rocm-hip-runtime`, `miopen-hip`, `rocblas`, `hipblaslt`, `rccl`,
   `hipfft`, `hipsparse`, `rocsolver`, `hiprand`) from the AMD repo. No SDK
   (`rocm-dev`/cmake files) is needed with the manual-link mode.
3. Device access: `/dev/kfd` and `/dev/dri/renderD*` are `root:render`
   with an ACL (`+`), and `test -r /dev/kfd -a -w /dev/kfd` already
   succeeds from the desktop login (logind grants the seat), so nothing is
   needed for runs from the local session. Only for ssh/cron runs add the
   user to `render` and `video` (`sudo usermod -aG render,video joerg`,
   re-login). Without access HIP reports no devices even though
   `rocm-smi` (which reads sysfs) sees them.
4. For a CUDA box instead: the `+cuXYZ` archive as `libtorch-cuda/` and
   the NVIDIA driver; everything else is identical with `cuda` as the
   variant.

**Local implementation list** (mechanical; order matters):
1. `csrc/libtch/CMakeLists.txt`: the `TCH_LIBTORCH_DIR` branch of DESIGN
   4.8; rebuild `build/libtch` the old way to confirm nothing changed, and
   once with `-DTCH_LIBTORCH_DIR=$PWD/libtorch` into `build/libtch-manual`
   to confirm the manual path (CPU) passes `tests/api`.
2. `src/TchSys.pas`: the L2 externals above (QWord for `uint64_t`, Int64
   for `int64_t`, `Boolean` for C `bool`, `cint` for `int`); trim the unit
   comment's "Not declared" list to nothing.
3. `src/Tch.pas`: the 4.7 block additions, `TGpuBackend`, `GpuBackend`,
   `TDevice.ToString`; API comments stating which calls are CPU-safe.
4. `src/TchJit.pas`: `JitFuserCudaEnabled`/`SetJitFuserCudaEnabled`.
5. `src/TchNN.pas`: the flatten call in `TRNN`'s protected constructor
   (after `FFlatWeights` is complete), guarded and swallowed as above.
6. `tests/mock/mock_tch.c` stubs + `mock_test.pas` checks; `tests/jit`
   adds the fuser-flag round trip on CPU. Run `tests/mock/run.sh`,
   `tests/api`, `tests/nn`, `tests/jit` on the CPU build: unchanged.
7. `tests/gpu/gpu_test.pas` and `tests/gpu/run.sh` (variant argument,
   visibility variable, `-Flbuild/libtch-$v -k-rpath=...`); first run it
   against `build/libtch`: exactly `SKIP: no GPU in this libtorch`, exit 0.
8. After the user-side prerequisites: `cmake -S csrc/libtch -B
   build/libtch-rocm -DTCH_LIBTORCH_DIR=$PWD/libtorch-rocm
   -DCMAKE_BUILD_TYPE=Release && cmake --build build/libtch-rocm -j 8`;
   `ldd build/libtch-rocm/libtch.so | grep 'not found'` must be empty.
9. `tests/gpu/run.sh rocm`: record item 1's capability print, the H13 and
   H30 outcomes, and H31–H35 results next to each hypothesis (DESIGN 4.8
   and here); then `LD_LIBRARY_PATH=build/libtch-rocm` in front of the
   CPU-linked `tests/nn` and `tests/jit` binaries — everything still passes
   on CPU tensors with the GPU libtorch loaded (same ABI).
10. Add a "Local session 7" section here (it became local session 8) (en +
    jp); mark H31–H36 resolved
    or amended; update DESIGN 4.8 numbers if any differ.

## Design session 7 (2026-10-11): milestone 3 layers designed — implemented in local session 7 above

Answers item 2 of "Handing back (end of local session 4)" below; item 3
(CUDA/ROCm) stays open. Design only: nothing committed, no code outside the
scratchpad. Ran on the local machine against the real libtorch 2.13.0 (CPU)
and FPC 3.2.2, so the numbers in DESIGN 4.6.10 are measured, not derived.

**Decisions** (details and rationale in DESIGN 4.6.10):
- tch-rs's generic `ConvConfigND<ND>` becomes one non-generic
  `TConvConfigND` with `TInt64Array` fields (`Nd = Length`), one base class
  `TConvND` (`Forward` switches on `FNd`), and `TConv1D/2D/3D` subclasses
  holding only the scalar-config constructors. `TConv2D`'s API is unchanged;
  its body moves to the base. `ConvConfigToND(Cfg, Nd)`,
  `DefaultConvConfigND(Nd)`, `PadWithMode` (public, `'constant'` for
  `pmZeros`) are added. Same shape for `TConvTransposeND` + `1D/2D/3D`
  with `TConvTransposeConfig[ND]` (weight `[In, Out div Groups, K..]`).
- `TGroupNorm(NumGroups, NumChannels[, Cfg])` over `group_norm`, weight
  then bias `[NumChannels]` when `Affine`.
- `TRNN = class abstract` (not a `TModule`, as tch-rs) with `ZeroState`,
  `Step`, `Seq`, `SeqInit`; one `TRNNState = record H, C: TTensor end`
  for both `TLSTM` and `TGRU` (GRU leaves `C` nil); `TRNNConfig` shared;
  `FFlatWeights` registered as `weight_ih_l{L}[_reverse]`, `weight_hh_…`,
  `bias_ih_…`, `bias_hh_…` per layer per direction. `SeqInit` calls the
  generated `lstm`/`gru` with the `out` results going to the function
  result (output) and `OutState.H`/`.C`.
- `TIdentity` for `nn::Id`.
- Deviations from tch-rs, each stated in 4.6.10: PyTorch's order for
  reversed-padding-repeated-twice (tch-rs's is wrong for asymmetric ND
  padding, measured); `BatchFirst False` handled in `Seq`/`Step`; plain
  `TInit` instead of `Option<Init>` for the bias inits; `ZeroState` reads
  kind/device from the first weight at call time; cuDNN flattening left to
  the CUDA/ROCm milestone.

**Measurements** (probe `scratchpad/fable-layers/probe*.pas`, FPC 3.2.2 +
libtorch 2.13.0 CPU):
- M-A: base constructor with `array of Int64` + two `overload` subclass
  constructors resolve for the scalar calls and for the inherited ND call
  through the subclass. Group-8 conv2d numbers unchanged through the base.
- M-B / M-B2: `SeqInit(X, S, S)` nils `S` on entry (H8) → `ETorchError
  "undefined tensor passed as hx to gru"`; `S := Step(X, S)` is safe
  (result through a temporary).
- M-C: every number in the 4.6.10 test list (conv1d four modes, conv3d,
  asymmetric ND padding, conv_transpose 1D/2D/3D incl. stride 2 and
  `OutputPadding`, group_norm ±0.99998, GRU `0.380797`, LSTM `0.1817 /
  0.380797`, the five tch-rs RNN configs' shapes and variable names,
  `batch_first` False shapes).

**New hypotheses** (not measurable with CPU libtorch here): **H29** `.ot`
interchange with tch-rs-written RNN/conv-transpose/group-norm files (needs
Rust); **H30** the cuDNN flatten call on a CUDA/ROCm store (item 3).

**Local implementation list** (mechanical; all in `src/TchNN.pas` and
`tests/nn/nn_test.pas`, no L2 or C changes):
1. Add `TConvConfigND`, `DefaultConvConfigND`, `ConvConfigToND`,
   `PadWithMode` (replace the private `ModeNames`, `'constant'` for
   `pmZeros`); `TConvND` with the ND constructor (length checks raise
   `ETorchError`), `FPadTwice[2*I] = FPadTwice[2*I+1] = Padding[Nd-1-I]`,
   `Forward` as the 4.6.10 table. Rebase `TConv2D` on it (body →
   `inherited Create(P, In, Out, [K, K], ConvConfigToND(Cfg, 2))`); add
   `TConv1D`, `TConv3D`. Run the existing group 8: unchanged.
2. `TConvTransposeConfig`, `TConvTransposeConfigND`, the two `Default*`,
   `ConvTransposeConfigToND`, `TConvTransposeND`, `TConvTranspose1D/2D/3D`.
3. `TGroupNormConfig`, `DefaultGroupNormConfig`, `TGroupNorm`.
4. `TRNNConfig`, `DefaultRNNConfig`, `TRNNState`, `TRNN` (protected
   constructor = `rnn_weights` with `GateMul` 4 or 3; `Step`, `Seq`,
   `BatchDimIndex`), `TLSTM`, `TGRU` (`ZeroState`, `SeqInit`). Document
   the `InState`/`OutState` distinct-variables rule at the declaration.
5. `TIdentity`.
6. `tests/nn/nn_test.pas` group 11, items 1–11 of the 4.6.10 list; keep
   the final `TchLiveHandles = 0`.
7. Mock (`tests/mock`): nothing numerical to add; optionally construction
   and registration order of `TLSTM(2, 4)` (16 names for 2 layers
   bidirectional) over the shape-only stubs.
8. Regression: `tests/nn`, `tests/api`, `tests/mock/run.sh`, `tests/jit`.
9. Record results next to H29/H30 only if they become testable; add a
   "Local session 6" section here (it became local session 7) (en + jp).
   Item 3 of "Handing back (end
   of local session 4)" remains the open design work.

## Design session 6 (2026-10-11): TorchScript objects designed — implemented in local session 6 above

Answers item 1 of "Handing back (end of local session 4)" below; items 2
(milestone 3 layers) and 3 (CUDA/ROCm) stay open. Design only: nothing
committed, no code outside the scratchpad. The session ran on the local
machine, so the design is measured against the real libtorch 2.13.0 (CPU)
and the built `libtch.so`. Everything is in **DESIGN 4.6.9**; summary:

1. **Handle.** `IJitObject`, an interface refcounted by `TInterfacedObject`
   (as 4.1's `TTensorHandle`; interlocked, deterministic release), with
   the implementing class `TJitObjectHandle` hidden in the implementation
   section. An interface rather than a record because `TIValue` needs the
   field and the object's methods take and return `TIValue`: FPC cannot
   forward-declare a record but can forward-declare an interface
   (measured). `JitLiveObjects` counts live handles like `TchLiveHandles`.
2. **`TIValue`.** `ivObject = 14`, field `Obj: IJitObject`, `FromObject`,
   `:=` from `IJitObject`, `AsObject`, `TagName = 'Object'`. **`FromC`
   keeps the pointer for tag 14** (`Keep` flag; the handle owns it);
   **`ToC` returns `ati_clone(Obj.Raw)`** — the caller frees the clone,
   the handle's pointer is never handed to C for freeing (tch-rs's fix
   for issue 475). Inside containers nothing special: `ati_to_*` hand out
   fresh `IValue`s (measured) and a failing later child releases the
   converted siblings through the compiler's unwinding of
   `IValuesFromC`'s local array.
3. **API.** `IJitObject.MethodIV(Name, Inputs)` and `GetAttr(Name)`
   (shape of `TJitModule.MethodIV`); `TJitModule.CreateClass(ClassName,
   Inputs): IJitObject` (qualified name, e.g. `'__torch__.InputObject'`;
   returns the object rather than tch-rs's `IValue`, stated). An object
   does not keep its module alive and does not need to (measured: method
   calls work after `atm_free`).
4. **L2.** Four externals (`atm_create_class_`, `ati_object_method_`,
   `ati_object_getattr_`, `ati_clone`), declarations in 4.6.9. **Fork
   patch: none** (the unused local in `atm_create_class_` is harmless).
5. **Tests.** Group 14 of `tests/jit/jit_test.pas`: `jit6` (`foo6.pt`,
   `y()` → `[6, 8, 10]`), `jit_double_free` (`foo7.pt`,
   `make_input_object` then `add_them` → `[5, 7, 9]`, twice with the same
   handle), `CreateClass`, the measured error messages, objects through
   tuple/list/dict, sharing, object after `M.Free`, counts back, a `10^4`
   loop. Mock: tag-14 `mock_ivalue` with named attributes `a0, a1, …`,
   `mock_last_class`, `TestObjects` (eight checks).

**Measurements (libtorch 2.13.0 CPU, FPC 3.2.2; probes in the scratchpad
only):**
- M9: raw C entry points on `foo6.pt`/`foo7.pt` — `ati_tag` 14 for both
  object results; `y()` → `[6, 8, 10]`; `getattr x` → `[3, 4, 5]`;
  `ati_clone` works after `ati_free` of the original and after `atm_free`
  of the module; a tuple child comes back as a fresh pointer usable after
  the tuple is freed; `add_them` twice with the same object pointer →
  `[5, 7, 9]` twice; `atm_create_class_('__torch__.InputObject', [T1,
  T2])` → `add_them` `[5, 7, 9]`; the bare class name fails. Nine error
  messages recorded verbatim in 4.6.9 (missing attribute/method, arity,
  `to_tensor`/`length` on an object, unknown class, missing `__init__`
  argument, wrong argument type, non-object pointer).
- M10: FPC accepts `IJitObject = interface;` forward → record field →
  full interface naming the record; copies share the handle, release is
  deterministic, the count returns to 0 after the routine returns (inside
  it a temporary keeps 1, H1); `SizeOf` 16.
- M11: `atm_forward` throws "forward did not return a tensor" for any
  non-tensor result (read in `torch_api.cpp`, not run on an object → H27).

**New hypotheses (need real libtorch):**
- **(H26)** an object round-trips through `GenericList` and as a
  `GenericDict` value as through a tuple (fresh tag-14 `IValue`s that
  outlive the container).
- **(H27)** `CreateClass('__torch__.TorchScriptClass', [x])` on `foo6.pt`
  gives `y() = 2x`; `Forward` (tensor API) on `foo6.pt` raises "forward
  did not return a tensor".
- **(H28)** `10^4` object method calls leave `JitLiveObjects`,
  `TchLiveHandles` and RSS flat.

**Local session, in this order (mechanical):**
1. `src/TchSys.pas`: the four externals of 4.6.9 after `ati_free`; the
   unit comment's "Not declared" list becomes `atm_fuser_cuda_*` only.
2. `src/TchJit.pas`, interface: `ivObject = 14` in `TIValueTag`;
   `IJitObject = interface;` before `TIValue`; `Obj: IJitObject` field,
   `FromObject`, `class operator :=(const O: IJitObject)`, `AsObject`;
   the full `IJitObject` (new GUID; `Raw`, `MethodIV`, `GetAttr`) after
   `TIValue`; `TJitModule.CreateClass`; `function JitLiveObjects:
   Integer`. Header comment: drop "objects not supported", add "an object
   does not keep its module alive".
3. `src/TchJit.pas`, implementation: `TagNames` → `'Object'`;
   `LiveObjects` counter + `JitLiveObjects`; `TJitObjectHandle`
   (constructor order `inherited; Inc; FRaw := P`, destructor `ati_free`
   + `Dec`); `AsObject`/`FromObject`/`:=`; `ToC` branch `ivObject`
   (`ati_clone`, nil `Obj` → `ETorchError('ivalue: nil object')`); `FromC`
   case 14 with the `Keep` local and `if not Keep then ati_free(P)` in the
   `finally`; `TJitObjectHandle.MethodIV`/`GetAttr` and
   `TJitModule.CreateClass` as 4.6.9 step 5 (`IValuesToC`, call,
   `TchCheck`, `finally FreeIValues`, `FromC`; `CreateClass` ends with
   `.AsObject`). Compile `tests/mock/run.sh` (the mock lacks the symbols
   until step 4: the test programs reference only what they use, so the
   existing three still link).
4. Mock: `cls`/`names` in `mock_ivalue`, `iv_clone`/`ati_free` updates,
   `atm_create_class_` + `mock_last_class`, `ati_object_getattr_`,
   `ati_object_method_`, `ati_clone`, `ati_length` failing on 14.
5. `tests/mock/jit_mock_test.pas`: `mock_last_class` external,
   `TestObjects` with the eight checks of 4.6.9 (counts compared from the
   caller), `JitLiveObjects = 0` at the end. `tests/mock/run.sh`.
6. `tests/jit/jit_test.pas`: group 14 "Objects" items a–g of 4.6.9,
   checking H26–H28; `JitLiveObjects = 0` next to the final
   `TchLiveHandles = 0`. Build and run as the file header says.
7. Regression: `tests/jit`, `tests/mock/run.sh`, `tests/abi` (C
   unchanged), `tests/api`, `tests/nn`.
8. Record H26–H28 results next to the hypotheses in DESIGN 4.6.9 (en +
   jp), add a "Local session 5" section here (en + jp); items 2 and 3 of
   "Handing back (end of local session 4)" remain the open design work.

## Local session 4 (2026-10-11): milestone 3 (JIT) implemented

Steps 1–8 of cloud session 5's list (below) are done; every test passes.
Commits `8a912eb` … `b816950`.

1. **`FromArray` revision** (DESIGN 4.3): the `Single` overload is gone, the
   four others take `AKind` (`Double` + `kFloat` rounds into a `Single`
   buffer in Pascal), `FromSingleArray` added. `tests/compiler` has the new
   `h2_fromarray.pas` and `xfail_h2_single.pas`; `tests/api` and `tests/nn`
   updated. Trap found in the test: a real literal in a comparison is
   `Extended` on x86-64, so `T.DoubleValue([0]) = 0.9` is False; compare
   with `Double(0.9)`.
2. **4.6.7 follow-ups:** `{$push}{$warn 3018 off}` around
   `TOptimizer.CreateRaw` (no warning left); the half-precision
   `LoadPartial` check added to `TestSaveLoad` (passes: `u2` `kHalf` mean
   42, `v2` `kFloat` mean 1, missing `['a.b.t2']`).
3. **Fork patch:** `atm_create`, `atm_register_parameter`, `atm_define` in
   `csrc/libtch/torch_api.{h,cpp}`; libtch rebuilt; `tests/abi`, mock pass.
4. **Vendored** `tests/jit/models/foo.pt` … `foo8.pt` with tch-rs's two
   license files and a short README naming the source.
5. **`src/TchSys.pas`:** the JIT types and externals of 4.6.8.
6. **`src/TchJit.pas`:** `TIValue`, `TJitModule`, `TTrainableJitModule`,
   `Jit*` globals. Small deviations, recorded at the end of DESIGN 4.6.8:
   `TagName` is a `case` (FPC rejects an enum with explicit values as an
   array index); `CreateByTracing` cleans up after *any* failure past
   `atm_create_for_tracing`; `Item`/`Key`/`Value` range-check.
7. **Mock:** `atm_*`/`ati_*` in `tests/mock/mock_tch.c`,
   `tests/mock/jit_mock_test.pas` (also: a child failing in `ToC` or
   `FromC` leaves no C ivalue behind), added to `run.sh`.
8. **`tests/jit/jit_test.pas`:** all thirteen groups pass; **H19–H25 all
   hold** (results next to the hypotheses at the end of DESIGN 4.6.8).

- Regression: `tests/api`, `tests/nn`, `tests/generated`,
  `tests/ownership`, `tests/abi`, `tests/compiler/run.sh`,
  `tests/mock/run.sh` pass. `tests/fpu/run.sh` behaves as before (it is a
  measurement: the unmasked modes crash or hang by design).
- Not touched, as asked: milestone 3 layers and CUDA/ROCm.

## Handing back (2026-10-11, end of local session 4) — items 1, 2 and 3 answered in design sessions 6, 7 and 8 above

**Design only** (CLAUDE.md). Nothing is blocked; all of the following are
optional next milestones. Mark what needs real libtorch as H26, ...

1. **TorchScript objects (ati_tag 14)** — *designed: DESIGN 4.6.9, design
   session 6 above.* `ati_object_method_`,
   `ati_object_getattr_`, `ati_clone`, `atm_create_class_`; tch-rs
   `Object`, tests `jit6` and `jit_double_free`, assets `foo6.pt`/`foo7.pt`
   (already vendored, unused). Needs the refcounted handle DESIGN 4.6.8
   mentions (an object keeps its C `IValue` alive), and how `TIValue`
   carries it (copies share; `FromC` currently raises on tag 14).
2. **Milestone 3 layers** (DESIGN 4.6.5 deferred list) — *designed: DESIGN
   4.6.10, design session 7 above.*
3. **CUDA/ROCm** — *designed: DESIGN 4.8, design session 8 above.* Only
   CPU so far (H13's real device move untested;
   `atm_fuser_cuda_*` not declared). The local machine has an AMD Radeon
   RX 9070 (Navi 48) and `rocm-smi`; no ROCm libtorch has been tried yet.
   So the GPU route to design first is ROCm (libtorch's HIP build reports
   the device as CUDA).

**How the next design session runs is still open:** the cloud service
credits are used up, and the user will decide later how to send the
design work to Fable. Until then, nothing here waits on it. (Settled in local session
5: Fable 5.1 sub-agents run from the local session, one design at a time.)

## Cloud session 5 (2026-10-10): JIT designed, `FromArray` revised, local session 3 reviewed — implemented in local session 4 above

The request below ("Handing back to the cloud, end of local session 3") is
answered for items 1–3; item 4 (milestone 3 layers, CUDA/ROCm) is
untouched. Design only: nothing committed, no code outside the scratchpad.
Summary:

1. **JIT (DESIGN 4.6.8).** L2: `PTorchModule`, `PTorchIValue`, the 21
   `atm_*`/`at_set_graph_executor_optimize` and 31 `ati_*` externals listed
   there (`atm_named_parameters` reuses `TTorchLoadCallback`). Not declared:
   TorchScript objects (tag 14: `ati_object_*`, `ati_clone`,
   `atm_create_class_`, tests `jit6`/`jit_double_free`, `foo6.pt`/`foo7.pt`
   — a later milestone, they need a refcounted handle) and `atm_fuser_cuda_*`
   (CUDA). `TIValue`: tag values = `ati_tag`'s 0–13 plus input-only
   `ivStringList`/`ivDevice`; static constructors, implicit `:=` from
   `TTensor`/`Int64`/`Double`/`Boolean`/`String`, `As*`/`Item`/`Key`/`Value`
   accessors; `ToC` (caller frees) / `FromC` (consumes) with nil-initialised
   child slots and `finally` cleanup; `GenericDict` as flat key/value
   `Items`. `TJitModule`: `Create(file[, device])`, `CreateFromStream`,
   `CreateByTracing`, `CreateEmpty` + `RegisterParameter` + `Define` (fork
   addition `atm_create`/`atm_register_parameter`/`atm_define`: three
   one-line functions over `Module::define`, `register_parameter` and a
   `training` attribute), `Forward`/`Method` (tensor), `ForwardIV`/`MethodIV`
   (ivalue), `SetEval`/`SetTrain`, `ToDevice(device, kind)`, `Save`,
   `NamedParameters` (4.6.3's trampoline, duplicated in `TchJit`), the
   `Jit*` globals. `TTrainableJitModule = class(TModule)`:
   `atm_load_on_device(P.Device)`, `P.Add(name with '.'→'_', T,
   T.RequiresGrad)`, shared handles; `Forward(X, Train)` applies `Train`
   through `atm_train`/`atm_eval` when it changes (stated deviation: tch-rs
   ignores it). **Test assets:** vendor tch-rs's `tests/foo*.pt` into
   `tests/jit/models/` (license files next to them) **and** build the
   parameterised / `self.training` modules from TorchScript source in the
   test; tracing alone cannot produce parameters. Thirteen test groups with
   hand-checkable numbers in 4.6.8.
2. **`FromArray` (DESIGN 4.3).** Drop the `Single` overload; the
   `Double`/`Int64`/`Byte`/`Boolean` overloads get a trailing `AKind` with
   the natural default (`kFloat`/`kInt64`/`kUInt8`/`kBool`);
   `FromSingleArray` for `Single` buffers (as `FromInt32Array`). Measured:
   every real literal then resolves to `Double` (float32 tensor by default,
   PyTorch's rule and the same bits as Python); `[1, 2]` stays `Int64`; a
   `Single` variable inside a literal converts, an `array of Single`
   variable needs the named function.
3. **Review of local session 3.** lr 1e-10: `‖g‖² = 80.955² + 8.974² =
   6634.2`, `49 · 1e-10 · 6634.2 = 3.25e-5`, confirmed; the measured
   `-3.24e-5` is within the float32 resolution of a loss near 23 (1.9e-6).
   4.6.7 test 7 corrected to tch-rs's *add* / *add_partial* layouts (`t123`
   once). `TVarPath.Group`, the `FullName` validation (`TryGet` raises on
   `.`/`|`) and the `AddEntry` nil check confirmed. `CreateRaw` stays
   private; warning 3018 silenced with `{$push}{$warn 3018 off} … {$pop}`
   (measured: `strict private` warns too). All extra tests adopted into
   4.6.7; one to add (the half-precision half of
   `save_and_load_partial_var_store_incomplete_file`).

**Measurements in the cloud (FPC 3.2.2; scratchpad only):**
- M4: open-array literals against `FromArray` overload sets (table in 4.3):
  a real literal is `Single` only if exactly representable in binary32 —
  `[1.0, 2.0]`, `[2.5]`, `[1.0, 0.1]` pick `Single`, `[0.9, -1.9]`,
  `[3.14]`, `[0.1]` are ambiguous; without the `Single` overload all resolve
  to `Double`; the `AKind` defaults change nothing in the resolution.
- M5: `TIValue` record with explicit tag values (`TIValueTag(12) =
  ivGenericList`), nested tuples, implicit `:=` from `Int64`/`Double`/
  `Boolean`/`String`/record; **`[T, 42, 1.5]` converts element-wise into an
  `array of TIValue` argument** through those operators; `SizeOf(Boolean) =
  1`.
- M6: FPC warns `(3018) Constructor should be public` for `private` and
  `strict private` constructors; `{$push}{$warn 3018 off} … {$pop}` removes
  it.
- M7: the fork's `atm_*`/`ati_*` region of `torch_api.{h,cpp}` is identical
  to upstream 4227b89b (`diff`); the `foo*.pt` are zip format version 3 and
  their TorchScript (`code/__torch__.py`) is summarised in 4.6.8.
- M8: the whole 4.6.8 interface (`TIValue`, `TJitModule`,
  `TTrainableJitModule`, the `Jit*` functions) compiled as a throwaway unit
  with stub bodies against the real `Tch`/`TchNN` units: no errors, no
  warning from the pragma-wrapped `CreateRaw`, and the test call shapes
  `ForwardIV([X, 42, 1.5])`, `Tuple([2.0, 3.0, 4])`, `GenericDict(['bar', X,
  'foo', X])`, `StringList([...])` resolve.

**New hypotheses (need real libtorch):**
- **(H19)** the vendored `tests/jit/models/foo*.pt` load with libtorch 2.13
  and give `jit_tests.rs`'s numbers (4.6.8 items 1–7).
- **(H20)** fork addition: `Module::define` compiles the two TorchScript
  sources of 4.6.8 (`torch.matmul`, `torch.t`, `self.training`); the
  registered `training` attribute makes `self.training` and
  `atm_train`/`atm_eval` work; `named_parameters` returns the registered
  parameters with their `requires_grad`.
- **(H21)** parameter sharing: after one SGD step on the var-store,
  `TTrainableJitModule.Forward(X) = 3.0`, and `Save` + reload keeps the
  trained values.
- **(H22)** `ati_generic_dict`'s `Dict[str, Tensor]` specialisation is
  accepted by `foo8.pt`'s `generate`.
- **(H23)** `atm_end_tracing(M, name, nil, 0)` after a raising trace
  closure leaves tracing usable for the next `CreateByTracing`.
- **(H24)** a `GenericDict` round trip keeps insertion order (`c10::Dict`
  is order-preserving).
- **(H25)** `JitGetProfilingMode` is True in a fresh process.

**Local session, in this order (mechanical):**
1. `FromArray` revision (DESIGN 4.3): `src/Tch.pas` — drop the
   `array of Single` overload, add `AKind` defaults to the four others
   (`kFloat` fast path: convert to a `Single` buffer in Pascal; otherwise
   `totype`), add `FromSingleArray`. Tests: `tests/api/api_test.pas`
   (`TDoubleArray.Create(0.25, 0.5)` is now `kFloat` — pass `kDouble` or
   change the expectation; add `FromArray([0.9, -1.9])`, `FromArray([1, 2],
   kFloat)`, `FromSingleArray`), `tests/nn/nn_test.pas` (the `Double(…)`
   casts can go), `tests/compiler/h2_fromarray.pas` (the new set; `[0.9,
   -1.9]`, `[3.14]`, `[1.0, 0.1]` → Double) plus a new
   `xfail_h2_single.pas` showing `[0.9, -1.9]` ambiguous with both `Single`
   and `Double` overloads. Run `tests/compiler/run.sh`, `tests/api`,
   `tests/nn`, `tests/mock/run.sh`.
2. 4.6.7 follow-ups: `{$push}{$warn 3018 off}` / `{$pop}` around
   `TOptimizer.CreateRaw`; add the half-precision `LoadPartial` check of
   4.6.7 item 7d to `TestSaveLoad`.
3. Fork patch: `atm_create`, `atm_register_parameter`, `atm_define` in
   `csrc/libtch/torch_api.{h,cpp}` (code in 4.6.8); rebuild libtch;
   `tests/abi`, `tests/mock/run.sh` (the mock gets its stubs in step 7; the
   test programs reference only what they use).
4. Vendor `tests/jit/models/foo.pt` … `foo8.pt` plus `LICENSE-MIT` and
   `LICENSE-APACHE` from tch-rs 4227b89b (`tests/`); commit message names
   the source files and `tests/create_jit_models.py`.
5. `src/TchSys.pas`: the JIT types and externals of 4.6.8.
6. `src/TchJit.pas`, in this order so each part compiles alone: `TIValue`
   (record, constructors, operators, accessors) → `ToC`/`FromC` →
   `TJitModule` (load/free/forward/method/eval/train/to/save) →
   `NamedParameters` trampoline → `CreateByTracing` → `CreateEmpty` /
   `RegisterParameter` / `Define` → `Jit*` globals → `TTrainableJitModule`.
7. Mock: `atm_*`/`ati_*` stubs in `tests/mock/mock_tch.c` (4.6.8 "Mock"),
   `tests/mock/jit_mock_test.pas`, add it to `run.sh`.
8. `tests/jit/jit_test.pas` with the thirteen groups of 4.6.8, checking
   H19–H25 as they come up; record each result next to the (Hn) in
   DESIGN.md.
9. Update DESIGN/HANDOFF (en + jp); then hand TorchScript objects (tag
   14), the milestone 3 layers (4.6.5 deferred list) and CUDA/ROCm back to
   a design session if needed.

## Local session 3 (2026-10-10): milestone 2 implemented

Steps 1–5 of cloud session 4's list (below) are done; every test passes.

1. **Fork patch** `ato_set_momentum_group`: `} if` → `} else if`
   (`csrc/libtch/torch_api.cpp`); libtch rebuilt; `tests/abi` and
   `tests/mock/run.sh` pass. A build of the *unpatched* source is kept in
   `build/libtch-unpatched/` (git-ignored) only to measure H16.
2. **L2/L3 additions:** `src/TchSys.pas` has `TTorchLoadCallback`,
   `at_save_multi`, `at_load_callback_with_device`,
   `at_loadz_callback_with_device`, `PTorchOptimizer` and the 14 `ato_*`
   (13 + `ato_free`); `src/Tch.pas` has `TTensor.CopyFrom` (`at_copy_`).
3. **`src/TchNN.pas`** as sketched in DESIGN 4.6, with these small
   additions: `TVarPath.Group`; `TVarPath.FullName` validates the name, so
   `TryGet` rejects `.`/`|` too; `AddEntry` raises on a nil tensor. FPC
   warns "Constructor should be public" for the private
   `TOptimizer.CreateRaw` (harmless, kept private as designed). tch-rs
   sources used for reference: a sparse clone of `src/nn` and `tests` at
   4227b89b in the session scratchpad (not committed).
4. **`tests/nn/nn_test.pas`** (build line in its header): all ten groups of
   4.6.7 pass, plus AdamW's first step (`[0.899, -1.898]`, decay before the
   update), Load of a missing file (`ETorchNativeError`), `CopyFrom`,
   `TVarPath.SetKind` on a sub-tree, an MLP in `TSequential`, method
   pointers in `AddFn`. **H13–H18 all resolved** (results in DESIGN.md next
   to each). Two corrections to 4.6.7: (a) the `LoadPartial →
   ['a.b.t2']` case needs tch-rs's `add_partial` layout (`t123` once), the
   full layout also misses `a.b.ccc.t123__3`; (b) the lr 1e-10 check is
   signed in tch-rs: the loss really drops by 3.25e-5 (hand) / 3.24e-5
   (measured), so an absolute 1e-5 bound fails. `build/nn_test h16` runs
   only the H16 check (for `LD_LIBRARY_PATH=build/libtch-unpatched`).
5. **Mock:** `tests/mock/mock_tch.c` got the 4.6.7 stubs (optimizer
   counters, save names, load callback with `err_after_first` /
   `nil_tensor` file names, `set_requires_grad`, `set_data`, `fill_`,
   `uniform_`, `copy_`, grad mode); `tests/mock/nn_mock_test.pas`; `run.sh`
   runs both programs. In the mock an in-place op returns a copy, not a
   second handle (plumbing only).

- Regression: `tests/api`, `tests/generated`, `tests/ownership`,
  `tests/compiler/run.sh` pass.
- **New measurement, (H2) addendum** (DESIGN 4.3): an open-array literal
  takes its element type from its first element. `[0.9, -1.9]`, `[3.14]`,
  `[1e-3]` are **ambiguous** between the `Single` and `Double` overloads of
  `FromArray`; `[1.0, 0.1]` silently picks `Single`. Workaround in the
  tests: `[Double(0.9), -1.9]`.

## Handing back to the cloud (2026-10-10, end of local session 3) — answered in cloud session 5 above

**Design only** (CLAUDE.md). Mark what needs real libtorch as H19, ...

1. **JIT milestone (`TchJit`).** DESIGN 4.6 fixes the shape (`TJitModule`
   class, `TIValue` tagged record, H10 resolved, `TTrainableJitModule =
   class(TModule)`). Not yet mechanical: which `atm_*` / `ati_*` entry
   points of `torch_api.h` (lines 201–264) are declared and how
   (`atm_forward` vs `atm_forward_` with `ivalue`, `atm_method[_]`,
   `atm_named_parameters` callback — same trampoline pattern as 4.6.3?);
   `TIValue` ↔ `ivalue` conversion and ownership (`ati_free`, tuples,
   generic lists/dicts, which tags in milestone 3); `atm_to`/`atm_eval`/
   `atm_train`; how `TTrainableJitModule` registers parameters into a
   `TVarPath` (names, trainable flag). **Test assets without Python:**
   tch-rs's tests load `tests/foo*.pt` made by `create_jit_models.py`;
   either vendor those files (license: same MIT/Apache as tch-rs) or create
   modules in Pascal through `atm_create_for_tracing` / `atm_end_tracing` /
   `atm_save`. Hand-checkable numbers as in 4.6.7.
2. **`FromArray` literals (H2 addendum).** Decide: document the first-element
   rule only, add named `FromSingles`/`FromDoubles`, or change the overload
   set (e.g. drop `Single` so real literals pick `Double`, which breaks
   PyTorch's float32 default). Consider also `[1.0, 0.1]` rounding silently.
3. **Review the deviations of local session 3 from the design** (details in
   "Local session 3" above and next to each item in DESIGN.md). Confirm
   or correct each, and make DESIGN 4.6 / 4.6.7 say what the code and
   tests now do:
   - 4.6.7 test 3: the lr 1e-10 bound. DESIGN now says tch-rs's check is
     signed and the loss drops by 3.25e-5 (hand) / 3.24e-5 (measured);
     check the arithmetic.
   - 4.6.7 test 7: `LoadPartial → ['a.b.t2']` needs tch-rs's
     `add_partial` layout (`t123` once); with the full layout
     `a.b.ccc.t123__3` is missing too. Not yet corrected in DESIGN 4.6.7.
   - Additions not in the sketch: `TVarPath.Group`; `TVarPath.FullName`
     validates the name (so `TryGet` rejects `.`/`|` too); `AddEntry`
     raises on a nil tensor.
   - FPC warning "Constructor should be public" for the private
     `TOptimizer.CreateRaw`: keep private, or another construction
     pattern?
   - Test additions beyond 4.6.7 (AdamW first step `[0.899, -1.898]`, Load
     of a missing file, `CopyFrom`, sub-tree `SetKind`, MLP, method
     pointers in `AddFn`): adopt into 4.6.7 or leave as test-only.
4. **Optional, later:** milestone 3 layers (DESIGN 4.6.5 deferred list),
   CUDA/ROCm (H13's real device move is untested; CPU only so far).

Keep DESIGN/HANDOFF (en + jp) current at the end, as usual.

## Cloud session 4 (2026-10-10): DESIGN 4.6 settled — implemented in local session 3 above

The request below ("Handing back to the cloud, end of local session 2") is
answered: DESIGN.md 4.6 now has the interface sketch plus subsections
4.6.1–4.6.7, each grounded in a named tch-rs file at 4227b89b. Nothing was
committed; no code was written outside the scratchpad. Summary of the
decisions:

1. **Init (4.6.1).** `TInit` carries `Kind`, `A`, `B`, and the Kaiming
   payload (`Dist`, `Fan`, `NonLin`, `Gain`). Fan and gain formulas from
   `init.rs` (fan-in = `Dims[1] * prod(Dims[2..])`, ReLU gain `sqrt(2)`,
   uniform bound `sqrt(3) * gain / sqrt(fan)`). `Make` = `f_init` op for op
   (`zeros`/`ones`/`mul_scalar`, `uniform_`, `randn`, `empty.normal_`,
   `t_`, `linalg_qr(…, 'reduced', Q, R)`, `diag(0).sign`, `mul_`,
   `mul_scalar_`, `contiguous`, then `.view(Dims)` — the one stated
   deviation, PyTorch's shape rule). No grad guard in `Make`; `Apply`
   (tch-rs `Init::set`) guards itself. Store kind is used everywhere tch-rs
   hard-codes `Float`.
2. **Store (4.6.2).** Append-only `array of TVarEntry` (name, tensor,
   trainable, group) + `array of Integer` of trainable indices (tch-rs
   `trainable_variables`) + `TDictionary<String, Integer>` index (measured
   on 3.2.2, case-sensitive). Duplicate `Add` renames to `<name>__<Count>`
   as tch-rs; `.`, `|` and empty names raise; `GetOrAdd` returns the
   existing tensor regardless of dims. `Freeze`/`Unfreeze` =
   `set_requires_grad` on the trainable entries; `SetDevice`/`SetKind` =
   `set_data(to/totype)` in place; `SetFloatKind` for the `half()` family;
   `CopyFrom` two-pass, under `NoGradGuard`. Equality/containers question
   closed: no `=` on `TTensor` needed.
3. **Save/Load (4.6.3).** `Save` → `at_save_multi` (`.`→`|`); `Load` /
   `LoadPartial` → `at_load_callback_with_device` (`.bin`/`.pt` →
   `at_loadz_callback_with_device`; `.safetensors` raises). `cdecl`
   trampoline with a `TLoadContext` class instance in `data`, exception
   captured with `AcquireExceptionObject`, re-raised after the C call,
   error slot cleared on every path; measured with a C harness. Precision
   update on load (`set_data(totype(file kind))` then copy) as tch-rs.
   `at_load_multi[_]` unused.
4. **Optimizers (4.6.4).** `PTorchOptimizer = Pointer`, externals as in
   `torch_api.h`; construction adds all trainable entries, `ZeroGrad`/
   `Step`/`BackwardStep*` lazily add new ones (`FVarsInOptimizer`).
   Defaults from `nn/optimizer.rs` (AdamW weight decay 0.01, RmsProp alpha
   0.99). `ClipGradValue` = `grad.clamp_`; `ClipGradNorm` = `norm` per
   grad, `stack(...).norm`, `mul_scalar_(coef)`; both under `NoGradGuard`.
   **Fork bug found:** `ato_set_momentum_group` has `} if` instead of
   `} else if` (`torch_api.cpp:564`), to be patched locally.
5. **Layers (4.6.5).** `TLinear` (`linear`), `TConv2D` (`conv2d`, scalar
   `ConvConfig`, padding modes via `pad`), `TBatchNorm` (Nd 1/2/3,
   `batch_norm`, uses `Train`), `TLayerNorm` (`layer_norm`), `TEmbedding`
   (`embedding`), dropout as a function. Defaults and variable names per
   tch-rs file in the 4.6.5 table. Conv1D/3D, ConvTranspose, GroupNorm,
   LSTM/GRU deferred to milestone 3.
6. **Sequential (4.6.6).** `specialize TObjectList<TModule>`; `AddFn` with
   four overloads (function, function+Train, method, method+Train;
   resolution measured); `Train` passed to every child; `ForwardAll`.
7. **Unit/tests (4.6.7).** Single `TchNN.pas`; `TchSys` gets the
   optimizer/save/load externals, `Tch.pas` gets `TTensor.CopyFrom`;
   `tests/nn/nn_test.pas` with the hand-checkable numbers listed there
   (initial loss 22.99967, first SGD step `w = 0.80955`, `b = 0.08974`,
   Adam first step `lr·sign(g)`, clip numbers from tch-rs, conv padding
   vectors, `LoadPartial → ['a.b.t2']`); the mock covers bookkeeping,
   callback and lazy-add counts only.

**Measurements in the cloud (FPC 3.2.2, gcc 13; scratchpad only):**
- M1: `cdecl` trampoline called three times from a C harness with a class
  instance in `data`; `AcquireExceptionObject` in the handler, re-raise after
  the call: class and message intact, later callbacks skipped, all handles
  owned.
- M2: `Generics.Collections` with `TTensor` against the mock libtch:
  `TDictionary<String, Integer>` case-sensitive, duplicate `Add` →
  `EListError`; `array of record` with a `TTensor` field and
  `TList<TTensor>` release handles correctly (only the H1 expression
  temporary lingers until routine exit); `TObjectList<TModule>` with an
  abstract base and `for .. in` works; `Contnrs.TObjectList` too.
- M3: four `AddFn` overloads on function / method pointers with and without
  `Train` resolve unambiguously in objfpc, chaining works.
- Side effect: `build/mock/libtch.so` was stale (built before local session
  2's stubs) and has been rebuilt with `tests/mock/run.sh` (45 checks pass).
  `build/` is git-ignored; nothing to commit.

**New hypotheses (need real libtorch):**
- **(H13)** `set_data` keeps shared handles in sync: after `SetDevice` /
  `SetKind` a layer's `Weight` and an optimizer created earlier see the new
  storage (tch-rs relies on this).
- **(H14)** Orthogonal init: `Q·Qᵀ = I` for `[3, 5]`, `Qᵀ·Q = I` for `[5, 3]`
  (`allclose` 1e-5), result shape = `Dims`.
- **(H15)** Load precision update: a `kFloat` store loading a `kHalf` file
  ends up `kHalf` with the right values.
- **(H16)** Unpatched `ato_set_momentum_group` raises "unexpected optimizer"
  on Adam; patched (`} else if`) does not.
- **(H17)** `TInit.Apply` on a trainable leaf fails without `NoGradGuard`
  ("a leaf Variable that requires grad is being used in an in-place
  operation") and works with it.
- **(H18)** `Freeze; Step` leaves the values unchanged and raises nothing
  (libtorch skips parameters with an undefined grad).

**Local session, in this order (mechanical):** (steps 1–6 done in local session 3)
1. Fork patch: `csrc/libtch/torch_api.cpp` `ato_set_momentum_group`:
   `} if (auto sgd` → `} else if (auto sgd`; rebuild libtch; `tests/abi`,
   `tests/mock/run.sh`.
2. `src/TchSys.pas`: `PTorchOptimizer`, `TTorchLoadCallback`,
   `at_save_multi`, `at_load_callback_with_device`,
   `at_loadz_callback_with_device`, the 13 `ato_*` (4.6.4 lists the
   signatures). `src/Tch.pas`: `TTensor.CopyFrom`.
3. `src/TchNN.pas`, in this order so each part compiles alone: `TInit`
   (4.6.1) → `TVarPath`/`TVarStore` without save/load (4.6.2) → save/load
   (4.6.3) → `TOptimizer` (4.6.4) → `TModule`, `TFuncModule`, `TSequential`
   (4.6.6) → the five layers (4.6.5).
4. `tests/nn/nn_test.pas` with the ten groups of 4.6.7, checking H13–H18
   as they come up; record each result next to the (Hn) in DESIGN.md.
5. Mock: add the stubs of 4.6.7 to `tests/mock/mock_tch.c` and a
   `tests/mock/nn_mock_test.pas` (or extend `mock_test.pas`) for the
   bookkeeping checks.
6. Update DESIGN/HANDOFF (en + jp); then hand JIT (`TchJit`) and CUDA back
   to a design session if needed.

## Handing back to the cloud (2026-10-10, end of local session 2) — answered in cloud session 4 above

**Design only.** Do not write committed code (units, generator, tests);
small throwaway measurements to settle a question are fine (CLAUDE.md).
Implementation comes back to a local session. Mark anything that needs
real libtorch to confirm as a new hypothesis (H13, ...) for the local
session; the cloud container has FPC but no libtorch (the mock
`libtch.so` in `tests/mock` is available).

Milestone 1 is done (see "Local session 2" below). Next is DESIGN 4.6
(optimizers, VarStore/nn). 4.6 fixes the overall shape, but these points
are not settled enough to implement mechanically:

1. **Initialisation (`TInit.Make`).** Exact formulas, following tch-rs
   `nn/init.rs` at 4227b89b: fan-in/fan-out from `Dims` (which dims, conv
   kernels), `KaimingUniform` gain and bound (tch-rs
   `DEFAULT_KAIMING_UNIFORM`), `KaimingNormal`, `Orthogonal` (which
   generated ops: `linalg_qr`? sign fix?), and whether `Make` runs under
   `NoGradGuard`. Which generated ops each one calls (names as in
   `src/TchTensorGenerated_intf.inc`).
2. **`TVarStore` internals.** Container for named variables (ordering
   matters for `Variables`/`TrainableVariables` and for the optimizer's
   lazy add), the trainable flag and group per variable, name validation
   (tch-rs rejects `.` inside a name), behaviour of `Add` on a duplicate
   name, and what `Freeze`/`Unfreeze`/`SetDevice`/`SetKind`/`CopyFrom` do
   to existing handles (in place, under no-grad?). Equality/containers
   open question: `Generics.Collections` vs a plain array.
3. **Save/Load.** C entry points: `at_save_multi(tensors, names, n,
   filename)`, `at_load_multi(tensors, names, n, filename)` (nil array in,
   new tensors out), `at_load_multi_` (copies into existing), and the
   callback forms `at_load_callback[_with_device]` /
   `at_loadz_callback[_with_device]` with
   `void (*f)(void *data, char *name, tensor t)`. Decide which ones
   `Save`/`Load`/`LoadPartial` use, as tch-rs does. **Constraint from
   step 4:** a Pascal exception must never propagate through C++ frames,
   so a Pascal callback must not raise; it records the error and the
   caller raises after the C call returns. The callback declaration
   (`cdecl`, how `data` carries the Pascal object) belongs in the design.
   `.` ↔ `|` in names as tch-rs.
4. **Optimizers.** C signatures: `ato_sgd(lr, momentum, dampening,
   weight_decay, int nesterov)`, `ato_adam/ato_adamw(lr, beta1, beta2,
   weight_decay, eps, bool amsgrad)`, `ato_rms_prop(lr, alpha, eps,
   weight_decay, momentum, int centered)`, `ato_add_parameters(opt, t,
   size_t group)`, `ato_set_learning_rate[_group]`, `ato_set_momentum[_group]`,
   `ato_set_weight_decay[_group]`, `ato_zero_grad`, `ato_step`, `ato_free`
   (all `PROTECT`ed, so `TchCheck` after each). Decide: the
   `PTorchOptimizer` type, how lazy adding of new trainable variables
   works (tch-rs `variables_in_optimizer`), the gradient clipping
   implementation (`ClipGradValue`/`ClipGradNorm` in Pascal over
   generated ops: which ones, under no-grad), and AdamW/RmsProp default
   values (tch-rs's).
5. **Layers for milestone 2.** Which ones besides `TLinear` (tch-rs order
   of popularity: `Conv2D`, `BatchNorm`, `Embedding`, `LayerNorm`,
   `Dropout` as a function, `LSTM`/`GRU`?), each with its config record,
   default init (tch-rs's), and the generated op its `Forward` calls.
   `TLinear` forward: `matmul` + bias or `linear`?
6. **`TSequential` / `TFuncModule`.** `TObjectList` vs
   `specialize TObjectList<TModule>`, `AddFn` with plain function
   pointers vs `of object`, and whether the sequential needs the train
   flag passed down (it does: `Forward(X, Train)` to every child).
7. **Unit layout and tests.** `TchNN` (and maybe `TchOptim`) units, what
   goes where; a milestone 2 acceptance test with hand-checkable numbers
   (e.g. linear regression with SGD converging to known weights, Adam on
   a tiny problem, save → load round trip, `LoadPartial` missing names),
   plus which parts the mock library can cover.

Keep DESIGN/HANDOFF (en + jp) current at the end, as usual. Local state
the design can rely on: everything in "Local session 2" below.

## State (2026-10-10, end of cloud design session 2)

- Design for all three layers written in DESIGN.md. Nothing built yet.
- Reference: tch-rs `4227b89b72059b0651ff83a38637693574e0a2d6`,
  libtorch 2.13.0.
- Decided: fork tch-rs libtch (keep C names, harden `PROTECT`, guard
  `extern "C"`), shared `libtch.so`, `TTensor` = record wrapping a refcounted
  interface, exceptions via `ETorchError`/`ETorchNativeError` after every
  call, nil-handle check before every non-optional tensor argument,
  snake_case generated ops (`&`-escaped where reserved) + CamelCase
  hand-written API, `out` parameters for multi-output ops, generated default
  parameters for trailing simple arguments, `TOptIntList` for
  `IntListOption`, Pascal emitter fed by a manifest from a forked gen.ml.
- License: MIT OR Apache-2.0, same as tch-rs (`LICENSE-MIT`,
  `LICENSE-APACHE` at the repo root). Vendored tch-rs files keep tch-rs's own
  license copies next to them.
- OCaml: `gen/gen.ml` is vendored as source (part of the fork). The OCaml
  compiler and opam packages (`yaml`, `base`, `stdio`, `dune`) are installed
  on the local machine to run it; they are a maintainer tool, needed only
  when regenerating for a new libtorch, never to build tch-fpc.
- Session 2 reviewed section 4 against tch-rs's sources and added 4.6 (nn /
  VarStore / optimizers / JIT as classes + `TVarPath` record, single
  `Forward(X, Train = False)`) and 4.7 (grad mode). Unverified FPC 3.2.2
  behaviour is marked **(H1)**–**(H11)** in DESIGN.md and listed below.

## Cloud session 3 (2026-10-10): parameter-name rule revised

- **Parameter names vs members, revised (H12).** The `_` suffix rule from
  local session 1 is replaced by `{$modeswitch duplicatelocals}` in
  `Tch.pas`; generated parameter and `out` names stay verbatim (`dim`,
  `indices`, `pow`), `NumDims` is `Dim` again, and reserved words use the
  same `&` escape as op names (`&out`, `&end`, `&to`); only `result`
  (3 ops) becomes `result_`. Measured on FPC 3.2.2 in the cloud container:
  `tests/compiler/h12_duplicatelocals.pas` and three `xfail_h12_*.pas`;
  the whole `tests/compiler/run.sh` passes. Details and the compiler-source
  check in DESIGN.md 4.4; README (en/jp) updated.
- Changed code: `src/Tch.pas` (switch, `Dim`, stand-ins use `dim`),
  `tests/ownership/ownership_test.pas` (`V.Dim`). The ownership test needs
  libtorch and was **not** re-run in the cloud; run it locally once
  (`Tch.pas` itself compiles).
## Cloud session 3, part 2: step 8 implemented in the cloud

The generator was designed *and built* here, because the design could be
measured: FPC 3.2.2 was installed in the container, OCaml came from apt
(`yaml` could not: opam's repository is unreachable through the proxy, so
gen.ml ran against a 150-line stand-in `Yaml` module kept in the session's
scratchpad, not committed). Everything in DESIGN.md section 5 is what the
committed code does.

- **gen.ml** writes `gen/ops.tsv` (`gen.exe <yaml> <csrc/libtch> <ops.tsv>`;
  no arguments = upstream behaviour). The regenerated
  `torch_api_generated.{h,cpp}` were byte-identical to the committed ones.
  `gen/ops.tsv` is committed (2,657 ops).
- **gen/tchgen.pas** (Pascal emitter) generates `src/TchSysGenerated.pas`,
  `src/TchTensorGenerated_intf.inc`, `src/TchTensorGenerated_impl.inc`
  (committed). `src/Tch.pas` now includes them, has the full `TKind`
  (`kNone`, 0..45), `TLayout`, `TDevice`, `TScalar`, `TOptInt64`,
  `TOptDouble`, `TOptIntList`, the nested `TTensorArray`, the helpers the
  generated bodies use, and `ETorchNativeError.Detail`; the stand-in ops
  are gone. `Tch.pas` with all ops compiles in 2.7 s.
- **Decisions** (details in DESIGN.md): error message split at
  `Exception raised from` (4.2); `TKind` extended to 45 with `kNone` for
  optional dtypes, no `TOptKind` (4.3); autocast functions in the fork
  patched to the device-taking API, warning handler deferred because c10's
  handler is thread-local (4.2); default-parameter rule and its numbers
  (4.4); results-that-are-inputs ops have no out parameters (4.4);
  `FromArray` overloads without `Int32` (4.3).
- **tests/mock**: a mock `libtch.so` (C, no libtorch) and `mock_test.pas`
  exercising every generated op shape; 45 checks pass (`tests/mock/run.sh`).
  `tests/compiler/run.sh` still passes. `tests/ownership/ownership_test.pas`
  compiles (not linked here; `ones` now takes `TDevice.Cpu`).

## Local session 2 (2026-10-10): steps 1-6 below done

- **1.** libtch rebuilt with the autocast patch; `tests/abi` passes and no
  `[W` warning line is printed any more (the deprecated-API warning on every
  `at_autocast_*` call is gone). `tests/fpu/run.sh`: same results as local
  session 1.
- **2.** gen.exe built with the real opam `yaml`; run against
  `Declarations-v2.13.0.yaml` from tch-rs 4227b89b (sparse clone of that one
  file, outside the repo): `ops.tsv` and `torch_api_generated.{h,cpp}`
  **byte-identical**, so the cloud's stand-in YAML parser typed every
  default the same way.
- **3.** tchgen: no diff (2657 ops; 479 with defaults, 149 with out
  parameters, 97 fill tensors passed in).
- **4.** `tests/ownership` passes against real libtorch (RSS flat over 10^6
  iterations, 1.4 s); `tests/mock/run.sh` passes.
- **5.** New `tests/generated/generated_test.pas`, all passing, against
  hand-worked values (no Python). Measured:
  - `sum_dim_intlist`: `NoIntList` and `IntList([])` both reduce over all
    dims (0-d, 15) for `sum`, as in PyTorch.
  - `pad(..., 'constant', NoDouble)` fills 0; `9.0` converts to `TOptDouble`.
  - `linalg_norm(2, NoIntList)` on a 2-d tensor is the **matrix** 2-norm
    (largest singular value, sqrt 54 here), not the vector norm; my first
    expectation (sqrt 55) was wrong, libtorch follows PyTorch's
    `torch.linalg.norm`. On a 1-d tensor it is the vector norm.
  - `to_sparse(ltNone, ...)` (nullopt layout) gives sparse COO like
    `ltSparse`.
  - **Error split verified** on a real message: `Message` = "mat1 and mat2
    shapes cannot be multiplied (2x3 and 2x3)", `Detail` = 2,217 chars
    starting with `Message` and containing `Exception raised from`.
  - `TchLiveHandles = 0` at the end.

- **6. Done: hand-written API of milestone 1** (`src/Tch.pas`), test
  `tests/api/api_test.pas` (build line in its header), all passing:
  - `Numel`, `Device`, `ShallowClone`, `Copy`; `FromArray` (Single,
    Double, Int64, Byte, Boolean), `FromInt32Array`, `FromBytes`;
    `To{Single,Double,Int64}Array` (convert first), `CopyData` (own kind).
  - Operators: `+ - * /` on tensors, unary `-`, `** Double`, and
    `+ - * /` with `Double`/`Int64` in both orders. `d - A` is
    `rsub_scalar`; `d / A` is `reciprocal * d` because `pow` with integer
    exponent `-1` fails on Int64 tensors (measured; DESIGN 4.1 table).
  - Autograd: `Backward(KeepGraph, CreateGraph)`, `RequiresGrad`,
    `ZeroGrad`; `grad`/`requires_grad_` are the generated ops. Grad mode
    and globals (4.7): `SetGradEnabled`, `NoGradGuard` (bare call has no
    effect, tested), `ManualSeed`, `Set/GetNumThreads`, `CudaIsAvailable`,
    `CudaDeviceCount`, `TDevice.CudaIfAvailable`.
  - Factory overloads `zeros/ones/empty/rand/randn(ASize; AKind = kFloat)`
    on CPU. Measured: no `overload` directive needed in `objfpc`; tchgen
    now treats `overload;` members as deliberate overloads and reads
    multi-line declarations whole (DESIGN 4.4).
  - `KindElementSize`: table checked against libtorch for all 46 kinds
    (DESIGN 4.3).
  - A local named `T` in a `TTensor` method collides with the generated
    `t` op (compile error, the 4.4 pitfall); hand-written locals avoid
    single-letter op names.
  - `tests/mock/mock_tch.c` got stubs for the new hand-written externals.

Next: done in cloud session 4 (see the top of this file); then the local implementation of milestone 2 as listed there.

Local session, in this order (all mechanical):

1. **Build libtch** again: `csrc/libtch/torch_api.cpp` has the autocast
   patch (`is_autocast_enabled(at::kCUDA)` / `set_autocast_enabled`), not
   compiled in the cloud. Then `tests/abi` and `tests/fpu` as before.
2. **Reproduce `gen/ops.tsv`** with the real toolchain:
   `(cd gen && dune build ./gen.exe)` then
   `gen/_build/default/gen.exe <tch-rs>/third_party/pytorch/Declarations-v2.13.0.yaml csrc/libtch gen/ops.tsv`;
   `git diff` should be empty (ops.tsv and the C files). If a default
   string differs, the stand-in YAML parser typed a scalar differently:
   fix `default_text` in gen.ml, rerun tchgen, commit the real output.
3. **Run tchgen** (`fpc -FUbuild/gen -obuild/gen/tchgen gen/tchgen.pas &&
   build/gen/tchgen gen/ops.tsv src`), expect no diff.
4. **Run the ownership test** against real libtorch (build line in its
   header), then `tests/mock/run.sh` (needs only gcc + fpc).
5. **First real-libtorch test of generated ops** (new, small):
   `split`, `max_dim`, `sort_values`, `cat`, `add_scalar(1.5)`,
   `sum_dim_intlist(IntList([]))` vs `NoIntList`, `pad(..., 'constant',
   NoDouble)`, `linalg_norm` (Scalar + IntListOption + defaults), one
   `LayoutOption` op (`to_sparse`), and an error whose `Message` has no
   backtrace while `Detail` has one. Compare with tch-rs/PyTorch results.
6. Then the hand-written API of milestone 1 (DESIGN section 6):
   `FromArray` overloads, `ToDoubles`/`CopyData`, scalar operators
   (`+ 1`, `* 2.5`), `Zeros/Ones` convenience overloads (check whether
   overloading a generated name needs `overload`), autograd.

## Handing back to the cloud (2026-10-10, end of local session 1)

Steps 1-7 are done and every hypothesis H1-H11 is measured (results in
"Local progress" below and next to each (Hn) in DESIGN.md). The OCaml
toolchain for step 8 is installed locally. The user chose to design step 8
in the cloud. Asked of the next cloud session:

1. **Design the generator (step 8)** so that local work can be mechanical:
   - exact `ops.tsv` schema emitted by gen.ml (columns, escaping, how
     defaults, optional/list kinds, receiver and result shape are encoded);
   - `gen/tchgen.pas` structure and the exact Pascal it emits for each op
     shape (fixed-N, dynamic, scalar, nothing), with the stand-ins in
     `src/Tch.pas` (`add`, `cat`, `max_dim`, `ones`) as the reference
     bodies that already work;
   - naming: the parameter/out-name rule (DESIGN 4.4; revised in cloud
     session 3, see above), the `&` escapes, and the op-name check against
     our hand-written members (see the tch-rs notes under "Parameter names
     vs `TTensor` members").
2. **Decide the open questions the generator touches:** native error
   message format (`Message` vs `Detail`), `TKind` 27-45, `TORCH_WARN`
   handling (patch `at_autocast_*` and/or a warning handler), default
   arguments scope, and dropping the `array of Int32` `FromArray`
   overload (H2).
3. Update DESIGN/HANDOFF (en + jp) at the end, as usual.

Local state the design can rely on: `src/TchSys.pas`, `src/Tch.pas`
(minimal `TTensor`, `Dim`, `TchLiveHandles`), tests under `tests/`,
build commands in each test's header and in "Local progress".

## Local progress (2026-10-10)

- **Step 1 done.** FPC 3.2.2, g++ 15.2.0, CMake 4.2.3 (reads libtorch's
  CMake config fine; "kineto not found" warning is harmless), libtorch
  2.13.0+cpu (cxx11 ABI) in `libtorch/`. OCaml not installed yet (step 8).
- **Step 2 done.** `d81f48b` pristine import, `a15806e` fork patch
  (`git diff d81f48b a15806e` shows the patch). Build:
  `cmake -S csrc/libtch -B build/libtch -DCMAKE_PREFIX_PATH=$PWD/libtorch
  -DCMAKE_BUILD_TYPE=Release && cmake --build build/libtch -j 8` (~30 s).
  2657 `atg_*` exported, `ldd -r` clean. gen.ml patched too, so the
  `extern "C"` guard survives regeneration.
- **Step 3 done.** `tests/c/smoke.c` passes (build command in its header).
  Found while doing it: the C branch of `torch_api.h` needed `<stddef.h>`
  (`size_t`) and `<stdbool.h>` (`bool`, before C23); added to the patch.
  An error message is libtorch's text followed by a C++ origin/backtrace
  starting at `Exception raised from ...` (see open questions).

- **Step 4 done.** `tests/abi/abi_test.pas` passes (build command in its
  header; `build/abi_test nilarg` runs the nil-argument case). Measured:
  - C99 `bool` returns: declare as `Boolean`. FPC tests only `%al`
    (`testb %al,%al`), which is all the SysV ABI guarantees. **Do not use
    `LongBool` or ctypes' `cbool`** (a `LongBool`): FPC then tests the whole
    `%eax` (`orl %eax,%eax`). It happened to read correctly here because
    g++ left the upper bits zero, but that is not guaranteed. `Boolean`
    arguments (`at_autocast_set_enabled`) also work.
  - `c10::ScalarType` 0-26 match `TKind` exactly, Float8 23-26 included.
    libtorch 2.13 also has **27-45** (UInt16/32/64, UInt1-7, Int1-7,
    Float8_e8m0fnu, Float4_e2m1fn_x2); `atg_empty` creates all of them on
    CPU and `at_scalar_type` round-trips (see open questions).
  - `uint8_t x_null` as `Byte` and `int8_t` layout as `ShortInt` (incl.
    -1 = nullopt) behave as intended (`atg_argmax`, `atg_to_sparse`).
  - `external 'tch'` + `-Flbuild/libtch -k-rpath=$PWD/build/libtch` finds
    `libtch.so` at run time; libtorch is found through libtch's RUNPATH.
  - nil tensor to a non-optional argument (`atg_add(@S, nil, A)`): FPC's
    SIGSEGV handler turns it into a catchable `EAccessViolation`, raised
    from inside C++ frames, and the process keeps running. That looks
    recoverable but is not safe (no C++ unwinding, libtorch state
    unknown), so the nil-handle check before every call stays necessary.
  - libtorch writes `TORCH_WARN` messages straight to stderr (e.g. the
    deprecated autocast API on every `at_autocast_*` call, quantized and
    ComplexHalf creation). See open questions.

- **Step 5 done.** `tests/fpu/fpu_test.pas` + `tests/fpu/run.sh` (one
  process per mode, 30 s timeout). FPC default MXCSR=$1900, x87CW=$1372;
  after `SetExceptionMask(all)` $1F80/$137F.
  - Unmasked, 4 elements: `EZeroDivide` raised from inside C++ frames.
  - Unmasked, 10^7 elements: unhandled exceptions in the OpenMP workers,
    then a **hang** (and corrupted Pascal state on the main thread).
  - Masked first: 1/0 gives +Inf for both sizes.
  - Pool created unmasked, masked afterwards: still hangs. Confirms that
    the mask belongs in `Tch`'s `initialization`.
  - **(H4) resolved:** a `TThread` started after the mask is masked too;
    small and parallel ops in it give Inf.
  - **(H5) resolved:** the mask survives a caught integer `EDivByZero`.
  - DESIGN.md 4.2 updated accordingly.

- **Step 6 done.** First code: `src/TchSys.pas` (hand-written externals,
  plus stand-ins for the few `atg_*` the tests need) and `src/Tch.pas`
  (`TTensor`/`TTensorHandle`, `TchCheck`, nil-argument check, FPU mask in
  `initialization`, `TchLiveHandles` counter; stand-in ops `add`, `mul`,
  `cat`, `max_dim`, `ones` in the generator's shape). Test:
  `tests/ownership/ownership_test.pas`, all passing.
  - `B := A; A := Default(TTensor)` keeps B alive; `A := A + B` correct.
  - 10^6 x `D := (A + B) * C`: handle count and RSS flat (~1.5 s total).
  - **(H1) resolved:** temporaries live until their slot is reused or the
    routine exits, not until the end of the statement.
  - **(H6) resolved:** `cat([A, B], 0)` and `cat` of temporaries: correct,
    no leak, no double free.
  - **(H8) resolved:** receiver passed as `out` gives `ETorchError`; a
    previous value in an `out` variable is released.
  - New problem found (see open questions): generated parameter names
    collide with `TTensor` members (`dim` vs `Dim`). The stand-ins used
    `dim_` until cloud session 3 (H12).

- **Step 7 done.** `tests/compiler/run.sh` (h*.pas must compile and run,
  xfail_*.pas must be rejected, plus a generated big record). All as
  expected; details recorded in DESIGN.md next to each hypothesis.
  - **(H2)** `[1.0, 2.0]` picks `Single`; `[1, 2]` is ambiguous with both
    `Int32` and `Int64` overloads → drop the `Int32` overload.
  - **(H3)** scalar operators resolve as intended; `**` and unary `-` ok.
  - **(H7)** all escaped names compile and are callable.
  - **(H9)** repeat the default in overrides (else `Derived.Forward(X)`
    does not compile).
  - **(H10)** compiles; `Items` is shared on copy (no copy-on-write).
  - **(H11)** guard local released at routine exit; a discarded
    `NoGradGuard;` is released immediately (pitfall).
  - Record with 2,657 methods via `{$I}` (with a managed field): compiles
    in about 1 s (-O2: 1.6 s); a `default` indexed property gives `T[5][2]`.

- **OCaml toolchain ready (step 8 prerequisite).** `ocaml` 5.4.0 and
  `opam` 2.5.0 from apt (`yaml` is not packaged in Ubuntu, hence opam).
  `opam init --bare --no-setup` (shell rc files untouched), switch
  `tch-fpc` on `ocaml-system`, `opam install dune base stdio yaml`. Use
  `eval $(opam env --switch=tch-fpc)` before building; then
  `cd gen && dune build ./gen.exe` builds warning-free. Run from a tch-rs
  checkout at 4227b89b (it reads
  `third_party/pytorch/Declarations-v2.13.0.yaml` and writes upstream
  paths), the patched gen.exe reproduces `csrc/libtch/torch_api_generated.{h,cpp}`
  byte for byte (2657 functions, 0.5 s).

## Next steps (local implementation)

1. **Toolchain check.** FPC 3.2.2, g++, CMake, libtorch 2.13.0 CPU
   (cxx11 ABI) unpacked into `libtorch/` (already git-ignored).
2. **Vendor libtch.** Copy from tch-rs at the commit above into
   `csrc/libtch/` and `gen/`; add LICENSE-MIT / LICENSE-APACHE. Commit the
   pristine copy first (message names the source files), then a second
   commit with our patch: remove stream/python/image code, add `tchf_free`,
   add `catch (...)` to `PROTECT`, wrap `extern "C"` in
   `torch_api_generated.h` with `#ifdef __cplusplus`. Build `libtch.so` with
   CMake. The YAML (`third_party/pytorch/Declarations-v2.13.0.yaml`) is in
   the tch-rs repo at that commit but may be missing from a shallow/partial
   clone; fetch it only when regenerating.
3. **Smoke test in C** (tiny C program, not Python): create a tensor,
   `atg_add`, print, trigger an error and read `get_and_reset_last_err`.
   Needs the `extern "C"` guard from step 2 to include the generated header.
4. **ABI verification in Pascal** — measure, do not assume:
   - `bool` returns (e.g. `at_autocast_is_enabled`) read correctly as
     `Boolean`.
   - `c10::ScalarType` numeric values match `TKind`, including the Float8
     kinds 23–26 (print `at_scalar_type` for tensors of each kind).
   - `uint8_t x_null` option flags as `Byte`, `int8_t` as `ShortInt`.
   - Linking: `external 'tch'` + rpath so the program finds `libtch.so` and
     libtorch at run time.
   - Passing a `nullptr` tensor to a non-optional argument really is a
     segfault (confirms the need for the nil-handle check).
5. **FPU mask tests.**
   - Without `SetExceptionMask`, does `1.0 / 0` on a CPU tensor crash an FPC
     program? Confirm the fix in `Tch` initialization.
   - **(H4)** After `SetExceptionMask` in the main thread, does a `TThread`
     started afterwards see the mask, or the FPC defaults? Decides whether
     `TchMaskFpuExceptions` must be called in every thread.
   - **(H5)** Raise and catch an `EZeroDivide` in Pascal code, then run a
     tensor op that produces NaN. Does the mask survive the RTL's SIGFPE
     handling?
6. **Ownership tests.** Loop 10^6 times over `(A + B) * C` and check RSS is
   flat; check `B := A` then `A := Default(TTensor)` keeps B alive.
   - **(H1)** Instrument `TTensorHandle.Destroy` and determine when the
     temporaries of `(A + B) * C` are released: end of statement, end of
     loop iteration, or end of routine.
   - **(H6)** `TTensor.cat([A, B], 0)` with an open-array constructor of
     `TTensor`: no leak, no double free.
   - **(H8)** `out` parameter semantics: `T.max_dim(1, False, T, I)` raises
     `ETorchError` (receiver finalised on entry) instead of crashing, and
     `V` passed as `out` with a previous value releases the old handle.
7. **Compiler acceptance tests** (tiny programs before the generator exists):
   - **(H2)** `FromArray([1.0, 2.0])` with overloads on `array of Single`
     and `array of Double` compiles unambiguously; same for `[1, 2]` with
     `Int32`/`Int64`. If not, use `FromSingles`/`FromDoubles`/...
   - **(H3)** Operator overloads `+(TTensor, Double)` and `+(TTensor, Int64)`
     coexist and `T + 1`, `T + 1.5`, `2 * T` all resolve; `**` can be
     overloaded for records; unary `-` too.
   - **(H7)** Record methods named `&div`, `&var`, `&to`, `&set`, `&repeat`,
     and modifiers `absolute`, `alias`, `index` (with and without `&`)
     compile and can be called as `t.&div(u)`.
   - **(H9)** `Forward(const X: TTensor; Train: Boolean = False)` as a
     virtual abstract method with the default repeated in overrides; calling
     `Model.Forward(X)` through the base and derived type.
   - **(H10)** `TIValue = record ... Items: array of TIValue; end;` compiles.
   - **(H11)** An `IInterface` local assigned from `NoGradGuard` is released
     at routine exit (and not earlier).
   - A record with ~2,650 methods pulled in via `{$I}` inside the record
     declaration compiles in reasonable time; `{$modeswitch advancedrecords}`
     properties on records if the `T[i]` sugar is wanted.
   - **(H12, resolved in cloud session 3)** `{$modeswitch duplicatelocals}`
     lets a method parameter share a member's name in `objfpc`; `&out` as
     a parameter name compiles; `result` as a function parameter and a
     local variable named like a member are still rejected.
8. **Generator. Done in cloud session 3** (see the top of this file):
   gen.ml writes `ops.tsv`, `gen/tchgen.pas` writes `TchSysGenerated.pas`
   and the `.inc` files, everything compiles, and `tests/mock` runs the
   generated layer. Local: reproduce `ops.tsv` with the real `yaml`
   library, run the ownership test, then the first real-libtorch test of
   generated ops.

## Open questions

- **Native error message format: decided** (cloud session 3). `Message`
  = text before `#10'Exception raised from '`, `Detail` = full text
  (DESIGN.md 4.2, implemented in `ETorchNativeError.Create`). Verified
  against a real libtorch message in local session 2.
- **`TKind` beyond 26: decided** (cloud session 3). Extended to all 46
  values of 2.13 plus `kNone = -1` for optional dtypes (DESIGN.md 4.3).
  Element sizes resolved in local session 2: `FromBytes`/`CopyData`
  accept every kind, sizes checked against libtorch (DESIGN.md 4.3).
- **libtorch warnings: decided** (cloud session 3). Fork patch of the two
  autocast functions only (compiled and tested in local session 2); c10's warning
  handler is thread-local, so a Pascal callback is deferred until needed
  (DESIGN.md 4.2).
- **Parameter names vs `TTensor` members: decided, then revised**
  (2026-10-10, user). First decision (local session 1): append `_` until
  unique, rename `Dim` → `NumDims`. Revised in cloud session 3 (H12):
  `{$modeswitch duplicatelocals}` in `Tch.pas` removes the collision, names
  stay verbatim, `Dim` is back, reserved words get `&`, `result` →
  `result_`. Rule, measurements and the compiler-source check in DESIGN.md
  4.4.
  Why tch-rs never needed this: in Rust, parameters are plain locals and
  methods are reachable only as `self.dim()` / `Tensor::dim`, so
  `fn dim(&self)` and ~250 ops with `dim: i64` coexist. (`duplicatelocals`
  is the part of `{$mode delphi}` that behaves like Rust here.) What
  tch-rs *does* handle, and step 8 must
  check against our own names: op-name collisions in gen.ml's
  `replace_map` (`where`/`unsafe` for Rust keywords; `view`, `to_device`,
  `t` → `tr` for hand-written methods) and `prefixed_functions` (`g_`
  prefix for `add`, `mul`, `div`, `sub`, ... whose plain names go to
  hand-written versions). These only rename the Rust wrappers; the
  `atg_*` names our generator reads are unaffected. Also in README.
- **Multi-output ops**: `out` parameters (decided; 149 ops), and no out
  parameters for the 97 ops whose results are the tensors passed in
  (cloud session 3, DESIGN.md 4.4). Still open: whether a few hand-written
  result records (`max_dim`, `sort`, `topk`, `lstm`) are worth adding
  after writing real code, and `var` instead of `out` if (H8) bites (H8
  resolved since: `out` kept, DESIGN.md 4.4).
- **Default arguments: decided** (cloud session 3, DESIGN.md 4.4): 479 ops
  get defaults; ops with out parameters none (107 would have had some).
  Local session 2: `zeros`, `ones`, `empty`, `rand`, `randn` overloads
  added; no `overload` directive is needed in `objfpc` (measured), but
  tchgen uses `overload;` as the marker for deliberate overloads
  (DESIGN.md 4.4). Still open: `topk(k)`-style defaults for out-param ops.
- **`FromArray` with real literals: decided** (cloud session 5, H2
  addendum in DESIGN 4.3). The first element decides the element type and
  a real literal is `Single` only when exactly representable, so `[0.9, …]`
  was ambiguous and `[1.0, 0.1]` rounded to `Single`. The `Single` overload
  is dropped, the others take `AKind` with the natural default (`kFloat`
  for `Double`), and `FromSingleArray` covers `Single` buffers. Implemented
  in local session 4.
- **Equality and containers: decided** (cloud session 4). `=` stays
  un-overloaded for `TTensor`. `TchNN` needs only an append-only
  `array of TVarEntry`, a `TDictionary<String, Integer>` name index and
  `TObjectList<TModule>`; all measured on 3.2.2 (DESIGN.md 4.6.2, 4.6.6).
  `TList<TTensor>` also works (byte comparer = handle identity) if ever
  needed.
- **Train flag: decided** (cloud session 4). `Forward(X, Train)` stays;
  `TSequential` passes `Train` to every child and a user `TModule` passes it
  on explicitly (DESIGN.md 4.6.6). No `Training` property.
- **`TIValue` representation.** Tagged record: decided, and H10 confirmed
  it compiles (step 7). Copies share `Items`; treat as immutable.
- **VarStore thread safety.** Documented as not thread-safe; add a
  `TCriticalSection` if a data-loader thread ever needs it.
- **Streams / images.** If needed later: replace tch-rs's Rust-implemented
  stream symbols with a struct of C function pointers passed in by Pascal
  (wrapping `TStream`). Images can be done in Pascal (FPImage) instead of stb.
- **Windows / macOS.** Not designed yet beyond `cdecl` and `tchf_free`.
  Windows additionally needs symbol export from `tch.dll`
  (`--export-all-symbols` or a `.def` file).
- **ROCm (deferred; CPU first) — done in local session 8** (DESIGN.md 4.8
  "Results"; the notes below are the original plan). Steps 2-7 use the CPU libtorch. The local
  GPU is an RX 9070 (Navi 48, gfx1201) plus a Raphael iGPU; no system ROCm.
  libtorch 2.13.0 for ROCm exists only as `+rocm7.1` (~5.8 GB, checked
  2026-10-10). Expected, unmeasured: the ROCm build presents HIP as CUDA
  (`cuda` device, `torch::cuda::is_available`), so the C API, `TDevice` and
  `CudaIsAvailable` stay unchanged. To check when added as `libtorch-rocm/`:
  link `torch_hip` with `-Wl,--no-as-needed` (libtch never references it
  directly); gfx1201 kernels present in the zip; iGPU enumeration
  (`HIP_VISIBLE_DEVICES`); whether the bundled deps suffice without system
  ROCm. DESIGN.md section 1 lists only `libtorch_cuda.so`.
- **Release packaging.** Generated files (`ops.tsv`, `torch_api_generated.*`,
  `TchSysGenerated.pas`, `.inc`) are committed, so every tag's source archive
  builds without OCaml. Open: whether releases also attach prebuilt
  `libtch.so`/`tch.dll` per libtorch version and platform, or users always
  build it with CMake. libtorch itself is never bundled.
- **Going public.** The license is decided (above, "State" section); the
  repo stays private until the user decides to publish it.
