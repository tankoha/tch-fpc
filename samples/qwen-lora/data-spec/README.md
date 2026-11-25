# Data specification v3 (qwen-lora)

Version: **spec-v4** (2026-11-26). Written by Claude (Opus) for the
harness; read by the generator, Mistral Large 3 (DESIGN 8.16).

| Version | Change |
|---|---|
| spec-v1 | first version; 100-job pilot: 43 % accepted |
| spec-v2 | after the first pilot: "common mistakes" per dialect in dialects.md (each checked with the dialect's compiler), rules for `expected_stdout` and the repair with the actual output in spec.md; second pilot: 44 % |
| spec-v3 | after the second pilot (44 %): `override` in Delphi records added to the common mistakes; the driver now shows the source lines around a compile error and says when an error repeats; third pilot: 62 % (Mistral), 42 % (local Qwen3.8 27B, same jobs) |
| spec-v4 | after the Qwen pilot: few-shot examples per dialect (`fewshot.jsonl`, accepted replies of the generator from pilots 2 and 3), put before each job as earlier exchanges |

The specification is instructions and lists. It contains **no example
programs written by Claude**: nothing in the `pascal-dialects` or
`tch-fpc` skills is handed to the generator. The only examples are in
`fewshot.jsonl`: replies that the generator (Mistral Large 3) wrote
itself in pilots 2 and 3 and that the verifier accepted, chosen by the
harness owner and not edited (DESIGN 8.16).

| File | Content |
|---|---|
| [spec.md](spec.md) | what a sample is, the record format, task types and quotas, prompt language, prohibitions, the generation protocol |
| [dialects.md](dialects.md) | per dialect: how a source declares it, the allowed libraries, the feature list with minimum counts |
| [topics.md](topics.md) | topics and the length distribution |
| [fewshot.jsonl](fewshot.jsonl) | per dialect two or three example exchanges (job line, accepted reply), each with its source run and record; made by `gen.bb fewshot` |

How it is used:
1. The driver `gen.bb` (harness owner, M3) sends the generator one job
   at a time: the three Markdown files above as the system
   text, the example exchanges of the job's dialect from
   `fewshot.jsonl`, and a job line (dialect, task type, topic, features to use,
   prompt language, job id).
2. The generator answers with one record (spec.md, "Record").
3. The driver compiles and runs the record with `qlora_verify` (T06,
   T22) and returns the compiler's or the run's message for at most
   three repairs (spec.md, "Protocol").
4. `qlora_dataset` (T19) decides independently what is accepted: the
   generator's own claim that a sample compiles is never trusted.

What changes the version: any edit of these files. The manifest records
`spec-v4` for every accepted record (the pilots of spec-v1 to spec-v3 are not part of the data set; the jobs of the few-shot examples' source records are left out of the pilot comparisons).
