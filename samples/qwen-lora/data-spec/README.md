# Data specification v1 (qwen-lora)

Version: **spec-v1** (2026-11-25). Written by Claude (Opus) for the
harness; read by the generator, Mistral Large 3 (DESIGN 8.16).

The specification is instructions and lists. It contains **no example
programs**: the generator writes every sample itself, and nothing here
or in the `pascal-dialects` skills is handed to it.

| File | Content |
|---|---|
| [spec.md](spec.md) | what a sample is, the record format, task types and quotas, prompt language, prohibitions, the generation protocol |
| [dialects.md](dialects.md) | per dialect: how a source declares it, the allowed libraries, the feature list with minimum counts |
| [topics.md](topics.md) | topics and the length distribution |

How it is used:
1. The driver `gen.bb` (harness owner, M3) sends the generator one job
   at a time: the three files above as the system
   text, and a job line (dialect, task type, topic, features to use,
   prompt language, job id).
2. The generator answers with one record (spec.md, "Record").
3. The driver compiles and runs the record with `qlora_verify` (T06,
   T22) and returns the compiler's or the run's message for at most
   three repairs (spec.md, "Protocol").
4. `qlora_dataset` (T19) decides independently what is accepted: the
   generator's own claim that a sample compiles is never trusted.

What changes the version: any edit of these files. The manifest records
`spec-v1` for every accepted record.
