# Evaluation tasks v1 (qwen-lora, DESIGN 8.17)

Held-out tasks for the "before" (M3) and "after" (M5) evaluations. Written
by Claude (Opus), never training data: the data set's leak check
(`qlora_dataset --eval eval/tasks.jsonl`) rejects any record close to a
prompt or a reference here.

| Part | Count |
|---|---|
| Pascal, per dialect (`objfpc`, `fpc-delphi`, `unleashed`, `delphi`, `oxygene`): 30 `write`, 5 `fix` (the prompt shows a program that does not compile), 5 `convert` (the prompt shows a program of another dialect) | 200 |
| regression, `contains` (plain questions, shell, other languages, two in Japanese): the reply must contain a literal text | 20 |
| regression, `tool-call` (weather, currency, time, web search; three in Japanese): the reply must call the named tool | 10 |

Files:
- `src/<dialect>.txt`: the Pascal tasks (`=== id type`, `--- prompt`,
  `--- given`, `--- ref`); `src/regression.tsv`: id, check, tool or text,
  prompt.
- `refs/<id>.pas` and `refs/<id>.expected`: each reference solution and
  its output. Every reference is accepted by `qlora_verify` (Delphi as the
  FPC approximation, Oxygene with EBuild and dotnet, T22); every `fix`
  task's given program fails to compile and every `convert` task's given
  program compiles and runs in its own dialect (`make-eval.bb givens`).
- `tasks.jsonl`: what `ask.bb` and `qlora_eval` read: `{id, dialect,
  type, prompt, expected_stdout, reference}` per Pascal task (the prompt
  ends with the dialect and "Reply with the complete program in one fenced
  code block."), `{id, dialect: "regression", type, prompt, check, tool |
  text}` per regression task.
- `make-eval.bb`: `split`, `fill` (compile and run the references,
  `.expected.new` to check by eye), `check V`, `givens V`, `build`.
- `ask.bb`: records an Ollama model's replies (`/api/chat`, two seeds,
  thinking on and off; the tools `get_weather`, `convert_currency`,
  `get_time`, `search_web` on tool-call tasks).

Checked 2026-11-25: the references as replies ("perfect replies") pass
460 of 460 judgements of `qlora_eval` (both thinking modes), 2 min 21 s.

Usage (the baseline, M3; GPUs used by Ollama, nothing else running on
them):

```bash
bb eval/ask.bb qwen3.8-27b-128k eval/tasks.jsonl work/eval/before.jsonl
```

then `qlora_eval eval/tasks.jsonl work/eval/before.jsonl work/eval/before`.
