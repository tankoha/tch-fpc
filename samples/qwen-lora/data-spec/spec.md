# Generator specification (spec-v1)

You write training samples for a model that learns to program in five
Pascal dialects. Each sample is a task (the prompt a user might write) and
its answer (a short explanation and the source). The samples are
compiled, run and checked by a separate tool; only samples that compile,
run and print exactly the expected output are kept.

## 1. What a sample is

- **One self-contained console program**, or (task type `unit`) one unit
  plus a test program that uses it. Oxygene samples are always **one
  file** (a namespace file; for `unit` tasks the class and its test
  program are in the same file).
- **Deterministic output**: the program prints the same text on every
  run. No reading from standard input, no command-line arguments, no
  network, no files outside the working directory (prefer none), no
  GUI, no clock or date of today, no random numbers without a fixed seed
  that is printed nowhere, no environment variables, no threads whose
  order shows in the output.
- It ends within **5 seconds** and exits with code 0 (the program
  returns normally; an Oxygene `Main` returns 0 or nothing).
- The output is plain text written with `WriteLn` / `Write` (Oxygene:
  `writeLn` / `write`), at most 100 lines, ASCII unless the task is about
  text processing. Every line ends with a line break; floating-point
  values are printed with a fixed format so the output does not depend on
  the platform's default.
- Source comments and identifiers are in English, whatever the prompt's
  language.

## 2. Reply format and record

Reply with exactly these four sections and nothing else (each heading on a
line of its own, in this order):

````
### prompt
<the task as a user would write it (section 4)>
### answer
<one or two sentences, then the source: exactly one fenced block per
file, ```pascal … ```; for `unit` tasks the unit's block first, the
program's block second (Oxygene: one block)>
### expected_stdout
```text
<the exact standard output of the program, every line as printed>
```
### features
<the feature ids (dialects.md) the source really uses, comma-separated;
at least the ones the job asks for>
````

The harness builds the record from the job and your sections:

| Field | Content |
|---|---|
| `id`, `dialect`, `task_type`, `topic`, `language` | from the job |
| `prompt`, `answer`, `features` | your sections |
| `expected_stdout` | the text inside the ```` ```text ```` fence, each line ending with `\n` |
| `generator` | filled by the harness: `model`, `version`, `date`, `prompt_id` (the job id), `temperature`, `repair_attempts` |

A `unit` block starts with `unit <Name>;` and is saved as `<Name>.pas`
next to the program; the program uses it by that name.

## 3. Task types and quotas

Per dialect, of the accepted samples:

| `task_type` | Share | The prompt asks to … |
|---|---|---|
| `write` | 50 % | write a program from a description of what it must do and print |
| `extend` | 15 % | change or extend a given program (the prompt contains it, at most 60 lines) |
| `fix` | 15 % | fix a program that does not compile (built by the harness from your own failed attempts, section 8) |
| `convert` | 10 % | convert a given program from another of the five dialects (the prompt contains it) |
| `unit` | 10 % | write a unit (Oxygene: a class) and a small test program that prints its results |

The data set v1 keeps 1000 accepted samples per dialect; 5 % of each
dialect × task type goes to validation.

## 4. Prompts

- Language: **English 90 %, Japanese 10 %** (the job says which). A
  Japanese prompt is natural Japanese as a Japanese programmer writes it;
  the source stays in English.
- The prompt names the dialect the way a user does ("in Free Pascal's
  objfpc mode", "Delphi", "Unleashed Pascal", "Oxygene"), states what the
  program must print, and gives any input data inside the prompt. Vary
  the wording and the length (one sentence to a short paragraph); do not
  start every prompt the same way.
- `extend` and `convert` prompts contain the given program in a fenced
  block; it must be a correct program of the stated dialect.

## 5. Answers

- One or two sentences on the approach, then the source. No further
  explanation after the source.
- Idiomatic code of the dialect, using the features the job asks for in
  a way that is natural for the task (not forced in as decoration).
- Lengths: section "Length" of topics.md.

## 6. Prohibitions

- No verbatim or near-verbatim reproduction of existing code (books,
  documentation, repositories, well-known examples); write it yourself.
- No personal data: no real names of private persons, addresses, e-mail
  addresses, telephone numbers, user names or host names; invented data
  is generic (`Alice`, `Item 3`, `example.com`).
- No dialect mixing: nothing from another dialect, nothing outside the
  dialect's feature list and allowed libraries (dialects.md).
- No code that needs a library, package or file not named in
  dialects.md; no inline assembler; no platform calls.
- No compile-error markers (`expect-error:`) in a sample.

## 7. Repairs

When the harness returns a compiler or run message, answer with the
whole corrected reply in the same four sections. At most
three repairs; then the job is dropped.

## 8. Protocol (the harness side)

- One job per request; the job line gives `id`, `dialect`, `task_type`,
  `topic`, `features` (one to three ids to use), `language`.
- Temperature 0.7, a fixed seed per job; the harness records the model's
  version and the date.
- The harness compiles and runs every record with its verifier (the
  compiler of the dialect, a fresh directory, 10 s limit) and compares the
  output with `expected_stdout`.
- **`fix` records** are made by the harness from a `write` or `extend`
  job whose first answer did not compile and whose repair did: the
  prompt asks to fix the broken source (given in a fenced block with the
  compiler's message, reproduced by the verifier), the answer is the
  repaired source. Such records carry two more fields:
  `broken_source` and `compiler_message`.
- Rejected records and the raw output stay outside the repository; the
  accepted data set, its manifest and card are published.
