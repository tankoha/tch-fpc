# T19 External Design: qlora_dataset

## Purpose

`qlora_dataset` prepares a training data set from raw generator output. It
filters, deduplicates, checks for leaks, verifies, and splits records for
training and validation. It is the "Dedup and leaks", "Provenance", and
"Quotas" step of DESIGN 8.16–8.18.

## Command Line

```
qlora_dataset [options] RAW OUTDIR
```

Arguments:
- `RAW`: path to the raw JSONL file (one record per line)
- `OUTDIR`: directory for the output files (created if missing)

Options (in order of parsing, all optional):
- `--model F`: GGUF model file for tokenizer check
- `--ref-tokens F`: reference token file (JSONL from llama-server)
  - Both `--model` and `--ref-tokens` must be given together, or neither
- `--eval F`: evaluation task file (repeatable, JSONL with `id`, `prompt`, `reference`)
- `--leak-dir D`: example directory to check for leaks (repeatable)
- `--seed N`: random seed for the split (default: 1)
- `--valid-percent P`: validation percentage 0–100 (default: 5)
- `--timeout S`: verification timeout per record in seconds, 1–3600 (default: 10)
- `--spec-version V`: specification version string (default: `v1`)

Defaults:
- Seed: 1
- ValidPercent: 5
- Timeout: 10
- SpecVersion: `v1`
- Model, RefTokens, Evals, LeakDirs: empty

## Record Schema

Each raw record is a JSON object with these fields (checked in order):

1. `id` (string, non-empty): unique identifier
2. `dialect` (string, one of): `objfpc`, `fpc-delphi`, `unleashed`, `delphi`, `oxygene`
3. `task_type` (string, one of): `write`, `extend`, `fix`, `convert`, `unit`
4. `topic` (string): topic description
5. `language` (string): programming language
6. `prompt` (string, non-empty): the instruction
7. `answer` (string, non-empty): the response, must contain at least one fenced code block
8. `expected_stdout` (string): expected output for the program
9. `features` (array): feature tags
10. `generator` (object): generation metadata

Block extraction from `answer`:
- Fenced blocks are code sections delimited by ``` lines
- The program block is the first block that does not start with `unit ` (case-insensitive after normalization)
- Other blocks are units, saved as `<unit name>.pas`

## Checks Per Record

Checks are applied in order; the first failing check is the reject reason:

1. **Schema**: valid JSON object with all required fields
2. **Program block**: at least one fenced block, and one is a program (not a unit)
3. **Duplicate ID**: `id` must be unique among accepted records
4. **Dialect match**: `DetectDialect` on the program block must equal the record's `dialect`
5. **Leak against evaluation tasks**: source shingles vs reference shingles (Jaccard ≥ 0.5), prompt shingles vs prompt shingles
6. **Leak against example files**: source shingles vs `*.pas`/`*.dpr` in `--leak-dir` directories (recursive)
7. **Exact duplicate**: `SourceKey` (MD5 of normalized blocks) matches an earlier accepted record
8. **Near duplicate**: Jaccard of 5-gram shingles ≥ 0.5 with an earlier accepted record
9. **Tokenizer**: if `--model` and `--ref-tokens` given, token IDs must match the reference
10. **Verification**: program compiles/runs, outcome `accepted` with class `compiled-run` or `fpc-approx`, no `expect-error:` marker

A duplicate (exact or near) keeps the earlier record; the later one is rejected.

## Deduplication Details

### Exact Duplicates
- `SourceKey`: MD5 hash of normalized blocks joined by newline
- Normalization: remove comments (braces, `(* *)`, `//`), lower case outside strings, collapse whitespace to single space

### Near Duplicates
- Tokenization: words (`[a-z_][a-z0-9_]*`), numbers (`[0-9]+`), other characters (not whitespace alone)
- Shingles: FNV-1a 64-bit hashes of 5-grams (joined by space), sorted unique
- Jaccard: `|A ∩ B| / |A ∪ B|`, threshold 0.5 or higher

## Tokenizer Check

Requires both `--model` and `--ref-tokens`:
- Reference file: one JSON object per line with `id`, `prompt` (token array), `answer` (token array)
- Made by bundled `llama-server` `/tokenize` endpoint (special tokens parsed, no BOS)
- Compare: `QwenTokenizer.Encode` of prompt and answer vs reference token IDs
- Missing `id` or token mismatch → reject
- First line of an `id` is used; later lines ignored

Without both options, the check is skipped (recorded in card).

## Verification

A record is accepted when:
- Verifier outcome: `accepted`
- Class: `compiled-run` or `fpc-approx`
- Program block has no `expect-error:` marker

Oxygene records: class stays `static-only` (rejected with `verify: static-ok`)

## Split

Records are split into train and validation:
1. Group by dialect × task type
2. Within each group, order by `SplitKey` (MD5 of `<seed>:<id>`)
3. First `ValidCount` records go to valid, rest to train
4. `ValidCount = (n * percent + 50) div 100` (rounded half-up)
5. Both outputs maintain raw order of accepted records

## Output Files

All files written to `OUTDIR`:

1. `train.jsonl`: raw JSON object plus `class` field
2. `valid.jsonl`: same format, validation records
3. `manifest.jsonl`: per-record metadata
   - Fields: `id`, `record_md5`, `source_md5`, `class`, `split`, `spec`, `generator`
4. `rejects.jsonl`: rejected records
   - Fields: `line` (1-based), `id`, `reason`, `detail`
5. `card.md`: summary statistics
   - Totals, yield percentage
   - Table: dialect × task type × class with train/valid counts
   - Reject reasons up to colon with counts

## Exit Codes

| Code | Meaning |
|---|---|
| 0 | Success: records processed, files written |
| 1 | Runtime error: file not found, GGUF error, tokenizer error, etc. |
| 2 | Usage error: wrong arguments, unknown option, missing value |

## Error Conditions

### Usage Errors (exit 2)

From `ParseDatasetArgs`:
- `<option>: value missing`
- `--seed: an integer, got "<value>"`
- `--valid-percent: an integer in 0..100, got "<value>"`
- `--timeout: an integer in 1..3600, got "<value>"`
- `unknown option <arg>`
- `expected RAW OUTDIR, got <n> arguments`
- `--model and --ref-tokens go together`

### Runtime Errors (exit 1)

From `RunDatasetCLI`:
- `no raw file <path>`
- `no file <path>` (evaluation or reference file)
- `no leak directory <path>`
- GGUF errors
- Tokenizer errors

### Reject Reasons

Written to `rejects.jsonl` in check order:
1. `schema: not a JSON object`
2. `schema: id`, `schema: dialect`, `schema: task_type`, `schema: topic`, `schema: language`, `schema: prompt`, `schema: answer`, `schema: expected_stdout`, `schema: features`, `schema: generator`
3. `schema: answer has no fenced block`
4. `schema: answer has no program block`
5. `duplicate-id`
6. `dialect: <detected>`
7. `leak-eval: <task id>`
8. `leak-example: <file path>`
9. `duplicate: <earlier id>`
10. `near-duplicate: <earlier id>`
11. `tokenizer: no reference`
12. `tokenizer`
13. `verify: <outcome>`

## Success Output

On success (exit 0), stdout contains:
```
<a> of <n> records accepted (train <t>, valid <v>) into <outdir>
```

## Units Used

- `DialectVerify` (T06): `DetectDialect`, `VerifyFile`, `OUT_ACCEPTED`, `VS_COMPILED_RUN`, `VS_FPC_APPROX`
- `QwenTokenizer` (T05): `TQwenTokenizer`, `Encode`
- `GgufFile` (T01): `TGgufFile`, opens the GGUF model

## Environment

The program reads compiler paths from environment variables (as `qlora_verify`):
- `QLORA_FPC`, `QLORA_FPC_UNLEASHED`, `HOME`
