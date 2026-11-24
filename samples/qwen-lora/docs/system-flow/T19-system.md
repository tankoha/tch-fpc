# T19 System Flow: qlora_dataset

## Components Used

| Component | Purpose |
|---|---|
| `DialectVerify` | Detect dialect, verify programs, `VerifyFile` |
| `QwenTokenizer` | Encode text to token IDs for tokenizer check |
| `GgufFile` | Open GGUF model for tokenizer |

## Who Runs It

### Generator (M3)
- Processes raw output from the generator pipeline
- Filters out invalid, duplicate, and leaked records
- Creates training-ready JSONL files

### Harness Owner
- Unit tests with stock FPC and tiny tokenizer (T19b)
- ITa with rehearsal model (T19c)

### User
- Preparing final training data from generated records

## Inputs

### Primary Input
- Raw JSONL from generator (one record per line)
- Each record: `id`, `dialect`, `task_type`, `topic`, `language`, `prompt`, `answer`, `expected_stdout`, `features`, `generator`

### Evaluation Tasks (optional, `--eval`)
- JSONL with `id`, `prompt`, `reference` fields
- Checked for both prompt leaks and reference/source leaks

### Example Directories (optional, `--leak-dir`)
- Pascal source files (`*.pas`, `*.dpr`) checked recusively
- Compared against record sources for near-duplicate leaks

### Tokenizer Reference (optional, `--model` + `--ref-tokens`)
- GGUF model file for `QwenTokenizer`
- Reference tokens: JSONL from bundled `llama-server` `/tokenize` endpoint
- Made by `tests/ita/T19/make-ref.bb`

## Outputs

### Training Data
- `train.jsonl`: Raw object plus `class` field, for `TrainData` (T15)
- `valid.jsonl`: Same format for validation

### Provenance
- `manifest.jsonl`: Per-record metadata (MD5, split, class, spec, generator)
- `rejects.jsonl`: Line, id, reason, detail for every rejected record

### Summary
- `card.md`: Totals, yield, dialect × task type table, reject reasons

## Downstream Consumers

### TrainData (T15)
- Reads `train.jsonl` and `valid.jsonl` by class
- Loads records into training batches

### Evaluation (M3)
- The evaluation tasks themselves come from a separate source
- `qlora_dataset` checks for leaks against them

## Unit Test Environment

### Test Fixture
- `tests/ut/DatasetFilter_test.pas`: helpers for `DatasetFilter`
- `tests/ut/DatasetCli_test.pas`: helpers for `DatasetCli`

### Tiny Model
- Tiny tokenizer from T05 (`TinyPath` from `QwenBlocksFixture`)
- GPUs hidden with `HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1`

### Test Cases
- `DatasetFilter`: rows UT-DatasetFilter-001 to 010 (T19b1)
- `DatasetCli`: rows UT-DatasetCli-101 to 109 (T19b2)

Build and run:
```
fpc -Mobjfpc -Sh -vew -l- -Fu<src> -Fu<tests/ut> -FU<work/bin> -Fl<libtch> -k-rpath=<libtch> tests/ut/<Unit>_test.pas
HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 work/bin/<Unit>_test --all --format=plain
```

## ITa Environment

### Driver
- `tests/ita/T19-run.sh`
- Builds `src/qlora_dataset.pas`
- Runs twice from fresh directories outside the project

### Test Data
- `tests/ita/T19/raw.jsonl`: 16 good records in four dialects + 17 planted faults
- Made by `make-raw.bb`
- `eval.jsonl`: evaluation tasks
- `examples/`: example Pascal files
- `ref-tokens.jsonl`: tokenizer reference (one record altered)
- Made by `make-ref.bb` from bundled `llama-server`

### Command
```
QLORA_MODEL=<rehearsal GGUF> <project root>/tests/ita/T19-run.sh
```

### Result File
- `<project root>/work/ita/T19/summary.tsv`
- Columns: case, OK/NG, value, limit, note
- OK when value ≤ limit
- Under a minute

### Validation
- `tests/ita/T19/check.bb` recomputes references
- Compares outputs against expected outcomes

## ITa Cases

| Case ID | Title | Expected (short) |
|---|---|---|
| ITa-T19-001 | Record outcomes vs expected.tsv | 0 mismatches |
| ITa-T19-002 | Valid records vs split recomputed | 0 mismatches |
| ITa-T19-003 | train/valid format, raw order | 0 mismatches |
| ITa-T19-004 | manifest: MD5, split, class | 0 mismatches |
| ITa-T19-005 | card totals vs rejects.jsonl | 0 mismatches |
| ITa-T19-006 | Second run: files byte-identical | 0 files differ |

## Acceptance

From DESIGN 8.18: "on a synthetic raw set with planted duplicates, leaks and failures, exactly the planted ones are rejected"

Measured with prototype:
- All ITa cases: 0 (limit 0)
- 16 of 33 records accepted, 17 rejected as planted
- Approximately 3 seconds per run
