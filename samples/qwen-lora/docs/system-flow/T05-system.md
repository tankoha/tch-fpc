# T05: Qwen Tokenizer — System Flow

## Callers

### T07: Chat Rendering

Task T07 uses `QwenTokenizer` to render chat messages in the Qwen chat format:
- Encodes system, user, and assistant messages
- Insert special tokens like `<|im_start|>` and `<|im_end|>`
- Prepares the tokenized prompt for the model

### T15: Training Record Tokenization

Task T15 (LoRA fine-tuning data preparation) uses `QwenTokenizer`:
- Tokenizes training records
- Prepares input/target sequences for fine-tuning

### Data Pipeline

The `qlora_tokenize` program is used by the data pipeline:
- Tokenizes corpus files in batch
- Outputs token IDs for downstream processing
- Supports JSONL format for structured data

## Unit Test Fixture

**`tests/ut/TokenizerFixture.pas`** provides:

- `TinyVocab` — A minimal vocabulary for unit testing:
  - 274 tokens (IDs 0-273)
  - Single-byte tokens (0-255) via `ByteToUnicode`
  - Test tokens: `he`, `ll`, `hell`, `hello`, `Ġw`, `or`, `Ġwor`, `ld`, `Ġworld`, `aa`, `ab`, `bc`
  - Special tokens: `<|im_start|>` (266, type 3), `<|im_end|>` (267, type 3), `_chg_` (268, type 4), `<tts_text_bos>` (269, type 3), `<tts_text_bos_single>` (270, type 3)
  - Non-special unused token: `<|unused|>` (273, type 5)

- `NewTinyTokenizer` — Creates a `TQwenTokenizer` from `TinyVocab`

- `WriteTinyGguf(Path, Model, Pre, OmitKey)` — Writes a minimal GGUF for testing

- Helper functions:
  - `IdsText(Tokens)` — Formats token IDs as `|id1 id2|`
  - `WordsText(Words)` — Formats words as `|w1|w2|`
  - `WriteBytes(Path, Data)` — Writes raw bytes to a file

## Integration Test (ITa)

### Oracle

The ITa compares `Encode` results against `llama-server`'s `/tokenize` endpoint:

- **Oracle location**: `work/oracle/T05/` (not in git)
  - `corpus.jsonl` — Test corpus (from `make-corpus.bb`)
  - `expected.tsv` — Token IDs from `/tokenize` (from `make-oracle.bb`)
  - `special.tsv` — Special token dump (from `vocabdump`)

- **Test corpus composition**:
  - 75 Pascal source files across five dialects
  - 402 English paragraphs (from DESIGN.md)
  - 402 Japanese paragraphs (from DESIGN_jp.md)
  - 68 special token test lines
  - 31 boundary test lines
  - 2000 random strings (fixed seed)
  - **Total**: 2978 records, 260,253 tokens

### Test Runner

```bash
QLORA_MODEL=<absolute path of rehearsal GGUF> <project root>/tests/ita/T05-run.sh
```

The test runner:
1. Builds `src/qlora_tokenize.pas`
2. Builds `tests/ita/T05/roundtrip_check.pas`
3. Runs both programs from a fresh directory outside the project
4. Compares output with oracle
5. Writes results to `<project root>/work/ita/T05/summary.tsv`

**Execution time**: Approximately 5 seconds

### Result File

`<project root>/work/ita/T05/summary.tsv` columns:
- Case ID
- OK/NG status
- Diff lines count
- Lines in our output
- Lines in oracle output
- Note
- Exit codes
- Stderr line count

## Dependencies

### From tch-fpc (shared)

- `Tch` — LibTorch bindings
- `TchNN` — Neural network utilities

### Pascal Standard Library

- `fphttpclient` — HTTP client (for potential `/tokenize` calls)
- `fpjson` — JSON parsing
- `process` — Process execution
- `unicodedata` — Unicode character properties (`GetProps`, `UGC_*` categories)
- `Generics.Collections` — `TDictionary` for token and merge lookups

### Project Units

- `GgufFile` — GGUF file reading (`TGgufFile.Open`, `FindKey`, `KV` access)
- `GgufFile` types: `TGgufKV`, arrays via `.Strs`, `.Ints`, `.StrVal`, `.IntVal`

## Directory Structure

```
samples/qwen-lora/
├── src/
│   ├── QwenTokenizer.pas          # Unit (T05b1, T05b2)
│   ├── TokenizeCli.pas            # Unit (T05b3)
│   └── qlora_tokenize.pas         # Program (complete, harness owner)
├── tests/
│   ├── ut/
│   │   ├── TokenizerFixture.pas   # UT fixture (complete, harness owner)
│   │   ├── QwenTokenizer_test.pas # UT stage 1 (T05b1)
│   │   ├── QwenTokenizerEnc_test.pas # UT stage 2 (T05b2)
│   │   ├── TokenizeCli_test.pas   # UT stage 3 (T05b3)
│   │   └── cases/
│   │       ├── QwenTokenizer.cases.md
│   │       └── TokenizeCli.cases.md
│   └── ita/
│       ├── T05-run.sh
│       ├── T05/
│       └── cases/
│           └── T05-ita.cases.md
└── work/
    └── oracle/
        └── T05/                    # Oracle (not in git)
```
