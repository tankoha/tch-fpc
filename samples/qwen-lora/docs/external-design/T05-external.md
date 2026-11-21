# T05: Qwen Tokenizer — External Design

## Purpose

T05 implements the byte-level BPE tokenizer for the Qwen GGUF model. It provides:

1. **Unit `QwenTokenizer`** — The tokenizer core: vocabulary loading, pre-tokenization (the qwen35 scanner), BPE encoding, and decoding.
2. **Unit `TokenizeCli` and program `qlora_tokenize`** — Command-line interface for tokenizing files, used by the data pipeline.

## Acceptance Criterion (H46)

`Encode` equals the bundled `llama-server`'s `/tokenize` on a corpus of Pascal sources of all five dialects, English and Japanese prose, and the special tokens, 100% of records.

Measured on the rehearsal model (`Qwen3.5-0.8B-Q4_K_M.gguf`): every one of 2978 records (75 Pascal files, 402 English and 402 Japanese paragraphs, 68 special-token lines, 31 boundary lines, 2000 random strings; 260,253 tokens) has the token IDs of `/tokenize`, and `Decode(Encode(text))` gives every text back, in 1.5 seconds including loading.

## Unit `QwenTokenizer` Interface

```pascal
type
  ETokenizerError = class(Exception);
  TTokenArray = array of Integer;
  TWordArray = array of RawByteString;
  TQwenTokenizer = class
  public
    constructor Create(AGguf: TGgufFile);
    constructor CreateFromVocab(const ATokens: array of RawByteString;
      const ATypes: array of Integer;
      const AMerges: array of RawByteString; AEosId: Integer);
    destructor Destroy; override;
    function PreTokenize(const AText: RawByteString): TWordArray;
    function Encode(const AText: RawByteString): TTokenArray;
    function Decode(const ATokens: array of Integer): RawByteString;
    function SpecialId(const AText: RawByteString): Integer;
    property VocabSize: Integer;
    property EosId: Integer;
  end;

function ByteToUnicode(AByte: Byte): RawByteString;
```

### Methods

- **`Create(AGguf: TGgufFile)`** — Reads tokenizer metadata from the GGUF file:
  - `tokenizer.ggml.model` (must be `'gpt2'`)
  - `tokenizer.ggml.pre` (must be `'qwen35'`)
  - `tokenizer.ggml.tokens`, `tokenizer.ggml.token_type`, `tokenizer.ggml.merges`
  - `tokenizer.ggml.eos_token_id`

- **`CreateFromVocab(...)`** — Creates the tokenizer from arrays (used by unit tests).

- **`PreTokenize(AText)`** — Splits a raw text fragment into words (the qwen35 scanner). No special token handling.

- **`Encode(AText)`** — Tokenizes text: parses special tokens, pre-tokenizes raw fragments, applies BPE. No BOS/EOS added.

- **`Decode(ATokens)`** — Converts token IDs back to text.

- **`SpecialId(AText)`** — Returns the ID of the special token with that text, or -1 if not a special token.

- **`VocabSize`** — Number of tokens in the vocabulary.

- **`EosId`** — The EOS token ID from the GGUF metadata.

### Errors

All errors raise `ETokenizerError`:

**From `Create` and `Init`:**
- `'tokenizer: key K missing'` — Required GGUF key not found
- `'tokenizer: model M, expected gpt2'` — Wrong tokenizer model
- `'tokenizer: pre P, expected qwen35'` — Wrong pre-tokenizer type
- `'tokenizer: N tokens but M types'` — Token and type array lengths differ
- `'tokenizer: eos id E out of range'` — EOS ID not in 0..VocabSize-1
- `'tokenizer: no token for byte B'` — A single-byte token missing from vocabulary

**From `PreTokenize` and `Encode`:**
- `'invalid UTF-8 at byte K'` — Malformed UTF-8 in input text

**From `Decode`:**
- `'token id N out of range'` — Token ID not in 0..VocabSize-1
- `'token N: character not in the byte map'` — Token string contains a character not mapped to a byte

## Unit `TokenizeCli` and Program `qlora_tokenize`

```pascal
function RunTokenizeCLI(const AArgs: array of string; AOut, AErr: TStrings): Integer;
```

### Command Line

```
qlora_tokenize [--jsonl] MODEL FILE
```

- `MODEL` — Path to the GGUF model file
- `FILE` — Path to the input file (read as bytes, then interpreted as UTF-8)
- `--jsonl` — Optional flag for JSONL mode

### Modes

**Plain mode** (no `--jsonl`):
- Encodes the entire file as one text
- Output: token IDs joined by single spaces (empty line for no tokens)

**JSONL mode** (`--jsonl`):
- Each line parsed as JSON
- Each object must have a `"text"` string field
- Optional `"id"` field (default: 1-based line number)
- Output per line: `id<TAB>token_ids`

### Exit Codes

- **0** — Success
- **1** — Error (file open, read, tokenizer, JSON parse); error message prefixed with `qlora_tokenize: `
- **2** — Usage error (wrong number of arguments, unknown argument starting with `-`)

### Error Messages

- `'Usage: qlora_tokenize [--jsonl] MODEL FILE'` — Usage error (exit 2)
- `'qlora_tokenize: <exception message>'` — Any exception during execution (exit 1)
- `'line L: not JSON (<message>)'` — JSON parse error on line L
- `'line L: no string "text"'` — JSON object missing the `"text"` field

## Vocabulary (GGUF Metadata)

| Key | Value |
|---|---|
| `tokenizer.ggml.model` | `'gpt2'` |
| `tokenizer.ggml.pre` | `'qwen35'` |
| `tokenizer.ggml.tokens` | Array of 248,320 strings (byte-encoded) |
| `tokenizer.ggml.token_type` | Array of 248,320 int32: 1=normal, 3=control, 4=user-defined, 5=unused |
| `tokenizer.ggml.merges` | Array of 247,587 strings `"left right"` (rank = index) |
| `tokenizer.ggml.eos_token_id` | 248046 (`<|im_end|>`) |

**Special tokens**: Tokens with type 3 (control) or 4 (user-defined). Type 5 (unused) is NOT special.

## Byte Map (`ByteToUnicode`)

Every byte `b` maps to one UTF-8 code point:
- `b` in `$21..$7E`, `$A1..$AC`, `$AE..$FF`: the code point is `b` itself
- Every other byte (0..32, 127..160, 173), in increasing order, maps to 256, 257, ...

Examples:
- `A` (65) → `$41`
- Space (32) → U+0120 (`$C4 $A0`) — the `Ġ` character
- 0 → U+0100 (`$C4 $80`)
- 127 → U+0121 (`$C4 $A1`)
- 233 → U+00E9 (`$C3 $A9`)
- 255 → U+00FF (`$C3 $BF`)

All 256 byte strings are tokens in the vocabulary.

## Character Classes (for Pre-Tokenization)

From FPC's `unicodedata` unit:
- **Letter**: Categories `UGC_UppercaseLetter .. UGC_OtherLetter` (L*)
- **Mark**: `UGC_NonSpacingMark .. UGC_EnclosingMark` (M*)
- **Number**: `UGC_DecimalNumber .. UGC_OtherNumber` (N*)
- **Space**: Specific code points (`$09..$0D`, `$20`, `$85`, `$A0`, `$1680`, `$2000..$200A`, `$2028`, `$2029`, `$202F`, `$205F`, `$3000`)
- **Defined**: Every code point whose category is not `UGC_Unassigned`, plus all spaces
