# QwenTokenizer Detailed Design (T05)

## 1) Module structure

- **Unit `TQwenTokenizer`** – public tokenizer class.
- **Unit `TokenizeCli` / program `qlora_tokenize`** – CLI wrapper.
- No auxiliary units; all logic lives in the two units.

## 2) Function specifications

### `function ByteToUnicode(AByte: Byte): RawByteString;`
- **Params**: `AByte: Byte` – byte value.
- **Returns**: UTF‑8 encoded `RawByteString` corresponding to the byte map.
- **Exceptions**: None.
- **Algorithm**: See *Byte Map* section in external design.

### `constructor TQwenTokenizer.Create(AGguf: TGgufFile);`
- **Params**: `AGguf: TGgufFile` – GGUF model file.
- **Returns**: Initialized tokenizer.
- **Exceptions**: `ETokenizerError` – missing keys, wrong model/pre, length mismatch, EOS out of range, missing byte token.
- **Algorithm**: Load metadata, call `Init`.

### `constructor TQwenTokenizer.CreateFromVocab(const ATokens: array of RawByteString; const ATypes: array of Integer; const AMerges: array of RawByteString; AEosId: Integer);`
- **Params**: Arrays `ATokens`, `ATypes`, `AMerges`, integer `AEosId`.
- **Returns**: Initialized tokenizer.
- **Exceptions**: `ETokenizerError` – token/ type mismatch, EOS out of range, missing byte token.
- **Algorithm**: Assign arrays to internal fields, call `Init`.

### `destructor TQwenTokenizer.Destroy; override;`
- **Algorithm**: Free internal dictionaries.

### `function TQwenTokenizer.PreTokenize(const AText: RawByteString): TWordArray;`
- **Params**: `AText: RawByteString` – raw fragment (no special token handling).
- **Returns**: Array of words (`TWordArray`).
- **Exceptions**: `ETokenizerError('invalid UTF‑8…')`.
- **Algorithm**: **Qwen35 scanner** – pseudo‑code verbatim (lines 90‑124).

### `function TQwenTokenizer.Encode(const AText: RawByteString): TTokenArray;`
- **Params**: `AText: RawByteString` – arbitrary text.
- **Returns**: Token ID array (`TTokenArray`).
- **Exceptions**: `ETokenizerError` – from `PreTokenize` or BPE.
- **Algorithm**:
  1. Special tokens: split by longest‑first scan; each special fragment yields its ID via `SpecialId`.
  2. Each raw fragment is pre‑tokenized (see *PreTokenize*).
  3. Each word is BPE‑encoded (see *BPE loop* below).
  4. Concatenate all ids.

### `function TQwenTokenizer.EncodeWord(const AWord: RawByteString): TTokenArray;`
- **Params**: `AWord` – single word produced by `PreTokenize`.
- **Returns**: Token ID array for that word.
- **Exceptions**: None.
- **Algorithm**: **BPE loop** – pseudo‑code verbatim (lines 144‑154).

### `function TQwenTokenizer.Decode(const ATokens: array of Integer): RawByteString;`
- **Params**: `ATokens` – ID array.
- **Returns**: UTF‑8 text.
- **Exceptions**: `ETokenizerError('token id N out of range')`, `ETokenizerError('token N: character not in byte map')`.
- **Algorithm**: For each id, if special → token string; else split string into UTF‑8 characters, map each character to an original byte via inverse byte map; concatenate bytes.

### `function TQwenTokenizer.SpecialId(const AText: RawByteString): Integer;`
- **Params**: Candidate special‑token text.
- **Returns**: ID or `-1`.
- **Exceptions**: None.
- **Algorithm**: Linear search in `FSpecials` list; compare text, return ID.

### `function TQwenTokenizer.VocabSize: Integer;`
- **Returns**: `Length(FTokens)`.

### `function TQwenTokenizer.EosId: Integer;`
- **Returns**: Stored EOS id.

### `function RunTokenizeCLI(const AArgs: array of string; AOut, AErr: TStrings): Integer;`
- **Params**: CLI args, output and error string lists.
- **Returns**: Exit code.
- **Exceptions**: `AErr.Add` messages; no raise.
- **Algorithm**: Parse args, open model, create tokenizer, read file bytes, `Encode`, format output per mode (plain or JSONL). Handle errors as described in external design.

## 3) Data structures

```pascal
const
  NoCp = $FFFFFFFF;

type
  TTokenArray = array of Integer;
  TWordArray = array of RawByteString;

var
  FTokens    : array of RawByteString; // token string -> all
  FTypes     : array of Integer;        // token type
  FTokenId   : TDictionary<RawByteString, Integer>; // string → first id
  FMergeRank : TDictionary<RawByteString, Integer>; // merge string → rank
  FByteDec   : array[0..255] of Byte; // inverse of ByteToUnicode
  FSpecials  : array of record
    Text : RawByteString;
    Id   : Integer;
  end;
```

## 4) Processing details

| External rule | Implemented by | Notes |
|---------------|---------------|-------|
| *Load metadata* | `Create / CreateFromVocab` + `Init` | key checks, error handling
| *Special‑token parsing* | `Encode` (special‑first scan) | longest‑first, lower‑id tie
| *UTF‑8 decoding* | `PreTokenize` | strict UTF‑8 error
| *Scanner* | `PreTokenize` – Qwen35 scanner pseudo‑code | lines 90‑124
| *Byte map* | `ByteToUnicode` + inverse `FByteDec` | table from facts
| *BPE* | `EncodeWord` – BPE loop pseudo‑code | lines 144‑154
| *Decoding* | `Decode` – inverse mapping and special token handling | token out of range, invalid char
| *CLI modes* | `RunTokenizeCLI` | plain, jsonl, error messages

## 5) Change history

- _None_ – new design.
