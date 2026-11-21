# T05 facts (checked by the harness owner)

T05 is the unit `QwenTokenizer` (the byte-level BPE tokenizer of the Qwen
GGUF), the unit `TokenizeCli` and the program `qlora_tokenize`. **H46**:
`Encode` equals the bundled `llama-server`'s `/tokenize` on a corpus of
Pascal sources of all five dialects, English and Japanese prose and the
special tokens, 100 % of records.

Measured by the harness owner with a prototype written exactly as below:
on the rehearsal model (`Qwen3.5-0.8B-Q4_K_M.gguf`) every one of 2978
records (75 Pascal files, 402 English and 402 Japanese paragraphs, 68
special-token lines, 31 boundary lines, 2000 random strings; 260 253
tokens) has the token ids of `/tokenize`, and `Decode(Encode(text))`
gives every text back, in 1.5 s including loading. The upstream code read:
`llama-vocab.cpp` (`llm_tokenizer_bpe_session`, `tokenizer_st_partition`)
and `unicode.cpp` (`unicode_regex_split_custom_qwen35`) of llama.cpp
`39febd2ad`. These facts are final; do not investigate further or write
probes.

## Vocabulary (GGUF metadata, measured on the rehearsal model)

| Key | Value |
|---|---|
| `tokenizer.ggml.model` | `gpt2` |
| `tokenizer.ggml.pre` | `qwen35` |
| `tokenizer.ggml.tokens` | array of 248 320 strings, byte-encoded (see "Byte map"; a space is `Ġ`) |
| `tokenizer.ggml.token_type` | array of 248 320 int32: 1 normal (248 044), 3 control (27), 4 user-defined (6), 5 unused (243) |
| `tokenizer.ggml.merges` | array of 247 587 strings `"left right"` (one space; tokens never contain a space); the index is the rank, lower merges first |
| `tokenizer.ggml.eos_token_id` | 248046 (`<|im_end|>`) |
| no `add_bos_token` key | no BOS is added; `/tokenize` adds none |

The **special tokens** are the tokens of type 3 and 4 (33 here, ids
248044 to 248076, for example `<|im_start|>` 248045, `<|im_end|>` 248046,
`<think>` 248068, `</think>` 248069, `<tool_call>` 248058). Type 5
(unused) is **not** special. `<tts_text_bos>` (248073) is a prefix of
`<tts_text_bos_single>` (248075): the longer one must be found first.

## Byte map (`ByteToUnicode`)

Every byte b maps to one code point, written as UTF-8:
- b in `$21..$7E`, `$A1..$AC`, `$AE..$FF`: the code point b itself
  (`'A'` → `41`; 233 → U+00E9, bytes `C3 A9`; 255 → `C3 BF`);
- every other byte (0..32, 127..160, 173), in increasing order, gets
  256, 257, …: 0 → U+0100 (`C4 80`), 10 → U+010A (`C4 8A`), 32 → U+0120
  (`C4 A0`, the `Ġ`), 127 → U+0121 (`C4 A1`), 160 → U+0142 (`C5 82`),
  173 → U+0143 (`C5 83`).
All 256 such strings are tokens of the vocabulary (measured; `Init`
checks it).

## Character classes (for the scanner)

From FPC's unit `unicodedata` (`GetProps(cp)^.Category`, the constants
`UGC_*`):
- **letter**: categories `UGC_UppercaseLetter .. UGC_OtherLetter` (L*);
- **mark**: `UGC_NonSpacingMark .. UGC_EnclosingMark` (M*);
- **number**: `UGC_DecimalNumber .. UGC_OtherNumber` (N*);
- **space**: exactly the code points `$09..$0D, $20, $85, $A0, $1680,
  $2000..$200A, $2028, $2029, $202F, $205F, $3000` (U+200B and U+FEFF are
  not spaces; measured);
- **defined**: every code point whose category is not `UGC_Unassigned`,
  and every space.
Positions outside the text have no class (all false) and the code point
`NoCp = $FFFFFFFF`.

## Encode, step by step

1. **Specials first.** Start with one raw fragment, the whole text (none
   for an empty text). For each special token in the order "longest text
   first" (for equal lengths, lower id first): in every raw fragment find
   every occurrence from left to right (byte comparison) and split the
   fragment into raw text, the special token, raw text, … Fragments that
   are already special tokens are kept. Empty raw pieces are dropped.
2. Each special fragment gives its id. Each raw fragment is **pre-tokenized**
   into words (below), and each word is **BPE-encoded** (below); the ids
   are appended in order.
3. Nothing is added at the start or the end (no BOS, no EOS).

### PreTokenize (the `qwen35` scanner)

Decode the UTF-8 text into code points `Cps[0..N-1]` with the byte offset
of each (strict UTF-8: a lead byte `$80..$C1` or `$F5..$FF`, a missing or
non-`10xxxxxx` continuation byte, or a sequence cut by the end raise
`ETokenizerError('invalid UTF-8 at byte K')`, K the 1-based byte position
in the text given to `PreTokenize`). Then, with `Pos := 0; Prev := 0`,
`F(p)` the class set of position p and `Cp(p)` its code point, and
`AddTok(e)` = "emit the bytes of code points Prev..e-1 as one word if
e > Prev; Prev := e":

```
while Pos < N:
  cp := Cp(Pos); f := F(Pos)
  -- contractions (ASCII letters, case-insensitive: lower only 'A'..'Z')
  if cp = '''' and Pos+1 < N:
    c1 := lower(Cp(Pos+1))
    if c1 in [s, t, m, d]: AddTok(Pos+2); Pos += 2; continue
    if Pos+2 < N:
      c2 := lower(Cp(Pos+2))
      if (c1,c2) in [(r,e), (v,e), (l,l)]: AddTok(Pos+3); Pos += 3; continue
  -- [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
  if not (cp = CR or cp = LF or f.number):
    if f.letter or f.mark or F(Pos+1).letter or F(Pos+1).mark:
      Pos += 1
      while F(Pos).letter or F(Pos).mark: Pos += 1
      AddTok(Pos); continue
  -- \p{N}  (one digit per word)
  if f.number: Pos += 1; AddTok(Pos); continue
  -- ' '?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
  f2 := F(Pos+1) if cp = ' ' else f
  if not (f2.space or f2.letter or f2.mark or f2.number) and f.defined:
    if cp = ' ': Pos += 1
    while not (f2.space or f2.letter or f2.mark or f2.number) and f2.defined:
      Pos += 1; f2 := F(Pos)
    while Cp(Pos) in [CR, LF]: Pos += 1
    AddTok(Pos); continue
  -- spaces
  nw := 0; lastRN := 0
  while F(Pos+nw).space:
    if Cp(Pos+nw) in [CR, LF]: lastRN := Pos+nw+1
    nw += 1
  if lastRN > 0: Pos := lastRN; AddTok(Pos); continue          -- \s*[\r\n]+
  if nw > 1 and Cp(Pos+nw) <> NoCp: Pos += nw-1; AddTok(Pos); continue   -- \s+(?!\S)
  if nw > 0: Pos += nw; AddTok(Pos); continue                  -- \s+
  Pos += 1; AddTok(Pos)                                        -- anything else alone
```

Note `f.defined` in the punctuation rule tests the **current** code point
(`f`), the loop tests `f2`; this is upstream's code and is what matches.

### BPE of one word

The word's bytes become symbols `ByteToUnicode(b)` (one per byte). For
each adjacent pair (i, i+1) its rank is the index of the merge
`Sym[i] + ' ' + Sym[i+1]` (none: no merge). Repeat: take the pair with the
**lowest rank; on equal ranks the leftmost**; replace the two symbols by
their concatenation; recompute only the ranks of the pairs with its left
and right neighbours. Stop when no pair has a rank. Each final symbol is a
vocabulary string; its id is the **first** id with that string. (On this
vocabulary every final symbol is a token, since merges only build tokens
and every single byte is a token.)

Prototype's loop (O(n²) scan of an `Integer` rank array, fast enough: the
whole corpus in about 1 s):

```
N := Length(word); Sym[i] := ByteToUnicode(word[i]); Rank[i] := rank(Sym[i], Sym[i+1]) or MaxInt; Rank[N-1] := MaxInt
while N > 1:
  best := MaxInt; bi := -1
  for i in 0..N-2: if Rank[i] < best: best := Rank[i]; bi := i      -- strict <: leftmost on ties
  if bi < 0: break
  Sym[bi] := Sym[bi] + Sym[bi+1]
  shift Sym[bi+2..N-1] and Rank[bi+2..N-2] one place left; N -= 1; Rank[N-1] := MaxInt
  if bi > 0: Rank[bi-1] := rank(Sym[bi-1], Sym[bi]) or MaxInt
  if bi < N-1: Rank[bi] := rank(Sym[bi], Sym[bi+1]) or MaxInt
```

## Decode

For each id (out of range → `ETokenizerError('token id N out of range')`):
a special token (type 3 or 4) gives its text as is; any other token's
string is split into UTF-8 characters, each mapped back to its byte by the
inverse of the byte map (a character that is not in it →
`ETokenizerError('token N: character not in the byte map')`). The bytes
are concatenated. `Decode(Encode(t)) = t` for every valid UTF-8 text.

## Unit `QwenTokenizer` (interface, final; in the skeleton)

```pascal
type
  ETokenizerError = class(Exception);
  TTokenArray = array of Integer;
  TWordArray = array of RawByteString;
  TQwenTokenizer = class
  public
    { Reads tokenizer.ggml.model (must be 'gpt2'), .pre (must be 'qwen35'),
      .tokens, .token_type, .merges, .eos_token_id; then Init. Errors:
      'tokenizer: key K missing', 'tokenizer: model M, expected gpt2',
      'tokenizer: pre P, expected qwen35', and Init's. }
    constructor Create(AGguf: TGgufFile);
    { The same from arrays (the UT's way). }
    constructor CreateFromVocab(const ATokens: array of RawByteString; const ATypes: array of Integer;
      const AMerges: array of RawByteString; AEosId: Integer);
    destructor Destroy; override;
    function PreTokenize(const AText: RawByteString): TWordArray;  { one raw fragment, no special handling }
    function Encode(const AText: RawByteString): TTokenArray;      { specials parsed, no BOS/EOS }
    function Decode(const ATokens: array of Integer): RawByteString;
    function SpecialId(const AText: RawByteString): Integer;       { id of the special token with that text, else -1 }
    property VocabSize: Integer;   { number of tokens }
    property EosId: Integer;
  end;
function ByteToUnicode(AByte: Byte): RawByteString;
```

`Init` (private, called by both constructors), errors in this order:
`'tokenizer: N tokens but M types'` (lengths differ), `'tokenizer: eos id
E out of range'` (not in 0..N-1), `'tokenizer: no token for byte B'` (a
byte string missing). It fills `FTokens`, `FTypes`, `FTokenId` (string →
first id), `FMergeRank` (merge string → first index), `FByteDec`
(`ByteToUnicode(b)` → b), `FSpecials` (types 3 and 4, sorted longest text
first, equal lengths by id; an insertion sort is fine, there are 33).

Implementation notes (from the prototype):
- `TTokDict = specialize TDictionary<RawByteString, Integer>`
  (`Generics.Collections`; `TryGetValue`, `ContainsKey`, `Add`). Loading
  the rehearsal vocabulary takes about 0.3 s.
- Keep every byte string a `RawByteString`; build strings with `Copy`,
  `+` and `Chr`, not `StringReplace`/`Format` (code-page conversions).
- `TGgufKV.Strs` and `.Ints` of `TGgufFile.KV[FindKey(...)]` hold the
  arrays; `.StrVal` the strings; `.IntVal` the eos id.
- `ByteToUnicode` for b ≥ 256 code points is two UTF-8 bytes
  (`$C0 or (cp shr 6)`, `$80 or (cp and $3F)`).

## Unit `TokenizeCli` and the program `qlora_tokenize`

`function RunTokenizeCLI(const AArgs: array of string; AOut, AErr: TStrings): Integer;`
(in the skeleton). `src/qlora_tokenize.pas` is complete (written by the
harness owner): it only forwards.

- Arguments `[--jsonl] MODEL FILE`. Anything else (fewer or more
  arguments, or any other argument starting with `-`) → `AErr.Add('Usage:
  qlora_tokenize [--jsonl] MODEL FILE')`, result 2.
- Open MODEL with `TGgufFile.Open`, create the tokenizer, read FILE **as
  bytes** (`TFileStream` into a `RawByteString`, then
  `SetCodePage(S, CP_UTF8, False)`).
- Plain: `AOut.Add(ids)`: the ids of `Encode(whole file)` joined by single
  spaces (an empty line for none).
- `--jsonl`: split the bytes on LF; drop one CR at the end of a line; skip
  empty lines. Each line: `SetCodePage(Line, CP_UTF8, False)`, then
  **`GetJSON(Line, False)`** (unit `jsonparser`): measured in FPC 3.2.2,
  `UseUTF8 = True` (the default) turns non-ASCII into `?`, while `False`
  keeps raw UTF-8 and decodes `é` to UTF-8. Not parseable →
  `'line L: not JSON (<message>)'`; not an object with a string `"text"`
  → `'line L: no string "text"'`. Take the text as `UTF8String`
  (`Strings['text']`), the id as `Get('id', IntToStr(L))` (L = 1-based
  line number), and `AOut.Add(id + #9 + ids)`.
- Every exception (open, read, tokenizer, JSON) → `AErr.Add('qlora_tokenize: '
  + E.Message)`, result 1; lines already added to AOut stay. Otherwise 0.
  Free the tokenizer and the GGUF in a `finally`.

## UT vocabulary (`tests/ut/TokenizerFixture.pas`, harness owner's)

`TinyVocab`: ids 0..255 are `ByteToUnicode(0..255)` (type 1); then
256 `he`, 257 `ll`, 258 `hell`, 259 `hello`, 260 `Ġw`, 261 `or`, 262
`Ġwor`, 263 `ld`, 264 `Ġworld`, 265 `aa`, 266 `<|im_start|>` (3), 267
`<|im_end|>` (3), 268 `<think>` (4), 269 `<tts_text_bos>` (3), 270
`<tts_text_bos_single>` (3), 271 `ab`, 272 `bc`, 273 `<|unused|>` (5);
others type 1. Merges in rank order: `h e`, `l l`, `he ll`, `hell o`,
`Ġ w`, `o r`, `Ġw or`, `l d`, `Ġwor ld`, `a a`, `b c`, `a b`. EOS 267.
`NewTinyTokenizer` = `CreateFromVocab` on it. `WriteTinyGguf(Path, Model,
Pre, OmitKey)` writes a GGUF without tensors holding it (defaults `gpt2`,
`qwen35`, nothing omitted). Also `IdsText`, `WordsText` (`|w1|w2|`),
`WriteBytes`. For a printable ASCII byte the token id is the byte
(`'x'` = 120, `' '` alone = 32).

## Unit tests (rows; every row has one method, every method belongs to a row)

Escapes below: `\n` LF, `\r` CR, `\t` TAB, `#$..` bytes; `·` separates
words. All rows use the tiny vocabulary unless they say otherwise.

**Stage 1, `tests/ut/QwenTokenizer_test.pas`, rows UT-QwenTokenizer-001..**

| Row | Input | Expected |
|---|---|---|
| 001 | `ByteToUnicode` of 65, 32, 0, 10, 127, 160, 173, 233, 255 | bytes `41`; `C4 A0`; `C4 80`; `C4 8A`; `C4 A1`; `C5 82`; `C5 83`; `C3 A9`; `C3 BF` |
| 002 | `NewTinyTokenizer` | VocabSize 274, EosId 267 |
| 003 | `CreateFromVocab` with 273 types | ETokenizerError, message `tokenizer: 274 tokens but 273 types` |
| 004 | eos -1 | ETokenizerError `tokenizer: eos id -1 out of range` |
| 005 | eos 274 | ETokenizerError `tokenizer: eos id 274 out of range` |
| 006 | token 0 replaced by `X0` | ETokenizerError `tokenizer: no token for byte 0` |
| 007 | PreTokenize `hello world` | `hello` · ` world` |
| 008 | `I'm sure it's THEY'LL we'd'VE` | `I` · `'m` · ` sure` · ` it` · `'s` · ` THEY` · `'LL` · ` we` · `'d` · `'VE` |
| 009 | `'x 're'll' ` | `'x` · ` '` · `re` · `'ll` · `'` · ` ` |
| 010 | `abc123 4.5` | `abc` · `1` · `2` · `3` · ` ` · `4` · `.` · `5` |
| 011 | `a  b   c` | `a` · ` ` · ` b` · `  ` · ` c` |
| 012 | `x   ` | `x` · `   ` |
| 013 | `a\nb\r\n\r\nc` | `a` · `\n` · `b` · `\r\n\r\n` · `c` |
| 014 | `  \n  x` | `  \n` · ` ` · ` x` |
| 015 | `\ta\t\tb` | `\ta` · `\t` · `\tb` |
| 016 | `foo(bar); x:=1;\n` | `foo` · `(bar` · `);` · ` x` · `:=` · `1` · `;\n` |
| 017 | ` !x ?y .z` | ` !` · `x` · ` ?` · `y` · ` .` · `z` |
| 018 | `!!!\n\nx` | `!!!\n\n` · `x` |
| 019 | `caf#$C3#$A9 cafe#$CC#$81` | `caf#$C3#$A9` · ` cafe#$CC#$81` |
| 020 | `#$CC#$81a b#$CC#$81` | `#$CC#$81a` · ` b#$CC#$81` |
| 021 | `日本語。１２` (UTF-8) | `日本語` · `。` · `１` · `２` |
| 022 | `x#$C2#$A0y#$E3#$80#$80z` | `x` · `#$C2#$A0y` · `#$E3#$80#$80z` |
| 023 | `#$F0#$9F#$98#$80 ok` | `#$F0#$9F#$98#$80` · ` ok` |
| 024 | empty | no words |
| 025 | `#$FF` | ETokenizerError `invalid UTF-8 at byte 1` |
| 026 | `a#$C3` | ETokenizerError `invalid UTF-8 at byte 2` |
| 027 | `#$C3#$41` | ETokenizerError `invalid UTF-8 at byte 2` |
| 028 | `#$C0#$80` | ETokenizerError `invalid UTF-8 at byte 1` |
| 029 | `#$F5#$80#$80#$80` | ETokenizerError `invalid UTF-8 at byte 1` |
| 030 | `#$CD#$B8` (U+0378, unassigned: no class), then `#$CD#$B8#$CD#$B8!` | one word `#$CD#$B8`; then `#$CD#$B8` · `#$CD#$B8` · `!` (the scanner's last rule; measured against llama-server: ids 137 116 137 116 0) |

**Stage 2, `tests/ut/QwenTokenizerEnc_test.pas`, rows UT-QwenTokenizer-101..**

| Row | Input | Expected |
|---|---|---|
| 101 | Encode `hello world` | 259 264 |
| 102 | `abc` | 97 272 (rank of `b c` is lower than `a b`) |
| 103 | `aaa` | 265 97 (equal ranks: leftmost) |
| 104 | `aaaa` | 265 265 |
| 105 | `<|im_start|>hello<|im_end|>` | 266 259 267 |
| 106 | `<tts_text_bos_single>x<tts_text_bos>` | 270 120 269 |
| 107 | `<think><think>` | 268 268 |
| 108 | `<|im_start` | 60 124 105 109 95 115 116 97 114 116 |
| 109 | `<|unused|>` (type 5, not special) | 60 124 117 110 117 115 101 100 124 62 |
| 110 | ` ` | 32 |
| 111 | empty | no ids |
| 112 | `#$C3#$A9 #$E3#$81#$82` | 195 169 32 227 129 130 |
| 113 | `x#$FF` | ETokenizerError `invalid UTF-8 at byte 2` |
| 114 | Decode 266 259 264 267 | `<|im_start|>hello world<|im_end|>` |
| 115 | Decode 195 169 | bytes `C3 A9` |
| 116 | Decode of no ids | empty |
| 117 | Decode 273 | `<|unused|>` |
| 118 | Decode 274 | ETokenizerError `token id 274 out of range` |
| 119 | Decode -1 | ETokenizerError `token id -1 out of range` |
| 120 | vocabulary + token 274 `日` (UTF-8, type 1); Decode 274 | ETokenizerError `token 274: character not in the byte map` |
| 121 | Decode(Encode(s)) for every input of rows 007..023 and 101..112 | equals s |
| 122 | SpecialId `<think>`, `<|im_end|>`, `hello`, `<|unused|>` | 268, 267, -1, -1 |
| 123 | `Create` on `WriteTinyGguf(Dir + '/t.gguf')` opened with `TGgufFile.Open` | VocabSize 274, EosId 267, Encode `hello world` = 259 264 |
| 124 | model `llama` | ETokenizerError `tokenizer: model llama, expected gpt2` |
| 125 | pre `llama3` | ETokenizerError `tokenizer: pre llama3, expected qwen35` |
| 126 | key `tokenizer.ggml.merges` omitted | ETokenizerError `tokenizer: key tokenizer.ggml.merges missing` |
| 127 | key `tokenizer.ggml.eos_token_id` omitted | ETokenizerError `tokenizer: key tokenizer.ggml.eos_token_id missing` |
| 128 | key `tokenizer.ggml.tokens` omitted | ETokenizerError `tokenizer: key tokenizer.ggml.tokens missing` |
| 129 | key `tokenizer.ggml.token_type` omitted | ETokenizerError `tokenizer: key tokenizer.ggml.token_type missing` |
| 130 | key `tokenizer.ggml.model` omitted | ETokenizerError `tokenizer: key tokenizer.ggml.model missing` |
| 131 | key `tokenizer.ggml.pre` omitted | ETokenizerError `tokenizer: key tokenizer.ggml.pre missing` |

**Stage 3, `tests/ut/TokenizeCli_test.pas`, rows UT-TokenizeCli-001..**
(model `Dir + '/tiny.gguf'` from `WriteTinyGguf`; files by `WriteBytes`;
"out"/"err" are the lines of AOut/AErr)

| Row | Arguments and file | Expected |
|---|---|---|
| 001 | MODEL, a file `hello world<|im_end|>` | exit 0; out `259 264 267`; err none |
| 002 | MODEL, an empty file | exit 0; out one empty line |
| 003 | `--jsonl`, lines `{"id":"a","text":"hello"}`, `{"text":"aaa"}`, empty, `{"id":"c","text":"<think>x"}\r` | exit 0; out `a\t259`, `2\t265 97`, `c\t268 120` |
| 004 | `--jsonl`, `{"text":"é あ"}` | exit 0; out `1\t195 169 32 227 129 130` |
| 005 | `--jsonl`, the same text as raw UTF-8 (`#$C3#$A9 #$E3#$81#$82`) | the same as 004 |
| 006 | no arguments | exit 2; err `Usage: qlora_tokenize [--jsonl] MODEL FILE` |
| 007 | MODEL only | exit 2; the usage line |
| 008 | MODEL, FILE, `x` | exit 2; the usage line |
| 009 | `--x`, MODEL, FILE | exit 2; the usage line |
| 010 | a missing model file | exit 1; err starts with `qlora_tokenize: ` and names the path |
| 011 | a missing text file | exit 1; err names the path |
| 012 | model written with model `llama` | exit 1; err `qlora_tokenize: tokenizer: model llama, expected gpt2` |
| 013 | pre `llama3` | exit 1; err `qlora_tokenize: tokenizer: pre llama3, expected qwen35` |
| 014 | merges omitted | exit 1; err `qlora_tokenize: tokenizer: key tokenizer.ggml.merges missing` |
| 015 | `--jsonl`, lines `{"text":"x"}`, `not json` | exit 1; out `1\t120`; err starts with `qlora_tokenize: line 2: not JSON (` |
| 016 | `--jsonl`, `{"text": 5}` | exit 1; err `qlora_tokenize: line 1: no string "text"` |
| 017 | `--jsonl`, `["text"]` | exit 1; err `qlora_tokenize: line 1: no string "text"` |
| 018 | a file `ok#$FF` | exit 1; err `qlora_tokenize: invalid UTF-8 at byte 3` |
| 019 | after `SetCurrentDir(Dir)` (restore it in a `finally`): `tiny.gguf`, `a.txt` (bare names) | exit 0; out `259 264 267` |
| 020 | `-x`, FILE (two arguments; added in T05b3c) | exit 2; the usage line; no output |

Compare ids as text with `IdsText`, words with `WordsText`, bytes with
`=` on `RawByteString`. Build every test program with
`-Fu{proj}/tests/ut` (the fixture lives there).

## ITa (prepared by the harness owner)

`QLORA_MODEL=<absolute path of the rehearsal GGUF> {proj}/tests/ita/T05-run.sh`
builds `src/qlora_tokenize.pas` and `tests/ita/T05/roundtrip_check.pas`,
runs them from a fresh directory outside the project and writes
`{proj}/work/ita/T05/summary.tsv` (case, OK/NG, diff lines, lines ours,
lines oracle, note; then the drivers' exit codes and the stderr line
count). About 5 s.

| Case | Compared | Lines |
|---|---|---|
| ITa-T05-001 | token ids = `/tokenize`: Pascal files of five dialects | 75 |
| ITa-T05-002 | English paragraphs (DESIGN.md) | 402 |
| ITa-T05-003 | Japanese paragraphs (DESIGN_jp.md) | 402 |
| ITa-T05-004 | special tokens alone, adjacent, in text, chat | 68 |
| ITa-T05-005 | boundary lines | 31 |
| ITa-T05-006 | random strings (fixed seed) | 2000 |
| ITa-T05-007 | `Decode(Encode(text)) = text` for every record | 2978 |
| ITa-T05-008 | plain mode on one whole source file (absolute path) | 1 |

The oracle (`work/oracle/T05`, not in git): `corpus.jsonl`
(`make-corpus.bb`), `expected.tsv` (`make-oracle.bb`, `/tokenize` of the
bundled `llama-server` on the CPU), `special.tsv` (`vocabdump`).

## Stages and case IDs

- Stage 1 (task T05b1): `ByteToUnicode`, `CreateFromVocab`/`Init`,
  `GetVocabSize`, `PreTokenize`; rows UT-QwenTokenizer-001 to 030 (030 added in T05b3).
- Stage 2 (task T05b2): `EncodeWord`, `Encode`, `Decode`, `SpecialId`,
  `Create`; rows UT-QwenTokenizer-101 to 131 (128 to 131 added in T05b3).
- Stage 3 (task T05b3): `RunTokenizeCLI`; rows UT-TokenizeCli-001 to 020 (020 added in T05b3c).
