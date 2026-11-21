# UT cases — unit QwenTokenizer (T05b1)

| Case ID | Title | Target function | Input (short) | Expected result | Notes |
|---|---|---|---|---|---|
| UT-QwenTokenizer-001 | | | ByteToUnicode of 65, 32, 0, 10, 127, 160, 173, 233, 255 | bytes 41; C4 A0; C4 80; C4 8A; C4 A1; C5 82; C5 83; C3 A9; C3 BF | |
| UT-QwenTokenizer-002 | | | NewTinyTokenizer | VocabSize 274, EosId 267 | |
| UT-QwenTokenizer-003 | | | CreateFromVocab with 273 types | ETokenizerError, message `tokenizer: 274 tokens but 273 types` | |
| UT-QwenTokenizer-004 | | | eos -1 | ETokenizerError `tokenizer: eos id -1 out of range` | |
| UT-QwenTokenizer-005 | | | eos 274 | ETokenizerError `tokenizer: eos id 274 out of range` | |
| UT-QwenTokenizer-006 | | | token 0 replaced by X0 | ETokenizerError `tokenizer: no token for byte 0` | |
| UT-QwenTokenizer-007 | | | PreTokenize `hello world` | `hello` · ` world` | |
| UT-QwenTokenizer-008 | | | `I'm sure it's THEY'LL we'd'VE` | `I` · `'m` · ` sure` · ` it` · `'s` · ` THEY` · `'LL` · ` we` · `'d` · `'VE` | |
| UT-QwenTokenizer-009 | | | `'x 're'll' ` | `'x` · ` '` · `re` · `'ll` · `'` · ` ` | |
| UT-QwenTokenizer-010 | | | `abc123 4.5` | `abc` · `1` · `2` · `3` · ` ` · `4` · `.` · `5` | |
| UT-QwenTokenizer-011 | | | `a  b   c` | `a` · ` ` · ` b` · `  ` · ` c` | |
| UT-QwenTokenizer-012 | | | `x   ` | `x` · `   ` | |
| UT-QwenTokenizer-013 | | | `a\n b\r\n\r\nc` | `a` · `\n` · `b` · `\r\n\r\n` · `c` | |
| UT-QwenTokenizer-014 | | | `  \n  x` | `  \n` · ` ` · ` x` | |
| UT-QwenTokenizer-015 | | | `\ta\ta\tb` | `\ta` · `\t` · `\tb` | |
| UT-QwenTokenizer-016 | | | `foo(bar); x:=1;\n` | `foo` · `(bar` · `);` · ` x` · `:=` · `1` · `;\n` | |
| UT-QwenTokenizer-017 | | | ` !x ?y .z` | ` !` · `x` · ` ?` · `y` · ` .` · `z` | |
| UT-QwenTokenizer-018 | | | `!!!\n\nx` | `!!!\n\n` · `x` | |
| UT-QwenTokenizer-019 | | | `caf#$C3#$A9 cafe#$CC#$81` | `caf#$C3#$A9` · ` cafe#$CC#$81` | |
| UT-QwenTokenizer-020 | | | `#$CC#$81a b#$CC#$81` | `#$CC#$81a` · ` b#$CC#$81` | |
| UT-QwenTokenizer-021 | | | `日本語。１２` (UTF-8) | `日本語` · `。` · `１` · `２` | |
| UT-QwenTokenizer-022 | | | `x#$C2#$A0y#$E3#$80#$80z` | `x` · `#$C2#$A0y` · `#$E3#$80#$80z` | |
| UT-QwenTokenizer-023 | | | `#$F0#$9F#$98#$80 ok` | `#$F0#$9F#$98#$80` · ` ok` | |
| UT-QwenTokenizer-024 | | | empty | no words | |
| UT-QwenTokenizer-025 | | | `#$FF` | ETokenizerError `invalid UTF-8 at byte 1` | |
| UT-QwenTokenizer-026 | | | `a#$C3` | ETokenizerError `invalid UTF-8 at byte 2` | |
| UT-QwenTokenizer-027 | | | `#$C3#$41` | ETokenizerError `invalid UTF-8 at byte 2` | |
| UT-QwenTokenizer-028 | | | `#$C0#$80` | ETokenizerError `invalid UTF-8 at byte 1` |
| UT-QwenTokenizer-029 | | | `#$F5#$80#$80#$80` | ETokenizerError `invalid UTF-8 at byte 1` |
| UT-QwenTokenizer-030 | | | `#$CD#$B8` (U+0378, unassigned: no class), then `#$CD#$B8#$CD#$B8!` | one word `#$CD#$B8`; then `#$CD#$B8` · `#$CD#$B8` · `!` (ids 137 116 137 116 0) |
| UT-QwenTokenizer-101 | | | Encode `hello world` | 259 264 |
| UT-QwenTokenizer-102 | | | abc | 97 272 (rank of `b c` is lower than `a b`) | |
| UT-QwenTokenizer-103 | | | aaa | 265 97 (equal ranks: leftmost) | |
| UT-QwenTokenizer-104 | | | aaaa | 265 265 | |
| UT-QwenTokenizer-105 | | | `<|im_start|>hello<|im_end|>` | 266 259 267 | |
| UT-QwenTokenizer-106 | | | `<tts_text_bos_single>x<tts_text_bos>` | 270 120 269 | |
| UT-QwenTokenizer-107 | | | `<think><think>` | 268 268 | |
| UT-QwenTokenizer-108 | | | `<|im_start` | 60 124 105 109 95 115 116 97 114 116 | |
| UT-QwenTokenizer-109 | | | `<|unused|>` (type 5, not special) | 60 124 117 110 117 115 101 100 124 62 | |
| UT-QwenTokenizer-110 | | | ` ` | 32 | |
| UT-QwenTokenizer-111 | | | empty | no ids | |
| UT-QwenTokenizer-112 | | | `#$C3#$A9 #$E3#$81#$82` | 195 169 32 227 129 130 | |
| UT-QwenTokenizer-113 | | | `x#$FF` | ETokenizerError `invalid UTF-8 at byte 2` | |
| UT-QwenTokenizer-114 | | | Decode 266 259 264 267 | `<|im_start|>hello world<|im_end|>` | |
| UT-QwenTokenizer-115 | | | Decode 195 169 | bytes `C3 A9` | |
| UT-QwenTokenizer-116 | | | Decode of no ids | empty | |
| UT-QwenTokenizer-117 | | | Decode 273 | `<|unused|>` | |
| UT-QwenTokenizer-118 | | | Decode 274 | ETokenizerError `token id 274 out of range` | |
| UT-QwenTokenizer-119 | | | Decode -1 | ETokenizerError `token id -1 out of range` | |
| UT-QwenTokenizer-120 | | | vocabulary + token 274 `日` (UTF-8, type 1); Decode 274 | ETokenizerError `token 274: character not in the byte map` | |
| UT-QwenTokenizer-121 | | | Decode(Encode(s)) for every input of rows 007..023 and 101..112 | equals s | |
| UT-QwenTokenizer-122 | | | SpecialId `<think>`, `<|im_end|>`, `hello`, `<|unused|>` | 268, 267, -1, -1 | |
| UT-QwenTokenizer-123 | | | `Create` on `WriteTinyGguf(Dir + '/t.gguf')` opened with `TGgufFile.Open` | VocabSize 274, EosId 267, Encode `hello world` = 259 264 | |
| UT-QwenTokenizer-124 | | | model `llama` | ETokenizerError `tokenizer: model llama, expected gpt2` | |
| UT-QwenTokenizer-125 | | | pre `llama3` | ETokenizerError `tokenizer: pre llama3, expected qwen35` | |
| UT-QwenTokenizer-126 | | | key `tokenizer.ggml.merges` omitted | ETokenizerError `tokenizer: key tokenizer.ggml.merges missing` |
| UT-QwenTokenizer-127 | | | key `tokenizer.ggml.eos_token_id` omitted | ETokenizerError `tokenizer: key tokenizer.ggml.eos_token_id missing` |
| UT-QwenTokenizer-128 | | | key `tokenizer.ggml.tokens` omitted | ETokenizerError `tokenizer: key tokenizer.ggml.tokens missing` |
| UT-QwenTokenizer-129 | | | key `tokenizer.ggml.token_type` omitted | ETokenizerError `tokenizer: key tokenizer.ggml.token_type missing` |
| UT-QwenTokenizer-130 | | | key `tokenizer.ggml.model` omitted | ETokenizerError `tokenizer: key tokenizer.ggml.model missing` |
| UT-QwenTokenizer-131 | | | key `tokenizer.ggml.pre` omitted | ETokenizerError `tokenizer: key tokenizer.ggml.pre missing` |
| UT-QwenTokenizer-132 | | | vocabulary + token 274 `#$F0#$9F#$98#$80` (4‑byte UTF‑8, type 1); Decode 274 | ETokenizerError `token 274: character not in the byte map` |
