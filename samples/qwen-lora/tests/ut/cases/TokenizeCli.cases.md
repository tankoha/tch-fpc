# UT cases — unit TokenizeCli (T05b3)

| Case ID | Title | Target function | Input (short) | Expected result | Notes |
|---|---|---|---|---|---|
| UT-TokenizeCli-001 | | | MODEL, a file `hello world<|im_end|>` | exit 0; out `259 264 267`; err none |
| UT-TokenizeCli-002 | | | MODEL, an empty file | exit 0; out one empty line |
| UT-TokenizeCli-003 | | | `--jsonl`, lines `{"id":"a","text":"hello"}`, `{"text":"aaa"}`, empty, `{"id":"c","text":"<think>x"}` | exit 0; out `a\t259`, `2\t265 97`, `c\t268 120` |
| UT-TokenizeCli-004 | | | `--jsonl`, `{"text":"é あ"}` | exit 0; out `1\t195 169 32 227 129 130` |
| UT-TokenizeCli-005 | | | `--jsonl`, the same text as raw UTF-8 (`#$C3#$A9 #$E3#$81#$82`) | the same as 004 |
| UT-TokenizeCli-006 | | | no arguments | exit 2; err `Usage: qlora_tokenize [--jsonl] MODEL FILE` |
| UT-TokenizeCli-007 | | | MODEL only | exit 2; the usage line |
| UT-TokenizeCli-008 | | | MODEL, FILE, `x` | exit 2; the usage line |
| UT-TokenizeCli-009 | | | `--x`, MODEL, FILE | exit 2; the usage line |
| UT-TokenizeCli-010 | | | a missing model file | exit 1; err starts with `qlora_tokenize: ` and names the path |
| UT-TokenizeCli-011 | | | a missing text file | exit 1; err names the path |
| UT-TokenizeCli-012 | | | model written with model `llama` | exit 1; err `qlora_tokenize: tokenizer: model llama, expected gpt2` |
| UT-TokenizeCli-013 | | | pre `llama3` | exit 1; err `qlora_tokenize: tokenizer: pre llama3, expected qwen35` |
| UT-TokenizeCli-014 | | | merges omitted | exit 1; err `qlora_tokenize: tokenizer: key tokenizer.ggml.merges missing` |
| UT-TokenizeCli-015 | | | `--jsonl`, lines `{"text":"x"}`, `not json` | exit 1; out `1\t120`; err starts with `qlora_tokenize: line 2: not JSON (` |
| UT-TokenizeCli-016 | | | `--jsonl`, `{"text": 5}` | exit 1; err `qlora_tokenize: line 1: no string "text"` |
| UT-TokenizeCli-017 | | | `--jsonl`, `"["text"]"` | exit 1; err `qlora_tokenize: line 1: no string "text"` |
| UT-TokenizeCli-018 | | | a file `ok#$FF` | exit 1; err `qlora_tokenize: invalid UTF-8 at byte 3` |
| UT-TokenizeCli-019 | | | after `SetCurrentDir(Dir)` (restore it in a `finally`): `tiny.gguf`, `a.txt` (bare names) | exit 0; out `259 264 267` |
| UT-TokenizeCli-020 | | | `-x`, FILE (two arguments) | exit 2; err `Usage: qlora_tokenize [--jsonl] MODEL FILE` |
