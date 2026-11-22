# T11 External Design — QwenModel, LogprobsCli, qlora_logprobs

## Purpose

T11 provides the whole `qwen35` model for logits and a CLI to compute
next-token log-probabilities. It comprises:

- Unit `QwenModel`: token embedding, trunk blocks as T13 `Recompute` stages
  split over up to two devices, the final RMS norm, and the head (the
  output projection to vocabulary logits).
- Unit `LogprobsCli`: the command-line behaviour of `qlora_logprobs`.
- Program `qlora_logprobs`: a thin wrapper that calls `RunLogprobsCLI` and
  exits with its code.

The same code runs the rehearsal 0.8B model (24 trunk blocks, vocabulary
248,320, tied head) and the 27B model (65 blocks of which the last is the
MTP block, with its own `output.weight`).

## Forward Pass

The forward pass computes logits from token ids:

1. **Embed**: Look up the embedding rows for the input token ids from
   `token_embd.weight` (packed, dequantised per row) on device 0.
2. **Trunk**: Pass the embedded sequence through the trunk blocks, each
   implementing attention and FFN with gated linear units. The trunk is
   exposed both as a monolithic `Trunk` function and as `Stages` (an array
   of T13 `Recompute` stage interfaces) for gradient checkpointing.
3. **Norm**: Apply RMS norm with `output_norm.weight` on device 1.
4. **Head**: Multiply by the transposed head matrix (`output.weight` or
   `token_embd.weight` if tied) to get logits over the vocabulary.

The head is chunked (8,192 rows by default) to bound memory usage.

## Tied Head

When `output.weight` is absent, the head is tied to the embedding
(`token_embd.weight`). The rehearsal model uses a tied head; the 27B has a
separate `output.weight`.

## Trunk Block Count

The GGUF file may carry `qwen35.nextn_predict_layers` (the MTP block). The
trunk excludes this block:

```
TrunkBlocks = block_count - qwen35.nextn_predict_layers (if present)
```

For the rehearsal model: 24 blocks, no MTP → TrunkBlocks = 24.
For the 27B: 65 blocks, MTP = 1 → TrunkBlocks = 64.

## Device Split

The model is split over two devices at a given block index:

- Blocks 0 to Split-1 run on device 0.
- Blocks Split to TrunkBlocks-1 run on device 1.

Split = 0 places all blocks on device 1; Split = TrunkBlocks places all on
device 0.

## Command Line

```
qlora_logprobs [--top N] [--split S] MODEL FILE
```

- `--top N`: Number of top log-probabilities to output (1–100, default 20).
- `--split S`: Split the trunk at block S over cuda:0 / cuda:1 (0–1000).
  Without `--split`, the model runs on the CPU.
- `MODEL`: Path to the GGUF model file.
- `FILE`: Path to a JSONL file with prompts (see below).

## Input Format

`FILE` is a JSONL file (one JSON object per line). Each object specifies a
prompt in one of two forms:

1. Token ids: `{"id": "...", "tokens": [5, 17, 2]}`
2. Text: `{"id": "...", "text": "hello world"}`

- `id` is optional (defaults to the line number).
- Blank lines are skipped.
- Text prompts are tokenised with `QwenTokenizer` from the model file.

## Output Format

For each prompt, one JSON line to stdout:

```json
{"id":"<id>","n":<token count>,"top":[{"id":<token>,logprob":<value>},...]}
```

The `top` array has `N` entries (from `--top`), sorted by descending
log-probability.

## Exit Codes

- 0: Success (all prompts processed).
- 1: Runtime error (a line has invalid JSON, missing keys, an out-of-range
  token id, the model file is not found, etc.). The error message goes to
  stderr.
- 2: Usage error (wrong number of arguments, unknown option, `--top` or
  `--split` out of range). Two lines to stderr: the reason and the usage.

## Errors

`EQwenModel` is raised by `QwenModel`:

- `split <S> outside 0..<TrunkBlocks>`
- `token_embd.weight must be [vocab, hidden]`
- `output.weight must be [vocab, hidden]` (when present and wrong shape)
- `head chunk must be positive, got <value>`
- `no token ids`
- `token id <id> outside 0..<vocab-1>`
- `H must be [T, <hidden>]` (wrong input shape to `Logits`)

`RunLogprobsCLI` usage errors (exit 2):

- `--top needs a value`
- `--split needs a value`
- `--top must be 1..100, got <text>`
- `--split must be 0..1000, got <text>`
- `unknown option <arg>`
- `MODEL and FILE expected`

Other errors (exit 1), prefixed with `qlora_logprobs: `:

- `line <N>: not JSON (<parser message>)`
- `line <N>: not an object`
- `line <N>: needs "tokens" (array) or "text" (string)`
- Plus messages from `TGgufFile`, `TQwenModel`, and libtorch.

## H43 — Acceptance

H43 is the acceptance criterion for T11. A prototype measured it against
the bundled `llama-server`:

- Reference: `llama-server /completion` with token-id prompts, 24 prompts
  of 1–600 tokens and 3 text prompts.
- The reference runs on an **all-F32 copy** of the rehearsal GGUF (every
  tensor dequantised) with `-ctk f32 -ctv f32 -fa off` and an **f32 KV
  cache**.
- On the original Q4_K_M file, `llama-server` quantises activations to Q8_K
  before K-quant products, causing differences up to 1.8 in log-probability.
- With the F32 copy and f32 KV cache: **max difference 1.3e-3**, top-1
  match 24/24 on token prompts, 1.4e-3 on text prompts.

The Pascal forward reads the same Q4_K_M file (its dequantisation is exact)
and computes in fp32, matching the reference within 1.3e-3.
