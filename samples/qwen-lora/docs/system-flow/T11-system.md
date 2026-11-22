# T11 System Flow — QwenModel, LogprobsCli, qlora_logprobs

## Callers

### Training (T17)

T17's training runs call:

- `TQwenModel.Embed` to embed prompt tokens.
- `TQwenModel.Stages` to get the trunk blocks as T13 `Recompute` stages.
- `TQwenModel.Logits` to project the final hidden state to vocabulary
  logits.

The stages are passed to `RecomputeForward` (T13) for gradient
checkpointing during the forward pass.

### qlora_logprobs (H43, M4 on 27B)

`qlora_logprobs` is the acceptance tool for H43 (rehearsal model) and will
be used at milestone M4 on the 27B model. It reads prompts from a JSONL
file, computes next-token log-probabilities, and outputs the top-N tokens
with their log-probabilities.

The ITa driver `tests/ita/T11/logprobs_check.pas` compares
`qlora_logprobs` output against the reference in `work/oracle/T11`.

## Unit Tests

### QwenModel_test.pas

The UT compares `TQwenModel` against the scalar reference
`RefLogits` from `tests/ut/QwenRefScalar.pas` (T10). The test fixtures are
tiny models written by `tests/ut/QwenBlocksFixture.pas`:

- `TiedPath`: 2 blocks, tied head (`token_embd.weight` only), vocabulary
  274.
- `OwnPath`: 2 blocks, separate `output.weight`.
- `NextPath`: 2 trunk blocks + MTP flag (`qwen35.nextn_predict_layers = 1`,
  `block_count = 3`), tied head.
- `NonePath`: blocks only (no embedding or head).
- `BadEmbdPath`: malformed `token_embd.weight` (wrong columns).
- `BadOutPath`: malformed `output.weight` (wrong rows).

The scalar oracle computes the forward pass row by row in Double:

1. Embedding rows copied and dequantised.
2. Trunk with `RefBlock` (T10).
3. RMS norm with `output_norm`.
4. Matrix multiplication with `output.weight` or `token_embd`.

### LogprobsCli_test.pas

The UT exercises `RunLogprobsCLI` with:

- Usage errors (exit 2): wrong arguments, out-of-range values, unknown
  options.
- Runtime errors (exit 1): invalid JSON, wrong record shape, out-of-range
  token ids, missing files.
- Successful runs on `TinyPath` and `PromptPath`, comparing output lines
  with `TopLine` of `LastLogProbs`.

The fixture `PromptPath` has four lines: a token-id prompt with id, one
without, a blank line, and a text prompt.

## ITa

`tests/ita/T11-run.sh` builds and runs `qlora_logprobs` on the rehearsal
GGUF against the oracle in `work/oracle/T11`:

- 24 token-id prompts of 1–600 tokens.
- 3 text prompts (tokenised through `QwenTokenizer`).

The driver `logprobs_check.pas` compares the output with `expected.jsonl`
and writes `summary.tsv`. The cases are:

- ITa-T11-001: top-1 token match count (limit 0 differences).
- ITa-T11-002: max |logprob - ref| over top-10, token prompts (limit 0.02).
- ITa-T11-003: text prompts, same check (limit 0.02).
- ITa-T11-004: `RecomputeForward` vs `Trunk`, chat-130 prompt (limit 0,
  exact match).
