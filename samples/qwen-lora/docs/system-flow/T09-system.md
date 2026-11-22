# T09 System Flow: DeltaRuleChunked

## Callers

T10's linear block uses `DeltaRuleChunked` for training and scoring, in the
Qwen fine-tuning pipeline.

## Unit tests

The UT (tests/ut/DeltaRule_test.pas) compares `DeltaRuleChunked` with
`DeltaRuleTokens` (T08):

- The hand cases of T08 come out exactly
- Random inputs at various sizes: difference ≤ 1e-5 (measured ≤ 7.2e-7)
- Strong decay (g down to −5 per token): difference ≤ 1e-5 (measured 4.4e-6)
- Gradients of all five inputs against the token form: ≤ 1e-5 (measured 2.1e-6)

The fixture's scalar reference `RefTokens` validates small cases.

## ITa

The ITa driver `tests/ita/T09/chunked_check.pas` runs the function at the
27B model's sizes (48 heads, 128 × 128):

- Against `DeltaRuleTokens` at T = 1, 63, 64, 65, 200, 2048
- Against the scalar Double reference `RefTokens` at T = 200
- Strong decay, gradients, different chunk sizes

The driver runs on the CPU (GPUs hidden), from a fresh directory outside the
project. It writes a TSV summary with case ID, OK/NG, measured value, limit,
and notes.
