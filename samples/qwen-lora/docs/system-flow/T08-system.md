# System flow — unit `DeltaRule` (T08)

## Unit summary

The unit `DeltaRule` implements the gated delta rule token by token.
It provides two functions: `DeltaStep` for a single token step, and
`DeltaRuleTokens` for processing a whole sequence.

## Callers

### T10 (Linear block)

The linear block of the Gated DeltaNet (T10) calls `DeltaRuleTokens`
per sequence. Before calling:

- The query is scaled by 1/sqrt(Dk) (DESIGN 8.10, step 6).
- Key heads are already tiled to the value heads (DESIGN 8.10, step 5).

### T09 (Chunked form)

T09 implements the chunked form of the gated delta rule. Its output is
checked against this reference `DeltaRuleTokens` implementation.

## Testing relationships

### Unit tests

The UT compares the unit's outputs with:

1. The fixture's scalar reference `RefTokens` in Double precision (oracle for
   random inputs).
2. Hand-computed values from the hand case (exact fp32 values).

The fixture `tests/ut/DeltaRuleFixture.pas` provides:
- `RandomInputs`: seeded random inputs for various sizes.
- `RefTokens`: the rule computed in Double, independent of the unit.

### ITa (integration test)

The ITa driver `tests/ita/T08/deltarule_check.pas` runs the unit at the
27B model's sizes:
- H = 48 value heads
- Dk = Dv = 128
- T = 200 tokens

It compares with the fixture's `RefTokens` and the hand values. Both are
outside the unit (double-checked by an independent oracle).

Measured performance: about 50 ms for H=48, T=200, Dk=Dv=128 on CPU.
Measured accuracy: 4.8e-7 maximum difference from the Double reference.
