# UT cases — unit `TrainStep` (T17)

| Case ID | Title | Target function | Input (short) | Expected result | Notes |
|---|---|---|---|---|---|
| UT-TrainStep-001 | sample loss without gradients | SampleLoss | `SampleLoss(M, FixSample, 3)` | Count 7; Sum = `RefLoss(M, FixSample)` rel 1e-5 | |
| UT-TrainStep-002 | backward: loss and grads against the reference | SampleLossBackward | Want := `RefBackward(M, S, FixSample, 3, Ref)`; `SampleLossBackward(M, FixSample, 3, 3)` | Count 7; Sum = Ref rel 1e-5; `RelDiff(Grads(S), Want)` ≤ 1e-5; the B grads of Want not zero (max > 0.1) | |
| UT-TrainStep-003 | backward: slice size 1, 3, 7, 100 | SampleLossBackward | Want := `RefBackward(.., 1, ..)`; slices 1, 3, 7, 100 (grads zeroed between) | for each: Sum = Ref rel 1e-5; grads within 1e-5 of Want | |
| UT-TrainStep-004 | denominator scaling and accumulation | SampleLossBackward | denominator 3; then 6; then 6 again without zeroing | grads at 6 × 2 within 1e-6 of those at 3; after the second call at 6 the grads equal those at 3 (rel 1e-6) | |
| UT-TrainStep-005 | adapters with B zero: same loss as without adapters | SampleLoss | S := `NewSet(M, -1)` (B zero); M0 a second model without adapters | `SampleLoss(M0, ..).Sum` = `SampleLoss(M, ..).Sum` exact | |
| UT-TrainStep-006 | SampleLoss defines no grads | SampleLoss | `SampleLoss(M, FixSample, 3)` | no A or B grad defined afterwards | |
| UT-TrainStep-007 | ETrainStep messages: slice, denominator, loss span | SampleLoss, SampleLossBackward | `CheckStep` without a set: slice 0 (both); denominator 0, −2; LossFrom 0; LossTo 12; LossTo 21 | `slice must be positive, got 0` (twice); `denominator must be positive, got 0`, `.. got -2`; `x: loss span 0..19 outside 1..20`; `x: loss span 12..12 outside 1..20`; `x: loss span 12..21 outside 1..20` | |
| UT-TrainStep-008 | ValidationLoss over a set; empty set NaN | ValidationLoss | a TTrainSet (the tiny model's tokenizer) of three records (`hello`/`world`, `ab`/`abc abc`, `x`/`y`, class `c`); `ValidationLoss(M, V, 2)`; then an empty file | Count 3; = Σ `SampleLoss(.., 256)`.Sum / Σ Count, rel 1e-5; empty: NaN | |
| UT-TrainStep-009 | backward with the loss span at the last token | SampleLossBackward | FixSample with span 19..20 (the last token): `RefBackward(.., 1, ..)`, `SampleLossBackward(.., 1, 256)` | Count 1; Sum = Ref rel 1e-5; grads within 1e-5 | |
