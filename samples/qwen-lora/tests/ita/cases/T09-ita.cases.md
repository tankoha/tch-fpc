# ITa cases — T09 DeltaRuleChunked

| Case ID | Title | Expected (short) |
|---|---|---|
| ITa-T09-001 | O and State vs DeltaRuleTokens, H 48, 128×128, T = 1, 63, 64, 65, 200 | ≤ 1e-5 |
| ITa-T09-002 | vs scalar Double reference RefTokens, H 48, T 200 | ≤ 1e-5 |
| ITa-T09-003 | strong decay, g in (−5, 0], H 16, T 200 | ≤ 1e-5 |
| ITa-T09-004 | training length T 2048, H 48 (time in note) | ≤ 1e-5 |
| ITa-T09-005 | gradients of all five inputs vs token form | ≤ 1e-5 |
| ITa-T09-006 | chunk sizes 1, 2, 3, 5, 7, 10, 49, 50, 51, 64 at T 50 | ≤ 1e-5 |

Command: `<project root>/tests/ita/T09-run.sh`

Result file: `<project root>/work/ita/T09/summary.tsv`
