# ITa cases — unit `DeltaRule` (T08)

| Case ID | Title | Expected (short) |
|---------|-------|------------------|
| ITa-T08-001 | hand case, two tokens (O and State) | limit 0 |
| ITa-T08-002 | hand case with g = ln(1/2) on token 1 | limit 0 |
| ITa-T08-003 | β = 0 at H 48, T 200, 128 × 128: max \|O\| and \|State\| | limit 0 |
| ITa-T08-004 | `RefTokens`, seed 13, H 48, T 200, g in (−0.1, 0] | limit 1e-5 |
| ITa-T08-005 | `RefTokens`, seed 14, H 16, T 200, g in (−5, 0] | limit 1e-5 |
| ITa-T08-006 | `RefTokens`, H 48, T = 1, 63, 64, 65, 200 (T09's lengths) | limit 1e-5 |
| ITa-T08-007 | g = −inf, H 48, T 20: o_t = β_t v_t (k_t · q_t) | limit 1e-6 |

## Command

```bash
<project root>/tests/ita/T08-run.sh
```

## Result file

`<project root>/work/ita/T08/summary.tsv`

Columns: case, OK/NG, value, limit, note; then the driver's exit code and
the count of stderr lines other than the GPU warnings. OK when value ≤ limit.
