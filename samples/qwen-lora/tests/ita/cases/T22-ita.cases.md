# T22 ITa Test Cases

| Case ID | Title | Expected (short) |
|---------|-------|------------------|
| ITa-T22-001 | 11 Oxygene files: outcome as planted, class compiled-run | ≤ 0 mismatches |
| ITa-T22-002 | Details free of home and work directory paths | ≤ 0 lines |
| ITa-T22-003 | Oxygene work directories left under ~/.cache/qlora-verify | 0 |
| ITa-T22-004 | Seconds per Oxygene file (build and run) | ≤ 5 s |
| ITa-T22-005 | T19 ITa with this src/: NG cases | 0 (6 OK expected) |
| ITa-T22-006 | T19's x-oxygene compiled, run and rejected (reason wrong-output) | 0 |

## Command

```sh
QLORA_MODEL=<rehearsal GGUF> <project root>/tests/ita/T22-run.sh
```

## Result File

`<project root>/work/ita/T22/summary.tsv`

The summary contains: case, OK/NG, value, limit, note; followed by exit codes
and the count of stderr lines other than GPU warnings. OK when value ≤ limit.

`qlora_verify` exits 1 in this run (planted faults); that is expected.
