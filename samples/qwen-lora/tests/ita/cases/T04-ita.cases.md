| Case ID | Title | Expected (short) |
|---|---|---|
| ITa-T04-001 | identity copy of the model: sha256 and size | match source sha256 and size |
| ITa-T04-002 | planned layout of the copy equals upstream | 322 lines, no diff |
| ITa-T04-003 | identity copy of the projector: sha256 and size | match source sha256 and size |
| ITa-T04-004 | planned layout after retyping 78 tensors equals upstream | 322 lines, no diff |
| ITa-T04-005 | retyped file: sha256 and size | match oracle sha256 and size |
| ITa-T04-006 | upstream reads the retyped file back with that layout | 322 lines, no diff |

---

Run all cases with:

```bash
QLORA_MODEL=<absolute path of the rehearsal GGUF> <project root>/tests/ita/T04-run.sh
```

Result file:

`<project root>/work/ita/T04/summary.tsv`
