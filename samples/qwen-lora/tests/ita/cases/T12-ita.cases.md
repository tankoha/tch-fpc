# ITa cases — T12 (Lora)

| Case ID | Title | Expected (short) |
|---|---|---|
| ITa-T12-001 | AttachLora on the rehearsal model: pairs, their dimensions and ParameterCount against the tensor directory | 0 mismatches |
| ITa-T12-002 | B = 0: LastLogProbs with the adapters against a model without them, prompts pas-001, ja-065, chat-130 of work/oracle/T11 | 0 (exact) |
| ITa-T12-003 | B random (0.02 N(0, 1)): LastLogProbs with the adapters against the base forward on a GGUF whose target matrices are W + s·B·A, computed in Double from T02's scalar dequantisation, same prompts | ≤ 1e-3 |
| ITa-T12-004 | chat-065, B random: gradients of every A and B, block-wise (Recompute) against one backward through the whole graph, relative to the largest gradient of each tensor | ≤ 1e-4 |
| ITa-T12-005 | Save, then Load into a fresh set on a fresh model: every A and B | 0 (exact) |

**Command:**

```
QLORA_MODEL=<rehearsal GGUF> <project root>/tests/ita/T12-run.sh
```

**Result file:**

```
<project root>/work/ita/T12/summary.tsv
```
