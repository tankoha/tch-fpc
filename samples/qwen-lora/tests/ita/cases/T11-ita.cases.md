# ITa cases — T11 (qlora_logprobs)

| Case ID | Title | Expected (short) |
|---|---|---|
| ITa-T11-001 | Top-1 token against the reference, 24 token prompts of 1–600 tokens | 0 differences |
| ITa-T11-002 | Max \|logprob − ref\| over the reference's top 10, token prompts | ≤ 0.02 |
| ITa-T11-003 | 3 text prompts through QwenTokenizer: same length and top-1, max \|logprob − ref\| over the top 10 | ≤ 0.02 |
| ITa-T11-004 | Trunk through RecomputeForward(M.Stages, ..) against M.Trunk, chat-130 | 0 (exact) |

**Command:**

```
QLORA_MODEL=<rehearsal GGUF> <project root>/tests/ita/T11-run.sh
```

**Result file:**

```
<project root>/work/ita/T11/summary.tsv
```
