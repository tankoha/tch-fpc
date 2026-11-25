# T21 ITa Cases

実行コマンド:
```
QLORA_MODEL=<rehearsal GGUF> <project root>/tests/ita/T21-run.sh
```

結果ファイル: `<project root>/work/ita/T21/summary.tsv`

| Case ID | Title | Expected (short) |
|---|---|---|
| ITa-T21-001 | Default kind logprobs and update lines | Byte-identical to code before T21 (count of differences ≤ 0) |
| ITa-T21-002 | --compute bf16 logprobs | Largest ∥logprob difference∥ ≤ 0.25 |
| ITa-T21-003 | --compute bf16 top-1 tokens | Prompts with different top-1 token where fp32 margin > 0.5: count ≤ 0 |
| ITa-T21-004 | --compute bf16 training | Largest relative loss difference ≤ 0.02 |
| ITa-T21-005 | fp32 then --resume with bf16 | First 2 fp32 identical, rest within 0.02: count of violations ≤ 0 |
| ITa-T21-006 | --compute bf16 twice | Logprobs byte-identical (1 if not, ≤ 0) |
