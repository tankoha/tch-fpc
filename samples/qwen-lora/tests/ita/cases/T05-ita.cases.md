# ITa cases — T05 Qwen Tokenizer

| Case ID | Title | Expected (short) |
|---|---|---|
| ITa-T05-001 | Token ids = `/tokenize`: Pascal files of five dialects | OK; 75 lines match oracle |
| ITa-T05-002 | English paragraphs (DESIGN.md) | OK; 402 lines match oracle |
| ITa-T05-003 | Japanese paragraphs (DESIGN_jp.md) | OK; 402 lines match oracle |
| ITa-T05-004 | Special tokens alone, adjacent, in text, chat | OK; 68 lines match oracle |
| ITa-T05-005 | Boundary lines | OK; 31 lines match oracle |
| ITa-T05-006 | Random strings (fixed seed) | OK; 2000 lines match oracle |
| ITa-T05-007 | `Decode(Encode(text)) = text` for every record | OK; 2978 lines roundtrip |
| ITa-T05-008 | Plain mode on one whole source file (absolute path) | OK; 1 line matches oracle |

## Test Execution

```bash
QLORA_MODEL=<absolute path of the rehearsal GGUF> <project root>/tests/ita/T05-run.sh
```

Result file: `<project root>/work/ita/T05/summary.tsv`

The test runner:
- Builds `src/qlora_tokenize.pas` and `tests/ita/T05/roundtrip_check.pas`
- Runs them from a fresh directory outside the project
- Compares output with the oracle at `work/oracle/T05/`
- Writes summary.tsv with columns: case, OK/NG, diff lines, lines ours, lines oracle, note

**Execution time**: ~5 seconds
