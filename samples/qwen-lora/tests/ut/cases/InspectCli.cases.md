# UT cases — unit `InspectCli` (T01c)

Unit under test: `src/InspectCli.pas` —
`function RunInspectCLI(const AArgs: array of string; AOut, AErr: TStrings): Integer`
(all of the logic of `qlora_inspect`; the program body is a thin wrapper,
R20). The UT calls `RunInspectCLI` directly (facts "Unit tests").
Base: `docs/detailed-design/T01-detailed.md` §2.12 and §5 (IDs in the Notes
column), the facts' output/exit-code block.

Test arrangement: test-built small GGUF files (same tiny writer as
GgufFile_test, under `{proj}/work/ut/gguf/`). UT never need the rehearsal model (facts "Unit tests"): the oracle
comparison is ITa-T01-001..005, and UT-012…016 are retired.

| Case ID | Title | Input (short) | Expected result | Notes |
|---|---|---|---|---|
| UT-InspectCli-001 | Valid file: summary lines, exit 0 | test file: 2 KV, tensor f32 [4] (16 B), data section ends exactly at file end; args [FILE] | rc = 0; AOut in order `version`, `alignment`, `kv`, `tensors`, `data_offset`, `data_end`, `file_size`, tab-separated, each with this file's own value; AErr empty | 2.12 step 4; R13, R17 |
| UT-InspectCli-002 | type lines ordered by type id, not file order | tensors in file order: q4_K, f16, f32; args [FILE] | exactly 3 type lines: `type\tf32\t1`, then `type\tf16\t1`, then `type\tq4_K\t1`; one line per type present | 2.12 step 5; R14 |
| UT-InspectCli-003 | No flags → no kv / tensor lines | test file as UT-001; args [FILE] | AOut = summary lines + type lines only; no `kv\t` or `tensor\t` lines | 2.12 steps 6–7 |
| UT-InspectCli-004 | --kv lines: value-type per key, arrays as array<elem>[count] | keys: uint8, string, array\<string\> with 2 elements, array\<int32\> with 3 elements; args ['--kv', FILE] | kv lines in file order: `kv\tK1\tuint8`, `kv\tK2\tstring`, `kv\tK3\tarray<string>[2]`, `kv\tK4\tarray<int32>[3]`; no values printed | 2.12 step 6; R15 |
| UT-InspectCli-005 | --tensors lines: format, dims order, file order | tensors f32 [4] and q4_K [256, 2]; args ['--tensors', FILE] | `tensor\t<name>\t<type name>\t<d1 d2>\t<offset>\t<bytes>` per tensor in file order; dims in GGUF order, single spaces (e.g. `256 2`) | 2.12 step 7; R16 |
| UT-InspectCli-006 | Both flags: kv lines before tensor lines | test file with both keys and tensors; args ['--kv', '--tensors', FILE] | AOut order: summary, type*, kv*, tensor*; rc = 0 if data_end = file_size | 2.12 steps 5–7 |
| UT-InspectCli-007 | Usage error: no FILE | args ['--kv'] | rc = 2; AErr = `Usage: qlora_inspect [--kv] [--tensors] FILE`; AOut empty | 2.12 check 1; R18 |
| UT-InspectCli-008 | Usage error: unrecognized option | args ['--bogus', FILE] | rc = 2; same usage line on stderr; AOut empty | 2.12 check 1; R18 |
| UT-InspectCli-009 | File not valid GGUF: exit 1, error on stderr, nothing on stdout | args [bad-magic file] | rc = 1; AErr = `error: ` + the EGgufError message; AOut empty | 2.12 check 2; R18, R19; §5 (cut / bad file) |
| UT-InspectCli-010 | data_end <> file_size: all lines, then the error, exit 1 | a file passing all Open checks with one trailing byte after the data section; args [FILE] | rc = 1; AOut = all lines (summary, type, kv/tensor per flags); AErr = `error: data ends at <data_end>, file size <file_size>` | 2.12 check 3; R18; §5 (cut one byte short) |
| UT-InspectCli-011 | Zero tensors: data_end = data_offset, exit 0 | valid file with n_tensors = 0, file size = DataOffset; args [FILE] | rc = 0; `data_end` line = the `data_offset` value; no type lines | 2.12 step 3; R17; §5 (zero tensors) |
| UT-InspectCli-012 | Oracle: summary lines equal summary.txt | QLORA_MODEL set; args ['--kv','--tensors', $QLORA_MODEL] | rc = 0; stdout lines 1–12 equal `{proj}/work/oracle/T01/summary.txt`: version 3, alignment 32, kv 46, tensors 320, data_offset 10961920, data_end 532517120, file_size 532517120, type f32 133 / q8_0 36 / q4_K 98 / q5_K 36 / q6_K 17 | facts "Oracle"; R13, R14 (I-5); [廃止] UT never need the model (facts "Unit tests"); the oracle comparison is ITa-T01-001..005 |
| UT-InspectCli-013 | Oracle: kv lines equal kv-types.tsv | same call as UT-012 | the `kv` lines with the leading `kv\t` removed equal `{proj}/work/oracle/T01/kv-types.tsv` byte-for-byte: 46 lines, file order | facts "Oracle"; R15 (I-5); [廃止] UT never need the model (facts "Unit tests"); the oracle comparison is ITa-T01-001..005 |
| UT-InspectCli-014 | Oracle: tensor lines equal tensors.tsv | same call as UT-012 | the `tensor` lines with the leading `tensor\t` removed equal `{proj}/work/oracle/T01/tensors.tsv` byte-for-byte: 320 lines, file order | facts "Oracle"; R16 (I-5); [廃止] UT never need the model (facts "Unit tests"); the oracle comparison is ITa-T01-001..005 |
| UT-InspectCli-015 | Oracle: per-type counts equal the llama-server load log | same call as UT-012 | the type-line counts (f32 133, q8_0 36, q4_K 98, q5_K 36, q6_K 17) equal the `llama_model_loader: - type ...` lines of `{proj}/work/oracle/T01/llama-server-load.log` | facts "Oracle"; R14 (I-5); [廃止] UT never need the model (facts "Unit tests"); the oracle comparison is ITa-T01-001..005 |
| UT-InspectCli-016 | QLORA_MODEL not set → oracle cases are skipped | QLORA_MODEL unset; the UT-012…015 test methods run | each is skipped in consoletestrunner (no assertion, no failure) | facts "Oracle" (tests are skipped without the model); [廃止] UT never need the model (facts "Unit tests"); the oracle comparison is ITa-T01-001..005 |
| UT-InspectCli-017 | Test the CLI through the program binary with real stdout/stderr | (originally planned) args [test file], run `src/qlora_inspect` as a process | output and exit code per a real invocation | [廃止] the logic lives in `InspectCli` and the UT calls `RunInspectCLI` directly (facts "Unit tests", R20); the thin program body is covered by the ITa cases instead |
