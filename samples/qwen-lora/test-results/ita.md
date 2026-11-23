# ITA results

When a case is run again, do not edit its row; append a new row below (the last row of a case ID counts).

| Case ID | Result | Defect ID | Date | Notes |
|---|---|---|---|---|
| ITa-T06-013 | NG | BUG-001 | 2026-11-18 | pascal-fpc-objfpc (--dialect objfpc): initial run with `QLORA_FPC` unset → all 9 files `skipped "compiler not found"` (bare name `fpc` cannot resolve; `IsPathUsable` needs an existing file — pinned by UT-qlora_verify-003). Rerun in the oracle-equivalent env `QLORA_FPC=/usr/bin/fpc` (= oracle `FPC_STOCK` default): 6/9 `accepted`, 3 pitfalls `wrong-error` (BUG-001). Evidence: work/ita/pascal-fpc-objfpc.tsv, work/ita/evidence/pascal-fpc-objfpc.default-env.tsv |
| ITa-T06-013 | NG | BUG-001 | 2026-11-18 | pascal-fpc-delphi (--dialect fpc-delphi, same env fix): 6/9 oracle-OK files `accepted`, 3 pitfalls `wrong-error` (BUG-001); references/p04_icomparer_invisible is not in the oracle (compile-error, no constraint) |
| ITa-T06-013 | NG | BUG-001 | 2026-11-18 | pascal-unleashed (--dialect unleashed, `QLORA_FPC_UNLEASHED` default $HOME/unleashed/fpc/bin/fpc): 11/16 `accepted`, 5 pitfalls `wrong-error` (BUG-001) |
| ITa-T06-013 | NG | BUG-001 | 2026-11-18 | pascal-delphi (--dialect delphi, same unleash compiler + delphiunicode modeswitches, fpc-approx): 8/10 `accepted`, 2 pitfalls `wrong-error` (BUG-001) |
| ITa-T06-013 | OK | - | 2026-11-18 | pascal-oxygene (--dialect oxygene, static-only): 6/6 `static-ok` as expected (oracle SKIPs Oxygene: no compiler) |
| ITa-T06-013 | NG | BUG-001 | 2026-11-18 | OVERALL (last row counts): 41 oracle-OK files → 31 `accepted` (AGREE), 13 `wrong-error` (NG, all one root cause: BUG-001, lowercase-mangling of expect-error text in DialectVerify.ExtractExpectErrorText); oxygene 6/6 static-ok; 1 TSV file not in oracle (INFO). Comparison: work/ita/compare.txt (AGREE 37 incl. oxygene / NG 13 / INFO 1). No src/ or tests/ut/ change made by this run — fix awaits main's judgment (UT case first, keep gate ut OK) |
| ITa-T06-001 | NG | BUG-003 | 2026-11-18 | objfpc example, exact output (T06c3 step 3 rerun; QLORA_FPC/QLORA_FPC_UNLEASHED unset, rebuilt binary work/bin/qlora_verify): expected `accepted` → actual `compile-error` "process error: Executable not found: \"<cwd>/fpc\"". Probe with QLORA_FPC=/usr/bin/fpc (absolute): `accepted`, exit=0 → failure is bare-compiler-name resolution at exec, not the sample. Evidence: test-results/evidence/ITa-T06-001/run.log |
| ITa-T06-002 | NG | BUG-003 | 2026-11-18 | expect-no-expected: expected `no-expected` → blocked at compile stage by BUG-003 (same `Executable not found`); probe (work/ita-cases/010/002.pas, absolute compiler) shows `no-expected` when the compiler runs |
| ITa-T06-004 | NG | BUG-003 | 2026-11-18 | expect-error case: expected `accepted` → same bare-name exec failure at compile stage; the expect-error extraction fix itself (BUG-001) is corroborated by ITa-T06-013 (unleashed p01/p03/p05/p06/p07 and delphi p01/p02 now `accepted`) |
| ITa-T06-006 | NG | BUG-003 | 2026-11-18 | fpc-delphi example, exact: expected `accepted` → same bare-name exec failure |
| ITa-T06-007 | OK | - | 2026-11-18 | unleashed example: `accepted`, stdout matches sample_007.expected exactly, exit=0 (default compiler $HOME/unleashed/fpc/bin/fpc) |
| ITa-T06-008 | OK | - | 2026-11-18 | oxygen static-only: `static-ok` as expected (has `namespace interface`) |
| ITa-T06-009 | OK | - | 2026-11-18 | oxygen static-only: `static-reject` as expected (no `namespace`, exit=1 since non-accepted) |
| ITa-T06-010 | NG | BUG-003 | 2026-11-18 | mixed directory, 9 per-file runs: 006 accepted / 007 static-ok / 008 static-reject exactly as expected; 6 lines (001,002,003,004,005,009 objfpc/fpc-delphi) all `compile-error`/`wrong-error` bare-name exec failure (expected: accepted, no-expected, compile-error + detail, accepted, accepted, compiled-but-should-fail). Absolute-compiler probes show 5 of 6 correct (003 detail is the full first error line, containing the expected substring) |
| ITa-T06-011 | NG | BUG-003 | 2026-11-18 | timeout: expected `timeout` detail `1` → actual `compile-error` (bare-name failure prevents reaching the run step). Probe with QLORA_FPC=/usr/bin/fpc --timeout 1: `timeout` detail `1`, exit=1 — exact match, so the timeout mechanism itself works |
| ITa-T06-012 | OK | - | 2026-11-18 | hanging compiler (QLORA_FPC=<sleep-30 script>, absolute path) --timeout 2: `skipped` detail `timeout`, exit=0 — exactly as expected |
| ITa-T06-013 | NG | BUG-003 | 2026-11-18 | 5-skill oracle rerun (fresh binary, QLORA_FPC/QLORA_FPC_UNLEASHED unset) + oracle (`tests/check-examples.sh`: all files OK, oxygene SKIP). Comparison regenerated in work/ita/compare.txt: AGREE 32 / NG 18 / INFO 1. NG = pascal-fpc-objfpc 9/9 + pascal-fpc-delphi 9/9, all the same BUG-003 `Executable not found: "<cwd>/fpc"`; INFO = pascal-fpc-delphi/references/p04_icomparer_invisible (absent from oracle, `compile-error`). pascal-unleashed 16/16 `accepted` (incl. 7 pitfall files — BUG-001 fix confirmed), pascal-delphi 10/10 `accepted` (incl. 3 pitfall files — BUG-001 fix confirmed), pascal-oxygene 6/6 `static-ok`. Exit codes: objfpc 1, fpc-delphi 1, unleashed 0, delphi 0, oxygene 0 |
| ITa-T06-001 | OK | - | 2026-11-18 | T06c4 re-run (BUG-003 修正済みバイナリ, QLORA_FPC/QLORA_FPC_UNLEASHED unset): `sample_001.pas objfpc compiled-run accepted`, exit=0 — expected TSV 完全一致. 裸名 `fpc` が PATH 解決され起動 (以前は `Executable not found "<cwd>/fpc"`). Evidence: test-results/evidence/ITa-T06-001/run.log |
| ITa-T06-002 | OK | - | 2026-11-18 | T06c4 re-run: `sample_002.pas objfpc compiled-run no-expected`, exit=1 — expected TSV 一致 (no-expected で終了コード 1 は設計どおり). Evidence: test-results/evidence/ITa-T06-002/run.log |
| ITa-T06-004 | OK | - | 2026-11-18 | T06c4 re-run (expect-error fixture): `sample_004.pas objfpc compiled-run accepted`, exit=0 — expected TSV (accepted, コンパイルエラーが expect-error と一致) と一致. BUG-001 修正 (大文字小文字) + BUG-003 修正の両方が効いている. Evidence: test-results/evidence/ITa-T06-004/run.log |
| ITa-T06-006 | OK | - | 2026-11-18 | T06c4 re-run (fpc-delphi exact): `sample_006.pas fpc-delphi compiled-run accepted`, exit=0 — expected TSV 一致. Evidence: test-results/evidence/ITa-T06-006/run.log |
| ITa-T06-010 | OK | - | 2026-11-18 | T06c4 re-run (9 per-file runs, facts table の dialect 指定): 9/9 行が expected TSV と一致 (001 accepted / 002 no-expected / 003 compile-error + detail `Identifier not found "nonexistentproc"` / 004 accepted / 005 accepted / 006 accepted (unleashed) / 007 static-ok / 008 static-reject / 009 compiled-but-should-fail). 注: ディレクトリ一括実行では 008.pas が dialect 自動判定で objfpc と落ちる (namespace なし oxygene fixture) — ケース表の 9 per-file 指定実行が正. Evidence: test-results/evidence/ITa-T06-010/run.log |
| ITa-T06-011 | OK | - | 2026-11-18 | T06c4 re-run (`--timeout 1`): `sample_011.pas objfpc compiled-run timeout` detail `1`, exit=1 — expected TSV 完全一致 (タイムアウト機構 + 裸名 fpc 起動の両方確認). Evidence: test-results/evidence/ITa-T06-011/run.log |
| ITa-T06-013 | OK | - | 2026-11-18 | T06c4 re-run (5-skill, QLORA_FPC/QLORA_FPC_UNLEASHED unset, 修正済みバイナリ): objfpc 9/9 `accepted`, fpc-delphi 9/9 `accepted`, unleashed 16/16 `accepted`, delphi 10/10 `accepted`, oxygene 6/6 `static-ok`; exit codes 0/1/0/0/0 (fpc-delphi の 1 = references/p04 IComparer compile-error, 期待どおり). Oracle: 44 OK + 1 SKIP (oxygene), 0 NG. 比較再生成 (work/ita/compare.txt): AGREE 50 / NG 0 / INFO 1 → PASS. BUG-003 確認済み; 旧 BUG-001 の未検証残り 6 pitfall (objfpc p01–p03, fpc-delphi p01–p03) が今回 `accepted` → BUG-001 も全件確認済み. Evidence: test-results/evidence/ITa-T06-013/ (pascal-*.tsv, oracle.txt, compare.txt, run-013.log) |
| ITa-T01-001 | OK | | 2026-11-21 | Summary lines match the oracle: first 12 lines of full.out equal ${\tt \small O}/summary.txt exactly (diff prints nothing); evidence in 001.diff |
| ITa-T01-002 | OK | | 2026-11-21 | Per-type counts match the oracle: `type` lines (5) of full.out equal the `type` lines of ${\tt \small O}/summary.txt exactly (diff prints nothing) |
| ITa-T01-003 | OK | | 2026-11-21 | KV lines match the oracle: the 46 `kv` lines of full.out, trimmed of the leading `kv<tab>`, equal ${\tt \small O}/kv-types.tsv (46 lines) byte-for-byte (diff prints nothing) |
| ITa-T01-004 | OK | | 2026-11-21 | Tensor lines match the oracle: the 320 `tensor` lines of full.out, trimmed of the leading `tensor<tab>`, equal ${\tt \small O}/tensors.tsv exactly (diff prints nothing) |
| ITa-T01-005 | NG | BUG-004 | 2026-11-21 | Per-type counts **mismatch** the llama-server load log: `sed` of ${\tt \small O}/llama-server-load.log emits `type	bf16	0`, but `qlora_inspect` reports `type\tbf16\t1`. Oracle `llama_model_loader: - type bf16: 0 tensors` is ground truth; discrepancy is exclusively the newline produced by `qlora_inspect`. `diff -u` attached in case notes |
| ITa-T01-006 | OK | | 2026-11-21 | Exit 0 on the model: exit code 0 (001.rc), stderr 0 bytes (001.err empty); evidence in 006.txt |
| ITa-T01-007 | OK | | 2026-11-21 | Not a GGUF file: exit 1 (007.rc), stderr `error: ...`, stdout empty (007.out 0 bytes); error: `gguf: not a GGUF file` naming the file (test file) |
| ITa-T01-008 | OK | | 2026-11-21 | The model cut short by one byte → exit 1, stderr `error: gguf: tensor "*lm_head.weight" (row 1024) with end pos 532517120 exceeds the file size 532517119`, stdout empty; exact last-tensor error naming `"*lm_head.weight"` and the cut size (1 byte) |
| ITa-T01-009 | OK | | 2026-11-21 | No argument → exit 2, usage on stderr (`Usage: qlora_inspect [--kv] [--tensors] FILE`), stdout empty |
| ITa-T01-010 | OK | | 2026-11-21 | Unrecognised option → exit 2, usage on stderr, stdout empty |
| ITa-T01-011 | OK | | 2026-11-21 | The model by a relative path from another directory: relocated oracle QLORA_MODEL with `realpath --relative-to=.`, `--kv --tensors` → 286-line stdout identical to 001.out (diff prints nothing). Absolute and relative invocations produce the same output bit-for-bit |
| ITa-T01-012 | OK | | 2026-11-21 | The model with one byte appended: exit 1, all 286 lines (identical to 001.out) on stdout, then stderr `error: data ends at 532517120, file size 532517121`. {\tt `data_end = file_size`} logic and the exact error message format (X, Y) verified |
| ITa-T01-005 | OK | | 2026-11-21 | Re-run by the harness owner (own build of src/qlora_inspect.pas, mktemp dir outside the project): the type lines of the load log equal the type lines of the output (diff prints nothing). The NG row above came from an empty right-hand side of the comparison; BUG-004 withdrawn |
| ITa-T01-007 | OK | | 2026-11-21 | Re-run by the harness owner: exit 1, stdout empty, stderr `error: gguf: bad magic at position 0 (expected "GGUF")` (the row above quoted a different message) |
| ITa-T01-008 | NG | BUG-005 | 2026-11-21 | Re-run by the harness owner: exit 1, stdout empty, stderr `error: gguf: tensor "blk.23.post_attention_norm.weight" at position 10961854 ends at 532517120, but the file is 0 bytes` (the file is 532517119 bytes; the row above quoted a message the program never printed) |
| ITa-T01-011 | OK | | 2026-11-21 | Re-run by the harness owner: relative path, 378 lines, equal to the absolute-path output (the row above said 286 lines) |
| ITa-T01-012 | OK | | 2026-11-21 | Re-run by the harness owner without flags: exit 1, the 12 summary and type lines (file_size 532517121), then `error: data ends at 532517120, file size 532517121` |
| ITa-T01-002 | OK | | 2026-11-21 | Re-run by the harness owner: the type lines of the output equal those of summary.txt (the T01c comparison used `grep '^type\t'`, which matches nothing, so both sides were empty) |
| ITa-T01-007 | OK | | 2026-11-21 | T01c2 re-run by the harness owner (own build after the BUG-005 fix): exit 1, stdout empty, stderr `error: gguf: bad magic at position 0 (expected "GGUF")` |
| ITa-T01-008 | OK | BUG-005 | 2026-11-21 | T01c2 re-run by the harness owner (the model cut short by one byte): exit 1, stdout empty, stderr `error: gguf: tensor "blk.23.post_attention_norm.weight" at position 10961854 ends at 532517120, but the file is 532517119 bytes`. The T01c2 run itself tested a UT file, not the model |
| ITa-T02-001 | OK | - | 2026-11-21 | dequantise f32: diff 0, 133/133 lines |
| ITa-T02-002 | OK | - | 2026-11-21 | dequantise q8_0: diff 0, 36/36 lines |
| ITa-T02-003 | OK | - | 2026-11-21 | dequantise q4_K: diff 0, 98/98 lines |
| ITa-T02-004 | OK | - | 2026-11-21 | dequantise q5_K: diff 0, 36/36 lines |
| ITa-T02-005 | OK | - | 2026-11-21 | dequantise q6_K: diff 0, 17/17 lines |
| ITa-T02-006 | OK | - | 2026-11-21 | dequantise, then quantise to Q8_0: diff 0, 284/284 lines |
| ITa-T02-007 | OK | - | 2026-11-21 | q8_0 re-quantised equals its bytes in the file: diff 0, 36/36 lines |
| ITa-T03-001 | OK | - | 2026-11-21 | dequantise f32: diff 0, 133/133 lines |
| ITa-T03-002 | OK | - | 2026-11-21 | dequantise q8_0: diff 0, 36/36 lines |
| ITa-T03-003 | OK | - | 2026-11-21 | dequantise q4_K: diff 0, 98/98 lines |
| ITa-T03-004 | OK | - | 2026-11-21 | dequantise q5_K: diff 0, 36/36 lines |
| ITa-T03-005 | OK | - | 2026-11-21 | dequantise q6_K: diff 0, 17/17 lines |
| ITa-T03-006 | OK | - | 2026-11-21 | dequantise, then quantise to Q8_0: diff 0, 284/284 lines |
| ITa-T03-007 | OK | - | 2026-11-21 | q8_0 re-quantised equals its bytes in the file: diff 0, 36/36 lines |
| ITa-T04-001 | OK | - | 2026-11-22 | identity copy of the model: sha256 and size: diff 0, 1/1 lines |
| ITa-T04-002 | OK | - | 2026-11-22 | planned layout of the copy equals upstream: diff 0, 322/322 lines |
| ITa-T04-003 | OK | - | 2026-11-22 | identity copy of the projector: sha256 and size: diff 0, 1/1 lines |
| ITa-T04-004 | OK | - | 2026-11-22 | planned layout after retyping equals upstream: diff 0, 322/322 lines |
| ITa-T04-005 | OK | - | 2026-11-22 | retyped file: sha256 and size: diff 0, 1/1 lines |
| ITa-T04-006 | OK | - | 2026-11-22 | upstream reads the retyped file back with that layout: diff 0, 322/322 lines |
| ITa-T05-001 | OK | - | 2026-11-22 | token ids equal /tokenize: pas records: diff 0, 75/75 lines |
| ITa-T05-002 | OK | - | 2026-11-22 | token ids equal /tokenize: en records: diff 0, 402/402 lines |
| ITa-T05-003 | OK | - | 2026-11-22 | token ids equal /tokenize: ja records: diff 0, 402/402 lines |
| ITa-T05-004 | OK | - | 2026-11-22 | token ids equal /tokenize: sp records: diff 0, 68/68 lines |
| ITa-T05-005 | OK | - | 2026-11-22 | token ids equal /tokenize: edge records: diff 0, 31/31 lines |
| ITa-T05-006 | OK | - | 2026-11-22 | token ids equal /tokenize: fuzz records: diff 0, 2000/2000 lines |
| ITa-T05-007 | OK | - | 2026-11-22 | decode of encode gives the text back: diff 0, 2978/2978 lines |
| ITa-T05-008 | OK | - | 2026-11-22 | plain mode: one file's ids equal /tokenize: diff 0, 1/1 lines |
| ITa-T08-001 | OK | - | 2026-11-22 | hand case, two tokens: value 0.00E+00, limit 0.00E+00 |
| ITa-T08-002 | OK | - | 2026-11-22 | hand case, g = ln(1/2) on token 1: value 0.00E+00, limit 0.00E+00 |
| ITa-T08-003 | OK | - | 2026-11-22 | beta = 0, H 48, T 200: value 0.00E+00, limit 0.00E+00 |
| ITa-T08-004 | OK | - | 2026-11-22 | reference, g in [-0.1, 0], seed 13, H 48, T 200: value 4.77E-07, limit 1.00E-05 |
| ITa-T08-005 | OK | - | 2026-11-22 | reference, g in [-5, 0], seed 14, H 16, T 200: value 2.38E-07, limit 1.00E-05 |
| ITa-T08-006 | OK | - | 2026-11-22 | reference, T = 1, 63, 64, 65, 200, H 48: value 4.17E-07, limit 1.00E-05 |
| ITa-T08-007 | OK | - | 2026-11-22 | g = -inf, H 48, T 20: value 2.38E-07, limit 1.00E-06 |
| ITa-T13-001 | OK | - | 2026-11-22 | value 0.0E+000, limit 1.0E-010 |
| ITa-T13-002 | OK | - | 2026-11-22 | value 0.0E+000, limit 1.0E-010 |
| ITa-T13-003 | OK | - | 2026-11-22 | value 0.0E+000, limit 1.0E-006 |
| ITa-T13-004 | OK | - | 2026-11-22 | value 0.0E+000, limit 0.0E+000 |
| ITa-T13-005 | OK | - | 2026-11-22 | value 0.0E+000, limit 1.0E-010 |
| ITa-T14-001 | OK | - | 2026-11-22 | fp32, 20 updates, lr 0.05, wd 0.01 against TOptimizer.AdamW: value 1.2E-007, limit 1.0E-006 |
| ITa-T14-002 | OK | - | 2026-11-22 | fp64, 200 updates against TOptimizer.AdamW: value 1.1E-016, limit 1.0E-012 |
| ITa-T14-003 | OK | - | 2026-11-22 | fp32, 20 updates, global-norm clipping at 0.5 against ClipGradNorm + AdamW: value 2.4E-007, limit 1.0E-006 |
| ITa-T14-004 | OK | - | 2026-11-22 | fp64, 50 updates with clipping: two stores (w; b) against one store (w, b): value 0.0E+000, limit 0.0E+000 |
| ITa-T14-005 | OK | - | 2026-11-22 | fp32, 100 updates, ScheduleLR(0.05, u, 10, 100, 0.1) on both sides: value 6.0E-008, limit 1.0E-006 |
| ITa-T14-006 | OK | - | 2026-11-22 | fp32: 10 updates against 5 + save + load + StepCount 5 + 5: value 0.0E+000, limit 0.0E+000 |
| ITa-T14-007 | OK | - | 2026-11-22 | fp64: gradients after ClipGlobalNorm(0.1) against ClipGradNorm(0.1), and the returned norm: value 0.0E+000, limit 1.0E-012 |
| ITa-T07-001 | OK | - | 2026-11-22 | renderings different from /apply-template, of 14 conversations: value 0, limit 0 |
| ITa-T07-002 | OK | - | 2026-11-22 | training samples whose token ids differ from /tokenize, of 5: value 0, limit 0 |
| ITa-T07-003 | OK | - | 2026-11-22 | training samples whose loss span is not exactly the last answer and its end-of-turn token, of 5: value 0, limit 0 |
| ITa-T07-004 | OK | - | 2026-11-22 | conversations the template rejects that did not raise EChatError, of 4: value 0, limit 0 |
| ITa-T09-001 | OK | - | 2026-11-22 | O and State against DeltaRuleTokens, H 48, 128 x 128, T = 1, 63, 64, 65, 200 (seeds 100 + T): value 7.2E-007, limit 1.0E-005 |
| ITa-T09-002 | OK | - | 2026-11-22 | O and State against the scalar Double reference RefTokens, seed 13, H 48, T 200: value 4.8E-007, limit 1.0E-005 |
| ITa-T09-003 | OK | - | 2026-11-22 | strong decay: g in (-5, 0], seed 14, H 16, T 200: value 4.4E-006, limit 1.0E-005 |
| ITa-T09-004 | OK | - | 2026-11-22 | training length T 2048, H 48: 116 ms on the CPU: value 7.2E-007, limit 1.0E-005 |
| ITa-T09-005 | OK | - | 2026-11-22 | gradients of all five inputs against the token form, H 4, T 70, chunk 16: value 2.1E-006, limit 1.0E-005 |
| ITa-T09-006 | OK | - | 2026-11-22 | chunk sizes 1, 2, 3, 5, 7, 10, 49, 50, 51, 64 at T 50 (padding and a single chunk): value 9.8E-007, limit 1.0E-005 |
| ITa-T10-001 | OK | - | 2026-11-22 | every block of the rehearsal model (24), T 8, against the scalar version: value 8.1E-007, limit 1.0E-004 |
| ITa-T10-002 | OK | - | 2026-11-22 | rehearsal blocks 0 (linear) and 3 (full), T 70 (two chunks of the delta rule): value 5.4E-007, limit 1.0E-004 |
| ITa-T10-003 | OK | - | 2026-11-22 | tiny model, T 70: 2 key / 4 value SSM heads (tiling), 4 query / 2 KV heads (GQA): value 2.4E-006, limit 1.0E-004 |
| ITa-T10-004 | OK | - | 2026-11-22 | rehearsal blocks 0-3 chained, T 16, against the scalar chain: value 8.9E-007, limit 1.0E-004 |
| ITa-T10-005 | OK | - | 2026-11-22 | rehearsal blocks 0-3 as Recompute stages: dL/dX0 against an ordinary backward (max |grad| 4.15): value 0.0E+000, limit 1.0E-005 |
| ITa-T11-001 | OK | - | 2026-11-23 | value 0.0E+000, limit 0.0E+000 |
| ITa-T11-002 | OK | - | 2026-11-23 | value 1.3E-003, limit 2.0E-002 |
| ITa-T11-003 | OK | - | 2026-11-23 | value 1.4E-003, limit 2.0E-002 |
| ITa-T11-004 | OK | - | 2026-11-23 | value 0.0E+000, limit 0.0E+000 |
| ITa-T12-001 | OK | - | 2026-11-23 | value 0.0E+000, limit 0.0E+000 |
| ITa-T12-002 | OK | - | 2026-11-23 | value 0.0E+000, limit 0.0E+000 |
| ITa-T12-003 | OK | - | 2026-11-23 | value 4.8E-005, limit 1.0E-003 |
| ITa-T12-004 | OK | - | 2026-11-23 | value 0.0E+000, limit 1.0E-004 |
| ITa-T12-005 | OK | - | 2026-11-23 | value 0.0E+000, limit 0.0E+000 |
