# Evidence — ITa-T06-013 (2026-11-18, sub-agent run, task T06c2)

## Environment
- `QLORA_DIALECT_SKILLS=$HOME/.claude/skills/polyglot-scripting/pascal-dialects/skills` (from the parent shell)
- stock compiler (pascal-fpc-objfpc, pascal-fpc-delphi): `QLORA_FPC=/usr/bin/fpc` — oracle-equivalent (the oracle script itself uses `FPC_STOCK:-/usr/bin/fpc`); **not** the program default (bare `fpc`, which `IsPathUsable` cannot resolve — pinned by UT-qlora_verify-003)
- unleashed/delphi compilers: program default `$HOME/unleashed/fpc/bin/fpc` (`QLORA_FPC_UNLEASHED` unset) — same default as the oracle's `FPC_UNLEASHED`
- `HOME=$HOME`; cwd for all runs: project root (work dir `work/verify` is relative)

## Commands (in order)
1. Build:
   `fpc -Mobjfpc -Sh -vew -l- -Fu<proj>/src -FU<proj>/work/bin <proj>/src/qlora_verify.pas -o<proj>/work/bin/qlora_verify`
   (981 lines compiled, 0 errors; FPC 3.2.2)
2. Per skill (final TSVs kept in `work/ita/`):
   `./work/bin/qlora_verify --dialect <D> $QLORA_DIALECT_SKILLS/<skill> > work/ita/<skill>.tsv`
   `<D>` per tasks/T06-facts.md: objfpc / fpc-delphi / unleashed / delphi / oxygene.
   The two stock-dialect skills were first run **without** `QLORA_FPC` (default env):
   all files `skipped "compiler not found"` — kept as
   `evidence/pascal-fpc-objfpc.default-env.tsv` and `evidence/pascal-fpc-delphi.default-env.tsv`,
   then rerun with `QLORA_FPC=/usr/bin/fpc` (the TSV kept in `work/ita/<skill>.tsv`).
3. Oracle (once):
   `bash $QLORA_DIALECT_SKILLS/../tests/check-examples.sh > work/ita/oracle.txt 2>&1`  → exit 0, 44 OK + 1 SKIP (oxygene), 0 NG
4. Comparison (awk, per-file):
   `awk -f work/ita/compare.awk work/ita/oracle.txt work/ita/pascal-*.tsv > work/ita/compare.txt`
   → AGREE 37 (incl. 6 oxygene static-ok) / NG 13 (all `wrong-error`) / INFO 1 (references/p04, not in oracle)

## Result files
- `work/ita/pascal-fpc-objfpc.tsv` — 9 lines: 6 accepted, 3 wrong-error
- `work/ita/pascal-fpc-delphi.tsv` — 10 lines: 6 accepted, 3 wrong-error, 1 compile-error (references/, not in oracle)
- `work/ita/pascal-unleashed.tsv` — 16 lines: 11 accepted, 5 wrong-error
- `work/ita/pascal-delphi.tsv` — 10 lines: 8 accepted, 2 wrong-error
- `work/ita/pascal-oxygene.tsv` — 6 lines: 6 static-ok
- `work/ita/oracle.txt`, `work/ita/compare.txt`, `work/ita/compare.awk`
- `work/ita/evidence/*.default-env.tsv` — initial default-env runs (stock dialects)

## Root-cause probe for BUG-001 (all 13 NG files, one common cause)
- hexdump of `pascal-unleashed/pitfalls/p01_generic_keyword.pas` line 3 (expect-error text, ASCII quotes)
  and of the unleashed-fpc compile log line `Fatal: Syntax error, "=" expected but "identifier TBOX" found`
  → the text is byte-for-byte a substring of the log; `grep -F` (oracle style) matches → oracle OK.
- Code audit: `src/DialectVerify.pas` `ExtractExpectErrorText` copies the text from the lowercased
  source (`Line := Copy(Low, S, E-S)`); `VerifyFile` then does a case-sensitive `Pos(Pitfall, Log)`.
  Every one of the 13 failing expect-error texts contains a capital (TBOX, Illegal, Identifier,
  Variable, Generics, Incompatible, Ordinal, Parser, ABC); the two fully-lowercase expect-error
  pitfalls (unleashed p02, p04) are `accepted`. → defect recorded as BUG-001.

## Re-run (2026-11-18 17:4x–18:1x, T06c3 step 3; BUG-001/002 fixes in the rebuilt binary)
- Build: `fpc -Mobjfpc -Sh -gl -Fusrc -Fu../../src -o work/bin/qlora_verify src/qlora_verify.pas` → 0 errors.
- Per skill, **program-default env** (`env -u QLORA_FPC -u QLORA_FPC_UNLEASHED`), cwd = project root:
  `./work/bin/qlora_verify --dialect <D> $QLORA_DIALECT_SKILLS/<skill> > work/ita/<skill>.tsv`
  exit codes: objfpc 1, fpc-delphi 1, unleashed 0, delphi 0, oxygene 0 (`work/ita/exit-codes.txt`, full log in `run-013.log`).
- Oracle (once): `bash $QLORA_DIALECT_SKILLS/../tests/check-examples.sh` → exit 0, 44 OK + 1 SKIP (oxygene), 0 NG (`oracle.txt`).
- Comparison: `awk -f work/ita/compare.awk work/ita/oracle.txt work/ita/pascal-*.tsv > work/ita/compare.txt`
  → **AGREE 32 / NG 18 / INFO 1** (all 18 NG = `Executable not found: "<cwd>/fpc"`, new defect BUG-003; INFO = references/p04, not in oracle).
- Result files (copies of `work/ita/`): `pascal-*.tsv`, `oracle.txt`, `compare.txt`, `exit-codes.txt`, `run-013.log`.
- Interpretation: BUG-001 fix confirmed (unleashed 16/16 and delphi 10/10 accepted, incl. all pitfall files);
  BUG-002 fix confirmed at the gate (no `skipped "compiler not found"` anymore); remaining failures all stem from
  BUG-003 (bare `fpc` name resolved against cwd in `RunLimited`, never via PATH — src/DialectVerify.pas:529-530).

## Re-run (2026-11-18, T06c4 step 2; BUG-003 fix in rebuilt binary)
- Build: `fpc -Mobjfpc -Sh -vew -l- -Fu<proj>/src -FU<proj>/work/bin <proj>/src/qlora_verify.pas -o<proj>/work/bin/qlora_verify` → 0 errors (rebuilt 19:13; src/DialectVerify.pas 18:49 > old binary 17:50).
- Per skill, **program-default env** (`env -u QLORA_FPC -u QLORA_FPC_UNLEASHED`), cwd = project root:
  `./work/bin/qlora_verify --dialect <D> $SKILLS/<skill> > work/ita/<skill>.tsv`
  exit codes: **objfpc 0, fpc-delphi 1, unleashed 0, delphi 0, oxygene 0** (`work/ita/exit-codes.txt`, `run-013.log`).
  The bare `fpc` now resolves via PATH (BUG-003 fix); no `Executable not found` anywhere.
- Oracle (once): `bash $SKILLS/../tests/check-examples.sh` → exit 0, **44 OK + 1 SKIP (oxygene)**, 0 NG (`oracle.txt`).
- Comparison: `awk -f work/ita/compare.awk work/ita/oracle.txt work/ita/pascal-*.tsv > work/ita/compare.txt`
  → **AGREE 50 / NG 0 / INFO 1 → PASS**.
  Per skill (AGREE / NG / INFO): objfpc 9/0/0 (all incl. 3 pitfalls `accepted`), fpc-delphi 9/0/1 (all 9 oracle files `accepted`),
  unleashed 16/0/0 (incl. 7 pitfalls `accepted`), delphi 10/0/0 (incl. 3 pitfalls `accepted`), oxygene 6/0/0 (6 `static-ok`).
  fpc-delphi exit=1 is the expected `compile-error` of references/p04_icomparer_invisible (INFO, absent from oracle; IComparer).
- Interpretation: BUG-003 fix confirmed on ITa; the 6 previously unverified BUG-001 pitfall files
  (pascal-fpc-objfpc p01–p03, pascal-fpc-delphi p01–p03) now reach compile and are `accepted` → BUG-001 fully 確認済み.
- All TSV/oracle/compare copies of this re-run kept in this evidence dir (refreshed 19:18).
