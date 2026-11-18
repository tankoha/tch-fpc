# UT cases — `DialectVerify` (T06)

Rules: one fpcunit `TTest` method per row, method `UT_DialectVerify_NNN`.
`dialect` column tags which compiler dialect path the case exercises.
All "fake" tools are shell scripts written to `work/ut/bin` by the test setup.

Corrections 2026-11-17 (per `tasks/T06-facts.md`, corrected external design):
retired rows keep their old expectations verbatim; surviving rows read under the
corrected contracts. Legacy notation in surviving rows: `dialect-objfpc` ≙
`objfpc`, `dialect-unleashed` ≙ `unleashed`, `dialect-oxygene` ≙ `oxygene`,
surviving `dialect-delphi` rows mean `{$mode delphi}` ≙ `fpc-delphi`,
`RunWithLimit` ≙ `RunLimited`, `DialectValidateMain` ≙ `VerifyMain`.
New sequential cases 052+ cover the corrected behaviour.

| case ID | target function | dialect | branch / viewpoint | input | expected result |
|---|---|---|---|---|---|
| UT-DialectVerify-001 | DetectDialect | objfpc | `{$MODE OBJFPC}` literal hit | `{$mode objfpc}{$H+}` | `dialect-objfpc` |
| UT-DialectVerify-002 | DetectDialect | delphi | `{$MODE DELPHI}` hit, case-insensitive | text containing `{$MODE Delphi}` | `dialect-delphi` |
| UT-DialectVerify-003 | DetectDialect | unleashed | `{$MODE UNLEASHED}` hit | text containing `{$MODE UNLEASHED}` | `dialect-unleashed` |
| UT-DialectVerify-004 | [廃止] facts により `{$LIBRARYNAME}` の検出ルール廃止; Oxygene は「ファイルの先頭語が `namespace`」のみ (062/063) `{$LIBRARYNAME <name>}` hit | line `  {$LIBRARYNAME _}` | `dialect-oxygene` |
| UT-DialectVerify-005 | [廃止] 検出はファイル先頭語のみになったため、`// NAMESPACE` 行はもうヒットしない (062/063) `NAMESPACE` word-boundary hit | line starting `// NAMESPACE foo` | `dialect-oxygene` |
| UT-DialectVerify-006 | DetectDialect | objfpc | `NAMESPACE` embedded in longer word must NOT hit | line `// MYNAMESPACE foo` | `dialect-objfpc` |
| UT-DialectVerify-007 | DetectDialect | objfpc | no indicator at all → default | `program x; begin end.` | `dialect-objfpc` |
| UT-DialectVerify-008 | DetectDialect | delphi | earliest position wins | delphi line before objfpc line | `dialect-delphi` |
| UT-DialectVerify-009 | [廃止] facts により `QLORA_DELPHI_PATH` は不使用; `delphi` は `QLORA_FPC_UNLEASHED`+`fpc-approx` に (065) delphi without `QLORA_DELPHI_PATH` | DelphiPath=`''` | `Result=Fpc`, class=`fpc-approx` |
| UT-DialectVerify-010 | [廃止] `QLORA_DELPHI_PATH` 廃止 (065) delphi with `QLORA_DELPHI_PATH` set | DelphiPath=`/dq/comp` | `Result=/dq/comp`, class=`compiled-run` |
| UT-DialectVerify-011 | ChooseCompiler | unleashed | unleashed selection | env defaults | `Result=FpcUnleashed`, class=`compiled-run` |
| UT-DialectVerify-012 | ChooseCompiler | objfpc | objfpc selection | env defaults | `Result=Fpc`, class=`compiled-run` |
| UT-DialectVerify-013 | ChooseCompiler | oxygen | oxygen selection | — | `Result=''` (static-only) |
| UT-DialectVerify-014 | [廃止] コンパイラ不在は outcome `skipped` (失敗扱いしない) に (101) VerifyFile | objfpc | compiler path missing on disk | Fpc=`/no/such/fpc`, real `{$mode objfpc}` src | line class `static-only`, JSON `{}` |
| UT-DialectVerify-015 | [廃止] 存在しないパスは VerifyMain での usage エラー exit 2; VerifyFile は読み取れるソースを前提 (112) VerifyFile | objfpc | source file missing | non-existent src name | dialect `dialect-objfpc`, class `static-only`, JSON `{}` |
| UT-DialectVerify-016 | [廃止] outcome は TSV; 意図したエラー一致は `accepted` (098) VerifyFile | objfpc | compile fail + expect-error text found in log | `fcomp_bad` (3 error lines incl. `Identifier not found "brokenproc"`), src has `// expect-error: Identifier not found "brokenproc"` | `compiled=false`, `match="expect-error"`, no `exit_code`, no `timeout` |
| UT-DialectVerify-017 | [廃止] error_lines 方式廃止; `compile-error` + 最初のエラー行 detail (097, 079) VerifyFile | delphi | compile fail, no expect-error → empty match + error_lines | `fcomp_bad`, DelphiPath set, src without expect-error | `match=""`, `error_lines` length 3, first triple `["error","f.pas:2","..."]` |
| UT-DialectVerify-018 | [廃止] detail は最初の `Error:`/`Fatal:` 行のみ (079) VerifyFile | objfpc | log > 10 error lines → first 10 only | `fcomp_many` (15 error lines) | `error_lines` length exactly 10, last is line 10's |
| UT-DialectVerify-019 | [廃止] outcome は `compiled-but-should-fail` (100) VerifyFile | objfpc | compile SUCCESS while expect-error present → defect path | `fcomp_ok`, src has expect-error comment | `compiled=true`, `match=""`, program not run (no exit_code, no timeout in JSON) |
| UT-DialectVerify-020 | [廃止] outcome 語彙変更; 一致は `accepted` (092) VerifyFile | objfpc | run success, expected present, exact | `fexe_out` prints `ok\n`; expected file `ok\n` | `exit_code=0`, `match="exact"` |
| UT-DialectVerify-021 | [廃止] `.expected` なしは `no-expected` (094) VerifyFile | objfpc | run success, NO expected file → exact | `fexe_out`, no expected file | `match="exact"` (ITa-002 rule) |
| UT-DialectVerify-022 | [廃止] `partial` 廃止; バイト一致不一致は `wrong-output` (093) VerifyFile | objfpc | expected ⊂ actual → partial | expected `ok`, actual `ok-verbose` | `match="partial"` |
| UT-DialectVerify-023 | [廃止] `partial` 廃止 (093) VerifyFile | objfpc | actual ⊂ expected → partial | expected `ok-verbose`, actual `ok` | `match="partial"` |
| UT-DialectVerify-024 | [廃止] 不一致は `wrong-output` (093) VerifyFile | objfpc | no overlap → empty match | expected `foo`, actual `bar` | `match=""` |
| UT-DialectVerify-025 | [廃止] 非ゼロ終了は `run-error`, detail=終了コード (095) VerifyFile | unleashed | non-zero exit code still compared | `fexe_err1` (prints `boom`, exit 1), expected `boom` | `exit_code=1`, `match="exact"` (line exit = 1 overall) |
| UT-DialectVerify-026 | [廃止] `QLORA_TIMEOUT` 廃止; limit は `--timeout`/default 10 のみ; outcome `timeout`+detail=limit (096) VerifyFile | objfpc | run timeout | `fexe_sleep` (sleep 5), QLORA_TIMEOUT=1 | `timeout=true`, NO `exit_code` key, line good=false → `DialectValidateMain` exit 1 |
| UT-DialectVerify-027 | RunWithLimit | general | failed start → exception branch | `/no/such/binary arg` | `Started=False`, output contains `process error` |
| UT-DialectVerify-028 | RunWithLimit | general | clean start/read path | `/usr/bin/echo hello` | `Started=True`, `TimedOut=False`, output `hello`, exit 0 |
| UT-DialectVerify-029 | [廃止] CompareOutputs はバイト一致判定に; 空 vs 空 → `accepted` (092) CompareOutputs | general | expected file EMPTY + actual empty → exact | `''`/`''`, present=True | `exact` |
| UT-DialectVerify-030 | [廃止] 空 vs 非空は `wrong-output` (093) CompareOutputs | general | expected file EMPTY + actual non-empty → '' | `''`/`x`, present=True | `''` |
| UT-DialectVerify-031 | [廃止] TSV は 5 列 path dialect class outcome detail VerifyFile | objfpc | TSV line = 4 tab-separated cells, correct dialect/class | case-020 input | cells[1]=`dialect-objfpc`, cells[2]=`compiled-run`, cells[0]=src path |
| UT-DialectVerify-032 | [廃止] JSON 出力廃止 (TSV; facts) BuildOutcomeJSON | general | special chars escaped + field order | actual `a"b<TAB>c` | JSON contains `\"b\\tc`, order compiled→exit_code→stdout→expected→match→compiler_output |
| UT-DialectVerify-033 | [廃止] バイト一致のため CRLF vs LF は等価でない → `wrong-output` (093) CompareOutputs | general | CRLF stdout vs LF expected → exact | `ok\r\n` vs `ok\n` | `exact` |
| UT-DialectVerify-034 | [廃止] `--dir`/FILES モデル廃止; `PATH...` 混在引数 (111, 107) DialectValidateMain | general | `--dir` recursion + FILES order + exit 0 | dir with good `a.pas`, bad `sub/d.pas` | lines order = FILES then dir-sorted, exit code 1 (bad line) — and all-good variant exit 0 |
| UT-DialectVerify-035 | [廃止] `QLORA_TIMEOUT` 廃止; `--timeout` は default 10s に優先 (111, 096) DialectValidateMain | general | `--timeout` override beats QLORA_TIMEOUT | `--timeout 1`, `fexe_sleep` | `timeout=true` in outcome (1 s applied) |
| UT-DialectVerify-036 | DialectValidateMain | general | `--timeout` invalid (non-numeric) | `--timeout abc file.pas` | `False`, exit 2, usage line |
| UT-DialectVerify-037 | DialectValidateMain | general | unknown `--opt` | `--frobnicate a.pas` | `False`, exit 2, usage line |
| UT-DialectVerify-038 | ExtractExpectErrorText | general | directive present, capture text after colon | `// expect-error: broken proc` | `broken proc` |
| UT-DialectVerify-039 | [廃止] ExtractErrors トリプル方式廃止; detail は最初の `Error:`/`Fatal:` 行 (079) ExtractErrors | general | FPC-style Error/Fatal lines, col stripped | 2 `Error:` + 1 `Fatal:` line | 3 triples, `FileLine`=`f.pas:12` (no `,col`) |
| UT-DialectVerify-040 | [廃止] FirstCompilerErrorLine: 該当なし → `''` (080) ExtractErrors | general | non-matching lines ignored | compiler banner lines | `Length=0` |
| UT-DialectVerify-041 | [廃止] `--dir` 廃止; PATH なし → usage exit 2 (112) DialectValidateMain | general | `--dir` present but empty argv otherwise | `--dir ./emptydir` | `False`, exit 2, usage line (no target) |
| UT-DialectVerify-042 | [廃止] ReadEnvRec は `QLORA_FPC`/`QLORA_FPC_UNLEASHED` のみ (052-054) ReadEnvRec | general | all env vars set (PutEnv) | set 4 vars + override | all read from env, `TimeoutSec` from override |
| UT-DialectVerify-043 | [廃止] `QLORA_DELPHI_PATH`/`QLORA_TIMEOUT` 廃止; 残るデフォルトのみ (053) ReadEnvRec | general | all env vars unset → defaults | cleared env | `fpc`, `$HOME/unleashed/fpc/bin/fpc`, `''`, `2` |
| UT-DialectVerify-044 | [廃止] 期待ファイルは常に `<base>.expected`、拡張子指定なし (055/056) ExpectedPathFor | general | expected file exists / not | file present / absent | `<dir>/<base>.<ext>` / `''` |
| UT-DialectVerify-045 | [廃止] `.expected` は ReadTextFile 経由 (084/085) ReadExpectedFile | general | file read / missing | present / absent | content / `''` |
| UT-DialectVerify-046 | ExtractExpectErrorText | general | directive absent | text without `expect-error:` | `''` |
| UT-DialectVerify-047 | [廃止] `--expect` フラグ廃止; `.expected` は固定 (055/056) DialectValidateMain | general | `--expect` custom extension honored | `--expect want`, `<base>.want` file | expected found, `match="exact"` |
| UT-DialectVerify-048 | [廃止] detail 上限は 200 文字 (NormalizeDetail) (083) TruncateStr | general | long / short / exact length | 40/40/41 chars vs max 40 | truncated / identical / identical |
| UT-DialectVerify-049 | [廃止] ReadFileText → ReadTextFile (Boolean+text) (084/085) ReadFileText | general | readable file / missing file | file / non-path | content / `''` |
| UT-DialectVerify-050 | [廃止] 収集は `*.pas`/`*.dpr` のみ (`.pp` は対象外) (107) CollectPascalFiles | general | recursive, extension filter, empty dir, missing dir | nested `.pas`/`.pp`/`.dpr`/`.txt` | all pascal files sorted, `.txt` excluded |
| UT-DialectVerify-051 | SortStrings | general | n=0 / n=1 / n>1 unsorted | `['c','a','b']` etc. | sorted / unchanged / `[]` |
| UT-DialectVerify-052 | [廃止] ReadEnvRec は環境変数を読み不再、値を引数で渡す形に変更 (2.1; 新ケース 121) ReadEnvRec | general | `QLORA_FPC` 設定時はその値 | PutEnv `QLORA_FPC=/fp` | `Fpc='/fp'` |
| UT-DialectVerify-053 | [廃止] ReadEnvRec は環境変数を読み不再、値を引数で渡す形に変更 (2.1; 新ケース 122) ReadEnvRec | general | 両未設定 → デフォルト | env クリア, `AHomeDir='/h'` | `Fpc='fpc'`, `FpcUnleashed='/h/unleashed/fpc/bin/fpc'` |
| UT-DialectVerify-054 | [廃止] ReadEnvRec は環境変数を読み不再、値を引数で渡す形に変更 (2.1; 新ケース 123) ReadEnvRec | general | `QLORA_FPC_UNLEASHED` 設定時はその値 | PutEnv `=/up` | `FpcUnleashed='/up'` |
| UT-DialectVerify-055 | ExpectedPathFor | general | `.pas` ソース | `s/unit.pas` | `s/unit.expected` |
| UT-DialectVerify-056 | ExpectedPathFor | general | `.dpr` ソースにも `<base>.expected` (拡張子無視) | `s/main.dpr` | `s/main.expected` |
| UT-DialectVerify-057 | DetectDialect | fpc-delphi | `{$MODE DELPHI}` → 修正済み名前 | text containing `{$MODE Delphi}` | `fpc-delphi` |
| UT-DialectVerify-058 | DetectDialect | unleashed | `{$MODE UNLEASHED}` → 修正済み名前 | text containing `{$MODE UNLEASHED}` | `unleashed` |
| UT-DialectVerify-059 | DetectDialect | delphi | mode なし + ファイル名 `.dpr` (新ルール) | file `main.dpr`, text `program x;` | `delphi` |
| UT-DialectVerify-060 | DetectDialect | delphi | mode なし + `{$APPTYPE CONSOLE}` 行 (新ルール) | file `x.pas`, text with `{$APPTYPE CONSOLE}` line | `delphi` |
| UT-DialectVerify-061 | DetectDialect | objfpc | コメント内の `APPTYPE` は不算出 | text `// {$APPTYPE CONSOLE}` | `objfpc` |
| UT-DialectVerify-062 | DetectDialect | oxygen | ファイル先頭語 `namespace` (修正済みルール) | text first word `namespace` | `oxygene` |
| UT-DialectVerify-063 | DetectDialect | objfpc | 本文の `namespace` (先頭語でなく) は不算出 | text `program x; ... namespace foo;` | `objfpc` |
| UT-DialectVerify-064 | ChooseCompiler | fpc-delphi | `fpc-delphi` は `QLORA_FPC` | `Fpc='/fp'` | `Result='/fp'`, class=`compiled-run` |
| UT-DialectVerify-065 | ChooseCompiler | delphi | `delphi` は `QLORA_FPC_UNLEASHED`+`fpc-approx` (修正済み) | `FpcUnleashed='/up'` | `Result='/up'`, class=`fpc-approx` |
| UT-DialectVerify-066 | ChooseCompiler | — | 未知の dialect 名 (守衛ブランチ) | `'wat'` | `Result=''`, class=`''` |
| UT-DialectVerify-067 | BuildCompileCmdLine | objfpc | ベースフラグと順序 `-vew -l- -FE<out> -FU<out>` | paths set, AIsDelphi=False | 4 flags present in order, all quoted |
| UT-DialectVerify-068 | BuildCompileCmdLine | delphi | `-Mdelphiunicode -Manonymousfunctions -Mfunctionreferences -Minlinevars` | AIsDelphi=True | 4 `-M` switches present |
| UT-DialectVerify-069 | RewriteDelphiUnits | delphi | `System.SysUtils` → bare (uses 内) | `uses System.SysUtils;` | `uses SysUtils;` |
| UT-DialectVerify-070 | RewriteDelphiUnits | delphi | 9 単純ユニット全置換 | `System.SysUtils/Math/StrUtils/DateUtils/Classes/Types/Variants/TypInfo/Rtti` | all → bare names |
| UT-DialectVerify-071 | RewriteDelphiUnits | delphi | `System.Generics.Collections` → `Generics.Collections` | uses clause with it | `Generics.Collections;` |
| UT-DialectVerify-072 | RewriteDelphiUnits | delphi | 混合行の同時置換 | `System.SysUtils, System.Generics.Collections` | `SysUtils, Generics.Collections` |
| UT-DialectVerify-073 | RewriteDelphiUnits | delphi | 語境界: `XSystem.SysUtils` は不変 | text with `XSystem.SysUtils` | unchanged |
| UT-DialectVerify-074 | RewriteDelphiUnits | delphi | 語境界: `System.SysUtilsX` は不変 | text with `System.SysUtilsX` | unchanged |
| UT-DialectVerify-075 | RewriteDelphiUnits | delphi | `System.` なし → 不変 | `program x; begin end.` | unchanged |
| UT-DialectVerify-076 | ExtractExpectErrorText | general | 修正済み構文: 括弧 コメント | `{ bad call expect-error: broken proc }` | `broken proc` |
| UT-DialectVerify-077 | ExtractExpectErrorText | general | 複数ある → 最初 | 2 directive lines | first text only |
| UT-DialectVerify-078 | FirstCompilerErrorLine | general | 最初の `Error:` 行を返す | 3rd of 5 lines has `Error: ...` | that line |
| UT-DialectVerify-079 | FirstCompilerErrorLine | general | `Fatal:` 行は対象 | `Fatal: ...` line | that line |
| UT-DialectVerify-080 | FirstCompilerErrorLine | general | 該当なし → `''` | banner + notes only | `''` |
| UT-DialectVerify-081 | NormalizeDetail | general | タブ → スペース (facts) | `a<TAB>b` | `a b` |
| UT-DialectVerify-082 | NormalizeDetail | general | 上限 200: ちょうど 200 → 不変 | 200 chars | identical |
| UT-DialectVerify-083 | NormalizeDetail | general | 上限 200: 201 → 200 文字 | 201 chars | length exactly 200 |
| UT-DialectVerify-084 | ReadTextFile | general | 読み込み成功 | existing file | `True` + content |
| UT-DialectVerify-085 | ReadTextFile | general | 存在しない → False | non-path | `False`, text `''` |
| UT-DialectVerify-086 | IsPathUsable | general | 存在ファイル / 不在 | `f` / `/no/x` | `True` / `False` |
| UT-DialectVerify-087 | CompileStep | objfpc | コンパイル成功 → csOk + 実行可能 | fake fpc (exit 0, exe writes) | `csOk`, `AExePath='<out>/<name>'` |
| UT-DialectVerify-088 | CompileStep | objfpc | コンパイル失敗 → csFail | fake fpc (exit 3, 出力 `Error: ...`) | `csFail`, `AExePath=''`, log holds output |
| UT-DialectVerify-089 | CompileStep | general | タイムアウト → csTimeout | fake sleep 5, limit 1 | `csTimeout` |
| UT-DialectVerify-090 | CompileStep | general | コンパイラ不在 → csNoCompiler | `Fpc='/no/fpc'` | `csNoCompiler` |
| UT-DialectVerify-091 | CompileStep | delphi | delphi は書き換えコピーを `<out>` にコンパイル | fake fpc が引数・コピー内容をログ | 引数=`<out>/<name>`, copy holds `Generics.Collections` |
| UT-DialectVerify-092 | VerifyFile | objfpc | fake fpc+exe: ok 結果 → `accepted` | sample ok + `.expected` | line outcome `accepted`, `AOVOK=True` |
| UT-DialectVerify-093 | VerifyFile | objfpc | 出力不一致 → `wrong-output` | actual `bar`, expected `foo` | `wrong-output`, `AOVOK=False` |
| UT-DialectVerify-094 | VerifyFile | objfpc | `.expected` なし → `no-expected` | no expected file | `no-expected` |
| UT-DialectVerify-095 | VerifyFile | objfpc | 実行終了コード 3 → `run-error` | fake exe exit 3 | `run-error`, detail `3` |
| UT-DialectVerify-096 | VerifyFile | general | 実行タイムアウト (limit 1) → `timeout` | fake sleep 5, ATimeoutSec=1 | `timeout`, detail `1` |
| UT-DialectVerify-097 | VerifyFile | objfpc | コンパイル失敗+pitfall なし → `compile-error` | fake fpc fail | `compile-error`, detail = 最初の `Error:` 行 |
| UT-DialectVerify-098 | VerifyFile | objfpc | pitfall+エラー一致 → `accepted` | `{ expect-error: X }` + log `Error: ... X` | `accepted` |
| UT-DialectVerify-099 | VerifyFile | objfpc | pitfall+エラー不一致 → `wrong-error` | mismatch text | `wrong-error`, detail = 最初の `Error:` 行 |
| UT-DialectVerify-100 | VerifyFile | objfpc | pitfall ありでコンパイル成功 → `compiled-but-should-fail` | fake fpc ok + directive | `compiled-but-should-fail` |
| UT-DialectVerify-101 | VerifyFile | objfpc | コンパイラ不在 → `skipped` (失敗扱いせず) | `Fpc='/no/fpc'` | `skipped`, `AOVOK=True` |
| UT-DialectVerify-102 | VerifyFile | general | コンパイルハング → `skipped`, detail `timeout` | fake sleep, limit 1 | `skipped`, detail `timeout` |
| UT-DialectVerify-103 | VerifyFile | oxygen | 先頭語 `namespace` → `static-ok` | text first word namespace | `static-ok`, class `static-only` |
| UT-DialectVerify-104 | VerifyFile | oxygen | 先頭語非 namespace → `static-reject` | text `program x;` | `static-reject` |
| UT-DialectVerify-105 | VerifyFile | delphi | forced dialect: `delphi` が `{$mode objfpc}` を上書き、`QLORA_FPC_UNLEASHED` 使用 | `--dialect delphi`, fake unleash fpc with `-M` | line dialect `delphi`, `accepted` |
| UT-DialectVerify-106 | VerifyFile | general | 2 ファイル連続 → 別々の fresh `<out>` | two ok samples, fake exe logs own path | both `accepted`, two distinct workdirs logged |
| UT-DialectVerify-107 | CollectPascalFiles | general | dir: `.pas`/`.dpr` のみ、`.pp`/`.txt` 除外、ソート | dir with 4 files | 2 files, sorted |
| UT-DialectVerify-108 | CollectPascalFiles | general | 大文字 `.DPR` も収集 | `UPPER.DPR` | collected |
| UT-DialectVerify-109 | IsGoodOutcome | general | accepted/static-ok/skipped → True | 3 outcomes | all `True` |
| UT-DialectVerify-110 | IsGoodOutcome | general | 残り 8 outcome → False | all others | all `False` |
| UT-DialectVerify-111 | VerifyMain | general | 混在引数 (file + dir) 両方検証 | `x.pas ./d` (good) | 2 lines, exit 0 |
| UT-DialectVerify-112 | VerifyMain | general | PATH なし → usage, exit 2 | `--timeout 5` only | `False`, exit 2, usage line |
| UT-DialectVerify-113 | VerifyMain | general | `--dialect` 不正値 → usage, exit 2 | `--dialect wat x.pas` | `False`, exit 2 |
| UT-DialectVerify-114 | VerifyMain | general | `--dialect delphi x.pas` 有効 | + fake unleash fpc | forced line, outcome decides exit |
| UT-DialectVerify-115 | VerifyMain | general | 同名 file 2 引数 → 2 行 | `x.pas x.pas` | 2 identical lines, exit per outcome |
| UT-DialectVerify-116 | VerifyMain | general | 良しと悪し混在 → exit 1 | good.x + bad.x | 2 lines, exit 1 |
| UT-DialectVerify-117 | VerifyMain | general | 全て `skipped` → exit 0 | no fpc in PATH, 1 src | 1 `skipped` line, exit 0 |
| UT-DialectVerify-118 | VerifyFile | objfpc | `.expected` 空 + 実行出力空 → バイト一致 → `accepted` (029 承継) | empty expected, empty stdout | `accepted` |
| UT-DialectVerify-119 | VerifyFile | objfpc | `.expected` 空 + 出力非空 → `wrong-output` (030 承継) | expected `''`, actual `x` | `wrong-output` |
| UT-DialectVerify-120 | VerifyFile | objfpc | CRLF stdout vs LF expected → バイト不一致 → `wrong-output` (033 承継) | expected `ok\n`, actual `ok\r\n` | `wrong-output` |
| UT-DialectVerify-121 | ReadEnvRec | general | `AFpc` 渡時はその値 (052 新形) | `ReadEnvRec('/fp','','/h')` | `Fpc='/fp'` |
| UT-DialectVerify-122 | ReadEnvRec | general | 両渡されず → デフォルト (053 新形, `AHome='/h'`) | `ReadEnvRec('','','/h')` | `Fpc='fpc'`, `FpcUnleashed='/h/unleashed/fpc/bin/fpc'` |
| UT-DialectVerify-123 | ReadEnvRec | general | `AFpcUnleashed` 渡時はその値 (054 新形) | `ReadEnvRec('','/up','/h')` | `FpcUnleashed='/up'` |
| UT-DialectVerify-124 | DetectDialect | objfpc | ファイル先頭語検出の前置き空白をスキップ (既定へ) | 先頭にスペースのテキスト `   hello world`、ファイル `f.pas` | `objfpc` |
| UT-DialectVerify-125 | DetectDialect | fpc-delphi | CRLF 改行の行でも mode 指示行を検出 | `{$MODE Delphi}` の CRLF 終了行 | `fpc-delphi` |
| UT-DialectVerify-126 | ExtractExpectErrorText | general | ディレクティブ文本後の CRLF は除去 | `// expect-error: broken proc` + CRLF 終了 | `broken proc` |
| UT-DialectVerify-127 | ExtractExpectErrorText | general | 長い単語に埋もれたマーカーは指令でない | `xexpect-error: broken proc` | `''` |
| UT-DialectVerify-128 | FirstCompilerErrorLine | general | CRLF 終了のエラー行も返す | 任意行 + CRLF で終わる `Error: boom` 行 | `Error: boom` |
| UT-DialectVerify-129 | ExpectedPathFor | general | ディレクトリ部分なし (裸ファイル名) | `unit.pas` | `unit.expected` |
| UT-DialectVerify-130 | IsPathUsable | general | 空パスは使用不可 | `''` | `False` |
| UT-DialectVerify-131 | IsPathUsable | general | 存在するが読めないファイル (read 権限なし) は False | `chmod 000` 済みのファイル | `False` |
| UT-DialectVerify-132 | CompileStep | delphi | delphi 書き換えでソースが読めない場合 → csFail | 存在しない `s.pas`, ADelphi=True | `csFail`, `AExePath=''` |
| UT-DialectVerify-133 | VerifyFile | objfpc | T06c: プログラムは **`<out>` 内で実行** — exe `s` は相対名 `data.txt` を読み取る。`data.txt` は fake fpc が exe と同じ `<out>` に書き出す (facts: "run the program inside `<out>`") | `s.pas` 最小; `s.expected = FROM-OWN-DIR`+`read-ok` (LF); カスタム fake fpc: `data.txt` → `<out>`, exe = `cat data.txt`/`echo read-ok` | `OK=True`; `accepted` (修正前は継承 cwd なので相対 `data.txt` 読込失敗 → `wrong-output`) |
| UT-DialectVerify-134 | RunVerifyCLI | general | T06c: `--help` は usage を出力し終了コード 0 を返す (VerifyMain の True パス) | `AArgs=['--help']`, `AEnv=ReadEnvRec('','','')`, 任意のワークルート | 戻り値 `0`; 標準出力に usage 文 |
| UT-DialectVerify-135 | RunVerifyCLI | general | T06c: 引数なしは usage + usage-error 分岐 (`if not VerifyMain` → `Result := 2`) | `AArgs=[]` (空配列), `AEnv=ReadEnvRec('','','')` | 戻り値 `2`; 標準出力非空 (usage 文) |
| UT-DialectVerify-136 | RunVerifyCLI | objfpc | T06c: 実ファイル 1 つ + fake fpc は `accepted` 行を出力し終了コード 0 を返す (VerifyMain が 0 で True を返すパス) | `s.pas` 最小; `WriteFakeFpc('echo ok')`; `AEnv.Fpc=<fake>`; `AArgs=['--dialect','objfpc','<s.pas 絶対パス>']` | 戻り値 `0`; 標準出力に `accepted` を含む 1 行 |
| UT-DialectVerify-140 | ExtractExpectErrorText | general | T06c2 BUG-001: 大文字小文字が混ざった pitfall 指示 → 文字の大小はそのまま保持 (ログ一致は大文字小文字区別、`grep -F` と同) | `{ expect-error: SyNtAx ErRoR }` (mixed case) | `SyNtAx ErRoR` (unchanged) |
| UT-DialectVerify-141 | ChooseCompiler | general | T06c2 BUG-002: `fpc` のような裸のコンパイラ名は `PATH` で解決される (ゲートは `IsPathUsable`; 修正前は裸名 → `skipped`) | `Fpc='fpc', fpc in $PATH` | `Result='fpc', class='compiled-run'` |
| UT-DialectVerify-142 | RunLimited | general | T06c4 BUG-003: 裸のコンパイラ名 (no `/`) は `PATH` で解決 — `RunLimited` は cwd 前の再解決 (relative→absolute) が先で `<cwd>/<name>` になり ENOENT → `process error`/`Result=False`. `IsPathUsable` (BUG-002) は裸名を `ExeSearch` 済みだが、`RunLimited` は解いていなかった. 修正後: 裸名は `ExeSearch` で絶対パス化し実行. 裸名が `ExeSearch` できなければ従来の cwd 接頭 (相対 exe の挙動) にフォールバック | 裸名 `echo` (PATH 上、cwd にファイルなし), `RunLimited('echo', ['dv142-marker'], '', 5000, …)` | `Result=True`, exit 0, 出力 `dv142-marker` (修正前は `Result=False`, 出力 `process error`) |
