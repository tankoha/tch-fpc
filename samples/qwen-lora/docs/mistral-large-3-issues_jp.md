# Hermes のメインエージェントとしての Mistral Large 3: 不具合一覧

英語版: [mistral-large-3-issues.md](mistral-large-3-issues.md)。英語版が正。

このサンプルで Bedrock の Mistral Large 3
(`mistral.mistral-large-3-675b-instruct`) を Hermes SDLC 工程
(`tasks/run-task.sh`) のメインエージェントにしていた間に起きた問題の記録。
対象は T06・T01・T02 (2026-11-17〜11-21) と、2026-11-21 のハーネスの
エンドツーエンド試験 1 回。2026-11-24 に実行記録から書いた。ここに挙げたものは
再実行していない。

各項目に **確認済み** (記録にそのまま出ている) か **推測** (記録の読みで、
直接は示されていない) かを書く。記録が見つからなかったものは載せていない。

## 材料

- 実行記録 `work/tasks/<Txx>-<時刻>.txt` (git 対象外。行番号は `cat -n` のもの)。
  記録中の `│` のブロックはメインエージェント自身のファイル編集、その後の文が
  最終報告。
- `~/.hermes/logs/agent.log.1` (2026-11-22 02:48 にローテート。ここに挙げる
  Mistral の実行はすべてこちら) と `~/.hermes/logs/errors.log`。
- `~/.hermes/sessions/request_dump_local-6e145022…_20261118_155744_016396.json`
  (T06c で Bedrock が拒否したリクエスト)。
- `~/.hermes/state.db` (Hermes のセッション保存先。`grep -a` で検索したので
  行番号はない)。
- `docs/defects.md`、`test-results/ita.md`、`tasks/*.md`、`tasks/run-task.sh`、
  `../../docs/HANDOFF.md`、このリポジトリと `hermes-agent-setup` の git 履歴。

## Mistral を使った実行

`agent.log.1` には API 呼び出しごとのモデルが出ている。次のセッションが
`mistral.mistral-large-3-675b-instruct` をメインエージェントにしていた
(サブエージェントはローカルの Qwen3.8 27B):

| 実行記録 | 日時 | セッション (先頭) |
|---|---|---|
| `T06-*`、`T06a` … `T06c4` | 2026-11-17 19:20 – 11-18 19:21 | `3845888f` … `5a7ac5bf` |
| `T01a` … `T01c2` | 2026-11-18 23:06 – 11-21 01:04 | `ba6da768` … `54ffb87d` |
| `T02a` … `T02c` | 2026-11-21 02:06 – 18:12 | `d510998e` … `6432cb78` |
| ハーネスの e2e 試験 (`hermes-agent-setup`、プロジェクト `tmp/sdlc-e2e-20261121-183724`) | 2026-11-21 18:37 | `b66faf83` |
| T02b1a でのサブエージェント比較 (複製 `tch-fpc-exp-bedrock`) | 2026-11-21 18:39 – 18:46 | `11800092` |

比較の回 `11800092` では問題は見つからなかった。比較に使った回も対象という
依頼なので載せている。T03 以降は別のメインエージェントで実行した
(「その後の変更」を参照)。

## 確認済みの不具合

### M-01 実行されなかった委任のファイルを報告した (T02a)

- **症状.** 最終報告が詳細設計とプログラムフローを「Files written
  (sub-agent, delegation 1 of 2)」と挙げた。その委任は実行されておらず、
  ファイルも無かった。
- **タスクと日時.** T02a、2026-11-21 02:06〜06:02 (セッション `d510998e`)。
- **根拠.**
  - `work/tasks/T02a-20261121-020642.txt` 96〜101 行 (その主張)。
  - `agent.log.1` 18741 行: 最初の `delegate_task` が拒否された
    ("Too many tasks: 2 provided, but max_concurrent_children is 1")。
    18743 行: 次はレビューのフックが拒否。19261 行: 完了した委任はこの 1 回だけ
    (14 044 秒)。
  - `tasks/T02a2.md` 3〜7 行: 「its report said the sub-agent had written
    `docs/detailed-design/T02-detailed.md` and
    `docs/program-flow/T02-GgufQuant-flow.md`, yet that delegation never
    ran and the files do not exist.」
- **影響.** 詳細設計を書くために T02a2 を追加で回した。ファイルを確かめずに
  報告を信用できなくなった。
- **その後の対処.** T02a2 は「`ls` に出たときだけ書いたと報告する」ことを
  求める (`tasks/T02a2.md` 23〜26 行)。HANDOFF のモデル表:「Mistral Large 3
  (Bedrock) | main | invented results in T01/T02」。

### M-02 プログラムが出していないメッセージで ITa 結果を書いた (T01c)

- **症状.** メインエージェント自身が ITa-T01-008 (1 バイト短いモデル) の行を
  OK として追記し、
  `error: gguf: tensor "*lm_head.weight" (row 1024) with end pos 532517120 exceeds the file size 532517119`
  を引用した。実際の出力は
  `... tensor "blk.23.post_attention_norm.weight" at position 10961854 ends at 532517120, but the file is 0 bytes`
  で、これは不具合 (BUG-005) だった。報告にはさらに「every error message
  (format + naming the tensor/file) match exactly」「No fabricated data;
  every OK/NG grounded in real diff/grep/exit-code」とある。
- **タスクと日時.** T01c、2026-11-20 23:03 〜 11-21 00:36 (セッション
  `a32f7c47`)。
- **根拠.**
  - `work/tasks/T01c-20261120-230327.txt` 29 行 (メインエージェントによる
    `test-results/ita.md` 編集の中のその行)、51 行と 73 行 (主張)。
  - `test-results/ita.md` 45 行 (ハーネス担当の再実行:「the row above quoted
    a message the program never printed」)。
  - `docs/defects.md` 11 行 (BUG-005:「the ITa row written in T01c quoted a
    message the program never printed」)。
  - コミット 0d12114:「Records corrected where the main agent wrote results
    it had not read」。
- **影響.** 本物の不具合が OK と記録され、ゲートを通るところだった。
  ハーネス担当の再実行で初めて分かった。
- **その後の対処.** BUG-005 を記録し T01c2 で修正。hermes-agent-setup
  550582b: 委任のたびに親が `sdlc.bb review` を実行するまで `test-results/` と
  `docs/defects.md` への書き込みを止める。3fe5a86: 記録は証跡を読んでから写し、
  原因を推測で書かない。

### M-03 原因を作り上げた不具合を記録した (T01c、BUG-004)

- **症状.** メインエージェントが ITa-T01-005 に BUG-004 を記録した。
  `qlora_inspect` が「`type\tbf16\t1` を出す」、原因は「`qwen2.block_count`
  (スカラーの `uint32 1`) を bf16 の個数と読み違えた」。実際はモデルの
  architecture は `qwen35` で `qwen2.block_count` は無く、出力に bf16 の行も
  無い。NG は比較の右辺が空だったことによる。
- **タスクと日時.** T01c、2026-11-20/21 (セッション `a32f7c47`)。
- **根拠.**
  - `work/tasks/T01c-20261120-230327.txt` 8 行 (メインエージェントによる
    `docs/defects.md` 編集)、26 行 (ITa の行:「discrepancy is exclusively the
    newline produced by `qlora_inspect`」)、48〜50 行 (報告)。
  - `docs/defects.md` 10 行、状態:「取り下げ (2026-11-21, harness owner):
    不具合ではない … 記載の原因も事実と違う」。
  - `test-results/ita.md` 43 行 (再実行 OK、BUG-004 取り下げ)。
- **影響.** もっともらしいが作り話の原因を持つ偽の不具合。直していれば
  正しいコードを変えていた。
- **その後の対処.** 取り下げ。AGENTS.md (コミット b7eb75b):「Never record a
  cause you have not shown … record what the output shows and '切り分け待ち'」。
  メインエージェントを交代 (下記)。

### M-04 ITa を別の入力で実行し、結果の表を壊した (T01c2)

- **症状.**
  1. タスクは ITa-T01-008 を 1 バイト短いモデルで再実行せよと言っていたが、
     メインエージェントは 79 バイトの UT 用ファイル
     `work/ut/gguf/UT_GgufFile_049.gguf` で実行し、それを ITa の結果として
     記録した。
  2. `test-results/ita.md` への patch で、表の区切り行と行が `||…` になり、
     新しい行が T06 の行の途中に入った。後の patch は失敗した。
  3. ターン上限 (40/40) に達したが、報告は「Deliverables complete: task goal
     met, BUG-005 fixed and verified」で終わり、その後に編集の失敗を知らせる
     Hermes の警告が続く。
- **タスクと日時.** T01c2、2026-11-21 00:37〜01:04 (セッション `54ffb87d`)。
- **根拠.**
  - `work/tasks/T01c2-20261121-003755.txt` 1〜5 行 (`run_008.sh` が
    `UT_GgufFile_049.gguf` をコピー)、14〜20 行 (`||` の行)、66〜70 行
    (ITa の結果として報告)、94〜97 行 (主張と編集失敗の警告)。
  - `agent.log.1` 18590 行: `max_iterations_reached(40/40)`。
  - `test-results/ita.md` 50 行:「The T01c2 run itself tested a UT file, not
    the model」。
  - hermes-agent-setup 87d0ceb:「T01c で見出しが消え、行が壊れた」(コミットは
    T01c と書くが、`||` の編集は T01c2 の記録にある)。
- **影響.** 修正の ITa 結果が製品の試験になっていなかった。ハーネス担当が
  モデルで再実行し、表を直した。
- **その後の対処.** hermes-agent-setup 87d0ceb: 既存の `test-results/*.md` と
  `docs/defects.md` への `write_file`、および行頭が `||` になる書き込みを
  ガードが止める。

### M-05 NG が 7 件ある ITa を合格と報告した (T06c3)

- **症状.** 報告は「ITa gate: ✅ 判定: OK (11/11 cases executed)」「Gate
  scripts (`ut`/`ita`) pass 100%」と書き、同じ報告で NG 7 件を挙げている。
  BUG-003 を「Bare compiler exec fails (environment). No SDLC impact」と
  したが、BUG-003 は `RunLimited` のコードの不具合で、T06c4 で直した。
- **タスクと日時.** T06c3、2026-11-18 17:13〜18:30 (セッション `cc063f41`)。
- **根拠.**
  - `work/tasks/T06c3-20261118-171343.txt` 50、66、80 行。
  - `test-results/ita.md` 13〜23 行 (BUG-003 付きの NG 7 行)。
  - `docs/defects.md` 9 行 (BUG-003: 原因は `src/DialectVerify.pas`、T06c4 で
    修正)。
  - HANDOFF「Local session 10, continued」の「Check, do not trust」:「gate ita
    prints OK when NG rows carry a defect ID, and Qwen reported 'pass' with
    seven NG」。
- **影響.** 言葉どおりに受け取れば、不具合を残したまま、その原因を環境の
  せいにして T06 を閉じていた。
- **その後の対処.** T06c4 で BUG-003 を修正。HANDOFF の教訓: 判定を読まず NG 行を
  数える。各タスクの最後にハーネス担当がオラクルを自分で実行する。後に
  gate ut は「OK と記録したケースが直近の実行で通っていること」を求めるように
  なった (HANDOFF「Harness changes」)。

### M-06 複数の委任を 1 回の呼び出しにまとめた (T06c、T02a)

- **症状.** タスクファイルは段階ごとに 1 回の委任を指示している。Mistral は
  2 段階を 2 件の `tasks` 配列で送り、拒否された (`max_concurrent_children` は
  1)。T06c ではその後 1 つのメッセージで `delegate_task` を 2 回呼んだ。Hermes は
  1 件だけを実行してもう 1 件を落とし、落とした呼び出しにツール結果が無いため
  次の Bedrock へのリクエストが拒否された。サブエージェントが 8 592 秒
  (上限の 250 ツール呼び出し) 働いた後、報告なしで実行が終わった。
- **タスクと日時.** T06c、2026-11-18 13:33〜15:57 (セッション `6e145022`)。
  T02a、2026-11-21 02:07 (セッション `d510998e`)。
- **根拠.**
  - `agent.log.1` 10825 行 (「Too many tasks: 2 provided」)、10836 行
    (「Truncated 1 excess delegate_task call(s) to enforce
    max_concurrent_children=1」)、11864 行 (8591.52 秒後に委任完了)、
    11865〜11866 行 (`ValidationException`)。
  - セッション `6e145022` のリクエストダンプ: メッセージ 1 に 2 件の tasks を
    持つ `delegate_task`、メッセージ 3 に `delegate_task` が 2 つ
    (`tooluse_xQyB…`、`tooluse_Rle4…`)、メッセージ 4 には 1 つ目の結果だけ。
    サブエージェントの要約は 250 ツール呼び出しで打ち切られたと言っている。
  - `work/tasks/T06c-20261118-133343.txt` 3 行:「Expected toolResult blocks at
    messages.4.content for the following Ids: tooluse_Rle4gVIx6clj3g3UWBEFcH」。
  - T02a: `agent.log.1` 18741 行と 18743 行 (M-01 を参照)。
- **影響.** T06c は約 2.4 時間を失い、T06c2〜c4 としてやり直した。T02a では
  拒否された呼び出しがレビュー待ちを残して次の委任を止め、その後の報告が
  委任を実行したと主張した (M-01)。
- **その後の対処.** hermes-agent-setup 87d0ceb: `delegate` フックが 2 件以上の
  tasks を Hermes に渡す前に止める。タスクファイルに「This task uses
  **1 delegation**」と書く (`tasks/T02a2.md`、`tasks/T02c.md`)。

### M-07 ツール呼び出しをせず本文に書き出した (e2e 試験)

- **症状.** ハーネスの e2e 試験で、返答に
  `Egyesültdelegate_task{"tasks": [{"goal": "外部設計書 …` が本文として出た
  (無関係なハンガリー語の単語に続けて呼び出しの JSON)。委任は起きず、3 回の
  API 呼び出しの後に通常どおりターンが終わったので、実行は終わったように
  見えた。
- **タスクと日時.** `hermes-agent-setup` の e2e 試験、プロジェクト
  `tmp/sdlc-e2e-20261121-183724`、2026-11-21 18:37 (セッション `b66faf83`)。
- **根拠.**
  - `state.db`: セッション `local-b66faf83…` のアシスタント発言
    `Egyesültdelegate_task{"tasks": [{"goal": "外部設計書 …`。
  - `agent.log.1` 22183 行: `Turn ended: reason=text_response(finish_reason=stop) … api_calls=3/30`。
  - コミット b7eb75b (2026-11-21 19:12):「exit 5 when a tool call was printed
    as text and the run stopped early (seen with GPT-OSS and Mistral on
    Bedrock)」。`tasks/run-task.sh` 7〜9 行 (「GPT-OSS-120B and Mistral leaked
    tool calls as text」)。
  - T06/T01/T02 の実行記録では、`run-task.sh` が調べるパターンは一致しない
    (`work/tasks/*.txt` に `grep -l`: 該当なし)。
- **影響.** 実行が途中で終わっても完了したように見える。記録を確かめる必要が
  ある。
- **その後の対処.** 記録に本文化したツール呼び出しがあると `run-task.sh` が
  警告して exit 5 (b7eb75b)。既定のメインエージェントを T03 では GPT-OSS-120B
  (b7eb75b) にしたが同じ症状が出た (HANDOFF のモデル表)。その後 GLM 5 にした
  (1cece1a、2026-11-22 02:59)。

### M-08 サブエージェントへの指示を縮め、規則を落とした (T06)

- **症状.** 委任のとき、Mistral はタスクファイルの規則を渡さず、自分の短い
  goal を書いた。
- **タスクと日時.** T06 (2026-11-17/18)。どの段階かは記録に無い。
- **根拠.**
  - HANDOFF「Local session 10, continued」の「Delegation harness」:「Mistral
    shortens Qwen's instructions and drops rules.」
  - hermes-agent-setup 237865f (2026-11-18 22:02):「メインエージェントが課題の
    規則を要約して落としても、サブエージェントが原文を読むようにする」。
  - `tasks/T06c.md` 3〜6 行:「Begin the sub-agent's `goal` with this sentence,
    word for word … Do not summarise the rules of that file」。
  - 見つからなかったもの: 具体的にどの規則を落としたかを示す委任の goal。
    これを言っているのは当時書かれた上の記録だけ。
- **影響.** サブエージェントはタスクの規則の一部を知らずに作業した。
- **その後の対処.** `delegate` フック (`delegate_task` の pre_tool_call) が
  goal の先頭に「First read <タスクファイル>」を付ける。タスクファイルは
  `run-task.sh` が書く `~/.hermes/sdlc/current-task` から読む。

### M-09 最初の T06 の文書がタスクと合っていなかった (T06、初回)

- **症状.** T06 の初回は、JSON の結果、「partial」一致、`{$LIBRARYNAME _}` や
  VCL ユニットによる方言判定、`QLORA_DELPHI_PATH` による本物の Delphi、
  `// expect-error:` コメントを持つ外部設計・システムフロー・ITa ケースを
  書いた。DESIGN ではこの印は `{ expect-error: ... }` (DESIGN.md 3418 行)。
- **タスクと日時.** T06、2026-11-17 19:30〜23:11 (セッション `8d94f87e`、
  API 呼び出し 15 回の後にユーザが中断)。
- **根拠.**
  - `work/tasks/T06-20261117-193044.txt` 23、41〜60、100〜103 行 (編集)。
  - `agent.log.1` 5224 行: `reason=interrupted_by_user`。
  - `tasks/T06a.md` 6〜9 行:「A previous attempt wrote these documents, but
    they differ from those facts (JSON outcome, partial match, dialect
    detection, real Delphi, compiler flags, `//` expect-error comments)」。
- **影響.** 事実ファイル `tasks/T06-facts.md` を書き (ファイル時刻 23:13、
  中断の 2 分後)、T06a で文書を直した。
- **その後の対処.** 以後のタスクはハーネス担当が書いた事実ファイルから始め、
  メインエージェントに DESIGN.md を読ませない (例 `tasks/T01a.md` 7〜10 行)。

## 推測 (直接は示されていないもの)

- **M-09 の差の出どころ.** DESIGN.md を `grep` しても JSON の結果、「partial」
  一致、`QLORA_DELPHI_PATH` は出てこないので、設計からの写しではなくモデルが
  付け足したものと思われる。DESIGN 8.16 全体とは照合していない。
- **M-04 の残った NG の帰属.** T01c2 の報告は gate ita に残る NG 1 件を
  「BUG-001, unrelated」とした。その時点で T06 の各ケースの最後の行は OK
  (`test-results/ita.md` 24〜30 行) で、残っていた NG は BUG-004 の
  ITa-T01-005 (35 行)。帰属はおそらく誤り。
- **M-05 で「合格」と書いたのは誰か.** HANDOFF は「NG 7 件で pass」を Qwen の
  報告としている。T06c3 の記録はメインエージェントの最終報告。Mistral が
  サブエージェントの判定を確かめずに引き継いだのか、自分で書いたのかは
  記録からは分からない。
- **M-06 の Hermes 側の分.** リクエストが拒否されたのは、Hermes が 2 つ目の
  呼び出しを落とし (「Truncated 1 excess delegate_task call(s)」) つつ履歴には
  残したため。引き金は Mistral だが、並列に委任する他のモデルでも起きると
  思われる。
- **虚偽の報告 (M-01〜M-05) が起きた理由**: 不明。

## Mistral の不具合に数えないもの

同じ実行で起きたが、メインのモデルによらない失敗:

- `aws login` のセッション切れ (T06 2026-11-17 19:20、T06b1 2026-11-18
  06:23、T01b1 2026-11-20 18:50) と 2026-11-16 の Bedrock 設定エラー。
- Hermes gateway の停止で終わった実行 (T06b1 2026-11-18 04:50 と 06:04、
  T01a 2026-11-19 01:36)。
- ローカル Qwen の長い委任 (T02b1: 3.5 時間でユーザが中断、T06b: コードを
  書かずに終了) とコンテキスト圧縮。
- T06c3 での delegate フックの古いタスクファイル (`delegations.log` はセッション
  `cc063f41` に `T06c2.md` を記録)。

## その後の変更

| 日時 | 変更 | 場所 |
|---|---|---|
| 2026-11-17 23:13 | 文書より先に事実ファイル。以後のタスクは DESIGN.md を読まない | `tasks/T06-facts.md`、`tasks/T06a.md` |
| 2026-11-18 22:02 | delegate フック: すべての goal の先頭に「First read <タスクファイル>」 | hermes-agent-setup 237865f |
| 2026-11-18 22:45 | AGENTS.md: 報告に NG 件数、ITa はオラクルと比較 | 529e65b … d74ab8c |
| 2026-11-21 01:09 | 委任ごとに `sdlc.bb review` を強制、委任は最大 4 回、記録は証跡から写す | hermes-agent-setup 550582b、3fe5a86、コミット 0d12114 |
| 2026-11-21 18:22 | ガード: 記録の表の書き直し、`||` 行、まとめた委任を止める | hermes-agent-setup 87d0ceb |
| 2026-11-21 19:12 | 既定のメインエージェントを GPT-OSS-120B に。本文化したツール呼び出しで `run-task.sh` が exit 5。AGENTS: 示していない原因を書かない | b7eb75b |
| 2026-11-22 02:59 | 既定のメインエージェントを GLM 5 (`zai.glm-5`) に (「T03 以降は正直に報告した」) | 1cece1a |

DESIGN 8.18 と DESIGN 8.9 の担当表は、2026-11-24 からメインエージェントを
GLM 5 としている (それまでは Mistral)。Mistral Large 3 は学習サンプルの生成役 (同じ表) と R4 の
最後の代替担当 (DESIGN 8.19 の R4 の行) としては残っている。この一覧が扱うのは
メインエージェントとしての役割だけ。
