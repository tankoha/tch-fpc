# 開発プロセス規約 (SDLC)

（[AGENTS.md](AGENTS.md) の日本語訳です。内容が食い違う場合は英語版が正です。）

このプロジェクトは工程と担当モデルを分けて開発する。メインエージェント (Mistral Large 3) と
サブエージェント (Qwen3.8、`delegate_task` で起動) の両方がこの規約に従う。
規約の一部はシェルフック (`sdlc.bb`) が機械的に強制する。フックにブロックされたら、
回避せずにブロック理由に従うこと。

ファイル操作 (`write_file`、`patch`) のパスと `terminal` の `workdir` は、常に絶対パスで指定する。
相対パスだとフックがプロジェクトを判断できず、操作が止められる。

## 担当

| 工程 | 担当 | 成果物 |
|---|---|---|
| 外部設計 | メイン (Mistral) | `docs/external-design/`、`docs/system-flow/` |
| ITa / ITb / ST テストケース作成 | メイン (Mistral) | `tests/ita/cases/`、`tests/itb/cases/`、`tests/st/cases/` |
| 詳細設計 | サブ (Qwen) | `docs/detailed-design/`、`docs/program-flow/` |
| UT ケース作成・実装・UT 実施 | サブ (Qwen) | `tests/ut/`、`src/`、`test-results/ut.md` |
| ITa / ITb 実施 | サブ (Qwen) | `test-results/ita.md`、`test-results/itb.md` |
| ITa / ITb の NG 判定の確定・原因の切り分け | メイン (Mistral) | `docs/defects.md` |

- メインは詳細設計以降 (`src/`、`docs/detailed-design/`、`docs/program-flow/`、`tests/ut/`) を
  自分で書かない。必ず `delegate_task` で委任する。
- サブは外部設計書・システムフロー・ITa/ITb/ST のテストケースを書き換えない。
  変更が必要だと判断したら、最終報告に「外部設計変更依頼」として理由と箇所を書いて返す。

## 委任のしかた (メイン向け)

サブは親の会話を一切知らない。`delegate_task` の `goal` と `context` に次を必ず入れる。

- `skill_view` で読ませるスキル：詳細設計なら `sdlc-detailed-design`、UT・実装なら `sdlc-ut-tdd`、
  ITa/ITb 実施なら `sdlc-it-execution`。それに加えて常に `pascal-fpc-objfpc` と `tch-fpc`
- 対象の外部設計書・システムフローのパス
- そのタスクの元になる `../../docs/DESIGN.md` の節
- 完了条件 (どの成果物をどう更新したら完了か)

1 回の委任は 1 ユニットか 1 プログラムに絞る。サブのコンテキストは 64k しかない。

## 設計書の更新

- 修正・変更で外部設計書やシステムフローの内容が変わる場合、メインは同じ作業の中で更新する。
- 修正・変更で詳細設計書やプログラムフローの内容が変わる場合、サブは同じ作業の中で更新する。
- 更新しない場合は、更新不要と判断した理由を最終報告に書く。

## テスト

- 実装の前にテストケースを作成する。UT ケース表 `tests/ut/cases/<ユニット>.cases.md` がない
  ソースファイルは書けない (フックが止める)。
- テストケースは削除しない。ケース・テスト手続き・結果の記録を減らす変更はフックが止める。
  不要になったケースは削除せず、備考に「[廃止]」と理由を書く。
- テストが OK にならない場合は報告する。可能であれば事象と原因を `docs/defects.md` に記録し、
  結果表の不具合 ID 欄にその ID を書く。書式はスキル `sdlc-defect-record` に従う。

## UT 完了条件

次の 2 つを両方満たしたときに限り UT 工程完了とする。判定は `sdlc.bb gate ut . --run` で行う。

1. カバレッジ：UT ケースを実行したとき、`src/` の行が 100% 実行される
   (`:coverage-kind :c0`。kcov は行を測り、分岐は測らない)。
2. 消化率：全ケースが「OK」か「NG (不具合 ID 記録済み)」で 100%。

ITa / ITb / ST の完了は消化率 100% (`sdlc.bb gate ita` など) で判定する。

## このプロジェクト：Pascal と tch-fpc

このサンプルは、Qwen3.8 27B を 2 枚の GPU で LoRA ファインチューニングするもので、tch-fpc の
上に Pascal で書く。設計は `../../docs/DESIGN.md` の 8.9〜8.19 にある。タスクの一覧と、
タスクごとの受け入れの基準 (オラクル) は 8.18。

- すべて Free Pascal 3.2.2 で、どのファイルも `{$mode objfpc}{$H+}`。ユニットは `Tch`、
  `TchNN` (tch-fpc。`../../src` にある)。HTTP と JSON は `fphttpclient`、`fpjson`、プロセスは
  `process`。Python と Node.js は使わない。bash は 1 行のラッパーだけ。
- `src/` は 1 ファイル 1 ユニットで、名前は DESIGN 8.12 のとおり (`GgufFile`、`GgufQuant` など)。
  プログラムは `src/qlora_*.pas`。
- UT は fpcunit で書く。`tests/ut/<ユニット>_test.pas` は、published なメソッド
  `procedure Test...` を持つ `TTestCase` のクラスを含むプログラムで、`consoletestrunner` で動かす。
  ケース ID は `UT-<ユニット>-NNN`。
- ビルドの行 (`.sdlc/ut-coverage.sh` が使うもの)：
  `fpc -Mobjfpc -Sh -gl -Fusrc -Fu../../src -FU<units> -Fl<libtch> -k-rpath=<libtch> <file.pas>`。
- モデルのファイルが要るテストは、パスを環境変数 (リハーサル用の GGUF なら `QLORA_MODEL`)
  から読み、無ければ飛ばす。
- 生成したデータ、モデルのファイル、結果はリポジトリに書かない。`work/` (git 管理外) に置く。
- テストは GPU を使わない。`.sdlc/ut-coverage.sh` が GPU を隠す。
- 数値計算のコードは、先に書いたオラクルで確かめる (DESIGN 8.18)：手で計算した例、遅い
  スカラー版、または `llama-server` からの参照値。
