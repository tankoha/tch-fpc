# tch-fpc

PyTorch の C++ API (libtorch) に対する Free Pascal バインディング。
[tch-rs](https://github.com/LaurentMazare/tch-rs) に触発されたプロジェクト。

**状態:** 実装の初期段階。C ラッパーは libtorch 2.13.0 でビルドでき、設計の核は
FPC 3.2.2 で検証済み。生成した 2,657 個のテンソル op はコンパイルできるが、
動かしたのはモックの libtch に対してだけで、libtorch ではまだ。まだ使えるものはない。

## 名前の付け方と tch-rs との違い

- 生成する op は PyTorch/tch-rs の名前をスネークケースのまま使う
  （`t.max_pool2d(...)`、`t.add_(u)`）。手書きの API は Pascal 流のキャメルケース
  （`Shape`、`Kind`、`ToString`）。
- Pascal の予約語や修飾子と同じ名前は、op も引数も `&` でエスケープする：
  `t.&div(u)`、`t.&var(0)`、`add_out(&out, ...)`。例外は `result` という引数
  （3 op）だけで、Pascal では `Result` が関数の戻り値なので `result_` にする。
- それ以外の引数名もそのまま（`dim`、`indices`、`pow`）。`Dim` や op の `indices` の
  ような `TTensor` のメンバーと同じ名前でも変えない。そのために `Tch.pas` では
  `{$modeswitch duplicatelocals}` を有効にしている。Delphi モードのうち「引数が
  メンバーを隠してよい」という部分だけで、Rust が暗黙にそうしているのと同じ
  （引数はローカル変数、メソッドは `self.dim()` で呼ぶ）。利用側のユニットには
  影響しない。スイッチは `Tch.pas` の中だけ。

## ライセンス

次のいずれかを選択できる:

- Apache License, Version 2.0（[LICENSE-APACHE](LICENSE-APACHE)）
- MIT ライセンス（[LICENSE-MIT](LICENSE-MIT)）

tch-rs と同じ。本プロジェクトは tch-rs の C ラッパーとコード生成器をフォークしている。

明示的に別段の表明がない限り、本プロジェクトへの取り込みを意図して提出された
貢献（Apache-2.0 ライセンスの定義による）は、追加の条件なしに上記の
デュアルライセンスで提供されるものとする。

正は英語版: [README.md](README.md)
