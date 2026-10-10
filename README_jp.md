# tch-fpc

PyTorch の C++ API (libtorch) に対する Free Pascal バインディング。
[tch-rs](https://github.com/LaurentMazare/tch-rs) に触発されたプロジェクト。

**状態:** 実装の初期段階。C ラッパーは libtorch 2.13.0 でビルドでき、設計の核は
FPC 3.2.2 で検証済み。生成するテンソル API はまだない。まだ使えるものはない。

## 名前の付け方と tch-rs との違い

- 生成する op は PyTorch/tch-rs の名前をスネークケースのまま使う
  （`t.max_pool2d(...)`、`t.add_(u)`）。手書きの API は Pascal 流のキャメルケース
  （`Shape`、`Kind`、`ToString`）。
- Pascal の予約語や修飾子と同じ名前の op は `&` でエスケープする：`t.&div(u)`、
  `t.&var(0)`。
- **引数名の末尾に `_` が付くことがある**（`dim_`、`indices_`、`pow__`）。
  `{$mode objfpc}` では、メソッドの引数に record のメンバーと同じ名前を使えない。
  すべての op は `TTensor` のメンバーなので、手書きのメソッドや別の op（`indices` など）と
  同じ名前の引数には、重ならなくなるまで `_` を足す。Rust にはこの衝突がない：引数は
  ただのローカル変数で、メソッドは `self.dim()` の形でしか呼べないので、tch-rs では
  `fn dim(&self)` と `dim: i64` が共存できる。引数名は呼び出し側には出てこないので、
  影響はシグネチャの見た目だけ。
- 同じ理由で、手書きの API はよくある引数名を避ける：次元数は `Dim` ではなく
  `NumDims`（`dim` は約250の op の引数）。

## ライセンス

次のいずれかを選択できる:

- Apache License, Version 2.0（[LICENSE-APACHE](LICENSE-APACHE)）
- MIT ライセンス（[LICENSE-MIT](LICENSE-MIT)）

tch-rs と同じ。本プロジェクトは tch-rs の C ラッパーとコード生成器をフォークしている。

明示的に別段の表明がない限り、本プロジェクトへの取り込みを意図して提出された
貢献（Apache-2.0 ライセンスの定義による）は、追加の条件なしに上記の
デュアルライセンスで提供されるものとする。

正は英語版: [README.md](README.md)
