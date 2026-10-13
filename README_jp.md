# tch-fpc

PyTorch の C++ API (libtorch) に対する Free Pascal バインディング。
[tch-rs](https://github.com/LaurentMazare/tch-rs) に触発されたプロジェクト。

**状態:** 実装の初期段階（Linux x86-64、FPC 3.2.2、libtorch 2.13.0）。
CPU で動き、libtorch の ROCm ビルドを通じて GPU でも動く（AMD RX 9070 と
`2.13.0+rocm7.1` で確認）。CUDA 向けのビルド経路はあるが、NVIDIA のハードウェアが
ないため未確認。生成した 2,657 個の op と手書きの基本 API を持つテンソル、自動微分、
`TchNN`（初期化、save/load 付きの var store、SGD/Adam/AdamW/RmsProp、Linear、
Conv1D/2D/3D、ConvTranspose1D/2D/3D、BatchNorm、LayerNorm、GroupNorm、Embedding、
LSTM/GRU、Identity、Sequential）、`TchJit`（TorchScript モジュールの読み込み、実行、
トレース、ファインチューニング、TorchScript のオブジェクト）が本物の libtorch で動く。
API は今後も変わりうる。

ROCm または CUDA 版の libtorch については [docs/DESIGN.md](docs/DESIGN.md) の 4.8 節を
参照。libtorch のディレクトリは手動でリンクし（`-DTCH_LIBTORCH_DIR=...`）、バリアント
ごとに `build/libtch-<variant>` を 1 つ作る。ROCm 版の libtorch の zip にはシンボリック
リンクが含まれず、展開後に 8 つ追加する必要がある。一覧は 4.8 の "Results" にある。

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
