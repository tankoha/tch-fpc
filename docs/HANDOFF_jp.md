# 引き継ぎ

（[HANDOFF.md](HANDOFF.md) の日本語訳です。内容が食い違う場合は英語版が正です。）

再開するときは最初にこれを読んでください。決定事項と理由は [DESIGN_jp.md](DESIGN_jp.md) にあります。

## 現状（2026-10-10、クラウド設計セッション2回目の終了時点）

- 3つの層すべての設計を DESIGN.md に書きました。まだ何もビルドしていません。
- 基準：tch-rs `4227b89b72059b0651ff83a38637693574e0a2d6`、libtorch 2.13.0。
- 決定済み：
  - tch-rs の libtch をフォークする（C の関数名は維持し、`PROTECT` を強化し、
    `extern "C"` をガードで囲む）。libtch は共有ライブラリ `libtch.so` にする。
  - `TTensor` は参照カウント付きインターフェースを包む record にする。
  - libtch を呼ぶたびにエラーを確認し、`ETorchError`/`ETorchNativeError` を投げる。
    省略不可のテンソル引数は呼び出し前に必ず nil ハンドルをチェックする。
  - 生成する op はスネークケース（予約語は `&` でエスケープ）、手書きの API は
    キャメルケースにする。
  - 複数出力の op は `out` 引数にする。末尾の単純な引数には生成時にデフォルト値を付ける。
    `IntListOption` には `TOptIntList` を使う。
  - フォークした gen.ml に一覧ファイルを出力させ、Pascal の生成器がそれを読む。
  - ライセンス：tch-rs と同じ MIT OR Apache-2.0（リポジトリ直下に `LICENSE-MIT`、
    `LICENSE-APACHE`）。取り込んだ tch-rs のファイルの横には tch-rs 自身の
    ライセンスのコピーを置く。
- セッション2では、4 章を tch-rs のソースと突き合わせてレビューし、4.6
  （nn / VarStore / オプティマイザ / JIT をクラスで、`TVarPath` は record で、
  `Forward(X, Train = False)` は1つに）と 4.7（勾配モード）を追加しました。
  FPC 3.2.2 の未検証の挙動は DESIGN.md で **(H1)**〜**(H11)** と印を付け、下に並べています。

## 次の作業（ローカル実装）

1. **ツールの確認。** FPC 3.2.2、g++、CMake を用意し、libtorch 2.13.0 の CPU 版
   （cxx11 ABI）を `libtorch/` に展開する（`.gitignore` 済み）。
2. **libtch の取り込み。** 上記コミットの tch-rs から `csrc/libtch/` と `gen/` に
   コピーし、LICENSE-MIT / LICENSE-APACHE を追加する。まず無改変のコピーを
   コミットし（メッセージに取り込み元のファイル名を書く）、2つ目のコミットで
   こちらのパッチを当てる：
   - ストリーム・Python・画像のコードを削除する。
   - `tchf_free` を追加する。
   - `PROTECT` に `catch (...)` を追加する。
   - `torch_api_generated.h` の `extern "C"` を `#ifdef __cplusplus` で囲む。

   CMake で `libtch.so` をビルドする。YAML
   （`third_party/pytorch/Declarations-v2.13.0.yaml`）はそのコミットの tch-rs
   リポジトリにありますが、浅い／部分的なクローンには含まれないことがあります。
   取得するのは再生成するときだけでよいです。
3. **C での動作確認**（Python ではなく小さな C プログラムで）：テンソルを作り、
   `atg_add` を呼び、表示する。わざとエラーを起こし、`get_and_reset_last_err` で
   読めることを確かめる。生成ヘッダーを include するには手順2の `extern "C"` の
   ガードが必要。
4. **Pascal での ABI 確認**（推測せず、実際に測る）：
   - `bool` の戻り値（`at_autocast_is_enabled` など）が `Boolean` として正しく読めるか。
   - `c10::ScalarType` の数値が `TKind` と一致するか。Float8 の 23〜26 も含む
     （型ごとのテンソルについて `at_scalar_type` を表示する）。
   - `uint8_t x_null` の省略フラグを `Byte` として、`int8_t` を `ShortInt` として渡せるか。
   - リンク：`external 'tch'` と rpath で、実行時に `libtch.so` と libtorch が見つかるか。
   - 省略不可の引数に `nullptr` のテンソルを渡すと本当にセグメンテーション違反になるか
     （nil ハンドルのチェックが必要なことの確認）。
5. **FPU マスクのテスト。**
   - `SetExceptionMask` なしで、CPU テンソルで `1.0 / 0` を計算すると FPC の
     プログラムが落ちるか。`Tch` の initialization での対処で直ることを確かめる。
   - **(H4)** メインスレッドで `SetExceptionMask` したあとに起動した `TThread` は
     そのマスクを引き継ぐか、FPC の既定に戻るか。全スレッドで
     `TchMaskFpuExceptions` を呼ぶ必要があるかがこれで決まる。
   - **(H5)** Pascal コードで `EZeroDivide` を発生させて捕まえ、そのあと NaN を出す
     テンソル演算を実行する。RTL の SIGFPE 処理のあともマスクは残っているか。
6. **所有権のテスト。** `(A + B) * C` を 10^6 回ループして RSS（常駐メモリ）が増えないことを
   確認する。`B := A` のあと `A := Default(TTensor)` としても B が生きていることを確認する。
   - **(H1)** `TTensorHandle.Destroy` に計測を入れ、`(A + B) * C` の一時値が
     いつ解放されるか調べる：文の終わりか、ループ1周の終わりか、ルーチンの終わりか。
   - **(H6)** `TTensor` のオープン配列コンストラクタを使った `TTensor.cat([A, B], 0)`：
     リークも二重解放もないこと。
   - **(H8)** `out` 引数の意味論：`T.max_dim(1, False, T, I)` がクラッシュではなく
     `ETorchError` になる（レシーバーが入口で解放される）こと。また、以前の値を持つ
     `V` を `out` に渡すと古いハンドルが解放されること。
7. **コンパイラが受け付けるかのテスト**（生成器ができる前に、小さなプログラムで）：
   - **(H2)** `array of Single` と `array of Double` のオーバーロードがあるとき、
     `FromArray([1.0, 2.0])` が曖昧にならずにコンパイルできるか。`Int32`/`Int64` での
     `[1, 2]` も同様。だめなら `FromSingles`/`FromDoubles`/... にする。
   - **(H3)** 演算子オーバーロード `+(TTensor, Double)` と `+(TTensor, Int64)` が
     共存でき、`T + 1`、`T + 1.5`、`2 * T` がすべて解決されるか。record で `**` を
     オーバーロードできるか。単項の `-` も。
   - **(H7)** `&div`、`&var`、`&to`、`&set`、`&repeat` という名前の record メソッドと、
     修飾子 `absolute`、`alias`、`index`（`&` あり・なしの両方）がコンパイルでき、
     `t.&div(u)` のように呼べるか。
   - **(H9)** 仮想抽象メソッド `Forward(const X: TTensor; Train: Boolean = False)` で、
     オーバーライドでもデフォルト値を繰り返したとき、基底型と派生型の両方から
     `Model.Forward(X)` と呼べるか。
   - **(H10)** `TIValue = record ... Items: array of TIValue; end;` がコンパイルできるか。
   - **(H11)** `NoGradGuard` を代入した `IInterface` のローカル変数が、ルーチンの終わりで
     （それより早くではなく）解放されるか。
   - `{$I}` で record 宣言の中に約 2,650 個のメソッドを取り込んでも、妥当な時間で
     コンパイルできるか。`T[i]` の糖衣が欲しければ、`{$modeswitch advancedrecords}` の
     record でプロパティが使えるか。
8. **生成器。**
   - gen.ml に `ops.tsv` の出力を追加する（公開名、種類、レシーバー、結果の形、
     デフォルト値付きの引数、戻り値名）。
   - `gen/tchgen.pas` を書き、`TchSysGenerated.pas` と `.inc` ファイルを生成して、
     全体をコンパイルする。
   - 手書きの名前（`data`、`grad`、`detach`、`to_device`、`clone` など）との、
     大文字小文字を無視した衝突チェックを入れる。

## 未解決の論点

- **OCaml の処理系。** `gen/gen.ml` はフォークの一部なので、どちらにしてもソースとして
  リポジトリに入る。論点は、それを *実行する* ために OCaml のコンパイラと opam
  パッケージ（`yaml`、`base`、`stdio`）をローカルのマシンに入れるかどうか。必要なのは
  libtorch を上げるときだけで、tch-fpc のビルドには不要。代案：gen.ml の実行は
  クラウドセッションだけで行い、`ops.tsv` をコミットする。
- **複数出力の op**：`out` 引数（決定済み）。実際にコードを書いてから、よく使う op
  （`max_dim`、`sort`、`topk`、`lstm`）に手書きの結果用 record を足す価値があるか
  見直す。(H8) でエイリアシングの危険が実際に問題になるなら、`out` の代わりに `var` に
  するかも検討する。
- **デフォルト引数。** 生成器は末尾に連続する Bool/Int64/Double/String 引数に
  デフォルト値を付ける。それで恩恵を受ける op がどれだけあるか。また、最初の
  マイルストーンで手書きの便利なオーバーロード（例：`kFloat`/CPU を既定にした
  `TTensor.zeros([2, 3])`）をどれだけ足すか。
- **等価性とコンテナ。** `TTensor` では `=` を意図的にオーバーロードしていないので、
  FGL（`TFPGList<TTensor>`）は使えない。`TchNN` に必要なことが
  `Generics.Collections.TList<TTensor>` と素の `TTensorArray` で足りるか確認する。
  足りなければ、ハンドルの同一性による `=` を再検討する。
- **Train フラグ。** `Forward(X, Train)` の引数（決定済み）か、子へ伝播する PyTorch 風の
  `Training` プロパティか。`TSequential` と独自モデルの例ができてから見直す。
- **`TIValue` の表現。** タグ付き record（決定済み。H10 次第）か、fpjson 風の
  クラス階層か。
- **VarStore のスレッドセーフ性。** スレッドセーフでないと明記している。データ読み込み用の
  スレッドが必要になったら `TCriticalSection` を足す。
- **ストリーム／画像。** 後で必要になったら：tch-rs の Rust 実装のストリーム関数の代わりに、
  Pascal から C の関数ポインタをまとめた構造体を渡す形にする（`TStream` を包む）。
  画像は stb ではなく Pascal 側（FPImage）で処理できる。
- **Windows / macOS。** `cdecl` と `tchf_free` 以外はまだ設計していない。Windows では
  さらに `tch.dll` からのシンボルのエクスポートが必要（`--export-all-symbols` か
  `.def` ファイル）。
- **公開。** ライセンスは決定済み（上記）。リポジトリはユーザーが公開を決めるまで
  非公開のまま。
