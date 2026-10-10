# 引き継ぎ

（[HANDOFF.md](HANDOFF.md) の日本語訳です。内容が食い違う場合は英語版が正です。）

再開するときは最初にこれを読んでください。決定事項と理由は [DESIGN_jp.md](DESIGN_jp.md) にあります。

## 現状（2026-10-10、クラウド設計セッション1回目の終了時点）

- 3つの層すべての設計を DESIGN.md に書きました。まだ何もビルドしていません。
- 基準：tch-rs `4227b89b72059b0651ff83a38637693574e0a2d6`、libtorch 2.13.0。
- 決定済み：
  - tch-rs の libtch をフォークする（C の関数名は維持）。
  - libtch は共有ライブラリ `libtch.so` にする。
  - `TTensor` は参照カウント付きインターフェースを包む record にする。
  - libtch を呼ぶたびにエラーを確認し、`ETorchError` を投げる。
  - 生成する op はスネークケース、手書きの API はキャメルケースにする。
  - フォークした gen.ml に一覧ファイルを出力させ、Pascal の生成器がそれを読む。

## 次の作業（ローカル実装）

1. **ツールの確認。** FPC 3.2.2、g++、CMake を用意し、libtorch 2.13.0 の CPU 版
   （cxx11 ABI）を `libtorch/` に展開する（`.gitignore` 済み）。
2. **libtch の取り込み。** 上記コミットの tch-rs から `csrc/libtch/` と `gen/` に
   コピーし、LICENSE-MIT / LICENSE-APACHE を追加する。まず無改変のコピーを
   コミットし（メッセージに取り込み元のファイル名を書く）、2つ目のコミットで
   こちらのパッチを当てる：ストリーム・Python・画像のコードを削除し、`tchf_free` を
   追加する。CMake で `libtch.so` をビルドする。
3. **C での動作確認**（Python ではなく小さな C プログラムで）：テンソルを作り、
   `atg_add` を呼び、表示する。わざとエラーを起こし、`get_and_reset_last_err` で
   読めることを確かめる。
4. **Pascal での ABI 確認**（推測せず、実際に測る）：
   - `bool` の戻り値（`at_autocast_is_enabled` など）が `Boolean` として正しく読めるか。
   - `c10::ScalarType` の数値が `TKind` と一致するか（型ごとのテンソルについて
     `at_scalar_type` を表示する）。
   - リンク：`external 'tch'` と rpath で、実行時に `libtch.so` と libtorch が見つかるか。
5. **FPU マスクのテスト。** `SetExceptionMask` なしで、CPU テンソルで `1.0 / 0` を
   計算すると FPC のプログラムが落ちるか。`Tch` の initialization での対処で
   直ることを確かめる。
6. **所有権のテスト。** `(A + B) * C` を 10^6 回ループして RSS（常駐メモリ）が増えないことを
   確認する。`B := A` のあと `A := Default(TTensor)` としても B が生きていることを
   確認する。これで FPC 3.2.2 上での「インターフェースを持つ record」設計を検証する。
7. **生成器。** gen.ml に `ops.tsv` の出力を追加し、`gen/tchgen.pas` を書き、
   `TchSysGenerated.pas` と `.inc` ファイルを生成して、全体をコンパイルする。

## 未解決の論点

- **OCaml への依存。** 再生成には OCaml と opam パッケージ `yaml`/`base`/`stdio` が
  必要。ローカルに入れてよいか？ 代案：`ops.tsv` の再生成だけをクラウドセッションで
  行い、コミットする。
- **Pascal での `IntListOption`。** オープン配列では「なし」と「空」を区別できない。
  候補は、`TInt64Array` からの暗黙変換演算子を持つ `TOptIntList` record、
  または引数あり・なしのオーバーロード。
- **複数出力の op**：`out` 引数（現在の案）か、小さな結果用 record（`TTensorPair`）か。
  `max_dim`、`sort`、`lstm` で実際にコードを書いてから見直す。
- **デフォルト引数。** 便利なオーバーロードをどこまで生成し、どこから手書きにするか。
- **ストリーム／画像。** 後で必要になったら：tch-rs の Rust 実装のストリーム関数の代わりに、
  Pascal から C の関数ポインタをまとめた構造体を渡す形にする（`TStream` を包む）。
  画像は stb ではなく Pascal 側（FPImage）で処理できる。
- **Windows / macOS。** `cdecl` と `tchf_free` 以外はまだ設計していない。
- **ライセンス。** 未定のまま。リポジトリは非公開を維持。フォークで MIT/Apache-2.0 の
  コードが入るので、tch-fpc も同じ二重ライセンスにするのが一番摩擦が少ない ——
  ユーザーが決めること。
