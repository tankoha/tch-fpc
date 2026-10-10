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
  - OCaml：`gen/gen.ml` はフォークの一部としてソースでリポジトリに入る。それを
    実行するための OCaml のコンパイラと opam パッケージ（`yaml`、`base`、`stdio`、
    `dune`）はローカルのマシンに入れる。メンテナー用の道具で、新しい libtorch 向けに
    再生成するときだけ必要。tch-fpc のビルドには不要。
- セッション2では、4 章を tch-rs のソースと突き合わせてレビューし、4.6
  （nn / VarStore / オプティマイザ / JIT をクラスで、`TVarPath` は record で、
  `Forward(X, Train = False)` は1つに）と 4.7（勾配モード）を追加しました。
  FPC 3.2.2 の未検証の挙動は DESIGN.md で **(H1)**〜**(H11)** と印を付け、下に並べています。

## クラウドセッション3（2026-10-10）：引数名のルールを改訂

- **引数名とメンバーの衝突、改訂（H12）。** ローカルセッション1の `_` を足す
  ルールをやめ、`Tch.pas` に `{$modeswitch duplicatelocals}` を入れる。生成する
  引数名・`out` の名前はそのまま（`dim`、`indices`、`pow`）、`NumDims` は `Dim` に
  戻し、予約語は op 名と同じ `&` エスケープ（`&out`、`&end`、`&to`）。`result`
  （3 op）だけ `result_`。クラウドのコンテナの FPC 3.2.2 で実測：
  `tests/compiler/h12_duplicatelocals.pas` と `xfail_h12_*.pas` 3本。
  `tests/compiler/run.sh` 全体が通る。詳細とコンパイラのソースの確認は
  DESIGN_jp.md の 4.4。README（en/jp）も更新。
- 変えたコード：`src/Tch.pas`（スイッチ、`Dim`、仮の op は `dim`）、
  `tests/ownership/ownership_test.pas`（`V.Dim`）。ownership テストは libtorch が
  要るのでクラウドでは **再実行していない**。ローカルで一度走らせること
  （`Tch.pas` 自体はコンパイルできる）。
- 下の引き継ぎのうち、まだ残っているもの：1（ops.tsv の形式、tchgen.pas の
  op の形ごとの出力）と 2（エラーメッセージの形、`TKind` 27〜45、`TORCH_WARN`、
  デフォルト引数の範囲、`FromArray` の Int32）。

## クラウドへの引き継ぎ（2026-10-10、ローカルセッション1の終了時点）

手順1〜7は完了し、仮説 H1〜H11 はすべて実測済み（結果は下の「ローカルの進捗」と、
DESIGN_jp.md の各 (Hn) の横）。手順8のための OCaml の環境はローカルに入っている。
ユーザーの判断で、手順8の設計はクラウドで行う。次のクラウドセッションへのお願い：

1. **生成器（手順8）を設計する。** ローカルの作業が機械的に進められる程度に：
   - gen.ml が出力する `ops.tsv` の正確な形式（列、エスケープ、デフォルト値・省略可能や
     リストの型・レシーバー・結果の形の表し方）。
   - `gen/tchgen.pas` の構成と、op の形（固定個数、可変個数、スカラー、戻り値なし）ごとに
     出力する Pascal の正確な形。`src/Tch.pas` の仮の op（`add`、`cat`、`max_dim`、
     `ones`）が、すでに動いている本体の見本になる。
   - 名前：引数名・結果名のルール（DESIGN_jp.md の 4.4。クラウドセッション3で改訂、
     上を参照）、`&` のエスケープ、
     手書きのメンバーと op 名の衝突チェック（「引数名と `TTensor` のメンバーの衝突」の
     tch-rs についてのメモを参照）。
2. **生成器に関わる未解決の論点を決める：** ネイティブのエラーメッセージの形
   （`Message` と `Detail`）、`TKind` の 27〜45、`TORCH_WARN` の扱い（`at_autocast_*` の
   修正か警告ハンドラー、または両方）、デフォルト引数の範囲、`FromArray` の
   `array of Int32` のオーバーロードをなくすこと（H2）。
3. 最後にいつもどおり DESIGN/HANDOFF（英語版と jp）を更新する。

設計の前提にしてよいローカルの状態：`src/TchSys.pas`、`src/Tch.pas`（最小限の
`TTensor`、`Dim`、`TchLiveHandles`）、`tests/` 以下のテスト。ビルド方法は各テストの
先頭と「ローカルの進捗」にある。

## ローカルの進捗（2026-10-10）

- **手順1 完了。** FPC 3.2.2、g++ 15.2.0、CMake 4.2.3（libtorch の CMake 設定を問題なく
  読める。「kineto not found」の警告は無害）、`libtorch/` に libtorch 2.13.0+cpu
  （cxx11 ABI）。OCaml は未導入（手順8で必要）。
- **手順2 完了。** `d81f48b` が無改変の取り込み、`a15806e` がパッチ
  （`git diff d81f48b a15806e` でパッチだけが見える）。ビルド：
  `cmake -S csrc/libtch -B build/libtch -DCMAKE_PREFIX_PATH=$PWD/libtorch
  -DCMAKE_BUILD_TYPE=Release && cmake --build build/libtch -j 8`（約30秒）。
  `atg_*` を 2657 個公開、`ldd -r` で未解決シンボルなし。gen.ml も直したので、
  `extern "C"` のガードは再生成しても残る。
- **手順3 完了。** `tests/c/smoke.c` が通る（ビルド方法はファイル先頭）。
  その過程で分かったこと：`torch_api.h` の C 側の分岐に `<stddef.h>`（`size_t`）と
  `<stdbool.h>`（C23 より前の `bool`）が必要だったので、パッチに追加した。
  エラーメッセージは libtorch の本文のあとに `Exception raised from ...` で始まる
  C++ の発生箇所・バックトレースが続く（未解決の論点を参照）。

- **手順4 完了。** `tests/abi/abi_test.pas` が通る（ビルド方法はファイル先頭。
  `build/abi_test nilarg` で nil 引数のケースを実行）。測った結果：
  - C99 の `bool` の戻り値は `Boolean` で宣言する。FPC は `%al` だけを見る
    （`testb %al,%al`）。SysV ABI が保証するのもそこだけ。**`LongBool` や ctypes の
    `cbool`（実体は `LongBool`）は使わない**：FPC が `%eax` 全体を見る
    （`orl %eax,%eax`）。今回は g++ が上位ビットを0にしていたので正しく読めたが、
    保証はない。`Boolean` の引数（`at_autocast_set_enabled`）も正しく渡る。
  - `c10::ScalarType` の 0〜26 は `TKind` と完全に一致（Float8 の 23〜26 も含む）。
    libtorch 2.13 には **27〜45** もある（UInt16/32/64、UInt1〜7、Int1〜7、
    Float8_e8m0fnu、Float4_e2m1fn_x2）。CPU で `atg_empty` がすべて作れて、
    `at_scalar_type` も一致する（未解決の論点を参照）。
  - `uint8_t x_null` を `Byte`、`int8_t` の layout を `ShortInt`（-1 = nullopt を含む）
    として渡して、意図どおりに動く（`atg_argmax`、`atg_to_sparse`）。
  - `external 'tch'` と `-Flbuild/libtch -k-rpath=$PWD/build/libtch` で、実行時に
    `libtch.so` が見つかる。libtorch は libtch の RUNPATH 経由で見つかる。
  - 省略不可の引数に nil のテンソルを渡す（`atg_add(@S, nil, A)`）と、FPC の
    SIGSEGV ハンドラーが C++ のフレームの中から捕まえられる `EAccessViolation` に
    変え、プロセスは動き続ける。回復できるように見えるが安全ではない（C++ の
    巻き戻しが起きず、libtorch の状態は不明）。呼び出し前の nil ハンドルのチェックは
    やはり必要。
  - libtorch は `TORCH_WARN` のメッセージを直接 stderr に書く（例：`at_autocast_*`
    を呼ぶたびに非推奨 API の警告、量子化型や ComplexHalf の作成時）。
    未解決の論点を参照。

- **手順5 完了。** `tests/fpu/fpu_test.pas` と `tests/fpu/run.sh`（モードごとに別プロセス、
  タイムアウト30秒）。FPC の既定は MXCSR=$1900、x87CW=$1372。
  `SetExceptionMask(all)` のあとは $1F80/$137F。
  - マスクなし・4要素：C++ のフレームの中から `EZeroDivide` が投げられる。
  - マスクなし・10^7 要素：OpenMP のワーカーで未処理の例外になり、そのあと
    **固まる**（メインスレッドの Pascal の状態も壊れる）。
  - 先にマスク：どちらの大きさでも 1/0 は +Inf。
  - マスクせずにプールを作り、あとでマスク：やはり固まる。マスクは `Tch` の
    `initialization` で設定すべきことの確認。
  - **(H4) 解決：** マスクのあとに起動した `TThread` もマスクされている。その中の
    小さな op も並列の op も Inf を返す。
  - **(H5) 解決：** 整数の `EDivByZero` を捕まえたあともマスクは残る。
  - DESIGN_jp.md の 4.2 もこれに合わせて更新した。

- **手順6 完了。** 最初のコード：`src/TchSys.pas`（手書きの external 宣言と、テストに
  必要な少数の `atg_*` の仮宣言）と `src/Tch.pas`（`TTensor`/`TTensorHandle`、
  `TchCheck`、nil 引数のチェック、`initialization` での FPU マスク、
  `TchLiveHandles` カウンター。生成器が出す形で手書きした仮の op：`add`、`mul`、
  `cat`、`max_dim`、`ones`）。テストは `tests/ownership/ownership_test.pas` で、全部通る。
  - `B := A; A := Default(TTensor)` のあとも B は生きている。`A := A + B` も正しい。
  - `D := (A + B) * C` を 10^6 回：ハンドル数も RSS も一定（全体で約1.5秒）。
  - **(H1) 解決：** 一時値は文の終わりではなく、枠が再利用されるかルーチンを
    抜けるまで生きている。
  - **(H6) 解決：** `cat([A, B], 0)` も一時値の `cat` も正しく、リークも二重解放もない。
  - **(H8) 解決：** レシーバーを `out` に渡すと `ETorchError`。`out` に渡した変数の
    以前の値は解放される。
  - 新しく見つかった問題（未解決の論点を参照）：生成する引数名が `TTensor` の
    メンバーと衝突する（`dim` と `Dim`）。仮の op はクラウドセッション3（H12）まで
    `dim_` にしていた。

- **手順7 完了。** `tests/compiler/run.sh`（h*.pas はコンパイルできて動くこと、
  xfail_*.pas は拒否されること、それに生成した大きな record）。すべて想定どおり。
  詳細は DESIGN_jp.md の各仮説の横に書いた。
  - **(H2)** `[1.0, 2.0]` は `Single` を選ぶ。`Int32` と `Int64` の両方のオーバーロードが
    あると `[1, 2]` が曖昧になる → `Int32` のオーバーロードは置かない。
  - **(H3)** スカラーの演算子は意図どおりに解決される。`**` と単項の `-` も使える。
  - **(H7)** エスケープした名前はすべてコンパイルでき、呼べる。
  - **(H9)** オーバーライドでもデフォルト値を繰り返す（でないと `Derived.Forward(X)` が
    コンパイルできない）。
  - **(H10)** コンパイルできる。コピーすると `Items` は共有される（コピーオンライトなし）。
  - **(H11)** ローカル変数のガードはルーチンの終わりで解放される。戻り値を捨てた
    `NoGradGuard;` はすぐ解放される（落とし穴）。
  - 2,657 個のメソッドを `{$I}` で取り込んだ record（管理型のフィールドあり）：
    約1秒でコンパイルできる（-O2 で 1.6 秒）。`default` のインデックス付きプロパティで
    `T[5][2]` と書ける。

- **OCaml の環境を用意（手順8の前提）。** apt の `ocaml` 5.4.0 と `opam` 2.5.0
  （Ubuntu に `yaml` のパッケージがないので opam を使う）。
  `opam init --bare --no-setup`（シェルの設定ファイルには触れない）、`ocaml-system` を
  使うスイッチ `tch-fpc`、`opam install dune base stdio yaml`。ビルドの前に
  `eval $(opam env --switch=tch-fpc)` を実行し、`cd gen && dune build ./gen.exe` で
  警告なしにビルドできる。4227b89b の tch-rs のチェックアウトで実行すると
  （`third_party/pytorch/Declarations-v2.13.0.yaml` を読み、上流のパスに書き出す）、
  パッチ済みの gen.exe は `csrc/libtch/torch_api_generated.{h,cpp}` をバイト単位で
  再現する（2657 関数、0.5 秒）。

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
   - **(H12、クラウドセッション3で解決)** `{$modeswitch duplicatelocals}` で、
     `objfpc` のままメソッドの引数にメンバーと同じ名前を使える。引数名 `&out` も
     コンパイルできる。関数の引数名 `result` と、メンバーと同名のローカル変数は
     引き続き拒否される。
8. **生成器。**
   - gen.ml に `ops.tsv` の出力を追加する（公開名、種類、レシーバー、結果の形、
     デフォルト値付きの引数、戻り値名）。
   - `gen/tchgen.pas` を書き、`TchSysGenerated.pas` と `.inc` ファイルを生成して、
     全体をコンパイルする。
   - 手書きの名前（`data`、`grad`、`detach`、`to_device`、`clone` など）との、
     大文字小文字を無視した衝突チェックを入れる。
   - 引数名・out の結果名に衝突の処理は不要（H12）。予約語は `&` でエスケープし、
     `result` は `result_` にする（DESIGN_jp.md の 4.4）。

## 未解決の論点

- **ネイティブのエラーメッセージの形。** libtorch のメッセージは本文のあとに、
  複数行の C++ の発生箇所・バックトレース（`Exception raised from ...`）が続く。
  tch-rs はそのまま渡している。`ETorchNativeError` の案：`Message` は
  `Exception raised from` より前の本文、全文は `Detail` プロパティに入れる。
  `TchRaiseNative` を書くとき（手順4以降）に決める。
- **26 を超える `TKind`。** libtorch 2.13 には ScalarType の 27〜45 があり、CPU で
  作れる（手順4）。DESIGN の `TKind` は 26 までで、未知の型では例外を投げる。
  決めること：`TKind` を 45 まで広げる（少なくとも UInt16/32/64 と
  Float8_e8m0fnu は役に立ちそう）か、0〜26 のままにして `Kind` で例外にするか。
- **libtorch の警告。** `TORCH_WARN` は無条件に stderr に出る。上流の
  `at_autocast_*` は呼ぶたびに警告を出す（非推奨の
  `at::autocast::is_enabled`/`set_enabled`）。案：フォークでこの2つの関数を
  デバイスを取る非推奨でない API に直す。または libtch から `c10::WarningHandler`
  を設定して Pascal のコールバックに渡す（両方でもよい）。
- **引数名と `TTensor` のメンバーの衝突：決定後に改訂**（2026-10-10、ユーザー判断）。
  最初の決定（ローカルセッション1）：重ならなくなるまで `_` を足し、`Dim` → `NumDims`。
  クラウドセッション3で改訂（H12）：`Tch.pas` の `{$modeswitch duplicatelocals}` で
  衝突そのものがなくなり、名前はそのまま、`Dim` に戻し、予約語は `&`、`result` →
  `result_`。ルール、実測、コンパイラのソースの確認は DESIGN_jp.md の 4.4。
  tch-rs でこれが要らなかった理由：Rust では引数はただのローカル変数で、メソッドは
  `self.dim()` や `Tensor::dim` の形でしか呼べないので、`fn dim(&self)` と `dim: i64` を
  持つ約250の op が共存できる（`duplicatelocals` は `{$mode delphi}` のうち、この点で
  Rust と同じ振る舞いをする部分）。
  tch-rs が *対処している* のは op 名の衝突で、手順8ではこちらの名前に対して同じ
  チェックが必要：gen.ml の `replace_map`（Rust の予約語の `where`/`unsafe`、手書きの
  メソッドと重なる `view`、`to_device`、`t` → `tr`）と `prefixed_functions`（素の名前を
  手書き版に譲る `add`、`mul`、`div`、`sub` などに `g_` を付ける）。これらは Rust の
  ラッパーの名前を変えるだけで、生成器が読む `atg_*` の名前には影響しない。README にも記載。
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
- **`TIValue` の表現。** タグ付き record：決定済みで、H10 でコンパイルできることを確認
  （手順7）。コピーすると `Items` を共有するので、変更しないものとして扱う。
- **VarStore のスレッドセーフ性。** スレッドセーフでないと明記している。データ読み込み用の
  スレッドが必要になったら `TCriticalSection` を足す。
- **ストリーム／画像。** 後で必要になったら：tch-rs の Rust 実装のストリーム関数の代わりに、
  Pascal から C の関数ポインタをまとめた構造体を渡す形にする（`TStream` を包む）。
  画像は stb ではなく Pascal 側（FPImage）で処理できる。
- **Windows / macOS。** `cdecl` と `tchf_free` 以外はまだ設計していない。Windows では
  さらに `tch.dll` からのシンボルのエクスポートが必要（`--export-all-symbols` か
  `.def` ファイル）。
- **ROCm（保留。まず CPU）。** 手順2〜7は CPU 版 libtorch で進める。手元の GPU は
  RX 9070（Navi 48、gfx1201）と内蔵の Raphael で、システムに ROCm は入っていない。
  libtorch 2.13.0 の ROCm 版は `+rocm7.1` のみ（約5.8GB、2026-10-10 確認）。
  未測定の見込み：ROCm 版は HIP を CUDA として見せる（`cuda` デバイス、
  `torch::cuda::is_available`）ので、C API・`TDevice`・`CudaIsAvailable` は変更不要。
  `libtorch-rocm/` として追加するときに確認すること：`torch_hip` を
  `-Wl,--no-as-needed` でリンクする（libtch は直接参照しない）、zip に gfx1201 の
  カーネルが入っているか、内蔵 GPU の列挙（`HIP_VISIBLE_DEVICES`）、同梱の依存だけで
  システムの ROCm なしに動くか。DESIGN.md の1章は `libtorch_cuda.so` しか書いていない。
- **リリースの中身。** 生成ファイル（`ops.tsv`、`torch_api_generated.*`、
  `TchSysGenerated.pas`、`.inc`）はコミットされるので、どのタグのソースアーカイブも
  OCaml なしでビルドできる。未決：リリースに libtorch のバージョン・プラットフォーム別の
  ビルド済み `libtch.so`/`tch.dll` も添付するか、利用者が常に CMake でビルドするか。
  libtorch 本体は同梱しない。
- **公開。** ライセンスは決定済み（上記）。リポジトリはユーザーが公開を決めるまで
  非公開のまま。
