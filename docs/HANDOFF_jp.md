# 引き継ぎ

（[HANDOFF.md](HANDOFF.md) の日本語訳です。内容が食い違う場合は英語版が正です。）

再開するときは最初にこれを読んでください。決定事項と理由は [DESIGN_jp.md](DESIGN_jp.md) にあります。

## クラウドセッション4（2026-10-10）：DESIGN 4.6 を確定

下の依頼（「クラウドへの引き継ぎ、ローカルセッション2の終了時点」）に回答しました：
DESIGN.md 4.6 にインターフェースのスケッチと小節 4.6.1〜4.6.7 を書き、それぞれ
4227b89b の tch-rs の具体的なファイルに根拠を置いています。何もコミットしておらず、
スクラッチパッドの外にコードは書いていません。決定事項の要約：

1. **初期化（4.6.1）。** `TInit` は `Kind`、`A`、`B` と Kaiming のペイロード（`Dist`、
   `Fan`、`NonLin`、`Gain`）を持つ。fan と gain の式は `init.rs` から（fan-in =
   `Dims[1] * prod(Dims[2..])`、ReLU の gain `sqrt(2)`、一様分布の範囲
   `sqrt(3) * gain / sqrt(fan)`）。`Make` = `f_init` を op 単位で（`zeros`/`ones`/
   `mul_scalar`、`uniform_`、`randn`、`empty.normal_`、`t_`、
   `linalg_qr(…, 'reduced', Q, R)`、`diag(0).sign`、`mul_`、`mul_scalar_`、`contiguous`、
   最後に `.view(Dims)` — 唯一の明示した差異で、PyTorch の形の規則）。`Make` に勾配
   ガードはなし。`Apply`（tch-rs の `Init::set`）は自分でガードする。tch-rs が `Float` を
   直書きしている箇所はどこでもストアの kind を使う。
2. **ストア（4.6.2）。** 追加のみの `array of TVarEntry`（名前、テンソル、trainable、
   グループ）+ 学習対象の添字の `array of Integer`（tch-rs の `trainable_variables`）+
   `TDictionary<String, Integer>` の索引（3.2.2 で計測、大文字小文字を区別）。重複する
   `Add` は tch-rs と同じく `<name>__<Count>` に改名。`.`、`|`、空の名前は例外。
   `GetOrAdd` は次元にかかわらず既存のテンソルを返す。`Freeze`/`Unfreeze` = 学習対象の
   要素への `set_requires_grad`。`SetDevice`/`SetKind` = その場での `set_data(to/totype)`。
   `half()` 系には `SetFloatKind`。`CopyFrom` は2パスで、`NoGradGuard` の下。
   等価性/コンテナの論点は閉じた：`TTensor` に `=` は不要。
3. **Save/Load（4.6.3）。** `Save` → `at_save_multi`（`.`→`|`）。`Load` / `LoadPartial` →
   `at_load_callback_with_device`（`.bin`/`.pt` → `at_loadz_callback_with_device`、
   `.safetensors` は例外）。`data` に `TLoadContext` クラスのインスタンスを渡す `cdecl` の
   トランポリン。例外は `AcquireExceptionObject` で捕まえ、C の呼び出し後に再 raise。
   エラースロットはどの経路でも空にする。C のハーネスで計測済み。読み込み時の精度の
   更新（`set_data(totype(ファイルの kind))` してからコピー）は tch-rs と同じ。
   `at_load_multi[_]` は使わない。
4. **オプティマイザ（4.6.4）。** `PTorchOptimizer = Pointer`、external は `torch_api.h`
   のとおり。構築時にすべての学習対象の要素を追加し、`ZeroGrad`/`Step`/`BackwardStep*` が
   新しいものを遅延追加（`FVarsInOptimizer`）。デフォルト値は `nn/optimizer.rs` から
   （AdamW の weight decay 0.01、RmsProp の alpha 0.99）。`ClipGradValue` =
   `grad.clamp_`。`ClipGradNorm` = 勾配ごとの `norm`、`stack(...).norm`、
   `mul_scalar_(coef)`。どちらも `NoGradGuard` の下。**フォークのバグを発見：**
   `ato_set_momentum_group` が `} else if` ではなく `} if`（`torch_api.cpp:564`）。
   ローカルでパッチする。
5. **レイヤー（4.6.5）。** `TLinear`（`linear`）、`TConv2D`（`conv2d`、スカラーの
   `ConvConfig`、padding モードは `pad` 経由）、`TBatchNorm`（Nd 1/2/3、`batch_norm`、
   `Train` を使う）、`TLayerNorm`（`layer_norm`）、`TEmbedding`（`embedding`）、関数と
   しての dropout。デフォルト値と変数名は 4.6.5 の表に tch-rs のファイルごとに。
   Conv1D/3D、ConvTranspose、GroupNorm、LSTM/GRU はマイルストーン3へ。
6. **Sequential（4.6.6）。** `specialize TObjectList<TModule>`。`AddFn` は4つの
   オーバーロード（関数、関数+Train、メソッド、メソッド+Train。解決を計測済み）。`Train`
   を全ての子に渡す。`ForwardAll`。
7. **ユニット/テスト（4.6.7）。** `TchNN.pas` 1つ。`TchSys` にオプティマイザ/save/load の
   external、`Tch.pas` に `TTensor.CopyFrom`。そこに挙げた手計算できる数値による
   `tests/nn/nn_test.pas`（初期 loss 22.99967、SGD の最初のステップ `w = 0.80955`、
   `b = 0.08974`、Adam の最初のステップ `lr·sign(g)`、tch-rs のクリッピングの数値、conv の
   padding のベクトル、`LoadPartial → ['a.b.t2']`）。mock は帳簿、コールバック、遅延追加の
   回数だけを確認する。

**クラウドでの計測（FPC 3.2.2、gcc 13。スクラッチパッドのみ）：**
- M1：`data` にクラスのインスタンスを渡し、C のハーネスから `cdecl` のトランポリンを
  3回呼ぶ。ハンドラで `AcquireExceptionObject`、呼び出し後に再 raise：クラスと
  メッセージは無傷、以後のコールバックは飛ばされ、ハンドルはすべて所有される。
- M2：mock の libtch に対する `TTensor` と `Generics.Collections`：
  `TDictionary<String, Integer>` は大文字小文字を区別し、重複する `Add` → `EListError`。
  `TTensor` フィールドを持つ `array of record` と `TList<TTensor>` はハンドルを正しく
  解放する（H1 の式の一時変数だけがルーチン終了まで残る）。抽象基底クラスと `for .. in`
  での `TObjectList<TModule>` が動く。`Contnrs.TObjectList` も動く。
- M3：関数ポインタ / メソッドポインタ、`Train` あり・なしの4つの `AddFn` オーバーロードは
  objfpc で曖昧さなく解決し、チェーンも動く。
- 副作用：`build/mock/libtch.so` が古かった（ローカルセッション2のスタブより前にビルド）
  ので `tests/mock/run.sh` で再ビルドした（45件のチェックが通る）。`build/` は
  git-ignore されているので、コミットするものはない。

**新しい仮説（本物の libtorch が必要）：**
- **(H13)** `set_data` は共有ハンドルを同期させる：`SetDevice` / `SetKind` のあと、
  レイヤーの `Weight` と先に作ったオプティマイザが新しいストレージを見る（tch-rs は
  これに依存している）。
- **(H14)** Orthogonal 初期化：`[3, 5]` で `Q·Qᵀ = I`、`[5, 3]` で `Qᵀ·Q = I`
  （`allclose` 1e-5）、結果の形 = `Dims`。
- **(H15)** 読み込み時の精度の更新：`kHalf` のファイルを読んだ `kFloat` のストアは
  正しい値の `kHalf` になる。
- **(H16)** パッチ前の `ato_set_momentum_group` は Adam で "unexpected optimizer" を
  投げ、パッチ後（`} else if`）は投げない。
- **(H17)** 学習対象の葉への `TInit.Apply` は `NoGradGuard` なしで失敗し（"a leaf
  Variable that requires grad is being used in an in-place operation"）、ありで成功する。
- **(H18)** `Freeze; Step` は値を変えず、例外も出ない（libtorch は grad が未定義の
  パラメータを飛ばす）。

**ローカルセッション、この順に（機械的）：**
1. フォークのパッチ：`csrc/libtch/torch_api.cpp` の `ato_set_momentum_group`：
   `} if (auto sgd` → `} else if (auto sgd`。libtch を再ビルド。`tests/abi`、
   `tests/mock/run.sh`。
2. `src/TchSys.pas`：`PTorchOptimizer`、`TTorchLoadCallback`、`at_save_multi`、
   `at_load_callback_with_device`、`at_loadz_callback_with_device`、13個の `ato_*`
   （シグネチャは 4.6.4 に）。`src/Tch.pas`：`TTensor.CopyFrom`。
3. `src/TchNN.pas`。各部分が単独でコンパイルできるようこの順に：`TInit`（4.6.1）→
   save/load なしの `TVarPath`/`TVarStore`（4.6.2）→ save/load（4.6.3）→ `TOptimizer`
   （4.6.4）→ `TModule`、`TFuncModule`、`TSequential`（4.6.6）→ 5つのレイヤー（4.6.5）。
4. 4.6.7 の10グループによる `tests/nn/nn_test.pas`。出てきた順に H13〜H18 を確認し、
   結果を DESIGN.md の各 (Hn) の横に記録する。
5. mock：4.6.7 のスタブを `tests/mock/mock_tch.c` に足し、帳簿のチェックのために
   `tests/mock/nn_mock_test.pas`（または `mock_test.pas` の拡張）を書く。
6. DESIGN/HANDOFF（英語・日本語）を更新。その後、必要なら JIT（`TchJit`）と CUDA を
   設計セッションに戻す。

## クラウドへの引き継ぎ（2026-10-10、ローカルセッション2の終了時点）— 上のクラウドセッション4で回答済み

**設計のみ。** コミットするコード（ユニット、生成器、テスト）は書かないこと。
論点を決めるための小さな使い捨ての計測はよい（CLAUDE.md）。実装はローカル
セッションに戻す。本物の libtorch がないと確認できないことは、ローカル向けの
新しい仮説（H13 以降）として印を付ける。クラウドのコンテナには FPC はあるが
libtorch はない（`tests/mock` の mock `libtch.so` は使える）。

マイルストーン1は完了（下の「ローカルセッション2」）。次は DESIGN 4.6
（オプティマイザ、VarStore/nn）。4.6 で全体の形は決まっているが、次の点は
機械的に実装できるほどには決まっていない：

1. **初期化（`TInit.Make`）。** tch-rs 4227b89b の `nn/init.rs` に沿った正確な式：
   `Dims` からの fan-in/fan-out（どの次元か、畳み込みのカーネル）、`KaimingUniform`
   のゲインと範囲（tch-rs の `DEFAULT_KAIMING_UNIFORM`）、`KaimingNormal`、
   `Orthogonal`（どの生成 op を使うか：`linalg_qr`？符号の補正は？）、`Make` を
   `NoGradGuard` の下で実行するか。それぞれが呼ぶ生成 op（名前は
   `src/TchTensorGenerated_intf.inc` のとおり）。
2. **`TVarStore` の内部。** 名前付き変数のコンテナ（`Variables`/`TrainableVariables`
   とオプティマイザの遅延追加には順序が効く）、変数ごとの trainable フラグとグループ、
   名前の検証（tch-rs は名前の中の `.` を拒否）、同名で `Add` したときの動作、
   `Freeze`/`Unfreeze`/`SetDevice`/`SetKind`/`CopyFrom` が既存のハンドルに何をするか
   （その場で？no-grad の下で？）。等価性とコンテナの論点：`Generics.Collections` か
   素の配列か。
3. **Save/Load。** C の入口：`at_save_multi(tensors, names, n, filename)`、
   `at_load_multi(tensors, names, n, filename)`（nil の配列を渡すと新しいテンソルが
   返る）、`at_load_multi_`（既存のテンソルにコピー）、コールバック形式の
   `at_load_callback[_with_device]` / `at_loadz_callback[_with_device]`
   （`void (*f)(void *data, char *name, tensor t)`）。tch-rs と同様に
   `Save`/`Load`/`LoadPartial` がどれを使うか決める。**手順4からの制約：** Pascal の
   例外を C++ のフレームを通して伝播させてはならないので、Pascal のコールバックは
   例外を出さない。エラーを記録し、C の呼び出しから戻った後で呼び出し側が raise する。
   コールバックの宣言（`cdecl`、`data` で Pascal のオブジェクトをどう渡すか）も設計に
   含める。名前の `.` ↔ `|` は tch-rs と同じ。
4. **オプティマイザ。** C のシグネチャ：`ato_sgd(lr, momentum, dampening,
   weight_decay, int nesterov)`、`ato_adam/ato_adamw(lr, beta1, beta2,
   weight_decay, eps, bool amsgrad)`、`ato_rms_prop(lr, alpha, eps,
   weight_decay, momentum, int centered)`、`ato_add_parameters(opt, t,
   size_t group)`、`ato_set_learning_rate[_group]`、`ato_set_momentum[_group]`、
   `ato_set_weight_decay[_group]`、`ato_zero_grad`、`ato_step`、`ato_free`
   （すべて `PROTECT` 付きなので毎回 `TchCheck`）。決めること：`PTorchOptimizer` 型、
   新しい trainable 変数の遅延追加の方法（tch-rs の `variables_in_optimizer`）、
   勾配クリッピングの実装（`ClipGradValue`/`ClipGradNorm` を Pascal 側で生成 op を
   使って書く：どの op か、no-grad の下か）、AdamW/RmsProp のデフォルト値（tch-rs のもの）。
5. **マイルストーン2のレイヤー。** `TLinear` 以外に何を入れるか（tch-rs でよく使われる
   順：`Conv2D`、`BatchNorm`、`Embedding`、`LayerNorm`、関数としての `Dropout`、
   `LSTM`/`GRU`？）。それぞれの設定 record、デフォルトの初期化（tch-rs のもの）、
   `Forward` が呼ぶ生成 op。`TLinear` の forward は `matmul` + bias か `linear` か。
6. **`TSequential` / `TFuncModule`。** `TObjectList` か
   `specialize TObjectList<TModule>` か。`AddFn` は素の関数ポインタか `of object` か。
   train フラグを子に渡すか（渡す：すべての子に `Forward(X, Train)`）。
7. **ユニット構成とテスト。** `TchNN`（と、必要なら `TchOptim`）のユニットと
   何をどこに置くか。手で確かめられる数値を使ったマイルストーン2の受け入れテスト
   （例：SGD の線形回帰が既知の重みに収束する、小さな問題での Adam、
   save → load の往復、`LoadPartial` が欠けた名前を返す）と、mock ライブラリで
   どこまで検証できるか。

いつもどおり最後に DESIGN/HANDOFF（英語・日本語）を更新すること。設計の前提に
してよいローカルの状態は、下の「ローカルセッション2」にあるものすべて。

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
## クラウドセッション3の後半：手順8をクラウドで実装

設計を実測で確かめられたので、生成器は設計するだけでなく *作った*。コンテナに
FPC 3.2.2 を入れ、OCaml は apt から（`yaml` は入らなかった：opam のリポジトリに
プロキシ越しで届かないため、このセッションの scratchpad に置いた 150 行の代替
`Yaml` モジュールで gen.ml を動かした。コミットはしていない）。DESIGN_jp.md の
5 章に書いてあることは、すべてコミットしたコードが実際にしていること。

- **gen.ml** が `gen/ops.tsv` を書く（`gen.exe <yaml> <csrc/libtch> <ops.tsv>`。
  引数なしなら上流と同じ振る舞い）。再生成した `torch_api_generated.{h,cpp}` は
  コミット済みのものとバイト一致。`gen/ops.tsv` はコミット済み（2,657 op）。
- **gen/tchgen.pas**（Pascal 生成器）が `src/TchSysGenerated.pas`、
  `src/TchTensorGenerated_intf.inc`、`src/TchTensorGenerated_impl.inc` を生成
  （コミット済み）。`src/Tch.pas` はこれらを取り込み、`TKind` 全部（`kNone`、0..45）、
  `TLayout`、`TDevice`、`TScalar`、`TOptInt64`、`TOptDouble`、`TOptIntList`、入れ子の
  `TTensorArray`、生成した本体が使う補助関数、`ETorchNativeError.Detail` を持つ。
  仮の op はなくなった。全 op 入りの `Tch.pas` は 2.7 秒でコンパイルできる。
- **決定**（詳細は DESIGN_jp.md）：エラーメッセージは `Exception raised from` で
  分割（4.2）。`TKind` は 45 まで広げ、省略可能な dtype には `kNone`。`TOptKind` は
  なし（4.3）。フォークの autocast 関数はデバイスを取る API に修正、警告ハンドラーは
  c10 のハンドラーがスレッドローカルなので保留（4.2）。デフォルト引数のルールと
  その数（4.4）。結果が入力そのものの op には out 引数なし（4.4）。`FromArray` の
  オーバーロードは `Int32` なし（4.3）。
- **tests/mock**：モックの `libtch.so`（C、libtorch 不要）と、生成した op の形を
  すべて動かす `mock_test.pas`。45 項目が通る（`tests/mock/run.sh`）。
  `tests/compiler/run.sh` も引き続き通る。`tests/ownership/ownership_test.pas` は
  コンパイルできる（ここではリンクしていない。`ones` は `TDevice.Cpu` を取るようになった）。

## ローカルセッション2（2026-10-10）：下の手順1〜6完了

- **1.** autocast のパッチ入りで libtch を再ビルド。`tests/abi` は合格し、
  `[W` の警告行はもう出ない（`at_autocast_*` を呼ぶたびに出ていた非推奨 API の
  警告が消えた）。`tests/fpu/run.sh` はローカルセッション1と同じ結果。
- **2.** opam の本物の `yaml` で gen.exe をビルドし、tch-rs 4227b89b の
  `Declarations-v2.13.0.yaml`（そのファイルだけを repo の外に sparse clone）で
  実行：`ops.tsv` と `torch_api_generated.{h,cpp}` は**バイト単位で一致**。
  クラウドの代替 YAML パーサーはデフォルト値をすべて同じ型で読んでいた。
- **3.** tchgen：差分なし（2657 op、デフォルト引数あり 479、out 引数あり 149、
  渡したテンソルに書き込むもの 97）。
- **4.** `tests/ownership` は本物の libtorch で合格（10^6 回で RSS 一定、1.4 秒）。
  `tests/mock/run.sh` も合格。
- **5.** 新規 `tests/generated/generated_test.pas`、全件合格。期待値は手計算
  （Python は使っていない）。実測したこと：
  - `sum_dim_intlist`：`sum` では `NoIntList` も `IntList([])` も全次元で
    縮約する（0 次元、15）。PyTorch と同じ。
  - `pad(..., 'constant', NoDouble)` は 0 で埋める。`9.0` は `TOptDouble` に変換される。
  - 2 次元テンソルの `linalg_norm(2, NoIntList)` は**行列**の 2 ノルム
    （最大特異値、ここでは sqrt 54）で、ベクトルノルムではない。最初の期待値
    （sqrt 55）は私の誤りで、libtorch は PyTorch の `torch.linalg.norm` どおり。
    1 次元ならベクトルノルム。
  - `to_sparse(ltNone, ...)`（レイアウトが nullopt）は `ltSparse` と同じく sparse COO。
  - **エラーの分割を本物のメッセージで確認**：`Message` = "mat1 and mat2
    shapes cannot be multiplied (2x3 and 2x3)"、`Detail` は 2,217 文字で
    `Message` から始まり `Exception raised from` を含む。
  - 終了時 `TchLiveHandles = 0`。

- **6. 完了：マイルストーン1の手書き API**（`src/Tch.pas`）。テストは
  `tests/api/api_test.pas`（ビルド方法はファイル先頭）、全件合格：
  - `Numel`、`Device`、`ShallowClone`、`Copy`。`FromArray`（Single、Double、Int64、
    Byte、Boolean）、`FromInt32Array`、`FromBytes`。`To{Single,Double,Int64}Array`
    （先に変換）、`CopyData`（自分の kind のまま）。
  - 演算子：テンソル同士の `+ - * /`、単項 `-`、`** Double`、`Double`/`Int64` との
    `+ - * /`（左右両方）。`d - A` は `rsub_scalar`、`d / A` は `reciprocal * d`。
    Int64 テンソルに整数の指数 `-1` で `pow` すると失敗するため（実測、DESIGN 4.1 の表）。
  - 自動微分：`Backward(KeepGraph, CreateGraph)`、`RequiresGrad`、`ZeroGrad`。
    `grad`/`requires_grad_` は生成 op。grad モードとグローバル設定（4.7）：
    `SetGradEnabled`、`NoGradGuard`（代入しない呼び出しは効かないことをテスト済み）、
    `ManualSeed`、`Set/GetNumThreads`、`CudaIsAvailable`、`CudaDeviceCount`、
    `TDevice.CudaIfAvailable`。
  - CPU 上のファクトリのオーバーロード `zeros/ones/empty/rand/randn(ASize; AKind = kFloat)`。
    実測：`objfpc` では `overload` 指示子は不要。tchgen は `overload;` 付きのメンバーを
    意図したオーバーロードとして扱い、複数行の宣言をまとめて読むようにした（DESIGN 4.4）。
  - `KindElementSize`：46 種すべてについて libtorch と照合済み（DESIGN 4.3）。
  - `TTensor` のメソッドで `T` という名前のローカル変数を使うと生成 op の `t` と
    衝突する（コンパイルエラー、4.4 の落とし穴）。手書きのローカル変数は1文字の
    op 名を避ける。
  - `tests/mock/mock_tch.c` に新しい手書き external のスタブを追加。

次：クラウドセッション4で完了（このファイルの先頭）。続けて、そこに挙げたマイルストーン2のローカル実装。

ローカルセッションでやること（この順で。すべて機械的）：

1. **libtch を再ビルド**：`csrc/libtch/torch_api.cpp` に autocast のパッチ
   （`is_autocast_enabled(at::kCUDA)` / `set_autocast_enabled`）が入っているが、
   クラウドではコンパイルしていない。そのあと `tests/abi` と `tests/fpu` を前と同じく。
2. **本物のツールチェーンで `gen/ops.tsv` を再現**：
   `(cd gen && dune build ./gen.exe)` のあと
   `gen/_build/default/gen.exe <tch-rs>/third_party/pytorch/Declarations-v2.13.0.yaml csrc/libtch gen/ops.tsv`。
   `git diff` が空になるはず（ops.tsv と C のファイル）。デフォルト値の文字列が
   違ったら、代替の YAML パーサーがスカラーに別の型を付けていたということ：
   gen.ml の `default_text` を直し、tchgen を再実行して、本物の出力をコミットする。
3. **tchgen を実行**（`fpc -FUbuild/gen -obuild/gen/tchgen gen/tchgen.pas &&
   build/gen/tchgen gen/ops.tsv src`）。差分なしのはず。
4. **ownership テストを本物の libtorch で実行**（ビルド方法はファイル先頭）。
   それから `tests/mock/run.sh`（gcc と fpc だけで動く）。
5. **生成した op を本物の libtorch で試す最初のテスト**（新規、小さく）：
   `split`、`max_dim`、`sort_values`、`cat`、`add_scalar(1.5)`、
   `sum_dim_intlist(IntList([]))` と `NoIntList`、`pad(..., 'constant', NoDouble)`、
   `linalg_norm`（Scalar + IntListOption + デフォルト）、`LayoutOption` の op を1つ
   （`to_sparse`）、そして `Message` にはバックトレースがなく `Detail` にはあるエラー。
   tch-rs/PyTorch の結果と比べる。
6. その次にマイルストーン1の手書き API（DESIGN 6 章）：`FromArray` の
   オーバーロード、`ToDoubles`/`CopyData`、スカラーの演算子（`+ 1`、`* 2.5`）、
   `Zeros/Ones` の便利なオーバーロード（生成した名前をオーバーロードするのに
   `overload` が要るか確認）、自動微分。

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
8. **生成器。クラウドセッション3で完了**（このファイルの先頭を参照）：gen.ml が
   `ops.tsv` を書き、`gen/tchgen.pas` が `TchSysGenerated.pas` と `.inc` を書き、
   全体がコンパイルでき、`tests/mock` で生成層が動く。ローカルでは本物の `yaml`
   ライブラリで `ops.tsv` を再現し、ownership テストを走らせ、生成した op を本物の
   libtorch で試す最初のテストを書く。

## 未解決の論点

- **ネイティブのエラーメッセージの形：決定済み**（クラウドセッション3）。`Message` は
  `#10'Exception raised from '` より前の本文、`Detail` は全文（DESIGN_jp.md の 4.2。
  `ETorchNativeError.Create` に実装済み）。ローカルセッション2で
  本物の libtorch のメッセージで確認済み。
- **26 を超える `TKind`：決定済み**（クラウドセッション3）。2.13 の 46 個全部に
  広げ、省略可能な dtype 用に `kNone = -1` を足した（DESIGN_jp.md の 4.3）。
  要素サイズはローカルセッション2で解決：`FromBytes`/`CopyData` はすべての kind を
  扱い、サイズは libtorch と照合済み（DESIGN_jp.md の 4.3）。
- **libtorch の警告：決定済み**（クラウドセッション3）。autocast の2関数のフォーク
  パッチだけ（ローカルではまだコンパイルしていない）。c10 の警告ハンドラーは
  スレッドローカルなので、Pascal のコールバックは必要になるまで保留
  （DESIGN_jp.md の 4.2）。
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
- **複数出力の op**：`out` 引数（決定済み。149 op）。結果が渡したテンソルそのものの
  97 op には out 引数なし（クラウドセッション3、DESIGN_jp.md の 4.4）。まだ未決：
  実際にコードを書いてから、よく使う op（`max_dim`、`sort`、`topk`、`lstm`）に
  手書きの結果用 record を足す価値があるか。(H8) が実際に問題になるなら `var`。
- **デフォルト引数：決定済み**（クラウドセッション3、DESIGN_jp.md の 4.4）：479 op に
  デフォルトが付く。out 引数のある op には付かない（107 op には付いたはず）。
  ローカルセッション2：`zeros`、`ones`、`empty`、`rand`、`randn` のオーバーロードを
  追加。`objfpc` では `overload` 指示子は不要（実測）だが、tchgen は意図した
  オーバーロードの目印に `overload;` を使う（DESIGN_jp.md の 4.4）。未決：out 引数の op に `topk(k)` 型のデフォルト。
- **等価性とコンテナ：決定**（クラウドセッション4）。`TTensor` の `=` は
  オーバーロードしないまま。`TchNN` に必要なのは追加のみの `array of TVarEntry`、
  `TDictionary<String, Integer>` の名前の索引、`TObjectList<TModule>` だけで、
  すべて 3.2.2 で計測済み（DESIGN_jp.md 4.6.2、4.6.6）。必要になれば
  `TList<TTensor>` も動く（バイト比較 = ハンドルの同一性）。
- **Train フラグ：決定**（クラウドセッション4）。`Forward(X, Train)` のまま。
  `TSequential` が `Train` を全ての子に渡し、ユーザーの `TModule` は明示的に渡す
  （DESIGN_jp.md 4.6.6）。`Training` プロパティは作らない。
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
