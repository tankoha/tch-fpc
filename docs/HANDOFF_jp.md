# 引き継ぎ

（[HANDOFF.md](HANDOFF.md) の日本語訳です。内容が食い違う場合は英語版が正です。）

再開するときは最初にこれを読んでください。決定事項と理由は [DESIGN_jp.md](DESIGN_jp.md) にあります。

## 全体レビュー（2026-10-12、ローカルセッション8の後）— ここから再開

独立した Opus のレビュアー3体（コードと設計、英日ドキュメント、テストの網羅。読み取り
専用で、各セッションの文脈は渡していない）がリポジトリ全体を見た。指摘はすべてコードで
確かめてから対応した。修正してコミットしたもの：
- DESIGN 4.8 の本文と 6 章の状況欄を実測の「結果」に合わせた（GpuBackend の規則、L2 一覧
  に `at_context_has_cudart`、CPU ビルドで例外になる呼び出し、GPU テストの項目 1b/2/6/7/8、
  モックの項目、フューザーフラグ）。4.6.10 の平坦化の項目は 4.8 を指す。4.6.9 のモックの
  数に注記。マイルストーン3のレイヤーの状況段落。ROCm アーカイブの実際の名前。autocast
  パッチはコンパイル済みと明記。
- HANDOFF：古い記述への前方参照（H32 修正、Fable による設計セッションの決定、未決事項の
  完了、ROCm の完了）。設計セッション6〜8の見出しから対応するローカルセッションを指す。
  「(below)」→「(above)」。
- UML の状況表と TchSys の箱、`CLAUDE.md` の作業の流れ、`Tch.pas`・`TchSys.pas`・
  `tests/jit`・`tests/gpu` のソースコメント。
- テスト：埋め込みの `PaddingIdx` を勾配で確認（旧チェックは失敗し得なかった）。GPU
  テストで `GpuBackend`、H33、H34 を判定（変種は `run.sh` が渡す）。GPU 上のスカラーへの
  `DoubleValue` は 6 を返すこと。`tests/api` で CPU ビルドの GPU API を判定。H26 の辞書の
  場合も新しい `Raw` を確認。モックの GPU スタブは本物が失敗する所で失敗し、
  `jit_mock_test` にフューザーフラグのテスト。

**レビューで出て未修正のもの（ユーザーに聞く）：**
- コード：改名後の `name__N` が既にあると `TVarStore.AddEntry` がストアを半端な状態に
  する。例外を出したレイヤーのコンストラクタは変数をストアに残す（実測：`Groups = 0` の
  `TConv2D`）。`TBatchNorm` は `Nd` を検査しない。2つ目を巻き戻すか事前検査にするかは
  設計判断。
- DESIGN 3 は区別されたポインタ型（`type Pointer`）と `{$packrecords c}` を求めるが、
  コードは素の `Pointer` で packrecords もない。どちらに合わせるか決める。
- テスト：コンパイラの仮説（H1、H3、H4、H5、H10、H11 の一部）は印字だけで判定して
  いない。テストのない公開 API（メソッドを渡す `CreateByTracing`、
  `TTrainableJitModule` のほかのメソッド、`BackwardStepClipNorm` など）。GPU テストは
  `nn_test` の準備を共有せず写している。

## ローカルセッション8（2026-10-12）：GPU（4.8）を実装し、RX 9070 で動作

**現状。** 4.8 は実装・コミット済み。作業ツリーはクリーン。設計セッション8のローカル手順
1〜10 は完了。ローカルセッション4の終了時に戻した設計項目はすべて実装済みになった。

- `csrc/libtch/CMakeLists.txt`：`-DTCH_LIBTORCH_DIR=<dir>` の手動リンクモード
  （`option()` ではなく `CACHE PATH`）。既定の経路はビット単位で同一。ビルド：
  `build/libtch`（CPU、不変）、`build/libtch-rocm`（`cmake -S csrc/libtch -B
  build/libtch-rocm -DTCH_LIBTORCH_DIR=$PWD/libtorch-rocm -DCMAKE_BUILD_TYPE=Release
  && cmake --build build/libtch-rocm -j 12`、約1分）、`build/libtch-manual`（CPU
  アーカイブでの手動モード、確認用のみ）。
- `TchSys`：`atc_*`、`at_context_*`（`at_context_has_cudart` を追加）、
  `atm_fuser_cuda_*` の external。`Tch`：`CudnnIsAvailable`、`CudaManualSeed(All)`、
  `CudaSynchronize`、`CudnnUserEnabled`/`Set…`、`SetCudnnBenchmark`、
  `TGpuBackend`/`GpuBackend`、`GpuRuntimeVersion`、`CudnnVersion`、
  `TDevice.ToString`。`TchJit`：`JitFuserCudaEnabled`/`SetJitFuserCudaEnabled`。
  `TchNN`：`TRNN` の条件付き cuDNN 平坦化呼び出し。
- `tests/gpu/gpu_test.pas` ＋ `run.sh [rocm|cuda|cpu]`：RX 9070 で全項目 OK。`cpu` は
  `SKIP` を出す。モック：CPU ビルド相当のスタブと `TestGpuBackend`。CPU の jit テスト：
  fuser フラグ。CPU での回帰（abi、api、nn、jit、モック）：すべて OK。
- `libtorch-rocm/lib` には zip に含まれない **シンボリックリンクが8つ** 必要だった
  （SONAME のリンク7つと `libnuma.so`）。ないとプロセスが起動しない。コマンドは
  DESIGN 4.8「結果」にある。展開し直したらやり直すこと。

**実測で設計と違ったこと**（すべて DESIGN 4.8「結果」）：
- H34 不成立：ROCm に cuDNN はない（`CudnnIsAvailable` は False）ので、そこでは RNN の
  平坦化は呼ばれない（H30 は対象外）。RNN の数値は再現。
- H35：ROCm では `has_hip` False、`has_cuda` True、`has_cudart` False。`GpuBackend` は
  「HIP、または CUDART なしの CUDA」を ROCm とする。`GpuRuntimeVersion` は ROCm で例外。
- libtorch 2.13 では nvfuser が削除済み：`SetJitFuserCudaEnabled(True)` はどのビルド
  でも例外。
- `libtorch_rocshmem.so` は起動時に stderr へ `(null): No such file or directory` を
  4回出す（RDMA 用の `dlopen`）。害はない。
- Haiku がモックのスタブと2回の回帰を担当した。今回は作業も報告も正確だった（差分は
  レビュー済み）。

**未解決 / 次（設計済みで未実装のものは残っていない）：**
- CUDA の経路は未実測（ここに NVIDIA の GPU はない）：`gbCuda` の分岐、cuDNN の
  平坦化呼び出し、`GpuRuntimeVersion`/`CudnnVersion` の値。
- H29（tch-rs が書いた `.ot` との相互運用）は依然 Rust ツールチェーンが要る。
- 次のマイルストーンはまだ設計がない：ユーザーとテーマを決めて設計セッションを行う
  （セッション6〜8と同じく Fable サブエージェントで）。

## ローカルセッション7（2026-10-12）：マイルストーン3のレイヤー（4.6.10）を実装

**現状。** 4.6.10 は実装・コミット済み。作業ツリーはクリーン。設計セッション7のローカル
手順 1〜6、8、9 は完了。手順7（任意のモック確認）は省いた：名前と順序は libtorch に対して
確認している。

- `src/TchNN.pas`：`TConvConfigND`、`DefaultConvConfigND`、`ConvConfigToND`、
  `PadWithMode`（非公開の `ModeNames` を置き換え）、`TConvND` ＋ `TConv1D/2D/3D`
  （`TConv2D` は基底を付け替え、API は不変）、転置畳み込みの一族、`TGroupNorm`、
  `TRNNConfig`/`TRNNState`/`TRNN`/`TLSTM`/`TGRU`、`TIdentity`。
- `tests/nn/nn_test.pas` のグループ 11、4.6.10 の項目 1〜11（44 チェック）：libtorch
  2.13.0 CPU ですべて OK。実測済みの数値をすべて再現（conv1d の各モード、PyTorch 順の
  ND パディング `[[7, 10], [5, 8]]`、転置 1D/2D/3D、group norm ±0.99998、GRU
  `0.380797`、LSTM `0.380797 / 0.181700`、tch-rs の RNN 設定 10 通り、`BatchFirst
  False`）。グループ 8 の畳み込みの数値は新しい基底を通しても不変。
- 回帰：`tests/abi`、`tests/api`、`tests/nn`、`tests/jit`、モック — すべて OK。
- `docs/UML.md`/`UML_jp.md`：4.6.10 の要素から planned を外した。

**実装中に見つかった設計との違い**（DESIGN 4.6.10 の末尾に記録）：
- RNN のコンストラクタの引数は `AHiddenDim`：`HiddenDim` はプロパティ `HiddenDim` と
  重複識別子になる。
- protected の `TRNN.Create` は警告 3018 を出す。`TJitModule.CreateRaw` と同じく
  `{$push}{$warn 3018 off}` で抑える。

**次：** **4.8 GPU** — 未実装で残る唯一の設計項目。ROCm の前提はそろっている
（`libtorch-rocm/`、H32/H33）。下の設計セッション8のローカル手順の 1 から始める。
H31、H34、H35 は未解決。`TRNN` の cuDNN 平坦化呼び出し（H30）は入れ先の `TRNN` ができた。

## ローカルセッション6（2026-10-12）：TorchScript のオブジェクト（4.6.9）を実装

**現状。** 4.6.9 は実装・コミット済み。作業ツリーはクリーン。設計セッション6のローカル
手順 1〜8 は完了。

- `src/TchSys.pas`：external 4つ（`atm_create_class_`、`ati_object_method_`、
  `ati_object_getattr_`、`ati_clone`）。「未宣言」は `atm_fuser_cuda_*` だけになった。
- `src/TchJit.pas`：`ivObject = 14`、`IJitObject`（`TIValue` の前で前方宣言）、
  `TJitObjectHandle`、`FromObject`/`:=`/`AsObject`、`ToC` はクローン、`FromC` は
  ポインタを保持（`Keep`）、`TJitModule.CreateClass`、`JitLiveObjects`。
- `tests/jit/jit_test.pas`：`TestObjects`、4.6.9 の項目 a〜g（29 チェック）、
  libtorch 2.13.0 CPU ですべて OK。**H26、H27、H28 は成立**（結果は DESIGN 4.6.9 に
  記録）。`[O, 42]` は `:=` で要素ごとに変換される（未実測だった）。
- モック：`mock_tch.c` のタグ 14 オブジェクト、`jit_mock_test.pas` の `TestObjects`。
  `tests/mock/run.sh` は OK。
- 回帰：`tests/abi`、`tests/api`、`tests/nn`、`tests/jit`、モック — すべて OK。
- `docs/UML.md`/`UML_jp.md`：4.6.9 の要素から planned を外した。

**実装中に見つかった設計との違い：**
- `CreateClass(const ClassName: …)` はコンパイルできない：`ClassName` は
  `TObject.ClassName`（重複識別子）。引数名は `QualifiedName`。DESIGN 4.6.9 を更新。
- モックの数：モックのオブジェクトは1セル **＋属性ごとに1セル**（属性はクローンした
  ivalue）なので、設計の `mock_live_ivalues = 1` はテストでは `Cells`（`[1, 2.5]` で 3）。
  項目7（Device の兄弟が失敗）は文の前後の差分で確認する：ルーチン内では項目5〜6の
  一時変数がまだハンドルを持っている（H1）。差分は 0 — 変換済みの兄弟はすぐ解放される。

**Haiku との分担（このセッションから）。** ユーザーは、機械的な作業とテスト実行・報告に
Claude Haiku 4.5 のサブエージェントを使いたい。このセッションでは Haiku がモックのスタブ
（4.6.9 の仕様から）、UML の en/jp 同期、回帰の実行を担当した。モックにはリークがあり
（`ati_object_method_` が要素をクローンし、`ati_tuple` がさらにクローンしていた）、
その結果の失敗を「無関係」と報告した。UML の編集は1文を取りこぼした。どちらもここで
修正済み。したがって：Haiku には仕様が完全に決まった作業を渡し、失敗は解釈せずそのまま
報告させ、差分はレビューする。

**次（指定がなければどれからか聞く）：**
- **4.6.10 マイルストーン3のレイヤー**（CPU のみ。ローカル手順は下の設計セッション7）。
  4.8 の `TRNN` の flatten 呼び出しはこれに依存する。
- **4.8 GPU**（ROCm の前提はそろっている。設計セッション8のローカル手順の 1 から。
  H31、H34、H35 は未解決）。

## ローカルセッション5（2026-10-11〜12）：3つの設計がそろい、ROCm 版 libtorch を入手

**現状。** 以下はすべてコミット・push 済み。やりかけのものはなく、作業ツリーはクリーン。

- **設計セッション6〜8（下の各節）** は、このローカルセッションから Claude Fable 5.1 の
  サブエージェントとして1件ずつ実行し、ここで確認してコミットした：TorchScript の
  オブジェクト（DESIGN 4.6.9）、マイルストーン3のレイヤー（4.6.10）、CUDA/ROCm（4.8）。
  費用は Anthropic のクレジットで1件あたり約5〜6ドル。読む節を指定し、ツール呼び出しの
  目安を示し、ダウンロードを禁じた絞った依頼にすると後の回ほど安くなった。H36 はここで
  tch-rs 4227b89b の `src/nn/rnn.rs` と照合して解決（バイアスなしで stride0 を 2 にするのは
  明記済みの逸脱）。
- **`docs/UML.md` / `UML_jp.md`**：実装済みと予定の要素を描いた Mermaid の図8枚（ユニット、
  ユニットごとのクラス図、ForwardIV の所有権とファインチューニングのシーケンス、ビルドの
  変種）と状況表。予定の要素を実装したら更新する。
- **ROCm 版 libtorch を入手**（設計セッション8のユーザー側の前提1）：
  `https://download.pytorch.org/libtorch/rocm7.1/` の
  `libtorch-shared-with-deps-2.13.0+rocm7.1.zip` を `libtorch-rocm/` に展開（12GB、
  `build-version` は `2.13.0+rocm7.1`。zip は展開後に削除）。`.gitignore` に `libtorch-*/`
  を追加。アーカイブ名から `cxx11-abi` は消えているが cxx11 ABI。2.14.1+rocm7.14（1.6GB）も
  あったが、生成 op が 2.13.0 由来なので採らなかった。
  - **(H33) 解決：** `libtorch_hip.so` に `gfx1201`（RX 9070）がある。gfx1200、gfx11xx、
    gfx1030、gfx9xx もある。`gfx1036`（iGPU）はないので `HIP_VISIBLE_DEVICES=0` は引き続き
    必須。
  - **(H32) 解決：** `ldd libtorch_hip.so` で not found なし。`libamdhip64`、`libMIOpen`、
    `librocblas`、`libhipblaslt`、`librccl`、`libhsa-runtime64` が `lib/` に同梱。AMD の
    ROCm パッケージは不要。
    **ローカルセッション8で修正：** zip にはシンボリックリンクがなく、何かを読み込む前に
    8つ足す必要がある（DESIGN 4.8「結果」）。
  - **罠：** `LD_LIBRARY_PATH=libtorch-rocm/lib` にするとシステムのツールが同梱の
    `libtinfo.so.6` を読む（bash が "no version information available" と警告）。
    全体の `LD_LIBRARY_PATH` は使わず、DESIGN 4.8 のビルドごとの rpath を使う。
- **ユーザーとの約束（自動メモリにもある）：** 大きなダウンロードが終わったら報告して
  止まる。展開・検証は指示があってから。

**次（ユーザーが 18:00 のトークンリセット後に始める。指定がなければどれからか聞く）：**
- **4.8 GPU** — ROCm の前提はそろったので、下の設計セッション8のローカル手順を1から
  始められる（ユーザー側の前提1〜3は完了：アーカイブあり、H32 成立、ローカルログインから
  デバイスにアクセス可）。未検証の仮説は H31、H34、H35。
- **4.6.9 TorchScript のオブジェクト** と **4.6.10 マイルストーン3のレイヤー** — CPU だけで
  できる。ローカル手順は下の設計セッション6と7の節。
- いつもどおり最後に DESIGN/HANDOFF/UML（en + jp）を更新する。

## 設計セッション8（2026-10-11）：CUDA/ROCm を設計 — 上のローカルセッション8で実装済み

下の「引き継ぎ（ローカルセッション4の終了時点）」の項目3への回答 — 最後に残っていた
項目。設計のみ：何もコミットせず、何もダウンロード・インストールしていない。スクラッチ
パッドで使い捨ての cmake ビルドを 1 回。詳細と理由は DESIGN 4.8。

**決定**（DESIGN 4.8）：
- 変種ごとに展開した libtorch を 1 つ（`libtorch/` は今のまま CPU、`libtorch-rocm/`、
  `libtorch-cuda/`）、ビルドディレクトリも変種ごと（`build/libtch`、`build/libtch-rocm`、
  `build/libtch-cuda`）。Pascal ユニット、C ラッパー、ABI は不変：HIP は CUDA を装う
  （`libtorch/include/ATen/Context.h:638-650` で確認）ので、`TDevice.Cuda(0)`、
  `device_of_int`、`atc_*` はそのまま AMD GPU を指す。
- `csrc/libtch/CMakeLists.txt` に `-DTCH_LIBTORCH_DIR=<dir>` を追加：`find_package(Torch)`
  の代わりに手動のインクルード/リンク行（`-Wl,--no-as-needed libtorch.so`、続いて
  `libtorch_cpu.so libc10.so`、`-D_GLIBCXX_USE_CXX11_ABI=1`、`BUILD_RPATH <dir>/lib`）。
  理由は ROCm アーカイブの `Caffe2Config.cmake` が `LoadHIP.cmake` を include し、それが
  `/opt/rocm` の ROCm SDK を要求するから。CPU アーカイブ向けの既定経路は不変。
- L2：`atc_cudnn_is_available`、`atc_manual_seed(_all)`、`atc_synchronize`、
  `atc_user_enabled_cudnn`、`atc_set_user_enabled_cudnn`、`atc_set_benchmark_cudnn`、
  `at_context_has_cuda/hip/cudnn`、`at_context_version_cudart/cudnn`、`atm_fuser_cuda_*`。
  L3（4.7 のブロック）：`CudnnIsAvailable`、`CudaManualSeed(All)`、
  `CudaSynchronize(Index = -1)`、`CudnnUserEnabled`/`SetCudnnUserEnabled`、
  `SetCudnnBenchmark`、`TGpuBackend = (gbNone, gbCuda, gbRocm)` + `GpuBackend`、
  `GpuRuntimeVersion`、`CudnnVersion`、`TDevice.ToString`；`TchJit`：
  `JitFuserCudaEnabled`/`SetJitFuserCudaEnabled`。CPU ビルドで安全なのは
  `CudaIsAvailable` / `CudaDeviceCount` だけで、ほかは `ETorchNativeError` になる。
- `TRNN`：`Device.IsCuda and CudnnIsAvailable` のとき tch-rs の平坦化呼び出し
  （`_cudnn_rnn_flatten_weight(FFlatWeights, 4 または 2, InputSize, Mode 2/3, HiddenSize, 0,
  NumLayers, BatchFirst, Bidirectional)`）、結果は捨て、`ETorchNativeError` は握りつぶす
  （ヒントに過ぎない。MIOpen は拒否しうる）。
- `tests/gpu/gpu_test.pas` + `run.sh <変種>`：`CudaIsAvailable` が False なら `SKIP` と
  exit 0；それ以外は 4.6.7/4.6.8/4.6.10 の CPU の数値（`3 5 7`、`6`、`[0.9, 0.8]`、
  `0.380797`、`0.1817`）を再利用する 10 項目、H13 の実デバイス移動（項目 5）と H30
  （項目 8）。Raphael iGPU（`gfx1036`、非対応）を隠して `CudaDeviceCount = 1` にするため
  `HIP_VISIBLE_DEVICES=0` で走らせる。
- モック：新しい `atc_*`/`at_context_*` スタブは CPU ビルドと同じ値を返す；`mock_test.pas`
  は `GpuBackend = gbNone` とフューザーフラグの往復を確認。

**確認したこと**（何も変えていない）：`rocm-smi --showproductname` → GPU[0] RX 9070
`gfx1201`、GPU[1] Raphael iGPU `gfx1036`；`/sys/class/kfd` に両方ある（kernel 7.0.0 の
amdgpu/KFD は動作）。インストール済みの ROCm ユーザー空間は **`rocm-smi 7.1.1` +
`librocm_smi64`（Ubuntu）だけ**；`/opt/rocm` なし、`libamdhip64` なし、MIOpen なし、
`rocminfo` なし。ユーザーのグループ：`joerg adm cdrom sudo dip plugdev users lxd ollama` —
`render`/`video` には入っていないが、シートの ACL により `/dev/kfd` は既にユーザーから
読み書きできる。実測：手動リンクの CMakeLists で CPU アーカイブに
対して libtch が 28 秒でビルドできる；できた .so の `NEEDED` は `libtorch_cpu.so` +
`libc10.so` だけ（as-needed が傘を落とした — ゆえに GPU ビルドでは明示の
`--no-as-needed`。バックエンドは `libtorch.so` 経由でしか届かない）。

**新しい仮説**（ROCm 版 libtorch が要る。ここでは何も計測できない）：
- **H31** 手動リンクの libtch は ROCm アーカイブに対してもそのまま動く（`torch_api.cpp`
  に `__HIP_PLATFORM_AMD__` などの追加 define は不要）。
- **H32** `libtorch-cxx11-abi-shared-with-deps-2.13.0+rocmX.Y` アーカイブは HIP ランタイム、
  MIOpen、rocBLAS、hipBLASLt、RCCL を `lib/` に同梱している（`ldd
  libtorch-rocm/lib/libtorch_hip.so | grep 'not found'` で確認）。していなければ同じ X.Y の
  AMD ROCm ランタイムパッケージ。
- **H33** そのアーカイブに `gfx1201` のコードオブジェクトがある（`strings -n 7
  libtorch_hip.so | grep -oE 'gfx[0-9a-f]+' | sort -u`）。ローカルでは見えない；PyTorch の
  ROCm 7.x ビルドにはあるはずで、ROCm 6.x にはない見込み。
- **H34** ROCm で `atc_cudnn_is_available` は 1（hipify されたフック経由で MIOpen が cuDNN の
  代わりになる）。
- **H35** ROCm で `at_context_has_hip` は True；`at_context_has_cuda` も True かは不明
  （`GpuBackend` は HIP を先に見るのでどちらでも正しい。テストが両方を印字）。
- **H36** tch-rs `rnn.rs` の平坦化呼び出しは記憶から再構成（リポジトリはローカルにない）：
  `is_cuda && cudnn_is_available`、stride0 4、mode 2（LSTM）/ 3（GRU）、proj_size 0。
  **解決**：取りまとめ側のセッションが tch-rs 4227b89b `src/nn/rnn.rs`（154〜166 行、
  239〜251 行）で確認し一致。tch-rs はバイアスなしでも stride0 を 4 に固定するが、DESIGN 4.8
  はその場合 2（PyTorch）とし、明記済みの逸脱。

**ユーザー側の前提**（設計セッションでは行っていない。説明のみ）：
1. download.pytorch.org から `libtorch-cxx11-abi-shared-with-deps-2.13.0+rocmX.Y.zip`
   （CPU と同じ 2.13.0 の ROCm 版。2.13.0 向けに提供される最も高い ROCm バージョンを選ぶ —
   RDNA4/gfx1201 は ROCm 7.x が要る）を取得し、リポジトリ直下に `libtorch-rocm/` として
   展開（git-ignore）。何かビルドする前に上の 2 つのワンライナーで H33 と H32 を確認。
2. H32 が外れたら：X.Y に対応する AMD の ROCm ランタイムパッケージ（`rocm-hip-runtime`、
   `miopen-hip`、`rocblas`、`hipblaslt`、`rccl`、`hipfft`、`hipsparse`、`rocsolver`、
   `hiprand`）を AMD のリポジトリから入れる。手動リンクモードなら SDK（`rocm-dev`/cmake
   ファイル）は不要。
3. デバイスへのアクセス：`/dev/kfd` と `/dev/dri/renderD*` は `root:render` に ACL（`+`）
   付きで、デスクトップのログインからは `test -r /dev/kfd -a -w /dev/kfd` が既に通る
   （logind がシートに付与）ので、ローカルセッションからの実行には何も要らない。ssh/cron
   から走らせる場合だけユーザーを `render` と `video` に追加（`sudo usermod -aG
   render,video joerg`、再ログイン）。アクセスがないと `rocm-smi`（sysfs を読む）には
   見えていても HIP はデバイスなしと報告する。
4. CUDA 機の場合：`+cuXYZ` アーカイブを `libtorch-cuda/` に、NVIDIA ドライバ。ほかは変種を
   `cuda` にするだけで同一。

**ローカル実装リスト**（機械的。順序に意味がある）：
1. `csrc/libtch/CMakeLists.txt`：DESIGN 4.8 の `TCH_LIBTORCH_DIR` 分岐；従来の方法で
   `build/libtch` を再ビルドして変化なしを確認し、`-DTCH_LIBTORCH_DIR=$PWD/libtorch` で
   `build/libtch-manual` に 1 回ビルドして手動経路（CPU）が `tests/api` を通すことを確認。
2. `src/TchSys.pas`：上の L2 外部宣言（`uint64_t` は QWord、`int64_t` は Int64、C `bool` は
   `Boolean`、`int` は `cint`）；ユニットコメントの「未宣言」リストを空にする。
3. `src/Tch.pas`：4.7 ブロックへの追加、`TGpuBackend`、`GpuBackend`、`TDevice.ToString`；
   どの呼び出しが CPU で安全かを API コメントに書く。
4. `src/TchJit.pas`：`JitFuserCudaEnabled`/`SetJitFuserCudaEnabled`。
5. `src/TchNN.pas`：`TRNN` の protected コンストラクタ（`FFlatWeights` が完成したあと）に
   上記のガード付き・握りつぶしの平坦化呼び出し。
6. `tests/mock/mock_tch.c` のスタブ + `mock_test.pas` の確認；`tests/jit` に CPU での
   フューザーフラグの往復を追加。CPU ビルドで `tests/mock/run.sh`、`tests/api`、`tests/nn`、
   `tests/jit` を実行：変化なし。
7. `tests/gpu/gpu_test.pas` と `tests/gpu/run.sh`（変種引数、可視性変数、
   `-Flbuild/libtch-$v -k-rpath=...`）；まず `build/libtch` に対して走らせる：ちょうど
   `SKIP: no GPU in this libtorch`、exit 0。
8. ユーザー側の前提が済んだら：`cmake -S csrc/libtch -B build/libtch-rocm
   -DTCH_LIBTORCH_DIR=$PWD/libtorch-rocm -DCMAKE_BUILD_TYPE=Release && cmake --build
   build/libtch-rocm -j 8`；`ldd build/libtch-rocm/libtch.so | grep 'not found'` が空で
   あること。
9. `tests/gpu/run.sh rocm`：項目 1 の能力印字、H13 と H30 の結果、H31〜H35 の結果を各仮説の
   横（DESIGN 4.8 とここ）に記録；次に CPU リンクの `tests/nn` と `tests/jit` のバイナリの
   前に `LD_LIBRARY_PATH=build/libtch-rocm` — GPU 版 libtorch をロードしても CPU テンソルで
   全部通る（同じ ABI）。
10. ここに「ローカルセッション7」の節を追加（実際はローカルセッション8になった）（en + jp）；H31〜H36 を解決または修正済みと
    記す；数値が違えば DESIGN 4.8 を更新。

## 設計セッション7（2026-10-11）：マイルストーン3のレイヤーを設計 — 上のローカルセッション7で実装済み

下の「引き継ぎ（ローカルセッション4の終了時点）」の項目2への回答。項目3
（CUDA/ROCm）は未着手のまま。設計のみ：何もコミットせず、スクラッチパッド外に
コードなし。ローカル機で本物の libtorch 2.13.0（CPU）と FPC 3.2.2 に対して
実行したので、DESIGN 4.6.10 の数値は導出ではなく実測。

**決定事項**（詳細と理由は DESIGN 4.6.10）：
- tch-rs のジェネリックな `ConvConfigND<ND>` は、`TInt64Array` フィールドを持つ
  ジェネリックでない1つの `TConvConfigND`（`Nd = Length`）、1つの基底クラス
  `TConvND`（`Forward` は `FNd` で分岐）、スカラー設定のコンストラクタだけを持つ
  サブクラス `TConv1D/2D/3D` になる。`TConv2D` の API は不変、本体は基底へ移る。
  `ConvConfigToND(Cfg, Nd)`、`DefaultConvConfigND(Nd)`、`PadWithMode`（公開、
  `pmZeros` は `'constant'`）を追加。`TConvTransposeND` + `1D/2D/3D` と
  `TConvTransposeConfig[ND]` も同じ形（weight は `[In, Out div Groups, K..]`）。
- `TGroupNorm(NumGroups, NumChannels[, Cfg])` は `group_norm` 上。`Affine` のとき
  weight、bias `[NumChannels]` の順。
- `TRNN = class abstract`（tch-rs と同じく `TModule` ではない）に `ZeroState`、
  `Step`、`Seq`、`SeqInit`。`TLSTM` と `TGRU` で1つの
  `TRNNState = record H, C: TTensor end`（GRU は `C` を nil のまま）。`TRNNConfig`
  は共通。`FFlatWeights` は層ごと方向ごとに `weight_ih_l{L}[_reverse]`、
  `weight_hh_…`、`bias_ih_…`、`bias_hh_…` として登録。`SeqInit` は生成された
  `lstm`/`gru` を呼び、`out` の結果を関数結果（出力）と `OutState.H`/`.C` に受ける。
- `nn::Id` は `TIdentity`。
- tch-rs からの逸脱（それぞれ 4.6.10 に明記）：reversed-padding-repeated-twice は
  PyTorch の順（tch-rs のは非対称な ND パディングで誤り、実測）。`BatchFirst False`
  を `Seq`/`Step` で扱う。バイアス初期化は `Option<Init>` でなく素の `TInit`。
  `ZeroState` は kind/device を呼び出し時に最初の重みから読む。cuDNN 平坦化は
  CUDA/ROCm マイルストーンへ。

**計測**（プローブ `scratchpad/fable-layers/probe*.pas`、FPC 3.2.2 + libtorch 2.13.0
CPU）：
- M-A：`array of Int64` を取る基底コンストラクタ + サブクラスの `overload`
  コンストラクタ2つは、スカラー呼び出しも、サブクラス経由の継承 ND 呼び出しも解決する。
  グループ8の conv2d の数値は基底経由で不変。
- M-B / M-B2：`SeqInit(X, S, S)` は入口で `S` を nil にする（H8）→ `ETorchError
  "undefined tensor passed as hx to gru"`。`S := Step(X, S)` は安全（結果は一時変数経由）。
- M-C：4.6.10 のテスト一覧の全数値（conv1d の4モード、conv3d、非対称 ND パディング、
  conv_transpose 1D/2D/3D（stride 2 と `OutputPadding` を含む）、group_norm
  ±0.99998、GRU `0.380797`、LSTM `0.1817 / 0.380797`、tch-rs の5つの RNN 設定の形と
  変数名、`batch_first` False の形）。

**新しい仮説**（ここの CPU libtorch では測れない）：**H29** tch-rs が書いた
RNN/conv-transpose/group-norm の `.ot` ファイルとの相互運用（Rust が要る）。
**H30** CUDA/ROCm の store での cuDNN 平坦化呼び出し（項目3）。

**ローカル実装リスト**（機械的。すべて `src/TchNN.pas` と `tests/nn/nn_test.pas`、
L2 と C の変更なし）：
1. `TConvConfigND`、`DefaultConvConfigND`、`ConvConfigToND`、`PadWithMode`
   （private の `ModeNames` を置き換え、`pmZeros` は `'constant'`）を追加。
   `TConvND` に ND コンストラクタ（長さ検査は `ETorchError`）、
   `FPadTwice[2*I] = FPadTwice[2*I+1] = Padding[Nd-1-I]`、`Forward` は 4.6.10 の表
   どおり。`TConv2D` をその上に載せ替え（本体 →
   `inherited Create(P, In, Out, [K, K], ConvConfigToND(Cfg, 2))`）、`TConv1D`、
   `TConv3D` を追加。既存のグループ8を実行：不変。
2. `TConvTransposeConfig`、`TConvTransposeConfigND`、`Default*` 2つ、
   `ConvTransposeConfigToND`、`TConvTransposeND`、`TConvTranspose1D/2D/3D`。
3. `TGroupNormConfig`、`DefaultGroupNormConfig`、`TGroupNorm`。
4. `TRNNConfig`、`DefaultRNNConfig`、`TRNNState`、`TRNN`（protected コンストラクタ
   = `rnn_weights`、`GateMul` は 4 か 3。`Step`、`Seq`、`BatchDimIndex`）、`TLSTM`、
   `TGRU`（`ZeroState`、`SeqInit`）。`InState`/`OutState` は別変数という規則を宣言の
   そばに書く。
5. `TIdentity`。
6. `tests/nn/nn_test.pas` グループ11、4.6.10 の一覧の項目 1–11。最後の
   `TchLiveHandles = 0` は維持。
7. モック（`tests/mock`）：数値的に足すものはない。任意で、形だけのスタブ上で
   `TLSTM(2, 4)` の生成と登録順（2層双方向で16名）。
8. 回帰：`tests/nn`、`tests/api`、`tests/mock/run.sh`、`tests/jit`。
9. H29/H30 の結果はテスト可能になった場合だけ記録。ここに「ローカルセッション6」
   の節を追加（実際はローカルセッション7になった）（en + jp）。「引き継ぎ（ローカルセッション4の終了時点）」の項目3が
   残る設計作業。

## 設計セッション6（2026-10-11）：TorchScript のオブジェクトを設計 — 上のローカルセッション6で実装済み

下の「引き継ぎ（ローカルセッション4の終了時点）」の項目1への回答。項目2
（マイルストーン3のレイヤー）と3（CUDA/ROCm）は未着手のまま。設計のみ：何もコミット
せず、スクラッチパッド外にコードは置いていない。セッションはローカル機で動かしたので、
設計は本物の libtorch 2.13.0（CPU）とビルド済みの `libtch.so` に対して実測している。
すべて **DESIGN 4.6.9** にある。要約：

1. **ハンドル。** `IJitObject`、`TInterfacedObject` で参照カウントするインターフェース
   （4.1 の `TTensorHandle` と同様。インターロック、決定的な解放）。実装クラス
   `TJitObjectHandle` は implementation 部に隠す。レコードではなくインターフェースなのは、
   `TIValue` がフィールドを必要とし、オブジェクトのメソッドが `TIValue` を受け取り返す
   から：FPC はレコードを前方宣言できないがインターフェースはできる（実測）。
   `JitLiveObjects` が `TchLiveHandles` と同様に生きているハンドルを数える。
2. **`TIValue`。** `ivObject = 14`、フィールド `Obj: IJitObject`、`FromObject`、
   `IJitObject` からの `:=`、`AsObject`、`TagName = 'Object'`。**`FromC` はタグ 14 では
   ポインタを保持する**（`Keep` フラグ。ハンドルが所有）。**`ToC` は
   `ati_clone(Obj.Raw)` を返す** — 呼び出し側がクローンを解放し、ハンドルのポインタは
   解放のために C に渡されない（tch-rs の issue 475 の修正）。コンテナ内では特別な
   ことはない：`ati_to_*` は新しい `IValue` を渡し（実測）、後の子が失敗すれば
   変換済みの兄弟は `IValuesFromC` のローカル配列をコンパイラが巻き戻すことで解放される。
3. **API。** `IJitObject.MethodIV(Name, Inputs)` と `GetAttr(Name)`
   （`TJitModule.MethodIV` の形）。`TJitModule.CreateClass(ClassName, Inputs): IJitObject`
   （修飾名、例 `'__torch__.InputObject'`。tch-rs の `IValue` ではなくオブジェクトを返す、
   明記）。オブジェクトはモジュールを生かしておかず、その必要もない（実測：`atm_free`
   の後もメソッド呼び出しが動く）。
4. **L2。** external 4つ（`atm_create_class_`、`ati_object_method_`、
   `ati_object_getattr_`、`ati_clone`）、宣言は 4.6.9。**フォークへのパッチ：なし**
   （`atm_create_class_` の未使用ローカルは無害）。
5. **テスト。** `tests/jit/jit_test.pas` のグループ 14：`jit6`（`foo6.pt`、`y()` →
   `[6, 8, 10]`）、`jit_double_free`（`foo7.pt`、`make_input_object` の後 `add_them` →
   `[5, 7, 9]`、同じハンドルで2回）、`CreateClass`、実測したエラーメッセージ、タプル／
   リスト／辞書を通るオブジェクト、共有、`M.Free` 後のオブジェクト、数が戻ること、
   `10^4` 回のループ。モック：名前付き属性 `a0, a1, …` を持つタグ 14 の `mock_ivalue`、
   `mock_last_class`、`TestObjects`（8つの確認）。

**計測（libtorch 2.13.0 CPU、FPC 3.2.2。プローブはスクラッチパッドのみ）：**
- M9：`foo6.pt`/`foo7.pt` に生の C エントリポイント — 両オブジェクト結果で `ati_tag` は
  14。`y()` → `[6, 8, 10]`。`getattr x` → `[3, 4, 5]`。`ati_clone` は元の `ati_free` の後も
  モジュールの `atm_free` の後も動く。タプルの子は新しいポインタとして戻り、タプル解放
  後も使える。同じオブジェクトポインタで `add_them` を2回 → `[5, 7, 9]` が2回。
  `atm_create_class_('__torch__.InputObject', [T1, T2])` → `add_them` `[5, 7, 9]`。素の
  クラス名は失敗。エラーメッセージ9つを 4.6.9 に原文で記録（属性／メソッドなし、引数の
  数、オブジェクトへの `to_tensor`/`length`、未知のクラス、`__init__` の引数不足、引数の
  型違い、非オブジェクトのポインタ）。
- M10：FPC は `IJitObject = interface;` の前方宣言 → レコードのフィールド → レコードを
  使う完全なインターフェースを受け付ける。コピーはハンドルを共有、解放は決定的、
  ルーチンが戻った後に数は 0 に戻る（ルーチン内では一時変数が 1 を保つ、H1）。
  `SizeOf` は 16。
- M11：`atm_forward` はテンソルでない結果すべてに "forward did not return a tensor" を
  投げる（`torch_api.cpp` を読んだだけで、オブジェクトでは走らせていない → H27）。

**新しい仮説（本物の libtorch が必要）：**
- **(H26)** オブジェクトはタプルと同様に `GenericList` と `GenericDict` の値としても
  往復する（コンテナより長生きする新しいタグ 14 の `IValue`）。
- **(H27)** `foo6.pt` への `CreateClass('__torch__.TorchScriptClass', [x])` は
  `y() = 2x`。`foo6.pt` への `Forward`（テンソル API）は "forward did not return a
  tensor" を出す。
- **(H28)** `10^4` 回のオブジェクトメソッド呼び出しで `JitLiveObjects`、
  `TchLiveHandles`、RSS が平坦。

**ローカルセッション、この順で（機械的に）：**
1. `src/TchSys.pas`：4.6.9 の external 4つを `ati_free` の後に。ユニットコメントの
   「未宣言」リストは `atm_fuser_cuda_*` だけにする。
2. `src/TchJit.pas`、interface 部：`TIValueTag` に `ivObject = 14`。`TIValue` の前に
   `IJitObject = interface;`。フィールド `Obj: IJitObject`、`FromObject`、
   `class operator :=(const O: IJitObject)`、`AsObject`。`TIValue` の後に完全な
   `IJitObject`（新しい GUID。`Raw`、`MethodIV`、`GetAttr`）。`TJitModule.CreateClass`。
   `function JitLiveObjects: Integer`。ヘッダコメント：「オブジェクト未対応」を消し、
   「オブジェクトはモジュールを生かしておかない」を足す。
3. `src/TchJit.pas`、implementation 部：`TagNames` → `'Object'`。`LiveObjects` カウンタと
   `JitLiveObjects`。`TJitObjectHandle`（コンストラクタの順序は `inherited; Inc; FRaw := P`、
   デストラクタは `ati_free` + `Dec`）。`AsObject`/`FromObject`/`:=`。`ToC` の `ivObject`
   分岐（`ati_clone`、nil の `Obj` → `ETorchError('ivalue: nil object')`）。`FromC` の
   case 14 に `Keep` ローカル、`finally` に `if not Keep then ati_free(P)`。
   `TJitObjectHandle.MethodIV`/`GetAttr` と `TJitModule.CreateClass` は 4.6.9 の手順5の
   とおり（`IValuesToC`、呼び出し、`TchCheck`、`finally FreeIValues`、`FromC`。
   `CreateClass` は `.AsObject` で終わる）。`tests/mock/run.sh` をコンパイル（手順4まで
   モックにシンボルはないが、テストプログラムは使うものだけを参照するので既存の3つは
   リンクできる）。
4. モック：`mock_ivalue` に `cls`/`names`、`iv_clone`/`ati_free` の更新、
   `atm_create_class_` + `mock_last_class`、`ati_object_getattr_`、`ati_object_method_`、
   `ati_clone`、14 で失敗する `ati_length`。
5. `tests/mock/jit_mock_test.pas`：`mock_last_class` の external、4.6.9 の8つの確認を
   持つ `TestObjects`（数は呼び出し側で比較）、最後に `JitLiveObjects = 0`。
   `tests/mock/run.sh`。
6. `tests/jit/jit_test.pas`：4.6.9 のグループ 14「Objects」項目 a〜g、H26〜H28 を確認。
   最後の `TchLiveHandles = 0` の隣に `JitLiveObjects = 0`。ファイルヘッダのとおり
   ビルドして実行。
7. 回帰：`tests/jit`、`tests/mock/run.sh`、`tests/abi`（C は不変）、`tests/api`、
   `tests/nn`。
8. H26〜H28 の結果を DESIGN 4.6.9 の仮説の横に記録（en + jp）、ここに「ローカル
   セッション5」の節を追加（en + jp）。「引き継ぎ（ローカルセッション4の終了時点）」の
   項目2と3が残りの設計作業。

## ローカルセッション4（2026-10-11）：マイルストーン3（JIT）を実装

下のクラウドセッション5の手順1〜8を完了。すべてのテストが通る。
コミットは `8a912eb` … `b816950`。

1. **`FromArray` の改訂**（DESIGN 4.3）：`Single` のオーバーロードを削除し、残り4つに
   `AKind` を追加（`Double` + `kFloat` は Pascal 側で `Single` のバッファに丸める）、
   `FromSingleArray` を追加。`tests/compiler` に新しい `h2_fromarray.pas` と
   `xfail_h2_single.pas`、`tests/api` と `tests/nn` も更新。テストで見つけた罠：
   x86-64 では比較式中の実数リテラルは `Extended` なので `T.DoubleValue([0]) = 0.9` は
   False。`Double(0.9)` と比較する。
2. **4.6.7 の積み残し：** `TOptimizer.CreateRaw` を `{$push}{$warn 3018 off}` で囲んだ
   （警告は消えた）。半精度の `LoadPartial` の確認を `TestSaveLoad` に追加（通る：
   `u2` は `kHalf` で平均 42、`v2` は `kFloat` で平均 1、欠けているのは `['a.b.t2']`）。
3. **フォークへのパッチ：** `csrc/libtch/torch_api.{h,cpp}` に `atm_create`、
   `atm_register_parameter`、`atm_define`。libtch を再ビルド、`tests/abi` とモックは通る。
4. `tests/jit/models/foo.pt` … `foo8.pt` を tch-rs の2つのライセンスファイル、出典を書いた
   短い README とともに**取り込み**。
5. **`src/TchSys.pas`：** 4.6.8 の JIT の型と external。
6. **`src/TchJit.pas`：** `TIValue`、`TJitModule`、`TTrainableJitModule`、`Jit*` の
   グローバル。小さな逸脱（DESIGN 4.6.8 の末尾に記録）：`TagName` は `case`（値を明示
   した列挙型を FPC は配列の添字にできない）。`CreateByTracing` は
   `atm_create_for_tracing` 以降の *あらゆる* 失敗で後始末する。`Item`/`Key`/`Value` は
   範囲を確認する。
7. **モック：** `tests/mock/mock_tch.c` に `atm_*`/`ati_*`、
   `tests/mock/jit_mock_test.pas`（`ToC` や `FromC` で子が失敗しても C の ivalue が
   残らないことも確認）、`run.sh` に追加。
8. **`tests/jit/jit_test.pas`：** 13 グループすべて通る。**H19〜H25 はすべて成立**
   （結果は DESIGN 4.6.8 末尾の仮説の横）。

- 回帰：`tests/api`、`tests/nn`、`tests/generated`、`tests/ownership`、`tests/abi`、
  `tests/compiler/run.sh`、`tests/mock/run.sh` は通る。`tests/fpu/run.sh` は従来どおり
  （計測用で、マスクしないモードは設計上クラッシュまたはハングする）。
- 依頼どおり手を付けていないもの：マイルストーン3のレイヤー、CUDA/ROCm。

## 引き継ぎ（2026-10-11、ローカルセッション4の終了時点）— 項目1、2、3は上の設計セッション6、7、8で回答済み

**設計のみ**（CLAUDE.md）。止まっているものはなく、以下はどれも任意の次の
マイルストーン。本物の libtorch が必要なものは H26 以降として記すこと。

1. **TorchScript のオブジェクト（ati_tag 14）** — *設計済み：DESIGN 4.6.9、上の
   設計セッション6。* `ati_object_method_`、
   `ati_object_getattr_`、`ati_clone`、`atm_create_class_`。tch-rs の `Object`、
   テスト `jit6` と `jit_double_free`、アセット `foo6.pt`/`foo7.pt`（取り込み済みだが
   未使用）。DESIGN 4.6.8 にある参照カウント付きハンドル（オブジェクトは C の `IValue`
   を生かし続ける）と、`TIValue` がそれをどう持つか（コピーは共有。現在 `FromC` は
   タグ 14 で例外）が必要。
2. **マイルストーン3のレイヤー**（DESIGN 4.6.5 の先送りリスト）— *設計済み：DESIGN
   4.6.10、上の設計セッション7。*
3. **CUDA/ROCm** — *設計済み：DESIGN 4.8、上の設計セッション8。* これまで CPU のみ
   （H13 の実デバイスへの移動は未検証、
   `atm_fuser_cuda_*` は未宣言）。ローカル機には AMD Radeon RX 9070（Navi 48）と
   `rocm-smi` がある。ROCm 版 libtorch はまだ試していない。よって先に設計する GPU の
   経路は ROCm（libtorch の HIP ビルドはデバイスを CUDA として報告する）。

**次の設計セッションの進め方は未定：** クラウドのサービスクレジットを使い切ったため、
Fable への設計作業の渡し方はユーザーが後で決める。それまでここで待つものはない。（ローカル
セッション5で決定：Fable 5.1 のサブエージェントをローカルセッションから1件ずつ走らせる。）

## クラウドセッション5（2026-10-10）：JIT を設計、`FromArray` を改訂、ローカルセッション3を確認 — 上のローカルセッション4で実装済み

下の依頼（「クラウドへの引き継ぎ、ローカルセッション3の終了時点」）の項目1〜3に
回答しました。項目4（マイルストーン3のレイヤー、CUDA/ROCm）には手を付けていません。
設計のみ：何もコミットしておらず、スクラッチパッドの外にコードは書いていません。
要約：

1. **JIT（DESIGN 4.6.8）。** L2：`PTorchModule`、`PTorchIValue`、そこに列挙した 21 個の
   `atm_*`/`at_set_graph_executor_optimize` と 31 個の `ati_*` の external
   （`atm_named_parameters` は `TTorchLoadCallback` を再利用）。宣言しないもの：
   TorchScript のオブジェクト（タグ 14：`ati_object_*`、`ati_clone`、`atm_create_class_`、
   テスト `jit6`/`jit_double_free`、`foo6.pt`/`foo7.pt` — 参照カウント付きハンドルが要る
   ので後のマイルストーン）と `atm_fuser_cuda_*`（CUDA）。`TIValue`：タグの値 = `ati_tag`
   の 0〜13 と入力専用の `ivStringList`/`ivDevice`。静的コンストラクタ、`TTensor`/`Int64`/
   `Double`/`Boolean`/`String` からの暗黙の `:=`、`As*`/`Item`/`Key`/`Value` アクセサ。
   `ToC`（呼び出し側が解放）／`FromC`（消費する）は nil 初期化した子のスロットと
   `finally` での後始末。`GenericDict` は平坦なキー／値の `Items`。`TJitModule`：
   `Create(file[, device])`、`CreateFromStream`、`CreateByTracing`、`CreateEmpty` +
   `RegisterParameter` + `Define`（フォークの追加 `atm_create`/`atm_register_parameter`/
   `atm_define`：`Module::define`、`register_parameter`、`training` 属性の上の1行関数3つ）、
   `Forward`/`Method`（テンソル）、`ForwardIV`/`MethodIV`（ivalue）、`SetEval`/`SetTrain`、
   `ToDevice(device, kind)`、`Save`、`NamedParameters`（4.6.3 のトランポリンを `TchJit` に
   複製）、`Jit*` のグローバル関数。`TTrainableJitModule = class(TModule)`：
   `atm_load_on_device(P.Device)`、`P.Add('.'→'_' した名前, T, T.RequiresGrad)`、ハンドルは
   共有。`Forward(X, Train)` は `Train` が変わったときに `atm_train`/`atm_eval` で適用する
   （明示した差異：tch-rs は無視する）。**テスト資材：** tch-rs の `tests/foo*.pt` を
   `tests/jit/models/` に取り込む（ライセンスファイルを隣に）**かつ**、パラメータ付き／
   `self.training` のモジュールはテスト内で TorchScript のソースから作る。トレースだけでは
   パラメータを作れない。手計算できる数値を持つ13のテストグループは 4.6.8 に。
2. **`FromArray`（DESIGN 4.3）。** `Single` のオーバーロードを外す。`Double`/`Int64`/
   `Byte`/`Boolean` のオーバーロードは末尾に自然な既定値の `AKind` を持つ（`kFloat`/
   `kInt64`/`kUInt8`/`kBool`）。`Single` のバッファには `FromSingleArray`
   （`FromInt32Array` と同様）。実測：これですべての実数リテラルが `Double` に解決する
   （既定で float32 のテンソル。PyTorch の規則で、Python と同じビット列）。`[1, 2]` は
   `Int64` のまま。リテラル内の `Single` 変数は変換されるが、`array of Single` の変数には
   名前付きの関数が要る。
3. **ローカルセッション3の確認。** lr 1e-10：`‖g‖² = 80.955² + 8.974² = 6634.2`、
   `49 · 1e-10 · 6634.2 = 3.25e-5`、確認。実測の `-3.24e-5` は 23 付近の loss の float32 の
   分解能（1.9e-6）の範囲内。4.6.7 テスト7は tch-rs の *add* / *add_partial* の構成
   （`t123` は1回）に訂正。`TVarPath.Group`、`FullName` の検査（`TryGet` は `.`/`|` で
   例外）、`AddEntry` の nil 検査を承認。`CreateRaw` は private のまま。警告 3018 は
   `{$push}{$warn 3018 off} … {$pop}` で消す（実測：`strict private` でも警告は出る）。
   追加テストはすべて 4.6.7 に採用。1つ追加予定
   （`save_and_load_partial_var_store_incomplete_file` の半精度の後半）。

**クラウドでの計測（FPC 3.2.2。スクラッチパッドのみ）：**
- M4：オープン配列リテラルと `FromArray` のオーバーロードの組（表は 4.3）：実数リテラルが
  `Single` になるのは binary32 で正確に表せるときだけ — `[1.0, 2.0]`、`[2.5]`、
  `[1.0, 0.1]` は `Single` を選び、`[0.9, -1.9]`、`[3.14]`、`[0.1]` は曖昧。`Single` の
  オーバーロードがなければすべて `Double` に解決。`AKind` の既定値は解決を変えない。
- M5：明示したタグ値を持つ `TIValue` record（`TIValueTag(12) = ivGenericList`）、入れ子の
  タプル、`Int64`/`Double`/`Boolean`/`String`/record からの暗黙の `:=`。
  **`[T, 42, 1.5]` はこれらの演算子を通して `array of TIValue` 引数に要素ごとに
  変換される**。`SizeOf(Boolean) = 1`。
- M6：FPC は `private` と `strict private` のコンストラクタに `(3018) Constructor should
  be public` と警告する。`{$push}{$warn 3018 off} … {$pop}` で消える。
- M7：フォークの `torch_api.{h,cpp}` の `atm_*`/`ati_*` の領域は上流 4227b89b と同一
  （`diff`）。`foo*.pt` は zip 形式バージョン3で、その TorchScript（`code/__torch__.py`）
  は 4.6.8 に要約。
- M8：4.6.8 のインターフェース全体（`TIValue`、`TJitModule`、`TTrainableJitModule`、
  `Jit*` 関数）を、本体をスタブにした使い捨てのユニットとして本物の `Tch`/`TchNN` に対して
  コンパイル：エラーなし、プラグマで囲んだ `CreateRaw` からの警告なし、テストの呼び出し
  形 `ForwardIV([X, 42, 1.5])`、`Tuple([2.0, 3.0, 4])`、`GenericDict(['bar', X, 'foo',
  X])`、`StringList([...])` が解決する。

**新しい仮説（本物の libtorch が必要）：**
- **(H19)** 取り込んだ `tests/jit/models/foo*.pt` は libtorch 2.13 で読み込め、
  `jit_tests.rs` の数値（4.6.8 の項目1〜7）を与える。
- **(H20)** フォークの追加：`Module::define` は 4.6.8 の2つの TorchScript ソース
  （`torch.matmul`、`torch.t`、`self.training`）をコンパイルする。登録した `training`
  属性で `self.training` と `atm_train`/`atm_eval` が働く。`named_parameters` は登録した
  パラメータをその `requires_grad` とともに返す。
- **(H21)** パラメータの共有：var-store への1回の SGD ステップの後に
  `TTrainableJitModule.Forward(X) = 3.0` で、`Save` + 読み戻しで学習済みの値が保たれる。
- **(H22)** `ati_generic_dict` の `Dict[str, Tensor]` への特化が `foo8.pt` の `generate`
  に受け入れられる。
- **(H23)** 例外を投げるトレースのクロージャの後の `atm_end_tracing(M, name, nil, 0)` で、
  次の `CreateByTracing` が使える。
- **(H24)** `GenericDict` の往復は挿入順を保つ（`c10::Dict` は順序を保つ）。
- **(H25)** 新しいプロセスで `JitGetProfilingMode` は True。

**ローカルセッション、この順に（機械的）：**
1. `FromArray` の改訂（DESIGN 4.3）：`src/Tch.pas` — `array of Single` のオーバーロードを
   外し、他の4つに `AKind` の既定値を足し（`kFloat` の高速経路：Pascal 側で `Single` の
   バッファに変換。それ以外は `totype`）、`FromSingleArray` を足す。テスト：
   `tests/api/api_test.pas`（`TDoubleArray.Create(0.25, 0.5)` は今後 `kFloat` — `kDouble`
   を渡すか期待値を変える。`FromArray([0.9, -1.9])`、`FromArray([1, 2], kFloat)`、
   `FromSingleArray` を追加）、`tests/nn/nn_test.pas`（`Double(…)` のキャストは外せる）、
   `tests/compiler/h2_fromarray.pas`（新しい組。`[0.9, -1.9]`、`[3.14]`、`[1.0, 0.1]` →
   Double）と、`Single` と `Double` の両方があると `[0.9, -1.9]` が曖昧になることを示す
   新しい `xfail_h2_single.pas`。`tests/compiler/run.sh`、`tests/api`、`tests/nn`、
   `tests/mock/run.sh` を実行。
2. 4.6.7 の積み残し：`TOptimizer.CreateRaw` を `{$push}{$warn 3018 off}` / `{$pop}` で
   囲む。4.6.7 項目 7d の半精度の `LoadPartial` の確認を `TestSaveLoad` に足す。
3. フォークのパッチ：`csrc/libtch/torch_api.{h,cpp}` に `atm_create`、
   `atm_register_parameter`、`atm_define`（コードは 4.6.8）。libtch を再ビルド。
   `tests/abi`、`tests/mock/run.sh`（モックのスタブは手順7で。テストプログラムは使う
   ものしか参照しない）。
4. tch-rs 4227b89b（`tests/`）から `tests/jit/models/foo.pt` … `foo8.pt` と `LICENSE-MIT`、
   `LICENSE-APACHE` を取り込む。コミットメッセージに出典のファイルと
   `tests/create_jit_models.py` を書く。
5. `src/TchSys.pas`：4.6.8 の JIT の型と external。
6. `src/TchJit.pas`。各部分が単独でコンパイルできるようこの順に：`TIValue`（record、
   コンストラクタ、演算子、アクセサ）→ `ToC`/`FromC` → `TJitModule`（load/free/forward/
   method/eval/train/to/save）→ `NamedParameters` のトランポリン → `CreateByTracing` →
   `CreateEmpty` / `RegisterParameter` / `Define` → `Jit*` のグローバル →
   `TTrainableJitModule`。
7. モック：`tests/mock/mock_tch.c` に `atm_*`/`ati_*` のスタブ（4.6.8「モック」）、
   `tests/mock/jit_mock_test.pas`、`run.sh` に追加。
8. 4.6.8 の13グループによる `tests/jit/jit_test.pas`。出てきた順に H19〜H25 を確認し、
   結果を DESIGN.md の各 (Hn) の横に記録する。
9. DESIGN/HANDOFF（英語・日本語）を更新。その後、必要なら TorchScript のオブジェクト
   （タグ 14）、マイルストーン3のレイヤー（4.6.5 の先送り一覧）、CUDA/ROCm を
   設計セッションに戻す。

## ローカルセッション3（2026-10-10）：マイルストーン2を実装

下のクラウドセッション4の一覧の手順1〜5を完了し、すべてのテストが通ります。

1. **フォークのパッチ** `ato_set_momentum_group`：`} if` → `} else if`
   （`csrc/libtch/torch_api.cpp`）。libtch を再ビルドし、`tests/abi` と
   `tests/mock/run.sh` が通る。*パッチ前*のソースのビルドを H16 の計測専用に
   `build/libtch-unpatched/`（git 管理外）に残してある。
2. **L2/L3 の追加：** `src/TchSys.pas` に `TTorchLoadCallback`、`at_save_multi`、
   `at_load_callback_with_device`、`at_loadz_callback_with_device`、`PTorchOptimizer`
   と 14 個の `ato_*`（13 個 + `ato_free`）。`src/Tch.pas` に `TTensor.CopyFrom`
   （`at_copy_`）。
3. **`src/TchNN.pas`** は DESIGN 4.6 のスケッチどおり。小さな追加：`TVarPath.Group`。
   `TVarPath.FullName` が名前を検査するので `TryGet` も `.`/`|` を拒否する。`AddEntry` は
   nil テンソルで例外。private の `TOptimizer.CreateRaw` に FPC が「Constructor should be
   public」と警告する（無害。設計どおり private のまま）。参照に使った tch-rs のソースは
   4227b89b の `src/nn` と `tests` のスパースクローンで、セッションのスクラッチパッドに
   置いた（コミットしていない）。
4. **`tests/nn/nn_test.pas`**（ビルド方法はヘッダ）：4.6.7 の10グループすべてが通る。
   加えて AdamW の1ステップ目（`[0.899, -1.898]`、減衰が更新より先）、存在しないファイルの
   Load（`ETorchNativeError`）、`CopyFrom`、部分木への `TVarPath.SetKind`、`TSequential` の
   MLP、`AddFn` へのメソッドポインタも確認。**H13〜H18 はすべて解決**（結果は DESIGN.md の
   各仮説の横）。4.6.7 の訂正が2点：(a) `LoadPartial → ['a.b.t2']` は tch-rs の
   `add_partial` の構成（`t123` は1回）が必要。完全な構成だと `a.b.ccc.t123__3` も欠ける。
   (b) lr 1e-10 の判定は tch-rs では符号付き。loss は実際に 3.25e-5（手計算）／3.24e-5
   （実測）下がるので、絶対値 1e-5 の上限は成り立たない。`build/nn_test h16` は H16 だけを
   実行する（`LD_LIBRARY_PATH=build/libtch-unpatched` 用）。
5. **モック：** `tests/mock/mock_tch.c` に 4.6.7 のスタブ（オプティマイザのカウンタ、
   保存した名前、`err_after_first` / `nil_tensor` というファイル名で挙動が変わる load
   コールバック、`set_requires_grad`、`set_data`、`fill_`、`uniform_`、`copy_`、勾配
   モード）。`tests/mock/nn_mock_test.pas` を追加し、`run.sh` は両方を実行する。モックの
   インプレース op は2つ目のハンドルではなくコピーを返す（配線の確認のみ）。

- 回帰：`tests/api`、`tests/generated`、`tests/ownership`、`tests/compiler/run.sh` が通る。
- **新しい計測、(H2) 追記**（DESIGN 4.3）：オープン配列リテラルの要素型は先頭の要素で
  決まる。`[0.9, -1.9]`、`[3.14]`、`[1e-3]` は `FromArray` の `Single` と `Double` の
  オーバーロードの間で**曖昧**。`[1.0, 0.1]` は黙って `Single` を選ぶ。テストでの回避策は
  `[Double(0.9), -1.9]`。

## クラウドへの引き継ぎ（2026-10-10、ローカルセッション3の終了時点）— 上のクラウドセッション5で回答済み

**設計のみ**（CLAUDE.md）。本物の libtorch で確認が必要なことは H19 以降として記す。

1. **JIT マイルストーン（`TchJit`）。** 形は DESIGN 4.6 で決まっている（`TJitModule`
   クラス、`TIValue` タグ付きレコード、H10 解決済み、`TTrainableJitModule =
   class(TModule)`）。まだ機械的に進められない点：`torch_api.h`（201〜264 行）の
   `atm_*` / `ati_*` のどれをどう宣言するか（`atm_forward` と `ivalue` を返す
   `atm_forward_`、`atm_method[_]`、`atm_named_parameters` のコールバック — 4.6.3 と同じ
   トランポリン方式か）。`TIValue` ↔ `ivalue` の変換と所有権（`ati_free`、タプル、汎用
   リスト／辞書、マイルストーン3でどのタグまで）。`atm_to`/`atm_eval`/`atm_train`。
   `TTrainableJitModule` がパラメータを `TVarPath` に登録する方法（名前、学習対象フラグ）。
   **Python なしのテスト資材：** tch-rs のテストは `create_jit_models.py` で作った
   `tests/foo*.pt` を読む。それらを取り込む（ライセンスは tch-rs と同じ MIT/Apache）か、
   Pascal から `atm_create_for_tracing` / `atm_end_tracing` / `atm_save` でモジュールを
   作るか。4.6.7 のように手で確かめられる数値で。
2. **`FromArray` のリテラル（H2 追記）。** 先頭要素の規則を文書化するだけか、名前付きの
   `FromSingles`/`FromDoubles` を足すか、オーバーロードの組を変えるか（例：`Single` を
   外して実数リテラルを `Double` にする。PyTorch の既定 float32 とはずれる）を決める。
   `[1.0, 0.1]` が黙って丸められる点も考慮。
3. **ローカルセッション3で設計どおりにならなかった点の確認**（詳細は上の
   「ローカルセッション3」と DESIGN_jp.md の各項目の横）。それぞれ承認または修正し、
   DESIGN 4.6 / 4.6.7 をコードとテストの現状に合わせる：
   - 4.6.7 テスト3：lr 1e-10 の上限。DESIGN には tch-rs の判定は符号付きで、loss は
     3.25e-5（手計算）／3.24e-5（実測）下がると書いた。計算を確認すること。
   - 4.6.7 テスト7：`LoadPartial → ['a.b.t2']` は tch-rs の `add_partial` の構成
     （`t123` は1回）が必要。完全な構成だと `a.b.ccc.t123__3` も欠ける。DESIGN 4.6.7 は
     まだ直していない。
   - スケッチにない追加：`TVarPath.Group`。`TVarPath.FullName` が名前を検査する
     （`TryGet` も `.`/`|` を拒否する）。`AddEntry` は nil テンソルで例外。
   - private の `TOptimizer.CreateRaw` への FPC の警告「Constructor should be public」：
     private のままでよいか、別の生成パターンにするか。
   - 4.6.7 にないテストの追加（AdamW の1ステップ目 `[0.899, -1.898]`、存在しない
     ファイルの Load、`CopyFrom`、部分木の `SetKind`、MLP、`AddFn` へのメソッド
     ポインタ）：4.6.7 に取り込むか、テストだけに置くか。
4. **任意、後で：** マイルストーン3のレイヤー（DESIGN 4.6.5 の先送り一覧）、CUDA/ROCm
   （H13 の実際のデバイス移動は未確認。ここまで CPU のみ）。

最後にいつもどおり DESIGN/HANDOFF（英日）を更新すること。

## クラウドセッション4（2026-10-10）：DESIGN 4.6 を確定 — 上のローカルセッション3で実装済み

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

**ローカルセッション、この順に（機械的）：**（手順1〜6はローカルセッション3で完了）
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
  パッチだけ（ローカルセッション2でコンパイル・テスト済み）。c10 の警告ハンドラーは
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
  手書きの結果用 record を足す価値があるか。(H8) が実際に問題になるなら `var`。（その後
  H8 は解決：`out` のまま、DESIGN_jp.md の 4.4）
- **デフォルト引数：決定済み**（クラウドセッション3、DESIGN_jp.md の 4.4）：479 op に
  デフォルトが付く。out 引数のある op には付かない（107 op には付いたはず）。
  ローカルセッション2：`zeros`、`ones`、`empty`、`rand`、`randn` のオーバーロードを
  追加。`objfpc` では `overload` 指示子は不要（実測）だが、tchgen は意図した
  オーバーロードの目印に `overload;` を使う（DESIGN_jp.md の 4.4）。未決：out 引数の op に `topk(k)` 型のデフォルト。
- **実数リテラルの `FromArray`：決定**（クラウドセッション5、DESIGN 4.3 の H2 追記）。
  要素型は先頭の要素で決まり、実数リテラルが `Single` になるのは正確に表せるときだけ
  なので、`[0.9, …]` は曖昧で `[1.0, 0.1]` は `Single` に丸められていた。`Single` の
  オーバーロードを外し、他は自然な既定値の `AKind` を取り（`Double` には `kFloat`）、
  `Single` のバッファは `FromSingleArray` で。ローカルセッション4で実装済み。
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
- **ROCm（保留。まず CPU）— ローカルセッション8で実施**（DESIGN_jp.md の 4.8「結果」。
  以下は当初の計画）。手順2〜7は CPU 版 libtorch で進める。手元の GPU は
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
