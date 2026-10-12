# tch-fpc 設計

（[DESIGN.md](DESIGN.md) の日本語訳です。内容が食い違う場合は英語版が正です。）

決定事項とその理由をまとめます。未解決の論点と次の作業は
[HANDOFF_jp.md](HANDOFF_jp.md) にあります。FPC 3.2.2 の挙動のうち、まだビルドして
確かめていないものには **(H*n*)** を付け、HANDOFF の検証手順に載せています。
確認が済むまでは仮説として扱ってください。

以下はすべて tch-rs のコミット
`4227b89b72059b0651ff83a38637693574e0a2d6`（2026-08-23、crate 0.26.0）を基準にしています。
このコミットは **libtorch 2.13.0** 向けで、
`third_party/pytorch/Declarations-v2.13.0.yaml` からコードを生成しています。

## 0. 全体像

```
 ユーザーコード (Pascal)
      |
 [L3] Tch           TTensor（管理型 record）, TDevice, TKind, TScalar,
      |             ETorchError, 生成された op メソッド, 勾配モードの補助
      |  TchNN      TVarStore / TVarPath / TInit, TModule と各レイヤー,
      |             TSequential, TOptimizer
      |  TchJit     TJitModule, TIValue
 [L2] TchSys        生の `external` 宣言。C ABI と 1対1
      |             （手書きの TchSys + 生成の TchSysGenerated）
 ---- C ABI ---------------------------------------------------------------
 [L1] libtch.so     tch-rs の torch-sys/libtch をフォークした C ラッパー。
      |             libtorch に対して g++ でコンパイル
 [L0] libtorch      libtorch_cpu.so / libtorch_cuda.so / libc10.so
```

FPC から C++ は一切見えません。Pascal と C++ の間の約束は、`libtch` が公開する
素の C ABI だけです。

## 1. C ラッパー：tch-rs の libtch をフォークする（決定）

検討した選択肢：

| 選択肢 | 判定 |
|---|---|
| tch-rs の `torch-sys/libtch` をそのまま使う（git submodule など） | 不採用 |
| **このリポジトリにフォークし、ABI と関数名は維持する** | **採用** |
| C ラッパーを自作する | 不採用 |

そのまま使わない理由：
- `torch_api.cpp` が `tch_write_stream_*` / `tch_read_stream_*` を呼んでいますが、
  これらは *Rust 側で実装* されています（`torch-sys/src/lib.rs`）。Rust なしで
  libtch をリンクすると未解決シンボルが残ります。Pascal 側から同じ名前を
  エクスポートすることはできますが、ビルドが tch-rs の内部実装に縛られます。
- `torch_python.*`（pybind）や stb の画像ヘッダーも付いてきますが、当面どちらも不要です。

自作しない理由：
- 価値があるのは生成部分です。`torch_api_generated.{h,cpp}` は約 1.9 万行で、
  2,657 個の op をカバーしています。PyTorch の `Declarations.yaml` から
  `gen/gen.ml` が生成しており、何年分もの特例処理（除外関数、
  `no_tensor_options`、省略可能なスカラーの扱いなど）が入っています。
  作り直しても得るものがありません。
- ABI はすでに言語に依存しない形です。不透明ポインタ、`int64_t`、`double`、
  `int`、ポインタと長さの組、スレッドローカルな文字列によるエラー通知だけで
  できています。

フォークのルール：
- `csrc/libtch/` に `torch_api.h`、`torch_api.cpp`、`torch_api_generated.h`、
  `torch_api_generated.cpp` を、`gen/` に `gen/gen.ml` を取り込みます。
  tch-rs の `LICENSE-MIT` と `LICENSE-APACHE` も横に置きます。
  コミットメッセージには取り込み元のファイル名を書きます（CLAUDE.md 参照）。
  `third_party/pytorch/Declarations-v2.13.0.yaml`（再生成のときだけ必要）は
  同じコミットの tch-rs リポジトリにあります。大きいので取り込みません。
- **tch-rs の C の関数名と呼び出しの形は変えません**（`at_*`、`atg_*`、`ats_*`、
  `ato_*`、`atm_*`、`ati_*`）。上流との差分を小さく保てるので、libtorch の
  バージョンを上げる作業は「tch-rs から取り込み直し、こちらのパッチを当て直す」で
  済みます。
- フォークから削除するもの：ストリーム関数（`*_to_stream`、`*_from_stream*`、
  `tch_*_stream_*`）、`torch_python.*`、画像関数（`at_load_image*`、
  `at_save_image`、`at_resize_image`）と stb ヘッダー。必要になれば Pascal で
  扱いやすい形で戻せます（HANDOFF 参照）。
- `PROTECT` を強化する：上流は `const std::exception&` しか捕まえません。
  固定メッセージを入れる `catch (...)` 節を追加し、C++ のどんな throw も
  Pascal のフレームまで巻き戻らないようにします（巻き戻るとプロセスが異常終了します）。
- `torch_api_generated.h` は無条件に `extern "C" {` を開いています（tch-rs では
  C++ としてしかコンパイルされないため）。HANDOFF の C での動作確認でこのヘッダーを
  include できるよう、`#ifdef __cplusplus` で囲みます。どちらのパッチも数行で、
  当て直すのは簡単です。
- 独自に追加する関数には別の接頭辞 `tchf_`（tch-fpc）を付け、こちらで足したものだと
  一目で分かるようにします。最初の1つは `void tchf_free(void *)` です。libtch が
  `strdup`/`malloc` で確保した文字列や配列を解放します。これで Pascal 側が
  どの C ランタイムの `free` を使うべきかを気にせずに済みます（Windows で効きます）。
- libtch は **共有ライブラリ**（`libtch.so` / `tch.dll`）としてビルドします。
  C++ を FPC のプログラムに静的リンクすると、libstdc++ や libtorch の静的初期化まで
  FPC のリンカー呼び出しに持ち込むことになり、割に合いません。

## 2. Pascal 側が前提にする C ABI の事実

tch-rs を読んで得た内容です（CLAUDE.md に従い、**ビルドして確認すること**）。

- ハンドル：`tensor`、`scalar`、`optimizer`、`module`、`ivalue` は、ヒープに確保した
  C++ オブジェクト（`new torch::Tensor(...)`）への不透明ポインタです。どれも
  `at_free` / `ats_free` / `ato_free` / `atm_free` / `ati_free` でちょうど1回
  解放する必要があります。`at_free` はただの `delete` で、失敗せず、
  エラーの格納場所にも触れません。
- エラー：すべての入口は本体を `PROTECT(...)` で包んでいます。これは
  `std::exception`（`c10::Error` を含む）を捕まえ、`strdup(e.what())` を
  `thread_local char *torch_last_err` に入れます。
  `get_and_reset_last_err()` はそれを返してクリアし、文字列の所有権は呼び出し側に移ります。
  ハンドルを返す手書きの関数はエラー時に `nullptr` を、`int` のフラグを返す関数は
  `-1` を返します。
- **省略不可の `tensor` 引数は null チェックなしで参照外しされます**
  （生成された呼び出しの `*self`）。`TensorOption` ではなく `Tensor` が要る場所に
  `nullptr` を渡すと、捕まえられるエラーではなくセグメンテーション違反になります。
  L3 は呼び出し前に検証する必要があります（4.2）。
- 生成された op（`atg_*`、全 2,657 個）の形：
  - 結果のテンソル数が固定：`void atg_X(tensor *out__, ...)`。呼び出し側が
    N 個分の配列を渡し、C 側は op が成功した *あとで* `out__[0..N-1]` を埋めます。
    2個以上（最大7個）のテンソルを返す op は 246 個です。
  - 結果の数が可変（`split` など）：`tensor *atg_X(...)` が `malloc` で確保した
    `nullptr` 終端の配列を返します。呼び出し側は各要素の所有権を受け取ったあと、
    配列だけを解放します（テンソルは解放しない）。該当する op は 33 個です。
  - スカラーの結果：`bool`（`int` として）/ `int64_t` / `double` を直接返します。
    44 個。何も返さない op が 21 個あります。
- データのメモリ所有権：
  - `at_tensor_of_data` は **コピーします**（`torch::zeros` + `memcpy`）。
    呼び出しから戻った時点で Pascal 側のバッファを解放して構いません。
  - `at_tensor_of_blob` は **コピーしません**。バッファはテンソルとその全ビューより
    長く生きている必要があります。最初のマイルストーンでは公開しません。
  - `at_copy_data` はデータを外へコピーします。必要なら CPU への移動と
    `contiguous()` を経由します。
  - `at_to_string`、`ats_to_string`、`ati_to_string` の文字列とエラー文字列は
    `strdup` されたもの、可変個の結果配列は `malloc` されたものです。
    すべて `tchf_free` で解放します。
- コールバック式のローダー（`at_load_callback*`、`at_loadz_callback*`、
  `atm_named_parameters`）は、呼び出したスレッド上で `PROTECT` の内側から
  `f(data, name, new torch::Tensor(...))` を呼びます。受け取ったテンソルは
  **呼ばれた側が所有** します。名前は借り物で、呼び出しの間だけ有効です。
- 引数のエンコード：

  | gen.ml の型 | C の引数 | 意味 |
  |---|---|---|
  | Tensor | `tensor x` | 非 null |
  | TensorOption | `tensor x` | `nullptr` = なし |
  | Scalar | `scalar x` | `ats_int` / `ats_float` で作る |
  | Bool | `int x` | 0/1 |
  | Int64 / Double | `int64_t` / `double` | |
  | Int64Option / DoubleOption | `T x_v, uint8_t x_null` | `x_null<>0` = なし |
  | IntList | `int64_t *x_data, int x_len` | |
  | IntListOption | 同上 | `x_data == nullptr` = なし |
  | DoubleList | `double *x_data, int x_len` | |
  | TensorList / TensorOptList | `tensor *x_data, int x_len` | opt の方は null 要素 = なし |
  | String | `char *x_ptr, int x_len` | NUL 終端ではない |
  | TensorOptions | `int x_kind, int x_device` | dtype + device |
  | ScalarType | `int` | `c10::ScalarType` の値 |
  | ScalarTypeOption | `int` | `<0` = なし |
  | Device | `int` | `-1` CPU、`-2` MPS、`-3` Vulkan、`>=0` CUDA の番号 |
  | Layout / LayoutOption | `int8_t` | `-1` = なし（2.13 では 7 op：`to_sparse`、`_to_sparse`、`_spdiags` など） |

- 手書きの `torch_api.h` では、フラグに `int` と C++ の `bool` が混在しています
  （例：`at_autocast_is_enabled` は `bool` を返す）。Pascal 側は1つずつ正確に
  宣言する必要があります。`bool` には `Boolean`（1バイト。`LongBool` は不可）、
  `int` には `cint`、`uint8_t` には `Byte`、`int8_t` には `ShortInt`、
  `size_t` には `csize_t`。

重要な帰結：**C ヘッダーだけでは良い Pascal API は生成できません。**
`Tensor` と `TensorOption`、`Bool`・`Int`・`ScalarType`・`Device` は、どれも
同じ C の型に潰れてしまいます。Pascal の生成器には gen.ml が持っている
より詳しい型情報が必要です（5 章）。

## 3. Pascal の L2 層：TchSys（生の宣言）

- ユニット `TchSys`（`torch_api.h` の手書き部分）と `TchSysGenerated`
  （すべての `atg_*`）。中身は `external` 宣言だけで、ロジックは持ちません。
- 型：`type PTorchTensor = type Pointer;` など（別個のポインタ型にして、
  テンソルのハンドルが要る場所にスカラーのハンドルを渡せないようにする）。
- 規約：すべての宣言に `cdecl`（x86-64 の Linux/Windows では意味を持たず、
  i386 で正しくなる）、`{$packrecords c}`、`cint` などは `ctypes` から。
- ライブラリ名は定数1つにまとめます：`const LibTch = 'tch';` と
  `external LibTch name 'atg_add'`。リンク時の静的な `external` 束縛にし、
  `LoadLibrary` で 2,657 個の関数ポインタを取る方式は使いません。
- 名前は C と完全に同じにします（`atg_add`、`at_free`）。引数名もそのままで、
  予約語（`out`、`end`、`to`）は `&` でエスケープし（`&out`）、`result` だけは
  `result_` にします（4.4 と同じルール）。引数名は ABI に影響しません。

スケッチ：

```pascal
unit TchSys;
{$mode objfpc}{$H+}{$packrecords c}
interface
uses ctypes;
const
  LibTch = 'tch';
type
  PTorchTensor = type Pointer;  PPTorchTensor = ^PTorchTensor;
  PTorchScalar = type Pointer;
  TTorchLoadCallback = procedure(data: Pointer; name: PChar; t: PTorchTensor); cdecl;

function  get_and_reset_last_err: PChar; cdecl; external LibTch;
procedure tchf_free(p: Pointer); cdecl; external LibTch;

function  at_new_tensor: PTorchTensor; cdecl; external LibTch;
function  at_tensor_of_data(vs: Pointer; dims: PInt64; ndims: csize_t;
            element_size_in_bytes: csize_t; type_: cint): PTorchTensor;
            cdecl; external LibTch;
procedure at_copy_data(t: PTorchTensor; vs: Pointer; numel: csize_t;
            element_size_in_bytes: csize_t); cdecl; external LibTch;
function  at_shallow_clone(t: PTorchTensor): PTorchTensor; cdecl; external LibTch;
function  at_defined(t: PTorchTensor): cint; cdecl; external LibTch;
function  at_dim(t: PTorchTensor): csize_t; cdecl; external LibTch;
procedure at_shape(t: PTorchTensor; dims: PInt64); cdecl; external LibTch;
function  at_scalar_type(t: PTorchTensor): cint; cdecl; external LibTch;
function  at_device(t: PTorchTensor): cint; cdecl; external LibTch;
function  at_to_string(t: PTorchTensor; line_size: cint): PChar; cdecl; external LibTch;
function  at_autocast_is_enabled: Boolean; cdecl; external LibTch;  // C++ の bool
procedure at_load_callback(filename: PChar; data: Pointer;
            f: TTorchLoadCallback); cdecl; external LibTch;
procedure at_free(t: PTorchTensor); cdecl; external LibTch;

function  ats_int(v: Int64): PTorchScalar; cdecl; external LibTch;
function  ats_float(v: Double): PTorchScalar; cdecl; external LibTch;
procedure ats_free(s: PTorchScalar); cdecl; external LibTch;
```

```pascal
unit TchSysGenerated;  // GENERATED, DO NOT EDIT
...
procedure atg_add(out__: PPTorchTensor; self, other: PTorchTensor); cdecl; external LibTch;
procedure atg_max_dim(out__: PPTorchTensor; self: PTorchTensor; dim: Int64;
            keepdim: cint); cdecl; external LibTch;
function  atg_split(self: PTorchTensor; split_size, dim: Int64): PPTorchTensor;
            cdecl; external LibTch;
procedure atg_zeros(out__: PPTorchTensor; size_data: PInt64; size_len: cint;
            options_kind, options_device: cint); cdecl; external LibTch;
procedure atg_narrow(out__: PPTorchTensor; self: PTorchTensor; dim, start,
            length: Int64); cdecl; external LibTch;
procedure atg_add_out(out__: PPTorchTensor; &out, self, other: PTorchTensor);
            cdecl; external LibTch;
```

## 4. Pascal の L3 層：Tch（オブジェクト指向）

### 4.1 テンソルの所有権：インターフェースを包む管理型 record（決定）

検討した選択肢：

| 選択肢 | 判定 |
|---|---|
| 素の `class` で明示的に `Free` | 不採用：一時値のたびに `try..finally` が必要で、`(a + b) * c` では途中の値がリークする。 |
| COM 風の `ITensor` インターフェースを公開型にする | 不採用：FPC の演算子オーバーロードは record を中心に作られており、インターフェース型に二項演算子が書けるかどうかも不明。1つのインターフェースに 2,657 個の仮想メソッドを持たせるのは record の静的メソッドより重い。公開インターフェースは実装オブジェクトをクラス参照で持たせる誘惑を生み、参照カウントの典型的な落とし穴になる。 |
| 生ポインタを持つ record + 管理演算子（`Initialize/Finalize/Copy`） | 当面は不採用：3.2.0 以降でしか使えず、3.2.x では一時値や関数の戻り値まわりのバグ報告がある。テンソル1つにつきヒープ確保が1回減る。その確保が問題だと計測で分かったときだけ再検討する。 |
| 生成 op は record helper、中核は record | 不採用（4.5 参照）。 |
| **参照カウント付きの `ITensorHandle` を包む record** | **採用** |

```pascal
unit Tch;
{$mode objfpc}{$H+}{$modeswitch advancedrecords}
type
  ITensorHandle = interface
    function Raw: PTorchTensor;
  end;

  TTensorHandle = class(TInterfacedObject, ITensorHandle)
  private
    FRaw: PTorchTensor;
  public
    constructor Create(ARaw: PTorchTensor);  // 所有権を受け取る
    destructor Destroy; override;            // at_free(FRaw)。例外は投げない
    function Raw: PTorchTensor;
  end;

  TTensor = record
  private
    FHandle: ITensorHandle;   // nil = テンソルが何もない
  public
    // --- 手書きの API（Pascal 流の名前） ---
    class function FromRaw(ARaw: PTorchTensor): TTensor; static; // 所有権を受け取る
    function Raw: PTorchTensor; inline;      // 借り物。IsNil なら nil
    function IsNil: Boolean; inline;         // FHandle = nil
    function Defined: Boolean;               // IsNil でなく、かつ at_defined <> 0
    function Dim: Integer;
    function Shape: TInt64Array;
    function Numel: Int64;
    function Kind: TKind;
    function Device: TDevice;
    function ToString(LineWidth: Integer = 80): AnsiString;
    function ShallowClone: TTensor;          // 新しいハンドル、同じストレージ
    function Copy: TTensor;                  // 新しいストレージ（zeros_like + copy_）
    // データの出し入れ (H2)
    class function FromArray(const A: array of Single): TTensor; overload; static;
    class function FromArray(const A: array of Double): TTensor; overload; static;
    class function FromArray(const A: array of Int64): TTensor; overload; static;
    class function FromInt32Array(const A: array of Int32): TTensor; static;  // no Int32 overload (H2)
    class function FromArray(const A: array of Byte): TTensor; overload; static;
    class function FromArray(const A: array of Boolean): TTensor; overload; static;
    class function FromBytes(const Data; const Shape: array of Int64; AKind: TKind): TTensor; static;
    function ToSingleArray: TSingleArray;    // 外へコピー。型も変換する
    function ToDoubleArray: TDoubleArray;
    function ToInt64Array: TInt64Array;
    procedure CopyData(var Dest; ACount: Int64);  // own kind, no conversion
    function DoubleValue(const Index: array of Int64): Double;
    function Int64Value(const Index: array of Int64): Int64;
    // 自動微分
    procedure Backward(KeepGraph: Boolean = False; CreateGraph: Boolean = False);
    function RequiresGrad: Boolean;
    procedure ZeroGrad;                      // grad (generated) .detach_.zero_ if defined
    // 演算子（4.1「演算子」）
    class operator +(const A, B: TTensor): TTensor;
    class operator -(const A, B: TTensor): TTensor;
    class operator *(const A, B: TTensor): TTensor;    // 要素ごと (mul)
    class operator /(const A, B: TTensor): TTensor;
    class operator -(const A: TTensor): TTensor;       // neg
    class operator **(const A: TTensor; B: Double): TTensor;
    class operator +(const A: TTensor; B: Double): TTensor;  // Int64 版と左右逆の版も
    ...
    // --- 生成された op メソッド（4.4） ---
    {$I TchTensorGenerated_intf.inc}
  end;
```

この形にした理由：
- **解放のタイミングが決まる。** インターフェースの参照カウントにより、Pascal 側の
  最後の参照が消えた時点で C++ のテンソル（と GPU メモリ）が解放されます。
  tch-rs の `Drop` と同じです。
- **式が書ける。** `{$mode objfpc}{$modeswitch advancedrecords}` の record は
  `class operator` による演算子オーバーロードに対応しています。`(a + b) * c` の
  途中結果はコンパイラが管理する一時値になります。**(H1) 解決**（手順6、FPC 3.2.2）：
  一時値は文の終わりでは解放 *されません*。ルーチンの中の式の場所ごとに一時値の
  枠があり、枠が再利用される（次に評価される。たとえばループの次の周）か、
  ルーチンを抜けるまで最後の値を持ち続けます。したがってループで溜まることは
  ありません（10^6 回で、ハンドル数も RSS も一定）が、大きな途中結果はルーチンが
  戻るまで生き残ります。大きなテンソルでの対策：式を小さな関数に入れる、または
  名前付きの変数に分けて `X := Default(TTensor)` で解放する。
- **FPC 3.2.2 で堅牢。** インターフェースのフィールドを持つ record は、昔からある
  インターフェースの参照カウント機構でコンパイラが管理します。あらゆる Lazarus
  アプリケーションで使われている仕組みです。管理演算子は使いません（上の表）。
- **参照カウントがスレッドセーフ。** `TInterfacedObject` はアトミックな
  増減（interlocked increment/decrement）を使います。
- **「なし」が自然に表せる。** `Default(TTensor)`（ハンドルが nil）が、そのまま
  `TensorOption = nullptr` に対応します。また `TTensor` はポインタ1個分の
  大きさなので、受け渡しのコストはほぼありません。

意味論：
- `B := A` は同じ C++ の `torch::Tensor` オブジェクトを共有します。libtorch の
  テンソル自体が共有ストレージへの参照カウント付きハンドルなので、C++ や Python の
  ユーザーが期待するエイリアシングと同じです。`ShallowClone`（`at_shallow_clone`）は
  同じストレージへの独立したハンドルを返します。`Copy` はデータをコピーします。
- 「何もない」には2段階あります。`IsNil`（ハンドルがない。`Default(TTensor)`。
  生成器が `TensorOption` に `nullptr` として渡すもの）と、`Defined`（libtorch 自身の
  概念。`at_new_tensor` は未定義テンソルへの nil でないハンドルを返し、勾配がないときの
  `grad` はこれを返す）です。省略不可の `Tensor` 引数に nil のハンドルを渡すと、
  C 側が参照外しする前（2 章）に `ETorchError` を投げます。
- `Raw` は借り物のポインタです。どこかの `TTensor` がハンドルを持っている間だけ
  有効なので、保存しないこと。
- `TTensorHandle.Destroy` は `at_free` を呼ぶだけです。例外は投げず、エラーの
  格納場所も読みません（例外の巻き戻し中に走るデストラクタが2つ目の例外を
  投げてはいけないため）。
- インプレース op は PyTorch と同じく末尾にアンダースコアを付け（`add_`）、
  tch-rs と同様にチェーンできるよう同じテンソルを返します。戻り値を無視しても
  構いません（`{$X+}` が既定）。
- 早めに解放したいとき：`A := Default(TTensor);`。
- **`=` / `<>` はオーバーロードしない。** テンソルの等価性は曖昧です（ハンドルが
  同じか、要素ごとか、`allclose` か）。明示的に書かせる（`SameHandle`、生成された
  `equal`、`allclose`）ことで、気づかない驚きを避けます。結果として、FGL のコンテナ
  （`TFPGList<TTensor>`）は `=` を要求するためコンパイルできません。
  `TTensorArray`（動的配列）か、バイト単位で比較する
  `Generics.Collections.TList<TTensor>` を使います。HANDOFF の未解決の論点に記載。
- データの入力：`FromArray` は要素型でオーバーロードしてあるので、
  `TTensor.FromArray([1.0, 2.0])` と自然に書けます。**(H2)** 実数定数のオープン配列
  リテラルが `array of Single` と `array of Double` のどちらかに曖昧さなく決まるか
  （整数リテラルの `Int32` と `Int64` も同様）はテストが必要です。だめなら名前を
  分けます（`FromSingles`、`FromDoubles` など）。Pascal は戻り値の型で
  オーバーロードできないので、データの出力は型名付きにします（`ToDoubleArray`）。
  **(H2) 解決**（手順7）：実数リテラルは曖昧になりません。`Single` と `Double` の両方の
  オーバーロードがあると、`[1.0, 2.0]` は **`Single`** を選びます（Double 型の変数なら
  `Double`）。これは PyTorch の既定の float32 と一致します。整数リテラル（`[1, 2]`）は
  **`Int32` と `Int64` の両方のオーバーロードがあると曖昧になります**。`Int32` がなければ
  `Int64`（PyTorch の既定の int64）を選び、`Single, Double, Int64, Byte, Boolean` の組では
  試したすべてのリテラルが解決しました。したがって `FromArray` に `array of Int32` の
  オーバーロードは置きません。int32 のデータは名前付きの関数（例：`FromInt32Array`）か
  `FromBytes` を使います。
  **(H2) 追記**（ローカルセッション3、3.2.2 で計測）：オープン配列コンストラクタの要素型は
  **先頭**の要素で決まります。先頭が `Single` で正確に表せる値（`1.0`、`0.5`、整数）なら
  全体が `Single` になるので、`[1.0, 0.1]` は `Single` を選び `0.1` を黙って丸めます。
  先頭がそうでない値（`0.9`、`3.14`、`1e-3`）だと呼び出しが**曖昧**になります
  （`[0.9, 1.0]`、`[3.14]` はコンパイルできない）。回避策は先頭の要素に型を付けること
  （`[Double(0.9), -1.9]` は `Double`、`[Single(0.9)]` は `Single`）。HANDOFF の
  未解決の論点に記載。

演算子（手書き。生成された op に対応付ける）：

| Pascal | op | 備考 |
|---|---|---|
| `A + B`, `A - B`, `A * B`, `A / B` | `add`, `sub`, `mul`, `div` | `*` は **要素ごと**。行列積は `A.matmul(B)` |
| `A + d`, `A - d`, `A * d`, `A / d` | `add_scalar` など | `d: Double` と `d: Int64` を明示的なオーバーロードとして用意する **(H3)**。`TScalar` の暗黙変換は通さず、`T + 1` と `T + 1.5` がどちらもユーザー定義の変換なしで解決されるようにする |
| `d + A`, `d - A`, `d * A`, `d / A` | 左右逆 | `d - A` = `A.rsub_scalar(d)`（op 1回）。`d / A` は以前の `A.pow(-1) * d` ではなく `A.reciprocal * d`。ローカルセッション2で実測：Int64 テンソルに整数の指数 `-1` で `pow` すると失敗する（"Integers to negative integer powers are not allowed."）。`-1.0` なら通る。`reciprocal` は真の除算と同じく float に昇格し、`4 / Int64 テンソル` は kFloat |
| `-A` | `neg` | |
| `A ** d` | `pow_tensor_scalar` | FPC は `**` のオーバーロードを許す |

**(H3) 解決**（手順7）：`Double` と `Int64` のオーバーロードを両方の順で用意すると、
`T + 1` と `2 * T` は `Int64`、`T + 1.5` と `2.5 * T` は `Double` を選びます。
`Integer`/`Byte` の変数は `Int64`、`Single` の変数は `Double` を選びます。単項の `-` と
`**`（record の演算子）も使えます。

オーバーロードしないもの：`=`、`<>`、`<`、`>`（PyTorch では比較の結果は
テンソルなので、Boolean を返すのは嘘になる）、`@`、`in`。

コスト：テンソル1つにつきヒープ確保が2回（C++ の `torch::Tensor` と Pascal の
`TTensorHandle`）。当面は許容し、最適化は計測してから行います（逃げ道は
管理演算子版）。

### 4.2 エラー処理（決定）

- API は1種類だけにします（tch-rs の `f_add`/`add` のような、失敗を返す版と
  返さない版は分けない）。Pascal には例外があるので、すべての op は失敗時に例外を投げます。
- クラス：`ETorchError = class(Exception)` が基底で、Pascal 側の検証（nil ハンドル、
  形状の補助関数、未知の型）が投げるのはこれです。
  `ETorchNativeError = class(ETorchError)` は libtorch のエラー格納場所から来た
  メッセージを持ちます。`ETorchError` を捕まえれば両方捕まえられます。
- L3 は libtch を呼ぶ **たびに** 次を呼びます。

  ```pascal
  procedure TchCheck; inline;
  var P: PChar;
  begin
    P := get_and_reset_last_err;
    if P <> nil then TchRaiseNative(P);  // メッセージをコピーし、tchf_free(P) してから raise
  end;
  ```

  `TchRaiseNative` はインライン化しない別手続きにして、通常の経路を小さく保ちます。
  エラーの格納場所は C 側でスレッドローカルで、同じスレッドの直後に読むので、
  スレッド同士が干渉しません。生成されたコードが使う補助関数には `Tch` を付け、
  op の引数名（`self`、`&out`、`data`、`index` など）に隠されないようにします。
- **ネイティブのメッセージの形（決定、クラウドセッション3）。** libtorch の
  `c10::Error::what()` は、メッセージ、文脈の行、そして
  `"\nException raised from <関数> at <ファイル>:<行> (most recent call
  first):\n<バックトレース>"` の順です（`c10/util/Exception.cpp` の
  `Error::compute_what`）。`ETorchNativeError.Create` は最初の
  `#10'Exception raised from '` で分割し、`Message` はその前の本文、`Detail`
  プロパティに全文を残します。C 側は変えません（`what_without_backtrace()` を
  使うには `c10::Error` を捕まえる節が要り、しかもバックトレースが完全に失われる）。
  `tests/mock` のモックで確認。実際の文面は手順3で見たとおり。
- **libtorch の警告（決定、クラウドセッション3）。** `TORCH_WARN` は c10 の既定の
  `WarningHandler` を通して stderr に出ます。`c10/util/Exception.cpp`（2.13）で
  分かったこと2点：ハンドラーは **スレッドローカル**
  （`ThreadWarningHandler::warning_handler_`）なので、Pascal から入れたハンドラーは
  入れたスレッドにしか効かず、libtorch のワーカースレッドやほかの `TThread` には
  効かない。手順4で見た非推奨の警告は `at::autocast::is_enabled()`/`set_enabled()`
  が出すもので、フォークの `torch_api.cpp` ではこれを
  `is_autocast_enabled(at::kCUDA)` / `set_autocast_enabled(at::kCUDA, b)` に
  置き換えた（意味は同じ、警告なし。当時は **まだコンパイルしていない**。クラウドには
  libtorch がない。ローカルセッション2でコンパイル・テスト済み）。決定：最初のマイルストーンではこのパッチだけ。libtorch 自身の
  警告は C++ と同じく stderr のまま。プログラムが警告を捕まえる必要が出たら、
  `Warning::msg()` と `source_location()` を Pascal に渡すスレッドごとの
  `tch_set_warning_handler(callback, data)` を libtch に足すのが次の手。
- 呼び出し前に、省略不可の `Tensor` 引数（レシーバーを含む）すべてについて
  `IsNil` を確認し、nil なら
  `ETorchError('undefined tensor passed as <param> to <op>')` を投げます。
  コストがかかるのはこのチェックだけで、これがセグメンテーション違反の代わりになります。
- 固定 N 個の出力：スロットは `nil` で初期化し、エラー時は nil でないスロットを
  解放してから例外を投げます（実際には C が埋めるのは成功後だけですが、
  `new torch::Tensor` が途中で `bad_alloc` を投げる可能性はあります）。
- **Pascal の例外を C++ のフレームをまたいで伝わらせない。** libtch に渡すコールバック
  （`at_load_callback`、`atm_named_parameters`）は `cdecl` の中継関数にし、本体を
  `try..except` で包みます。ハンドラーは例外オブジェクトを
  （`AcquireExceptionObject` で）コンテキスト用の record に保存して、普通に戻ります。
  C の呼び出しから戻ったあとで、L3 がそれを投げ直します。中継関数は、捨てる場合でも
  渡されたテンソルの所有権を受け取ります（2 章）。
- **FPU の例外マスク。** x86 の FPC プログラムは、x87 の制御ワードと MXCSR の両方で
  浮動小数点例外（不正演算、ゼロ除算、オーバーフロー）が *有効* なまま動くので、
  `0/0` は `EInvalidOp` を投げます。libtorch、MKL、oneDNN は C の既定（すべて
  マスク済み）を前提にしていて、NaN や Inf を正当に生成します。Pascal の既定のままだと、
  ゼロ除算する CPU カーネルで SIGFPE が起きます。測定結果（手順5、FPC の既定は
  MXCSR=$1900）：呼び出し元のスレッドでは、FPC が C++ のフレームの中から
  `EZeroDivide` を投げます。libtorch の OpenMP ワーカーでは未処理の例外になり、
  プロセスは並列区間で **固まります**。決定：ユニット `Tch` は
  `initialization` で、libtorch が呼ばれうるより前に
  `SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide, exOverflow,
  exUnderflow, exPrecision])` を呼び、スレッド用に
  `procedure TchMaskFpuExceptions;` を公開します。
  - libtorch 自身のワーカープールは最初の並列 op のときに遅延生成され、新しい
    pthread は生成元スレッドの浮動小数点制御状態を引き継ぎます。したがって、
    `Tch` が最初の呼び出しより前にマスクする *からこそ*、プールのスレッドも
    マスクされた状態になります。マスクを遅延ではなく `initialization` で設定しなければ
    ならないのはこのためです。手順5で確認済み：マスクせずにプールを作り、あとから
    マスクしても、次の並列の 1/0 で固まります。
  - Pascal から作ったスレッド（`TThread`、`BeginThread`）は RTL が初期化します。
    **(H4) 解決**（FPC 3.2.2、Linux x86_64）：`SetExceptionMask` のあとに起動した
    `TThread` は MXCSR=$1F80 / x87CW=$137F（すべてマスク）で始まり、その中での
    小さな op も並列の op も Inf を返します。`TchMaskFpuExceptions` は RTL の外で
    作られたスレッド（別の C ライブラリが作るものなど）のために公開したままにしますが、
    `TThread` では不要です。`BeginThread` は個別には測っていません。
  - **(H5) 解決**：整数の `EDivByZero`（RTL のハンドラーを通る本物の SIGFPE）を
    捕まえたあともマスクは $1F80 のままで、テンソルの 1/0 も Inf を返します。
    すべてマスクしている間は Pascal コードで浮動小数点の `EZeroDivide` は起きないので、
    かけ直しは不要です。
  - トレードオフをはっきり書いておくと：`Tch` を使うプログラムでは、普通の Pascal の
    `x / 0.0` も例外ではなく `Inf` になります。これは C/C++/Python と同じ挙動で、
    数値計算ライブラリとしては許容範囲です。隠さず README に書きます。

### 4.3 小さな値型

```pascal
type
  { libtorch 2.13 の c10::ScalarType 全部（torch/headeronly/core/ScalarType.h）。
    クラウドセッション3で決定：0〜26 は手順4で検証済み、27〜45 も存在し CPU で
    作れる。kNone = -1 は省略可能な dtype 引数のための「dtype なし」で、
    TTensor.Kind は 0..45 の外では ETorchError を投げる。 }
  TKind = (kNone = -1,
           kUInt8 = 0, kInt8 = 1, kInt16 = 2, kInt = 3, kInt64 = 4,
           kHalf = 5, kFloat = 6, kDouble = 7, kComplexHalf = 8,
           kComplexFloat = 9, kComplexDouble = 10, kBool = 11,
           kQInt8 = 12, kQUInt8 = 13, kQInt32 = 14, kBFloat16 = 15,
           kQUInt4x2 = 16, kQUInt2x4 = 17, kBits1x8 = 18, kBits2x4 = 19,
           kBits4x2 = 20, kBits8 = 21, kBits16 = 22,
           kFloat8e5m2 = 23, kFloat8e4m3fn = 24, kFloat8e5m2fnuz = 25,
           kFloat8e4m3fnuz = 26, kUInt16 = 27, kUInt32 = 28, kUInt64 = 29,
           kUInt1 = 30, ..., kUInt7 = 36, kInt1 = 37, ..., kInt7 = 43,
           kFloat8e8m0fnu = 44, kFloat4e2m1fnx2 = 45);

  { c10::Layout。ltNone = -1 は省略可能な layout 引数のためだけ }
  TLayout = (ltNone = -1, ltStrided = 0, ltSparse, ltSparseCsr, ltMkldnn,
             ltSparseCsc, ltSparseBsr, ltSparseBsc, ltJagged);

  TDeviceType = (dtCpu, dtCuda, dtMps, dtVulkan);
  TDevice = record
    DeviceType: TDeviceType;
    Index: Integer;               // CUDA のみ
    function ToC: cint;           // -1 / >=0 / -2 / -3
    class function FromC(V: cint): TDevice; static;
    class function Cpu: TDevice; static;
    class function Cuda(AIndex: Integer = 0): TDevice; static;
    class function CudaIfAvailable: TDevice; static;
    function IsCuda: Boolean;
  end;

  TInt64Array  = array of Int64;
  TDoubleArray = array of Double;
  TSingleArray = array of Single;
  TTensorPtrArray = array of PTorchTensor;   // 生成した本体用：テンソルのリスト引数

  TScalar = record                // 呼び出しごとに一時的な ats_* ハンドルに変換する
    IsFloat: Boolean;
    I: Int64; F: Double;
    class operator :=(V: Int64): TScalar;
    class operator :=(V: Double): TScalar;
  end;

  TOptInt64  = record HasValue: Boolean; Value: Int64;  class operator :=(V: Int64): TOptInt64;  end;
  TOptDouble = record HasValue: Boolean; Value: Double; class operator :=(V: Double): TOptDouble; end;
  TOptIntList = record HasValue: Boolean; Value: TInt64Array; end;

  TTensor = record
  public type
    TTensorArray = array of TTensor;   // 入れ子：TTensor より前には宣言できない
  ...
  end;
  TTensorArray = TTensor.TTensorArray; // 利用側向けの別名

const
  NoInt64:   TOptInt64   = (HasValue: False; Value: 0);
  NoDouble:  TOptDouble  = (HasValue: False; Value: 0);
  NoIntList: TOptIntList = (HasValue: False; Value: nil);

function IntList(const A: array of Int64): TOptIntList;  // 値あり（空でもよい）
```

- `Default(TOptXxx)` も「なし」です。`NoXxx` 定数は呼び出し箇所で読みやすいだけです
  （`t.sum_dim_intlist(IntList([0]))`）。
- ジェネリクス（`TOpt<T>`）にせず手書きにします。ジェネリックな record の中の
  暗黙変換演算子 `:=` は 3.2.x でバグ報告がある分野で、しかも必要なのは3つだけです。
  実測（クラウドセッション3）：`t.argmax(1)`、`t.clamp_min_(0.5)`、`t.add_scalar(1)` は
  リテラルも変数もキャストなしで `:=` 演算子を通ります。
- **`TOptKind` はなし（クラウドセッション3で改訂）。** record 型の引数には Pascal の
  デフォルト値を付けられず（実測：「It is not possible to define a default value
  for a parameter of type TOptInt64」）、`dtype: ScalarType? = None` は末尾の
  省略可能引数として最も多い（54 op：`sum`、`mean`、`cumsum`、`softmax` など）。
  そこで `ScalarTypeOption` は `TKind` にし、`kNone` を「なし」とします。C の
  エンコード（`-1`）そのままで、こうすると `= kNone` のデフォルトが付きます。
  `LayoutOption` も `ltNone` で同じ。`Int64Option`/`DoubleOption` は record の
  まま：`Int64`/`Double` には「なし」に使える余りの値がありません。
- **`FromArray` のオーバーロード（決定、H2 の続き。クラウドセッション5で改訂）。**
  最初の決定（クラウドセッション3、ローカルセッション2で実装）：`array of Single`、
  `array of Double`、`array of Int64`、`array of Byte`、`array of Boolean`。`Int32` は
  やめる（`Int64` と曖昧、H2）。`Int32` のデータは `FromInt32Array` で、`Int16`/`Int8` は
  `FromBytes` で。**H2 追記、実測**（ローカルセッション3、クラウドセッション5。FPC 3.2.2）：
  オープン配列コンストラクタの要素型は**先頭**の要素で決まり、実数リテラルの型が
  `Single` になるのは、その値が binary32 で正確に表せるときだけです。`Single` と `Double`
  の両方のオーバーロードがあると、`[1.0, 2.0]`、`[2.5]`、`[0.75]`、`[100.0]`、`[1.5e2]` は
  `Single` を選び、`[1.0, 0.1]` も同様です — `0.1` や `[1.5, D]` の `Double` 変数 `D` は
  黙って `Single` に丸められます。`[0.9, -1.9]`、`[3.14]`、`[0.1]`、`[1e-3]`、`[0.3]`、
  `[0.1, 1.0]` はコンパイルエラー（"Can't determine which overloaded function to
  call"）。つまりリテラルがコンパイルできるかどうかが先頭要素の二進展開に依存します。
  **改訂した決定：** `Single` のオーバーロードを外し、残る4つのオーバーロードの末尾に
  自然な既定値を持つ kind 引数を付けます：

  ```pascal
  class function FromArray(const A: array of Double;  AKind: TKind = kFloat): TTensor; overload; static;
  class function FromArray(const A: array of Int64;   AKind: TKind = kInt64): TTensor; overload; static;
  class function FromArray(const A: array of Byte;    AKind: TKind = kUInt8): TTensor; overload; static;
  class function FromArray(const A: array of Boolean; AKind: TKind = kBool):  TTensor; overload; static;
  class function FromSingleArray(const A: array of Single; AKind: TKind = kFloat): TTensor; static;
  ```

  ちょうどこの組で実測：`[1.0, 2.0]`、`[0.9, -1.9]`、`[1.0, 0.1]`、`[3.14]`、`[1.5, 2]`、
  `[D, 1.5]`、`[1.5, D]`、`[S]`（`Single` 変数）はすべて `Double` に解決。`[1, 2]`、
  `[1, 5000000000]`、`[N, 2]` は `Int64` のまま。`[Byte(200), 1]` は `Byte`、
  `[True, False]` は `Boolean`。`FromArray([0.9, -1.9], kDouble)` と
  `FromArray([1, 2], kFloat)` は kind をそのまま渡す。`array of Single` の*変数*はどの
  オーバーロードにも合わなくなるので `FromSingleArray` を置く（`Int32` と同じ扱い）。
  `[1, 2.5]` は引き続きエラー（`Int64` の配列になり、次に `Single`）：`[1.0, 2.5]` と
  書く。利用者向けに書く規則：リテラル配列の型は先頭要素で決まる。先頭が整数なら
  整数配列になる。
  - 既定の `kFloat` は PyTorch の `torch.tensor([1.0, 2.0])` の既定（float32）を保ち、
    Python と同じビット列になります（10進 → binary64 → binary32）。kind 引数は結果への
    `.to_dtype` の代わりです。
  - 実装：`Double` で `AKind = kFloat` のときは Pascal 側で `Single` のバッファに変換して
    `at_tensor_of_data` を1回呼ぶ（2つ目のテンソルは作らない）。それ以外の組み合わせは
    自然な kind で作り、違えば `totype(AKind)`。`FromSingleArray` は旧 `Single` 版の本体に
    kind を足したものです。
  - 既存の呼び出し側：`tests/api` は `TDoubleArray.Create(0.25, 0.5)` が `kDouble` になる
    ことを期待している（今後は `kDouble` を渡さなければ `kFloat`）、`tests/nn` は
    `[Double(0.9), -1.9]` とキャストしている（不要になる）、
    `tests/compiler/h2_fromarray.pas` は旧い組を記録している — すべてローカルセッション4で
    更新します（HANDOFF）。
  - **実施済み（ローカルセッション4）。** このとおり実装。`tests/compiler` に新しい
    `h2_fromarray.pas`（上の各行を再実測）と `xfail_h2_single.pas`（両方の
    オーバーロードがあると `[0.9, -1.9]` は曖昧）。`tests/api` は `[0.9, -1.9]` の
    float32 のビット、`kDouble`、`[1, 2], kFloat`、`FromSingleArray` を確認し、
    `tests/nn` は `array of Single` 変数に `FromSingleArray` を使う。テストで踏んだ罠で、
    利用者にも伝えるべきもの：x86-64 では比較式中の実数リテラルは `Extended` なので、
    binary64 で正確な `0.9` に対しても `T.DoubleValue([0]) = 0.9` は False。
    `Double(0.9)` と比較する（実測）。
- **要素サイズ（ローカルセッション2で実測）。** `KindElementSize(AKind)` は Pascal 側の
  表で、`tests/api` が 46 種すべてについて libtorch の `element_size()` と照合済み
  （libtch の `at_copy_data` はサイズが違うと拒否する）。1 バイト未満の型と
  Float8/Float4 はすべて 1 バイト、`kNone` は `ETorchError`。よって `FromBytes` と
  `CopyData` はどの型も受け付け、`To{Single,Double,Int64}Array` は先に `to_dtype` で変換する。
- `IntListOption`（103 箇所）は未解決の論点でした。オープン配列も動的配列も
  「なし」と「空」を区別できない（どちらも nil／長さ0）ので、`HasValue` を持つ
  record にし、`IntList([...])` で作ることにしました。
- `TensorOptList`（13 箇所）には追加の型は要りません。`array of TTensor` で、
  `IsNil` の要素が「なし」を意味します。

生成器が使う引数の対応：

| gen.ml の型 | Pascal の引数 |
|---|---|
| Tensor | `const X: TTensor`（nil ハンドルは呼び出し前に拒否） |
| TensorOption | `const X: TTensor`（nil ハンドル = なし） |
| Scalar | `const X: TScalar` |
| Bool | `X: Boolean` |
| Int64 / Double | `X: Int64` / `X: Double` |
| Int64Option / DoubleOption | `const X: TOptInt64` / `TOptDouble` |
| IntList | `const X: array of Int64`（`[2, 3]` と書ける） |
| IntListOption | `const X: TOptIntList`（`IntList([..])` / `NoIntList`） |
| DoubleList | `const X: array of Double` |
| TensorList / TensorOptList | `const X: array of TTensor` |
| String | `const X: AnsiString`（ポインタ＋長さで渡す。`None` は tch-rs と同じく空文字列として C に届く） |
| TensorOptions | `X_kind: TKind; const X_device: TDevice`（YAML の名前は 80 op すべて `options`） |
| ScalarType / ScalarTypeOption | `X: TKind`（option では `kNone` = なし） |
| Device | `const X: TDevice` |
| Layout / LayoutOption | `X: TLayout`（option では `ltNone` = なし） |

戻り値：テンソル1つ → `TTensor`、可変個 → `TTensorArray`、固定 N 個 →
`out` 引数、または結果が渡したテンソルそのものなら なし（4.4 参照）、
`bool`/`int64_t`/`double` → `Boolean`/`Int64`/`Double`。

**(H6) 解決**（手順6）：`TTensor.cat([A, B], 0)` と
`TTensor.cat([A + B, B, A * B], 1)` は正しい形の結果を返し、20万回呼んでも RSS は
一定で、ルーチンを抜けるとすべてのハンドルが解放されます（リークも二重解放もなし）。
オープン配列コンストラクタの要素も、ほかの一時値と同じように保持されます（H1）。

### 4.4 生成する op メソッドと命名（決定）

- レシーバーの規則は tch-rs と同じです。引数に `self` という名前の `Tensor` が
  あれば、それをレシーバーとする `TTensor` のメソッドにします。なければ `input`
  という名前のもの（152 個。`conv2d`、`batch_norm` など。`x.conv2d(w, ...)` と
  PyTorch のように読める）。それもなければ `TTensor` の
  `class function ... static` にします（563 個。例：
  `TTensor.zeros([2, 3], kFloat, TDevice.Cpu)`）。tch-rs の `Tensor::zeros` に
  合わせた形です。レシーバーは `const` です。インプレース op でも `const` のままです
  （record は変わらず、変わるのはその背後の C++ テンソル）。
- **op 名は tch-rs/PyTorch のスネークケースのまま使います**
  （`t.max_pool2d(...)`、`t.add_(u)`、`t.sum_dim_intlist(...)`）。
  手書きの API は Pascal 流のキャメルケースにします（`Shape`、`Kind`、`ToString`）。
  理由：スネークケースからキャメルケースへの変換は情報が落ちる（`_foo`/`foo`/`foo_` が
  衝突する）、元の名前のままなら PyTorch や tch-rs のドキュメントで検索できる、
  そして呼び出し箇所で「生成か手書きか」が見分けられる。先頭や二重の
  アンダースコア（`_cast_byte`、`__and__`、530 個）も FPC の識別子として正しいので
  そのまま残します。tch-rs の `internal_` への改名は不要です。
- tch-rs 4227b89 で確認済み：2,657 個の `atg_*` 名には大文字が含まれず、
  大文字小文字を無視しても重複がありません。また、1つの op の中で大文字小文字だけが
  違う引数もありません。それでも生成器は、生成する名前と手書きのメンバー名との衝突を
  大文字小文字を無視してチェックする必要があります。手書きの API が避けるべき既知の
  重複：`data`、`grad`、`detach`、`to_device`、`absolute`、`alias`、`index`、
  そして `clone`。`clone` は gen.ml が `clone` そのものを除外し、オーバーロードの
  重複除去で素の名前が一番短い生き残りに与えられるため、実体は `clone_out` です。
  だから手書きのデータコピーは `Clone` ではなく `Copy` にしています。
- 予約語。FPC の予約語と衝突する op 名：`div`、`repeat`、`set`、`to`、`var`。
  修飾子と衝突するもの：`absolute`、`alias`、`index`。メソッドは名前を保ったまま、
  宣言でも呼び出しでも FPC の `&` エスケープを使います（`t.&div(u)`、`t.&var(0)`）。
  op に `_` を付ける方法は使えません。末尾のアンダースコアはインプレースを
  意味するからです。修飾子もエスケープします。正しい識別子に `&` を付けても害はなく、
  FPC がどの修飾子をメソッド名として受け付けるかを推測せずに済みます **(H7、手順7で
  解決**：`&div`、`&var`、`&to`、`&set`、`&repeat`、`&absolute`、`&alias`、`&index` は
  コンパイルでき、`t.&div(u)` のように呼べる。`absolute`、`alias`、`index` は `&` なしでも
  コンパイルできる）。
  予約語と同じ引数名も同じ `&` エスケープにします：`out`（819 箇所）、`end`（31）、
  `to`（7）は `&out`、`&end`、`&to`。こうすると op 名も引数名も1つのエスケープ
  ルールで済みます（H12 で確認）。例外は `result`（3 op：`_linalg_det_result`、
  `_linalg_solve_ex_result`、`linalg_solve_ex_out`）。`&result` にしても識別子は
  `result` のままで、関数の戻り値のシンボルと衝突するので `result_` にします。
  必要な箇所には、手書きの分かりやすい別名（`DivBy`、`ToDevice`、`Variance`）を
  上に足せます。
- **引数名・out の結果名と `TTensor` のメンバー（決定。クラウドセッション3で改訂）。**
  素の `{$mode objfpc}` では、メソッドの引数に、それより前に宣言された record の
  メンバーと同じ名前（大文字小文字を無視）を使えません（`Dim` の横の `dim`、
  op `indices` の横の `indices`）。コンパイルが通るかは宣言の順番で変わります。
  最初の決定（ローカルセッション1）は `_` を足すルール（`dim_`、`indices_`、`pow__`）
  と `Dim` → `NumDims` の改名でした。これを **(H12、実測済み)** で置き換えます：
  `Tch.pas` に `{$modeswitch duplicatelocals}` を足す。これは Delphi モードのうち
  まさにこれを許す部分だけで、ほかは `objfpc` のままです。FPC 3.2.2 で実測
  （`tests/compiler/h12_*`、`xfail_h12_*`）：
  - 引数名や `out` の名前は、前後どちらに宣言されたメンバーとも同じにできる。
    op `indices` の横の `out indices` も、`pow`/`pow_` の横の引数 `pow` も通る。
    本体の中では裸の名前が引数、`Self.Dim` がメンバー。
  - スイッチが要るのはメソッドを宣言するユニット（`Tch.pas` とその `.inc`）だけ。
    利用側のユニットはどのモードでもよい。
  - *ローカル変数* がメンバーと同名になるのは `objfpc` では引き続きエラー
    （これを許すのは `{$mode delphi}` だけ）。手書きの本体がローカル変数で
    うっかりメンバーを隠すことはない。
  - *関数* の引数名 `result` は引き続き重複識別子（メンバーではなく戻り値の
    シンボル）。上の `result_` はそのため。
  - コンパイラのソースでこのスイッチ（`m_duplicate_names`）を見ているのは
    `symtable.pas` の3箇所だけ：引数とメンバーの重複チェックと、TP/Delphi の
    「関数と同名のローカル変数」の扱い。ほかには何も変わらない。
  結果：生成するシグネチャは PyTorch の引数名をそのまま保ち（`dim`、`indices`、
  `pow`）、生成器にメンバー向けの接尾辞ロジックはなく、手書きの API は `Dim` を
  使えます（`NumDims` から戻した）。`Tch.pas` の手書きコードでの注意：メンバーと
  同名の引数はその本体の中で黙ってメンバーを隠すので、`Self.Member` と書くか、
  手書きの引数にはメンバーと重ならない名前を付けます。
- 固定個数の複数テンソルを返す op（246 個、2〜7 個。例：`max_dim` → values、indices）は
  `procedure max_dim(dim: Int64; keepdim: Boolean; out values, indices: TTensor)`
  にします。出力の名前は、`Declarations.yaml` の `returns[].name` があればそれを使い、
  なければ `result0..N` にします。理由：Pascal には分割代入がなく、`out` が定石
  （`DecodeDate`、`DivMod`）で、op ごとに結果用の record を作ると 246 個の型が増えます。
  よく使う少数の op（`max_dim`、`sort`、`topk`、`lstm`）には、実際のコードで
  必要性が見えたら手書きの結果用 record を足せます。
  注意：FPC は `out` 引数を関数に入った時点で解放する **(H8、手順6で確認済み)** ので、
  レシーバーや入力を `out` 引数に渡す（`t.max_dim(1, False, t, i)`）と、本体が
  走る前にそれが解放されます。その場合は nil ハンドルのチェックにより、クラッシュではなく
  `ETorchError` になります。ドキュメントに書き、小細工はしません。
  あわせて確認済み：以前の値を持つ変数を `out` に渡すと、その値は解放されます
  （呼び出しの前後でハンドル数が変わらない）。
- **結果が渡したテンソルそのものである op（決定、クラウドセッション3）。**
  複数結果の 246 op のうち 97 は「out=」版（`max_dim_out`、`sort_values`、
  `topk_values`、`qr_q`、`linalg_solve_ex_out` など）で、結果名がすべて引数名でもあり、
  libtorch はまさにそのテンソルへのハンドルを返します。ops.tsv で実測：97 op すべてで
  *全部の* 結果が重なり、一部だけ重なる op はない。ルール：こういう op は
  **out 引数のない `procedure`** にする。生成した本体は呼び出し後に N 個の結果ハンドルを
  解放し（`TchFreeSlots`）、テンソルは呼び出し側がすでに持っている
  （`t.sort_values(V, I)` が `V` と `I` を埋める）。一部だけ重なる場合は生成器のエラー。
  結果1つの「out=」op（`add_out` など 800 以上）はこれまでどおり `TTensor` を返す
  （`out` への新しいハンドル。C++ の `Tensor&` と同じ）。
- デフォルト引数（クラウドセッション3で改訂）。gen.ml は各引数の `default` を正規化した
  文字列で ops.tsv に書き（5 章）、tchgen はデフォルト値を付けられる引数が *末尾に*
  連続する最大の範囲を Pascal のデフォルト引数にします。付けられる型：`Bool`、`Int64`、
  `Double`、`String`、`ScalarType`、`ScalarTypeOption`（`= kNone`）、`Layout`、
  `LayoutOption`（`= ltNone`）。付けられない型（そこで範囲が止まる）：record
  （`TOptInt64`、`TOptDouble`、`TOptIntList`、`TScalar`、`TDevice`）、オープン配列、
  `TTensor`（実測：FPC は record 引数のデフォルト値を拒否する）。Pascal の型で
  「なし」を表せない型の `::std::nullopt`（`bool?` 23 op、`str?` 50 op：C には
  ただの bool／空文字列が届く）には **デフォルトを付けない**。tch-rs と同じく
  呼び出し側が選ぶ。`at::Reduction::Mean` → `1`、`"constant"` → `'constant'`
  （C のエスケープを解く）。`out` 引数のある op にはデフォルトなし（デフォルトは
  最後の引数でなければならない）。渡したテンソルを埋める op には付く。2.13 での実測：
  479 op に1つ以上のデフォルトが付く。out 引数のある 149 op のうち 107 にはデフォルトが
  付いたはず（`topk`、`sort` など）で、手書きのラッパーで後から足せる。
- `TensorOptions` 引数は明示的なまま（`options_kind: TKind; const
  options_device: TDevice`。C の名前）にします。`kFloat`/CPU を既定にした手書きの
  `TTensor.zeros([2, 3])` オーバーロードは安く作れ、tch-rs の `FLOAT_CPU` に相当します。
  **生成された名前のオーバーロード（ローカルセッション2で実測）。** `objfpc` では
  record のメソッドのオーバーロードに `overload` 指令は要らない（宣言のどちらにも
  付けない／片方／両方、いずれでも `zeros([2, 3])` と `zeros([2, 3, 4], kFloat, Dev)`
  の解決は同じ）。それでも生成 op の手書きオーバーロードには、tchgen が読む目印として
  `overload;` を付ける：宣言に `overload;` がある手書きメンバーは意図したオーバーロード、
  それ以外で op と同名の手書きメンバーは衝突（エラー）。このため tchgen は複数行の
  宣言をまとめて読む（継続行をフィールドと誤認しなくなった）。マイルストーン1には
  CPU 上の `zeros`、`ones`、`empty`、`rand`、`randn` `(ASize; AKind = kFloat)` がある。
- **生成する本体（クラウドセッション3）。** ローカル変数はすべて `__` で終わる
  （`Outs__`、`P__`、`R__`、Scalar のハンドルとテンソルリストのポインタ配列は
  `<引数>__`）。YAML の引数名で `__` で終わるものはない（tchgen が確認）。
  レシーバーは常に `Self.Raw` と書き、組み込み関数は修飾する（`System.Ord`、
  `System.Length`、`System.PInt64`、`TchSys.PPTorchTensor`）。`ord` や `length` という
  引数があり、`duplicatelocals` の下では組み込み関数を隠してしまうため
  （`linalg_norm` がコンパイルできずに判明）。Scalar のハンドルは `try` の中で作り
  `finally` で解放する。テンソルのリストは nil チェックのあと `TTensorPtrArray` に
  写す（`TchTensorPtrs`、`TchTensorOptPtrs`）。「値あり・空」の `IntListOption` は
  nil でないポインタと長さ 0 で渡す（`TchOptIntListPtr`）。すべて `tests/mock` で
  動かしている。

### 4.5 ユニット構成

```
src/
  TchSys.pas                 L2 手書きの external 宣言
  TchSysGenerated.pas        L2 生成された external 宣言          （生成）
  Tch.pas                    L3 中核：TTensor, TDevice, TKind, TScalar,
                             ETorchError, 勾配モードの補助
  TchTensorGenerated_intf.inc  TTensor 内のメソッド宣言           （生成）
  TchTensorGenerated_impl.inc  メソッドの本体                     （生成）
  TchNN.pas                  TInit, TVarStore, TVarPath, TModule, 各レイヤー,
                             TSequential, TOptimizer（4.6）
  TchJit.pas                 TJitModule, TIValue, TTrainableJitModule（4.6）
gen/
  gen.ml, dune, dune-project  tch-rs の生成器のフォーク（+ ops.tsv 出力）
  ops.tsv                    tchgen が読む一覧                   （生成）
  tchgen.pas                 Pascal 生成器（5 章）
csrc/libtch/                 フォークした C ラッパー + CMakeLists.txt
tests/mock/                  モックの libtch + 生成層のテスト（libtorch 不要）
tests/jit/                   TchJit のテスト（本物の libtorch）。models/ = tch-rs の tests/foo*.pt
                             （tch-rs のライセンスファイルを隣に置く。4.6.8）
```

- 生成する record のメンバーは record の中で宣言しなければならないので、
  include ファイルを2つに分けています。生成した op を別ユニットの
  `record helper for TTensor` に置く案も検討しましたが、不採用です。FPC はある型に
  対して、スコープ内の *最後の* ヘルパーしか解決しないので、ユーザーが `TTensor` の
  ヘルパーを1つ定義しただけで 2,657 個の op がすべて黙って見えなくなります。
- オプティマイザは別の `TchOptim` ではなく `TchNN` に置きます。`TOptimizer` は
  var-store の学習対象変数の一覧とグループを必要とし（tch-rs と同じく、オプティマイザより
  後に作られた変数も遅延登録する）、それはユニットをまたいで公開しない方がよい
  内部状態だからです。tch-rs も同じ理由で `nn` に置いています。

### 4.6 nn、VarStore、オプティマイザ、JIT（決定）

tch-rs の形と、それに対応する Pascal：

| tch-rs | 役割 | Pascal | 理由 |
|---|---|---|---|
| `VarStore`（`Arc<Mutex<Variables>>`、device、kind） | 全パラメータを名前付きで所有 | `TVarStore = class` | 寿命が長く、明示的な寿命を持つ実体。モデルを使い終えたら `Free` する |
| `Path<'a>`（借用、`vec<String>` + group） | 名前空間。コピーが安い | `TVarPath = record`（所有しない `TVarStore` 参照 + ドット区切りのパス + group） | 値の意味論、所有権なし。`P.Sub('layer1')` / `P / 'layer1'` |
| `Entry::or_*` | 取得または作成 | `TVarPath.GetOrAdd(...)` | 別の型は不要 |
| ペイロード付きの `Init` enum | 初期化の方法 | `Kind` + フィールドと静的コンストラクタを持つ `TInit = record` | Pascal の列挙型はペイロードを持てない |
| `Module` / `ModuleT` トレイト | 順伝播 | `TModule = class abstract`。仮想メソッド `Forward(const X; Train: Boolean = False)` 1つ | Pascal にはブランケット実装がない。デフォルト引数で2つのトレイトをまとめる |
| `Linear`、`Conv2D`、`BatchNorm`、`LSTM`... の struct | `Tensor` フィールドを持つレイヤー | `TModule` を継承したクラス。パラメータは公開の `TTensor` フィールド（`Weight`、`Bias`） | 所有権の考え方は tch-rs と同じ：レイヤーと var-store がハンドルを共有する |
| `Sequential`（`Vec<Box<dyn Module>>`） | 合成 | `TSequential = class(TModule)`。子を所有する | 所有関係の木が明示的で、Pascal らしい |
| `Func`（クロージャのレイヤー） | その場限りのレイヤー | 普通の関数ポインタか `of object` メソッドを受け取る `TFuncModule` | FPC 3.2.2 には無名関数がない。`add_fn(|x| ...)` の代わりに名前付き関数か小さな `TModule` の派生クラスを使う |
| `Optimizer` + `OptimizerConfig` の struct | 学習 | `class function Adam(VS, LR; Beta1 = 0.9; ...)` というコンストラクタを持つ `TOptimizer = class` | Pascal のデフォルト引数がビルダーや設定用 struct の代わりになる |
| `no_grad(|| ...)`、`NoGradGuard` | 勾配モード | `SetGradEnabled(B): Boolean` + `NoGradGuard: IInterface` | 4.7 参照 |
| `CModule`、`TrainableCModule`、`IValue` | TorchScript | `TJitModule = class`、`TTrainableJitModule = class(TModule)`、タグ付き record の `TIValue` | 下記参照 |
| `Iter2` | ミニバッチ | `for .. in` 用に `GetEnumerator` を持つ `TBatchIter` | 後で |

上の形はクラウドセッション2で決めたものです。クラウドセッション4では、HANDOFF が
「まだ機械的には実装できない」としていた7点（初期化の式、ストアの内部、save/load、
オプティマイザ、レイヤー、sequential、ユニットとテスト）を、同じ tch-rs のコミットに
照らして決めました。以下のファイル名は tch-rs のものです。本物の libtorch がないと
確認できないことは **(H13)**〜**(H18)**、クラウドで FPC 3.2.2 上で計測したことは
*計測済み* と記します。すべて1つのユニット `TchNN` に置きます。

スケッチ（インターフェース。本体は 4.6.1〜4.6.7 に記述）：

```pascal
unit TchNN;
{$mode objfpc}{$H+}{$modeswitch advancedrecords}
uses SysUtils, Generics.Collections, TchSys, Tch;
type
  // --- 初期化（tch-rs nn/init.rs） ---
  TFanMode = (fmFanIn, fmFanOut);                              // FanInOut
  TInitDist = (idUniform, idNormal);                           // NormalOrUniform
  TNonLinearity = (nlReLU, nlLinear, nlSigmoid, nlTanh, nlSELU, nlExplicitGain);
  TInitKind = (ikConst, ikUniform, ikRandn, ikKaiming, ikOrthogonal);
  TInit = record
    Kind: TInitKind;
    A, B: Double;                 // Const: A; Uniform: lo, up; Randn: mean, std; Orthogonal: gain は A
    Dist: TInitDist;              // Kaiming のみ
    Fan: TFanMode;
    NonLin: TNonLinearity;
    Gain: Double;                 // nlExplicitGain 用
    class function Constant(V: Double): TInit; static;        // "Const" は予約語
    class function Uniform(Lo, Up: Double): TInit; static;
    class function Randn(Mean, Std: Double): TInit; static;
    class function Kaiming(ADist: TInitDist; AFan: TFanMode; ANonLin: TNonLinearity;
      AGain: Double = 0): TInit; static;
    class function KaimingUniform: TInit; static;   // Kaiming(idUniform, fmFanIn, nlReLU) = DEFAULT_KAIMING_UNIFORM
    class function KaimingNormal: TInit; static;    // Kaiming(idNormal, fmFanIn, nlReLU) = DEFAULT_KAIMING_NORMAL
    class function Orthogonal(AGain: Double = 1): TInit; static;
    function Make(const Dims: array of Int64; const ADevice: TDevice; AKind: TKind): TTensor;  // f_init
    procedure Apply(const T: TTensor);              // Init::set。その場で、NoGradGuard の下で
  end;
  function FanForDims(Fan: TFanMode; const Dims: array of Int64): Int64;   // FanInOut::for_weight_dims
  function NonLinearityGain(NL: TNonLinearity; ExplicitGain: Double = 0): Double;

  // --- var store（tch-rs nn/var_store.rs） ---
  TVarStore = class;

  TVarPath = record
  private
    FStore: TVarStore;            // 所有しない。パスより長く生きている必要がある
    FPath: String;                // 'encoder.layer1'（'' = ルート）
    FGroup: Integer;
  public
    function Sub(const Name: String): TVarPath;               // Name に '.' があれば例外
    class operator /(const P: TVarPath; const Name: String): TVarPath;
    function SetGroup(G: Integer): TVarPath;
    function Group: Integer; inline;                          // ローカルセッション3の追加、承認済み
    function Store: TVarStore; inline;
    function Device: TDevice;
    function Kind: TKind;
    function FullName(const Name: String): String;            // Path::path
    function Add(const Name: String; const T: TTensor; Trainable: Boolean): TTensor;
    function Variable(const Name: String; const Dims: array of Int64; const Init: TInit): TTensor; // var
    function Zeros(const Name: String; const Dims: array of Int64): TTensor;
    function Ones(const Name: String; const Dims: array of Int64): TTensor;
    function ZerosNoTrain(const Name: String; const Dims: array of Int64): TTensor;
    function OnesNoTrain(const Name: String; const Dims: array of Int64): TTensor;
    function RandnStandard(const Name: String; const Dims: array of Int64): TTensor;
    function Randn(const Name: String; const Dims: array of Int64; Mean, Std: Double): TTensor;
    function Uniform(const Name: String; const Dims: array of Int64; Lo, Up: Double): TTensor;
    function KaimingUniform(const Name: String; const Dims: array of Int64): TTensor;
    function KaimingNormal(const Name: String; const Dims: array of Int64): TTensor;
    function Orthogonal(const Name: String; const Dims: array of Int64; Gain: Double): TTensor;
    function VarCopy(const Name: String; const T: TTensor): TTensor;
    function TryGet(const Name: String; out T: TTensor): Boolean;   // Path::get
    function GetOrAdd(const Name: String; const Dims: array of Int64; const Init: TInit): TTensor; // entry().or_var
    procedure SetKind(AKind: TKind);        // この部分木の全変数
    procedure SetFloatKind(AKind: TKind);   // この部分木の浮動小数点の変数のみ
  end;

  TNamedTensor = record Name: String; Tensor: TTensor; end;
  TNamedTensorArray = array of TNamedTensor;

  TVarEntry = record
    Name: String;                 // ドット区切りの完全な名前
    Tensor: TTensor;
    Trainable: Boolean;
    Group: Integer;
  end;
  TVarEntryArray = array of TVarEntry;
  TNameIndex = specialize TDictionary<String, Integer>;

  TVarStore = class
  private
    FVars: TVarEntryArray;        // 登録順、追加のみ（named_variables + 順序）
    FTrainable: array of Integer; // FVars の添字、登録順（trainable_variables）
    FIndex: TNameIndex;           // 完全な名前 -> FVars の添字
    FDevice: TDevice;
    FKind: TKind;
    function AddEntry(const FullName: String; const T: TTensor; Trainable: Boolean;
      Group: Integer): TTensor;   // TVarPath.Add が使う
  public
    constructor Create(const ADevice: TDevice; AKind: TKind = kFloat);
    destructor Destroy; override;
    function Root: TVarPath;
    function Count: Integer;
    function TrainableVariables: TTensorArray;
    function Variables: TNamedTensorArray;                    // 登録順
    function TryGet(const FullName: String; out T: TTensor): Boolean;
    procedure Save(const FileName: String);
    procedure Load(const FileName: String);
    function LoadPartial(const FileName: String): TStringArray;   // 欠けていた名前
    procedure Freeze;
    procedure Unfreeze;
    procedure SetDevice(const ADevice: TDevice);
    procedure SetKind(AKind: TKind);                          // set_kind：全変数
    procedure SetFloatKind(AKind: TKind);                     // half/bfloat16/float/double
    procedure CopyFrom(Src: TVarStore);                       // VarStore::copy
    property Device: TDevice read FDevice;
    property Kind: TKind read FKind;
  end;

  // --- モジュールとレイヤー ---
  TModule = class abstract
  public
    function Forward(const X: TTensor; Train: Boolean = False): TTensor; virtual; abstract;
  end;

  TLinearConfig = record WsInit, BsInit: TInit; HasBsInit: Boolean; Bias: Boolean; end;
  TPaddingMode = (pmZeros, pmReflect, pmReplicate, pmCircular);
  TConvConfig = record
    Stride, Padding, Dilation, Groups: Int64; Bias: Boolean;
    WsInit, BsInit: TInit; PaddingMode: TPaddingMode;
  end;
  TBatchNormConfig = record CudnnEnabled: Boolean; Eps, Momentum: Double; Affine: Boolean; WsInit, BsInit: TInit; end;
  TLayerNormConfig = record CudnnEnabled: Boolean; Eps: Double; ElementwiseAffine: Boolean; WsInit, BsInit: TInit; end;
  TEmbeddingConfig = record Sparse, ScaleGradByFreq: Boolean; WsInit: TInit; PaddingIdx: Int64; end;
  function DefaultLinearConfig: TLinearConfig;      // DefaultConvConfig、DefaultBatchNormConfig なども同様

  TLinear = class(TModule)
  public
    Weight, Bias: TTensor;        // バイアスがないレイヤーでは Bias.IsNil
    constructor Create(const P: TVarPath; InDim, OutDim: Int64); overload;
    constructor Create(const P: TVarPath; InDim, OutDim: Int64; const Cfg: TLinearConfig); overload;
    function Forward(const X: TTensor; Train: Boolean = False): TTensor; override;
  end;
  TConv2D = class(TModule)       // tch-rs のスカラー ConvConfig による Conv2D
  private
    FCfg: TConvConfig;
    FPadTwice: TInt64Array;       // reversed_padding_repeated_twice
  public
    Weight, Bias: TTensor;
    constructor Create(const P: TVarPath; InDim, OutDim, KSize: Int64); overload;
    constructor Create(const P: TVarPath; InDim, OutDim, KSize: Int64; const Cfg: TConvConfig); overload;
    function Forward(const X: TTensor; Train: Boolean = False): TTensor; override;
  end;
  TBatchNorm = class(TModule)    // Nd = 1, 2, 3 が batch_norm1d/2d/3d。それ以外の Nd は登録前に ETorchError
  public
    Weight, Bias, RunningMean, RunningVar: TTensor;
    constructor Create(const P: TVarPath; Nd: Integer; OutDim: Int64); overload;
    constructor Create(const P: TVarPath; Nd: Integer; OutDim: Int64; const Cfg: TBatchNormConfig); overload;
    function Forward(const X: TTensor; Train: Boolean = False): TTensor; override;
  end;
  TLayerNorm = class(TModule)
  public
    Weight, Bias: TTensor;
    constructor Create(const P: TVarPath; const NormalizedShape: array of Int64); overload;
    constructor Create(const P: TVarPath; const NormalizedShape: array of Int64; const Cfg: TLayerNormConfig); overload;
    function Forward(const X: TTensor; Train: Boolean = False): TTensor; override;
  end;
  TEmbedding = class(TModule)
  public
    Weight: TTensor;
    constructor Create(const P: TVarPath; NumEmbeddings, EmbeddingDim: Int64); overload;
    constructor Create(const P: TVarPath; NumEmbeddings, EmbeddingDim: Int64; const Cfg: TEmbeddingConfig); overload;
    function Forward(const X: TTensor; Train: Boolean = False): TTensor; override;
  end;

  TTensorFunc    = function(const X: TTensor): TTensor;
  TTensorFuncT   = function(const X: TTensor; Train: Boolean): TTensor;
  TTensorMethod  = function(const X: TTensor): TTensor of object;
  TTensorMethodT = function(const X: TTensor; Train: Boolean): TTensor of object;

  TFuncModule = class(TModule)   // Func / FuncT
  public
    constructor Create(F: TTensorFunc); overload;
    constructor Create(F: TTensorFuncT); overload;
    constructor Create(F: TTensorMethod); overload;
    constructor Create(F: TTensorMethodT); overload;
    function Forward(const X: TTensor; Train: Boolean = False): TTensor; override;
  end;

  TModuleList = specialize TObjectList<TModule>;

  TSequential = class(TModule)   // SequentialT：Train を全ての子に渡す
  private
    FLayers: TModuleList;         // OwnsObjects
  public
    constructor Create;
    destructor Destroy; override;
    function Add(M: TModule): TSequential;                    // 所有権を取る。Self を返す
    function AddFn(F: TTensorFunc): TSequential; overload;
    function AddFn(F: TTensorFuncT): TSequential; overload;
    function AddFn(F: TTensorMethod): TSequential; overload;
    function AddFn(F: TTensorMethodT): TSequential; overload;
    function Forward(const X: TTensor; Train: Boolean = False): TTensor; override;
    function ForwardAll(const X: TTensor; Train: Boolean = False; N: Integer = -1): TTensorArray;
    function Count: Integer;
    property Items[I: Integer]: TModule read GetItem; default;
  end;

  // --- オプティマイザ（tch-rs nn/optimizer.rs、wrappers/optimizer.rs） ---
  TOptimizer = class
  private
    FOpt: PTorchOptimizer;
    FStore: TVarStore;            // 所有しない。オプティマイザをストアより先に解放する
    FVarsInOptimizer: Integer;    // variables_in_optimizer
    {$push}{$warn 3018 off}       // 「Constructor should be public」：API は下の class function（4.6.4）
    constructor CreateRaw(AOpt: PTorchOptimizer; VS: TVarStore);
    {$pop}
    procedure AddMissingVariables;
  public
    class function Sgd(VS: TVarStore; LR: Double; Momentum: Double = 0; Dampening: Double = 0;
      WeightDecay: Double = 0; Nesterov: Boolean = False): TOptimizer; static;
    class function Adam(VS: TVarStore; LR: Double; Beta1: Double = 0.9; Beta2: Double = 0.999;
      WeightDecay: Double = 0; Eps: Double = 1e-8; AmsGrad: Boolean = False): TOptimizer; static;
    class function AdamW(VS: TVarStore; LR: Double; Beta1: Double = 0.9; Beta2: Double = 0.999;
      WeightDecay: Double = 0.01; Eps: Double = 1e-8; AmsGrad: Boolean = False): TOptimizer; static;
    class function RmsProp(VS: TVarStore; LR: Double; Alpha: Double = 0.99; Eps: Double = 1e-8;
      WeightDecay: Double = 0; Momentum: Double = 0; Centered: Boolean = False): TOptimizer; static;
    destructor Destroy; override;                             // ato_free
    procedure ZeroGrad;
    procedure Step;
    procedure BackwardStep(const Loss: TTensor);
    procedure BackwardStepClip(const Loss: TTensor; MaxValue: Double);
    procedure BackwardStepClipNorm(const Loss: TTensor; MaxNorm: Double);
    procedure ClipGradValue(MaxValue: Double);
    procedure ClipGradNorm(MaxNorm: Double);
    procedure SetLearningRate(LR: Double);
    procedure SetLearningRateGroup(Group: Integer; LR: Double);
    procedure SetMomentum(M: Double);                         // Adam/AdamW では beta1
    procedure SetMomentumGroup(Group: Integer; M: Double);
    procedure SetWeightDecay(WD: Double);
    procedure SetWeightDecayGroup(Group: Integer; WD: Double);
    function TrainableVariables: TTensorArray;
  end;
```

決定事項と理由（全体の形）：
- **`TVarStore` はクラス、`TVarPath` は record。** ストアは同一性と可変の状態を持ち、
  所有者がはっきりしています。パスは接頭辞とグループを運ぶだけの使い捨ての値です。
  パスは素のクラス参照を持つので、「ストアはそのパス、レイヤー、オプティマイザより
  長く生きる」という規則は強制ではなくドキュメントで示します（tch-rs のライフタイム
  `'a` と同じで、コンパイラのチェックがないだけ）。
- **`Module` と `ModuleT` の代わりに、`Train: Boolean = False` 付きの `Forward` 1つ。**
  `Train` を使わないレイヤーは無視すればよいだけです。派生型経由の呼び出しで
  デフォルト値が効くよう、オーバーライドでもデフォルト値を繰り返す必要があります **(H9、
  手順7で解決**：デフォルト値を繰り返せば、基底型からも派生型からも `Forward(X)` と
  呼べる。繰り返さないオーバーライドは、基底型経由なら動くが、`Derived.Forward(X)` は
  コンパイルエラーになる）。
  PyTorch 風にモジュールに `Training` フラグを持たせる案も検討しましたが不採用です。
  フラグを伝播させるには子の登録の仕組み（Python では `__setattr__` の魔法）が必要で、
  Pascal ではそれを全部書き下すことになるからです。クラウドセッション4で決着：
  `TSequential` が `Train` を全ての子に渡し（4.6.6）、ユーザーの `TModule` は明示的に
  渡す。それ以上は不要です。
- **レイヤーはテンソル（参照カウント付き）以外何も所有しないクラス。** tch-rs と同じく
  `TVarPath` を受け取って作られ、自分のパラメータを登録します。`TSequential` は子を
  所有します。ユーザーのモデルは `TModule` の派生クラスで、コンストラクタで
  サブレイヤーを作り、デストラクタで解放します。Lazarus のコンポーネントと同じ型です。
- **設定** は `Default*` 関数を持つ record にします（Pascal の record はフィールドに
  デフォルト値を持てない）。よくある使い方には設定なしのオーバーロードを用意します。
  オプティマイザは選択肢が平坦なので、代わりにデフォルト引数を使います。
- **変数名** は区切りに `.` を使い、`.ot` ファイルには `|` で書きます。tch-rs と
  まったく同じなので、ファイルは tch-rs や `Tensor::save_multi` の読み手と互換です。
- **スレッドセーフ性：** tch-rs は Rust がこの問題を避けさせないので `Variables` を
  mutex で包んでいます。Pascal の最初の版では `TVarStore` をスレッドセーフでないと
  明記します。API を変えずに `TCriticalSection` を後から足せます。
- **JIT。** `TJitModule` はクラスです（インスタンスは少なく、ファイルのように
  明示的な寿命を持つ）。`TIValue` はタグ付き record（`Tag`、`Tensor`、`I`、`D`、`B`、
  `S`、`Items: array of TIValue`、`Tensors: TTensorArray` など）で、静的コンストラクタ
  （`TIValue.Tuple([...])`）、`TTensor`、`Int64`、`Double`、`Boolean`、`String` からの
  暗黙の `:=` 演算子、タグが合わなければ例外を投げる `As*` アクセサを持ちます。
  **(H10)** 自分自身の型の動的配列を含む record が 3.2.2 でコンパイルできる必要があります。
  だめなら fpjson 風のクラス階層にします。**(H10) 解決**（手順7）：コンパイルでき、
  入れ子（タプルのタプル）も作れる。ドキュメントに書くべき意味論：`W := V` は `Items` を
  共有する（動的配列は参照カウントで、コピーオンライトではない）ので、`W.Items[0]` に
  書くと `V` も変わる。`TIValue` は作ったあとは変更しないものとして扱う。`TTrainableJitModule` は `NamedParameters` を
  `TVarPath` に登録し、`TModule` を継承するので、スクリプト化したモデルも他のレイヤーと
  同じく `TOptimizer` で微調整できます。マイルストーン2の範囲外です。機械的な設計は
  4.6.8（クラウドセッション5）。

#### 4.6.1 初期化（`TInit`）、`src/nn/init.rs` より

- **Fan**（`FanInOut::for_weight_dims`）：`Receptive := Dims[2..] の積`
  （次元が3未満なら 1）。`fmFanIn` は `Length(Dims) < 2` なら 1、そうでなければ
  `Dims[1] * Receptive`。`fmFanOut` は `Length(Dims) = 0` なら 1、そうでなければ
  `Dims[0] * Receptive`。したがって線形層の重み `[Out, In]` の fan-in は `In`、
  畳み込みの重み `[Out, In div Groups, K, K]` の fan-in は `In div Groups * K * K`。
- **Gain**（`NonLinearity::gain`）：ReLU `sqrt(2)`、Tanh `5/3`、Linear と Sigmoid `1`、
  SELU `0.75`、ExplicitGain `g`。*実測*（全体レビュー）：FPC は `5.0 / 3.0` を Single で畳み込む
  （1.6666666269）ので、コードは `Double(5) / 3` と書く。`tests/nn` が全ゲインを 1e-12 で確認する。
- **`Make(Dims, Device, Kind)`** は tch-rs の `f_init` を op 単位でなぞります
  （生成 op の名前は `TchTensorGenerated_intf.inc` のとおり）：

  | `TInit` | テンソル |
  |---|---|
  | `Constant(0)` | `TTensor.zeros(Dims, Kind, Device)` |
  | `Constant(1)` | `TTensor.ones(Dims, Kind, Device)`（tch-rs は `abs(c - 1) <= f64::EPSILON` で判定。`A = 1.0` でも同じテンソルになる） |
  | `Constant(c)` | `TTensor.ones(...).mul_scalar(c)` |
  | `Uniform(Lo, Up)` | `TTensor.zeros(...).uniform_(Lo, Up)` |
  | `Randn(0, 1)` | `TTensor.randn(Dims, Kind, Device)` |
  | `Randn(M, S)` | `TTensor.randn(...).mul_scalar(S).add_scalar(M)` |
  | `Kaiming(idUniform, Fan, NL)` | `Std := Gain / sqrt(FanForDims(Fan, Dims))`、`Bound := sqrt(3) * Std`、`TTensor.zeros(...).uniform_(-Bound, Bound)` |
  | `Kaiming(idNormal, Fan, NL)` | `TTensor.randn(...).mul_scalar(Std)` |
  | `Orthogonal(G)` | 下記 |

  したがって線形層の重み `[Out, In]` への `KaimingUniform` は `±sqrt(6 / In)` の範囲から
  引きます（gain `sqrt(2)`、fan-in `In`）。テストで手計算できる範囲です。
- **Orthogonal**（`Init::Orthogonal`）：`Length(Dims) < 2` なら
  `ETorchError('Only tensors with 2 or more dimensions are supported')`。
  `Rows := Dims[0]`、`Cols := Dims[1..] の積`。
  `F := TTensor.empty([Rows, Cols], Kind, Device).normal_(0, 1)`。
  `Rows < Cols` なら `F := F.t_`。`TTensor.linalg_qr(F, 'reduced', Q, R)`
  （`out Q, R` を持つクラス手続き）。`Ph := R.diag(0).sign`。`Q.mul_(Ph)`
  （最後の次元でブロードキャスト。tch-rs の `q *= ph`）。`Rows < Cols` なら
  `Q := Q.t_`。`Q.mul_scalar_(G)`。`Result := Q.contiguous.view(Dims)`。
  **明示する差異：** tch-rs の `f_init` は `Dims` が3次元以上でも 2-D の `[Rows, Cols]`
  を返します（`Init::set` が `view_as` で形を直す）。PyTorch の `orthogonal_` は要求された
  形を保つので、`Make` は最後に `.view(Dims)` を付けます（2-D なら何もせず、両者は一致）。
  **(H14)**：本物の libtorch で `[3, 5]` なら `Q·Qᵀ = I`、`[5, 3]` なら `Qᵀ·Q = I`
  （`allclose(…, 1e-5)`）。
  **(H14) 解決**（ローカルセッション3）：どちらも 1e-5 以内（`eye` との差の最大絶対値）、
  形は `Dims` のまま。`[2, 3, 2]` で gain 2 なら形を保ち、各行のノルムは 2。
- **`Make` に `NoGradGuard` は不要：** 中で作るテンソルはすべて `requires_grad = False`
  なので履歴は記録されません（tch-rs の `f_init` にも `no_grad` はなく、gain の乗算を
  囲む冗長なものがあるだけ）。`requires_grad` はあとで `TVarPath.Add` が設定します。
- **`Apply(T)`** は tch-rs の `Init::set`（既存の変数をその場で初期化し直す：`fill_`、
  `uniform_`、`randn_like * std + mean` の `at_copy_`、Orthogonal は
  `Make(T.Shape, T.Device, T.Kind)` の `at_copy_`、Kaiming は fan を `T.Shape` から取る）。
  自前の `NoGradGuard` の下で実行します。対象は普通 requires_grad を持つ葉で、そのような
  葉へのその場演算は libtorch で例外になるからです。tch-rs は `no_grad` を呼び出し側に
  任せています（テストは `init` を `no_grad` で包む）が、Pascal 版は自分でやります。
  **(H17)**：ガードなしで失敗し、ありで成功することを確認。このために `at_copy_` を包む
  手書きの `TTensor.CopyFrom(const Src: TTensor)` を `Tch.pas` に足します（生成 op に
  `copy_` はなく、`_copy_from` は別の識別子なので tchgen の名前チェックは通る）。
  **(H17) 解決**（ローカルセッション3）：ガードなしで学習対象のリーフに `fill_` すると
  `ETorchNativeError`「a leaf Variable that requires grad is being used in an in-place
  operation.」。`Apply` はすべての `TInit` で成功し、その後も変数は勾配を要求する。
- **Kind。** `TVarPath.Variable` などはストアの `Kind` を渡します（tch-rs の `f_var` は
  `self.kind()`）。tch-rs は `zeros_no_train`、`ones_no_train`、`Entry::or_var` で
  `Kind::Float` を直書きしていますが、Pascal 版はどこでもストアの kind を使います
  （差異。デフォルトの `kFloat` では同じ）。tch-rs の `Entry::or_kaiming_uniform` は
  *normal* のデフォルトを使っています（上流の書き間違い）。`GetOrAdd` は明示的な `TInit`
  を取るので、この問題は生じません。

#### 4.6.2 `TVarStore` の内部、`src/nn/var_store.rs` より

- **tch-rs の状態：** `named_variables: HashMap<String, Tensor>`（順序なし）と、登録順の
  `trainable_variables: Vec<Var { tensor, group }>`。オプティマイザは遅延追加のために
  この順序と vector の長さに依存しています。
- **コンテナ（決定。クラウドセッション4の M2 で計測）：** 登録順で追加のみの
  `FVars: array of TVarEntry`（要素は削除しない）、学習対象の要素の添字を登録順に並べた
  `FTrainable: array of Integer`（= tch-rs の `trainable_variables`）、名前の検索用に
  `Generics.Collections` の `FIndex: specialize TDictionary<String, Integer>`。
  FPC 3.2.2 で計測：`TDictionary<String, Integer>` はコンパイルでき、キーは大文字小文字を
  区別し（`weight` と `Weight` は別のキー）、重複する `Add` は `EListError` を投げる。
  `TTensor` フィールドを持つ record の動的配列は `SetLength` で伸ばしてもすべての
  ハンドルを保持し、配列を空にするとすべて解放する。`TList<TTensor>` も動く
  （`IndexOf` は record のバイト列、つまりハンドルを比較するので同一性比較になる）が
  不要。これで「等価性とコンテナ」の論点は閉じました：`TTensor` に `=` は要りません。
  `Variables` と `TrainableVariables` は登録順で返します（tch-rs の `variables()` は
  HashMap なので、順序が決まるのは無害な改善）。
- **`Add(Name, T, Trainable)`**（`Path::add`）：完全な名前は `FPath + '.' + Name`
  （ルートでは `Name` のみ）。すでに存在すれば `<完全名>__<Count>` になります（tch-rs：
  `format!("{}__{}", path, named_variables.len())`）。tch-rs の `save_and_load_var_store`
  テストは `t123` を2回登録して4変数になることを期待しているので、tch-rs が書いた
  ファイルにはこの名前が含まれうるため、規則をそのまま保ちます。`<完全名>__<Count>` も
  既にあれば（例：`x`、`x__2`、もう一度 `x`）、次の空いている番号を使う。確認はすべて
  ストアを変える前に行う（全体レビューの後で追加。そこで辞書のエラーによりストアが
  半端な状態になることが見つかった）。`Trainable` なら
  `T.set_requires_grad(True)`（生成 op。その場で効き、返るハンドルは捨てる）を呼び、
  添字を `FTrainable` に追加します。関数は `T` 自身を返します：レイヤー、ストア、
  呼び出し側がハンドルを共有します（tch-rs の `shallow_clone` は Rust の所有権のためだけに
  あるもの）。
- **名前の検証：** `Sub` とすべての変数名は、文字列に `.` が含まれていれば `ETorchError`
  を投げます（tch-rs は panic："sub name cannot contain ."、"variable name cannot
  contain ."）。tch-rs より厳しい点（明示）：`|` も拒否し（save/load の往復で `.` に
  なってしまう）、空文字列も拒否します。検査は `TVarPath.FullName` にあり、パスの操作は
  すべてそこを通るので、`TVarPath.TryGet('a.b')` は False を返す代わりに例外を投げます
  （tch-rs の `Path::get` は文字列を連結するだけ。ドットを含む名前は追加され得ないので、
  例外にするのが一貫した選択）。ストア側の `TVarStore.TryGet(FullName)` はドット区切りの
  完全名を受け取り、検査しません。`AddEntry` はテンソルを `TchCheckArg` で検査します
  （4.2：さもなければ `set_requires_grad` が nil ハンドルへの最初の C 呼び出しになる）。
  `TVarPath.Group` は `Device`/`Kind` と並んでグループを返します。この3点はローカル
  セッション3の追加で、クラウドセッション5で承認しました。
- **`GetOrAdd(Name, Dims, Init)`**（`Entry::or_var`）：完全名で既存のテンソルがあれば、
  `Dims` と `Init` が何であれそれを返します（tch-rs のテスト `var_store_entry`：2回目の
  `or_zeros(&[1, 5, 9])` は `[3, 1, 4]` のテンソルを返す）。なければ学習対象の変数を
  追加します。
- **`Freeze` / `Unfreeze`：** `FTrainable` の全要素に `set_requires_grad(False / True)`。
  要素は学習対象のままで、オプティマイザにも残ります（libtorch のオプティマイザは `grad`
  が未定義のパラメータを飛ばす。**(H18)**：`Freeze; Step` は値を変えず、例外も出ない）。
  その場で行い、ハンドルは変わりません。**(H18) 解決**（ローカルセッション3）：
  `ZeroGrad; Freeze` のあと `A * E`（E は凍結していないテンソル）を逆伝播すると
  `A.grad` は未定義のまま。`Step` は `A` を変えず、例外も出ない。
- **`SetDevice(D)`**（`set_device`）：全要素に `T.set_data(T.&to(D))`（生成 op の
  `set_data` と `&to`）、その後 `FDevice := D`。`set_data` は *同じ* C++ テンソル
  オブジェクトの裏のストレージを差し替えるので、ハンドルを共有するすべての `TTensor`
  （レイヤー、オプティマイザのパラメータグループ）が移動後のデータを見ます。勾配ガードは
  不要です（`set_data` は autograd の op ではない）。tch-rs と同じ注意点：オプティマイザの
  状態バッファは移動しないので、オプティマイザは `SetDevice` のあとで作ること。
  **(H13)**：`SetDevice` のあとレイヤーの `Weight` とストアの要素が新しいデバイスを示し、
  `SetKind(kDouble)` より前に作ったオプティマイザがその変数をまだ更新できる。
  **(H13) 解決**（ローカルセッション3、CPU のみ）：`SetKind(kDouble)` のあと `TLinear`
  の `Weight`（同じハンドル）は `kDouble` になり、変換前に作った SGD がそれを手計算どおり
  `[0.9, 0.8]` に更新し、`kDouble` のまま。`SetDevice(Cpu)` 後もハンドルは使える。実際の
  デバイス移動は 4.8 の GPU テスト項目 5。
- **`SetKind(K)`**（`set_kind` → ルートの `Path::set_kind`）：全要素に
  `T.set_data(T.totype(K))`、その後 `FKind := K`。**`SetFloatKind(K)`**（tch-rs の
  `half` / `bfloat16` / `float` / `double`）：`is_floating_point` が真の要素のみ。
  明示する差異：tch-rs の `half()` は *新しい* 変数のデフォルト kind を `Float` のままに
  しますが、`SetFloatKind` は `FKind` も設定します。`TVarPath.SetKind` / `SetFloatKind`
  は部分木に限定します：名前がパスと等しいか、パス + `.` で始まる要素（tch-rs は素の
  `starts_with` で、パス `a` が `ab.x` にも一致する。修正、明示）。
- **`CopyFrom(Src)`**（`VarStore::copy`）：1回目のパスで `Self` のすべての名前が `Src` に
  あることを確認し、なければ何もコピーする前に
  `ETorchError('variable <name> not found in source var-store')` を投げます（tch-rs も
  同じ2パス）。2回目のパスは `NoGradGuard` の下で `Dst.CopyFrom(SrcT.&to(FDevice))`。
- **`Destroy`** は `FIndex` を解放します。配列を空にするとストアの参照が解放され、
  レイヤーと共有しているテンソルは生き続けます。解放済みのストアを指したままの
  `TVarPath` の値はぶら下がりポインタです（上記のとおりドキュメントで示す）。
  `Count = Length(FVars)`（tch-rs の `len()`）。

#### 4.6.3 保存と読み込み、`var_store.rs` と `wrappers/tensor.rs` より

- **`Save(FileName)`**（`VarStore::save` → `Tensor::save_multi`）：*すべての* 要素を
  登録順に、名前の `.` を `|` に置き換えて `at_save_multi(tensors, names, n, filename)`。
  Pascal 側は `array of AnsiString` を生かしたまま、その中を指す `array of PChar` を
  渡します。tch-rs は `.safetensors` 拡張子に対して Rust コードで safetensors を書きますが、
  libtch にその入口はないので、この拡張子は
  `ETorchError('safetensors files are not supported')` を投げます。
- **読み込み**（`VarStore::named_tensors`）：拡張子で分岐、tch-rs とまったく同じ：
  `.bin` / `.pt` → `at_loadz_callback_with_device`（Python から書き出した pickle の zip）、
  `.safetensors` → `ETorchError`、それ以外 →
  `at_load_callback_with_device(filename, data, @LoadCallback, FDevice.ToC)`。使わないので
  `TchSys` にも宣言しないもの：`at_load_multi`、`at_load_multi_`（tch-rs も使わない）と、
  デバイスなしの2つのコールバック形式（デバイスありの形式で足りる）。テンソルはストアの
  デバイス上に届きます。ファイル内の余分な名前は無視します。
- **コールバック**（tch-rs の `add_callback`。クラウドセッション4の M1 で、トランポリンを
  3回呼ぶ C のハーネスを使って計測）：

  ```pascal
  // TchSys
  TTorchLoadCallback = procedure(data: Pointer; name: PChar; t: PTorchTensor); cdecl;
  procedure at_load_callback_with_device(filename: PChar; data: Pointer;
    f: TTorchLoadCallback; device_id: cint); cdecl; external LibTch;
  // TchNN の implementation
  type
    TLoadContext = class
      Items: TNamedTensorArray;
      Error: Exception;                 // 取得した例外オブジェクト。nil = なし
    end;
  procedure LoadCallback(data: Pointer; name: PChar; t: PTorchTensor); cdecl;
  var Ctx: TLoadContext; N: String; T: TTensor;
  begin
    Ctx := TLoadContext(data);
    try
      T := TTensor.FromRaw(t);          // どの経路でもハンドルを所有する（2章）
      if Ctx.Error <> nil then Exit;    // 前のコールバックが失敗した：以後は飛ばす
      N := StringReplace(StrPas(name), '|', '.', [rfReplaceAll]);
      SetLength(Ctx.Items, Length(Ctx.Items) + 1);
      Ctx.Items[High(Ctx.Items)].Name := N;
      Ctx.Items[High(Ctx.Items)].Tensor := T;
    except
      Ctx.Error := Exception(AcquireExceptionObject);
    end;
  end;
  ```

  C の呼び出しから戻ったら：無条件に `P := get_and_reset_last_err`（どの経路でも
  スロットを空にしておく）。`Ctx.Error <> nil` なら `tchf_free(P)` し、オブジェクトを
  コンテキストから取り出して `raise`。そうでなく `P <> nil` なら `TchRaiseNative(P)`。
  計測結果：クラスのインスタンスは `data` を通して無傷で届く。トランポリン内で投げた
  例外はそこで捕まり、取得したオブジェクトは C のフレームを通って戻ったあとも生きていて、
  呼び出し後にクラスとメッセージを保ったまま再 raise できる。以後のコールバックは
  飛ばされる。渡されたハンドルはすべて所有される。借用の `name` はコールバックが
  戻る前にコピーします。
- **ストアへのコピー**（`load_internal`、`load_partial`、
  `copy_data_with_precision_update`）。`Load`：1回目のパスでストアのすべての名前が
  読み込んだ一覧にあることを確認し、なければ何も変更する前に
  `ETorchError('variable <name> not found in <file>')` を投げます（tch-rs は HashMap の
  順序で最初に欠けた名前で止まり、それより前のものは更新済みのまま。2パスにすれば
  決定的になる）。その後、各要素について `NoGradGuard` の下で
  `Dst.set_data(Dst.totype(Src.Kind))` してから `Dst.CopyFrom(Src)`：コピー先が
  *ファイルの* kind になるので、`kFloat` のストアが半精度のファイルを読むと `kHalf` に
  なります（tch-rs のテスト `save_and_load_var_store` の後半）。`LoadPartial`：1回目の
  パスなしで同じことをし、欠けていた名前を登録順に返します（tch-rs の
  `save_and_load_partial_var_store_incomplete_file` は `['a.b.t2']` を期待）。
  **(H15)**：本物の libtorch での精度の更新。
  **(H15) 解決**（ローカルセッション3）：`SetFloatKind(kHalf)` して保存し、新しい
  `kFloat` のストアに `Load` すると、両変数とも `kHalf` で平均は 42 と 2（tch-rs の
  `save_and_load_var_store` と同じ）。

#### 4.6.4 オプティマイザ、`nn/optimizer.rs`、`wrappers/optimizer.rs`、`torch_api.cpp` より

- **L2：** `TchSys` に `PTorchOptimizer = Pointer;`（既存の `PTorchTensor` と同じ流儀）と、
  `torch_api.h` の宣言どおりの external：

  ```pascal
  function ato_sgd(learning_rate, momentum, dampening, weight_decay: Double;
    nesterov: cint): PTorchOptimizer; cdecl; external LibTch;
  function ato_adam(learning_rate, beta1, beta2, weight_decay, eps: Double;
    amsgrad: Boolean): PTorchOptimizer; cdecl; external LibTch;      // C++ の bool（2章）
  function ato_adamw(...ato_adam と同じ...): PTorchOptimizer; cdecl; external LibTch;
  function ato_rms_prop(learning_rate, alpha, eps, weight_decay, momentum: Double;
    centered: cint): PTorchOptimizer; cdecl; external LibTch;
  procedure ato_add_parameters(opt: PTorchOptimizer; t: PTorchTensor; group: csize_t); cdecl; external LibTch;
  procedure ato_set_learning_rate(opt: PTorchOptimizer; learning_rate: Double); cdecl; external LibTch;
  procedure ato_set_learning_rate_group(opt: PTorchOptimizer; group: csize_t; learning_rate: Double); ...
  procedure ato_set_momentum(opt: PTorchOptimizer; momentum: Double); ...
  procedure ato_set_momentum_group(opt: PTorchOptimizer; group: csize_t; momentum: Double); ...
  procedure ato_set_weight_decay(opt: PTorchOptimizer; weight_decay: Double); ...
  procedure ato_set_weight_decay_group(opt: PTorchOptimizer; group: csize_t; weight_decay: Double); ...
  procedure ato_zero_grad(opt: PTorchOptimizer); ...
  procedure ato_step(opt: PTorchOptimizer); ...
  procedure ato_free(opt: PTorchOptimizer); ...
  ```

  `ato_free`（素の `delete`）以外はすべて `PROTECT` 付きなので、毎回 `TchCheck`。
  コンストラクタはエラー時に `nullptr` を返しますが、ポインタを見る前に `TchCheck` が
  例外にします。C++ の `bool` に対する `Boolean` 引数は手順4で計測済み
  （`at_autocast_set_enabled`）。
- **構築**（`OptimizerConfig::build`）：C のオプティマイザを作り、`FTrainable` の各要素を
  順に `ato_add_parameters(FOpt, Tensor.Raw, Group)`、
  `FVarsInOptimizer := Length(FStore.FTrainable)`。`ato_add_parameters` は `group` までの
  パラメータグループをデフォルトの複製として作るので、グループは飛び飛びでもよい
  （tch-rs と同じ）。`CreateRaw(AOpt, VS)` は4つの class function が使う private な
  コンストラクタで、FPC 3.2.2 は `(3018) Constructor should be public` と警告します
  （クラウドセッション5で計測：`strict private` でも同じ）。決定：private のまま — API は
  factory であり、生ポインタを取る public なコンストラクタは API にしない — 宣言を
  `{$push}{$warn 3018 off} … {$pop}` で囲んで警告を消します（計測：警告は出なくなる）。
- **遅延追加**（`add_missing_variables`）：`ZeroGrad`、`Step`、3つの `BackwardStep*` は
  まず `FStore.FTrainable[FVarsInOptimizer..]` をそれぞれのグループで追加し、カウントを
  更新します。`ClipGrad*` とセッターはしません。これは `TVarStore` の private
  フィールドを読むので、`TOptimizer` は `TchNN` に置きます（4.5）。
- **デフォルト値**（`nn/optimizer.rs` の `impl Default`）：SGD は momentum 0、
  dampening 0、weight decay 0、nesterov false。Adam は β₁ 0.9、β₂ 0.999、weight decay 0、
  eps 1e-8、amsgrad false。**AdamW は weight decay 0.01**、残りは Adam と同じ。RmsProp は
  alpha 0.99、eps 1e-8、weight decay 0、momentum 0、centered false。Pascal の引数の順序は
  tch-rs の struct に従い、`build(vs, lr)` と同じく `LR` を先頭にします。
- **`ClipGradValue(Max)`**（`clip_grad_value`）：学習対象の各要素について
  `G := T.grad; if G.Defined then G.clamp_(-Max, Max)`。
  **`ClipGradNorm(Max)`**（`clip_grad_norm`）：定義済みの各勾配の `G.norm`（引数なしの
  生成 op `norm`。平坦化した勾配の2ノルム）を集める。1つもなければ終了（tch-rs は空の
  リストの `stack` で panic する）。`Total := TTensor.stack(Norms, 0).norm.DoubleValue([])`、
  `Coef := Max / (Total + 1e-6)`、`Coef < 1` なら定義済みの各勾配に
  `G.mul_scalar_(Coef)`。どちらも `NoGradGuard` の下で実行します（tch-rs はノルム版だけ。
  PyTorch の `clip_grad_value_` も `@torch.no_grad()`）。tch-rs の `tests/nn_tests.rs`
  （`gradient_clip_test`）から手計算できる数値：`v1 = [1, 1]`、`v2 = [1]`、
  `v3 = [-1, -1]`、`loss = sum(cat([v1, 2·v2, 2·v3])²)` → 勾配は `[2, 2]`、`[8]`、
  `[-8, -8]`。`ClipGradValue(4)` → `[2, 2]`、`[4]`、`[-4, -4]`。`ClipGradNorm(1)`：
  全体のノルム `sqrt(4+4+64+64+64) = sqrt(200) = 14.1421`、係数 `0.070711` →
  `[0.1414, 0.1414]`、`[0.5657]`、`[-0.5657, -0.5657]`（小数4桁に丸め）。
- **セッター：** `SetLearningRate`（デフォルトと全グループ）、`SetLearningRateGroup`、
  `SetMomentum`（Adam/AdamW では β₁。libtch の `ato_set_momentum` のとおり）、
  `SetMomentumGroup`、`SetWeightDecay`、`SetWeightDecayGroup`。毎回 `TchCheck`。
  **`torch_api.cpp` を読んでいて見つけたフォークのパッチ：** `ato_set_momentum_group` の
  SGD の分岐が `} else if` ではなく `} if (auto sgd = ...)` になっているため、Adam、
  AdamW、RMSprop では値を設定したあとに `"unexpected optimizer"` を投げてしまいます
  （上流 tch-rs のバグ）。フォークでは `} else if` に直します（1トークン、
  `torch_api.cpp` の 564 行目）。**(H16)**：パッチ前のビルドでは Adam のオプティマイザへの
  `SetMomentumGroup(0, …)` が `ETorchNativeError('unexpected optimizer')` を投げ、
  パッチ後は投げない。
  **(H16) 解決**（ローカルセッション3）：パッチ前のソースのビルドに対して
  `build/nn_test h16` は「unexpected optimizer」で失敗し、パッチ後の `build/libtch` では通る。
- **`BackwardStep(Loss)`**：遅延追加、`ato_zero_grad`、`Loss.Backward`、`ato_step`。
  `BackwardStepClip` / `BackwardStepClipNorm` は backward と step の間にクリッピングを
  挟みます。`TrainableVariables` はストアのものを返します。`Destroy` は `ato_free` を
  呼び、ストアには触れません。パラメータグループは C++ の `Tensor` のコピーを持つので、
  ストアを先に解放しても libtorch のメモリは安全ですが、遅延追加が解放済みの `TVarStore`
  を参照してしまいます：オプティマイザをストアより先に解放すること（ドキュメントに書く）。

#### 4.6.5 マイルストーン2のレイヤー

対象：`TLinear`、`TConv2D`、`TBatchNorm`（Nd 1/2/3）、`TLayerNorm`、`TEmbedding`、
関数としての dropout（ユーザーの `Forward` の中で `X.dropout(P, Train)`、または
`TTensorFuncT` で `Seq.AddFn(@Drop)`）、それに `TFuncModule` と `TSequential`（4.6.6）。
同じ型でマイルストーン3に回すもの：`TConv1D` / `TConv3D` と次元ごとの `ConvConfigND`、
`TConvTranspose`、`TGroupNorm`（`group_norm` 上の `TLayerNorm` と同一）、
`TLSTM` / `TGRU`（`lstm` / `gru` は `out` 引数を持ち、状態の record と `ZeroState` が
要る）、`Id` — いずれも 4.6.10（設計セッション7）で設計済み。変数名、形、
デフォルトの初期化、各 `Forward` が呼ぶ生成 op は、すべて
名前を挙げた tch-rs のファイルから：

| レイヤー（tch-rs のファイル） | 変数：名前 `[形]`、デフォルトの初期化 | 設定のデフォルト | `Forward` |
|---|---|---|---|
| `TLinear`（`linear.rs`） | `weight [Out, In]` KaimingUniform。`Bias` なら `bias [Out]`：`HasBsInit` なら `BsInit`、そうでなければ `Uniform(-1/sqrt(In), 1/sqrt(In))` | `WsInit` KaimingUniform、`HasBsInit` False、`Bias` True | `X.linear(Weight, Bias)` — `atg_linear`、tch-rs と同じ。tch-rs の `matches_pytorch` テストは `linear` が `matmul(ws.tr()) + bs` と同等以上に PyTorch に近いことを示している。`Bias.IsNil` なら `TensorOption` の none |
| `TConv2D`（`conv.rs`、スカラーの `ConvConfig`） | `bias [Out]` `BsInit`（tch-rs と同じく先に登録）。`weight [Out, In div Groups, K, K]` `WsInit` | `Stride` 1、`Padding` 0、`Dilation` 1、`Groups` 1、`Bias` True、`WsInit` KaimingUniform、`BsInit` Constant(0)、`PaddingMode` pmZeros | `pmZeros`：`X.conv2d(Weight, Bias, [S, S], [P, P], [D, D], Groups)`。他のモード：`X.pad(FPadTwice, Mode, NoDouble).conv2d(Weight, Bias, [S, S], [0, 0], [D, D], Groups)`。`FPadTwice = [P, P, P, P]`（逆順の padding を2回繰り返したもの）、`Mode` は `'reflect'` / `'replicate'` / `'circular'`（`PaddingMode::to_string`） |
| `TBatchNorm`（`batch_norm.rs`） | `Affine` なら `weight [Out]` `WsInit`、`bias [Out]` `BsInit`。`running_mean [Out]` `ZerosNoTrain`。`running_var [Out]` `OnesNoTrain`（この登録順） | `CudnnEnabled` True、`Eps` 1e-5、`Momentum` 0.1、`Affine` True、`WsInit` Uniform(0, 1)、`BsInit` Constant(0) | まず次元チェック（Nd = 1：入力は 2-D か 3-D。それ以外は `Nd + 2` 次元。tch-rs の文言で `ETorchError`）、その後 `X.batch_norm(Weight, Bias, RunningMean, RunningVar, Train, Momentum, Eps, CudnnEnabled)`。マイルストーン2で唯一 `Train` を使うレイヤー |
| `TLayerNorm`（`layer_norm.rs`） | `ElementwiseAffine` なら `weight [NormalizedShape]` `WsInit`、`bias [NormalizedShape]` `BsInit` | `CudnnEnabled` True、`Eps` 1e-5、`ElementwiseAffine` True、`WsInit` Constant(1)、`BsInit` Constant(0) | `X.layer_norm(NormalizedShape, Weight, Bias, Eps, CudnnEnabled)` |
| `TEmbedding`（`sparse.rs`） | `weight [NumEmbeddings, EmbeddingDim]` `WsInit` | `Sparse` False、`ScaleGradByFreq` False、`WsInit` Randn(0, 1)、`PaddingIdx` -1 | `TTensor.embedding(Weight, X, PaddingIdx, ScaleGradByFreq, Sparse)`（クラス関数：レシーバの規則で `weight` が第1引数になった） |

各設定には `Default*Config` 関数があり、各レイヤーにはコンストラクタが2つ（設定あり・
なし）あります。変数名は tch-rs のものなので、`.ot` ファイルは相互に使えます。

#### 4.6.6 `TSequential` と `TFuncModule`、`sequential.rs` と `func.rs` より

- **コンテナ：** `TModuleList = specialize TObjectList<TModule>`（`Generics.Collections`、
  `OwnsObjects` はデフォルトで True）。計測（M2）：デフォルト引数付きの仮想メソッドを持つ
  抽象基底クラスで 3.2.2 上でコンパイルでき、`for M in FLayers` が動き、`Free` が子を
  解放する。`Contnrs.TObjectList` も動くが `Items[I]` のたびにキャストが要る。名前の
  索引ですでに `Generics.Collections` を使うので、型付きのリストを使います。
- **`AddFn` は1つの名前に4つのオーバーロード：** `TTensorFunc`、`TTensorFuncT`、
  `TTensorMethod`（`of object`）、`TTensorMethodT`。計測（M3、objfpc）：
  `AddFn(@PlainFunc)`、`AddFn(@PlainFuncT)`、`AddFn(@Obj.Method)`、`AddFn(@Obj.MethodT)`
  はそれぞれ対応する手続き型に曖昧さなく解決し、返された `Self` を通したチェーンも動く。
  それぞれ `TFuncModule` を追加します。`TFuncModule` は4つのポインタのどれか1つを持ち、
  単体でも作れます。メソッドポインタの場合、オブジェクトは sequential より長く生きている
  必要があります（ドキュメントに書く）。
- **`Forward(X, Train)`** は `Train` を全ての子に渡します（`SequentialT::forward_t`。
  `TModule` がすでに2つのトレイトをまとめているので、1つのクラスで tch-rs の
  `Sequential` と `SequentialT` の両方を担う）。空の sequential は `X` を返します
  （同じハンドルの record コピー。tch-rs の `shallow_clone`）。`ForwardAll(X, Train, N)`
  は最初の `N` レイヤーの出力を返します（`forward_all_t`。`N = -1` は全部）。
- `Add(M)` は所有権を取り、チェーン用に `Self` を返します。`Count` とデフォルトの
  `Items[I]` プロパティで子にアクセスできます。

#### 4.6.7 ユニット構成、L2 の追加、マイルストーン2のテスト

- **ユニットは `TchNN.pas` 1つ**（`TchOptim` はなし）：`TOptimizer` は
  `TVarStore.FTrainable` / `FVars` を読みますが、同じユニット内なら private に
  アクセスできます（4.5）。`TchJit.pas` は JIT のマイルストーンで追加（4.6.8）。
- **`TchSys.pas` への追加：** `PTorchOptimizer`、`TTorchLoadCallback`、`at_save_multi`、
  `at_load_callback_with_device`、`at_loadz_callback_with_device`、4.6.4 の14個の `ato_*`
  （13個 + `ato_free`）。**`Tch.pas` への追加：** `at_copy_` を包む
  `TTensor.CopyFrom(const Src: TTensor)`。**フォークのパッチ：** `ato_set_momentum_group`
  の `} else if`（4.6.4）、その後 `libtch.so` を再ビルド。
- **`tests/nn/nn_test.pas`**（本物の libtorch。ビルド方法はヘッダ）、手計算できる数値。
  この一覧はローカルセッション3以降のテストの現状で、クラウドセッション5で確認したもの
  （クラウドセッション4の一覧にない追加は採用。まだ足していないものが 7d に1つ）：
  1. 初期化：`[4, 16]` への `KaimingUniform` はすべての値が `±sqrt(6/16) = ±0.6124` の
     範囲にあり、平均は 0 から 0.1 以内。`[8, 3, 5, 5]` の `FanForDims` = 75（in）／200
     （out）。`Constant(3)` はすべて 3、`Constant(1)` は要求した kind を保つ。`Orthogonal`
     の `[3, 5]`、`[5, 3]`、gain 2 の `[2, 3, 2]`（H14）、結果の形 = `Dims`、1次元は例外。
     `Apply`（H17）：ガードなしの学習対象の葉への `fill_` は例外、すべての `TInit` の
     `Apply` は成功し、変数は引き続き grad を要求する。
  2. ストア：`Sub` と `/` を通した `a.b.t2` という命名。`Sub('x.y')`、`Zeros('a.b', …)`、
     `Zeros('a|b', …)`、`Sub('')` は例外で、何も追加しない。`t123` を2回追加すると
     `Count = 4` で、2つ目の名前は `a.b.ccc.t123__3`（tch-rs の `save_and_load_var_store`）。
     `TryGet` は完全名で、パス経由で、存在しない名前で。既存の名前への `GetOrAdd` は
     要求した次元にかかわらず `[3, 1, 4]` のテンソルを返す。`ZerosNoTrain` は grad を
     要求しない。`TrainableVariables` は登録順。`Freeze` → `RequiresGrad = False`、
     `Unfreeze` → True。`VarCopy`。`Freeze; Step` は何も変えない（H18）。`TLinear` と SGD
     オプティマイザを作った後の `SetKind(kDouble)`：レイヤーの `Weight` は `kDouble`、
     ステップは手計算の `[0.9, 0.8]` を与え `kDouble` のまま（H13）。`SetDevice(Cpu)` は
     ハンドルを使える状態に保つ。部分木 `lin` への `TVarPath.SetKind(kFloat)` は `weight`
     と `bias` だけを変換する。
  3. 線形回帰（tch-rs の `optimizer_test`）：`xs = 1..14` を `[14, 1]` の float、
     `ys = 0.42·xs + 1.337`、両方の初期化を `Constant(0)` にした `TLinear(1, 1)`、
     `Sgd(VS, 0)` の後に `SetLearningRate(1e-2)`。最初の `mse_loss` =
     `mean((0.42x + 1.337)²) = (0.1764·1015 + 1.12308·105 + 14·1.787569) / 14 = 22.99967`。
     **1回** のステップ後：`w = 0.80955`、`b = 0.08974`（`∂/∂w = -2/14·Σx·y =
     -2/14·566.685 = -80.955`、`∂/∂b = -2/14·Σy = -2/14·62.818 = -8.974`）、float32 で
     1e-3 以内。49回の `BackwardStep` の後は loss < 0.25。両方に `Apply(Constant(0))`
     すると loss は初期値に戻る。`SetLearningRate(1e-10)` でさらに49ステップ：tch-rs の
     判定は `final - initial < 1e-5`（**符号付き**）。絶対値で 1e-5 未満は成り立たない：
     勾配はほぼ一定なので、1ステップごとに loss は `lr·‖g‖² = 1e-10 · (80.955² + 8.974²)
     = 1e-10 · 6634.2 = 6.634e-7` 下がり、49ステップで `3.25e-5`。実測は `-3.24e-5`
     （ローカルセッション3）で、23 付近の loss の float32 の分解能（ulp 1.9e-6）の範囲内。
     テストは両方を確認する：tch-rs の符号付きの上限と `Near(L - Initial, -3.25e-5,
     5e-6)`。計算はクラウドセッション5で確認。加えて `Bias False` の `TLinear` は変数を
     1つ登録し、`Bias.IsNil`、forward は `[4, 3] → [4, 2]`。
  4. Adam：パラメータ `p = [1, -2]`、`loss = sum(p²)`、`Adam(VS, 0.1)`、1回の
     `BackwardStep` → `p = [0.9, -1.9]`、1e-6 以内：Adam の最初のステップは
     `-lr·g/(|g| + eps)`、すなわち `lr·sign(g)`。同じ `p` に既定の weight decay 0.01 の
     AdamW：PyTorch は先に減衰 `p ← p·(1 - lr·wd) = [0.999, -1.998]`、次に Adam の
     ステップ `lr·sign(g)` → **`[0.899, -1.898]`**、1e-6 以内（ローカルセッション3の追加、
     計算を確認）。RmsProp のステップと `SetMomentum`、SGD の `SetMomentumGroup`、Adam の
     setter は例外を出さない。Adam への `SetMomentumGroup(0, 0.8)` はパッチ済みのフォーク
     では例外を出さない（H16。`build/nn_test h16` はこの確認だけを実行する）。
  5. 勾配クリッピング：4.6.4 の数値。勾配がないときの `ClipGradNorm` は静かに抜ける。
     `BackwardStepClip(4)` はクリップした勾配でステップする（`v2 = 1 - 4`）。
  6. 遅延追加：オプティマイザを作ってから学習対象の変数を追加し、1回の `BackwardStep` で
     それが変わり、`TrainableVariables` にも見える。グループ1の変数は
     `SetLearningRateGroup(1, 0)` で動かず、グループ0は動く。
  7. Save → Load。構成は tch-rs の `tests/var_store.rs` のもの：*full*
     （`save_and_load_var_store`：`a.b.t2 [3]`、`t1 [4]`、`a.b.ccc.t123 [3]` を**2回** →
     `t123__3`）、*add*（`t1`、`t123` 1回）、*add_partial*（`a.b.t2`、`t1`、`t123` 1回）。
     (a) full 構成のストア2つ。`NoGradGuard` の下で `u += 42`、`v *= 2`。一時的な `.ot`
     に `Save`。2つ目のストアが `Load`：平均は 42 と 2、入れ子の名前は `|` を通して往復し、
     読み込んだ変数は引き続き grad を要求する。(b) 余分な `extra.x` を持つ full 構成の
     ストアへの `Load` はその名前を含む `ETorchError` で、ストアは変わらない（tch-rs の
     `save_and_load_var_store_incomplete_file`、`should_panic`）。そこでの `LoadPartial` は
     `['extra.x']` を返し、残りを読み込む。(c) `SetFloatKind(kHalf)` して保存し、新しい
     float のストアに `Load` → 変数は `kHalf` で平均は同じ（H15）。(d) tch-rs の
     `save_and_load_partial_var_store_incomplete_file`：ストア1は *add*、ストア2は
     *add_partial*、`u1 += 42`、`LoadPartial` は **`['a.b.t2']`** を返し、`u2 = 42`、`v2`
     は手つかず（1）。クラウドセッション4はこの場合を full 構成で書いたが、それだと
     `a.b.ccc.t123__3` も欠ける。訂正（ローカルセッション3／クラウドセッション5）。
     **追加済み**（ローカルセッション4）、この tch-rs テストの後半：ストア1に
     `SetFloatKind(kHalf)` して保存し、新しい float の *add_partial* ストアに
     `LoadPartial` → `u2` は `kHalf` で平均 42、欠けている `v2` は `kFloat` のまま平均 1、
     missing = `['a.b.t2']`。(e) `.safetensors` への `Save` は `ETorchError`。存在しない
     ファイルの `Load` は `ETorchNativeError`。(f) `CopyFrom`：名前でコピーする
     （`u2 = 7`）。コピー元にない名前はその名前を含む `ETorchError`。
  8. レイヤー：`Padding 1`、`Bias False`、重みをすべて 1 にした `TConv2D(1, 1, 3)` を
     `[[1, 2], [3, 4]]`（NCHW `[1, 1, 2, 2]`）に適用：zeros padding → `[10, 10, 10, 10]`、
     reflect → `[27, 24, 21, 18]`、circular → `[27, 24, 21, 18]`、replicate →
     `[18, 21, 24, 27]`（tch-rs の `conv` テスト）。`TConv2D(4, 6, 3)` は `c.bias`、次に
     `c.weight` を登録し、`[2, 4, 8, 8] → [2, 6, 6, 6]`。`TBatchNorm(1, 40)` は4つの変数を
     登録する（`running_mean`、`running_var` は学習対象でない。`Affine` False なら2つ）、
     `[10, 40]` への train モードの forward が動いて running stats が変わり、
     `[10, 40, 3, 3]` は例外。`BatchNorm2d` の eval forward は形を保つ。`TLayerNorm([2])`
     のフィット（tch-rs の `layer_norm_parameters_test`：`xs = [[42, 42], [42, 24]]`、SGD
     lr 1.0、初期 loss > 1、49ステップ後 < 0.25、リセットで初期 loss に戻る）。
     `TEmbedding(10, 4)` を `randint(10, [5, 7])` に適用すると `[5, 7, 4]` で、`PaddingIdx`
     の行は `Weight` の行と等しい。`TLinear` は `weight` と `bias` を登録する（`Count` 2、
     バイアスなしなら 1）。
  9. Sequential：空の sequential は同じハンドルを返し、その `ForwardAll` は `[X]`。
     `[1, -1, 2]` への `AddFn(@Relu)` → `[1, 0, 2]`。4つの `AddFn` オーバーロードの
     チェーン（関数、関数+Train、メソッド、メソッド+Train。`Factor = 3` のオブジェクト）：
     `Forward(X, True)` = `[9, 0, 18]` で `TTensorFuncT` は `Train` を見る、`Forward(X)` =
     `[3, 0, 6]`。`ForwardAll` はレイヤーごとに1つのテンソルを返し、`ForwardAll(N = 2)`
     は2つ。`Items[0] is TFuncModule`。`TLinear` 2つと ReLU の MLP：`[4, 3] → [4, 2]`、
     変数は4つ。
  10. オプティマイザ、レイヤー、ストアをこの順に解放したあと（間に MLP への Adam の
      5ステップの学習ループ）、`TchLiveHandles` が開始時の値に戻り、最後には 0。
- **mock で確認できること**（`tests/mock/mock_tch.c` を拡張。配線のみ）：カウンタ構造体
  上の `ato_*` スタブ（グループごとの追加数、zero_grad と step の回数を `mock_opt_*`
  として公開）→ 構築と遅延追加の回数。名前を記録する `at_save_multi` → `.`→`|` の変換と
  順序。`a|b|t2`、`t1` と mock のテンソルでコールバックを呼ぶ
  `at_load_callback_with_device`、および最初のコールバックの後でエラースロットを設定する
  ファイル名 → トランポリン、`|`→`.` の変換、エラーの順序。`atg_set_requires_grad`、
  `atg_set_data`、`atg_fill_`、`atg_uniform_`（形だけ）→ ストアの帳簿、`Freeze`、`TInit`
  の配線。`add_scalar` 上の `TSequential` / `TFuncModule` のチェーン。そこで確認できない
  もの：数値に関わるすべて（分布、QR、`linear`、`conv2d`、`batch_norm`、backward、
  `step`）。これは上の本物の libtorch のテストで確認します。

#### 4.6.8 JIT（`TchJit`）、`wrappers/jit.rs`、`nn/mod.rs`、`tests/jit_tests.rs`、`torch_api.cpp` より

クラウドセッション5で、同じ tch-rs のコミットに対して設計しました。フォークの
`torch_api.{h,cpp}` の `atm_*`/`ati_*` の領域は上流とバイト単位で同一です（`diff` で
確認）。本物の libtorch が必要なものは **(H19)**〜**(H25)**、クラウドの FPC 3.2.2 で
計測したものは*実測*と記します。下のインターフェース全体（`TIValue`、`TJitModule`、
`TTrainableJitModule`、`Jit*` 関数）は、本体をスタブにした使い捨てのユニットとして本物の
`Tch`/`TchNN` に対してコンパイルしました（*実測*、HANDOFF の M8）：エラーはなく、テストが
使う呼び出し形は解決します。

**範囲（マイルストーン3）。** TorchScript モジュールの読み込み、実行、保存、トレース。
`ati_tag` の値 0〜13（None、Tensor、Double、Int、Bool、Tuple、IntList、DoubleList、
BoolList、String、TensorList、GenericList、GenericDict）に対する `TIValue` と、入力専用の
2つの値 StringList と Device。`TTrainableJitModule`。TorchScript のソースから作る
スクリプト化モジュール（下記の3関数のフォーク追加）。プロファイリング／fuser の
スイッチ。**先送り**（L2 の行も含めて）：TorchScript のオブジェクト（タグ 14 —
`ati_object_method_`、`ati_object_getattr_`、`ati_clone`、`atm_create_class_`。tch-rs の
`Object`、テスト `jit6` と `jit_double_free`、資材 `foo6.pt`/`foo7.pt`）。オブジェクトは
C の `IValue` を生かし続けるので `TTensorHandle` のような参照カウント付きハンドルが必要で、
微調整の用途には要らないからです — **その後 4.6.9 で設計済み**（設計セッション6）。
それと `atm_fuser_cuda_*`（CUDA と一緒に）— **その後 4.8 で設計済み**（設計セッション8、
`JitFuserCudaEnabled`/`SetJitFuserCudaEnabled`）。

**L2（`TchSys.pas`）。** `PTorchModule = Pointer`（`torch::jit::script::Module*`）、
`PTorchIValue = Pointer`（`torch::jit::IValue*`）、`PPTorchIValue = ^PTorchIValue`。
external は `torch_api.h` のとおりで、型は2章のもの（`int` に `cint`、C++ の `bool` に
`Boolean`、`csize_t`、`PChar`、`PPChar`、`PInt64`、`PDouble`）。`atm_named_parameters` は
load コールバックと同じシグネチャなので `TTorchLoadCallback` を再利用します：

```pascal
function  atm_load(filename: PChar): PTorchModule; cdecl; external LibTch;
function  atm_load_on_device(filename: PChar; device: cint): PTorchModule; cdecl; external LibTch;
function  atm_load_str(data: PChar; sz: csize_t): PTorchModule; cdecl; external LibTch;
function  atm_load_str_on_device(data: PChar; sz: csize_t; device: cint): PTorchModule; cdecl; external LibTch;
function  atm_forward(m: PTorchModule; tensors: PPTorchTensor; ntensors: cint): PTorchTensor; cdecl; external LibTch;
function  atm_forward_(m: PTorchModule; ivalues: PPTorchIValue; nivalues: cint): PTorchIValue; cdecl; external LibTch;
function  atm_method(m: PTorchModule; method_name: PChar; tensors: PPTorchTensor; ntensors: cint): PTorchTensor; cdecl; external LibTch;
function  atm_method_(m: PTorchModule; method_name: PChar; ivalues: PPTorchIValue; nivalues: cint): PTorchIValue; cdecl; external LibTch;
procedure atm_eval(m: PTorchModule); cdecl; external LibTch;
procedure atm_train(m: PTorchModule); cdecl; external LibTch;
procedure atm_free(m: PTorchModule); cdecl; external LibTch;
procedure atm_to(m: PTorchModule; device, dtype: cint; non_blocking: Boolean); cdecl; external LibTch;
procedure atm_save(m: PTorchModule; filename: PChar); cdecl; external LibTch;
function  atm_get_profiling_mode: cint; cdecl; external LibTch;
procedure atm_set_profiling_mode(b: cint); cdecl; external LibTch;
procedure atm_set_tensor_expr_fuser_enabled(b: cint); cdecl; external LibTch;
function  atm_get_tensor_expr_fuser_enabled: Boolean; cdecl; external LibTch;
procedure at_set_graph_executor_optimize(b: Boolean); cdecl; external LibTch;
procedure atm_named_parameters(m: PTorchModule; data: Pointer; f: TTorchLoadCallback); cdecl; external LibTch;
function  atm_create_for_tracing(modl_name: PChar; inputs: PPTorchTensor; ninputs: cint): PTorchModule; cdecl; external LibTch;
procedure atm_end_tracing(m: PTorchModule; fn_name: PChar; outputs: PPTorchTensor; noutputs: cint); cdecl; external LibTch;
// フォークの追加（下記）
function  atm_create(modl_name: PChar): PTorchModule; cdecl; external LibTch;
procedure atm_register_parameter(m: PTorchModule; name: PChar; t: PTorchTensor; is_buffer: Boolean); cdecl; external LibTch;
procedure atm_define(m: PTorchModule; src: PChar); cdecl; external LibTch;

function  ati_none: PTorchIValue; cdecl; external LibTch;
function  ati_tensor(t: PTorchTensor): PTorchIValue; cdecl; external LibTch;
function  ati_int(v: Int64): PTorchIValue; cdecl; external LibTch;
function  ati_double(v: Double): PTorchIValue; cdecl; external LibTch;
function  ati_bool(v: cint): PTorchIValue; cdecl; external LibTch;
function  ati_string(s: PChar): PTorchIValue; cdecl; external LibTch;
function  ati_tuple(ivalues: PPTorchIValue; n: cint): PTorchIValue; cdecl; external LibTch;
function  ati_generic_list(ivalues: PPTorchIValue; n: cint): PTorchIValue; cdecl; external LibTch;
function  ati_generic_dict(ivalues: PPTorchIValue; npairs: cint): PTorchIValue; cdecl; external LibTch;  // 2*npairs 個の ivalue：k0, v0, k1, v1, ...
function  ati_int_list(v: PInt64; n: cint): PTorchIValue; cdecl; external LibTch;
function  ati_double_list(v: PDouble; n: cint): PTorchIValue; cdecl; external LibTch;
function  ati_bool_list(v: PChar; n: cint): PTorchIValue; cdecl; external LibTch;   // 0/1 のバイト
function  ati_string_list(v: PPChar; n: cint): PTorchIValue; cdecl; external LibTch;
function  ati_tensor_list(v: PPTorchTensor; n: cint): PTorchIValue; cdecl; external LibTch;
function  ati_device(device: cint): PTorchIValue; cdecl; external LibTch;
function  ati_tag(i: PTorchIValue): cint; cdecl; external LibTch;
function  ati_to_tensor(i: PTorchIValue): PTorchTensor; cdecl; external LibTch;
function  ati_to_int(i: PTorchIValue): Int64; cdecl; external LibTch;
function  ati_to_double(i: PTorchIValue): Double; cdecl; external LibTch;
function  ati_to_string(i: PTorchIValue): PChar; cdecl; external LibTch;   // strdup 済み：tchf_free
function  ati_to_bool(i: PTorchIValue): cint; cdecl; external LibTch;
function  ati_length(i: PTorchIValue): cint; cdecl; external LibTch;
function  ati_tuple_length(i: PTorchIValue): cint; cdecl; external LibTch;
procedure ati_to_tuple(i: PTorchIValue; outputs: PPTorchIValue; n: cint); cdecl; external LibTch;
procedure ati_to_generic_list(i: PTorchIValue; outputs: PPTorchIValue; n: cint); cdecl; external LibTch;
procedure ati_to_generic_dict(i: PTorchIValue; outputs: PPTorchIValue; npairs: cint); cdecl; external LibTch;  // 2*npairs 個を書く
procedure ati_to_int_list(i: PTorchIValue; outputs: PInt64; n: cint); cdecl; external LibTch;
procedure ati_to_double_list(i: PTorchIValue; outputs: PDouble; n: cint); cdecl; external LibTch;
procedure ati_to_bool_list(i: PTorchIValue; outputs: PChar; n: cint); cdecl; external LibTch;
procedure ati_to_tensor_list(i: PTorchIValue; outputs: PPTorchTensor; n: cint); cdecl; external LibTch;
procedure ati_free(i: PTorchIValue); cdecl; external LibTch;
```

`atm_free`、`ati_free`、`at_set_graph_executor_optimize` 以外はすべて `PROTECT` 付きなので
毎回 `TchCheck`。コンストラクタはエラー時に nil を返し、ポインタを見る前に `TchCheck` が
例外にします（`ati_to_bool`/`ati_to_int` はエラー時に -1 を返すが、これも同じ検査で
拾う）。C に渡すすべての `tensor` は先に `TchCheckArg` を通します（4.2）。宣言しないもの
（4.6.3 の「使わない入口は宣言しない」規則）：`atm_create_class_`、`ati_object_method_`、
`ati_object_getattr_`、`ati_clone`、`atm_fuser_cuda_set_enabled`、
`atm_fuser_cuda_is_enabled`。

**フォークの追加**（3関数。autocast と `ato_set_momentum_group` の修正と同じくパッチ。
`torch_api.h` では `atm_create_for_tracing` の隣に宣言）：

```c
module atm_create(char *modl_name) {
  PROTECT(auto m = new torch::jit::script::Module(modl_name);
          m->register_attribute("training", c10::BoolType::get(), true);
          return m;)
  return nullptr;
}
void atm_register_parameter(module m, char *name, tensor t, bool is_buffer) {
  PROTECT(m->register_parameter(name, *t, is_buffer);)
}
void atm_define(module m, char *src) { PROTECT(m->define(src);) }
```

理由：tch-rs のテスト用モジュールは `tests/create_jit_models.py` で作られています。
Python なしで、パラメータを持つモジュール、テンソル以外のシグネチャのメソッド、
`training` フラグをテスト時に作る唯一の方法が `torch::jit::Module::define` — libtorch の
TorchScript コンパイラ、公開 C++ API — です。`atm_create_for_tracing` ではできません：
トレース中の関数内で使ったテンソルは定数になり、grad を要求するテンソルはそもそも定数
としてトレースできません。`training` を登録するのは、`Module::train()` がこの属性のない
モジュールを黙って飛ばし（`module.cpp`。assert は上流でコメントアウト）、メソッド内の
`self.training` にも必要だからです。Python の `ScriptModule` も持っています。**(H20)**：
ビルドでき、下のテストのとおりに振る舞う。これは機能でもあります（Pascal からの
TorchScript）。`CreateEmpty`/`RegisterParameter`/`Define` として公開し、tch-rs からの差異
として明示します。

**`TIValue`。** タグの値は `ati_tag` にある分はそれと同じにします（*実測*：
`TIValueTag(12) = ivGenericList`）ので、`FromC` は範囲検査のあとキャストするだけです。
自分自身の配列は `TTensorArray` と同じく入れ子の型で（H10）：

```pascal
type
  TIValueTag = (ivNone = 0, ivTensor = 1, ivDouble = 2, ivInt = 3, ivBool = 4, ivTuple = 5,
    ivIntList = 6, ivDoubleList = 7, ivBoolList = 8, ivString = 9, ivTensorList = 10,
    ivGenericList = 12, ivGenericDict = 13,
    ivStringList = 20, ivDevice = 21);   // 入力専用：ati_tag の値がない（下記）
  TBooleanArray = array of Boolean;      // これを使う record より前に

  TIValue = record
  public type
    TIValueArray = array of TIValue;
  public
    Tag: TIValueTag;              // Default(TIValue) は None
    Tensor: TTensor;              // ivTensor
    I: Int64;                     // ivInt
    D: Double;                    // ivDouble
    B: Boolean;                   // ivBool
    S: String;                    // ivString
    Ints: TInt64Array;            // ivIntList
    Doubles: TDoubleArray;        // ivDoubleList
    Bools: TBooleanArray;         // ivBoolList
    Strings: TStringArray;        // ivStringList（入力専用）
    Tensors: TTensorArray;        // ivTensorList
    Items: TIValueArray;          // ivTuple, ivGenericList。ivGenericDict: k0, v0, k1, v1, ...（C 側と同じく平坦）
    Dev: TDevice;                 // ivDevice（入力専用）
    class function None: TIValue; static;
    class function FromTensor(const T: TTensor): TIValue; static;
    class function FromInt(V: Int64): TIValue; static;
    class function FromDouble(V: Double): TIValue; static;
    class function FromBool(V: Boolean): TIValue; static;
    class function FromString(const V: String): TIValue; static;
    class function FromDevice(const V: TDevice): TIValue; static;
    class function Tuple(const A: array of TIValue): TIValue; static;
    class function IntList(const A: array of Int64): TIValue; static;      // TIValue の中では Tch の IntList にユニット名が要る
    class function DoubleList(const A: array of Double): TIValue; static;
    class function BoolList(const A: array of Boolean): TIValue; static;
    class function StringList(const A: array of String): TIValue; static;
    class function TensorList(const A: array of TTensor): TIValue; static;
    class function GenericList(const A: array of TIValue): TIValue; static;
    class function GenericDict(const KeysAndValues: array of TIValue): TIValue; static;  // 偶数個。でなければ ETorchError
    class operator :=(const V: TTensor): TIValue;
    class operator :=(V: Int64): TIValue;
    class operator :=(V: Double): TIValue;
    class operator :=(V: Boolean): TIValue;
    class operator :=(const V: String): TIValue;
    function IsNone: Boolean;
    function TagName: String;                                 // 'Tensor', 'Tuple', ...（tch-rs の type_str）
    function AsTensor: TTensor;        // タグが合わなければ ETorchError('ivalue: expected Tensor, got Tuple')
    function AsInt: Int64;  function AsDouble: Double;  function AsBool: Boolean;  function AsString: String;
    function AsIntList: TInt64Array;  function AsDoubleList: TDoubleArray;
    function AsBoolList: TBooleanArray;  function AsTensorList: TTensorArray;
    function Count: Integer;           // タプル／リスト：Length(Items)。辞書：Length(Items) div 2。型付きリスト：その長さ
    function Item(Index: Integer): TIValue;                   // ivTuple または ivGenericList（tch-rs も両方を展開する）
    function Key(Index: Integer): TIValue;  function Value(Index: Integer): TIValue;   // ivGenericDict
    function ToC: PTorchIValue;                               // 新しい C の ivalue。呼び出し側が所有（ati_free）
    class function FromC(P: PTorchIValue): TIValue; static;   // P を消費する（どの経路でも ati_free）
  end;
  TIValueArray = TIValue.TIValueArray;
```

- *実測*（M5）：明示したタグ値を持つ record がコンパイルでき、入れ子にできる。`V := 42`
  / `2.5` / `0.9` / `True` / `'foo'` / record は `:=` 演算子を通して期待どおりのタグになる。
  そして **`array of TIValue` 引数に渡した `[T, 42, 1.5]` は、これらの演算子を通して
  要素ごとに変換される**ので、`M.ForwardIV([X, 42, 1.5])` と書けます。配列からの演算子は
  意図的に置きません（曖昧になる）。名前付きのコンストラクタで足ります。
- 意味論は 4.6 のとおり：コピーは `Items` を共有する。変更しないものとして扱う。
- **`ToC`**（tch-rs の `to_c`）、タグごとに：`ati_none`。`TchCheckArg` の後の
  `ati_tensor(T.Raw)`。`ati_int`/`ati_double`/`ati_bool(Ord(B))`。`ati_string(PChar(S))`。
  型付きリストは record 自身の配列の上で（`ati_bool_list(PChar(@Bools[0]), n)` —
  `SizeOf(Boolean) = 1`、実測、値は 0/1。`ati_string_list` は `Strings` から作った
  `array of PChar` の上で。`ati_tensor_list` は `Raw` の上で、各要素を `TchCheckArg`）。
  `ati_device(Dev.ToC)`。コンテナ：まず子を nil で初期化した `array of PTorchIValue` に
  変換し、`ati_tuple` / `ati_generic_list` / `ati_generic_dict(…, Length(Items) div 2)` で
  作り、`finally` で nil でない子をすべて解放する — C のコンテナは `IValue` をコピーして
  いる（tch-rs も解放する）。奇数個の辞書は `ETorchError`。空の配列は `nil` を渡す
  （空の配列の `@A[0]` は決して使わない。`ati_tuple(nullptr, 0)` は問題ない）。各 `ati_*`
  の後に `TchCheck`。
- **`FromC`**（tch-rs の `from_c`）：`try … finally ati_free(P)`。`nil` は `ETorchError`。
  `Tag := ati_tag(P); TchCheck` — 14 は `ETorchError('ivalue: TorchScript objects are not
  supported yet')`（4.6.9 まで）、列挙の外の他の値も例外（Device に対しては C 自身が "unsupported tag"
  を投げる）。その後：Tensor → `TchCheck` の後の `TTensor.FromRaw(ati_to_tensor(P))`。
  スカラー → `ati_to_int`/`double`/`bool` + `TchCheck`。String → `ati_to_string`、
  `StrPas`、`tchf_free`（`strdup` 済み、2章）。型付きリスト → `N := ati_length`、`N` 個の
  バッファ、`ati_to_*_list(P, PtrOrNil(buf), N)`（BoolList は `Byte` のバッファを通して
  `<> 0`）。TensorList → nil で初期化した `PTorchTensor` のスロット、エラー時は
  `TchCheckSlots`（Tch.pas）、でなければ各要素を `FromRaw`。Tuple / GenericList → `N :=
  ati_tuple_length` / `ati_length`、`ati_to_tuple` / `ati_to_generic_list` を nil で
  初期化した `array of PTorchIValue` に、そして子ごとに `C := Cs[K]; Cs[K] := nil;
  Items[K] := FromC(C)` として、後の変換が例外を投げたときに `finally` がまだ消費して
  いない子を `ati_free` できるようにする。GenericDict → `N := ati_length`（ペア数）、`2·N`
  個のスロット、`ati_to_generic_dict(P, Cs, N)`、同じ消費の仕方。C 側はサイズを検査して
  スロットを書く前に例外を投げ、`new` は途中で `bad_alloc` を出し得るので、スロットは nil
  で初期化する（4.2 の固定 N 出力の規則）。
- **入力専用の値。** `ivStringList` は `c10::List<string>` を作り、`ati_tag` はそれを 12
  と報告します：String 要素の GenericList として戻ってきます（tch-rs の `jit5` が
  まさにそれを期待している）。`ivDevice` には `ati_tag` の値がなく、Device の出力は
  `ati_tag` から `ETorchNativeError` になります。GenericDict のペアは挿入順で戻ります：
  `c10::Dict` は順序を保つハッシュマップで、tch-rs の往復テストもそれに依存しています
  **(H24)**。

**`TJitModule`**（tch-rs の `CModule`）：

```pascal
type
  TTraceFunc   = function(const Inputs: TTensorArray): TTensorArray;
  TTraceMethod = function(const Inputs: TTensorArray): TTensorArray of object;

  TJitModule = class
  private
    FModule: PTorchModule;
    {$push}{$warn 3018 off}
    constructor CreateRaw(M: PTorchModule);          // 所有権を取る。M <> nil（TchCheck 済み）
    {$pop}
  public
    constructor Create(const FileName: String); overload;                           // atm_load
    constructor Create(const FileName: String; const ADevice: TDevice); overload;   // atm_load_on_device
    constructor CreateFromStream(S: TStream); overload;                             // atm_load_str
    constructor CreateFromStream(S: TStream; const ADevice: TDevice); overload;     // atm_load_str_on_device
    constructor CreateEmpty(const ModuleName: String);                              // atm_create（フォーク）
    class function CreateByTracing(const ModuleName, FnName: String;
      const Inputs: array of TTensor; Fn: TTraceFunc): TJitModule; overload; static;
    class function CreateByTracing(const ModuleName, FnName: String;
      const Inputs: array of TTensor; Fn: TTraceMethod): TJitModule; overload; static;
    destructor Destroy; override;                                                   // atm_free
    function Forward(const Inputs: array of TTensor): TTensor;                      // atm_forward
    function ForwardIV(const Inputs: array of TIValue): TIValue;                    // atm_forward_
    function Method(const Name: String; const Inputs: array of TTensor): TTensor;   // atm_method
    function MethodIV(const Name: String; const Inputs: array of TIValue): TIValue; // atm_method_
    procedure SetEval;                                                              // atm_eval
    procedure SetTrain;                                                             // atm_train
    procedure ToDevice(const ADevice: TDevice; AKind: TKind; NonBlocking: Boolean = False);  // atm_to
    procedure Save(const FileName: String);                                         // atm_save
    function NamedParameters: TNamedTensorArray;                                    // atm_named_parameters
    procedure RegisterParameter(const Name: String; const T: TTensor; IsBuffer: Boolean = False);  // フォーク
    procedure Define(const Source: String);                                         // フォーク：TorchScript のソース
    property Raw: PTorchModule read FModule;
  end;

function  JitGetProfilingMode: Boolean;              procedure JitSetProfilingMode(B: Boolean);
function  JitGetTensorExprFuserEnabled: Boolean;     procedure JitSetTensorExprFuserEnabled(B: Boolean);
procedure JitSetGraphExecutorOptimize(B: Boolean);
```

- `Forward`/`Method` はテンソルを、`ForwardIV`/`MethodIV` は `TIValue` を返します。これは
  C 側の分け方です（タプルの結果に対して `atm_forward` は "forward did not return a
  tensor" を投げる。下のテストはこのメッセージに依存する）。テンソルの入力：各要素を
  `TchCheckArg`、`Raw` の `array of PTorchTensor`、空なら `nil`。IValue の入力：各要素を
  `ToC` で nil 初期化した配列に、呼び出し、`TchCheck`、`FromC(結果)`。入力は `finally` で
  解放します（tch-rs も呼び出しの後に解放する）。
- `NamedParameters`：4.6.3 の `LoadCallback` とまったく同じ `cdecl` のトランポリンを、
  `TchJit` 内の独自の `TParamContext` で（`TchNN` の private な補助を公開するより15行の
  重複を選ぶ）。名前はそのまま（`|`→`.` は残すが無害）。呼び出しの後に無条件で
  `get_and_reset_last_err`、取得した例外は再 raise。バッファはパラメータではない：
  `foo.pt` の `value` は空のリストになります。
- `CreateByTracing`：`atm_create_for_tracing(Name, inputs)`、`TchCheck`。`Outputs :=
  Fn(TTensorArray としての Inputs)`（同じハンドル。tch-rs もクロージャに `inputs` を
  渡す）。`atm_end_tracing(M, FnName, outputs, n)`、`TchCheck`。`Fn` が例外を投げたら：
  libtorch のトレース状態はグローバルで、libtch に「中断」の入口はないので、ハンドラは
  `atm_end_tracing(M, PChar(FnName), nil, 0)` を呼び、エラースロットを空にし
  （`get_and_reset_last_err` → `tchf_free`）、`atm_free(M)` して再 raise します — できる
  範囲で、**(H23)**：その後の `CreateByTracing` が動く（でなければ "cannot nest tracing
  calls"）。
- `CreateFromStream` は `S.Size - S.Position` バイトを `TBytes` に読み、
  `atm_load_str(@Buf[0], Length(Buf))` を呼びます（C は `std::string` にコピーする）。
  空のストリームは呼び出しの前に `ETorchError`。
- `ToDevice` は tch-rs の `to(device, kind, non_blocking)` を保ちます：`atm_to` の dtype
  は省略できません。
- `Define(Source)`：TorchScript のテキスト。`def` は0桁目、本体は4スペース、行は `#10`
  で結合。1回の呼び出しに複数の `def` があってもよい。コンパイルエラーは
  `ETorchNativeError` として届きます。
- グローバル：`atm_get_profiling_mode <> 0`、`atm_set_profiling_mode(Ord(B))`、
  `atm_set_tensor_expr_fuser_enabled(Ord(B))`、`atm_get_tensor_expr_fuser_enabled`
  （C++ の `bool` → `Boolean`）、`at_set_graph_executor_optimize(B)`（`PROTECT` なし。
  検査するものはない）。

**`TTrainableJitModule`**（tch-rs の `TrainableCModule`、`nn/mod.rs`）：

```pascal
type
  TJitMode = (jmUnknown, jmTrain, jmEval);
  TTrainableJitModule = class(TModule)
  private
    FInner: TJitModule;           // 所有する
    FMode: TJitMode;
    procedure RegisterInto(const P: TVarPath);
  public
    constructor Create(const FileName: String; const P: TVarPath);       // atm_load_on_device(P.Device)
    constructor CreateFromStream(S: TStream; const P: TVarPath);
    destructor Destroy; override;
    function Forward(const X: TTensor; Train: Boolean = False): TTensor; override;
    function ForwardIV(const Inputs: array of TIValue): TIValue;
    function Method(const Name: String; const Inputs: array of TTensor): TTensor;
    function MethodIV(const Name: String; const Inputs: array of TIValue): TIValue;
    procedure Save(const FileName: String);
    procedure SetTrain;  procedure SetEval;                                // FMode も設定する
    property Inner: TJitModule read FInner;
  end;
```

- 登録 = tch-rs の `TrainableCModule::load`：`FInner.NamedParameters` の各 `(Name, T)` に
  ついて `P.Add(StringReplace(Name, '.', '_', [rfReplaceAll]), T, T.RequiresGrad)` —
  変数名は `.` を拒否するので `.`→`_`（`encoder.layer1.weight` → `P` の接頭辞の下の
  `encoder_layer1_weight`）。学習対象 = 読み込んだテンソルの `requires_grad`。`Add` は
  ハンドルを共有します：ストアの要素はモジュールのパラメータ*そのもの*です
  （`new torch::Tensor(v)` は `TensorImpl` を共有する）。よってストアへの `TOptimizer`
  はスクリプト化モジュールをその場で更新し、`SetDevice`/`SetKind`（`set_data`）は
  それを動かし、`Save` は学習済みの値を書きます **(H21)**。ストアの kind は適用しません
  （tch-rs と同じ）。デバイスは `atm_load_on_device` による `P.Device`。
- `Forward(X, Train)` = `FInner.Forward([X])`。tch-rs は `train` を無視します
  （`forward_t(&self, xs, _train)`：`set_train` には `&mut self` が要る）。差異（明示）：
  Pascal 版はこれを適用します — `FMode` が `jmUnknown` か `Train` と食い違うときに
  `atm_train`/`atm_eval` を呼んで記憶する — ので、dropout や batch-norm を持つ
  スクリプト化モデルも `TSequential` の `Train` の下で `TchNN` のレイヤーと同じように
  振る舞います。読み込んだモジュールの `training` フラグは Python が保存したままなので、
  最初の `Forward` までは `jmUnknown`。`SetTrain`/`SetEval` も `FMode` を設定します。

**Python なしのテスト資材（決定）：次の両方。** それぞれ、もう一方にできないことのために。
1. **tch-rs の `tests/foo.pt` … `foo8.pt` を `tests/jit/models/` に取り込む**（9ファイル、
   15 KB、PyTorch の zip 形式バージョン3。ライセンスは我々と同じ MIT OR Apache-2.0。
   CLAUDE.md に従い tch-rs の2つのライセンスファイルを隣にコピーし、コミットメッセージに
   出典として `tch-rs 4227b89b tests/foo*.pt` と `tests/create_jit_models.py` を書く）。
   本物の Python が書き出したファイル — それこそが用途 — で、`jit_tests.rs` が期待値を
   与えます。`foo6.pt`/`foo7.pt` はオブジェクト待ち。内容（zip の `code/__torch__.py`
   から）：`foo` は `2x + y + value`、バッファ `value = [42.0]`。`foo1` は `2x + y`。`foo2`
   は `(2x + y, x - y)`。`foo3` は要素の積。`foo4` は `Tuple[float, float, int]` →
   `x0 + x1·x2`。`foo5` は `List[str]` → 各文字列の末尾1文字を除いたもの。`foo8` は
   `generate(Dict[str, Tensor])` → `(d['foo'], d['bar'])`。**(H19)**：libtorch 2.13 で
   読み込め（tch-rs の CI はこのコミットでそうしている）、下の数値を与える。
2. **Pascal でモジュールを作る**。パラメータや `self.training` が要るところは
   `CreateEmpty` + `RegisterParameter` + `Define`（フォーク、H20）で、tch-rs の
   `create_traced` には `CreateByTracing` で。トレースだけを唯一の手段にする案は却下：
   パラメータも、タプルも、テンソル以外のシグネチャも作れない。

**`tests/jit/jit_test.pas`**（本物の libtorch。tch-rs のテストと同じくリポジトリの
ルートから実行。0次元の入力は `TTensor.scalar_tensor(42, kInt64, TDevice.Cpu)`。ファイル
は `tests/jit/models/` の下）、手計算できる数値：
1. `foo.pt`（`jit`）：`x = [3, 1, 4, 1, 5]`、`y = [7]`（kFloat）：`Forward([x, y]) = 2x +
   y + 42 = [55, 51, 57, 51, 59]`。`Method('forward', …)` でも同じ。`NamedParameters` は空
   （`value` はバッファ）。`Create(file, TDevice.Cpu)` も同じ結果。
2. `foo1.pt`（`jit1`）：0次元 kInt64 の `42` と `1337` → `2·42 + 1337 = 1421`、kind は
   `kInt64`。`Forward` と `Method('forward')` で。
3. `foo2.pt`（`jit2`）：`ForwardIV([42 のテンソル, 1337 のテンソル])` → `Tag = ivTuple`、
   `Count = 2`、`Item(0).AsTensor` = `1421`、`Item(1)` = `42 - 1337 = -1295`。`MethodIV`
   でも同じ。タプルへの `AsTensor` は両方のタグを含む `ETorchError`。テンソル API の
   `Forward` は `ETorchNativeError` "forward did not return a tensor"。
4. `foo3.pt`（`jit3`）：`[1, 2, 3, 4, 5]` → `120`（0次元）。
5. `foo4.pt`（`jit4`）：`ForwardIV([TIValue.Tuple([2.0, 3.0, 4])])` → `ivDouble` の
   `2 + 3·4 = 14`。`NamedParameters` は空。
6. `foo5.pt`（`jit5`）：`ForwardIV([TIValue.StringList(['foo', 'bar', 'foobar'])])` →
   3つの `ivString` の `ivGenericList`：`'fo'`、`'ba'`、`'fooba'`。
7. `foo8.pt`（`specialized_dict`、tch-rs issue 597）：`MethodIV('generate',
   [TIValue.GenericDict(['bar', T[1, 7], 'foo', T[1, 2, 3]])])` → タプル `([1, 2, 3],
   [1, 7])`。**(H22)**：`ati_generic_dict` はすべてのキーが文字列ですべての値がテンソルの
   とき `Dict[str, Tensor]` に特化し、メソッドのスキーマはそれを要求する。
8. 往復（tch-rs の `ivalue_round_trip`。公開した `ToC`/`FromC` を通して）：`None`。`True`、
   `False`。`-1`、`42`、`15`。`''`、`'foobar'`。`Tuple([42, Pi])`。`IntList([42, 1337])`。
   `DoubleList([e, Pi, 299792458.00001])`。`Tuple([BoolList([T, F, T, T]), DoubleList(…)])`。
   `GenericList([42, 'foobar'])`。`GenericDict([42, 'foobar', 'foo', 'bar'])` は同じ順で
   戻る（H24）。加えて `Tensor` `[1, 2, 3]`（値と kind を保つ）、空の `Tuple([])`（nil
   ポインタの経路）、`ivString` の `ivGenericList` として戻る `StringList(['a', 'b'])`。
   奇数個の `GenericDict` は `ETorchError`。
9. トレース（`create_traced`）：`CreateByTracing('MyModule', 'MyFn', [0次元 kDouble の
   0.0, 1.0], @AddFn)`、`AddFn = [Inputs[0] + Inputs[1]]`。一時ファイルに `Save`、`Create`
   で読み戻し、`Method('MyFn', [[1, 2, 3, 4, 5], [41, 1335, Pi - 3, 4, 5]])`（kDouble）=
   `[42, 1337, Pi, 8, 10]` と正確に一致（`Pi - 3` は Sterbenz の補題により正確なので
   `(Pi - 3) + 3 = Pi`）。`Method('nope', …)` は `ETorchNativeError`。`ETorchError` を
   投げるクロージャはそれを伝播させ、その後の2回目の `CreateByTracing` は動く（H23）。
10. ソースからのスクリプト化モジュール（H20）：`M := TJitModule.CreateEmpty('Lin')`。
    `RegisterParameter('weight', requires_grad 付きの zeros [1, 3])`、`('bias',
    requires_grad 付きの zeros [1])`、`('scale', ones [1], grad なし)`。`Define('def
    forward(self, x):'#10'    return (torch.matmul(x, torch.t(self.weight)) + self.bias) *
    self.scale'#10)`。`Forward([[1, 2, 3]]) = 0`。`NamedParameters` = `weight`、`bias`、
    `scale`（3つ、登録順）。一時的な `.pt` に `Save`。次に `J :=
    TTrainableJitModule.Create(file, VS.Root / 'jit')`：`VS.Count = 3`（`jit.weight`、
    `jit.bias`、`jit.scale`）、`TrainableVariables` は2つ（`scale` は grad なし）。
    `Sgd(VS, 0.1)`、`X = [[1, 2, 3]]`、`Y = [[1]]`、`mse_loss = 1`。1回の `BackwardStep`：
    `∂/∂w = -2·x·(1 - 0) = [-2, -4, -6]`、`∂/∂b = -2` → `w = [0.2, 0.4, 0.6]`、`b = 0.2`
    で **`J.Forward(X) = 0.2 + 0.8 + 1.8 + 0.2 = 3.0`**（H21：モジュールがストアの
    ステップを見る）。`J.Save`、`TJitModule` として読み戻し、`Forward([X]) = 3.0`。
11. Train フラグ（H20）：`CreateEmpty('Mode')`、`Define('def forward(self, x):'#10'    if
    self.training:'#10'        return x * 2'#10'    return x'#10)`：新しいモジュールは
    `Forward([X]) = 2X`（`training` の既定は True）、`SetEval` → `X`、`SetTrain` → `2X`。
    保存して `TTrainableJitModule` として読み戻す：`Forward(X, True) = 2X`、`Forward(X) =
    X`、再び `Forward(X, True) = 2X`。
12. グローバル：新しいプロセスで `JitGetProfilingMode` は True **(H25)**、
    `JitSetProfilingMode(False)` の後は False、再び True。
    `JitSetTensorExprFuserEnabled(True/False)` を読み戻す。
    `JitSetGraphExecutorOptimize(True)` は例外を出さない。
13. エラーとハンドル：`TJitModule.Create('tests/jit/models/missing.pt')` →
    `ETorchNativeError`。`Define('def forward(self, x): return')` → `ETorchNativeError`。
    すべてのモジュール、ストア、オプティマイザを解放したあと `TchLiveHandles` は開始時の
    値に戻り、最後には 0。

**モック**（`tests/mock/mock_tch.c`、配線のみ。`tests/mock/jit_mock_test.pas`）：`ivalue`
は実際のペイロードを持つタグ付き構造体で、`mock_live_ivalues` カウンタ付き（`ati_*` は
タグ 0〜13 について忠実に実装、文字列リストの `ati_tag` は 12 を報告）→ 項目8の往復を
すべてモックで、その後 `mock_live_ivalues = 0`、奇数個の辞書、nil ポインタ。`module`
構造体と `mock_live_modules`：`atm_load*` は名前に `missing` を含むと失敗、
`atm_forward`/`atm_method` は最初のテンソルのコピーを返す（ないときはエラー）、
`atm_forward_`/`atm_method_` は入力のクローンのタプルを返しメソッド名を記録する
（`mock_last_method`）、`atm_eval`/`atm_train` → `mock_module_training`、`atm_to` は
デバイスと dtype を、`atm_save` はファイル名を記録、`atm_named_parameters` は `weight`
`[2, 3]`（grad あり）と `sub.bias` `[2]`（grad なし）でコールバック →
`TTrainableJitModule` は `jit.weight` を学習対象として、`jit.sub_bias` を学習対象でなく
登録する。`atm_create_for_tracing` / `atm_end_tracing` は呼び出し回数を数える（例外を
投げるクロージャの経路は `atm_end_tracing` を出力 0 で1回呼ぶ）。`atm_create` /
`atm_register_parameter` / `atm_define` は回数と最後のソースを記録。`atm_free` は減らす。
確認できないもの：TorchScript そのものと、項目1〜11のすべての数値。

**仮説（本物の libtorch が必要）：**
- **(H19)** 取り込んだ `foo*.pt` は libtorch 2.13 で読み込め、項目1〜7の数値を与える。
- **(H20)** フォークの追加：`Module::define` は上の2つのソース（`torch.matmul`、`torch.t`、
  `self.training`）を libtorch 2.13 でコンパイルし、`register_attribute("training", Bool,
  true)` で `self.training` と `atm_train`/`atm_eval` が働き、`named_parameters` は登録した
  3つのパラメータ（バッファではない）をその `requires_grad` とともに返す。
- **(H21)** パラメータの共有：ストアへの1回の SGD ステップの後に `J.Forward(X) = 3.0`
  で、保存したモジュールは学習済みの値で読み戻せる。
- **(H22)** `ati_generic_dict` の `Dict[str, Tensor]` への特化が `foo8.pt` の `generate`
  に受け入れられる。
- **(H23)** 例外を投げるクロージャの後の `atm_end_tracing(M, name, nil, 0)` でトレースは
  引き続き使える。
- **(H24)** `GenericDict` の往復は挿入順を保つ。
- **(H25)** プロファイリングモードは既定で on。

**結果（ローカルセッション4、libtorch 2.13.0 CPU）：H19〜H25 はすべて成立。**
`tests/jit/jit_test.pas` は上の項目 1〜13 を書かれたとおりに通る：
- **(H19) 解決：** 使う7つのモデルはすべて読み込め（`CreateFromStream` と
  `Create(file, TDevice.Cpu)` でも）、`jit_tests.rs` の数値になる。
- **(H20) 解決：** `Module::define` は両方のソースをコンパイルする。新しい
  `CreateEmpty` のモジュールは `self.training` の分岐を通り、`SetEval`/`SetTrain` で
  切り替わる。`NamedParameters` は `weight`、`bias`、`scale` を登録順に
  `requires_grad` 付きで返す。
- **(H21) 解決：** ストア上で SGD を1ステップ回すと `w = [0.2, 0.4, 0.6]`、`b = 0.2`、
  `J.Forward(X) = 3.0`。`J.Save` で保存したモジュールを読み直しても `Forward = 3.0`。
- **(H22) 解決：** `foo8.pt` の `generate` は特化した辞書を受け入れる。
- **(H23) 解決：** 例外を投げるクロージャの後でも、次の `CreateByTracing` は動く。
- **(H24) 解決：** `GenericDict([42, 'foobar', 'foo', 'bar'])` は挿入順で戻る。
- **(H25) 解決：** プログラム開始時の `JitGetProfilingMode` は True。

実装メモ（上のスケッチからの小さな逸脱）：
- `TIValueTag` は値を明示した列挙型で、FPC はそれを配列の添字に使えない
  （"enums with assignments cannot be used as array index"、実測）ので、`TagName`
  は `case` で書いた。
- `CreateByTracing` の後始末（`atm_end_tracing(M, FnName, nil, 0)`、エラースロットの
  クリア、`atm_free`）は、`Fn` の例外に限らず `atm_create_for_tracing` 以降の
  *あらゆる* 失敗で走る。出力の nil テンソルや（トレース状態を戻す前の）
  `atm_end_tracing` の失敗でもトレースが残るため。
- `FromC` はポインタを包む前に `ati_to_tensor` を確認する。`Item`、`Key`、`Value` は
  範囲外の添字で `ETorchError`。
- モック（`tests/mock/jit_mock_test.pas`）は、`ToC`（nil テンソル）や `FromC`
  （Device）で子が失敗しても C の ivalue が残らないことも確認する。

#### 4.6.9 TorchScript のオブジェクト（`ivObject`、タグ 14）、`wrappers/jit.rs`（`Object`）、`tests/jit_tests.rs`（`jit6`、`jit_double_free`）、`torch_api.cpp` より

設計セッション6（2026-10-11）でローカル機上で設計した。つまり本物の libtorch 2.13.0
（CPU）とビルド済みの `libtch.so` に対して行ったので、以下で *実測* と記したものはすべて
生の C エントリポイントを叩く使い捨てプローブの結果（HANDOFF の M9〜M11）。プローブで
届かなかったものが **(H26)**〜**(H28)**。範囲：4.6.8 が未宣言のまま残した4つのエントリ
ポイント（`ati_object_method_`、`ati_object_getattr_`、`ati_clone`、`atm_create_class_`）、
tch-rs の `Object`、テスト `jit6` と `jit_double_free`、取り込み済みの `foo6.pt`/`foo7.pt`。
4.6.8 のそれ以外は変えない。

**C 側がしていること**（`torch_api.cpp`、上流とバイト単位で同一）：
- `ati_tag` は `isObject()` に 14 を返す。*実測*：`foo6` の `forward` の結果と `foo7` の
  `make_input_object` の結果はタグ 14。
- `ati_object_method_(i, name, ivalues, n)` は `*i` の *コピー* を `self` として、続けて
  引数のコピーを積み、`i->toObjectRef().type()->getMethod(name)` を実行する。結果は新しい
  `IValue`。ポインタ `i` は読まれるだけ。*実測*：`foo6` のオブジェクトの `y()` は
  `x = [3, 4, 5]` に対して `[6, 8, 10]`。
- `ati_object_getattr_(i, name)` は属性を持つ新しい `IValue` を返す（*実測*：`x` →
  `[3, 4, 5]`。`foo7` の `foo`/`bar` → `[1, 2, 3]`/`[4, 5, 6]`）。
- `ati_clone(i)` は `new IValue(*i)`。Object の場合、`IValue` のコピーコンストラクタは
  `c10::intrusive_ptr<ivalue::Object>` をコピーする（`c10/core/ivalue.h`）。C のヒープ
  セルは2つ、TorchScript のオブジェクトは1つ。*実測*：元を `ati_free` した後も、
  **モジュールを `atm_free` した後も**、クローンの `y()` は動く — `ivalue::Object` は
  `StrongTypePtr`、つまりコンパイル単位への `shared_ptr` を持つので、オブジェクトは
  モジュールを生かしておく必要がない。
- `ati_to_tuple` / `ati_to_generic_list` / `ati_to_generic_dict` はスロットごとに新しい
  `new IValue(child)` を書く。*実測*：オブジェクトから作ったタプルを解放してから展開
  すると、元と異なる子ポインタが得られ、タプルが消えた後も `y()` が動く。
- `atm_create_class_(m, clz, ivalues, n)` は `m->create_class(QualifiedName(clz), stack)`：
  モジュールのコンパイル単位からクラスを **修飾名** で探し、`__init__` を実行する。
  *実測*：`foo7` に `'__torch__.InputObject'` とテンソル2つで、`add_them` が受け取る
  オブジェクトができる（`[5, 7, 9]`）。素の `'InputObject'` は失敗する。（この関数の
  未使用ローカル `base` は無害で、パッチしない。領域を上流と同一のままにする。）
- メッセージ（*実測*、`what()` の1行目、libtorch 2.13.0）：

  | 呼び出し | メッセージ |
  |---|---|
  | `getattr 'nope'` | `__torch__.TorchScriptClass (of Python compilation unit at: 0x…) does not have an attribute with name 'nope'` |
  | `method 'nope'` | `Couldn't find method: 'nope' on class: '__torch__.TorchScriptClass (…)'` |
  | 引数1つの `method 'y'` | `Expected at most 1 argument(s) for operator 'y', but received 2 argument(s). Declaration: y(__torch__.TorchScriptClass self) -> Tensor` |
  | オブジェクトへの `ati_to_tensor` | `Expected Tensor but got Object` |
  | オブジェクトへの `ati_length` | `unsupported tag for length Object`（-1 を返す） |
  | `create_class 'InputObject'` / `'__torch__.Nope'` | `Could not find class with name: 'InputObject' in module.` |
  | 引数1つの `create_class` | `__init__() is missing value for argument 'bar'. Declaration: __init__(__torch__.InputObject self, Tensor foo, Tensor bar) -> NoneType` |
  | `add_them(5)` | `add_them() Expected a value of type '__torch__.InputObject (…)' for argument 'data' but instead found type 'int'.` |
  | 非オブジェクトへの `ati_object_getattr_`/`_method_` | `isObject() INTERNAL ASSERT FAILED at ".../ivalue_inl.h":1699, please report a bug to PyTorch. Expected Object but got Int` |

  最後の行は生ポインタでしか到達できない。Pascal の API はタグ 14 のポインタしか包まない
  ので、これに対するガードは足さない。

**ハンドル（決定）。**

| 選択肢 | 判定 |
|---|---|
| `TIValue` に生の `PTorchIValue` を持ち手で解放 | 却下：`TIValue` は自由にコピーされるレコード（`Items`、開配列、戻り値）で、コピーが二重解放する — tch-rs の issue 475（`jit_double_free`）はまさにオブジェクトの `IValue` の2回目の解放だった。 |
| 管理オペレータ付きレコード | 4.1 と同じ理由で却下。 |
| `TTensor` のようにハンドルを包むレコード `TJitObject` | コンパイラの事実で却下：`TIValue` がフィールドを必要とし、オブジェクトのメソッドは `TIValue` を受け取り返すが、FPC はレコードを前方宣言できない。 |
| **`TInterfacedObject` で参照カウントするインターフェース `IJitObject`、実装クラスは非公開** | **採用。** FPC はインターフェースを前方宣言できる：`IJitObject = interface;` → `Obj: IJitObject` フィールドを持つ `TIValue` → `TIValue` を使う完全な宣言。*実測*（M10、FPC 3.2.2）：コンパイルできる。`B := A` はハンドルを共有（同じ `Raw`）、一方を解放しても他方は生き、最後の参照が消えると生存数は 0 に戻る。タグとインターフェースのレコードの `SizeOf` は 16。 |

4.1 が公開インターフェースを避けた理由はここでは当たらない：演算子オーバーロードなし、
メソッドは3つ、実装クラス `TJitObjectHandle` は implementation 部に宣言するので、誰も
クラス参照で持てない。スレッド安全性は 4.1 と同じ：`TInterfacedObject` のインターロック
された参照カウント。TorchScript のオブジェクト自体はテンソルと同様に同期されない。
生存数は `TchLiveHandles` と同じ方式：ハンドルのコンストラクタ／デストラクタで
`InterlockedIncrement`/`Decrement` する `LiveObjects` カウンタを `JitLiveObjects` で読む。

```pascal
unit TchJit;  // 追加分
type
  TIValueTag = (..., ivGenericDict = 13, ivObject = 14,
    ivStringList = 20, ivDevice = 21);

  IJitObject = interface;                       // 前方宣言：TIValue が持ち、そのメソッドが TIValue を使う

  TIValue = record
    ...
    Obj: IJitObject;                            // ivObject（それ以外は nil）
    class function FromObject(const O: IJitObject): TIValue; static;
    class operator :=(const O: IJitObject): TIValue;
    function AsObject: IJitObject;              // 不一致なら ETorchError('ivalue: expected Object, got Tuple')
    ...
  end;

  { TorchScript のオブジェクト（tch-rs の Object）。TIValue のコピーは共有し、
    最後の参照が消えたとき C の IValue が解放される。モジュールを生かしておかず、
    その必要もない。 }
  IJitObject = interface
    ['{...新しい GUID...}']
    function Raw: PTorchIValue;                 // 借用。常にタグ 14
    function MethodIV(const Name: String; const Inputs: array of TIValue): TIValue;  // ati_object_method_
    function GetAttr(const Name: String): TIValue;                                   // ati_object_getattr_
  end;

  TJitModule = class
    ...
    { atm_create_class_：QualifiedName は例 '__torch__.InputObject'（ClassName ではない：TObject.ClassName と重複識別子になる。ローカルセッション6で実測） }
    function CreateClass(const QualifiedName: String; const Inputs: array of TIValue): IJitObject;
  end;

function JitLiveObjects: Integer;               // 生きている TJitObjectHandle の数。TchLiveHandles と同じ

implementation
type
  TJitObjectHandle = class(TInterfacedObject, IJitObject)
  private
    FRaw: PTorchIValue;
  public
    constructor Create(P: PTorchIValue);        // 所有権を取る。本体は代入2つで例外を出せない
    destructor Destroy; override;               // ati_free(FRaw)。決して例外を出さない
    function Raw: PTorchIValue;
    function MethodIV(const Name: String; const Inputs: array of TIValue): TIValue;
    function GetAttr(const Name: String): TIValue;
  end;
```

- `TagName` は `'Object'`。`Count` はスカラーと同様にオブジェクトで例外。`IsNone` は
  変わらない。非オブジェクトへの `AsObject` は通常の不一致例外。`Obj` が nil の
  `ivObject` は `FromC` からは出てこず、nil テンソルと同じ扱い：`ToC` が
  `ETorchError('ivalue: nil object')`。
- `CreateClass` は `TIValue` ではなくオブジェクトを返す（tch-rs の `create_class_is`
  からの逸脱、明記）：`Module::create_class` は常に Object を作るので、結果を `FromC`
  と `AsObject` に通す（ありえない非オブジェクトには `AsObject` が例外を出す）。クラス名は
  実測したメッセージのとおり修飾名。
- `array of TIValue` 引数としての `[O, 42]` は、`[T, 42, 1.5]`（M5）と同様に `:=`
  オペレータで要素ごとに変換されるはず。ローカルセッション6で *実測*：変換される
  （`add_them([O])`、`Tuple([O, 42])`）。

**すべての経路での所有権**（設計の核心。各行で誰が何を解放するかを言う）：
1. **`FromC`、タグ 14。** `Result.Tag := ivObject; Result.Obj :=
   TJitObjectHandle.Create(P); Keep := True` — 以後 `P` はハンドルが所有する。`finally` は
   `FreeIValues(Cs); if not Keep then ati_free(P)` になる。`Keep` は False で初期化する
   ローカル。コンストラクタの本体は `inherited Create; InterlockedIncrement(LiveObjects);
   FRaw := P` — 代入を最後にして、その前で何かが例外を出しても（FPC はデストラクタを
   走らせる）`FRaw` は nil のまま、デストラクタは何も解放せず、`finally` が `P` を解放する
   （`Keep` は未設定）。他のタグは変わらず、引き続き `P` を消費する。
2. **`ToC`、`ivObject`。** `Result := ati_clone(Obj.Raw); TchCheck`。呼び出し側がクローンを
   所有し、他の `ToC` の結果と同様に解放する（`ForwardIV`/`MethodIV`/`CreateClass` と
   コンテナ分岐の `finally FreeIValues(Cs)`）。ハンドル自身のポインタは解放のために C に
   渡されることがない。これが tch-rs の issue 475 の修正：同じオブジェクトを2回渡しても、
   渡した後に Pascal の値が死んでも安全（C レベルで*実測*：同じポインタで `add_them` を
   2回呼ぶと `[5, 7, 9]` が2回。Pascal 経路は下のテスト項目 d）。
3. **コンテナ内のオブジェクト、`ToC`。** 子は他の子と同様にスロットへクローンされ、C の
   コンテナがそれをコピーし（intrusive_ptr がもう1つ増える）、`FreeIValues(Cs)` が
   クローンを解放する。後の子が `ToC` で失敗すると、すでに作ったクローンは
   `IValuesToC` の `except` が解放し、ハンドルには触れない。
4. **コンテナ内のオブジェクト、`FromC`。** `ati_to_*` はスロットごとに新しい `IValue` を
   書く（*実測*）。再帰する `FromC` がそれをハンドルで包む（手順1）。後の子が失敗した
   とき：未消費のスロットは `FreeIValues(Cs)` が解放する。すでに変換済みの子は
   `IValuesFromC` のローカルな結果配列にあり、巻き戻し中にコンパイラがそれを終了処理
   してハンドルを解放し、デストラクタがポインタを `ati_free` する。漏れも二重解放も
   ない。モックのテストが確認する（オブジェクトの後に Device）。
5. **`MethodIV` / `GetAttr` / `CreateClass`。** `TJitModule.MethodIV` と同じ形：
   `Cs := IValuesToC(Inputs)`（オブジェクトはクローン）。`P := ati_object_method_(FRaw,
   PChar(Name), Cs, n)` / `ati_object_getattr_(FRaw, PChar(Name))` /
   `atm_create_class_(FModule, PChar(QualifiedName), Cs, n)`。`try … finally FreeIValues(Cs)`
   の中で `TchCheck`。`Result := TIValue.FromC(P)`。C は `self` をコピーする
   （`inputs.push_back(*i)`）ので `FRaw` は有効なまま。`GetAttr` は tch-rs の nil 確認を
   残す（`ETorchError('ivalue: getattr returned nil')`。`PROTECT` 下では `TchCheck` の後は
   到達不能、`FromC` 自身のものと同様）。
6. **Destroy。** `if FRaw <> nil then ati_free(FRaw)`、`InterlockedDecrement(LiveObjects)`。
   決して例外を出さず、エラースロットも読まない（4.1）。
7. **モジュールの寿命。** オブジェクトは `TJitModule` より長生きする（上で*実測*）。
   `IJitObject` はモジュールへの参照を持たない。ユニットヘッダに書く。
8. **一時変数。** H1（4.1）と同じ：式の中で作られたオブジェクトはルーチンが戻るまで
   ルーチンの一時スロットで生きる（M10 で*実測*：名前付き変数をすべて解放した後も
   ルーチン内では生存数 1、戻った後に 0）。したがってテストは `JitLiveObjects` と
   `TchLiveHandles` を呼び出し側のルーチンで比較する。

**L2（`TchSys.pas`）**、`torch_api.h` のとおり：

```pascal
function  atm_create_class_(m: PTorchModule; clz_name: PChar; ivalues: PPTorchIValue;
  nivalues: cint): PTorchIValue; cdecl; external LibTch;
function  ati_object_method_(i: PTorchIValue; method_name: PChar; ivalues: PPTorchIValue;
  nivalues: cint): PTorchIValue; cdecl; external LibTch;
function  ati_object_getattr_(i: PTorchIValue; attr_name: PChar): PTorchIValue; cdecl; external LibTch;
function  ati_clone(i: PTorchIValue): PTorchIValue; cdecl; external LibTch;
```

4つとも `PROTECT` されエラー時は nil を返す（各呼び出し後に `TchCheck`）。ユニット
コメントの「未宣言」リストは `atm_fuser_cuda_*` だけになる。**フォークへのパッチ：なし。**

**`tests/jit/jit_test.pas`、グループ 14「Objects」**（手計算できる数値。`x = [3, 4, 5]`、
`T1 = [1, 2, 3]`、`T2 = [4, 5, 6]`、すべて kFloat。`Before := JitLiveObjects` と
`TchLiveHandles` は呼び出し側で取る）：
- a. `jit6`：`foo6.pt`、`R := M.ForwardIV([x])` → `R.Tag = ivObject`、`R.TagName =
  'Object'`。`O := R.AsObject`。`O.MethodIV('y', []).AsTensor = [6, 8, 10]`。
  `O.GetAttr('x').AsTensor = [3, 4, 5]`。`R.AsTensor` は `ETorchError` `'ivalue: expected
  Tensor, got Object'`。`M.Forward([x])` は `ETorchNativeError` "forward did not return a
  tensor"（`atm_forward` の `isTensor` 確認。**(H27)**）。
- b. エラー（`ETorchNativeError`、メッセージが含む）：`O.GetAttr('nope')` → `does not have
  an attribute with name 'nope'`。`O.MethodIV('nope', [])` → `Couldn't find method:
  'nope'`。`O.MethodIV('y', [7])` → `Expected at most 1 argument(s)`。
- c. `jit_double_free`：`foo7.pt`、`Inp := M.MethodIV('make_input_object', [T1, T2])` →
  `ivObject`。`M.MethodIV('add_them', [Inp]).AsTensor = [5, 7, 9]`。**同じ `Inp` で同じ
  呼び出しをもう一度** → `[5, 7, 9]`（issue 475 の回帰：ハンドルは渡されても生き残る）。
  `Inp.AsObject.GetAttr('foo') = [1, 2, 3]`、`'bar' = [4, 5, 6]`。
- d. `CreateClass`：`O := M.CreateClass('__torch__.InputObject', [T1, T2])`。
  `M.MethodIV('add_them', [O]) = [5, 7, 9]`（または `[TIValue.FromObject(O)]`）。
  `CreateClass('InputObject', [T1, T2])` → `Could not find class with name: 'InputObject'
  in module.`。`CreateClass('__torch__.InputObject', [T1])` → `__init__() is missing value
  for argument 'bar'`。`M.MethodIV('add_them', [5])` → `Expected a value of type
  '__torch__.InputObject`。`foo6.pt` で：`CreateClass('__torch__.TorchScriptClass',
  [x]).MethodIV('y', []) = [6, 8, 10]`（**(H27)**）。
- e. コンテナ：`W := TIValue.FromC(TIValue.Tuple([O, 42]).ToC)` → `W.Count = 2`、
  `W.Item(1).AsInt = 42`、`W.Item(0).AsObject.GetAttr('foo') = [1, 2, 3]`、
  `W.Item(0).AsObject.Raw <> O.Raw`（新しい `IValue`、同じオブジェクト）。同じ
  オブジェクトを `GenericList([O])`（`Item(0)`）と `GenericDict(['k', O])`（`Value(0)`）
  でも（**(H26)**）。
- f. 共有と寿命：`A := TIValue.FromObject(O); B := A; A.AsObject.Raw = B.AsObject.Raw`。
  `A := Default(TIValue)`。`B.AsObject.GetAttr('foo')` は引き続き `[1, 2, 3]`。
  `O.GetAttr('bar')` の **前に** `M.Free` → 引き続き `[4, 5, 6]`（オブジェクトは
  モジュールより長生き）。グループのルーチンが戻った後：`JitLiveObjects = Before`、
  `TchLiveHandles` は開始時の値。プログラム最終行に `TchLiveHandles = 0` と並べて
  `JitLiveObjects = 0` を足す。
- g. リークループ（**(H28)**）：小さなルーチン内で `foo6` のオブジェクトに `10^4` 回
  `O.MethodIV('y', [])`。その後 `JitLiveObjects` と `TchLiveHandles` は不変（C++ 側に
  カウンタはないので、Pascal のカウンタと 4.1 のループと同様の平坦な RSS が証拠）。

**モック**（`tests/mock/mock_tch.c`、配線のみ）：`mock_ivalue` にタグ 14 用の `char *cls`
と `char **names` を足す。属性は `items` に（`n` 個）、名前はコンストラクタ順に `a0`、
`a1`、…。`atm_create_class_(m, clz, ivs, n)` は `mock_last_class` を記録し、`clz` に
`missing` を含むと失敗、そうでなければ `n` 個のクローンした属性を持つオブジェクトを
返す。`ati_object_getattr_(o, name)`：`expect_tag(o, 14)`、名前の属性のクローン、
未知の名前は `set_err("mock: no attribute …")` で nil。`ati_object_method_(o, name, ivs,
n)`：`expect_tag(o, 14)`、`mock_last_method` を記録。`name == "fail"` はエラーで nil、
それ以外は属性のクローンの後に引数のクローンを並べたタプル。`ati_clone` → `iv_clone`
（モックではディープコピーで、生きている ivalue として数える。モックは配線を確認する
もので同一性は確認しない）。`iv_clone` と `ati_free` は `cls` と `names` をコピー／解放
する。14 への `ati_length` は libtorch と同様に失敗する。既存の `atm_forward_`/
`atm_method_`（クローンのタプル）はそのままオブジェクトを運べる。
`tests/mock/jit_mock_test.pas` に新しい `TestObjects`：
1. `O := M.CreateClass('__torch__.Foo', [1, 2.5])` → `mock_last_class`、
   `mock_live_ivalues = 1`、`JitLiveObjects = 1`。
2. `O.GetAttr('a1').AsDouble = 2.5`。`GetAttr('zz')` → `ETorchNativeError`。それぞれの後
   `mock_live_ivalues` は 1 に戻る。
3. `O.MethodIV('m', ['x'])` → タプル `(1, 2.5, 'x')`、`mock_last_method = 'm'`。
   `MethodIV('fail', [O])` は例外を出し `mock_live_ivalues = 1` のまま（クローンした
   入力は解放された）。
4. 共有：`A := TIValue.FromObject(O); B := A; A := Default(TIValue)`。
   `B.AsObject.Raw = O.Raw`。3つすべてを解放するまで数は不変（呼び出し側で確認）。
5. 往復 `FromC(Tuple([O, 42]).ToC)`：`Item(0).Tag = ivObject`、
   `Item(0).AsObject.GetAttr('a0').AsInt = 1`。両オブジェクトが生きている間
   `mock_live_ivalues = 2`（モックのクローンは別セル）、呼び出し側がすべて解放すると 0。
6. モックのモジュールで `M.MethodIV('f', [O])`：`Item(0).Tag = ivObject`、その後数は戻る。
7. 失敗する兄弟：`FromC(Tuple([O, TIValue.FromDevice(Cpu)]).ToC)` は Device の子で例外。
   `mock_live_ivalues` はオブジェクト自身のセルだけに戻り、`JitLiveObjects` は不変
   （所有権リストの手順4）。
8. プログラムの最終行に `JitLiveObjects = 0` を足す。
   *実装では*（ローカルセッション6）、モックのオブジェクトは1セル＋属性ごとに1セルなので、
   上の「1」と「2」は `Cells`（`[1, 2.5]` で 3）と `2 * Cells` になる。項目7は文の前後の
   差分で確認する。項目5〜6の一時変数がまだハンドルを持っているため（H1）。

**仮説（本物の libtorch が必要。上のそれ以外はすべて実測）：**
- **(H26)** オブジェクトは、タプル（実測）と同様に `GenericList` と `GenericDict` の値と
  しても往復する：コンテナより長生きする新しいタグ 14 の `IValue`。
- **(H27)** `foo6.pt` への `atm_create_class_('__torch__.TorchScriptClass', [x])` は
  `y()` が `2x` のオブジェクトを与え、`foo6.pt` への `atm_forward`（テンソル API）は
  "forward did not return a tensor" を出す（`torch_api.cpp` の `isTensor` 確認は読んだ
  だけで、走らせていない）。
- **(H28)** `10^4` 回のオブジェクトメソッド呼び出しで `JitLiveObjects`、`TchLiveHandles`、
  RSS が平坦。
- **結果（ローカルセッション6、2026-10-12、libtorch 2.13.0 CPU）：** H26、H27、H28 は
  成立。タプル、`GenericList`、`GenericDict` の値はそれぞれ新しいタグ 14 のポインタ
  （`Raw` が異なる）を与え、属性が読み戻せる。
  `CreateClass('__torch__.TorchScriptClass', [x]).y() = [6, 8, 10]`。`Forward` は
  "forward did not return a tensor" を出す。`10^4` 回のメソッド呼び出し：生存ハンドル 2、
  オブジェクト 1 が前後で同じ、RSS 127148 KiB → 127148 KiB。テストした6つのメッセージ（項目 a、b、d）は
  表どおりに一致。
- 取り込み済みの資材では確認できず、よって主張しないこと：一方のクローンを通じた変更が
  他方から見えること（変更するメソッドを持つ資材がなく、`Define` は TorchScript の
  クラスを宣言できない）。`IValue` のコピーコンストラクタから従うが、後の資材で確認
  してよい。

#### 4.6.10 マイルストーン3のレイヤー、`nn/conv.rs`、`conv_transpose.rs`、`group_norm.rs`、`rnn.rs`、`mod.rs`（`Id`）、`tests/nn_tests.rs` より

設計セッション7（ローカル機、本物の libtorch 2.13.0 CPU、FPC 3.2.2）。4.6.5 の
先送りリストを 4.6.5 と同じ型で：設定 record + `Default*Config`、コンストラクタ2つ、
tch-rs の変数名と登録順（`.ot` ファイルが相互に読める）。以下の数値はすべて生成 op に
対して*実測*（プローブはスクラッチパッド内、コミットしない）。レイヤークラス自体は
ローカルの実装者の仕事。

**tch-rs の `ConvConfigND<ND>` を FPC でどう表すか（決定）。** tch-rs は
`ND = [i64; 1|2|3]` 上のジェネリックな設定とレイヤーを1つ持ち、スカラーの `ConvConfig`
を取る `conv1d/2d/3d` と、スカラーを繰り返す `Create::make_array` を添えている。Pascal
では*ジェネリックでない1つの* `TConvConfigND`（次元ごとのフィールドは動的配列、
`Nd = Length`）、`Forward` で `FNd` により分岐する*1つの*基底クラス `TConvND`、そして
スカラー設定のコンストラクタだけを持つ薄いサブクラス `TConv1D/2D/3D`。FPC のジェネリック
record は測らずに却下：`specialize TConvConfigND<TInt3>` は互いに無関係な3つの型になり、
それぞれに `Default` 関数が要るうえ、基底クラスはどうせ次元で分岐する（Pascal には
`AsRef<[i64]>` も型ごとの trait 実装もない）ので、ジェネリックは何も買わない。採用した
形が頼るものは*実測*（M-A）：基底のコンストラクタ
`Create(P, In, Out, const KSizes: array of Int64; const Cfg: TConvConfigND)` とサブ
クラスの `overload` コンストラクタ2つは、`TConv2D.Create(P, 1, 1, 3)`、
`TConv2D.Create(P, 1, 1, 3, Cfg)`、**および**継承された
`TConv2D.Create(P, 1, 1, [3, 3], CfgND)` のどれも曖昧さなく解決する。サブクラスの
コンストラクタは `inherited Create(P, In, Out, [K, K], ConvConfigToND(Cfg, 2))` を呼ぶ。

```pascal
  TConvConfigND = record                     // ConvConfigND<[i64; N]>、N = Length(Stride)
    Stride, Padding, Dilation: TInt64Array;  // 3つとも長さ Nd
    Groups: Int64; Bias: Boolean;
    WsInit, BsInit: TInit; PaddingMode: TPaddingMode;
  end;
  function DefaultConvConfigND(Nd: Integer): TConvConfigND;          // [1..], [0..], [1..], 1, True, KaimingUniform, Constant(0), pmZeros
  function ConvConfigToND(const Cfg: TConvConfig; Nd: Integer): TConvConfigND;  // Create::conv の make_array
  function PadWithMode(const X: TTensor; Mode: TPaddingMode;
    const Pad: array of Int64): TTensor;     // PaddingMode::pad: X.pad(Pad, Name, NoDouble)

  TConvND = class(TModule)                   // Conv<ND>; nn::conv
  private
    FNd: Integer; FCfg: TConvConfigND;
    FPadTwice, FZeroPad: TInt64Array;        // reversed_padding_repeated_twice; [0, ..]
  public
    Weight, Bias: TTensor;
    constructor Create(const P: TVarPath; InDim, OutDim: Int64;
      const KSizes: array of Int64; const Cfg: TConvConfigND); overload;
    function Forward(const X: TTensor; Train: Boolean = False): TTensor; override;
    property Nd: Integer read FNd;
  end;
  TConv1D = class(TConvND)                   // conv1d。TConv2D と TConv3D も同一
  public
    constructor Create(const P: TVarPath; InDim, OutDim, KSize: Int64); overload;
    constructor Create(const P: TVarPath; InDim, OutDim, KSize: Int64; const Cfg: TConvConfig); overload;
  end;

  TConvTransposeConfig = record              // ConvTransposeConfig（スカラー）
    Stride, Padding, OutputPadding, Dilation, Groups: Int64; Bias: Boolean;
    WsInit, BsInit: TInit;                   // PaddingMode はない（tch-rs と同じ）
  end;
  TConvTransposeConfigND = record
    Stride, Padding, OutputPadding, Dilation: TInt64Array;
    Groups: Int64; Bias: Boolean; WsInit, BsInit: TInit;
  end;
  function DefaultConvTransposeConfig: TConvTransposeConfig;         // 1, 0, 0, 1, 1, True, KaimingUniform, Constant(0)
  function DefaultConvTransposeConfigND(Nd: Integer): TConvTransposeConfigND;
  function ConvTransposeConfigToND(const Cfg: TConvTransposeConfig; Nd: Integer): TConvTransposeConfigND;
  TConvTransposeND = class(TModule)          // ConvTransposeND<ND>
  private FNd: Integer; FCfg: TConvTransposeConfigND;
  public
    Weight, Bias: TTensor;
    constructor Create(const P: TVarPath; InDim, OutDim: Int64;
      const KSizes: array of Int64; const Cfg: TConvTransposeConfigND); overload;
    function Forward(const X: TTensor; Train: Boolean = False): TTensor; override;
  end;
  TConvTranspose1D = class(TConvTransposeND) { スカラーのコンストラクタ2つ } end;  // 2D、3D も

  TGroupNormConfig = record CudnnEnabled: Boolean; Eps: Double; Affine: Boolean; WsInit, BsInit: TInit; end;
  function DefaultGroupNormConfig: TGroupNormConfig;                 // True, 1e-5, True, Constant(1), Constant(0)
  TGroupNorm = class(TModule)
  private FCfg: TGroupNormConfig; FNumGroups, FNumChannels: Int64;
  public
    Weight, Bias: TTensor;                   // Affine が False なら IsNil
    constructor Create(const P: TVarPath; NumGroups, NumChannels: Int64); overload;
    constructor Create(const P: TVarPath; NumGroups, NumChannels: Int64; const Cfg: TGroupNormConfig); overload;
    function Forward(const X: TTensor; Train: Boolean = False): TTensor; override;
    property NumGroups: Int64 read FNumGroups;  property NumChannels: Int64 read FNumChannels;
  end;

  TRNNConfig = record                        // RNNConfig、LSTM と GRU で共通
    HasBiases: Boolean; NumLayers: Int64; Dropout: Double; Train: Boolean;
    Bidirectional, BatchFirst: Boolean;
    WIhInit, WHhInit, BIhInit, BHhInit: TInit;   // b_*_init は tch-rs では Option<Init>。逸脱の項を参照
  end;
  function DefaultRNNConfig: TRNNConfig;     // True, 1, 0, True, False, True, KaimingUniform x2, Constant(0) x2
  TRNNState = record H, C: TTensor; end;     // LSTMState((h, c))。GRUState(h) は C を nil のまま
  TRNN = class abstract                      // trait RNN。TModule ではない（tch-rs の LSTM/GRU も Module を実装しない）
  private
    FFlatWeights: TTensorArray; FHiddenDim: Int64; FCfg: TRNNConfig;
  protected
    constructor Create(const P: TVarPath; InDim, AHiddenDim, GateMul: Int64; const Cfg: TRNNConfig);  // rnn_weights
    function BatchDimIndex: Integer; inline; // BatchFirst なら 0、でなければ 1
  public
    function ZeroState(BatchDim: Int64): TRNNState; virtual; abstract;
    function Step(const Input: TTensor; const State: TRNNState): TRNNState;       // [batch, features]
    function Seq(const Input: TTensor; out State: TRNNState): TTensor;             // BatchFirst なら [batch, seq, features]
    function SeqInit(const Input: TTensor; const InState: TRNNState; out OutState: TRNNState): TTensor; virtual; abstract;
    property FlatWeights: TTensorArray read FFlatWeights;
    property HiddenDim: Int64 read FHiddenDim;
    property Config: TRNNConfig read FCfg;
  end;
  TLSTM = class(TRNN)
  public
    constructor Create(const P: TVarPath; InDim, AHiddenDim: Int64); overload;
    constructor Create(const P: TVarPath; InDim, AHiddenDim: Int64; const Cfg: TRNNConfig); overload;
    function ZeroState(BatchDim: Int64): TRNNState; override;
    function SeqInit(const Input: TTensor; const InState: TRNNState; out OutState: TRNNState): TTensor; override;
  end;
  TGRU = class(TRNN)  { 同じ4メンバ } end;

  TIdentity = class(TModule)                 // nn::Id
    function Forward(const X: TTensor; Train: Boolean = False): TTensor; override;  // Result := X
  end;
```

レイヤーごと — 変数（名前 `[形]`、デフォルトの初期化、登録順）、設定のデフォルト、
`Forward` が呼ぶ op：

| レイヤー（tch-rs のファイル） | 変数、登録順 | 設定のデフォルト | `Forward` |
|---|---|---|---|
| `TConvND` / `TConv1D` / `TConv3D`（`conv.rs`。`TConv2D` も同じ基底に移る、API は不変） | `Bias` なら `bias [Out]` `BsInit`。`weight [Out, In div Groups, K1, .., KNd]` `WsInit` | `DefaultConvConfig` は 4.6.5 のとおり。`DefaultConvConfigND(Nd)` は同じ値を `Nd` 回繰り返したもの | `pmZeros`：`X.conv{Nd}d(Weight, Bias, Stride, Padding, Dilation, Groups)`。他のモード：`PadWithMode(X, Mode, FPadTwice).conv{Nd}d(Weight, Bias, Stride, FZeroPad, Dilation, Groups)`。`case FNd of 1: conv1d; 2: conv2d; 3: conv3d`。コンストラクタは `Length(KSizes)` が 1..3 で `Stride`、`Padding`、`Dilation` がその長さでなければ `ETorchError` |
| `TConvTransposeND` / `1D` / `2D` / `3D`（`conv_transpose.rs`） | `Bias` なら `bias [Out]` `BsInit`。`weight [In, Out div Groups, K1, .., KNd]` `WsInit`（`In` が先） | `Stride` 1、`Padding` 0、`OutputPadding` 0、`Dilation` 1、`Groups` 1、`Bias` True、`WsInit` KaimingUniform、`BsInit` Constant(0) | `X.conv_transpose{Nd}d(Weight, Bias, Stride, Padding, OutputPadding, Groups, Dilation)` — 生成された引数順（groups が dilation の前）は C のもの。パディングモードなし |
| `TGroupNorm`（`group_norm.rs`） | `weight [NumChannels]` `WsInit`、`bias [NumChannels]` `BsInit`、どちらも `Affine` のときだけ | `CudnnEnabled` True、`Eps` 1e-5、`Affine` True、`WsInit` Constant(1)、`BsInit` Constant(0) | `X.group_norm(NumGroups, Weight, Bias, Eps, CudnnEnabled)` |
| `TLSTM` / `TGRU`（`rnn.rs`、`rnn_weights`） | `L` in `0..NumLayers-1`、`D` in `0..NumDirections-1`（`D = 1` なら `Sfx` = `'_reverse'`）の順に：`weight_ih_l{L}{Sfx} [GateDim, InL]` `WIhInit`、`weight_hh_l{L}{Sfx} [GateDim, Hidden]` `WHhInit`、次に `HasBiases` なら `bias_ih_l{L}{Sfx} [GateDim]` `BIhInit`、`bias_hh_l{L}{Sfx} [GateDim]` `BHhInit`。`InL = InDim`（`L = 0`）、それ以外は `Hidden * NumDirections`。`GateDim = 4 * Hidden`（LSTM、ゲート i, f, g, o）または `3 * Hidden`（GRU、ゲート r, z, n）。`FFlatWeights` は同じ並びの同じリスト（`lstm`/`gru` の `params`） | `HasBiases` True、`NumLayers` 1、`Dropout` 0、`Train` True、`Bidirectional` False、`BatchFirst` True、`WIhInit`/`WHhInit` KaimingUniform、`BIhInit`/`BHhInit` Constant(0) | `SeqInit`：LSTM は `Input.lstm([InState.H, InState.C], FFlatWeights, HasBiases, NumLayers, Dropout, Train, Bidirectional, BatchFirst, Result, OutState.H, OutState.C)`、GRU は `Input.gru(InState.H, FFlatWeights, ..., Result, OutState.H)`。`ZeroState(B)`：`TTensor.zeros([NumLayers * NumDirections, B, Hidden], FFlatWeights[0].Kind, FFlatWeights[0].Device)` を `H` に（LSTM は同じハンドルを `C` にも、tch-rs の `shallow_clone`）。`Step(X, S)`：`SeqInit(X.unsqueeze(1 - BatchDimIndex), S, Result)`、出力は捨てる。`Seq(X, S)`：`SeqInit(X, ZeroState(X.Shape[BatchDimIndex]), S)` |
| `TIdentity`（`mod.rs` の `Id`） | なし | — | `X` を返す（record のコピー、同じハンドル） |

決定事項と理由：
- **族ごとに基底1つ、サブクラスはスカラーのコンストラクタのためだけ。** `TConv2D` の
  公開 API（コンストラクタ2つ、`Weight`、`Bias`、`Forward`）は変わらない。本体は
  `TConvND` に移り、`TConv2D.Create(P, In, Out, K, Cfg)` は
  `inherited Create(P, In, Out, [K, K], ConvConfigToND(Cfg, 2))` になる。4.6.7 の
  グループ8の数値はこの形で再実行して不変（*実測*）。次元ごとの設定が欲しい
  ユーザーは `TConvConfigND` を組んで `TConvND.Create` か `TConv2D` の継承コンストラクタ
  を使う。
- **`PadWithMode` は公開**（tch-rs の `PaddingMode::pad`）。tch-rs の `pad` テストを
  移植でき、`TConvND.Forward` もこれを使う。`pmZeros` は `'constant'`（tch-rs
  `PaddingMode::to_string`。現在の private な `ModeNames` はそのスロットが `'zeros'`
  だが `Forward` は到達しない — 置き換える）。
- **reversed padding repeated twice は tch-rs ではなく PyTorch に従う（逸脱）。**
  `Padding = [Ph, Pw]` のとき `pad` が欲しいのは `[Pw, Pw, Ph, Ph]`（最後の次元の
  左右が先）。tch-rs の `conv()` は `[Pw, Ph, Pw, Ph]`（反転してもう一度反転）を作り、
  パディングが全部等しいとき — つまり皆が使うスカラー設定のとき — だけ正しい。*実測*：
  `TConv2D` の ND で `Padding [1, 0]`、reflect、`KSizes [3, 1]`、重み 1、入力
  `[[1, 2], [3, 4]]` は PyTorch の順で `[[7, 10], [5, 8]]`、tch-rs の順では
  `[1, 1, 1, 3]` の `[5, 8, 5]`。よって
  `FPadTwice[2*I] = FPadTwice[2*I + 1] = Padding[Nd - 1 - I]`。
- **`TRNNState` は両レイヤーで1つの record（逸脱）。** tch-rs は `LSTMState((h, c))`
  と `GRUState(h)` を `RNN` trait の関連型として持つ。関連型のジェネリックがないので、
  共通の抽象 `TRNN` には状態型が1つ要り、`TGRU` は `C` を nil のまま（`C.IsNil`）に
  する。アクセサ `h()`/`c()`/`value()` は単にフィールド。
- **`Seq`/`SeqInit` は出力を返し、状態は `out` で戻す**（tch-rs はタプル
  `(output, state)` を返す）。`Step` は tch-rs と同じく新しい状態を結果として返す。
  *実測*（M-B）：FPC は `TTensor` の record の `out` 引数を入口で finalize する
  （H8 の再現）ので、`L.SeqInit(X, S, S)` は本体の前に `S.H` を nil にし、op は
  `ETorchError: undefined tensor passed as hx to gru` を投げる — きれいな失敗。
  文書化：`InState` と `OutState` は別の変数でなければならない。*実測*（M-B2）：
  `S := L.Step(X, S)` は安全。関数結果は一時変数を経由し、本体は古い `S` を見る。
- **`BatchFirst = False` が動く（逸脱）。** tch-rs の `seq` は `batch_first` に
  関わらず `size()[0]` をバッチとし、`step` は `unsqueeze(1)` するので、
  `batch_first: false` では両方が間違ったバッチサイズを `zero_state` に渡す。Pascal
  は両所で `BatchDimIndex`（0 か 1）を使う。コストは `if` 2つ。*実測*：`batch_first`
  False の `gru` は `[3, 5, 2]` に対し `[3, 5, 4]`、`h [1, 5, 4]` を返す。
- **`BIhInit`/`BHhInit` は素の `TInit`**（tch-rs は `Option<Init>` で、`has_biases`
  のとき `unwrap()` — `None` は panic するだけ）。素のフィールドで panic 経路が消える。
  `HasBiases` False なら無視。
- **`ZeroState` は kind と device を呼び出し時に `FFlatWeights[0]` から読む**（tch-rs は
  `vs.device()` を生成時に保存し、kind は呼び出し時に読む）。生成後の store への
  `SetKind`/`SetDevice` が反映される。小さな逸脱、厳密により寛容。
- **cuDNN の重み平坦化** はここでは CUDA/ROCm マイルストーンに回し、4.8（H30）で実装
  済み：store が CUDA かつ cuDNN ありのとき、`TRNN` のコンストラクタが
  `_cudnn_rnn_flatten_weight` を呼ぶ。バイアスなしの重みのストライドは 2（PyTorch に従う。
  tch-rs は 4 を渡す）。CPU と ROCm（cuDNN なし）では呼ばない。
- **`TRNN` は `TModule` ではない**。tch-rs と同じ。op が受け取る `Train` は
  `Config.Train`（層間の dropout にだけ効く）で、`Forward` の引数ではない。
  `TSequential` に RNN を入れたいユーザーは `Seq` を `TFuncModule` のメソッドで包む。
- **`Id` は `TIdentity`**（名前だけ。Pascal で `Id` は「識別子」と読める）。
  `TSequential` のスロットを条件付きで埋めるための退化したレイヤー。

ここの CPU libtorch では測れず、仮説として残すもの：
- **(H29)** tch-rs の `lstm`/`gru`/`conv_transpose2d`/`group_norm` が書いた `.ot`
  ファイルとの相互運用（名前と形は構成上一致する。ファイルを作るには Rust ツール
  チェーンが要る）。
- **(H30)** CUDA/ROCm の store での cuDNN 平坦化呼び出し（項目3）— 条件・呼び出し・
  テストは 4.8（「GPU 上の H13 と H30」、GPU テスト項目 8）に移った。

**ローカルセッション7（2026-10-12）で `src/TchNN.pas` に実装。** `tests/nn/nn_test.pas`
のグループ 11（44 チェック）は libtorch 2.13.0 CPU で上の数値どおりにすべて通る。設計が
見落としていたコンパイラの事実が2つ（どちらも *実測*）：コンストラクタの引数名
`HiddenDim` はプロパティ `HiddenDim` と重複識別子になる（FPC は引数名をクラスのメンバと
照合する）ので、RNN のコンストラクタは `AHiddenDim` を取る。protected の `TRNN.Create` は
警告 3018（"Constructor should be public"）を出すので、`TJitModule.CreateRaw` と同じく
`{$push}{$warn 3018 off}` で抑える。任意のモック確認（ローカル手順の項目7）は省いた：
名前と順序は libtorch に対して確認している。

テスト一覧、`tests/nn/nn_test.pas` のグループ11「マイルストーン3のレイヤー」
（数値は tch-rs と記したもの以外、ここで生成 op に対して*実測*）：
  1. `PadWithMode`（tch-rs `pad`）：`[1, 2, 3]` を zeros `[1, 1]` →
     `[0, 1, 2, 3, 0]`。`[1, 2, 3, 4].view([1, 2, 2])` を reflect `[1, 1, 1, 1]` →
     `[4, 3, 4, 3, 2, 1, 2, 1, 4, 3, 4, 3, 2, 1, 2, 1]`。
  2. `TConv1D(1, 1, 3)`、`Padding 1`、`Bias False`、重み 1、入力 `[1, 2, 3]`
     （`[1, 1, 3]`）：zeros `[3, 6, 5]`、reflect `[5, 6, 7]`、replicate `[4, 6, 8]`、
     circular `[6, 6, 6]`。`TConv1D(4, 6, 3)` は `bias [6]` の次に `weight [6, 4, 3]`
     を登録。
  3. `TConv2D`：グループ8の4結果は不変。ND の非対称パディング `[1, 0]` reflect、
     `KSizes [3, 1]`、重み 1 → `[[7, 10], [5, 8]]`（形 `[1, 1, 2, 2]`）。
     `DefaultConvConfigND(2)` は `ConvConfigToND(DefaultConvConfig, 2)` とフィールド
     ごとに等しい。長さ4の `KSizes`、または長さの違う `Stride` は `ETorchError`。
  4. `TConv3D(1, 1, 2)`、重み 1、バイアスなし、`1..8` を `[1, 1, 2, 2, 2]` として →
     `[36]`、形 `[1, 1, 1, 1, 1]`。`TConvND(2, 3, [3, 3, 3])` に `Padding [1, 0, 0]`
     で `[1, 2, 4, 4, 4] → [1, 3, 4, 2, 2]`、weight `[3, 2, 3, 3, 3]`。
  5. `TConvTranspose2D(1, 1, 2)`、重み 1、`Bias False`、入力 `[[1, 2], [3, 4]]`：
     stride 1 → `[1, 1, 3, 3]` `[1, 3, 2, 4, 10, 6, 3, 7, 4]`。stride 2 →
     `[1, 1, 4, 4]` `[1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4]`。
     `TConvTranspose1D(1, 1, 2)` に `[1, 2, 3]` → `[1, 3, 5, 3]`。stride 2、
     `OutputPadding 1` → `[1, 1, 7]` `[1, 1, 2, 2, 3, 3, 0]`。
     `TConvTranspose3D(2, 3, 3)`：weight `[2, 3, 3, 3, 3]`（`bias [3]` の後に登録）、
     `[1, 2, 4, 4, 4] → [1, 3, 6, 6, 6]`。
  6. `TGroupNorm(2, 4)` に `[1, 2, 3, 4].view([1, 4, 1])` → `[-1, 1, -1, 1]` の
     1e-4 以内（`Eps` 1e-5 でちょうど ±0.99998）。`weight [4]`、`bias [4]` を登録。
     `Affine False` は何も登録せず、それでも正規化する。tch-rs `group_norm_test`：
     `TGroupNorm(2, 10)` に `randn [1, 10, 7]` で形を保つ。
  7. `TGRU` と `TLSTM`、tch-rs の5設定（デフォルト。`HasBiases` False。
     `Bidirectional`。`NumLayers 2`。両方）、`InDim 2`、`Hidden 4`、バッチ 5、
     seq 3：`ZeroState(5)` からの `randn [5, 2]` への `Step` は `H`（と `C`）
     `[LayerDim, 5, 4]`。`randn [5, 3, 2]` への `Seq` は `[5, 3, 4 * NumDirections]`。
     変数数は `4 * NumLayers * NumDirections`（バイアスなしなら `2 *`）。2層双方向の
     ケースの名前は `weight_ih_l0`、`weight_hh_l0`、`bias_ih_l0`、`bias_hh_l0`、
     `weight_ih_l0_reverse`、…、`weight_ih_l1 [12, 8]`（GRU）… `bias_hh_l1_reverse`、
     この順。`BatchFirst False`：`[3, 5, 2]` への `Seq` → `[3, 5, 4]`。
  8. 手計算の数値：`TGRU(1, 1)`、重みすべて 0、`bias_ih_l0 := [0, 0, 1]`
     （`NoGradGuard` 下）、`zeros [1, 1]` への `Step` → `H = 0.380797`
     （`= tanh(1) / 2`：r = z = σ(0) = ½、n = tanh(1)、h' = (1 − z)·n）。
     `TLSTM(1, 1)`、`bias_ih_l0 := [0, 0, 1, 0]` → `C = 0.380797`
     （`= ½·tanh(1)`：i = ½、g = tanh(1)）、`H = 0.181700`（`= ½·tanh(0.380797)`）。
     どちらも 1e-5 以内。
  9. `L.SeqInit(X, S, S)` は `ETorchError`（メッセージに `hx`）。
     `S := L.Step(X, S)` は動く。
  10. `TIdentity.Forward(X).Raw = X.Raw`。`TIdentity` + `AddFn(@Relu)` の
      `TSequential` は ReLU 単体と同じ振る舞い。
  11. レイヤーと store を解放した後、`TchLiveHandles` が開始時の値に戻る。

### 4.7 勾配モードと全体設定

```pascal
function  SetGradEnabled(Enabled: Boolean): Boolean;   // 以前の値を返す。libtorch ではスレッドローカル
function  NoGradGuard: IInterface;                     // 解放されると以前のモードに戻す
procedure ManualSeed(Seed: Int64);
procedure SetNumThreads(N: Integer); function GetNumThreads: Integer;
function  CudaIsAvailable: Boolean; function CudaDeviceCount: Integer;
```

使い方：`var G: IInterface; ... G := NoGradGuard; Loss := Model.Forward(X);`。
**(H11)** FPC はインターフェース型のローカル変数をルーチンの終わり（ブロックの
終わりではない）で解放するので、ガードはルーチンの残り全体に効きます。範囲を狭めたい
ときは、`SetGradEnabled` と `try..finally` を使うのが代わりの方法です。autocast も
後で同じ形にします。
**(H11) 解決**（手順7）：ローカル変数に持ったガードは、ルーチンの最後の文のあと
（内側の `begin..end` ブロックの終わりではない）に解放され、例外でルーチンを抜けるときは
巻き戻しの途中で解放されます。**落とし穴：** 戻り値を代入せずに `NoGradGuard;` と
呼ぶと、ガードは次の文より前にすぐ解放されるので効果がありません。API のコメントに
書くこと。

### 4.8 GPU デバイス：CUDA と ROCm（決定、設計セッション8）

範囲：既存の Pascal API を、libtorch 2.13.0 の CUDA 版または ROCm（HIP）版で、同じ
`TchSys`/`Tch`/`TchNN`/`TchJit` のソースのまま 1 枚の GPU で動かす。最初の対象は手元の
AMD Radeon RX 9070（`gfx1201`）での ROCm。CUDA との違いは展開する libtorch アーカイブ
だけ。**(H31)**〜**(H36)** には ROCm 版 libtorch が必要だった。すべてこの節末尾の
「結果」（ローカルセッション8）で答えが出ており、本文と食い違う箇所は「結果」が優先する。

**ローカルで確認したこと（設計セッション8、2026-10-11）：**
- `rocm-smi --showproductname`：GPU[0] RX 9070、GFX version `gfx1201`；GPU[1] Raphael の
  iGPU、`gfx1036`。`/sys/class/kfd/kfd/topology/nodes/` にも両方ある
  （`gfx_target_version` 120001 と 100306）ので、カーネル側（amdgpu + KFD、kernel 7.0.0）
  は動いている。
- インストール済みの ROCm ユーザー空間は `rocm-smi 7.1.1` と `librocm_smi64`（Ubuntu
  パッケージ）だけ。`/opt/rocm` なし、`libamdhip64` なし、MIOpen なし、
  `rocminfo`/`hipconfig` なし。ユーザーは `render`/`video` グループに入っていないが、
  `/dev/kfd` と `/dev/dri/renderD*` にはシートの ACL が付いていて、デスクトップのログイン
  からは既に読み書きできる。
- `libtorch/include/ATen/Context.h` 638〜650 行（CPU アーカイブのヘッダはどの変種とも
  同じ）：*"HIP masquerades to be CUDA (e.g., when you say CUDA, on a HIP build of ATen,
  this actually means HIP)"*。つまり ROCm 版 libtorch では `at::kCUDA`、
  `device_of_int(d >= 0)`、`torch::cuda::*`、`atc_*`、`atm_fuser_cuda_*`、`TDevice.Cuda(N)`
  がそのまま AMD GPU を指す。以下の API で「CUDA」は「GPU バックエンド」の意味。
- `libtorch/share/cmake/Caffe2/Caffe2Config.cmake`：CPU アーカイブでは
  `if (OFF) include(public/LoadHIP.cmake)`。ROCm アーカイブでは `if (ON)` で、
  `LoadHIP.cmake` は `$ROCM_PATH` か `/opt/rocm` の ROCm SDK を探す（`find_package(hip ...)`）。
  つまり **ROCm 版 libtorch に対する `find_package(Torch)` はランタイムだけでなく ROCm SDK
  の cmake ファイルを要求する**。
- 実測：libtch は **`find_package(Torch)` なしで**ビルド・リンクできる（スクラッチパッドの
  `manual-link/CMakeLists.txt`、28 秒）：インクルード `<dir>/include` と
  `<dir>/include/torch/csrc/api/include`、`-D_GLIBCXX_USE_CXX11_ABI=1`、リンク
  `libtorch.so libtorch_cpu.so libc10.so`、`BUILD_RPATH <dir>/lib`。できた .so の `NEEDED`
  は `libtorch_cpu.so` と `libc10.so` だけ：リンカの `--as-needed` が傘の `libtorch.so` を
  落とした。GPU ビルドではこれが重要：バックエンドは `libtorch_cuda.so` / `libtorch_hip.so`
  にあり、`libtorch.so` 経由でしか引き込まれないので、傘は `-Wl,--no-as-needed` で
  リンクしなければならない（TorchConfig.cmake は `find_package` 利用者に対してまさに
  それをしている）。

#### ビルドとリンク

- **ディレクトリ構成。** 変種ごとに libtorch を CPU 版の隣に置く：`libtorch/`（CPU、
  今のまま）、`libtorch-rocm/`、`libtorch-cuda/`（展開したアーカイブ。`libtorch/` と同様に
  git-ignore）。libtch のビルドディレクトリも変種ごと：`build/libtch`（CPU）、
  `build/libtch-rocm`、`build/libtch-cuda`。ほかは何も複製しない：`libtch.so` のエクスポート
  シンボルはどの変種でも同じで、Pascal ユニットは一切変わらない。
- **`csrc/libtch/CMakeLists.txt` に手動モードを追加**（フォークの変更。ファイル先頭の
  コメントに記す）：

  ```cmake
  option(TCH_LIBTORCH_DIR "libtorch root; when set, link it directly instead of find_package(Torch)" "")
  if(TCH_LIBTORCH_DIR)
    add_library(tch SHARED torch_api.cpp torch_api_generated.cpp)
    target_include_directories(tch PRIVATE ${TCH_LIBTORCH_DIR}/include
      ${TCH_LIBTORCH_DIR}/include/torch/csrc/api/include)
    target_compile_definitions(tch PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
    target_link_libraries(tch -Wl,--no-as-needed ${TCH_LIBTORCH_DIR}/lib/libtorch.so
      -Wl,--as-needed ${TCH_LIBTORCH_DIR}/lib/libtorch_cpu.so ${TCH_LIBTORCH_DIR}/lib/libc10.so)
    set_target_properties(tch PROPERTIES BUILD_RPATH ${TCH_LIBTORCH_DIR}/lib)
  else()
    find_package(Torch REQUIRED)         # 今の経路、CPU アーカイブ
    ...
  endif()
  ```

  既定（`find_package`）は CPU アーカイブ用にそのまま残すので、HANDOFF の既存コマンドは
  動き続ける。GPU 変種は `cmake -S csrc/libtch -B build/libtch-rocm
  -DTCH_LIBTORCH_DIR=$PWD/libtorch-rocm -DCMAKE_BUILD_TYPE=Release` で、ビルド機に ROCm SDK
  は要らない。**(H31)** この手動リンクで動く GPU 版 libtch ができる：変種間のコンパイル時
  の差はヘッダのマクロだけ（`USE_ROCM`/`USE_CUDA` は `torch_api.cpp` には不要。
  `torch::cuda::*` と `at::globalContext()` は実行時に解決されるフックを通る）。ROCm
  アーカイブのヘッダが正しい ABI に define（例 `__HIP_PLATFORM_AMD__`）を要するなら、
  本節のテストプログラムが落ちるので、`if(TCH_LIBTORCH_DIR MATCHES rocm)` の下に追加する。
- **ランタイムライブラリ。** ROCm の `libtorch_hip.so` は `libamdhip64.so`、`libMIOpen.so`、
  `librocblas.so`、`libhipblaslt.so`、`librccl.so`、… を `NEEDED` する。**(H32)**
  `libtorch-cxx11-abi-shared-with-deps` の ROCm アーカイブ（実際の名前は
  `libtorch-shared-with-deps-2.13.0+rocm7.1.zip`。`cxx11-abi` の部分はないが cxx11 ABI）
  はそれらを `lib/` に同梱している
  （ROCm 版 *wheel* は同梱。`ls libtorch-rocm/lib | grep -iE 'hip|miopen|rocblas'` と
  `ldd libtorch-rocm/lib/libtorch_hip.so | grep 'not found'` で確認）。していなければ、
  ユーザーが対応バージョンの AMD ROCm ランタイムパッケージ（`rocm-hip-runtime`、
  `miopen-hip`、`rocblas`、`hipblaslt`、`rccl`、`hipfft`、`hipsparse`、`rocsolver`、
  `hiprand`/`rocrand`）を入れる — アーカイブの ROCm バージョンは名前（`+rocmX.Y`）にあり、
  設計のほかの部分は何も変わらない。どちらにせよカーネルドライバは既にある。
  **(H32) 解決**（ローカルセッション5、`2.13.0+rocm7.1`）：`ldd` で not found なし。
  `libamdhip64`、`libMIOpen`、`librocblas`、`libhipblaslt`、`librccl`、`libhsa-runtime64` が
  `lib/` に同梱。全体の `LD_LIBRARY_PATH` をここに向けないこと（システムのツールが同梱の
  `libtinfo.so.6` を読む）。
  ローカルセッション8で **修正**：zip にはシンボリックリンクがなく、何かを読み込む前に
  8つ足す必要がある（「結果」参照）。
- **Pascal のリンク行は不変**：`external 'tch'`、`-Fl<build dir> -k-rpath=$PWD/<build dir>`；
  `libtch.so` 自身の RUNPATH（cmake が設定）が変種の `lib/` の `libtorch*.so` を見つける。
  変種の切り替え = 別の `-Fl`/`-k-rpath` でテストを再コンパイル、または CPU リンクの
  バイナリの前に `LD_LIBRARY_PATH=build/libtch-rocm`（同じ ABI、同じシンボル — 今
  `build/libtch-unpatched` を差し替えているのと同じやり方）。
- **デバイスの可視性。** iGPU（`gfx1036`）は対応ターゲットではなく `cuda:1` として
  現れてしまうので、GPU テストは必ず `HIP_VISIBLE_DEVICES=0`（CUDA：
  `CUDA_VISIBLE_DEVICES=0`）で走らせ、`CudaDeviceCount = 1` にして下の数値を決定的にする。
  **(H33)** libtorch 2.13 の ROCm アーカイブに `gfx1201` のコードオブジェクトが入っている：
  `strings -n 7 libtorch-rocm/lib/libtorch_hip.so | grep -oE 'gfx[0-9a-f]+' | sort -u` で
  手で確認できる（オフロードバンドルの id は平文）。`gfx1201` がなければこの GPU には
  新しいアーカイブが要る。`HSA_OVERRIDE_GFX_VERSION` の小細工はしない（ISA が違う）。
  **(H33) 解決**（ローカルセッション5）：`2.13.0+rocm7.1` のアーカイブに gfx1201 がある。
  gfx1200、gfx11xx、gfx1030、gfx9xx もあり、gfx1036 はない。
- **C ラッパーと Pascal 層：意味的な変更なし。** `device_of_int`、`at_device`、すべての
  `atc_*` は既に `torch_api.{h,cpp}`（tch-rs のもの）にある。追加するのは下の L2 宣言と
  L3 ラッパーだけ。

#### Pascal の API（`Tch.pas`、`TchSys.pas`、`TchJit.pas`）

既にあるもの：`TDevice`（4.3、`ToC` −1/≥0/−2/−3、`Cuda(N)`、`CudaIfAvailable`、`IsCuda`）、
`TTensor.Device`、生成された `to_device`、`TVarStore.SetDevice`（4.6.2）、store のデバイス
へのロード（4.6.3）、`TJitModule.ToDevice`（4.6.8）、`CudaIsAvailable`、`CudaDeviceCount`
（4.7）、L2 の `atc_cuda_device_count`、`atc_cuda_is_available`。

追加するもの。tch-rs の `Cuda`（`src/wrappers/device.rs`）に従い、`Tch.pas` に名前空間なしの
関数群として（既存の 2 つに合わせて接頭辞 `Cuda`）：

```pascal
// TchSys.pas (L2)、torch_api.h どおり
function  atc_cudnn_is_available: cint; cdecl; external LibTch;      // エラー時 -1
procedure atc_manual_seed(seed: QWord); cdecl; external LibTch;
procedure atc_manual_seed_all(seed: QWord); cdecl; external LibTch;
procedure atc_synchronize(device_index: Int64); cdecl; external LibTch;
function  atc_user_enabled_cudnn: cint; cdecl; external LibTch;      // エラー時 -1
procedure atc_set_user_enabled_cudnn(b: cint); cdecl; external LibTch;
procedure atc_set_benchmark_cudnn(b: cint); cdecl; external LibTch;
function  at_context_has_cuda: Boolean; cdecl; external LibTch;      // C bool = 1 バイト
function  at_context_has_cudart: Boolean; cdecl; external LibTch;    // ローカルセッション8で追加（H35）
function  at_context_has_hip: Boolean; cdecl; external LibTch;
function  at_context_has_cudnn: Boolean; cdecl; external LibTch;
function  at_context_version_cudart: Int64; cdecl; external LibTch;
function  at_context_version_cudnn: Int64; cdecl; external LibTch;
procedure atm_fuser_cuda_set_enabled(b: Boolean); cdecl; external LibTch;
function  atm_fuser_cuda_is_enabled: Boolean; cdecl; external LibTch;

// Tch.pas (L3)、4.7 のブロックはこうなる：
function  CudaIsAvailable: Boolean; function CudaDeviceCount: Integer;   // 既存
function  CudnnIsAvailable: Boolean;                 // atc_cudnn_is_available > 0（ROCm では False：H34 不成立）
procedure CudaManualSeed(Seed: QWord);               // 現在のデバイス
procedure CudaManualSeedAll(Seed: QWord);            // 全デバイス
procedure CudaSynchronize(DeviceIndex: Int64 = -1); // -1 = 現在のデバイス、torch::cuda::synchronize どおり
function  CudnnUserEnabled: Boolean; procedure SetCudnnUserEnabled(B: Boolean);
procedure SetCudnnBenchmark(B: Boolean);
type TGpuBackend = (gbNone, gbCuda, gbRocm);
function  GpuBackend: TGpuBackend;                   // 下記
function  GpuRuntimeVersion: Int64;                  // at_context_version_cudart（ROCm では HIP ランタイム）
function  CudnnVersion: Int64;                       // at_context_version_cudnn
```

- L3 の各関数は C 呼び出しのあと `TchCheck` を呼ぶ（`atc_*` のゲッターは `PROTECT` 内で
  −1 を返し、`TchCheck` が `ETorchNativeError` にする）。CPU ビルドでは（ローカル
  セッション8で *実測*。設計ではもっと多くが例外になると見ていた）例外になるのは
  `CudaSynchronize`（"No CUDA GPUs are available"）、`GpuRuntimeVersion`、`CudnnVersion`
  だけで、ほかは CPU ビルドの値を返す（`CudaIsAvailable` False、`CudaDeviceCount` 0、
  `GpuBackend = gbNone`、`CudnnIsAvailable` False、`CudnnUserEnabled` True、シードと cuDNN
  のスイッチは受け付ける）。`tests/api` がちょうどこれを確認する。GPU テストは今も最初に
  `CudaIsAvailable` を確かめる。
- **`GpuBackend`**（H35 の実測後の実装）：`at_context_has_hip` のとき、または
  `at_context_has_cuda` かつ `at_context_has_cudart` でないとき `gbRocm`。CUDA と CUDART が
  報告されれば `gbCuda`、それ以外 `gbNone`。**(H35)** は「ROCm 版 libtorch で
  `at_context_has_hip` は True」だったが、*実測* では False で、`has_cuda` は True、
  `has_cudart` は False。設計どおりの「has_hip なら gbRocm」では CUDA と報告してしまう。
  規則の CUDA 側はここでは測れない。`ManualSeed`（4.7、`at_manual_seed` = `torch::manual_seed`）
  は libtorch 内で既に全 CUDA デバイスにもシードを入れる；`CudaManualSeedAll` は tch-rs
  との対応のために残す。
- `TDevice`：`CudaIfAvailable(AIndex)` のオーバーロードは追加しない（tch-rs にも
  `cuda_if_available()` しかない）。テスト出力用に `function TDevice.ToString: String`
  （`'cpu'`、`'cuda:0'`、`'mps'`、`'vulkan'`）を追加。tch-rs にはないが費用はゼロ。
- **`TchJit`**：4.6.8 のプロファイリング/フューザーのスイッチの隣に `JitFuserCudaEnabled:
  Boolean` / `SetJitFuserCudaEnabled`；ユニットコメントの「未宣言」リストは空になる。設計では
  CPU ビルドでただのフラグとしていたが、*実測* では libtorch 2.13 のどのビルドでも有効化が
  例外になる（nvfuser は削除済み）。「結果」参照。
- **モック**（`tests/mock/mock_tch.c`）。CPU ビルドの実測の振る舞いに合わせて実装した：
  `atc_cudnn_is_available` → 0、シードと cuDNN のスイッチは何もしない、
  `atc_user_enabled_cudnn` → 1、`at_context_has_*`（`has_cudart` を含む）→ 0。
  `atc_synchronize` と `at_context_version_*` はエラーを設定する。
  `atm_fuser_cuda_set_enabled(true)` はエラーを設定し、`(false)` はフラグを下ろす。
  `mock_test.pas` は `GpuBackend = gbNone`、`CudaIsAvailable` と `CudnnIsAvailable` が
  False、`CudnnUserEnabled` が True、3つが例外になることを確認する。`jit_mock_test.pas` は
  フューザーフラグ（False は読み戻せ、True は例外）を確認する。

#### GPU 上の H13 と H30

- **H13（実デバイスへの移動）。** 4.6.7 の手で設定した重みで `TLinear(VS.Root / 'l', 2, 1)`
  を作ったあと `VS.SetDevice(TDevice.Cuda(0))`：その後 `VS.Device.IsCuda`、
  `L.Weight.Device.ToC = 0`、`VS.Variables` の全要素が `cuda:0` 上、移動の**あとに**作った
  SGD が `w` を CPU と同じ `[0.9, 0.8]` に更新する（`to_device(Cpu)` で読み戻す）。次に
  `SetDevice(Cpu)` で値を保ったまま戻り、`TchLiveHandles` は不変（set_data はストレージを
  差し替えるだけで新しいハンドルは作らない）。移動の*前に*作ったオプティマイザはテストに
  **含めない**：tch-rs が非対応と明記し、4.6.2 もそうしている。
- **H30（cuDNN/MIOpen の RNN 重み平坦化）。** tch-rs の `rnn.rs`（**H36 解決**：取りまとめ側のセッションが
  tch-rs 4227b89b `src/nn/rnn.rs` の 154〜166 行と 239〜251 行で確認。tch-rs はバイアスなしでも
  `4` を渡すので、下の `2` は PyTorch に合わせた明記済みの逸脱）：重みを作ったあと `if vs.device().is_cuda() &&
  Cuda::cudnn_is_available() { Tensor::internal_cudnn_rnn_flatten_weight(&flat_weights, 4,
  input_dim, mode, hidden_dim, proj_size = 0, num_layers, batch_first, bidirectional) }`、
  `mode` 2 = LSTM、3 = GRU（cuDNN `cudnnRNNMode_t`）、`4` = バイアスありのときの層×方向あたり
  の重み数。`TRNN` の protected コンストラクタにまさにその条件を入れ、生成された class
  function `TTensor._cudnn_rnn_flatten_weight(FFlatWeights, WeightStride0, InputSize, Mode,
  HiddenSize, 0, NumLayers, BatchFirst, Bidirectional)` を `WeightStride0 = 4`（`HasBiases`
  のとき、それ以外 `2`）で呼び、結果は捨て、`try ... except on ETorchNativeError do ; end`
  で囲む — 平坦化は性能のヒントであり、ROCm では拒否されうる（`_cudnn_rnn_flatten_weight`
  は cuDNN ディスパッチの op で、MIOpen には独自の `miopen_rnn` がある）。H30 の計測では
  gfx1201 でどちらが起きたかと、4.6.10 の GRU/LSTM の数値（`0.380797`、`0.1817`）が GPU
  でもどちらにせよ再現することを記録する。

#### GPU テスト：`tests/gpu/gpu_test.pas`（+ `run.sh`）

`tests/nn` と同じ作り方だが `build/libtch-rocm`（または `-cuda`）にリンクする（`run.sh` は
変種を `$1` で受け、既定 `rocm`、`HIP_VISIBLE_DEVICES=0` / `CUDA_VISIBLE_DEVICES=0` を設定）。
最初の文：`if not CudaIsAvailable then begin WriteLn('SKIP: no GPU in this libtorch');
Halt(0); end;` — 同じプログラムが `build/libtch`（CPU）の下では 0 で終了するので、回帰
テストの一覧に無条件で入れられる。数値は以前のグループの CPU の数値なので手で確認できる：

1. 能力の印字（検査せず HANDOFF に記録）：`GpuBackend`、`CudaDeviceCount`（可視性変数ありで
   1 のはず）、`CudnnIsAvailable`、`GpuRuntimeVersion`、`CudnnVersion`、
   `at_context_has_cuda`/`has_hip`。ローカルセッション8で項目 1b として追加し、検査する：`CudaDeviceCount = 1`
   （H33）、コマンドラインで指定した変種について `GpuBackend = gbRocm` と
   `CudnnIsAvailable` が False（H35、H34）、または `gbCuda`。
2. `T := TTensor.FromArray([1.0, 2.0, 3.0]).to_device(Cuda(0), kFloat)`：`T.Device.ToC = 0`、
   `T.Device.IsCuda`；`(T * 2 + 1).to_device(Cpu)` → `3 5 7`；`T.sum.to_device(Cpu)` → `6`；
   CUDA テンソルのまま `T.sum.DoubleValue([])` は例外になると見ていたが、*実測* では 6 を
   返す（libtorch がスカラーをホストへコピーする）。テストはそれを確認する。
3. `[64, 64]` の ones の `matmul` のあと `CudaSynchronize` → `to_device(Cpu)` が全部 `64`；
   `CudaManualSeedAll(7); A := randn([4], Cuda(0)); CudaManualSeedAll(7); B := randn([4],
   Cuda(0))` → `A.equal(B)` が True（同シードの再現性。値は手計算しない）。
4. GPU での autograd：`cuda:0` 上の `x := [3.0]` に `requires_grad_`；`(x * x).sum.Backward`；
   `x.grad.to_device(Cpu)` → `6`。
5. 上記の H13（`TLinear`、`SetDevice`、SGD → `[0.9, 0.8]`、CPU へ戻す）。
6. `Cuda(0)` で作った store への 4.6.7 の `.ot` ファイルの `TVarStore.Load`：変数は `cuda:0`
   に到着（`device_id = 0` の `at_load_callback_with_device`）、値は CPU ロードと等しい。
   実装では CPU の store が書いた1変数のファイルを使う（入れ子の名前は GPU では試して
   いない）。
7. JIT：`TJitModule.Create('tests/jit/models/foo1.pt')`、`ToDevice(Cuda(0), kFloat)`、入力を
   `cuda:0` に移して forward → 4.6.8 のグループ 1 と同じ結果を `to_device(Cpu)` で読み戻す。
   `SetJitFuserCudaEnabled(True)` で設定して True を読み戻す予定だったが、*実測* では
   libtorch 2.13 のどのビルドでも例外になる（nvfuser は削除済み）ので、テストは例外を
   確認する。実装では `foo1.pt`（0 次元 Int64）の代わりに `foo.pt`（float 入力2つ）を
   使い、CPU の結果と比べる。
8. H30：`Cuda(0)` 上の VS で `TGRU(…, 1, 1)` と `TLSTM(…, 1, 1)` を 4.6.10 の手計算の
   重みで → `0.380797` / `0.380797, 0.181700` を CPU で読み戻す。平坦化の条件を印字する
   （ROCm では False）。GPU での `TConv2D` グループ 8 の数値（MIOpen の conv 経路）。
   訓練モードの `TBatchNorm` は同じ入力で CPU と比べる（出力と `running_var` が 1e-5 以内）。
9. デバイス混在のエラー：`TTensor.FromArray([1.0]) + T`（CPU + CUDA）はメッセージに
   "Expected all tensors to be on the same device" を含む `ETorchNativeError`；プロセスは
   続行する。
10. 最後に `TchLiveHandles = 0`。

CPU のみでの実行の期待結果：`SKIP` の 1 行だけ。`tests/mock` は影響なし（GPU テストは
モックにリンクしない）。

#### 結果（ローカルセッション8、2026-10-12、RX 9070、`2.13.0+rocm7.1`）

上のとおり実装。`tests/gpu/run.sh rocm` は全項目を通り、`run.sh cpu` は `SKIP` の1行
だけを出す。仮説と設計に対する実測：

- **(H31) 成立。** 手動リンクの libtch は追加の define なしで `libtorch-rocm/` に対して
  ビルドでき（警告は CPU ビルドと同じく libtorch ヘッダ由来の `-Wc++20-extensions` 4件）、
  動作する。
- **(H32) 修正。** `ldd libtorch_hip.so` は問題なかったが、アーカイブ（zip）には
  シンボリックリンクがない：7つのライブラリはバージョンなしのファイル名でしか存在せず、
  SONAME と依存側はバージョン付きの名前を使う。また `libtorch_rocshmem.so` の初期化
  処理はバージョンなしの `libnuma.so` を `dlopen` し、見つからないとプロセスを終了
  させる。展開後の対処（git 管理外のディレクトリ、元に戻せる）：
  `cd libtorch-rocm/lib && for f in libamd_comgr.so libamdhip64.so
  libhipblas.so libhipsparse.so libhsa-runtime64.so librocblas.so
  libroctx64.so; do ln -s $f $(readelf -d $f | sed -n
  's/.*SONAME.*\[\(.*\)\]/\1/p'); done && ln -s libnuma.so.1 libnuma.so`。
  それ以外の `dlopen`（`libibverbs.so`、`libmlx5.so`、`libbnxt_re.so`、
  `libionic.so`、RDMA 用）は失敗しても害はないが、起動時に stderr へ
  `(null): No such file or directory` を4回出す。
- **(H33)** 実行時にも確認：`HIP_VISIBLE_DEVICES=0` で `CudaDeviceCount = 1`、使った
  カーネルはすべて gfx1201 で動く。
- **(H34) 不成立。** ROCm では `CudnnIsAvailable` も `at_context_has_cudnn` も False。
  これらのフック経由では MIOpen は cuDNN の代わりにならない。`CudnnVersion` は
  "Cannot query CuDNN version if ATen_cuda is not built with CuDNN" を出す。よって
  `TRNN` の平坦化の条件は ROCm では False で、呼び出しは行われない（H30：拒否される
  ものがない）。GRU `0.380797`、LSTM `0.380797 / 0.181700` は GPU で再現する。
- **(H35) 判明、設計を変更。** ROCm では `at_context_has_hip` は **False**、
  `at_context_has_cuda` は True なので、「has_hip なら ROCm」では CUDA と報告してしまう。
  そこでは `at_context_has_cudart` は False で、`at_context_version_cudart` は
  "Cannot query CUDART version because CUDART is not available" を出す。したがって
  `GpuBackend` は、HIP が報告されるか CUDART なしで CUDA が報告されるとき `gbRocm`、
  CUDART ありの CUDA で `gbCuda`（L2 に `at_context_has_cudart` を追加）。CUDA 側は
  ここでは測れない。`GpuRuntimeVersion` は ROCm と CPU のビルドでは例外になる。
- **nvfuser。** libtorch 2.13 では CPU を含むどのビルドでも
  `atm_fuser_cuda_set_enabled(true)` が例外になり（"nvfuser support in torchscript is
  removed and cannot be enabled!"）、両アクセサとも非推奨の警告を出す。
  `SetJitFuserCudaEnabled` は互換のため残す：False は何もせず、True は例外。CPU の jit
  テストと GPU の項目 7 がちょうどそれを確認する。
- **CMake。** `option()` は真偽値しか取らないので、`TCH_LIBTORCH_DIR` は `CACHE PATH`
  変数にした。既定の経路は不変（再構成後も CPU の `libtch.so` はビット単位で同一）。
- 項目 2：GPU 上のスカラーへの `DoubleValue` は値（6）を返し、例外にはならない
  （libtorch がホストへコピーする）。GPU 上の H13：新しいハンドルはできず、SGD で
  `[0.9, 0.8]`、CPU へ戻しても値はそのまま。訓練モードの BatchNorm2d は CPU との差が
  2.4e-7。CPU 向けにリンクした `tests/nn` と `tests/jit` は
  `LD_LIBRARY_PATH=build/libtch-rocm` でも通る。`tests/api` はそこでは CPU ビルド用の
  3つの確認が当然ながら失敗する。
- HIP の起動にはプロセスごとにシステム時間で約4秒かかる。

## 5. コード生成（決定）

クラウドセッション3で実装済み。以下はコード（`gen/gen.ml`、`gen/tchgen.pas`）が
実際にしていることで、libtorch 2.13.0 の `Declarations.yaml`（2,657 op）で実測
しています。

- gen.ml（OCaml）はすでに `Declarations-v2.13.0.yaml` を解析しており、
  引数の本当の種類を知っています。フォークでは出力を1つ足します。一覧ファイル
  `gen/ops.tsv` で、`gen.exe <Declarations.yaml> <csrc/libtch のディレクトリ>
  <ops.tsv>` として実行したときに書かれます（このモードでは Rust のファイルは
  書かない。引数なしなら上流と同じ振る舞いで、バイト一致の確認用）。1行に1つの op、
  TAB 区切り、`#` の行はコメントです。

  ```
  name  receiver  result  arg:Type[=default]...  [->ret,ret,...]
  ```

  | 列 | 値 |
  |---|---|
  | name | 公開名。C の関数は `atg_<name>` |
  | receiver | `self`、`input`（Pascal のレシーバーになる Tensor 引数。tch-rs のルール）、または `-`（静的なクラス関数） |
  | result | `nothing`、`fixed<N>`（N = 1..7）、`dynamic`、`bool`、`int64`、`double` |
  | args | C の順番の全引数（レシーバーを含む）。`Type` は gen.ml のコンストラクタ名（`Tensor`、`TensorOption`、`Int64Option`、`IntList`、…、`LayoutOption`）。`=default` は YAML にデフォルトがあるときだけ |
  | `->` | N ≥ 2 の `fixed<N>` だけ。`returns[].name` 由来の N 個の結果名（YAML にないときは `result0..`） |

  デフォルト値の文字列は正規化してあります（gen.ml の `default_text`）。yaml
  ライブラリが素のスカラーに型を付ける（`1` が `Float 1.` になる）ためです：
  `true`/`false`、整数は小数点なし（`-1`、`0`）、それ以外の数は `%.16g`
  （`1e-05`）、文字列は C++ リテラルの引用符も含めてそのまま（`"constant"`、
  `::std::nullopt`、`at::Reduction::Mean`、`{}`）。名前に `:`、`=`、TAB は含まれず、
  デフォルト値に TAB と改行は含まれません（検査済み）。例：

  ```
  add	self	fixed1	self:Tensor	other:Tensor
  max_dim	self	fixed2	self:Tensor	dim:Int64	keepdim:Bool=false	->values,indices
  conv2d	input	fixed1	input:Tensor	weight:Tensor	bias:TensorOption={}	stride:IntList=1	...	groups:Int64=1
  zeros	-	fixed1	size:IntList	options:TensorOptions={}
  split	self	dynamic	self:Tensor	split_size:Int64	dim:Int64=0
  sort_values	self	fixed2	values:Tensor	indices:Tensor	self:Tensor	dim:Int64=-1	descending:Bool=false	->values,indices
  ```

- Pascal の生成器 `gen/tchgen.pas`（FPC のプログラム。`fpc gen/tchgen.pas` か
  `instantfpc`）は `gen/ops.tsv` と `src/Tch.pas` を読み、`src/TchSysGenerated.pas`、
  `src/TchTensorGenerated_intf.inc`、`src/TchTensorGenerated_impl.inc` を書きます。
  Pascal 固有の方針はすべてここが受け持ちます（4.3、4.4）：引数の種類ごとの
  Pascal の型、op 名と引数名の予約語の `&` エスケープ（FPC の予約語と修飾子の表。
  `result` → `result_`。`self` は常にレシーバーだけ）、デフォルト引数の範囲、
  out 引数の有無、nil ハンドルのチェック、Scalar とテンソルリストの一時変数。
  `Tch.pas` の `TTensor = record` 宣言を走査して手書きのメンバー名を集め、次の場合は
  エラーで止まります：op 名がメンバー名と同じ（大文字小文字を無視）、生成する
  ローカル変数（`<引数>__`、`Outs__`、`P__`、`R__`）がメンバー名と同じ、引数名が
  `system`/`tch*`/`ttensor`/`self`、結果名と引数名が一部だけ重なる、変換できない
  デフォルト値、未知の引数の種類。黙って飛ばすことはしません。Python/Node は
  使いません（CLAUDE.md）。
- 再生成（libtorch を上げるときだけ。OCaml はメンテナーの道具）：

  ```
  eval $(opam env --switch=tch-fpc)
  (cd gen && dune build ./gen.exe)
  gen/_build/default/gen.exe path/to/Declarations-v2.XX.yaml csrc/libtch gen/ops.tsv
  fpc -FUbuild/gen -obuild/gen/tchgen gen/tchgen.pas && build/gen/tchgen gen/ops.tsv src
  ```

  クラウドセッションでは opam の `yaml` パッケージが入らなかった（プロキシ）ので、
  同じ gen.ml を、このファイルのブロック構文だけを ocaml-yaml と同じスカラーの
  型付けで読む 150 行の代替 `Yaml` モジュールと組み合わせて動かしました。できた
  `torch_api_generated.{h,cpp}` はコミット済みのものとバイト一致。コミットした
  `ops.tsv` はその実行の出力です。ローカルでは本物のライブラリで一度再生成して
  ください（一致するはず）。
- 生成したファイル（`ops.tsv`、`torch_api_generated.*`、`TchSysGenerated.pas`、
  `.inc`）はコミットするので、OCaml が要るのは libtorch を上げるときだけで、
  tch-fpc のビルドには不要です。2,657 op 入りの `Tch.pas` は 2.7 秒でコンパイル
  できます（48,691 行）。
- `tests/mock`（クラウドセッション3）：C で書いたモックの `libtch.so`（すべての
  `atg_*` を `gen_stubs.sh` でスタブ化して「mock: not implemented」を返し、十数個は
  double 配列の小さなテンソルで実装）と `mock_test.pas`。libtorch なしで生成層を
  動かします：結果1つ、out 引数、結果が入力そのもの、可変個、スカラーの結果、
  Scalar の一時ハンドルの解放、テンソルのリスト（nil 要素は Pascal で、空リストは
  C で拒否）、`IntListOption` の nil と空の区別、String + `DoubleOption`、
  デフォルト引数、エラーの分割、リークのカウンタ。45 項目、`tests/mock/run.sh`。
  配管のテストであって、数値のテストではありません。
- 不採用：`torch_api_generated.h` を直接解析する方法。2 章で挙げた型の区別が
  失われるため。当面は不採用：Pascal で YAML パーサーを書く方法。可能ではあるが、
  gen.ml に積み重なった特例処理を二重に持つことになるため。

## 6. 最初のマイルストーンの範囲

CPU のみ、Linux x86-64、libtorch 2.13.0：
Pascal の配列からテンソルを作る、データを読み戻す、表示する、算術演算子、
生成された op をいくつか、エラー → `ETorchError`、一時値を 10^6 回作るループで
リークしないこと、FPU マスクの確認。その後、自動微分（`at_backward`、
`requires_grad_`）、オプティマイザ、VarStore/nn、JIT モジュール、CUDA の順に進めます。

**状況（ローカルセッション2、2026-10-10）：マイルストーン1完了。** 手書き API は
4.1/4.3/4.7 の草案どおり `src/Tch.pas` に実装（`FromInt32Array`、`CopyData`、
`ZeroGrad`、`Backward(KeepGraph, CreateGraph)`、4.4 のファクトリのオーバーロードを含む）。
テスト：`tests/generated`（本物の libtorch での生成 op）、`tests/api`（手書き API、
要素サイズ）、それに既存の `tests/ownership`、`tests/fpu`、`tests/abi`、`tests/mock`、
`tests/compiler`。自動微分の基本（`requires_grad_`、`Backward`、`grad`、`NoGradGuard`）は
入った。次はオプティマイザ、VarStore/nn（4.6）、JIT、CUDA の順。

**マイルストーン2（クラウドセッション4、2026-10-10 で設計）：** JIT を除く 4.6 の
すべて — `TInit`、`TVarStore`/`TVarPath`、save/load、`TOptimizer`、`TLinear`、
`TConv2D`、`TBatchNorm`、`TLayerNorm`、`TEmbedding`、`TFuncModule`、`TSequential` —
と 4.6.7 の受け入れテスト。実装の順序と新しい仮説 H13〜H18 は HANDOFF_jp.md に
あります。

**状況（ローカルセッション3、2026-10-10）：マイルストーン2完了。** `src/TchNN.pas` が
JIT を除く 4.6 を実装。`tests/nn/nn_test.pas`（本物の libtorch）は 4.6.7 の数値すべてで
通り、H13〜H18 は解決（結果は上の各仮説の横）。配線は `tests/mock/nn_mock_test.pas` が
確認する。次は JIT（`TchJit`）、その後 CUDA。

**マイルストーン3（クラウドセッション5、2026-10-10 で設計）：JIT。** 4.6.8 のとおりの
`src/TchJit.pas` — タグ 0〜13 の `TIValue`、`TJitModule`、`TTrainableJitModule`、トレース、
3関数のフォーク追加による TorchScript のソースからのスクリプト化モジュール、
プロファイリング／fuser のスイッチ — と、`tests/jit/models/` に取り込む tch-rs の
`tests/foo*.pt`、`tests/jit/jit_test.pas`、モックのスタブ。TorchScript のオブジェクト
（タグ 14）と `atm_fuser_cuda_*` は先送り。同じローカルセッションで 4.3 の `FromArray` の
改訂と 4.6.7 の積み残しも適用する。順序と仮説 H19〜H25 は HANDOFF_jp.md にあります。

**状況（ローカルセッション4、2026-10-11）：マイルストーン3完了。** `src/TchJit.pas` が
4.6.8 を実装し、フォーク追加は `csrc/libtch` に。`tests/jit/jit_test.pas` は項目 1〜13 を
通り、H19〜H25 は成立（結果は 4.6.8 の末尾）。配線は `tests/mock/jit_mock_test.pas` が
確認する。`FromArray` の改訂（4.3）と 4.6.7 の積み残しも反映済み。残り：TorchScript の
オブジェクト（タグ 14）、マイルストーン3のレイヤー（4.6.5 の先送りリスト）、CUDA/ROCm。（3つとも
その後、下のローカルセッション6・7・8で実装済み。）

**TorchScript のオブジェクト（設計セッション6、2026-10-11 で設計。ローカルセッション6、2026-10-12 で実装）：** 4.6.9 —
`ivObject = 14`、`TIValue` 内の参照カウント付きハンドル `IJitObject`（`MethodIV`、
`GetAttr`）、`TJitModule.CreateClass`、`JitLiveObjects`、L2 の external 4つ、フォークへの
パッチなし。取り込み済みの `foo6.pt`/`foo7.pt` に対するテスト `jit6`/`jit_double_free` を
`tests/jit/jit_test.pas` のグループ 14 として、加えてモックの `TestObjects`。大半は
libtorch 2.13.0 に対して実測済み。H26〜H28 は成立（結果は 4.6.9）。その後も
残るもの：マイルストーン3のレイヤー（その後ローカルセッション7で実装）と CUDA/ROCm。

**マイルストーン3のレイヤー（設計セッション7、2026-10-11 で設計。ローカルセッション7、
2026-10-12 で実装）：** 4.6.10 — `TConvND` と `TConv1D/2D/3D`（`TConv2D` は基底を付け替え、
API は不変）、転置畳み込みの一族、`TGroupNorm`、`TRNN`/`TLSTM`/`TGRU`、`TIdentity`。
`tests/nn/nn_test.pas` のグループ 11 が実測済みの数値をすべて再現する。H29（tch-rs との
`.ot` の相互運用）は Rust ツールチェーンが要るため未解決のまま。

**CUDA/ROCm（設計セッション8、2026-10-11 で設計。ローカルセッション8、2026-10-12 で実装し RX 9070 で実行。結果は 4.8 の末尾）：** 4.8 — 変種ごとに libtorch ディレクトリ
1つと `build/libtch-<変種>` 1つ、フォークの CMakeLists に `TCH_LIBTORCH_DIR` の手動リンク
モード（CPU アーカイブに対してビルドできることを実測。ROCm SDK の cmake ファイルを不要に
する）、C と ABI の変更なし（HIP は CUDA を装う。ローカルのヘッダで確認）、残っていた
`atc_*` / `at_context_*` / `atm_fuser_cuda_*` の外部宣言とそのラッパー
`Cuda*`/`Cudnn*`/`GpuBackend`、`TRNN` の cuDNN RNN 平坦化呼び出し、CPU ビルドでは `SKIP` を
出して 0 で終了する `tests/gpu/gpu_test.pas`。RX 9070 での実測：H31 成立、H32 修正（展開後に
シンボリックリンク8つ）、H33 確認、H34 不成立、H35 判明（判定規則を変更）、H36 解決、H30 は
ROCm では対象外。CUDA の経路は未実測。これで「引き継ぎ（ローカルセッション4の終了時点）」
の設計項目はすべて実装済み。
