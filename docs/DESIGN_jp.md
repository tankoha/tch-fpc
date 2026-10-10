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
    class function FromArray(const A: array of Int32): TTensor; overload; static;
    class function FromArray(const A: array of Byte): TTensor; overload; static;
    class function FromArray(const A: array of Boolean): TTensor; overload; static;
    class function FromBytes(const Data; const Shape: array of Int64; AKind: TKind): TTensor; static;
    function ToSingleArray: TSingleArray;    // 外へコピー。型も変換する
    function ToDoubleArray: TDoubleArray;
    function ToInt64Array: TInt64Array;
    function DoubleValue(const Index: array of Int64): Double;
    function Int64Value(const Index: array of Int64): Int64;
    // 自動微分
    procedure Backward;
    function RequiresGrad: Boolean;
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

演算子（手書き。生成された op に対応付ける）：

| Pascal | op | 備考 |
|---|---|---|
| `A + B`, `A - B`, `A * B`, `A / B` | `add`, `sub`, `mul`, `div` | `*` は **要素ごと**。行列積は `A.matmul(B)` |
| `A + d`, `A - d`, `A * d`, `A / d` | `add_scalar` など | `d: Double` と `d: Int64` を明示的なオーバーロードとして用意する **(H3)**。`TScalar` の暗黙変換は通さず、`T + 1` と `T + 1.5` がどちらもユーザー定義の変換なしで解決されるようにする |
| `d + A`, `d - A`, `d * A`, `d / A` | 左右逆 | tch-rs と同じく `d - A` = `(-A) + d`、`d / A` = `A.pow(-1) * d` |
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
  置き換えた（意味は同じ、警告なし。**まだコンパイルしていない**。クラウドには
  libtorch がない）。決定：最初のマイルストーンではこのパッチだけ。libtorch 自身の
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
- **`FromArray` のオーバーロード（決定、H2 の続き）。** `array of Single`、
  `array of Double`、`array of Int64`、`array of Byte`、`array of Boolean`。実測で
  `[1, 2]` は `Int64`、`[1.0, 2.0]` は `Single`、`[True, False]` は `Boolean` を選び、
  曖昧にならない。`Int32` はやめる（`Int64` と曖昧、H2）。`Int32`/`Int16`/`Int8` の
  データは `FromInt32s` などで。手書き、最初のマイルストーン。
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

スケッチ：

```pascal
unit TchNN;
type
  TInitKind = (ikConst, ikUniform, ikRandn, ikKaimingUniform, ikKaimingNormal, ikOrthogonal);
  TInit = record
    Kind: TInitKind;
    A, B: Double;                       // 定数値 / lo,up / mean,std / gain
    class function Constant(V: Double): TInit; static;   // "Const" は予約語
    class function Uniform(Lo, Up: Double): TInit; static;
    class function Randn(Mean, Std: Double): TInit; static;
    class function KaimingUniform: TInit; static;        // tch-rs の DEFAULT_KAIMING_UNIFORM
    class function KaimingNormal: TInit; static;
    class function Orthogonal(Gain: Double): TInit; static;
    function Make(const Dims: array of Int64; const Device: TDevice; AKind: TKind): TTensor;
  end;

  TVarStore = class;

  TVarPath = record
  private
    FStore: TVarStore;                  // 所有しない。パスより長く生きている必要がある
    FPath: String;                      // 'encoder.layer1'（'' = ルート）
    FGroup: Integer;
  public
    function Sub(const Name: String): TVarPath;
    class operator /(const P: TVarPath; const Name: String): TVarPath;  // Sub の糖衣
    function SetGroup(G: Integer): TVarPath;
    function Store: TVarStore; inline;
    function Device: TDevice;
    function Kind: TKind;
    function Add(const Name: String; const T: TTensor; Trainable: Boolean): TTensor;
    function Variable(const Name: String; const Dims: array of Int64; const Init: TInit): TTensor; // tch-rs の "var"
    function Zeros(const Name: String; const Dims: array of Int64): TTensor;
    function Ones(const Name: String; const Dims: array of Int64): TTensor;
    function ZerosNoTrain(const Name: String; const Dims: array of Int64): TTensor;
    function OnesNoTrain(const Name: String; const Dims: array of Int64): TTensor;
    function Randn(const Name: String; const Dims: array of Int64; Mean, Std: Double): TTensor;
    function Uniform(const Name: String; const Dims: array of Int64; Lo, Up: Double): TTensor;
    function KaimingUniform(const Name: String; const Dims: array of Int64): TTensor;
    function VarCopy(const Name: String; const T: TTensor): TTensor;
    function TryGet(const Name: String; out T: TTensor): Boolean;
    function GetOrAdd(const Name: String; const Dims: array of Int64; const Init: TInit): TTensor;
  end;

  TNamedTensor = record Name: String; Tensor: TTensor; end;
  TNamedTensorArray = array of TNamedTensor;

  TVarStore = class
  public
    constructor Create(const ADevice: TDevice; AKind: TKind = kFloat);
    function Root: TVarPath;
    function TrainableVariables: TTensorArray;
    function Variables: TNamedTensorArray;
    procedure Save(const FileName: String);          // at_save_multi。tch-rs と同じく '.' -> '|'
    procedure Load(const FileName: String);          // 既存の変数へコピー。名前が欠けていれば例外
    function LoadPartial(const FileName: String): TStringArray;  // 欠けていた名前を返す
    procedure Freeze; procedure Unfreeze;
    procedure SetDevice(const ADevice: TDevice);
    procedure SetKind(AKind: TKind);
    procedure CopyFrom(Src: TVarStore);
    property Device: TDevice read FDevice;
    property Kind: TKind read FKind;
    property Count: Integer read GetCount;
  end;

  TModule = class abstract
  public
    function Forward(const X: TTensor; Train: Boolean = False): TTensor; virtual; abstract;
  end;

  TLinearConfig = record WsInit: TInit; BsInit: TInit; HasBsInit: Boolean; Bias: Boolean; end;
  function DefaultLinearConfig: TLinearConfig;

  TLinear = class(TModule)
  public
    Weight, Bias: TTensor;              // バイアスがないレイヤーでは Bias.IsNil
    constructor Create(const P: TVarPath; InDim, OutDim: Int64); overload;
    constructor Create(const P: TVarPath; InDim, OutDim: Int64; const Cfg: TLinearConfig); overload;
    function Forward(const X: TTensor; Train: Boolean = False): TTensor; override;
  end;

  TTensorFunc = function(const X: TTensor): TTensor;
  TTensorFuncT = function(const X: TTensor; Train: Boolean): TTensor;

  TSequential = class(TModule)
  private
    FLayers: TObjectList;               // OwnsObjects = True
  public
    function Add(M: TModule): TSequential;            // チェーン用に Self を返す
    function AddFn(F: TTensorFunc): TSequential;
    function AddFnT(F: TTensorFuncT): TSequential;
    function Forward(const X: TTensor; Train: Boolean = False): TTensor; override;
  end;

  TOptimizer = class
  public
    class function Sgd(VS: TVarStore; LR: Double; Momentum: Double = 0;
      Dampening: Double = 0; WeightDecay: Double = 0; Nesterov: Boolean = False): TOptimizer; static;
    class function Adam(VS: TVarStore; LR: Double; Beta1: Double = 0.9;
      Beta2: Double = 0.999; WeightDecay: Double = 0; Eps: Double = 1e-8;
      AmsGrad: Boolean = False): TOptimizer; static;
    class function AdamW(...): TOptimizer; static;
    class function RmsProp(...): TOptimizer; static;
    destructor Destroy; override;                      // ato_free
    procedure ZeroGrad;
    procedure Step;
    procedure BackwardStep(const Loss: TTensor);       // zero_grad; backward; step
    procedure BackwardStepClip(const Loss: TTensor; MaxValue: Double);
    procedure BackwardStepClipNorm(const Loss: TTensor; MaxNorm: Double);
    procedure ClipGradValue(MaxValue: Double);
    procedure ClipGradNorm(MaxNorm: Double);
    procedure SetLearningRate(LR: Double);             // + グループ別、momentum、weight decay
  private
    FOpt: PTorchOptimizer;
    FStore: TVarStore;                                 // 所有しない
    FVarsInOptimizer: Integer;                         // tch-rs と同じく新しい学習対象変数を遅延追加
  end;
```

決定事項と理由：
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
  PyTorch 風にモジュールに `Training` フラグを持たせる案も検討しましたが、当面は不採用です。
  フラグを伝播させるには子の登録の仕組み（Python では `__setattr__` の魔法）が必要で、
  Pascal ではそれを全部書き下すことになるからです。
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
  （`TIValue.Tuple([...])`）とタグが合わなければ例外を投げる `As*` アクセサを持ちます。
  **(H10)** 自分自身の型の動的配列を含む record が 3.2.2 でコンパイルできる必要があります。
  だめなら fpjson 風のクラス階層にします。**(H10) 解決**（手順7）：コンパイルでき、
  入れ子（タプルのタプル）も作れる。ドキュメントに書くべき意味論：`W := V` は `Items` を
  共有する（動的配列は参照カウントで、コピーオンライトではない）ので、`W.Items[0]` に
  書くと `V` も変わる。`TIValue` は作ったあとは変更しないものとして扱う。`TTrainableJitModule` は `NamedParameters` を
  `TVarPath` に登録し、`TModule` を継承するので、スクリプト化したモデルも他のレイヤーと
  同じく `TOptimizer` で微調整できます。

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
