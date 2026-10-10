# tch-fpc 設計

（[DESIGN.md](DESIGN.md) の日本語訳です。内容が食い違う場合は英語版が正です。）

決定事項とその理由をまとめます。未解決の論点と次の作業は
[HANDOFF_jp.md](HANDOFF_jp.md) にあります。

以下はすべて tch-rs のコミット
`4227b89b72059b0651ff83a38637693574e0a2d6`（2026-08-23、crate 0.26.0）を基準にしています。
このコミットは **libtorch 2.13.0** 向けで、
`third_party/pytorch/Declarations-v2.13.0.yaml` からコードを生成しています。

## 0. 全体像

```
 ユーザーコード (Pascal)
      |
 [L3] Tch           TTensor（管理型 record）, TDevice, TKind, ETorchError,
      |             生成された op メソッド。後で TchNN / TchOptim / TchJit
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
  約 2,650 個の op をカバーしています。PyTorch の `Declarations.yaml` から
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
- **tch-rs の C の関数名と呼び出しの形は変えません**（`at_*`、`atg_*`、`ats_*`、
  `ato_*`、`atm_*`、`ati_*`）。上流との差分を小さく保てるので、libtorch の
  バージョンを上げる作業は「tch-rs から取り込み直し、こちらのパッチを当て直す」で
  済みます。
- フォークから削除するもの：ストリーム関数（`*_to_stream`、`*_from_stream*`、
  `tch_*_stream_*`）、`torch_python.*`、画像関数（`at_load_image*`、
  `at_save_image`、`at_resize_image`）と stb ヘッダー。必要になれば Pascal で
  扱いやすい形で戻せます（HANDOFF 参照）。
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
  解放する必要があります。
- エラー：すべての入口は本体を `PROTECT(...)` で包んでいます。これは
  `std::exception`（`c10::Error` を含む）を捕まえ、`strdup(e.what())` を
  `thread_local char *torch_last_err` に入れます。
  `get_and_reset_last_err()` はそれを返してクリアし、文字列の所有権は呼び出し側に移ります。
- 生成された op（`atg_*`）の形：
  - 結果のテンソル数が固定：`void atg_X(tensor *out__, ...)`。呼び出し側が
    N 個分の配列を渡し、C 側が `out__[0..N-1]` を埋めます。
  - 結果の数が可変（`split` など）：`tensor *atg_X(...)` が `malloc` で確保した
    `nullptr` 終端の配列を返します。呼び出し側は各要素の所有権を受け取ったあと、
    配列だけを解放します（テンソルは解放しない）。該当する op は 33 個です。
  - スカラーの結果：`bool` / `int64_t` / `double` を直接返します。
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
  | Layout / LayoutOption | `int8_t` | `-1` = なし |

- 手書きの `torch_api.h` では、フラグに `int` と C++ の `bool` が混在しています
  （例：`at_autocast_is_enabled` は `bool` を返す）。Pascal 側は1つずつ正確に
  宣言する必要があります（`bool` には `Boolean`/`ByteBool` サイズ、`int` には `cint`）。

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
  `LoadLibrary` で 2,650 個の関数ポインタを取る方式は使いません。
- 名前は C と完全に同じにします（`atg_add`、`at_free`）。引数名が予約語
  （`out`、`end`、`to` など）の場合、生成器が `&` でエスケープする（`&end`）か
  `_` を付けます（`end_`）。引数名は ABI に影響しません。

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

function  get_and_reset_last_err: PChar; cdecl; external LibTch;
procedure tchf_free(p: Pointer); cdecl; external LibTch;

function  at_new_tensor: PTorchTensor; cdecl; external LibTch;
function  at_tensor_of_data(vs: Pointer; dims: PInt64; ndims: csize_t;
            element_size_in_bytes: csize_t; type_: cint): PTorchTensor;
            cdecl; external LibTch;
procedure at_copy_data(t: PTorchTensor; vs: Pointer; numel: csize_t;
            element_size_in_bytes: csize_t); cdecl; external LibTch;
function  at_shallow_clone(t: PTorchTensor): PTorchTensor; cdecl; external LibTch;
function  at_dim(t: PTorchTensor): csize_t; cdecl; external LibTch;
procedure at_shape(t: PTorchTensor; dims: PInt64); cdecl; external LibTch;
function  at_scalar_type(t: PTorchTensor): cint; cdecl; external LibTch;
function  at_device(t: PTorchTensor): cint; cdecl; external LibTch;
function  at_to_string(t: PTorchTensor; line_size: cint): PChar; cdecl; external LibTch;
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
```

## 4. Pascal の L3 層：Tch（オブジェクト指向）

### 4.1 テンソルの所有権：インターフェースを包む管理型 record（決定）

```pascal
type
  ITensorHandle = interface
    function Raw: PTorchTensor;
  end;

  TTensorHandle = class(TInterfacedObject, ITensorHandle)
  private
    FRaw: PTorchTensor;
  public
    constructor Create(ARaw: PTorchTensor);  // 所有権を受け取る
    destructor Destroy; override;            // at_free(FRaw)
    function Raw: PTorchTensor;
  end;

  TTensor = record
  private
    FHandle: ITensorHandle;   // nil = 未定義のテンソル
  public
    // 手書きの API（Pascal 流の名前）
    class function FromRaw(ARaw: PTorchTensor): TTensor; static;
    function Raw: PTorchTensor; inline;      // 未定義なら nil
    function Defined: Boolean; inline;
    function Dim: Integer;
    function Shape: TInt64Array;
    function Kind: TKind;
    function Device: TDevice;
    function ToString: AnsiString;
    function ShallowClone: TTensor;
    class operator +(const A, B: TTensor): TTensor;
    class operator -(const A, B: TTensor): TTensor;
    class operator *(const A, B: TTensor): TTensor;
    class operator /(const A, B: TTensor): TTensor;
    // 生成された op メソッド
    {$I TchTensorGenerated_intf.inc}
  end;
```

この形にした理由：
- **解放のタイミングが決まる。** インターフェースの参照カウントにより、Pascal 側の
  最後の参照が消えた時点で C++ のテンソル（と GPU メモリ）が解放されます。
  tch-rs の `Drop` と同じです。素の `class` だと一時値のたびに
  `try..finally Free` が必要になり、`a.Add(b).Mul(c)` では途中の値がリークします。
- **式が書ける。** `{$mode objfpc}{$modeswitch advancedrecords}` の record は
  `class operator` による演算子オーバーロードに対応しています。`(a + b) * c` の
  途中結果はコンパイラが管理する一時値になり、解放されます。
- **FPC 3.2.2 で堅牢。** インターフェースのフィールドを持つ record は、昔からある
  インターフェースの参照カウント機構でコンパイラが管理します。FPC の比較的新しい
  *管理演算子*（`class operator Initialize/Finalize/Copy`）は、3.2.x で一時値や
  関数の戻り値まわりにバグがあったので使いません。（記憶に頼らず、テストで確認すること。）
- **参照カウントがスレッドセーフ。** `TInterfacedObject` はアトミックな
  増減（interlocked increment/decrement）を使います。
- **「なし」が自然に表せる。** `Default(TTensor)`（ハンドルが nil）が、そのまま
  `TensorOption = nullptr` に対応します。

意味論：
- `B := A` は同じ C++ の `torch::Tensor` オブジェクトを共有します。libtorch の
  テンソル自体が共有ストレージへの参照カウント付きハンドルなので、C++ や Python の
  ユーザーが期待するエイリアシングと同じです。`ShallowClone`（`at_shallow_clone`）は
  同じストレージへの独立したハンドルを返します。`Copy`/`Clone`（生成された op）は
  データをコピーします。
- インプレース op は PyTorch と同じく末尾にアンダースコアを付け（`Add_`）、
  tch-rs と同様にチェーンできるよう同じテンソルを返します。
- 早めに解放したいとき：`A := Default(TTensor);`。

コスト：テンソル1つにつきヒープ確保が2回（C++ の `torch::Tensor` と Pascal の
`TTensorHandle`）。当面は許容し、最適化は計測してから行います。

### 4.2 エラー処理（決定）

- API は1種類だけにします（tch-rs の `f_add`/`add` のような、失敗を返す版と
  返さない版は分けない）。Pascal には例外があるので、すべての op は失敗時に例外を投げます。
- `ETorchError = class(Exception)`。L3 は libtch を呼ぶ **たびに** 次を呼びます。

  ```pascal
  procedure CheckTorch; inline;
  var P: PChar;
  begin
    P := get_and_reset_last_err;
    if P <> nil then RaiseTorchError(P);  // メッセージをコピーし、tchf_free(P) してから raise
  end;
  ```

  `RaiseTorchError` はインライン化しない別手続きにして、通常の経路を小さく保ちます。
  エラーの格納場所は C 側でスレッドローカルで、同じスレッドの直後に読むので、
  スレッド同士が干渉しません。
- エラー時は、すでに埋まった `out__` のスロットを解放してから例外を投げます
  （複数出力の配列を途中まで埋めたあとで PROTECT が失敗することがあるため）。
- **Pascal の例外を C++ のフレームをまたいで伝わらせない。** libtch に渡すコールバック
  （`at_load_callback` など）は `cdecl` の中継関数にし、中ですべての例外を捕まえて
  保存し、C の呼び出しから戻ったあとで投げ直します。
- **FPU の例外マスク。** x86 の FPC プログラムは浮動小数点例外が有効なまま
  動きます。libtorch/MKL/oneDNN は C の既定（例外がマスクされた状態）を前提にしていて、
  NaN や Inf を正当に生成します。ユニット `Tch` の `initialization` で
  `SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide, exOverflow,
  exUnderflow, exPrecision])` を呼びます。libtorch 自身のスレッドプールが作る
  ワーカースレッドには影響せず、問題になるのはメインスレッドと Pascal が作った
  スレッドです。CPU でゼロ除算するテストで確認すること。

### 4.3 小さな値型

```pascal
type
  TKind = (kUInt8 = 0, kInt8 = 1, kInt16 = 2, kInt = 3, kInt64 = 4,
           kHalf = 5, kFloat = 6, kDouble = 7, kComplexHalf = 8,
           kComplexFloat = 9, kComplexDouble = 10, kBool = 11,
           kQInt8 = 12, kQUInt8 = 13, kQInt32 = 14, kBFloat16 = 15);
           // 値は libtorch 2.13 の c10::ScalarType。頼る前に
           // c10/core/ScalarType.h と照合すること

  TDeviceType = (dtCpu, dtCuda, dtMps, dtVulkan);
  TDevice = record
    DeviceType: TDeviceType;
    Index: Integer;               // CUDA のみ
    function ToC: cint;           // -1 / >=0 / -2 / -3
    class function Cpu: TDevice; static;
    class function Cuda(AIndex: Integer = 0): TDevice; static;
    class function CudaIfAvailable: TDevice; static;
  end;

  TScalar = record                // 呼び出しごとに一時的な ats_* ハンドルに変換する
    IsFloat: Boolean;
    I: Int64; F: Double;
    class operator :=(V: Int64): TScalar;
    class operator :=(V: Double): TScalar;
  end;

  TOptInt64  = record HasValue: Boolean; Value: Int64;  class operator :=(V: Int64): TOptInt64;  end;
  TOptDouble = record HasValue: Boolean; Value: Double; class operator :=(V: Double): TOptDouble; end;
  TOptKind   = record HasValue: Boolean; Value: TKind;  class operator :=(V: TKind): TOptKind;   end;
  // Default(TOptXxx) = なし。ジェネリクスにせず手書きにする：ジェネリックな record の
  // 暗黙変換演算子で 3.2.2 が予想外の動きをする余地を減らすため。

  TInt64Array  = array of Int64;
  TTensorArray = array of TTensor;
```

生成器が使う引数の対応：

| gen.ml の型 | Pascal の引数 |
|---|---|
| Tensor | `const X: TTensor`（定義済みであること。チェックする） |
| TensorOption | `const X: TTensor`（未定義 = なし） |
| Scalar | `const X: TScalar` |
| Bool | `X: Boolean` |
| Int64 / Double | `X: Int64` / `X: Double` |
| Int64Option / DoubleOption | `const X: TOptInt64` / `TOptDouble` |
| IntList | `const X: array of Int64`（`[2, 3]` と書ける） |
| IntListOption | `const X: TOptIntList`（未決、HANDOFF 参照） |
| DoubleList | `const X: array of Double` |
| TensorList / TensorOptList | `const X: array of TTensor` |
| String | `const X: AnsiString` |
| TensorOptions | `AKind: TKind; const ADevice: TDevice` |
| ScalarType / ScalarTypeOption | `TKind` / `TOptKind` |
| Device | `const X: TDevice` |

戻り値：テンソル1つ → `TTensor`、可変個 → `TTensorArray`、固定 N 個 →
`out` 引数（4.4 参照）、`bool`/`int64_t`/`double` → `Boolean`/`Int64`/`Double`。

### 4.4 生成する op メソッドと命名（決定）

- 第1引数が `self: Tensor` の `atg_*` は `TTensor` のメソッドにし、それ以外は
  `TTensor` の `class function ... static` にします
  （例：`TTensor.zeros([2, 3], kFloat, TDevice.Cpu)`）。tch-rs の
  `Tensor::zeros` に合わせた形です。
- **op 名は tch-rs/PyTorch のスネークケースのまま使います**
  （`t.max_pool2d(...)`、`t.add_(u)`、`t.sum_dim_intlist(...)`）。
  手書きの API は Pascal 流のキャメルケースにします（`Shape`、`Kind`、`ToString`）。
  理由：スネークケースからキャメルケースへの変換は情報が落ちる（`_foo`/`foo`/`foo_` が
  衝突する）、元の名前のままなら PyTorch や tch-rs のドキュメントで検索できる、
  そして呼び出し箇所で「生成か手書きか」が見分けられる。
- tch-rs 4227b89 で確認済み：2,657 個の `atg_*` 名には大文字が含まれず、
  大文字小文字を無視しても重複がありません。したがって op 名同士では Pascal が
  大文字小文字を区別しないことは問題になりません。ただし生成器は、生成する名前と
  手書きのメンバー名との衝突を、大文字小文字を無視してチェックする必要があります。
- 予約語：op の `div`、`to`、`set`、`repeat`、`absolute` と、引数名の `out`
  （819 箇所）、`end`、`to` が Pascal のキーワードと衝突します。メソッドは FPC の
  `&` エスケープを使い（`t.&div(u)`、`t.&to(dev)`）、引数には `_` を付けます
  （`end_`）。必要な箇所には、手書きの分かりやすい別名（`DivBy`、`ToDevice`）を
  上に足せます。
- 固定個数の複数テンソルを返す op（例：`max_dim` → values と indices）は
  `procedure max_dim(Dim: Int64; KeepDim: Boolean; out Values, Indices: TTensor)`
  にします。出力の名前は、`Declarations.yaml` の `returns[].name` があればそれを使い、
  なければ `Result0..N` にします。
- デフォルト引数：Pascal は単純な型の末尾の引数にしかデフォルト値を付けられません。
  最初の版では gen.ml がすでに与えているもの以外はデフォルト値を付けず、
  便利なオーバーロードは後で手書きします。

### 4.5 ユニット構成

```
src/
  TchSys.pas                 L2 手書きの external 宣言
  TchSysGenerated.pas        L2 生成された external 宣言          （生成）
  Tch.pas                    L3 中核：TTensor, TDevice, TKind, ETorchError
  TchTensorGenerated_intf.inc  TTensor 内のメソッド宣言           （生成）
  TchTensorGenerated_impl.inc  メソッドの本体                     （生成）
  後で：TchNN.pas (VarStore, Linear, Conv...), TchOptim.pas, TchJit.pas
csrc/libtch/                 フォークした C ラッパー + CMakeLists.txt
gen/                         gen.ml（フォーク）+ Pascal 生成器
```

生成する record のメンバーは record の中で宣言しなければならないので、
include ファイルを2つに分けています。

## 5. コード生成（決定）

- gen.ml（OCaml）はすでに `Declarations-v2.13.0.yaml` を解析しており、
  引数の本当の種類を知っています。これをフォークし、出力をもう1つ増やします。
  1行に1つの op を書く一覧ファイル `gen/ops.tsv` で、例えば次のような形です。

  ```
  add	method	fixed1	self:Tensor	other:Tensor
  max_dim	method	fixed2	self:Tensor	dim:Int64	keepdim:Bool	->values,indices
  zeros	function	fixed1	size:IntList	options:TensorOptions
  split	method	dynamic	self:Tensor	split_size:Int64	dim:Int64
  ```

- Pascal の生成器は FPC のプログラム（`gen/tchgen.pas`、`instantfpc` で実行可能）で、
  `ops.tsv` を読んで `TchSysGenerated.pas` と2つの `.inc` ファイルを書き出します。
  Python/Node は使いません（CLAUDE.md）。
- 生成したファイルはコミットするので、OCaml が要るのは libtorch を上げるときだけで、
  tch-fpc のビルドには不要です。
- 不採用：`torch_api_generated.h` を直接解析する方法。2 章で挙げた型の区別が
  失われるため。当面は不採用：Pascal で YAML パーサーを書く方法。可能ではあるが、
  gen.ml に積み重なった特例処理を二重に持つことになるため。

## 6. 最初のマイルストーンの範囲

CPU のみ、Linux x86-64、libtorch 2.13.0：
Pascal の配列からテンソルを作る、データを読み戻す、表示する、算術演算子、
生成された op をいくつか、エラー → `ETorchError`、一時値を 10^6 回作るループで
リークしないこと。その後、自動微分（`at_backward`、`requires_grad_`）、
オプティマイザ、VarStore/nn、JIT モジュール、CUDA の順に進めます。
