# tch-fpc の UML

（[UML.md](UML.md) の日本語訳です。内容が食い違う場合は英語版が正です。）

実装済みのもの（マイルストーン1〜3、TorchScript のオブジェクト、マイルストーン3の
レイヤー、ROCm での GPU デバイス）の図です。まだ予定なのは CUDA ビルドだけで（破線）、
NVIDIA の機械が要ります。時点：2026-10-12、ローカルセッション8の後。正は
DESIGN.md で、このファイルはそれを図にしただけです。

目次：
1. [層とユニット](#1-層とユニット)
2. [Tch：テンソルの中核](#2-tchテンソルの中核)
3. [TchNN：ストア、モジュール、オプティマイザ](#3-tchnnストアモジュールオプティマイザ)
4. [TchNN：マイルストーン3のレイヤー](#4-tchnnマイルストーン3のレイヤー)
5. [TchJit：値とモジュール](#5-tchjit値とモジュール)
6. [シーケンス：ForwardIV と ivalue の所有権](#6-シーケンスforwardiv-と-ivalue-の所有権)
7. [シーケンス：TorchScript モジュールのファインチューニング](#7-シーケンスtorchscript-モジュールのファインチューニング)
8. [配置：CPU、ROCm、CUDA のビルド](#8-配置cpurocmcuda-のビルド)
9. [状況](#9-状況)

## 1. 層とユニット

C ラッパーは tch-rs の libtch のフォーク。L2 は生の `external` 宣言、L3 はオブジェクト層
（DESIGN 0、3、4.5）。

```mermaid
flowchart TB
  subgraph Native["C++ / C"]
    LT["libtorch 2.13.0<br/>(CPU and ROCm builds; CUDA planned, 4.8)"]
    LTCH["libtch.so<br/>torch_api.cpp (fork of tch-rs 4227b89b)<br/>torch_api_generated.cpp (gen.ml)<br/>+ tchf_free, atm_create / atm_register_parameter / atm_define"]
  end
  subgraph L2["Pascal L2 (raw)"]
    SYS["TchSys<br/>hand-written externals:<br/>at_*, atc_*, ats_*, ato_*, atm_*, ati_*"]
    SYSG["TchSysGenerated<br/>2,657 atg_* externals (generated)"]
  end
  subgraph L3["Pascal L3 (object layer)"]
    TCH["Tch<br/>TTensor, TDevice, TKind, TScalar,<br/>ETorchError, grad mode<br/>+ TchTensorGenerated_intf/impl.inc"]
    NN["TchNN<br/>TInit, TVarStore, TVarPath, TModule,<br/>layers, TSequential, TOptimizer"]
    JIT["TchJit<br/>TIValue, TJitModule,<br/>TTrainableJitModule"]
  end
  APP["User program / tests"]
  LT --> LTCH
  LTCH --> SYS
  LTCH --> SYSG
  SYS --> TCH
  SYSG --> TCH
  TCH --> NN
  NN --> JIT
  TCH --> JIT
  JIT --> APP
  NN --> APP
  TCH --> APP
```

`A --> B` は「B が A を使う」。`TchJit` が `TchNN` を使うのは `TModule`、`TVarPath`、
`TNamedTensorArray`（`TTrainableJitModule`）だけ。

## 2. Tch：テンソルの中核

`TTensor` は参照カウント付きインターフェースを包む管理型レコードなので、最後のコピーが
消えた時点で決定的に解放される（DESIGN 4.1）。生成された 2,657 個の op は `TTensor` の
メソッド（4.4）で、図には描いていない。

```mermaid
classDiagram
  direction LR
  class TTensor {
    <<record>>
    -FHandle: ITensorHandle
    +FromRaw(ARaw) TTensor$
    +Raw() PTorchTensor
    +IsNil() Boolean
    +Shape() TInt64Array
    +Kind() TKind
    +Device() TDevice
    +FromArray(A, AKind) TTensor$
    +FromSingleArray(A, AKind) TTensor$
    +ToDoubleArray() TDoubleArray
    +CopyFrom(Src)
    +Backward()
    +operators + - * /
    +2657 generated ops
  }
  class ITensorHandle {
    <<interface>>
    +Raw() PTorchTensor
  }
  class TTensorHandle {
    -FRaw: PTorchTensor
    +Create(ARaw)
    +Destroy() at_free
  }
  class TDevice {
    <<record>>
    +DeviceType: TDeviceType
    +Index: Integer
    +ToC() cint
    +Cpu() TDevice$
    +Cuda(AIndex) TDevice$
    +CudaIfAvailable() TDevice$
  }
  class TKind {
    <<enumeration>>
    kNone = -1
    kUInt8 .. kFloat4e2m1fnx2
  }
  class TScalar {
    <<record>>
    +IsFloat: Boolean
    +I: Int64
    +F: Double
  }
  class TOptInt64 {
    <<record>>
  }
  class TOptDouble {
    <<record>>
  }
  class TOptIntList {
    <<record>>
  }
  class ETorchError {
    <<exception>>
  }
  class ETorchNativeError {
    <<exception>>
    +Detail: String
  }
  class TGradModeGuard {
    returned by NoGradGuard
  }
  TTensor *-- ITensorHandle : shares
  ITensorHandle <|.. TTensorHandle
  TTensor ..> TDevice
  TTensor ..> TKind
  TTensor ..> TScalar
  TTensor ..> TOptInt64
  TTensor ..> TOptDouble
  TTensor ..> TOptIntList
  ETorchError <|-- ETorchNativeError
  TGradModeGuard ..|> IInterface
```

4.8 の GPU 関数（同じユニットの関数）：`CudnnIsAvailable`、`CudaManualSeed(All)`、
`CudaSynchronize`、`CudnnUserEnabled` / `SetCudnnUserEnabled`、`SetCudnnBenchmark`、
`GpuBackend: TGpuBackend (gbNone, gbCuda, gbRocm)`、`GpuRuntimeVersion`、`CudnnVersion`、
`TDevice.ToString`。`GpuBackend` は「HIP、または CUDART なしの CUDA」で `gbRocm`：ROCm
ビルドは hasHIP False を報告する（実測、H35）。

## 3. TchNN：ストア、モジュール、オプティマイザ

`TVarStore` が変数を所有し、`TVarPath` は軽い名前のスコープ。レイヤーはパス経由でテンソル
を登録してハンドルを共有し、`TOptimizer` はストアの学習対象変数を読み、後から増えた変数を
遅延追加する（DESIGN 4.6.1〜4.6.7）。

```mermaid
classDiagram
  class TInit {
    <<record>>
    +Kind: TInitKind
    +A, B: Double
    +Dist, Fan, NonLin, Gain
    +Constant(V)$
    +Uniform(Lo, Up)$
    +Randn(Mean, Std)$
    +KaimingUniform()$
    +Orthogonal(Gain)$
    +Make(Dims, Device, Kind) TTensor
    +Apply(T)
  }
  class TVarStore {
    -FVars: TVarEntryArray
    -FTrainable: array of Integer
    -FIndex: TNameIndex
    +Create(Device, Kind)
    +Root() TVarPath
    +Variables() TNamedTensorArray
    +TrainableVariables() TTensorArray
    +RollbackTo(ACount)
    +Save(FileName)
    +Load(FileName)
    +LoadPartial(FileName) TStringArray
    +Freeze() / Unfreeze()
    +SetDevice(Device)
    +SetKind(Kind) / SetFloatKind(Kind)
    +CopyFrom(Src)
  }
  class TVarPath {
    <<record>>
    -FStore: TVarStore
    -FPath: String
    -FGroup: Integer
    +Sub(Name) TVarPath
    +operator /(P, Name) TVarPath
    +Add(Name, T, Trainable) TTensor
    +Zeros / Ones / Randn / KaimingUniform ...
    +VarCopy(Name, T) TTensor
    +GetOrAdd(Name, Dims, Init) TTensor
  }
  class TVarEntry {
    <<record>>
    +Name: String
    +Tensor: TTensor
    +Trainable: Boolean
    +Group: Integer
  }
  class TModule {
    <<abstract>>
    +Forward(X, Train) TTensor*
  }
  class TLinear {
    +Weight, Bias: TTensor
  }
  class TConv2D {
    +Weight, Bias: TTensor
    -FCfg: TConvConfig
  }
  class TBatchNorm {
    +Weight, Bias, RunningMean, RunningVar
    -FNd: Integer
  }
  class TLayerNorm {
    +Weight, Bias: TTensor
  }
  class TEmbedding {
    +Weight: TTensor
  }
  class TFuncModule {
    function or method, with or without Train
  }
  class TSequential {
    -FLayers: TModuleList
    +Add(M) TSequential
    +AddFn(F) TSequential
    +ForwardAll(X, Train) TTensorArray
  }
  class TOptimizer {
    -FOpt: PTorchOptimizer
    -FStore: TVarStore
    -FVarsInOptimizer: Integer
    +Sgd / Adam / AdamW / RmsProp$
    +ZeroGrad()
    +Step()
    +BackwardStep(Loss)
    +BackwardStepClip / ClipNorm
    +SetLearningRate / SetMomentum / SetWeightDecay (+Group)
  }
  TVarStore *-- "many" TVarEntry
  TVarPath --> TVarStore : not owned
  TVarPath ..> TInit
  TModule <|-- TLinear
  TModule <|-- TConv2D
  TModule <|-- TBatchNorm
  TModule <|-- TLayerNorm
  TModule <|-- TEmbedding
  TModule <|-- TFuncModule
  TModule <|-- TSequential
  TSequential o-- "many" TModule : owns
  TLinear ..> TVarPath : registers via
  TOptimizer --> TVarStore : reads trainable, not owned
```

寿命の規則（DESIGN 4.6）：先にオプティマイザ、次にレイヤー、最後にストアを解放する。

## 4. TchNN：マイルストーン3のレイヤー

DESIGN 4.6.10。`TConv2D` は公開 API を変えずに共通の `TConvND` の下にある。RNN は tch-rs
と同じく `TModule` ではない。

```mermaid
classDiagram
  class TModule {
    <<abstract>>
    +Forward(X, Train) TTensor*
  }
  class TConvConfigND {
    <<record>>
    +Stride, Padding, Dilation: TInt64Array
    +Groups: Int64
    +Bias: Boolean
    +WsInit, BsInit: TInit
    +PaddingMode: TPaddingMode
  }
  class TConvND {
    -FNd: Integer
    -FCfg: TConvConfigND
    +Weight, Bias: TTensor
    +Create(P, InDim, OutDim, KSizes, Cfg)
    +Forward(X, Train) TTensor
  }
  class TConv1D {
  }
  class TConv2D {
    existing API, body moves to TConvND
  }
  class TConv3D {
  }
  class TConvTransposeND {
    -FCfg: TConvTransposeConfigND
    +Weight, Bias: TTensor
  }
  class TConvTranspose1D {
  }
  class TConvTranspose2D {
  }
  class TConvTranspose3D {
  }
  class TGroupNorm {
    +Weight, Bias: TTensor
    +NumGroups, NumChannels: Int64
  }
  class TIdentity {
    Forward returns X
  }
  class TRNNState {
    <<record>>
    +H, C: TTensor
  }
  class TRNNConfig {
    <<record>>
    +HasBiases, Bidirectional, BatchFirst
    +NumLayers: Int64
    +Dropout: Double
  }
  class TRNN {
    <<abstract>>
    -FFlatWeights: TTensorArray
    -FHiddenDim: Int64
    +ZeroState(BatchDim) TRNNState*
    +Step(Input, State) TRNNState
    +Seq(Input, out State) TTensor
    +SeqInit(Input, InState, out OutState) TTensor*
  }
  class TLSTM {
  }
  class TGRU {
    C stays nil
  }
  TModule <|-- TConvND
  TConvND <|-- TConv1D
  TConvND <|-- TConv2D
  TConvND <|-- TConv3D
  TConvND ..> TConvConfigND
  TModule <|-- TConvTransposeND
  TConvTransposeND <|-- TConvTranspose1D
  TConvTransposeND <|-- TConvTranspose2D
  TConvTransposeND <|-- TConvTranspose3D
  TModule <|-- TGroupNorm
  TModule <|-- TIdentity
  TRNN <|-- TLSTM
  TRNN <|-- TGRU
  TRNN ..> TRNNState
  TRNN ..> TRNNConfig
```

cuDNN のある GPU 上のストアでは、`TRNN` のコンストラクタが
`_cudnn_rnn_flatten_weight` も呼ぶ（4.8、H30）。ROCm には cuDNN がないので呼ばない。

## 5. TchJit：値とモジュール

`TIValue` はタグ付きレコードで、タグの値は `ati_tag` と同じ。`ToC` は呼び出し側が解放する
C の ivalue を作り、`FromC` は受け取った ivalue を消費する（DESIGN 4.6.8）。タグ 14 は
参照カウント付きのオブジェクトハンドルが保持する（4.6.9）。

```mermaid
classDiagram
  class TIValue {
    <<record>>
    +Tag: TIValueTag
    +Tensor: TTensor
    +I: Int64
    +D: Double
    +B: Boolean
    +S: String
    +Ints, Doubles, Bools, Strings, Tensors
    +Items: TIValueArray
    +Dev: TDevice
    +Obj: IJitObject
    +Tuple / GenericList / GenericDict ...$
    +operator := from TTensor, Int64, Double, Boolean, String
    +AsTensor() / AsInt() / AsString() ...
    +Item(I) / Key(I) / Value(I)
    +ToC() PTorchIValue
    +FromC(P) TIValue$
  }
  class TIValueTag {
    <<enumeration>>
    ivNone=0 .. ivTensorList=10
    ivGenericList=12
    ivGenericDict=13
    ivObject=14
    ivStringList=20 input only
    ivDevice=21 input only
  }
  class TJitModule {
    -FModule: PTorchModule
    +Create(FileName, Device)
    +CreateFromStream(S, Device)
    +CreateEmpty(Name)
    +CreateByTracing(Name, FnName, Inputs, Fn)$
    +Forward(Inputs) TTensor
    +ForwardIV(Inputs) TIValue
    +Method(Name, Inputs) TTensor
    +MethodIV(Name, Inputs) TIValue
    +SetEval() / SetTrain()
    +ToDevice(Device, Kind)
    +Save(FileName)
    +NamedParameters() TNamedTensorArray
    +RegisterParameter(Name, T, IsBuffer)
    +Define(Source)
    +CreateClass(QualifiedName, Inputs) IJitObject
  }
  class TTrainableJitModule {
    -FInner: TJitModule
    -FMode: TJitMode
    +Create(FileName, P: TVarPath)
    +Forward(X, Train) TTensor
    +ForwardIV / Method / MethodIV
    +Save(FileName)
    +SetTrain() / SetEval()
  }
  class TModule {
    <<abstract>>
  }
  class IJitObject {
    <<interface>>
    +Raw() PTorchIValue
    +MethodIV(Name, Inputs) TIValue
    +GetAttr(Name) TIValue
  }
  class TJitObjectHandle {
    -FRaw: PTorchIValue
    +Destroy() ati_free
  }
  TIValue --> TIValueTag
  TIValue o-- "many" TIValue : Items
  TIValue ..> IJitObject
  IJitObject <|.. TJitObjectHandle
  TModule <|-- TTrainableJitModule
  TTrainableJitModule *-- TJitModule : owns
  TJitModule ..> TIValue
  TJitModule ..> IJitObject : CreateClass
```

グローバル：`JitGetProfilingMode` / `JitSetProfilingMode`、
`JitGetTensorExprFuserEnabled` / `JitSetTensorExprFuserEnabled`、
`JitSetGraphExecutorOptimize`、`JitLiveObjects`、
`JitFuserCudaEnabled` / `SetJitFuserCudaEnabled`（4.8。有効化は例外：libtorch 2.13 で
nvfuser は削除済み）。

## 6. シーケンス：ForwardIV と ivalue の所有権

入力側で作った C の ivalue はすべて `finally` で解放され、結果は `FromC` が消費して
どの経路でも解放する（DESIGN 4.6.8）。子の変換が失敗すると、作り終えた兄弟は解放される。

```mermaid
sequenceDiagram
  participant U as Caller
  participant M as TJitModule
  participant V as TIValue
  participant C as libtch (atm_ / ati_)
  U->>M: ForwardIV([X, 42, 1.5])
  Note over U,M: implicit := turns each element into a TIValue
  loop each input
    M->>V: ToC
    V->>C: ati_tensor / ati_int / ati_double
    C-->>M: new C ivalue (into a nil-initialised slot)
  end
  M->>C: atm_forward_(module, slots, n)
  C-->>M: result ivalue
  M->>C: ati_free on every slot (finally)
  M->>V: FromC(result)
  V->>C: ati_tag, ati_to_* (children into nil-initialised slots)
  V->>C: ati_free(result) (finally)
  V-->>U: TIValue (Pascal-owned copy)
```

タグ 14 では `FromC` がポインタを解放せず `TJitObjectHandle` に持たせ、
`ToC` は `ati_clone(Obj.Raw)` を渡す。

## 7. シーケンス：TorchScript モジュールのファインチューニング

ストアの項目はモジュールのパラメータそのもの（共有ハンドル）なので、ストア上で
オプティマイザを1ステップ回すとスクリプト化モジュールがその場で変わる（DESIGN 4.6.8、
H21 解決済み）。

```mermaid
sequenceDiagram
  participant U as Caller
  participant VS as TVarStore
  participant J as TTrainableJitModule
  participant M as TJitModule
  participant O as TOptimizer
  U->>J: Create('model.pt', VS.Root / 'jit')
  J->>M: Create(file, P.Device)
  J->>M: NamedParameters
  M-->>J: (weight, T1), (bias, T2)
  J->>VS: P.Add('weight', T1, T1.RequiresGrad), P.Add('bias', ...)
  U->>O: Sgd(VS, 0.1)
  loop training
    U->>J: Forward(X, True)
    J->>M: SetTrain (only when the mode changes)
    J->>M: Forward([X])
    M-->>U: prediction
    U->>O: BackwardStep(loss)
    O->>VS: add missing trainable variables, then zero_grad, backward, step
    Note over VS,M: same TensorImpl: the module sees the update
  end
  U->>J: Save('trained.pt')
```

## 8. 配置：CPU、ROCm、CUDA のビルド

DESIGN 4.8：変種ごとに展開した libtorch と libtch のビルドディレクトリを1つずつ持つ。
Pascal のユニット、C ラッパー、ABI は変わらない。ROCm 版 libtorch は AMD GPU を CUDA として
見せるので、`TDevice.Cuda(0)` で選べる。

```mermaid
flowchart LR
  subgraph Archives["unpacked libtorch (repo root, git-ignored)"]
    CPU["libtorch/<br/>2.13.0 CPU (in use)"]
    ROCM["libtorch-rocm/<br/>2.13.0+rocm7.1 (in use;<br/>8 symlinks added after unpacking)"]
    CUDA["libtorch-cuda/<br/>2.13.0+cuXYZ (planned)"]
  end
  subgraph Builds["libtch builds"]
    B1["build/libtch<br/>find_package(Torch)"]
    B2["build/libtch-rocm<br/>-DTCH_LIBTORCH_DIR manual link,<br/>--no-as-needed libtorch.so"]
    B3["build/libtch-cuda<br/>same manual link"]
  end
  P["Pascal units (unchanged)<br/>Tch, TchNN, TchJit"]
  T["tests/gpu/gpu_test.pas<br/>SKIP when CudaIsAvailable is False"]
  HW["RX 9070 (gfx1201)<br/>HIP_VISIBLE_DEVICES=0 hides the iGPU"]
  CPU --> B1
  ROCM --> B2
  CUDA --> B3
  B1 --> P
  B2 --> P
  B3 --> P
  P --> T
  B2 -.-> HW
  classDef planned stroke-dasharray: 5 5
  class CUDA,B3 planned
```

## 9. 状況

| 領域 | DESIGN | 状態 |
|---|---|---|
| C ラッパーのフォーク、生成器、L2 | 1、2、3、5 | 実装済み（マイルストーン1） |
| `TTensor`、データの入出力、演算子、自動微分、`FromArray` の改訂 | 4.1〜4.4、4.7 | 実装済み（マイルストーン1、ローカルセッション4） |
| `TInit`、`TVarStore`、保存と読み込み、オプティマイザ、マイルストーン2のレイヤー | 4.6.1〜4.6.7 | 実装済み（マイルストーン2） |
| JIT：`TIValue`、`TJitModule`、`TTrainableJitModule` | 4.6.8 | 実装済み（マイルストーン3）、H19〜H25 成立 |
| TorchScript のオブジェクト（`IJitObject`） | 4.6.9 | 実装済み（ローカルセッション6、2026-10-12）、H26〜H28 成立 |
| Conv1D/3D、ConvTranspose、GroupNorm、LSTM/GRU、Identity | 4.6.10 | 実装済み（ローカルセッション7、2026-10-12）。H29（tch-rs との `.ot` の相互運用）は未解決、Rust が要る |
| GPU：ROCm / CUDA のビルドと API | 4.8 | 実装済み（ローカルセッション8、2026-10-12）。ROCm は RX 9070 で動作、H31・H33 成立、H32 修正（展開後にシンボリックリンク8つ）、H34 不成立、H35 判明。CUDA は未実測 |
