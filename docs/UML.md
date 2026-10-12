# tch-fpc in UML

Japanese version: [UML_jp.md](UML_jp.md)

Diagrams of what is implemented (milestones 1–3) and what is designed but
not implemented yet (DESIGN 4.8 GPU devices).
Planned elements carry the `planned` stereotype or the dashed style.
State: 2026-10-12, after local session 7. DESIGN.md stays the source of
truth; this file only draws it.

Contents:
1. [Layers and units](#1-layers-and-units)
2. [Tch: tensor core](#2-tch-tensor-core)
3. [TchNN: store, modules, optimizer](#3-tchnn-store-modules-optimizer)
4. [TchNN: milestone 3 layers](#4-tchnn-milestone-3-layers)
5. [TchJit: values and modules](#5-tchjit-values-and-modules)
6. [Sequence: ForwardIV and ivalue ownership](#6-sequence-forwardiv-and-ivalue-ownership)
7. [Sequence: fine-tuning a TorchScript module](#7-sequence-fine-tuning-a-torchscript-module)
8. [Deployment: CPU, ROCm and CUDA builds (planned)](#8-deployment-cpu-rocm-and-cuda-builds-planned)
9. [Status](#9-status)

## 1. Layers and units

The C wrapper is the fork of tch-rs's libtch; L2 is raw `external`
declarations, L3 the object layer (DESIGN 0, 3, 4.5).

```mermaid
flowchart TB
  subgraph Native["C++ / C"]
    LT["libtorch 2.13.0<br/>(CPU build; ROCm / CUDA planned, 4.8)"]
    LTCH["libtch.so<br/>torch_api.cpp (fork of tch-rs 4227b89b)<br/>torch_api_generated.cpp (gen.ml)<br/>+ tchf_free, atm_create / atm_register_parameter / atm_define"]
  end
  subgraph L2["Pascal L2 (raw)"]
    SYS["TchSys<br/>hand-written externals:<br/>at_*, ats_*, ato_*, atm_*, ati_*"]
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

`A --> B` reads "B uses A". `TchJit` uses `TchNN` only for `TModule`,
`TVarPath` and `TNamedTensorArray` (`TTrainableJitModule`).

## 2. Tch: tensor core

`TTensor` is a managed record around a reference-counted interface, so a
tensor is freed deterministically when its last copy goes (DESIGN 4.1).
The 2,657 generated ops are methods of `TTensor` (4.4), not drawn.

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

Planned in 4.8 (same unit, free functions): `CudnnIsAvailable`,
`CudaManualSeed(All)`, `CudaSynchronize`, `CudnnUserEnabled` /
`SetCudnnUserEnabled`, `SetCudnnBenchmark`, `GpuBackend: TGpuBackend
(gbNone, gbCuda, gbRocm)`, `GpuRuntimeVersion`, `CudnnVersion`,
`TDevice.ToString`.

## 3. TchNN: store, modules, optimizer

A `TVarStore` owns the variables; `TVarPath` is a cheap naming scope; layers
register their tensors through a path and share the handles; a
`TOptimizer` reads the store's trainable variables and adds new ones lazily
(DESIGN 4.6.1–4.6.7).

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

Lifetime rule (DESIGN 4.6): free optimizers first, then layers, then the
store.

## 4. TchNN: milestone 3 layers

DESIGN 4.6.10. `TConv2D` sits under a common `TConvND` with its public
API unchanged; RNNs are not `TModule`s, as in tch-rs.

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

On a GPU store, `TRNN`'s constructor also calls
`_cudnn_rnn_flatten_weight` (4.8, H30).

## 5. TchJit: values and modules

`TIValue` is a tagged record whose tags equal `ati_tag`'s; `ToC` makes a
C ivalue the caller frees, `FromC` consumes one (DESIGN 4.6.8). Tag 14 is held
by the refcounted object handle (4.6.9).

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

Globals: `JitGetProfilingMode` / `JitSetProfilingMode`,
`JitGetTensorExprFuserEnabled` / `JitSetTensorExprFuserEnabled`,
`JitSetGraphExecutorOptimize`, `JitLiveObjects`; planned:
`JitFuserCudaEnabled` / `SetJitFuserCudaEnabled` (4.8).

## 6. Sequence: ForwardIV and ivalue ownership

Every C ivalue made on the way in is freed in a `finally`; the result is
consumed by `FromC`, which frees it on every path (DESIGN 4.6.8). A failing
child conversion frees the siblings already made.

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

For tag 14, `FromC` keeps the pointer in a `TJitObjectHandle`
instead of freeing it, and `ToC` hands out `ati_clone(Obj.Raw)`.

## 7. Sequence: fine-tuning a TorchScript module

The store entries are the module's parameters (shared handles), so an
optimizer step on the store changes the scripted module in place
(DESIGN 4.6.8, H21 resolved).

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

## 8. Deployment: CPU, ROCm and CUDA builds (planned)

DESIGN 4.8: one unpacked libtorch and one libtch build directory per
variant; the Pascal units, the C wrapper and the ABI do not change. A ROCm
libtorch reports the AMD GPU as CUDA, so `TDevice.Cuda(0)` selects it.

```mermaid
flowchart LR
  subgraph Archives["unpacked libtorch (repo root, git-ignored)"]
    CPU["libtorch/<br/>2.13.0 CPU (in use)"]
    ROCM["libtorch-rocm/<br/>2.13.0+rocm7.x (planned, H31-H35)"]
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
  class ROCM,CUDA,B2,B3,T,HW planned
```

## 9. Status

| Area | DESIGN | State |
|---|---|---|
| C wrapper fork, generator, L2 | 1, 2, 3, 5 | implemented (milestone 1) |
| `TTensor`, data in/out, operators, autograd, `FromArray` revision | 4.1–4.4, 4.7 | implemented (milestone 1, local session 4) |
| `TInit`, `TVarStore`, save/load, optimizers, milestone 2 layers | 4.6.1–4.6.7 | implemented (milestone 2) |
| JIT: `TIValue`, `TJitModule`, `TTrainableJitModule` | 4.6.8 | implemented (milestone 3), H19–H25 hold |
| TorchScript objects (`IJitObject`) | 4.6.9 | implemented (local session 6, 2026-10-12) |
| Conv1D/3D, ConvTranspose, GroupNorm, LSTM/GRU, Identity | 4.6.10 | implemented (local session 7, 2026-10-12) |
| GPU: ROCm / CUDA builds and API | 4.8 | designed (session 8), H31–H35 open; needs a ROCm libtorch |
