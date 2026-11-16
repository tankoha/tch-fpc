# tch-fpc in UML

Japanese version: [UML_jp.md](UML_jp.md)

Diagrams of what is implemented (milestones 1–3, TorchScript objects, the
milestone 3 layers, GPU devices on ROCm) and of what is planned (dashed):
the CUDA build, which needs an NVIDIA machine, the knowledge package of
DESIGN 7, and the fine-tuning sample of DESIGN 8.
State: 2026-11-17, after design session 10. DESIGN.md stays the source of
truth; this file only draws it.

Contents:
1. [Layers and units](#1-layers-and-units)
2. [Tch: tensor core](#2-tch-tensor-core)
3. [TchNN: store, modules, optimizer](#3-tchnn-store-modules-optimizer)
4. [TchNN: milestone 3 layers](#4-tchnn-milestone-3-layers)
5. [TchJit: values and modules](#5-tchjit-values-and-modules)
6. [Sequence: ForwardIV and ivalue ownership](#6-sequence-forwardiv-and-ivalue-ownership)
7. [Sequence: fine-tuning a TorchScript module](#7-sequence-fine-tuning-a-torchscript-module)
8. [Deployment: CPU, ROCm and CUDA builds](#8-deployment-cpu-rocm-and-cuda-builds)
9. [Knowledge package (planned)](#9-knowledge-package-planned)
10. [Fine-tuning sample (planned)](#10-fine-tuning-sample-planned)
11. [Status](#11-status)

## 1. Layers and units

The C wrapper is the fork of tch-rs's libtch; L2 is raw `external`
declarations, L3 the object layer (DESIGN 0, 3, 4.5).

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

GPU functions of 4.8 (same unit, free functions): `CudnnIsAvailable`,
`CudaManualSeed(All)`, `CudaSynchronize`, `CudnnUserEnabled` /
`SetCudnnUserEnabled`, `SetCudnnBenchmark`, `GpuBackend: TGpuBackend
(gbNone, gbCuda, gbRocm)`, `GpuRuntimeVersion`, `CudnnVersion`,
`TDevice.ToString`. `GpuBackend` is `gbRocm` for "HIP, or CUDA without
CUDART": a ROCm build reports hasHIP False (measured, H35).

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

On a GPU store with cuDNN, `TRNN`'s constructor also calls
`_cudnn_rnn_flatten_weight` (4.8, H30); ROCm has no cuDNN, so not there.

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
`JitSetGraphExecutorOptimize`, `JitLiveObjects`,
`JitFuserCudaEnabled` / `SetJitFuserCudaEnabled` (4.8; enabling raises,
nvfuser is removed in libtorch 2.13).

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

## 8. Deployment: CPU, ROCm and CUDA builds

DESIGN 4.8: one unpacked libtorch and one libtch build directory per
variant; the Pascal units, the C wrapper and the ABI do not change. A ROCm
libtorch reports the AMD GPU as CUDA, so `TDevice.Cuda(0)` selects it.

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

## 9. Knowledge package (planned)

DESIGN 7: a plugin with one skill, loaded by Claude Code through the
repository's marketplace file and by Hermes through
`skills.external_dirs`. No Pascal types are added. The fine-tuning sample
of DESIGN 8 is drawn in section 10.

```mermaid
flowchart LR
  subgraph Repo["tch-fpc repository"]
    MP[".claude-plugin/marketplace.json<br/>source: ./plugin"]
    subgraph Plugin["plugin/"]
      PJ[".claude-plugin/plugin.json"]
      subgraph Skill["skills/tch-fpc/"]
        SK["SKILL.md<br/>rules, tables, doc index"]
        REF["reference/*.md"]
        EX["examples/ (.pas + .expected)"]
        PIT["pitfalls/ (must not compile)"]
        TR["traps/ (.pas + .expected)"]
        BE["scripts/build-example.sh"]
      end
    end
    DOCS["docs/, src/ (indexed, not copied)"]
    CHK["tests/skill/check-examples.sh<br/>tests/skill/check-index.sh"]
    TASKS["tests/skill/tasks/<br/>ten tasks with checkers (H39)"]
  end
  CC["Claude Code<br/>/plugin install tch-fpc"]
  HE["Hermes Agent<br/>skills.external_dirs"]
  CI["CI: libtorch (CPU) job"]
  MP --> Plugin
  CC --> MP
  HE --> Skill
  SK --> REF
  SK --> EX
  SK --> PIT
  SK --> TR
  SK -. "path + pinned URL" .-> DOCS
  CHK --> BE
  CHK --> EX
  CHK --> PIT
  CHK --> TR
  CI --> CHK
  classDef planned stroke-dasharray: 5 5
  class MP,PJ,SK,REF,EX,PIT,TR,BE,CHK,TASKS planned
```

## 10. Fine-tuning sample (planned)

DESIGN 8.9–8.19: the sample `samples/qwen-lora/` that LoRA-fine-tunes
Qwen3.8 27B on the GGUF's packed weights. Everything in this section is
planned; nothing exists yet.

Units and what they need (DESIGN 8.12). The left group is Pascal only, the
right group uses `Tch` / `TchNN`.

```mermaid
flowchart LR
  subgraph P["Pascal only"]
    GF["GgufFile<br/>read / write"]
    GQ["GgufQuant<br/>scalar Q4_K Q6_K Q8_0"]
    TK["QwenTokenizer"]
    CH["QwenChat"]
    DV["DialectVerify"]
  end
  subgraph T["on Tch / TchNN"]
    GQT["GgufQuantT"]
    CFG["QwenConfig"]
    PL["PackedLinear"]
    LO["Lora"]
    DR["DeltaRule"]
    BL["QwenBlocks"]
    MD["QwenModel"]
    RC["Recompute"]
    OP["LoraOptim"]
    TD["TrainData"]
    CK["Checkpoint"]
    MG["GgufMerge"]
  end
  subgraph X["Programs"]
    PI["qlora_inspect"]
    PT["qlora_tokenize"]
    PLG["qlora_logprobs"]
    PV["qlora_verify"]
    PD["qlora_dataset"]
    PTR["qlora_train"]
    PM["qlora_merge"]
    PE["qlora_eval"]
  end
  GQT --> GQ
  CFG --> GF
  PL --> GQT
  PL --> LO
  BL --> PL
  BL --> DR
  MD --> BL
  MD --> CFG
  CH --> TK
  TK --> GF
  TD --> CH
  CK --> LO
  CK --> OP
  MG --> GF
  MG --> GQT
  MG --> LO
  PI --> GF
  PT --> TK
  PLG --> MD
  PV --> DV
  PD --> DV
  PD --> TK
  PTR --> MD
  PTR --> RC
  PTR --> OP
  PTR --> TD
  PTR --> CK
  PM --> MG
  PE --> DV
  classDef planned stroke-dasharray: 5 5
  class GF,GQ,TK,CH,DV,GQT,CFG,PL,LO,DR,BL,MD,RC,OP,TD,CK,MG,PI,PT,PLG,PV,PD,PTR,PM,PE planned
```

The model and training types (DESIGN 8.13, 8.14).

```mermaid
classDiagram
  class TStage {
    <<abstract>>
    +Device() TDevice
    +Forward(X) TTensor
  }
  class TLinearAttnBlock {
    qkv, gate, out : TPackedLinear
    beta, alpha : TPackedLinear
    conv, dt, a, norms : TTensor
  }
  class TFullAttnBlock {
    q, k, v, output : TPackedLinear
    qNorm, kNorm, norms : TTensor
  }
  class TPackedLinear {
    packed : TTensor
    quantType
    +Forward(X, ComputeKind) TTensor
  }
  class TLoraPair {
    A, B : TTensor
    scale
    +Delta(X, ComputeKind) TTensor
    +MergedDelta() TTensor
  }
  class TLoraSet {
    one TVarStore per device
    +Save() +Load()
  }
  class TSavedInputs {
    onHost : Boolean
  }
  class Recompute {
    <<unit>>
    +RecomputeForward(Stages, X0, Saved) TTensor
    +RecomputeBackward(Stages, Saved, GradOut)
  }
  class TLoraAdamW {
    moments in a TVarStore per device
    +Step() +ZeroGrad()
    +ClipGlobalNorm(Max)
  }
  class TQwenModel {
    stages : TStage[64]
    +EmbedRows(Ids) TTensor
    +HeadLoss(H, Targets, Mask) loss, dH
    +LogProbs(Ids) TTensor
  }
  TStage <|-- TLinearAttnBlock
  TStage <|-- TFullAttnBlock
  TLinearAttnBlock o-- TPackedLinear
  TFullAttnBlock o-- TPackedLinear
  TPackedLinear o-- TLoraPair : optional
  TLoraSet o-- TLoraPair
  TQwenModel o-- TStage
  Recompute ..> TStage
  Recompute ..> TSavedInputs
  TLoraAdamW ..> TLoraSet
```

One training step on the two GPUs (DESIGN 8.14).

```mermaid
sequenceDiagram
  participant D as TrainData
  participant G0 as GPU 0 (embedding, blocks 0–35)
  participant H as host memory
  participant G1 as GPU 1 (blocks 36–63, head)
  participant O as LoraOptim
  D->>G0: token ids, loss mask
  Note over G0,G1: RecomputeForward, NoGrad
  loop block 0..35
    G0->>H: save block input
    G0->>G0: dequantise, forward
  end
  G0->>G1: hidden state (&to)
  loop block 36..63
    G1->>H: save block input
    G1->>G1: dequantise, forward
  end
  Note over G1: head: output matrix dequantised 8192 rows at a time,<br/>loss over 256-position slices, dL/dh
  Note over G0,G1: RecomputeBackward
  loop block 63..36
    H->>G1: saved input, requires grad
    G1->>G1: forward with graph, backward, keep x.grad
  end
  G1->>G0: gradient (&to)
  loop block 35..0
    H->>G0: saved input, requires grad
    G0->>G0: forward with graph, backward, keep x.grad
  end
  G0->>O: LoRA gradients (store 0)
  G1->>O: LoRA gradients (store 1)
  O->>O: every 2nd batch: clip over both stores, AdamW, zero gradients
```

From data specification to the deployed model (DESIGN 8.15–8.17).

```mermaid
flowchart TD
  SPEC["data-spec/<br/>(Opus/Sonnet)"] --> GEN["Mistral Large 3<br/>generate, compile, repair"]
  GEN --> RAW["raw JSONL (outside the repo)"]
  RAW --> VER["qlora_verify<br/>compiled-run / fpc-approx / static-only"]
  VER --> DS["qlora_dataset<br/>dedup, leak check, split, card"]
  TASKS["eval/tasks/ (held out)"] -. "leak check" .-> DS
  DS --> TRAINSET["work/dataset-v1<br/>train / valid"]
  DS --> HOLD["Oxygene, static-only:<br/>stored, held back"]
  TRAINSET --> TRAIN["qlora_train<br/>base GGUF Q4_K_M + LoRA"]
  TRAIN --> CKPT["checkpoints"]
  CKPT --> MERGE["qlora_merge<br/>adapted tensors as Q8_0"]
  MERGE --> GGUF["merged GGUF, 20.08 GB"]
  GGUF --> H47["llama-server (H47, rehearsal model)"]
  GGUF --> OLL["ollama create, new name (H45)<br/>second FROM: the projector"]
  TASKS --> EVAL["qlora_eval<br/>before / after, thinking on / off"]
  OLL --> EVAL
  BASE["original model in Ollama"] --> EVAL
  EVAL --> REP["report: verified aggregate,<br/>Delphi (FPC approx.), Oxygene (unverified),<br/>regression, MTP speed (H49)"]
  classDef planned stroke-dasharray: 5 5
  class SPEC,GEN,RAW,VER,DS,TASKS,TRAINSET,HOLD,TRAIN,CKPT,MERGE,GGUF,H47,OLL,EVAL,REP planned
```
## 11. Status

| Area | DESIGN | State |
|---|---|---|
| C wrapper fork, generator, L2 | 1, 2, 3, 5 | implemented (milestone 1) |
| `TTensor`, data in/out, operators, autograd, `FromArray` revision | 4.1–4.4, 4.7 | implemented (milestone 1, local session 4) |
| `TInit`, `TVarStore`, save/load, optimizers, milestone 2 layers | 4.6.1–4.6.7 | implemented (milestone 2) |
| JIT: `TIValue`, `TJitModule`, `TTrainableJitModule` | 4.6.8 | implemented (milestone 3), H19–H25 hold |
| TorchScript objects (`IJitObject`) | 4.6.9 | implemented (local session 6, 2026-10-12), H26–H28 hold |
| Conv1D/3D, ConvTranspose, GroupNorm, LSTM/GRU, Identity | 4.6.10 | implemented (local session 7, 2026-10-12); H29 (`.ot` interchange with tch-rs) open, needs Rust |
| GPU: ROCm / CUDA builds and API | 4.8 | implemented (local session 8, 2026-10-12); ROCm runs on the RX 9070, H31 and H33 hold, H32 amended (8 symlinks after unpacking), H34 false, H35 answered; CUDA unmeasured |
| Knowledge package: plugin, skill, verified examples | 7 | designed (design session 9, 2026-11-16); not implemented; H37–H39 open |
| Fine-tuning Qwen3.8 27B as a tch-fpc sample | 8 | gate passed (H40–H42 hold); designed in detail (design session 10, 2026-11-17: 8.9–8.19, section 10 here); not implemented; H43, H45–H50 open, H44 off the plan |
