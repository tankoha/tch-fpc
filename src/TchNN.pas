{ L3 nn: initialisation, var store, save/load, optimizers, modules and the
  milestone 2 layers (DESIGN.md 4.6). Ported from tch-rs 4227b89b
  src/nn/ (init, var_store, optimizer, linear, conv, batch_norm, layer_norm,
  sparse, sequential, func .rs); each section names its source file.

  Lifetimes: a TVarStore must outlive its TVarPath values, its layers' use of
  the store and every TOptimizer built on it. Free optimizers first, then the
  layers, then the store. Not thread-safe. }
unit TchNN;

{$mode objfpc}{$H+}
{$modeswitch advancedrecords}

interface

uses
  SysUtils, Math, ctypes, Generics.Collections, TchSys, Tch;

type
  { --- initialisation (nn/init.rs) --- }
  TFanMode = (fmFanIn, fmFanOut);
  TInitDist = (idUniform, idNormal);
  TNonLinearity = (nlReLU, nlLinear, nlSigmoid, nlTanh, nlSELU, nlExplicitGain);
  TInitKind = (ikConst, ikUniform, ikRandn, ikKaiming, ikOrthogonal);

  TInit = record
    Kind: TInitKind;
    { Const: A; Uniform: lo A, up B; Randn: mean A, std B; Orthogonal: gain A }
    A, B: Double;
    Dist: TInitDist;              // Kaiming only
    Fan: TFanMode;
    NonLin: TNonLinearity;
    Gain: Double;                 // for nlExplicitGain
    class function Constant(V: Double): TInit; static;
    class function Uniform(Lo, Up: Double): TInit; static;
    class function Randn(Mean, Std: Double): TInit; static;
    class function Kaiming(ADist: TInitDist; AFan: TFanMode;
      ANonLin: TNonLinearity; AGain: Double = 0): TInit; static;
    class function KaimingUniform: TInit; static;
    class function KaimingNormal: TInit; static;
    class function Orthogonal(AGain: Double = 1): TInit; static;
    { A new tensor of shape Dims (tch-rs f_init). No grad history: every
      tensor is created inside. }
    function Make(const Dims: array of Int64; const ADevice: TDevice;
      AKind: TKind): TTensor;
    { Re-initialises T in place (tch-rs Init::set), under its own
      NoGradGuard. }
    procedure Apply(const T: TTensor);
  end;

function FanForDims(Fan: TFanMode; const Dims: array of Int64): Int64;
function NonLinearityGain(NL: TNonLinearity; ExplicitGain: Double = 0): Double;

type
  { --- var store (nn/var_store.rs) --- }
  TVarStore = class;

  { A naming scope inside a store; a cheap value. Names are joined with '.';
    a name may not be empty or contain '.' or '|'. }
  TVarPath = record
  private
    FStore: TVarStore;            // not owned; must outlive the path
    FPath: String;                // 'encoder.layer1' ('' = root)
    FGroup: Integer;
  public
    function Sub(const Name: String): TVarPath;
    class operator /(const P: TVarPath; const Name: String): TVarPath;
    function SetGroup(G: Integer): TVarPath;
    function Store: TVarStore; inline;
    function Device: TDevice;
    function Kind: TKind;
    function Group: Integer; inline;
    function FullName(const Name: String): String;
    { Registers T under Name and returns it (the handle is shared). A name
      that already exists becomes '<name>__<Count>', as tch-rs. }
    function Add(const Name: String; const T: TTensor; Trainable: Boolean): TTensor;
    function Variable(const Name: String; const Dims: array of Int64;
      const Init: TInit): TTensor;
    function Zeros(const Name: String; const Dims: array of Int64): TTensor;
    function Ones(const Name: String; const Dims: array of Int64): TTensor;
    function ZerosNoTrain(const Name: String; const Dims: array of Int64): TTensor;
    function OnesNoTrain(const Name: String; const Dims: array of Int64): TTensor;
    function RandnStandard(const Name: String; const Dims: array of Int64): TTensor;
    function Randn(const Name: String; const Dims: array of Int64;
      Mean, Std: Double): TTensor;
    function Uniform(const Name: String; const Dims: array of Int64;
      Lo, Up: Double): TTensor;
    function KaimingUniform(const Name: String; const Dims: array of Int64): TTensor;
    function KaimingNormal(const Name: String; const Dims: array of Int64): TTensor;
    function Orthogonal(const Name: String; const Dims: array of Int64;
      Gain: Double): TTensor;
    { A trainable variable shaped like T, holding a copy of its data. }
    function VarCopy(const Name: String; const T: TTensor): TTensor;
    function TryGet(const Name: String; out T: TTensor): Boolean;
    { The existing variable whatever Dims and Init say, else a new one. }
    function GetOrAdd(const Name: String; const Dims: array of Int64;
      const Init: TInit): TTensor;
    { Converts the variables of this sub-tree (the path itself or below). }
    procedure SetKind(AKind: TKind);
    procedure SetFloatKind(AKind: TKind);
  end;

  TNamedTensor = record
    Name: String;
    Tensor: TTensor;
  end;
  TNamedTensorArray = array of TNamedTensor;

  TVarEntry = record
    Name: String;                 // full dotted name
    Tensor: TTensor;
    Trainable: Boolean;
    Group: Integer;
  end;
  TVarEntryArray = array of TVarEntry;
  TNameIndex = specialize TDictionary<String, Integer>;

  TVarStore = class
  private
    FVars: TVarEntryArray;        // insertion order, append-only
    FTrainable: array of Integer; // indices into FVars, insertion order
    FIndex: TNameIndex;           // full name -> index into FVars
    FDevice: TDevice;
    FKind: TKind;
    function AddEntry(const AFullName: String; const T: TTensor;
      Trainable: Boolean; AGroup: Integer): TTensor;
    procedure ConvertKind(const Prefix: String; AKind: TKind; FloatOnly: Boolean);
    function ReadFile(const FileName: String): TNamedTensorArray;
  public
    constructor Create(const ADevice: TDevice; AKind: TKind = kFloat);
    destructor Destroy; override;
    function Root: TVarPath;
    function Count: Integer;
    function TrainableVariables: TTensorArray;
    function Variables: TNamedTensorArray;     // insertion order
    function TryGet(const AFullName: String; out T: TTensor): Boolean;
    { '.ot' and any other extension: libtorch archive with '|' separators,
      the tch-rs format. Load also reads '.pt'/'.bin' (TorchScript pickle
      zip); '.safetensors' raises. }
    procedure Save(const FileName: String);
    { Every variable must be in the file; nothing changes otherwise.
      Variables take the file's kind. }
    procedure Load(const FileName: String);
    { Loads what is in the file; returns the missing names, insertion order. }
    function LoadPartial(const FileName: String): TStringArray;
    procedure Freeze;
    procedure Unfreeze;
    { Moves every variable in place (set_data): handles held by layers and
      optimizers follow. Optimizer state does not: create optimizers after. }
    procedure SetDevice(const ADevice: TDevice);
    procedure SetKind(AKind: TKind);
    procedure SetFloatKind(AKind: TKind);
    { Copies every variable of Self from the same name in Src. }
    procedure CopyFrom(Src: TVarStore);
    property Device: TDevice read FDevice;
    property Kind: TKind read FKind;
  end;

  { --- modules (nn/module.rs) --- }
  TModule = class abstract
  public
    { Overrides must repeat the default (H9). }
    function Forward(const X: TTensor; Train: Boolean = False): TTensor;
      virtual; abstract;
  end;

  { --- layers --- }
  TLinearConfig = record
    WsInit, BsInit: TInit;
    HasBsInit: Boolean;           // False: Uniform(-1/sqrt(In), 1/sqrt(In))
    Bias: Boolean;
  end;

  TPaddingMode = (pmZeros, pmReflect, pmReplicate, pmCircular);

  TConvConfig = record
    Stride, Padding, Dilation, Groups: Int64;
    Bias: Boolean;
    WsInit, BsInit: TInit;
    PaddingMode: TPaddingMode;
  end;

  TBatchNormConfig = record
    CudnnEnabled: Boolean;
    Eps, Momentum: Double;
    Affine: Boolean;
    WsInit, BsInit: TInit;
  end;

  TLayerNormConfig = record
    CudnnEnabled: Boolean;
    Eps: Double;
    ElementwiseAffine: Boolean;
    WsInit, BsInit: TInit;
  end;

  TEmbeddingConfig = record
    Sparse, ScaleGradByFreq: Boolean;
    WsInit: TInit;
    PaddingIdx: Int64;
  end;

function DefaultLinearConfig: TLinearConfig;
function DefaultConvConfig: TConvConfig;
function DefaultBatchNormConfig: TBatchNormConfig;
function DefaultLayerNormConfig: TLayerNormConfig;
function DefaultEmbeddingConfig: TEmbeddingConfig;

type
  TLinear = class(TModule)
  public
    Weight, Bias: TTensor;        // Bias.IsNil when the layer has none
    constructor Create(const P: TVarPath; InDim, OutDim: Int64); overload;
    constructor Create(const P: TVarPath; InDim, OutDim: Int64;
      const Cfg: TLinearConfig); overload;
    function Forward(const X: TTensor; Train: Boolean = False): TTensor; override;
  end;

  TConv2D = class(TModule)
  private
    FCfg: TConvConfig;
    FPadTwice: TInt64Array;       // reversed padding repeated twice
  public
    Weight, Bias: TTensor;
    constructor Create(const P: TVarPath; InDim, OutDim, KSize: Int64); overload;
    constructor Create(const P: TVarPath; InDim, OutDim, KSize: Int64;
      const Cfg: TConvConfig); overload;
    function Forward(const X: TTensor; Train: Boolean = False): TTensor; override;
  end;

  { Nd = 1, 2, 3 for batch_norm1d/2d/3d. Uses Train. }
  TBatchNorm = class(TModule)
  private
    FNd: Integer;
    FCfg: TBatchNormConfig;
  public
    Weight, Bias, RunningMean, RunningVar: TTensor;
    constructor Create(const P: TVarPath; Nd: Integer; OutDim: Int64); overload;
    constructor Create(const P: TVarPath; Nd: Integer; OutDim: Int64;
      const Cfg: TBatchNormConfig); overload;
    function Forward(const X: TTensor; Train: Boolean = False): TTensor; override;
  end;

  TLayerNorm = class(TModule)
  private
    FCfg: TLayerNormConfig;
    FShape: TInt64Array;
  public
    Weight, Bias: TTensor;
    constructor Create(const P: TVarPath;
      const NormalizedShape: array of Int64); overload;
    constructor Create(const P: TVarPath; const NormalizedShape: array of Int64;
      const Cfg: TLayerNormConfig); overload;
    function Forward(const X: TTensor; Train: Boolean = False): TTensor; override;
  end;

  TEmbedding = class(TModule)
  private
    FCfg: TEmbeddingConfig;
  public
    Weight: TTensor;
    constructor Create(const P: TVarPath; NumEmbeddings, EmbeddingDim: Int64); overload;
    constructor Create(const P: TVarPath; NumEmbeddings, EmbeddingDim: Int64;
      const Cfg: TEmbeddingConfig); overload;
    function Forward(const X: TTensor; Train: Boolean = False): TTensor; override;
  end;

  { --- function modules and sequential (nn/func.rs, nn/sequential.rs) --- }
  TTensorFunc    = function(const X: TTensor): TTensor;
  TTensorFuncT   = function(const X: TTensor; Train: Boolean): TTensor;
  TTensorMethod  = function(const X: TTensor): TTensor of object;
  TTensorMethodT = function(const X: TTensor; Train: Boolean): TTensor of object;

  { Wraps one function or method. A method's object must outlive the module. }
  TFuncModule = class(TModule)
  private
    FFunc: TTensorFunc;
    FFuncT: TTensorFuncT;
    FMethod: TTensorMethod;
    FMethodT: TTensorMethodT;
  public
    constructor Create(F: TTensorFunc); overload;
    constructor Create(F: TTensorFuncT); overload;
    constructor Create(F: TTensorMethod); overload;
    constructor Create(F: TTensorMethodT); overload;
    function Forward(const X: TTensor; Train: Boolean = False): TTensor; override;
  end;

  TModuleList = specialize TObjectList<TModule>;

  { Owns its children; Train goes to every child. }
  TSequential = class(TModule)
  private
    FLayers: TModuleList;
    function GetItem(I: Integer): TModule;
  public
    constructor Create;
    destructor Destroy; override;
    function Add(M: TModule): TSequential;
    function AddFn(F: TTensorFunc): TSequential; overload;
    function AddFn(F: TTensorFuncT): TSequential; overload;
    function AddFn(F: TTensorMethod): TSequential; overload;
    function AddFn(F: TTensorMethodT): TSequential; overload;
    function Forward(const X: TTensor; Train: Boolean = False): TTensor; override;
    { Outputs of the first N layers (N = -1: all); [X] when empty. }
    function ForwardAll(const X: TTensor; Train: Boolean = False;
      N: Integer = -1): TTensorArray;
    function Count: Integer;
    property Items[I: Integer]: TModule read GetItem; default;
  end;

  { --- optimizers (nn/optimizer.rs, wrappers/optimizer.rs) --- }

  { Holds the store's trainable variables; variables added to the store
    later join at the next ZeroGrad/Step/BackwardStep*. Free it before the
    store. }
  TOptimizer = class
  private
    FOpt: PTorchOptimizer;
    FStore: TVarStore;            // not owned
    FVarsInOptimizer: Integer;
    constructor CreateRaw(AOpt: PTorchOptimizer; VS: TVarStore);
    procedure AddMissingVariables;
  public
    class function Sgd(VS: TVarStore; LR: Double; Momentum: Double = 0;
      Dampening: Double = 0; WeightDecay: Double = 0;
      Nesterov: Boolean = False): TOptimizer; static;
    class function Adam(VS: TVarStore; LR: Double; Beta1: Double = 0.9;
      Beta2: Double = 0.999; WeightDecay: Double = 0; Eps: Double = 1e-8;
      AmsGrad: Boolean = False): TOptimizer; static;
    class function AdamW(VS: TVarStore; LR: Double; Beta1: Double = 0.9;
      Beta2: Double = 0.999; WeightDecay: Double = 0.01; Eps: Double = 1e-8;
      AmsGrad: Boolean = False): TOptimizer; static;
    class function RmsProp(VS: TVarStore; LR: Double; Alpha: Double = 0.99;
      Eps: Double = 1e-8; WeightDecay: Double = 0; Momentum: Double = 0;
      Centered: Boolean = False): TOptimizer; static;
    destructor Destroy; override;
    procedure ZeroGrad;
    procedure Step;
    procedure BackwardStep(const Loss: TTensor);
    procedure BackwardStepClip(const Loss: TTensor; MaxValue: Double);
    procedure BackwardStepClipNorm(const Loss: TTensor; MaxNorm: Double);
    procedure ClipGradValue(MaxValue: Double);
    procedure ClipGradNorm(MaxNorm: Double);
    procedure SetLearningRate(LR: Double);
    procedure SetLearningRateGroup(AGroup: Integer; LR: Double);
    { Beta1 for Adam/AdamW. }
    procedure SetMomentum(M: Double);
    procedure SetMomentumGroup(AGroup: Integer; M: Double);
    procedure SetWeightDecay(WD: Double);
    procedure SetWeightDecayGroup(AGroup: Integer; WD: Double);
    function TrainableVariables: TTensorArray;
  end;

implementation

{ ---------------------------------------------------------------- init }

const
  F64Epsilon = 2.220446049250313e-16;   // Rust f64::EPSILON

function FanForDims(Fan: TFanMode; const Dims: array of Int64): Int64;
var
  Receptive: Int64;
  I: Integer;
begin
  Receptive := 1;
  for I := 2 to High(Dims) do
    Receptive := Receptive * Dims[I];
  case Fan of
    fmFanIn:
      if Length(Dims) < 2 then
        Result := 1
      else
        Result := Dims[1] * Receptive;
  else
    if Length(Dims) = 0 then
      Result := 1
    else
      Result := Dims[0] * Receptive;
  end;
end;

function NonLinearityGain(NL: TNonLinearity; ExplicitGain: Double): Double;
begin
  case NL of
    nlReLU: Result := Sqrt(2.0);
    nlTanh: Result := 5.0 / 3.0;
    nlSELU: Result := 0.75;
    nlExplicitGain: Result := ExplicitGain;
  else
    Result := 1.0;                // nlLinear, nlSigmoid
  end;
end;

class function TInit.Constant(V: Double): TInit;
begin
  Result := Default(TInit);
  Result.Kind := ikConst;
  Result.A := V;
end;

class function TInit.Uniform(Lo, Up: Double): TInit;
begin
  Result := Default(TInit);
  Result.Kind := ikUniform;
  Result.A := Lo;
  Result.B := Up;
end;

class function TInit.Randn(Mean, Std: Double): TInit;
begin
  Result := Default(TInit);
  Result.Kind := ikRandn;
  Result.A := Mean;
  Result.B := Std;
end;

class function TInit.Kaiming(ADist: TInitDist; AFan: TFanMode;
  ANonLin: TNonLinearity; AGain: Double): TInit;
begin
  Result := Default(TInit);
  Result.Kind := ikKaiming;
  Result.Dist := ADist;
  Result.Fan := AFan;
  Result.NonLin := ANonLin;
  Result.Gain := AGain;
end;

class function TInit.KaimingUniform: TInit;
begin
  Result := Kaiming(idUniform, fmFanIn, nlReLU);
end;

class function TInit.KaimingNormal: TInit;
begin
  Result := Kaiming(idNormal, fmFanIn, nlReLU);
end;

class function TInit.Orthogonal(AGain: Double): TInit;
begin
  Result := Default(TInit);
  Result.Kind := ikOrthogonal;
  Result.A := AGain;
end;

function KaimingStd(const Init: TInit; const Dims: array of Int64): Double;
begin
  Result := NonLinearityGain(Init.NonLin, Init.Gain) /
    Sqrt(FanForDims(Init.Fan, Dims));
end;

function TInit.Make(const Dims: array of Int64; const ADevice: TDevice;
  AKind: TKind): TTensor;
var
  Std: Double;
  Rows, Cols: Int64;
  I: Integer;
  Flat, Q, R: TTensor;
begin
  case Kind of
    ikConst:
      if A = 0 then
        Result := TTensor.zeros(Dims, AKind, ADevice)
      else if Abs(A - 1) <= F64Epsilon then
        Result := TTensor.ones(Dims, AKind, ADevice)
      else
        Result := TTensor.ones(Dims, AKind, ADevice).mul_scalar(A);
    ikUniform:
      Result := TTensor.zeros(Dims, AKind, ADevice).uniform_(A, B);
    ikRandn:
      if (A = 0) and (Abs(B - 1) <= F64Epsilon) then
        Result := TTensor.randn(Dims, AKind, ADevice)
      else
        Result := TTensor.randn(Dims, AKind, ADevice).mul_scalar(B).add_scalar(A);
    ikKaiming:
      begin
        Std := KaimingStd(Self, Dims);
        if Dist = idUniform then
          Result := TTensor.zeros(Dims, AKind, ADevice).uniform_(
            -Sqrt(3.0) * Std, Sqrt(3.0) * Std)
        else
          Result := TTensor.randn(Dims, AKind, ADevice).mul_scalar(Std);
      end;
    ikOrthogonal:
      begin
        if Length(Dims) < 2 then
          raise ETorchError.Create(
            'Only tensors with 2 or more dimensions are supported');
        Rows := Dims[0];
        Cols := 1;
        for I := 1 to High(Dims) do
          Cols := Cols * Dims[I];
        Flat := TTensor.empty([Rows, Cols], AKind, ADevice).normal_(0, 1);
        if Rows < Cols then
          Flat := Flat.t_;
        TTensor.linalg_qr(Flat, 'reduced', Q, R);
        Q.mul_(R.diag(0).sign);
        if Rows < Cols then
          Q := Q.t_;
        Q.mul_scalar_(A);
        { tch-rs returns [Rows, Cols]; PyTorch keeps the requested shape. }
        Result := Q.contiguous.view(Dims);
      end;
  end;
end;

procedure TInit.Apply(const T: TTensor);
var
  G: IInterface;
  Std, Bound: Double;
begin
  G := NoGradGuard;
  case Kind of
    ikConst: T.fill_(A);
    ikUniform: T.uniform_(A, B);
    ikRandn: T.CopyFrom(T.randn_like.mul_scalar(B).add_scalar(A));
    ikKaiming:
      begin
        Std := KaimingStd(Self, T.Shape);
        if Dist = idUniform then
        begin
          Bound := Sqrt(3.0) * Std;
          T.uniform_(-Bound, Bound);
        end
        else
          T.CopyFrom(T.randn_like.mul_scalar(Std));
      end;
    ikOrthogonal:
      T.CopyFrom(Make(T.Shape, T.Device, T.Kind));
  end;
end;

{ ---------------------------------------------------------------- var path }

{ Stricter than tch-rs (which rejects only '.'): '|' would come back as '.'
  after a save/load round trip, and an empty name makes 'a..b'. }
procedure CheckName(const Name, What: String);
begin
  if (Name = '') or (Pos('.', Name) > 0) or (Pos('|', Name) > 0) then
    raise ETorchError.CreateFmt(
      '%s name cannot be empty or contain "." or "|": "%s"', [What, Name]);
end;

function TVarPath.Sub(const Name: String): TVarPath;
begin
  CheckName(Name, 'sub');
  Result := Self;
  if FPath = '' then
    Result.FPath := Name
  else
    Result.FPath := FPath + '.' + Name;
end;

class operator TVarPath./(const P: TVarPath; const Name: String): TVarPath;
begin
  Result := P.Sub(Name);
end;

function TVarPath.SetGroup(G: Integer): TVarPath;
begin
  Result := Self;
  Result.FGroup := G;
end;

function TVarPath.Store: TVarStore;
begin
  Result := FStore;
end;

function TVarPath.Device: TDevice;
begin
  Result := FStore.Device;
end;

function TVarPath.Kind: TKind;
begin
  Result := FStore.Kind;
end;

function TVarPath.Group: Integer;
begin
  Result := FGroup;
end;

function TVarPath.FullName(const Name: String): String;
begin
  CheckName(Name, 'variable');
  if FPath = '' then
    Result := Name
  else
    Result := FPath + '.' + Name;
end;

function TVarPath.Add(const Name: String; const T: TTensor;
  Trainable: Boolean): TTensor;
begin
  Result := FStore.AddEntry(FullName(Name), T, Trainable, FGroup);
end;

function TVarPath.Variable(const Name: String; const Dims: array of Int64;
  const Init: TInit): TTensor;
begin
  CheckName(Name, 'variable');
  Result := Add(Name, Init.Make(Dims, Device, Kind), True);
end;

function TVarPath.Zeros(const Name: String; const Dims: array of Int64): TTensor;
begin
  Result := Variable(Name, Dims, TInit.Constant(0));
end;

function TVarPath.Ones(const Name: String; const Dims: array of Int64): TTensor;
begin
  Result := Variable(Name, Dims, TInit.Constant(1));
end;

function TVarPath.ZerosNoTrain(const Name: String;
  const Dims: array of Int64): TTensor;
begin
  CheckName(Name, 'variable');
  Result := Add(Name, TInit.Constant(0).Make(Dims, Device, Kind), False);
end;

function TVarPath.OnesNoTrain(const Name: String;
  const Dims: array of Int64): TTensor;
begin
  CheckName(Name, 'variable');
  Result := Add(Name, TInit.Constant(1).Make(Dims, Device, Kind), False);
end;

function TVarPath.RandnStandard(const Name: String;
  const Dims: array of Int64): TTensor;
begin
  Result := Variable(Name, Dims, TInit.Randn(0, 1));
end;

function TVarPath.Randn(const Name: String; const Dims: array of Int64;
  Mean, Std: Double): TTensor;
begin
  Result := Variable(Name, Dims, TInit.Randn(Mean, Std));
end;

function TVarPath.Uniform(const Name: String; const Dims: array of Int64;
  Lo, Up: Double): TTensor;
begin
  Result := Variable(Name, Dims, TInit.Uniform(Lo, Up));
end;

function TVarPath.KaimingUniform(const Name: String;
  const Dims: array of Int64): TTensor;
begin
  Result := Variable(Name, Dims, TInit.KaimingUniform);
end;

function TVarPath.KaimingNormal(const Name: String;
  const Dims: array of Int64): TTensor;
begin
  Result := Variable(Name, Dims, TInit.KaimingNormal);
end;

function TVarPath.Orthogonal(const Name: String; const Dims: array of Int64;
  Gain: Double): TTensor;
begin
  Result := Variable(Name, Dims, TInit.Orthogonal(Gain));
end;

function TVarPath.VarCopy(const Name: String; const T: TTensor): TTensor;
var
  G: IInterface;
begin
  Result := Zeros(Name, T.Shape);
  G := NoGradGuard;
  Result.CopyFrom(T);
end;

function TVarPath.TryGet(const Name: String; out T: TTensor): Boolean;
begin
  Result := FStore.TryGet(FullName(Name), T);
end;

function TVarPath.GetOrAdd(const Name: String; const Dims: array of Int64;
  const Init: TInit): TTensor;
begin
  if not TryGet(Name, Result) then
    Result := Variable(Name, Dims, Init);
end;

procedure TVarPath.SetKind(AKind: TKind);
begin
  FStore.ConvertKind(FPath, AKind, False);
end;

procedure TVarPath.SetFloatKind(AKind: TKind);
begin
  FStore.ConvertKind(FPath, AKind, True);
end;

{ ---------------------------------------------------------------- var store }

constructor TVarStore.Create(const ADevice: TDevice; AKind: TKind);
begin
  inherited Create;
  FDevice := ADevice;
  FKind := AKind;
  FIndex := TNameIndex.Create;
end;

destructor TVarStore.Destroy;
begin
  FIndex.Free;
  FVars := nil;
  FTrainable := nil;
  inherited Destroy;
end;

function TVarStore.AddEntry(const AFullName: String; const T: TTensor;
  Trainable: Boolean; AGroup: Integer): TTensor;
var
  N: String;
  I: Integer;
begin
  TchCheckArg(T, 'tensor', 'TVarPath.Add');
  N := AFullName;
  if FIndex.ContainsKey(N) then
    N := N + '__' + IntToStr(Length(FVars));
  if Trainable then
    T.set_requires_grad(True);
  I := Length(FVars);
  SetLength(FVars, I + 1);
  FVars[I].Name := N;
  FVars[I].Tensor := T;
  FVars[I].Trainable := Trainable;
  FVars[I].Group := AGroup;
  FIndex.Add(N, I);
  if Trainable then
  begin
    SetLength(FTrainable, Length(FTrainable) + 1);
    FTrainable[High(FTrainable)] := I;
  end;
  Result := T;
end;

function TVarStore.Root: TVarPath;
begin
  Result.FStore := Self;
  Result.FPath := '';
  Result.FGroup := 0;
end;

function TVarStore.Count: Integer;
begin
  Result := Length(FVars);
end;

function TVarStore.TrainableVariables: TTensorArray;
var
  I: Integer;
begin
  Result := nil;
  SetLength(Result, Length(FTrainable));
  for I := 0 to High(FTrainable) do
    Result[I] := FVars[FTrainable[I]].Tensor;
end;

function TVarStore.Variables: TNamedTensorArray;
var
  I: Integer;
begin
  Result := nil;
  SetLength(Result, Length(FVars));
  for I := 0 to High(FVars) do
  begin
    Result[I].Name := FVars[I].Name;
    Result[I].Tensor := FVars[I].Tensor;
  end;
end;

function TVarStore.TryGet(const AFullName: String; out T: TTensor): Boolean;
var
  I: Integer;
begin
  T := Default(TTensor);
  Result := FIndex.TryGetValue(AFullName, I);
  if Result then
    T := FVars[I].Tensor;
end;

procedure TVarStore.Freeze;
var
  I: Integer;
begin
  for I in FTrainable do
    FVars[I].Tensor.set_requires_grad(False);
end;

procedure TVarStore.Unfreeze;
var
  I: Integer;
begin
  for I in FTrainable do
    FVars[I].Tensor.set_requires_grad(True);
end;

procedure TVarStore.SetDevice(const ADevice: TDevice);
var
  I: Integer;
begin
  for I := 0 to High(FVars) do
    FVars[I].Tensor.set_data(FVars[I].Tensor.&to(ADevice));
  FDevice := ADevice;
end;

{ Prefix '' = every variable; else the variable named Prefix and those below
  it (tch-rs's plain starts_with also matched 'ab.x' for 'a'). }
procedure TVarStore.ConvertKind(const Prefix: String; AKind: TKind;
  FloatOnly: Boolean);
var
  I: Integer;
  T: TTensor;
begin
  for I := 0 to High(FVars) do
    if (Prefix = '') or (FVars[I].Name = Prefix)
      or (Copy(FVars[I].Name, 1, Length(Prefix) + 1) = Prefix + '.') then
    begin
      T := FVars[I].Tensor;
      if (not FloatOnly) or T.is_floating_point then
        T.set_data(T.totype(AKind));
    end;
end;

procedure TVarStore.SetKind(AKind: TKind);
begin
  ConvertKind('', AKind, False);
  FKind := AKind;
end;

procedure TVarStore.SetFloatKind(AKind: TKind);
begin
  ConvertKind('', AKind, True);
  FKind := AKind;
end;

procedure TVarStore.CopyFrom(Src: TVarStore);
var
  I, J: Integer;
  G: IInterface;
begin
  for I := 0 to High(FVars) do
    if not Src.FIndex.ContainsKey(FVars[I].Name) then
      raise ETorchError.CreateFmt('variable %s not found in source var-store',
        [FVars[I].Name]);
  G := NoGradGuard;
  for I := 0 to High(FVars) do
  begin
    J := Src.FIndex[FVars[I].Name];
    FVars[I].Tensor.CopyFrom(Src.FVars[J].Tensor.&to(FDevice));
  end;
end;

{ --- save / load (var_store.rs, wrappers/tensor.rs) --- }

procedure TVarStore.Save(const FileName: String);
var
  Names: array of AnsiString;
  PNames: array of PChar;
  Ptrs: array of PTorchTensor;
  I: Integer;
begin
  if ExtractFileExt(FileName) = '.safetensors' then
    raise ETorchError.Create('safetensors files are not supported');
  Names := nil;
  PNames := nil;
  Ptrs := nil;
  SetLength(Names, Length(FVars));
  SetLength(PNames, Length(FVars));
  SetLength(Ptrs, Length(FVars));
  for I := 0 to High(FVars) do
  begin
    Names[I] := StringReplace(FVars[I].Name, '.', '|', [rfReplaceAll]);
    PNames[I] := PChar(Names[I]);
    Ptrs[I] := FVars[I].Tensor.Raw;
  end;
  at_save_multi(PPTorchTensor(Ptrs), PPChar(PNames), Length(FVars),
    PChar(FileName));
  TchCheck;
end;

type
  TLoadContext = class
    Items: TNamedTensorArray;
    Error: Exception;             // acquired exception object, nil = none
    destructor Destroy; override;
  end;

destructor TLoadContext.Destroy;
begin
  Error.Free;
  inherited Destroy;
end;

{ Called from C++ frames: must not let an exception escape. }
procedure LoadCallback(data: Pointer; name: PChar; t: PTorchTensor); cdecl;
var
  Ctx: TLoadContext;
  N: String;
  T1: TTensor;
begin
  Ctx := TLoadContext(data);
  try
    T1 := TTensor.FromRaw(t);     // own the handle on every path
    if Ctx.Error <> nil then
      Exit;                       // an earlier callback failed: keep skipping
    N := StringReplace(StrPas(name), '|', '.', [rfReplaceAll]);
    SetLength(Ctx.Items, Length(Ctx.Items) + 1);
    Ctx.Items[High(Ctx.Items)].Name := N;
    Ctx.Items[High(Ctx.Items)].Tensor := T1;
  except
    Ctx.Error := Exception(AcquireExceptionObject);
  end;
end;

function TVarStore.ReadFile(const FileName: String): TNamedTensorArray;
var
  Ctx: TLoadContext;
  Ext: String;
  P: PChar;
  E: Exception;
begin
  Ext := ExtractFileExt(FileName);
  if Ext = '.safetensors' then
    raise ETorchError.Create('safetensors files are not supported');
  Ctx := TLoadContext.Create;
  try
    if (Ext = '.bin') or (Ext = '.pt') then
      at_loadz_callback_with_device(PChar(FileName), Ctx, @LoadCallback,
        FDevice.ToC)
    else
      at_load_callback_with_device(PChar(FileName), Ctx, @LoadCallback,
        FDevice.ToC);
    P := get_and_reset_last_err;  // clear the slot on every path
    if Ctx.Error <> nil then
    begin
      if P <> nil then
        tchf_free(P);
      E := Ctx.Error;
      Ctx.Error := nil;
      raise E;
    end;
    if P <> nil then
      TchRaiseNative(P);
    Result := Ctx.Items;
  finally
    Ctx.Free;
  end;
end;

{ tch-rs copy_data_with_precision_update: the variable takes the file's
  kind. Caller holds a NoGradGuard. }
procedure CopyWithKind(const Dst, Src: TTensor);
begin
  Dst.set_data(Dst.totype(Src.Kind));
  Dst.CopyFrom(Src);
end;

function IndexByName(const Items: TNamedTensorArray): TNameIndex;
var
  I: Integer;
begin
  Result := TNameIndex.Create;
  for I := 0 to High(Items) do
    Result.AddOrSetValue(Items[I].Name, I);
end;

procedure TVarStore.Load(const FileName: String);
var
  Loaded: TNamedTensorArray;
  Idx: TNameIndex;
  I: Integer;
  G: IInterface;
begin
  Loaded := ReadFile(FileName);
  Idx := IndexByName(Loaded);
  try
    for I := 0 to High(FVars) do
      if not Idx.ContainsKey(FVars[I].Name) then
        raise ETorchError.CreateFmt('variable %s not found in %s',
          [FVars[I].Name, FileName]);
    G := NoGradGuard;
    for I := 0 to High(FVars) do
      CopyWithKind(FVars[I].Tensor, Loaded[Idx[FVars[I].Name]].Tensor);
  finally
    Idx.Free;
  end;
end;

function TVarStore.LoadPartial(const FileName: String): TStringArray;
var
  Loaded: TNamedTensorArray;
  Idx: TNameIndex;
  I, J: Integer;
  G: IInterface;
begin
  Result := nil;
  Loaded := ReadFile(FileName);
  Idx := IndexByName(Loaded);
  try
    G := NoGradGuard;
    for I := 0 to High(FVars) do
      if Idx.TryGetValue(FVars[I].Name, J) then
        CopyWithKind(FVars[I].Tensor, Loaded[J].Tensor)
      else
      begin
        SetLength(Result, Length(Result) + 1);
        Result[High(Result)] := FVars[I].Name;
      end;
  finally
    Idx.Free;
  end;
end;

{ ---------------------------------------------------------------- layers }

function DefaultLinearConfig: TLinearConfig;
begin
  Result.WsInit := TInit.KaimingUniform;
  Result.BsInit := TInit.Constant(0);
  Result.HasBsInit := False;
  Result.Bias := True;
end;

function DefaultConvConfig: TConvConfig;
begin
  Result.Stride := 1;
  Result.Padding := 0;
  Result.Dilation := 1;
  Result.Groups := 1;
  Result.Bias := True;
  Result.WsInit := TInit.KaimingUniform;
  Result.BsInit := TInit.Constant(0);
  Result.PaddingMode := pmZeros;
end;

function DefaultBatchNormConfig: TBatchNormConfig;
begin
  Result.CudnnEnabled := True;
  Result.Eps := 1e-5;
  Result.Momentum := 0.1;
  Result.Affine := True;
  Result.WsInit := TInit.Uniform(0, 1);
  Result.BsInit := TInit.Constant(0);
end;

function DefaultLayerNormConfig: TLayerNormConfig;
begin
  Result.CudnnEnabled := True;
  Result.Eps := 1e-5;
  Result.ElementwiseAffine := True;
  Result.WsInit := TInit.Constant(1);
  Result.BsInit := TInit.Constant(0);
end;

function DefaultEmbeddingConfig: TEmbeddingConfig;
begin
  Result.Sparse := False;
  Result.ScaleGradByFreq := False;
  Result.WsInit := TInit.Randn(0, 1);
  Result.PaddingIdx := -1;
end;

{ TLinear (linear.rs) }

constructor TLinear.Create(const P: TVarPath; InDim, OutDim: Int64);
begin
  Create(P, InDim, OutDim, DefaultLinearConfig);
end;

constructor TLinear.Create(const P: TVarPath; InDim, OutDim: Int64;
  const Cfg: TLinearConfig);
var
  BsInit: TInit;
begin
  inherited Create;
  Weight := P.Variable('weight', [OutDim, InDim], Cfg.WsInit);
  if Cfg.Bias then
  begin
    if Cfg.HasBsInit then
      BsInit := Cfg.BsInit
    else
      BsInit := TInit.Uniform(-1 / Sqrt(InDim), 1 / Sqrt(InDim));
    Bias := P.Variable('bias', [OutDim], BsInit);
  end;
end;

function TLinear.Forward(const X: TTensor; Train: Boolean): TTensor;
begin
  Result := X.linear(Weight, Bias);
end;

{ TConv2D (conv.rs, scalar ConvConfig) }

constructor TConv2D.Create(const P: TVarPath; InDim, OutDim, KSize: Int64);
begin
  Create(P, InDim, OutDim, KSize, DefaultConvConfig);
end;

constructor TConv2D.Create(const P: TVarPath; InDim, OutDim, KSize: Int64;
  const Cfg: TConvConfig);
var
  I: Integer;
begin
  inherited Create;
  FCfg := Cfg;
  SetLength(FPadTwice, 4);
  for I := 0 to 3 do
    FPadTwice[I] := Cfg.Padding;
  if Cfg.Bias then
    Bias := P.Variable('bias', [OutDim], Cfg.BsInit);
  Weight := P.Variable('weight', [OutDim, InDim div Cfg.Groups, KSize, KSize],
    Cfg.WsInit);
end;

function TConv2D.Forward(const X: TTensor; Train: Boolean): TTensor;
const
  ModeNames: array[TPaddingMode] of AnsiString =
    ('zeros', 'reflect', 'replicate', 'circular');
var
  S, D: Int64;
begin
  S := FCfg.Stride;
  D := FCfg.Dilation;
  if FCfg.PaddingMode = pmZeros then
    Result := X.conv2d(Weight, Bias, [S, S], [FCfg.Padding, FCfg.Padding],
      [D, D], FCfg.Groups)
  else
    Result := X.pad(FPadTwice, ModeNames[FCfg.PaddingMode], NoDouble).conv2d(
      Weight, Bias, [S, S], [0, 0], [D, D], FCfg.Groups);
end;

{ TBatchNorm (batch_norm.rs) }

constructor TBatchNorm.Create(const P: TVarPath; Nd: Integer; OutDim: Int64);
begin
  Create(P, Nd, OutDim, DefaultBatchNormConfig);
end;

constructor TBatchNorm.Create(const P: TVarPath; Nd: Integer; OutDim: Int64;
  const Cfg: TBatchNormConfig);
begin
  inherited Create;
  FNd := Nd;
  FCfg := Cfg;
  if Cfg.Affine then
  begin
    Weight := P.Variable('weight', [OutDim], Cfg.WsInit);
    Bias := P.Variable('bias', [OutDim], Cfg.BsInit);
  end;
  RunningMean := P.ZerosNoTrain('running_mean', [OutDim]);
  RunningVar := P.OnesNoTrain('running_var', [OutDim]);
end;

function ShapeText(const S: TInt64Array): String;
var
  I: Integer;
begin
  Result := '[';
  for I := 0 to High(S) do
  begin
    if I > 0 then
      Result := Result + ', ';
    Result := Result + IntToStr(S[I]);
  end;
  Result := Result + ']';
end;

function TBatchNorm.Forward(const X: TTensor; Train: Boolean): TTensor;
var
  D: Integer;
begin
  D := X.Dim;
  if (FNd = 1) and (D <> 2) and (D <> 3) then
    raise ETorchError.CreateFmt(
      'as nd=%d, expected an input tensor with 2 or 3 dims, got %d (%s)',
      [FNd, D, ShapeText(X.Shape)]);
  if (FNd > 1) and (D <> FNd + 2) then
    raise ETorchError.CreateFmt(
      'as nd=%d, expected an input tensor with %d dims, got %d (%s)',
      [FNd, FNd + 2, D, ShapeText(X.Shape)]);
  Result := X.batch_norm(Weight, Bias, RunningMean, RunningVar, Train,
    FCfg.Momentum, FCfg.Eps, FCfg.CudnnEnabled);
end;

{ TLayerNorm (layer_norm.rs) }

constructor TLayerNorm.Create(const P: TVarPath;
  const NormalizedShape: array of Int64);
begin
  Create(P, NormalizedShape, DefaultLayerNormConfig);
end;

constructor TLayerNorm.Create(const P: TVarPath;
  const NormalizedShape: array of Int64; const Cfg: TLayerNormConfig);
var
  I: Integer;
begin
  inherited Create;
  FCfg := Cfg;
  SetLength(FShape, Length(NormalizedShape));
  for I := 0 to High(NormalizedShape) do
    FShape[I] := NormalizedShape[I];
  if Cfg.ElementwiseAffine then
  begin
    Weight := P.Variable('weight', FShape, Cfg.WsInit);
    Bias := P.Variable('bias', FShape, Cfg.BsInit);
  end;
end;

function TLayerNorm.Forward(const X: TTensor; Train: Boolean): TTensor;
begin
  Result := X.layer_norm(FShape, Weight, Bias, FCfg.Eps, FCfg.CudnnEnabled);
end;

{ TEmbedding (sparse.rs) }

constructor TEmbedding.Create(const P: TVarPath;
  NumEmbeddings, EmbeddingDim: Int64);
begin
  Create(P, NumEmbeddings, EmbeddingDim, DefaultEmbeddingConfig);
end;

constructor TEmbedding.Create(const P: TVarPath;
  NumEmbeddings, EmbeddingDim: Int64; const Cfg: TEmbeddingConfig);
begin
  inherited Create;
  FCfg := Cfg;
  Weight := P.Variable('weight', [NumEmbeddings, EmbeddingDim], Cfg.WsInit);
end;

function TEmbedding.Forward(const X: TTensor; Train: Boolean): TTensor;
begin
  Result := TTensor.embedding(Weight, X, FCfg.PaddingIdx,
    FCfg.ScaleGradByFreq, FCfg.Sparse);
end;

{ ---------------------------------------------------------------- func / sequential }

constructor TFuncModule.Create(F: TTensorFunc);
begin
  inherited Create;
  FFunc := F;
end;

constructor TFuncModule.Create(F: TTensorFuncT);
begin
  inherited Create;
  FFuncT := F;
end;

constructor TFuncModule.Create(F: TTensorMethod);
begin
  inherited Create;
  FMethod := F;
end;

constructor TFuncModule.Create(F: TTensorMethodT);
begin
  inherited Create;
  FMethodT := F;
end;

function TFuncModule.Forward(const X: TTensor; Train: Boolean): TTensor;
begin
  if Assigned(FFunc) then
    Result := FFunc(X)
  else if Assigned(FFuncT) then
    Result := FFuncT(X, Train)
  else if Assigned(FMethod) then
    Result := FMethod(X)
  else
    Result := FMethodT(X, Train);
end;

constructor TSequential.Create;
begin
  inherited Create;
  FLayers := TModuleList.Create(True);
end;

destructor TSequential.Destroy;
begin
  FLayers.Free;
  inherited Destroy;
end;

function TSequential.GetItem(I: Integer): TModule;
begin
  Result := FLayers[I];
end;

function TSequential.Add(M: TModule): TSequential;
begin
  FLayers.Add(M);
  Result := Self;
end;

function TSequential.AddFn(F: TTensorFunc): TSequential;
begin
  Result := Add(TFuncModule.Create(F));
end;

function TSequential.AddFn(F: TTensorFuncT): TSequential;
begin
  Result := Add(TFuncModule.Create(F));
end;

function TSequential.AddFn(F: TTensorMethod): TSequential;
begin
  Result := Add(TFuncModule.Create(F));
end;

function TSequential.AddFn(F: TTensorMethodT): TSequential;
begin
  Result := Add(TFuncModule.Create(F));
end;

function TSequential.Forward(const X: TTensor; Train: Boolean): TTensor;
var
  M: TModule;
begin
  Result := X;
  for M in FLayers do
    Result := M.Forward(Result, Train);
end;

function TSequential.ForwardAll(const X: TTensor; Train: Boolean;
  N: Integer): TTensorArray;
var
  I: Integer;
  Cur: TTensor;
begin
  Result := nil;
  if FLayers.Count = 0 then
  begin
    SetLength(Result, 1);
    Result[0] := X;
    Exit;
  end;
  if (N < 0) or (N > FLayers.Count) then
    N := FLayers.Count;
  Cur := X;
  SetLength(Result, Max(N, 1));
  Result[0] := FLayers[0].Forward(Cur, Train);
  for I := 1 to N - 1 do
    Result[I] := FLayers[I].Forward(Result[I - 1], Train);
end;

function TSequential.Count: Integer;
begin
  Result := FLayers.Count;
end;

{ ---------------------------------------------------------------- optimizers }

constructor TOptimizer.CreateRaw(AOpt: PTorchOptimizer; VS: TVarStore);
begin
  inherited Create;
  FOpt := AOpt;
  FStore := VS;
  FVarsInOptimizer := 0;
  AddMissingVariables;
end;

destructor TOptimizer.Destroy;
begin
  if FOpt <> nil then
    ato_free(FOpt);
  inherited Destroy;
end;

procedure TOptimizer.AddMissingVariables;
var
  I: Integer;
begin
  while FVarsInOptimizer < Length(FStore.FTrainable) do
  begin
    I := FStore.FTrainable[FVarsInOptimizer];
    ato_add_parameters(FOpt, FStore.FVars[I].Tensor.Raw, FStore.FVars[I].Group);
    TchCheck;
    Inc(FVarsInOptimizer);
  end;
end;

class function TOptimizer.Sgd(VS: TVarStore; LR: Double; Momentum: Double;
  Dampening: Double; WeightDecay: Double; Nesterov: Boolean): TOptimizer;
var
  O: PTorchOptimizer;
begin
  O := ato_sgd(LR, Momentum, Dampening, WeightDecay, Ord(Nesterov));
  TchCheck;
  Result := TOptimizer.CreateRaw(O, VS);
end;

class function TOptimizer.Adam(VS: TVarStore; LR: Double; Beta1: Double;
  Beta2: Double; WeightDecay: Double; Eps: Double; AmsGrad: Boolean): TOptimizer;
var
  O: PTorchOptimizer;
begin
  O := ato_adam(LR, Beta1, Beta2, WeightDecay, Eps, AmsGrad);
  TchCheck;
  Result := TOptimizer.CreateRaw(O, VS);
end;

class function TOptimizer.AdamW(VS: TVarStore; LR: Double; Beta1: Double;
  Beta2: Double; WeightDecay: Double; Eps: Double; AmsGrad: Boolean): TOptimizer;
var
  O: PTorchOptimizer;
begin
  O := ato_adamw(LR, Beta1, Beta2, WeightDecay, Eps, AmsGrad);
  TchCheck;
  Result := TOptimizer.CreateRaw(O, VS);
end;

class function TOptimizer.RmsProp(VS: TVarStore; LR: Double; Alpha: Double;
  Eps: Double; WeightDecay: Double; Momentum: Double;
  Centered: Boolean): TOptimizer;
var
  O: PTorchOptimizer;
begin
  O := ato_rms_prop(LR, Alpha, Eps, WeightDecay, Momentum, Ord(Centered));
  TchCheck;
  Result := TOptimizer.CreateRaw(O, VS);
end;

procedure TOptimizer.ZeroGrad;
begin
  AddMissingVariables;
  ato_zero_grad(FOpt);
  TchCheck;
end;

procedure TOptimizer.Step;
begin
  AddMissingVariables;
  ato_step(FOpt);
  TchCheck;
end;

procedure TOptimizer.BackwardStep(const Loss: TTensor);
begin
  ZeroGrad;
  Loss.Backward;
  ato_step(FOpt);
  TchCheck;
end;

procedure TOptimizer.BackwardStepClip(const Loss: TTensor; MaxValue: Double);
begin
  ZeroGrad;
  Loss.Backward;
  ClipGradValue(MaxValue);
  ato_step(FOpt);
  TchCheck;
end;

procedure TOptimizer.BackwardStepClipNorm(const Loss: TTensor; MaxNorm: Double);
begin
  ZeroGrad;
  Loss.Backward;
  ClipGradNorm(MaxNorm);
  ato_step(FOpt);
  TchCheck;
end;

procedure TOptimizer.ClipGradValue(MaxValue: Double);
var
  G: IInterface;
  I: Integer;
  Gr: TTensor;
begin
  G := NoGradGuard;
  for I in FStore.FTrainable do
  begin
    Gr := FStore.FVars[I].Tensor.grad;
    if Gr.Defined then
      Gr.clamp_(-MaxValue, MaxValue);
  end;
end;

procedure TOptimizer.ClipGradNorm(MaxNorm: Double);
var
  G: IInterface;
  I: Integer;
  Grads, Norms: TTensorArray;
  Gr: TTensor;
  Total, Coef: Double;
begin
  G := NoGradGuard;
  Grads := nil;
  Norms := nil;
  for I in FStore.FTrainable do
  begin
    Gr := FStore.FVars[I].Tensor.grad;
    if Gr.Defined then
    begin
      SetLength(Grads, Length(Grads) + 1);
      Grads[High(Grads)] := Gr;
      SetLength(Norms, Length(Norms) + 1);
      Norms[High(Norms)] := Gr.norm;
    end;
  end;
  if Length(Grads) = 0 then
    Exit;
  Total := TTensor.stack(Norms, 0).norm.DoubleValue([]);
  Coef := MaxNorm / (Total + 1e-6);
  if Coef < 1 then
    for Gr in Grads do
      Gr.mul_scalar_(Coef);
end;

procedure TOptimizer.SetLearningRate(LR: Double);
begin
  ato_set_learning_rate(FOpt, LR);
  TchCheck;
end;

procedure TOptimizer.SetLearningRateGroup(AGroup: Integer; LR: Double);
begin
  ato_set_learning_rate_group(FOpt, AGroup, LR);
  TchCheck;
end;

procedure TOptimizer.SetMomentum(M: Double);
begin
  ato_set_momentum(FOpt, M);
  TchCheck;
end;

procedure TOptimizer.SetMomentumGroup(AGroup: Integer; M: Double);
begin
  ato_set_momentum_group(FOpt, AGroup, M);
  TchCheck;
end;

procedure TOptimizer.SetWeightDecay(WD: Double);
begin
  ato_set_weight_decay(FOpt, WD);
  TchCheck;
end;

procedure TOptimizer.SetWeightDecayGroup(AGroup: Integer; WD: Double);
begin
  ato_set_weight_decay_group(FOpt, AGroup, WD);
  TchCheck;
end;

function TOptimizer.TrainableVariables: TTensorArray;
begin
  Result := FStore.TrainableVariables;
end;

end.
