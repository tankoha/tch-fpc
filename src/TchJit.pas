{ L3 JIT: TorchScript values, modules and trainable modules (DESIGN.md
  4.6.8). Ported from tch-rs 4227b89b src/wrappers/jit.rs (IValue, CModule)
  and src/nn/mod.rs (TrainableCModule).

  Deviations from tch-rs, stated in DESIGN 4.6.8:
  - CreateEmpty, RegisterParameter and Define use the fork's atm_create,
    atm_register_parameter and atm_define (TorchScript from Pascal).
  - TTrainableJitModule.Forward applies Train through atm_train/atm_eval.
  - CreateClass returns the object, not an IValue (DESIGN 4.6.9).

  Lifetimes: a TTrainableJitModule shares its parameters with the var
  store; free optimizers, then the module, then the store. A
  TorchScript object does not keep its module alive and does not need to. }
unit TchJit;

{$mode objfpc}{$H+}
{$modeswitch advancedrecords}

interface

uses
  SysUtils, Classes, ctypes, TchSys, Tch, TchNN;

type
  { Values equal ati_tag's; ivStringList and ivDevice are input only (a
    string list comes back as a GenericList of String, a device output
    raises in ati_tag). }
  TIValueTag = (ivNone = 0, ivTensor = 1, ivDouble = 2, ivInt = 3,
    ivBool = 4, ivTuple = 5, ivIntList = 6, ivDoubleList = 7,
    ivBoolList = 8, ivString = 9, ivTensorList = 10, ivGenericList = 12,
    ivGenericDict = 13, ivObject = 14, ivStringList = 20, ivDevice = 21);
  TBooleanArray = array of Boolean;

  { forward: TIValue holds one, its methods use TIValue }
  IJitObject = interface;

  { A TorchScript value. Copies share Items: treat a value as immutable.
    Default(TIValue) is None. }
  TIValue = record
  public type
    TIValueArray = array of TIValue;
  public
    Tag: TIValueTag;
    Tensor: TTensor;              // ivTensor
    I: Int64;                     // ivInt
    D: Double;                    // ivDouble
    B: Boolean;                   // ivBool
    S: String;                    // ivString
    Ints: TInt64Array;            // ivIntList
    Doubles: TDoubleArray;        // ivDoubleList
    Bools: TBooleanArray;         // ivBoolList
    Strings: TStringArray;        // ivStringList (input only)
    Tensors: TTensorArray;        // ivTensorList
    { ivTuple, ivGenericList; ivGenericDict: k0, v0, k1, v1, ... }
    Items: TIValueArray;
    Dev: TDevice;                 // ivDevice (input only)
    Obj: IJitObject;              // ivObject (nil otherwise)
    class function None: TIValue; static;
    class function FromTensor(const T: TTensor): TIValue; static;
    class function FromInt(V: Int64): TIValue; static;
    class function FromDouble(V: Double): TIValue; static;
    class function FromBool(V: Boolean): TIValue; static;
    class function FromString(const V: String): TIValue; static;
    class function FromDevice(const V: TDevice): TIValue; static;
    class function FromObject(const O: IJitObject): TIValue; static;
    class function Tuple(const A: array of TIValue): TIValue; static;
    class function IntList(const A: array of Int64): TIValue; static;
    class function DoubleList(const A: array of Double): TIValue; static;
    class function BoolList(const A: array of Boolean): TIValue; static;
    class function StringList(const A: array of String): TIValue; static;
    class function TensorList(const A: array of TTensor): TIValue; static;
    class function GenericList(const A: array of TIValue): TIValue; static;
    { Keys and values alternating; an odd length raises ETorchError. }
    class function GenericDict(const KeysAndValues: array of TIValue): TIValue; static;
    class operator :=(const V: TTensor): TIValue;
    class operator :=(V: Int64): TIValue;
    class operator :=(V: Double): TIValue;
    class operator :=(V: Boolean): TIValue;
    class operator :=(const V: String): TIValue;
    class operator :=(const O: IJitObject): TIValue;
    function IsNone: Boolean;
    { 'Tensor', 'Tuple', ... as tch-rs's type_str }
    function TagName: String;
    { The As* accessors raise ETorchError on a tag mismatch. }
    function AsTensor: TTensor;
    function AsInt: Int64;
    function AsDouble: Double;
    function AsBool: Boolean;
    function AsString: String;
    function AsIntList: TInt64Array;
    function AsDoubleList: TDoubleArray;
    function AsBoolList: TBooleanArray;
    function AsTensorList: TTensorArray;
    function AsObject: IJitObject;
    { tuple/list: element count; dict: pair count; typed lists: length }
    function Count: Integer;
    function Item(Index: Integer): TIValue;    // ivTuple or ivGenericList
    function Key(Index: Integer): TIValue;     // ivGenericDict
    function Value(Index: Integer): TIValue;   // ivGenericDict
    { A new C ivalue, owned by the caller (ati_free). }
    function ToC: PTorchIValue;
    { Consumes P: it is freed on every path. }
    class function FromC(P: PTorchIValue): TIValue; static;
  end;
  TIValueArray = TIValue.TIValueArray;

  { A TorchScript object (tch-rs Object). Copies of a TIValue share it;
    the C IValue is freed when the last reference goes. }
  IJitObject = interface
    ['{6C2B8E41-3F7A-4D95-B0E2-91A4C5D7F318}']
    { borrowed; always tag 14 }
    function Raw: PTorchIValue;
    function MethodIV(const Name: String; const Inputs: array of TIValue): TIValue;
    function GetAttr(const Name: String): TIValue;
  end;

  TTraceFunc = function(const Inputs: TTensorArray): TTensorArray;
  TTraceMethod = function(const Inputs: TTensorArray): TTensorArray of object;

  { A TorchScript module (tch-rs CModule). }
  TJitModule = class
  private
    FModule: PTorchModule;
    { takes ownership; M <> nil }
    {$push}{$warn 3018 off}
    constructor CreateRaw(M: PTorchModule);
    {$pop}
    class function Trace(const ModuleName, FnName: String;
      const Inputs: array of TTensor; Fn: TTraceFunc;
      Meth: TTraceMethod): TJitModule; static;
  public
    constructor Create(const FileName: String); overload;
    constructor Create(const FileName: String; const ADevice: TDevice); overload;
    { Reads from the current position to the end; an empty rest raises. }
    constructor CreateFromStream(S: TStream); overload;
    constructor CreateFromStream(S: TStream; const ADevice: TDevice); overload;
    { An empty scripted module with a "training" attribute (fork). }
    constructor CreateEmpty(const ModuleName: String);
    { Fn runs once on Inputs while libtorch records the graph; its result
      becomes the outputs of method FnName. }
    class function CreateByTracing(const ModuleName, FnName: String;
      const Inputs: array of TTensor; Fn: TTraceFunc): TJitModule; overload; static;
    class function CreateByTracing(const ModuleName, FnName: String;
      const Inputs: array of TTensor; Fn: TTraceMethod): TJitModule; overload; static;
    destructor Destroy; override;
    { The tensor API raises ETorchNativeError when the result is not a
      tensor; use ForwardIV/MethodIV for other results. }
    function Forward(const Inputs: array of TTensor): TTensor;
    function ForwardIV(const Inputs: array of TIValue): TIValue;
    function Method(const Name: String; const Inputs: array of TTensor): TTensor;
    function MethodIV(const Name: String; const Inputs: array of TIValue): TIValue;
    procedure SetEval;
    procedure SetTrain;
    procedure ToDevice(const ADevice: TDevice; AKind: TKind; NonBlocking: Boolean = False);
    procedure Save(const FileName: String);
    { Parameters only, not buffers; the handles share the module's tensors. }
    function NamedParameters: TNamedTensorArray;
    procedure RegisterParameter(const Name: String; const T: TTensor;
      IsBuffer: Boolean = False);
    { TorchScript source: def at column 0, lines joined with #10. }
    procedure Define(const Source: String);
    { QualifiedName is e.g. '__torch__.InputObject'; runs __init__. }
    function CreateClass(const QualifiedName: String;
      const Inputs: array of TIValue): IJitObject;
    property Raw: PTorchModule read FModule;
  end;

  TJitMode = (jmUnknown, jmTrain, jmEval);

  { A TorchScript module whose parameters live in a var store (tch-rs
    TrainableCModule): an optimizer on the store trains the module in
    place. Parameter 'a.b' is registered as 'a_b' under the path. }
  TTrainableJitModule = class(TModule)
  private
    FInner: TJitModule;           // owned
    FMode: TJitMode;
    procedure RegisterInto(const P: TVarPath);
  public
    { Loads on P's device. }
    constructor Create(const FileName: String; const P: TVarPath);
    constructor CreateFromStream(S: TStream; const P: TVarPath);
    destructor Destroy; override;
    { Switches the module to train or eval mode when Train differs from
      the current mode (tch-rs ignores Train). }
    function Forward(const X: TTensor; Train: Boolean = False): TTensor; override;
    function ForwardIV(const Inputs: array of TIValue): TIValue;
    function Method(const Name: String; const Inputs: array of TTensor): TTensor;
    function MethodIV(const Name: String; const Inputs: array of TIValue): TIValue;
    procedure Save(const FileName: String);
    procedure SetTrain;
    procedure SetEval;
    property Inner: TJitModule read FInner;
  end;

function JitGetProfilingMode: Boolean;
procedure JitSetProfilingMode(B: Boolean);
function JitGetTensorExprFuserEnabled: Boolean;
procedure JitSetTensorExprFuserEnabled(B: Boolean);
procedure JitSetGraphExecutorOptimize(B: Boolean);
{ the CUDA fuser flag (tch-rs parity). libtorch 2.13 removed nvfuser:
  SetJitFuserCudaEnabled(True) raises ETorchNativeError, False is a no-op. }
function JitFuserCudaEnabled: Boolean;
procedure SetJitFuserCudaEnabled(B: Boolean);
{ live TorchScript object handles, as TchLiveHandles }
function JitLiveObjects: Integer;

implementation

type
  TIValuePtrArray = array of PTorchIValue;

  TJitObjectHandle = class(TInterfacedObject, IJitObject)
  private
    FRaw: PTorchIValue;
  public
    { takes ownership of P }
    constructor Create(P: PTorchIValue);
    { ati_free; never raises }
    destructor Destroy; override;
    function Raw: PTorchIValue;
    function MethodIV(const Name: String; const Inputs: array of TIValue): TIValue;
    function GetAttr(const Name: String): TIValue;
  end;

var
  LiveObjects: Integer = 0;

{ An enum with explicit values cannot index an array. }
function TagNames(T: TIValueTag): String;
begin
  case T of
    ivNone: Result := 'None';
    ivTensor: Result := 'Tensor';
    ivDouble: Result := 'Double';
    ivInt: Result := 'Int';
    ivBool: Result := 'Bool';
    ivTuple: Result := 'Tuple';
    ivIntList: Result := 'IntList';
    ivDoubleList: Result := 'DoubleList';
    ivBoolList: Result := 'BoolList';
    ivString: Result := 'String';
    ivTensorList: Result := 'TensorList';
    ivGenericList: Result := 'GenericList';
    ivGenericDict: Result := 'GenericDict';
    ivObject: Result := 'Object';
    ivStringList: Result := 'StringList';
    ivDevice: Result := 'Device';
  else
    Result := 'tag ' + IntToStr(Ord(T));
  end;
end;

procedure FreeIValues(const A: TIValuePtrArray);
var
  K: Integer;
begin
  for K := 0 to High(A) do
    if A[K] <> nil then
      ati_free(A[K]);
end;

{ New C ivalues for A; on an error the ones already made are freed. }
function IValuesToC(const A: array of TIValue): TIValuePtrArray;
var
  K: Integer;
begin
  Result := nil;
  SetLength(Result, Length(A));   // nil-initialised
  try
    for K := 0 to High(A) do
      Result[K] := A[K].ToC;
  except
    FreeIValues(Result);
    raise;
  end;
end;

{ Converts the C ivalues in Cs, nil-ing each slot as it is consumed, so
  the caller's finally frees only the ones not yet converted. }
function IValuesFromC(var Cs: TIValuePtrArray): TIValueArray;
var
  K: Integer;
  C: PTorchIValue;
begin
  Result := nil;
  SetLength(Result, Length(Cs));
  for K := 0 to High(Cs) do
  begin
    C := Cs[K];
    Cs[K] := nil;
    Result[K] := TIValue.FromC(C);
  end;
end;

{ TIValue }

class function TIValue.None: TIValue;
begin
  Result := Default(TIValue);
end;

class function TIValue.FromTensor(const T: TTensor): TIValue;
begin
  Result := Default(TIValue);
  Result.Tag := ivTensor;
  Result.Tensor := T;
end;

class function TIValue.FromInt(V: Int64): TIValue;
begin
  Result := Default(TIValue);
  Result.Tag := ivInt;
  Result.I := V;
end;

class function TIValue.FromDouble(V: Double): TIValue;
begin
  Result := Default(TIValue);
  Result.Tag := ivDouble;
  Result.D := V;
end;

class function TIValue.FromBool(V: Boolean): TIValue;
begin
  Result := Default(TIValue);
  Result.Tag := ivBool;
  Result.B := V;
end;

class function TIValue.FromString(const V: String): TIValue;
begin
  Result := Default(TIValue);
  Result.Tag := ivString;
  Result.S := V;
end;

class function TIValue.FromDevice(const V: TDevice): TIValue;
begin
  Result := Default(TIValue);
  Result.Tag := ivDevice;
  Result.Dev := V;
end;

class function TIValue.FromObject(const O: IJitObject): TIValue;
begin
  Result := Default(TIValue);
  Result.Tag := ivObject;
  Result.Obj := O;
end;

function CopyIValues(const A: array of TIValue): TIValueArray;
var
  K: Integer;
begin
  Result := nil;
  SetLength(Result, Length(A));
  for K := 0 to High(A) do
    Result[K] := A[K];
end;

class function TIValue.Tuple(const A: array of TIValue): TIValue;
begin
  Result := Default(TIValue);
  Result.Tag := ivTuple;
  Result.Items := CopyIValues(A);
end;

class function TIValue.IntList(const A: array of Int64): TIValue;
var
  K: Integer;
begin
  Result := Default(TIValue);
  Result.Tag := ivIntList;
  SetLength(Result.Ints, Length(A));
  for K := 0 to High(A) do
    Result.Ints[K] := A[K];
end;

class function TIValue.DoubleList(const A: array of Double): TIValue;
var
  K: Integer;
begin
  Result := Default(TIValue);
  Result.Tag := ivDoubleList;
  SetLength(Result.Doubles, Length(A));
  for K := 0 to High(A) do
    Result.Doubles[K] := A[K];
end;

class function TIValue.BoolList(const A: array of Boolean): TIValue;
var
  K: Integer;
begin
  Result := Default(TIValue);
  Result.Tag := ivBoolList;
  SetLength(Result.Bools, Length(A));
  for K := 0 to High(A) do
    Result.Bools[K] := A[K];
end;

class function TIValue.StringList(const A: array of String): TIValue;
var
  K: Integer;
begin
  Result := Default(TIValue);
  Result.Tag := ivStringList;
  SetLength(Result.Strings, Length(A));
  for K := 0 to High(A) do
    Result.Strings[K] := A[K];
end;

class function TIValue.TensorList(const A: array of TTensor): TIValue;
var
  K: Integer;
begin
  Result := Default(TIValue);
  Result.Tag := ivTensorList;
  SetLength(Result.Tensors, Length(A));
  for K := 0 to High(A) do
    Result.Tensors[K] := A[K];
end;

class function TIValue.GenericList(const A: array of TIValue): TIValue;
begin
  Result := Default(TIValue);
  Result.Tag := ivGenericList;
  Result.Items := CopyIValues(A);
end;

class function TIValue.GenericDict(const KeysAndValues: array of TIValue): TIValue;
begin
  if Odd(Length(KeysAndValues)) then
    raise ETorchError.CreateFmt('ivalue: GenericDict needs keys and values ' +
      'in pairs, got %d items', [Length(KeysAndValues)]);
  Result := Default(TIValue);
  Result.Tag := ivGenericDict;
  Result.Items := CopyIValues(KeysAndValues);
end;

class operator TIValue.:=(const V: TTensor): TIValue;
begin
  Result := TIValue.FromTensor(V);
end;

class operator TIValue.:=(V: Int64): TIValue;
begin
  Result := TIValue.FromInt(V);
end;

class operator TIValue.:=(V: Double): TIValue;
begin
  Result := TIValue.FromDouble(V);
end;

class operator TIValue.:=(V: Boolean): TIValue;
begin
  Result := TIValue.FromBool(V);
end;

class operator TIValue.:=(const V: String): TIValue;
begin
  Result := TIValue.FromString(V);
end;

class operator TIValue.:=(const O: IJitObject): TIValue;
begin
  Result := TIValue.FromObject(O);
end;

function TIValue.IsNone: Boolean;
begin
  Result := Tag = ivNone;
end;

function TIValue.TagName: String;
begin
  Result := TagNames(Tag);
end;

procedure RaiseTagMismatch(Expected: TIValueTag; const V: TIValue);
begin
  raise ETorchError.CreateFmt('ivalue: expected %s, got %s',
    [TagNames(Expected), V.TagName]);
end;

function TIValue.AsTensor: TTensor;
begin
  if Tag <> ivTensor then
    RaiseTagMismatch(ivTensor, Self);
  Result := Tensor;
end;

function TIValue.AsInt: Int64;
begin
  if Tag <> ivInt then
    RaiseTagMismatch(ivInt, Self);
  Result := I;
end;

function TIValue.AsDouble: Double;
begin
  if Tag <> ivDouble then
    RaiseTagMismatch(ivDouble, Self);
  Result := D;
end;

function TIValue.AsBool: Boolean;
begin
  if Tag <> ivBool then
    RaiseTagMismatch(ivBool, Self);
  Result := B;
end;

function TIValue.AsString: String;
begin
  if Tag <> ivString then
    RaiseTagMismatch(ivString, Self);
  Result := S;
end;

function TIValue.AsIntList: TInt64Array;
begin
  if Tag <> ivIntList then
    RaiseTagMismatch(ivIntList, Self);
  Result := Ints;
end;

function TIValue.AsDoubleList: TDoubleArray;
begin
  if Tag <> ivDoubleList then
    RaiseTagMismatch(ivDoubleList, Self);
  Result := Doubles;
end;

function TIValue.AsBoolList: TBooleanArray;
begin
  if Tag <> ivBoolList then
    RaiseTagMismatch(ivBoolList, Self);
  Result := Bools;
end;

function TIValue.AsTensorList: TTensorArray;
begin
  if Tag <> ivTensorList then
    RaiseTagMismatch(ivTensorList, Self);
  Result := Tensors;
end;

function TIValue.AsObject: IJitObject;
begin
  if Tag <> ivObject then
    RaiseTagMismatch(ivObject, Self);
  Result := Obj;
end;

function TIValue.Count: Integer;
begin
  case Tag of
    ivTuple, ivGenericList: Result := Length(Items);
    ivGenericDict: Result := Length(Items) div 2;
    ivIntList: Result := Length(Ints);
    ivDoubleList: Result := Length(Doubles);
    ivBoolList: Result := Length(Bools);
    ivStringList: Result := Length(Strings);
    ivTensorList: Result := Length(Tensors);
  else
    raise ETorchError.CreateFmt('ivalue: Count of a %s', [TagName]);
  end;
end;

function TIValue.Item(Index: Integer): TIValue;
begin
  if not (Tag in [ivTuple, ivGenericList]) then
    raise ETorchError.CreateFmt('ivalue: expected Tuple or GenericList, got %s',
      [TagName]);
  if (Index < 0) or (Index > High(Items)) then
    raise ETorchError.CreateFmt('ivalue: index %d out of range 0..%d',
      [Index, High(Items)]);
  Result := Items[Index];
end;

function TIValue.Key(Index: Integer): TIValue;
begin
  if Tag <> ivGenericDict then
    RaiseTagMismatch(ivGenericDict, Self);
  if (Index < 0) or (Index >= Count) then
    raise ETorchError.CreateFmt('ivalue: pair %d out of range 0..%d',
      [Index, Count - 1]);
  Result := Items[2 * Index];
end;

function TIValue.Value(Index: Integer): TIValue;
begin
  if Tag <> ivGenericDict then
    RaiseTagMismatch(ivGenericDict, Self);
  if (Index < 0) or (Index >= Count) then
    raise ETorchError.CreateFmt('ivalue: pair %d out of range 0..%d',
      [Index, Count - 1]);
  Result := Items[2 * Index + 1];
end;

{ tch-rs to_c. Containers copy their children's IValues, so the children
  are freed here whatever happens. Empty arrays pass nil. }
function TIValue.ToC: PTorchIValue;
var
  Cs: TIValuePtrArray;
  Ptrs: TTensorPtrArray;
  PStrs: array of PChar;
  K: Integer;
begin
  case Tag of
    ivNone: Result := ati_none;
    ivTensor:
      begin
        TchCheckArg(Tensor, 'ivalue', 'TIValue.ToC');
        Result := ati_tensor(Tensor.Raw);
      end;
    ivDouble: Result := ati_double(D);
    ivInt: Result := ati_int(I);
    ivBool: Result := ati_bool(Ord(B));
    ivString: Result := ati_string(PChar(S));
    ivIntList: Result := ati_int_list(PInt64(Pointer(Ints)), Length(Ints));
    ivDoubleList:
      Result := ati_double_list(PDouble(Pointer(Doubles)), Length(Doubles));
    { SizeOf(Boolean) = 1 holding 0 or 1, the bytes the C side expects }
    ivBoolList: Result := ati_bool_list(PChar(Pointer(Bools)), Length(Bools));
    ivStringList:
      begin
        PStrs := nil;
        SetLength(PStrs, Length(Strings));
        for K := 0 to High(Strings) do
          PStrs[K] := PChar(Strings[K]);
        Result := ati_string_list(PPChar(Pointer(PStrs)), Length(PStrs));
      end;
    ivTensorList:
      begin
        Ptrs := TchTensorPtrs(Tensors, 'ivalue', 'TIValue.ToC');
        Result := ati_tensor_list(PPTorchTensor(Pointer(Ptrs)), Length(Ptrs));
      end;
    ivDevice: Result := ati_device(Dev.ToC);
    { a clone: the caller frees it, the handle keeps its own (tch-rs issue 475) }
    ivObject:
      begin
        if Obj = nil then
          raise ETorchError.Create('ivalue: nil object');
        Result := ati_clone(Obj.Raw);
      end;
    ivTuple, ivGenericList, ivGenericDict:
      begin
        if (Tag = ivGenericDict) and Odd(Length(Items)) then
          raise ETorchError.CreateFmt('ivalue: GenericDict needs keys and ' +
            'values in pairs, got %d items', [Length(Items)]);
        Cs := IValuesToC(Items);
        try
          case Tag of
            ivTuple: Result := ati_tuple(PPTorchIValue(Pointer(Cs)), Length(Cs));
            ivGenericList:
              Result := ati_generic_list(PPTorchIValue(Pointer(Cs)), Length(Cs));
          else
            Result := ati_generic_dict(PPTorchIValue(Pointer(Cs)), Length(Cs) div 2);
          end;
          TchCheck;
        finally
          FreeIValues(Cs);
        end;
        Exit;
      end;
  else
    raise ETorchError.CreateFmt('ivalue: invalid tag %d', [Ord(Tag)]);
  end;
  TchCheck;
end;

{ tch-rs from_c. Fixed-N outputs go into nil-initialised slots: the C side
  may fail after writing some of them. }
class function TIValue.FromC(P: PTorchIValue): TIValue;
var
  CTag, N, K: cint;
  PT: PTorchTensor;
  Str: PChar;
  Bytes: TBytes;
  Slots: TTensorPtrArray;
  Cs: TIValuePtrArray;
  Keep: Boolean;
begin
  if P = nil then
    raise ETorchError.Create('ivalue: nil pointer');
  Result := Default(TIValue);
  Cs := nil;
  Keep := False;
  try
    CTag := ati_tag(P);
    TchCheck;
    case CTag of
      0: ;
      1:
        begin
          PT := ati_to_tensor(P);
          TchCheck;
          Result.Tag := ivTensor;
          Result.Tensor := TTensor.FromRaw(PT);
        end;
      2:
        begin
          Result.Tag := ivDouble;
          Result.D := ati_to_double(P);
          TchCheck;
        end;
      3:
        begin
          Result.Tag := ivInt;
          Result.I := ati_to_int(P);
          TchCheck;
        end;
      4:
        begin
          Result.Tag := ivBool;
          Result.B := ati_to_bool(P) <> 0;
          TchCheck;
        end;
      5, 12, 13:
        begin
          if CTag = 5 then
            N := ati_tuple_length(P)
          else
            N := ati_length(P);
          TchCheck;
          if CTag = 13 then
            SetLength(Cs, 2 * N)
          else
            SetLength(Cs, N);
          case CTag of
            5: ati_to_tuple(P, PPTorchIValue(Pointer(Cs)), N);
            12: ati_to_generic_list(P, PPTorchIValue(Pointer(Cs)), N);
          else
            ati_to_generic_dict(P, PPTorchIValue(Pointer(Cs)), N);
          end;
          TchCheck;
          Result.Tag := TIValueTag(CTag);
          Result.Items := IValuesFromC(Cs);
        end;
      6:
        begin
          N := ati_length(P);
          TchCheck;
          Result.Tag := ivIntList;
          SetLength(Result.Ints, N);
          ati_to_int_list(P, PInt64(Pointer(Result.Ints)), N);
          TchCheck;
        end;
      7:
        begin
          N := ati_length(P);
          TchCheck;
          Result.Tag := ivDoubleList;
          SetLength(Result.Doubles, N);
          ati_to_double_list(P, PDouble(Pointer(Result.Doubles)), N);
          TchCheck;
        end;
      8:
        begin
          N := ati_length(P);
          TchCheck;
          Bytes := nil;
          SetLength(Bytes, N);
          ati_to_bool_list(P, PChar(Pointer(Bytes)), N);
          TchCheck;
          Result.Tag := ivBoolList;
          SetLength(Result.Bools, N);
          for K := 0 to N - 1 do
            Result.Bools[K] := Bytes[K] <> 0;
        end;
      9:
        begin
          Str := ati_to_string(P);
          TchCheck;
          if Str = nil then
            raise ETorchError.Create('ivalue: nil string');
          Result.Tag := ivString;
          Result.S := StrPas(Str);
          tchf_free(Str);
        end;
      10:
        begin
          N := ati_length(P);
          TchCheck;
          Slots := nil;
          SetLength(Slots, N);
          ati_to_tensor_list(P, PPTorchTensor(Pointer(Slots)), N);
          TchCheckSlots(PPTorchTensor(Pointer(Slots)), N);
          Result.Tag := ivTensorList;
          SetLength(Result.Tensors, N);
          for K := 0 to N - 1 do
            Result.Tensors[K] := TTensor.FromRaw(Slots[K]);
        end;
      14:
        begin
          Result.Tag := ivObject;
          Result.Obj := TJitObjectHandle.Create(P);
          Keep := True;
        end;
    else
      raise ETorchError.CreateFmt('ivalue: unsupported tag %d', [CTag]);
    end;
  finally
    FreeIValues(Cs);
    { tag 14: the handle owns P }
    if not Keep then
      ati_free(P);
  end;
end;

{ TJitObjectHandle }

{ FRaw is assigned last: if anything before it raised, the destructor
  frees nothing and FromC's finally frees P. }
constructor TJitObjectHandle.Create(P: PTorchIValue);
begin
  inherited Create;
  InterLockedIncrement(LiveObjects);
  FRaw := P;
end;

destructor TJitObjectHandle.Destroy;
begin
  if FRaw <> nil then
    ati_free(FRaw);
  InterLockedDecrement(LiveObjects);
  inherited Destroy;
end;

function TJitObjectHandle.Raw: PTorchIValue;
begin
  Result := FRaw;
end;

{ C pushes a copy of FRaw as self, so FRaw stays valid. }
function TJitObjectHandle.MethodIV(const Name: String;
  const Inputs: array of TIValue): TIValue;
var
  Cs: TIValuePtrArray;
  P: PTorchIValue;
begin
  Cs := IValuesToC(Inputs);
  try
    P := ati_object_method_(FRaw, PChar(Name), PPTorchIValue(Pointer(Cs)),
      Length(Cs));
    TchCheck;
  finally
    FreeIValues(Cs);
  end;
  Result := TIValue.FromC(P);
end;

function TJitObjectHandle.GetAttr(const Name: String): TIValue;
var
  P: PTorchIValue;
begin
  P := ati_object_getattr_(FRaw, PChar(Name));
  TchCheck;
  if P = nil then
    raise ETorchError.Create('ivalue: getattr returned nil');
  Result := TIValue.FromC(P);
end;

{ TJitModule }

constructor TJitModule.CreateRaw(M: PTorchModule);
begin
  inherited Create;
  FModule := M;
end;

constructor TJitModule.Create(const FileName: String);
var
  M: PTorchModule;
begin
  inherited Create;
  M := atm_load(PChar(FileName));
  TchCheck;
  FModule := M;
end;

constructor TJitModule.Create(const FileName: String; const ADevice: TDevice);
var
  M: PTorchModule;
begin
  inherited Create;
  M := atm_load_on_device(PChar(FileName), ADevice.ToC);
  TchCheck;
  FModule := M;
end;

function ReadRest(S: TStream): TBytes;
begin
  Result := nil;
  SetLength(Result, S.Size - S.Position);
  if Length(Result) = 0 then
    raise ETorchError.Create('TJitModule.CreateFromStream: empty stream');
  S.ReadBuffer(Result[0], Length(Result));
end;

constructor TJitModule.CreateFromStream(S: TStream);
var
  Buf: TBytes;
  M: PTorchModule;
begin
  inherited Create;
  Buf := ReadRest(S);
  M := atm_load_str(PChar(@Buf[0]), Length(Buf));
  TchCheck;
  FModule := M;
end;

constructor TJitModule.CreateFromStream(S: TStream; const ADevice: TDevice);
var
  Buf: TBytes;
  M: PTorchModule;
begin
  inherited Create;
  Buf := ReadRest(S);
  M := atm_load_str_on_device(PChar(@Buf[0]), Length(Buf), ADevice.ToC);
  TchCheck;
  FModule := M;
end;

constructor TJitModule.CreateEmpty(const ModuleName: String);
var
  M: PTorchModule;
begin
  inherited Create;
  M := atm_create(PChar(ModuleName));
  TchCheck;
  FModule := M;
end;

{ libtorch's tracing state is global and libtch cannot abandon it, so a
  failure after atm_create_for_tracing ends the trace with no outputs and
  drops the module (H23). }
class function TJitModule.Trace(const ModuleName, FnName: String;
  const Inputs: array of TTensor; Fn: TTraceFunc;
  Meth: TTraceMethod): TJitModule;
var
  InPtrs, OutPtrs: TTensorPtrArray;
  Ins, Outs: TTensorArray;
  M: PTorchModule;
  P: PChar;
  K: Integer;
begin
  InPtrs := TchTensorPtrs(Inputs, 'inputs', 'CreateByTracing');
  Ins := nil;
  SetLength(Ins, Length(Inputs));
  for K := 0 to High(Inputs) do
    Ins[K] := Inputs[K];
  M := atm_create_for_tracing(PChar(ModuleName), PPTorchTensor(Pointer(InPtrs)),
    Length(InPtrs));
  TchCheck;
  try
    if Assigned(Fn) then
      Outs := Fn(Ins)
    else
      Outs := Meth(Ins);
    OutPtrs := TchTensorPtrs(Outs, 'outputs', 'CreateByTracing');
    atm_end_tracing(M, PChar(FnName), PPTorchTensor(Pointer(OutPtrs)),
      Length(OutPtrs));
    TchCheck;
  except
    atm_end_tracing(M, PChar(FnName), nil, 0);
    P := get_and_reset_last_err;
    if P <> nil then
      tchf_free(P);
    atm_free(M);
    raise;
  end;
  Result := TJitModule.CreateRaw(M);
end;

class function TJitModule.CreateByTracing(const ModuleName, FnName: String;
  const Inputs: array of TTensor; Fn: TTraceFunc): TJitModule;
begin
  Result := Trace(ModuleName, FnName, Inputs, Fn, nil);
end;

class function TJitModule.CreateByTracing(const ModuleName, FnName: String;
  const Inputs: array of TTensor; Fn: TTraceMethod): TJitModule;
begin
  Result := Trace(ModuleName, FnName, Inputs, nil, Fn);
end;

destructor TJitModule.Destroy;
begin
  if FModule <> nil then
    atm_free(FModule);
  inherited Destroy;
end;

function TJitModule.Forward(const Inputs: array of TTensor): TTensor;
var
  Ptrs: TTensorPtrArray;
  P: PTorchTensor;
begin
  Ptrs := TchTensorPtrs(Inputs, 'inputs', 'Forward');
  P := atm_forward(FModule, PPTorchTensor(Pointer(Ptrs)), Length(Ptrs));
  TchCheck;
  Result := TTensor.FromRaw(P);
end;

function TJitModule.ForwardIV(const Inputs: array of TIValue): TIValue;
var
  Cs: TIValuePtrArray;
  P: PTorchIValue;
begin
  Cs := IValuesToC(Inputs);
  try
    P := atm_forward_(FModule, PPTorchIValue(Pointer(Cs)), Length(Cs));
    TchCheck;
  finally
    FreeIValues(Cs);
  end;
  Result := TIValue.FromC(P);
end;

function TJitModule.Method(const Name: String;
  const Inputs: array of TTensor): TTensor;
var
  Ptrs: TTensorPtrArray;
  P: PTorchTensor;
begin
  Ptrs := TchTensorPtrs(Inputs, 'inputs', 'Method');
  P := atm_method(FModule, PChar(Name), PPTorchTensor(Pointer(Ptrs)),
    Length(Ptrs));
  TchCheck;
  Result := TTensor.FromRaw(P);
end;

function TJitModule.MethodIV(const Name: String;
  const Inputs: array of TIValue): TIValue;
var
  Cs: TIValuePtrArray;
  P: PTorchIValue;
begin
  Cs := IValuesToC(Inputs);
  try
    P := atm_method_(FModule, PChar(Name), PPTorchIValue(Pointer(Cs)),
      Length(Cs));
    TchCheck;
  finally
    FreeIValues(Cs);
  end;
  Result := TIValue.FromC(P);
end;

procedure TJitModule.SetEval;
begin
  atm_eval(FModule);
  TchCheck;
end;

procedure TJitModule.SetTrain;
begin
  atm_train(FModule);
  TchCheck;
end;

procedure TJitModule.ToDevice(const ADevice: TDevice; AKind: TKind;
  NonBlocking: Boolean);
begin
  atm_to(FModule, ADevice.ToC, Ord(AKind), NonBlocking);
  TchCheck;
end;

procedure TJitModule.Save(const FileName: String);
begin
  atm_save(FModule, PChar(FileName));
  TchCheck;
end;

{ The trampoline of TchNN's LoadCallback (DESIGN 4.6.3), duplicated rather
  than exported from TchNN. }
type
  TParamContext = class
    Items: TNamedTensorArray;
    Error: Exception;             // acquired exception object, nil = none
    destructor Destroy; override;
  end;

destructor TParamContext.Destroy;
begin
  Error.Free;
  inherited Destroy;
end;

{ Called from C++ frames: must not let an exception escape. }
procedure ParamCallback(data: Pointer; name: PChar; t: PTorchTensor); cdecl;
var
  Ctx: TParamContext;
  T1: TTensor;
begin
  Ctx := TParamContext(data);
  try
    T1 := TTensor.FromRaw(t);     // own the handle on every path
    if Ctx.Error <> nil then
      Exit;
    SetLength(Ctx.Items, Length(Ctx.Items) + 1);
    Ctx.Items[High(Ctx.Items)].Name := StrPas(name);
    Ctx.Items[High(Ctx.Items)].Tensor := T1;
  except
    Ctx.Error := Exception(AcquireExceptionObject);
  end;
end;

function TJitModule.NamedParameters: TNamedTensorArray;
var
  Ctx: TParamContext;
  P: PChar;
  E: Exception;
begin
  Ctx := TParamContext.Create;
  try
    atm_named_parameters(FModule, Ctx, @ParamCallback);
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

procedure TJitModule.RegisterParameter(const Name: String; const T: TTensor;
  IsBuffer: Boolean);
begin
  TchCheckArg(T, 't', 'RegisterParameter');
  atm_register_parameter(FModule, PChar(Name), T.Raw, IsBuffer);
  TchCheck;
end;

procedure TJitModule.Define(const Source: String);
begin
  atm_define(FModule, PChar(Source));
  TchCheck;
end;

function TJitModule.CreateClass(const QualifiedName: String;
  const Inputs: array of TIValue): IJitObject;
var
  Cs: TIValuePtrArray;
  P: PTorchIValue;
begin
  Cs := IValuesToC(Inputs);
  try
    P := atm_create_class_(FModule, PChar(QualifiedName),
      PPTorchIValue(Pointer(Cs)), Length(Cs));
    TchCheck;
  finally
    FreeIValues(Cs);
  end;
  Result := TIValue.FromC(P).AsObject;
end;

{ TTrainableJitModule }

constructor TTrainableJitModule.Create(const FileName: String;
  const P: TVarPath);
begin
  inherited Create;
  FInner := TJitModule.Create(FileName, P.Device);
  RegisterInto(P);
end;

constructor TTrainableJitModule.CreateFromStream(S: TStream;
  const P: TVarPath);
begin
  inherited Create;
  FInner := TJitModule.CreateFromStream(S, P.Device);
  RegisterInto(P);
end;

destructor TTrainableJitModule.Destroy;
begin
  FInner.Free;
  inherited Destroy;
end;

{ tch-rs TrainableCModule::load: the store entry is the module's parameter
  (shared handle); trainable = the loaded requires_grad. }
{ A failure part-way removes the parameters already added (RollbackTo). }
procedure TTrainableJitModule.RegisterInto(const P: TVarPath);
var
  Params: TNamedTensorArray;
  K, Mark: Integer;
begin
  Mark := P.Store.Count;
  try
    Params := FInner.NamedParameters;
    for K := 0 to High(Params) do
      P.Add(StringReplace(Params[K].Name, '.', '_', [rfReplaceAll]),
        Params[K].Tensor, Params[K].Tensor.RequiresGrad);
  except
    P.Store.RollbackTo(Mark);
    raise;
  end;
end;

function TTrainableJitModule.Forward(const X: TTensor; Train: Boolean): TTensor;
begin
  if (FMode = jmUnknown) or ((FMode = jmTrain) <> Train) then
    if Train then
      SetTrain
    else
      SetEval;
  Result := FInner.Forward([X]);
end;

function TTrainableJitModule.ForwardIV(const Inputs: array of TIValue): TIValue;
begin
  Result := FInner.ForwardIV(Inputs);
end;

function TTrainableJitModule.Method(const Name: String;
  const Inputs: array of TTensor): TTensor;
begin
  Result := FInner.Method(Name, Inputs);
end;

function TTrainableJitModule.MethodIV(const Name: String;
  const Inputs: array of TIValue): TIValue;
begin
  Result := FInner.MethodIV(Name, Inputs);
end;

procedure TTrainableJitModule.Save(const FileName: String);
begin
  FInner.Save(FileName);
end;

procedure TTrainableJitModule.SetTrain;
begin
  FInner.SetTrain;
  FMode := jmTrain;
end;

procedure TTrainableJitModule.SetEval;
begin
  FInner.SetEval;
  FMode := jmEval;
end;

{ globals }

function JitGetProfilingMode: Boolean;
begin
  Result := atm_get_profiling_mode <> 0;
  TchCheck;
end;

procedure JitSetProfilingMode(B: Boolean);
begin
  atm_set_profiling_mode(Ord(B));
  TchCheck;
end;

function JitGetTensorExprFuserEnabled: Boolean;
begin
  Result := atm_get_tensor_expr_fuser_enabled;
  TchCheck;
end;

procedure JitSetTensorExprFuserEnabled(B: Boolean);
begin
  atm_set_tensor_expr_fuser_enabled(Ord(B));
  TchCheck;
end;

procedure JitSetGraphExecutorOptimize(B: Boolean);
begin
  at_set_graph_executor_optimize(B);
end;

function JitFuserCudaEnabled: Boolean;
begin
  Result := atm_fuser_cuda_is_enabled;
  TchCheck;
end;

procedure SetJitFuserCudaEnabled(B: Boolean);
begin
  atm_fuser_cuda_set_enabled(B);
  TchCheck;
end;

function JitLiveObjects: Integer;
begin
  Result := LiveObjects;
end;

end.
