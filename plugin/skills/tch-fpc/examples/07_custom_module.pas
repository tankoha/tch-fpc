{ A TModule descendant, a TSequential, and TFuncModule with a named
  function (FPC 3.2.2 has no anonymous functions). }
program custom_module;

{$mode objfpc}{$H+}

uses
  SysUtils, Tch, TchNN;

type
  { Overrides of Forward must repeat the default value of Train. }
  TTwoLayer = class(TModule)
  private
    FSeq: TSequential;
  public
    constructor Create(const P: TVarPath);
    destructor Destroy; override;
    function Forward(const X: TTensor; Train: Boolean = False): TTensor; override;
  end;

function Relu(const X: TTensor): TTensor;
begin
  Result := X.relu;
end;

constructor TTwoLayer.Create(const P: TVarPath);
begin
  inherited Create;
  FSeq := TSequential.Create;
  { the sequential owns and frees the layers added to it }
  FSeq.Add(TLinear.Create(P / 'l1', 4, 8))
      .AddFn(@Relu)
      .Add(TLinear.Create(P / 'l2', 8, 2));
end;

destructor TTwoLayer.Destroy;
begin
  FSeq.Free;
  inherited Destroy;
end;

function TTwoLayer.Forward(const X: TTensor; Train: Boolean): TTensor;
begin
  Result := FSeq.Forward(X, Train);
end;

var
  VS: TVarStore;
  Net: TTwoLayer;
  Act: TFuncModule;
  Y: TTensor;
begin
  VS := TVarStore.Create(TDevice.Cpu);
  try
    Net := TTwoLayer.Create(VS.Root / 'net');
    Act := TFuncModule.Create(@Relu);
    try
      Y := Act.Forward(Net.Forward(TTensor.ones([5, 4])));
      WriteLn('output shape ', Y.Shape[0], 'x', Y.Shape[1]);
      WriteLn('non-negative: ', Y.min.DoubleValue([]) >= 0);
      WriteLn('variables: ', VS.Count);
    finally
      Act.Free;
      Net.Free;
    end;
  finally
    VS.Free;
  end;
end.
