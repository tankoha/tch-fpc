{ Hand-checkable layer setups shared by tests/nn (CPU) and tests/gpu, so the
  two cannot drift apart: the Conv2D padding modes of DESIGN.md 4.6.7
  group 8 and the RNN hand numbers of 4.6.10 item 8, on any device.
  Include after the uses clause (needs Tch and TchNN). }

const
  ConvPadModeNames: array[TPaddingMode] of String =
    ('zeros', 'reflect', 'replicate', 'circular');
  { TConv2D(1, 1, 3), Padding 1, no bias, weights 1, on [[1, 2], [3, 4]] }
  ConvPadExpected: array[TPaddingMode] of String =
    ('10 10 10 10', '27 24 21 18', '18 21 24 27', '27 24 21 18');
  { all weights 0; GRU bias_ih_l0 = [0, 0, 1]: h = tanh(1) / 2;
    LSTM bias_ih_l0 = [0, 0, 1, 0]: c = tanh(1) / 2, h = tanh(c) / 2 }
  HandGruBias: array[0..2] of Double = (0, 0, 1);
  HandLstmBias: array[0..3] of Double = (0, 0, 1, 0);
  HandGruH = 0.380797;
  HandLstmC = 0.380797;
  HandLstmH = 0.181700;

{ The conv above on Dev; the layer and its store are gone on return. }
function HandConvPadded(const Dev: TDevice; Mode: TPaddingMode): TTensor;
var
  VS: TVarStore;
  Conv: TConv2D;
  Cfg: TConvConfig;
begin
  VS := TVarStore.Create(Dev);
  Conv := nil;
  try
    Cfg := DefaultConvConfig;
    Cfg.Padding := 1;
    Cfg.Bias := False;
    Cfg.PaddingMode := Mode;
    Conv := TConv2D.Create(VS.Root, 1, 1, 3, Cfg);
    TInit.Constant(1).Apply(Conv.Weight);
    Result := Conv.Forward(TTensor.FromArray([1.0, 2, 3, 4]).view([1, 1, 2, 2])
      .to_device(Dev, kFloat));
  finally
    Conv.Free;
    VS.Free;
  end;
end;

{ Zeros every weight of R, then sets bias_ih_l0 (FlatWeights[2]) to BiasIh. }
procedure SetHandRNNWeights(R: TRNN; const BiasIh: array of Double);
var
  W: TTensor;
  G: IInterface;
begin
  G := NoGradGuard;
  for W in R.FlatWeights do
    W.CopyFrom(W.zeros_like);
  W := R.FlatWeights[2];
  W.CopyFrom(TTensor.FromArray(BiasIh).to_device(W.Device, W.Kind));
end;
