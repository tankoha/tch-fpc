{ UT program for src/DeltaRule.pas: the rows UT-DeltaRule-NNN of
  tests/ut/cases/DeltaRule.cases.md (stage 1 from 001: DeltaStep; stage 2
  from 101: DeltaRuleTokens). Random inputs and the scalar Double reference
  come from the harness owner's tests/ut/DeltaRuleFixture.pas.
  Runs on the CPU (.sdlc/ut-coverage.sh hides the GPUs).
  Invoked with `--all --format=plain` (see .sdlc/ut-coverage.sh).

  Skeleton by the harness owner; the test methods are written in T08b. }
program DeltaRule_test;

{$mode objfpc}{$H+}

uses
  SysUtils, Math, Tch, DeltaRule, DeltaRuleFixture,
  fpcunit, testregistry, consoletestrunner;

type
  TDeltaRuleTest = class(TTestCase)
  public
    { A kFloat tensor of the given shape from the values (row-major). }
    class function Make(const AValues: array of Double; const AShape: array of Int64): TTensor; static;
    { Fails unless T has the shape (as ShapeText) and the values (as ValuesText). }
    procedure CheckTensor(const AWhat, AShape, AValues: string; const T: TTensor);
  published
    { Stage 1: DeltaStep }
    procedure TestUT_DeltaRule_001;
    procedure TestUT_DeltaRule_002;
    procedure TestUT_DeltaRule_003;
    procedure TestUT_DeltaRule_004;
    procedure TestUT_DeltaRule_005;
    procedure TestUT_DeltaRule_006;
    procedure TestUT_DeltaRule_007;
    procedure TestUT_DeltaRule_008;
    procedure TestUT_DeltaRule_009;
    procedure TestUT_DeltaRule_010;
    procedure TestUT_DeltaRule_011;
    procedure TestUT_DeltaRule_012;
    procedure TestUT_DeltaRule_013;
    procedure TestUT_DeltaRule_014;
    procedure TestUT_DeltaRule_015;
    procedure TestUT_DeltaRule_016;
    { Stage 1: DeltaStep (kind checks added for coverage) }
    procedure TestUT_DeltaRule_017;
    procedure TestUT_DeltaRule_018;
    procedure TestUT_DeltaRule_019;
    procedure TestUT_DeltaRule_020;
    { Stage 2: DeltaRuleTokens (T08b2) }
    procedure TestUT_DeltaRule_101;
    procedure TestUT_DeltaRule_102;
    procedure TestUT_DeltaRule_103;
    procedure TestUT_DeltaRule_104;
    procedure TestUT_DeltaRule_105;
    procedure TestUT_DeltaRule_106;
    procedure TestUT_DeltaRule_107;
    procedure TestUT_DeltaRule_108;
    procedure TestUT_DeltaRule_109;
    procedure TestUT_DeltaRule_110;
    procedure TestUT_DeltaRule_111;
    procedure TestUT_DeltaRule_112;
    procedure TestUT_DeltaRule_113;
    procedure TestUT_DeltaRule_114;
    procedure TestUT_DeltaRule_115;
    procedure TestUT_DeltaRule_116;
    procedure TestUT_DeltaRule_117;
    { Stage 2: DeltaRuleTokens (kind checks added for coverage) }
    procedure TestUT_DeltaRule_118;
    procedure TestUT_DeltaRule_119;
    procedure TestUT_DeltaRule_120;
    { Stage T09b1: DeltaRuleChunked checks }
    procedure TestUT_DeltaRule_201;
    procedure TestUT_DeltaRule_202;
    procedure TestUT_DeltaRule_203;
    procedure TestUT_DeltaRule_204;
    procedure TestUT_DeltaRule_205;
    procedure TestUT_DeltaRule_206;
    procedure TestUT_DeltaRule_207;
    procedure TestUT_DeltaRule_208;
    procedure TestUT_DeltaRule_209;
    procedure TestUT_DeltaRule_210;
    procedure TestUT_DeltaRule_211;
    procedure TestUT_DeltaRule_212;
    { Stage T09b2: DeltaRuleChunked values }
    procedure TestUT_DeltaRule_213;
    procedure TestUT_DeltaRule_214;
    procedure TestUT_DeltaRule_215;
    procedure TestUT_DeltaRule_216;
    procedure TestUT_DeltaRule_217;
    procedure TestUT_DeltaRule_218;
    procedure TestUT_DeltaRule_219;
    procedure TestUT_DeltaRule_220;
    procedure TestUT_DeltaRule_221;
    procedure TestUT_DeltaRule_222;
    procedure TestUT_DeltaRule_223;
    procedure TestUT_DeltaRule_224;
  end;

class function TDeltaRuleTest.Make(const AValues: array of Double; const AShape: array of Int64): TTensor;
begin
  Result := TTensor.FromArray(AValues).reshape(AShape);
end;

procedure TDeltaRuleTest.CheckTensor(const AWhat, AShape, AValues: string; const T: TTensor);
begin
  AssertEquals(AWhat + ' shape', AShape, ShapeText(T));
  AssertEquals(AWhat + ' values', AValues, ValuesText(T));
end;

{ Stage 1: DeltaStep tests (001-016) }

procedure TDeltaRuleTest.TestUT_DeltaRule_001;
var
  M, Q, K, V, Beta, G, NewM, O: TTensor;
begin
  M := TTensor.zeros([1, 2, 2]);
  Q := Make([1, 1], [1, 2]);
  K := Make([1, 0], [1, 2]);
  V := Make([2, 4], [1, 2]);
  Beta := Make([0.5], [1]);
  G := Make([0], [1]);
  NewM := DeltaStep(M, Q, K, V, Beta, G, O);
  CheckTensor('result', '[1, 2, 2]', '1 0 2 0', NewM);
  CheckTensor('O', '[1, 2]', '1 2', O);
end;

procedure TDeltaRuleTest.TestUT_DeltaRule_002;
var
  M, Q, K, V, Beta, G, NewM, O: TTensor;
begin
  M := Make([1, 0, 2, 0], [1, 2, 2]);
  Q := Make([1, 2], [1, 2]);
  K := Make([1, 1], [1, 2]);
  V := Make([3, 3], [1, 2]);
  Beta := Make([0.5], [1]);
  G := Make([0], [1]);
  NewM := DeltaStep(M, Q, K, V, Beta, G, O);
  CheckTensor('result', '[1, 2, 2]', '2 1 2.5 0.5', NewM);
  CheckTensor('O', '[1, 2]', '4 3.5', O);
end;

procedure TDeltaRuleTest.TestUT_DeltaRule_003;
var
  M, Q, K, V, Beta, G, NewM, O: TTensor;
begin
  M := Make([1, 0, 2, 0], [1, 2, 2]);
  Q := Make([1, 2], [1, 2]);
  K := Make([1, 1], [1, 2]);
  V := Make([3, 3], [1, 2]);
  Beta := Make([0.5], [1]);
  G := Make([Ln(0.5)], [1]);
  NewM := DeltaStep(M, Q, K, V, Beta, G, O);
  CheckTensor('result', '[1, 2, 2]', '1.75 1.25 2 1', NewM);
  CheckTensor('O', '[1, 2]', '4.25 4', O);
end;

procedure TDeltaRuleTest.TestUT_DeltaRule_004;
var
  M, Q, K, V, Beta, G, NewM, O: TTensor;
begin
  M := Make([0, 0, 0, 0, 1, 0, 2, 0], [2, 2, 2]);
  Q := Make([1, 1, 1, 2], [2, 2]);
  K := Make([1, 0, 1, 1], [2, 2]);
  V := Make([2, 4, 3, 3], [2, 2]);
  Beta := Make([0.5, 0.5], [2]);
  G := Make([0, 0], [2]);
  NewM := DeltaStep(M, Q, K, V, Beta, G, O);
  CheckTensor('result', '[2, 2, 2]', '1 0 2 0 2 1 2.5 0.5', NewM);
  CheckTensor('O', '[2, 2]', '1 2 4 3.5', O);
end;

procedure TDeltaRuleTest.TestUT_DeltaRule_005;
var
  Q, K, V, Beta, G, M, NewM, O, ExpM, ExpO: TTensor;
begin
  RandomInputs(23, 3, 1, 4, 5, 1.0, Q, K, V, Beta, G);
  M := TTensor.randn([3, 5, 4]);
  Beta := TTensor.zeros([3]);
  NewM := DeltaStep(M, Q.select(1, 0), K.select(1, 0), V.select(1, 0), Beta, G.select(1, 0), O);
  ExpM := M * G.select(1, 0).exp.view([3, 1, 1]);
  ExpO := ExpM.matmul(Q.select(1, 0).unsqueeze(2)).squeeze_dim(2);
  CheckTensor('result', ShapeText(ExpM), ValuesText(ExpM), NewM);
  CheckTensor('O', ShapeText(ExpO), ValuesText(ExpO), O);
end;

procedure TDeltaRuleTest.TestUT_DeltaRule_006;
var
  Q, K, V, Beta, G, M, NewM, O: TTensor;
begin
  RandomInputs(23, 3, 1, 4, 5, 1.0, Q, K, V, Beta, G);
  M := TTensor.randn([3, 5, 4]);
  Beta := TTensor.zeros([3]);
  G := TTensor.zeros([3]);
  NewM := DeltaStep(M, Q.select(1, 0), K.select(1, 0), V.select(1, 0), Beta, G, O);
  CheckTensor('result', ShapeText(M), ValuesText(M), NewM);
end;

procedure TDeltaRuleTest.TestUT_DeltaRule_007;
var
  Q, K, V, Beta, G, M, NewM, O, RO, RS: TTensor;
begin
  RandomInputs(21, 3, 1, 4, 5, 1.0, Q, K, V, Beta, G);
  M := TTensor.zeros([3, 5, 4]);
  NewM := DeltaStep(M, Q.select(1, 0), K.select(1, 0), V.select(1, 0), Beta.select(1, 0), G.select(1, 0), O);
  RefTokens(Q, K, V, Beta, G, RO, RS);
  AssertTrue('O diff', MaxAbsDiff(O, RO.select(1, 0)) <= 1e-6);
  AssertTrue('M diff', MaxAbsDiff(NewM, RS) <= 1e-6);
end;

procedure TDeltaRuleTest.TestUT_DeltaRule_008;
var
  Q, K, V, Beta, G, M, NewM, O, RO, RS: TTensor;
begin
  RandomInputs(22, 3, 2, 4, 5, 1.0, Q, K, V, Beta, G);
  M := TTensor.zeros([3, 5, 4]);
  { First step }
  NewM := DeltaStep(M, Q.select(1, 0), K.select(1, 0), V.select(1, 0), Beta.select(1, 0), G.select(1, 0), O);
  { Second step }
  NewM := DeltaStep(NewM, Q.select(1, 1), K.select(1, 1), V.select(1, 1), Beta.select(1, 1), G.select(1, 1), O);
  RefTokens(Q, K, V, Beta, G, RO, RS);
  AssertTrue('O diff', MaxAbsDiff(O, RO.select(1, 1)) <= 1e-6);
  AssertTrue('M diff', MaxAbsDiff(NewM, RS) <= 1e-6);
end;

procedure TDeltaRuleTest.TestUT_DeltaRule_009;
var
  M, Q, K, V, Beta, G, NewM, O: TTensor;
begin
  M := TTensor.zeros([2, 5, 4], kDouble);
  Q := TTensor.zeros([2, 4]);
  K := TTensor.zeros([2, 4]);
  V := TTensor.zeros([2, 5]);
  Beta := TTensor.zeros([2]);
  G := TTensor.zeros([2]);
  try
    NewM := DeltaStep(M, Q, K, V, Beta, G, O);
    Fail('no EDeltaRule');
  except
    on E: EDeltaRule do
      AssertEquals('M must be kFloat (fp32)', E.Message);
  end;
end;

procedure TDeltaRuleTest.TestUT_DeltaRule_010;
var
  M, Q, K, V, Beta, G, NewM, O: TTensor;
begin
  M := TTensor.zeros([2, 5, 4]);
  Q := TTensor.zeros([2, 4]);
  K := TTensor.zeros([2, 4]);
  V := TTensor.zeros([2, 5]);
  Beta := TTensor.zeros([2]);
  G := TTensor.zeros([2], kBFloat16);
  try
    NewM := DeltaStep(M, Q, K, V, Beta, G, O);
    Fail('no EDeltaRule');
  except
    on E: EDeltaRule do
      AssertEquals('G must be kFloat (fp32)', E.Message);
  end;
end;

procedure TDeltaRuleTest.TestUT_DeltaRule_011;
var
  M, Q, K, V, Beta, G, NewM, O: TTensor;
begin
  M := TTensor.zeros([5, 4]);
  Q := TTensor.zeros([2, 4]);
  K := TTensor.zeros([2, 4]);
  V := TTensor.zeros([2, 5]);
  Beta := TTensor.zeros([2]);
  G := TTensor.zeros([2]);
  try
    NewM := DeltaStep(M, Q, K, V, Beta, G, O);
    Fail('no EDeltaRule');
  except
    on E: EDeltaRule do
      AssertEquals('M must be [H, Dv, Dk], got [5, 4]', E.Message);
  end;
end;

procedure TDeltaRuleTest.TestUT_DeltaRule_012;
var
  M, Q, K, V, Beta, G, NewM, O: TTensor;
begin
  M := TTensor.zeros([2, 5, 4]);
  Q := TTensor.zeros([2, 3]);
  K := TTensor.zeros([2, 4]);
  V := TTensor.zeros([2, 5]);
  Beta := TTensor.zeros([2]);
  G := TTensor.zeros([2]);
  try
    NewM := DeltaStep(M, Q, K, V, Beta, G, O);
    Fail('no EDeltaRule');
  except
    on E: EDeltaRule do
      AssertEquals('Q must be [2, 4], got [2, 3]', E.Message);
  end;
end;

procedure TDeltaRuleTest.TestUT_DeltaRule_013;
var
  M, Q, K, V, Beta, G, NewM, O: TTensor;
begin
  M := TTensor.zeros([2, 5, 4]);
  Q := TTensor.zeros([2, 4]);
  K := TTensor.zeros([1, 4]);
  V := TTensor.zeros([2, 5]);
  Beta := TTensor.zeros([2]);
  G := TTensor.zeros([2]);
  try
    NewM := DeltaStep(M, Q, K, V, Beta, G, O);
    Fail('no EDeltaRule');
  except
    on E: EDeltaRule do
      AssertEquals('K must be [2, 4], got [1, 4]', E.Message);
  end;
end;

procedure TDeltaRuleTest.TestUT_DeltaRule_014;
var
  M, Q, K, V, Beta, G, NewM, O: TTensor;
begin
  M := TTensor.zeros([2, 5, 4]);
  Q := TTensor.zeros([2, 4]);
  K := TTensor.zeros([2, 4]);
  V := TTensor.zeros([2, 4]);
  Beta := TTensor.zeros([2]);
  G := TTensor.zeros([2]);
  try
    NewM := DeltaStep(M, Q, K, V, Beta, G, O);
    Fail('no EDeltaRule');
  except
    on E: EDeltaRule do
      AssertEquals('V must be [2, 5], got [2, 4]', E.Message);
  end;
end;

procedure TDeltaRuleTest.TestUT_DeltaRule_015;
var
  M, Q, K, V, Beta, G, NewM, O: TTensor;
begin
  M := TTensor.zeros([2, 5, 4]);
  Q := TTensor.zeros([2, 4]);
  K := TTensor.zeros([2, 4]);
  V := TTensor.zeros([2, 5]);
  Beta := TTensor.zeros([3]);
  G := TTensor.zeros([2]);
  try
    NewM := DeltaStep(M, Q, K, V, Beta, G, O);
    Fail('no EDeltaRule');
  except
    on E: EDeltaRule do
      AssertEquals('Beta must be [2], got [3]', E.Message);
  end;
end;

procedure TDeltaRuleTest.TestUT_DeltaRule_016;
var
  M, Q, K, V, Beta, G, NewM, O: TTensor;
begin
  M := TTensor.zeros([2, 5, 4]);
  Q := TTensor.zeros([2, 4]);
  K := TTensor.zeros([2, 4]);
  V := TTensor.zeros([2, 5]);
  Beta := TTensor.zeros([2]);
  G := TTensor.zeros([2, 1]);
  try
    NewM := DeltaStep(M, Q, K, V, Beta, G, O);
    Fail('no EDeltaRule');
  except
    on E: EDeltaRule do
      AssertEquals('G must be [2], got [2, 1]', E.Message);
  end;
end;

{ Stage 1: DeltaStep kind check tests (017-020, added for coverage) }

procedure TDeltaRuleTest.TestUT_DeltaRule_017;
var
  M, Q, K, V, Beta, G, NewM, O: TTensor;
begin
  M := TTensor.zeros([2, 5, 4]);
  Q := TTensor.zeros([2, 4], kDouble);
  K := TTensor.zeros([2, 4]);
  V := TTensor.zeros([2, 5]);
  Beta := TTensor.zeros([2]);
  G := TTensor.zeros([2]);
  try
    NewM := DeltaStep(M, Q, K, V, Beta, G, O);
    Fail('no EDeltaRule');
  except
    on E: EDeltaRule do
      AssertEquals('Q must be kFloat (fp32)', E.Message);
  end;
end;

procedure TDeltaRuleTest.TestUT_DeltaRule_018;
var
  M, Q, K, V, Beta, G, NewM, O: TTensor;
begin
  M := TTensor.zeros([2, 5, 4]);
  Q := TTensor.zeros([2, 4]);
  K := TTensor.zeros([2, 4], kDouble);
  V := TTensor.zeros([2, 5]);
  Beta := TTensor.zeros([2]);
  G := TTensor.zeros([2]);
  try
    NewM := DeltaStep(M, Q, K, V, Beta, G, O);
    Fail('no EDeltaRule');
  except
    on E: EDeltaRule do
      AssertEquals('K must be kFloat (fp32)', E.Message);
  end;
end;

procedure TDeltaRuleTest.TestUT_DeltaRule_019;
var
  M, Q, K, V, Beta, G, NewM, O: TTensor;
begin
  M := TTensor.zeros([2, 5, 4]);
  Q := TTensor.zeros([2, 4]);
  K := TTensor.zeros([2, 4]);
  V := TTensor.zeros([2, 5], kDouble);
  Beta := TTensor.zeros([2]);
  G := TTensor.zeros([2]);
  try
    NewM := DeltaStep(M, Q, K, V, Beta, G, O);
    Fail('no EDeltaRule');
  except
    on E: EDeltaRule do
      AssertEquals('V must be kFloat (fp32)', E.Message);
  end;
end;

procedure TDeltaRuleTest.TestUT_DeltaRule_020;
var
  M, Q, K, V, Beta, G, NewM, O: TTensor;
begin
  M := TTensor.zeros([2, 5, 4]);
  Q := TTensor.zeros([2, 4]);
  K := TTensor.zeros([2, 4]);
  V := TTensor.zeros([2, 5]);
  Beta := TTensor.zeros([2], kDouble);
  G := TTensor.zeros([2]);
  try
    NewM := DeltaStep(M, Q, K, V, Beta, G, O);
    Fail('no EDeltaRule');
  except
    on E: EDeltaRule do
      AssertEquals('Beta must be kFloat (fp32)', E.Message);
  end;
end;

{ Stage 2: DeltaRuleTokens tests (101-117) }

procedure TDeltaRuleTest.TestUT_DeltaRule_101;
var
  Q, K, V, Beta, G, O, State: TTensor;
begin
  Q := Make([1, 1, 1, 2], [1, 2, 2]);
  K := Make([1, 0, 1, 1], [1, 2, 2]);
  V := Make([2, 4, 3, 3], [1, 2, 2]);
  Beta := Make([0.5, 0.5], [1, 2]);
  G := Make([0, 0], [1, 2]);
  O := DeltaRuleTokens(Q, K, V, Beta, G, State);
  CheckTensor('O', '[1, 2, 2]', '1 2 4 3.5', O);
  CheckTensor('State', '[1, 2, 2]', '2 1 2.5 0.5', State);
end;

procedure TDeltaRuleTest.TestUT_DeltaRule_102;
var
  Q, K, V, Beta, G, O, State: TTensor;
begin
  Q := Make([1, 1, 1, 2], [1, 2, 2]);
  K := Make([1, 0, 1, 1], [1, 2, 2]);
  V := Make([2, 4, 3, 3], [1, 2, 2]);
  Beta := Make([0.5, 0.5], [1, 2]);
  G := Make([0, Ln(0.5)], [1, 2]);
  O := DeltaRuleTokens(Q, K, V, Beta, G, State);
  CheckTensor('O', '[1, 2, 2]', '1 2 4.25 4', O);
  CheckTensor('State', '[1, 2, 2]', '1.75 1.25 2 1', State);
end;

procedure TDeltaRuleTest.TestUT_DeltaRule_103;
var
  Q, K, V, Beta, G, O, State: TTensor;
begin
  RandomInputs(7, 3, 5, 4, 6, 1.0, Q, K, V, Beta, G);
  Beta := TTensor.zeros([3, 5]);
  O := DeltaRuleTokens(Q, K, V, Beta, G, State);
  AssertEquals('O shape', '[3, 5, 6]', ShapeText(O));
  AssertEquals('State shape', '[3, 6, 4]', ShapeText(State));
  AssertTrue('O.abs.max is 0', O.abs.max.DoubleValue([]) <= 0);
  AssertTrue('State.abs.max is 0', State.abs.max.DoubleValue([]) <= 0);
end;

procedure TDeltaRuleTest.TestUT_DeltaRule_104;
var
  Q, K, V, Beta, G, O, State, ExpO: TTensor;
begin
  RandomInputs(8, 3, 5, 4, 6, 1.0, Q, K, V, Beta, G);
  G := TTensor.zeros([3, 5]) + Double(NegInfinity);
  O := DeltaRuleTokens(Q, K, V, Beta, G, State);
  ExpO := (V * Beta.unsqueeze(2)) * (K * Q).sum_dim_intlist(IntList([2]), True, kNone);
  AssertTrue('O diff', MaxAbsDiff(O, ExpO) <= 1e-6);
end;

procedure TDeltaRuleTest.TestUT_DeltaRule_105;
var
  Q, K, V, Beta, G, O, State, StepO, StepM: TTensor;
begin
  RandomInputs(9, 3, 5, 4, 6, 1.0, Q, K, V, Beta, G);
  O := DeltaRuleTokens(Q.narrow(1, 0, 1), K.narrow(1, 0, 1), V.narrow(1, 0, 1),
                       Beta.narrow(1, 0, 1), G.narrow(1, 0, 1), State);
  StepM := DeltaStep(TTensor.zeros([3, 6, 4]), Q.select(1, 0), K.select(1, 0),
                     V.select(1, 0), Beta.select(1, 0), G.select(1, 0), StepO);
  CheckTensor('O', ShapeText(StepO), ValuesText(StepO), O.select(1, 0));
  CheckTensor('State', ShapeText(StepM), ValuesText(StepM), State);
end;

procedure TDeltaRuleTest.TestUT_DeltaRule_106;
var
  Q, K, V, Beta, G, O, State, HeadO, HeadState: TTensor;
begin
  RandomInputs(9, 3, 5, 4, 6, 1.0, Q, K, V, Beta, G);
  O := DeltaRuleTokens(Q, K, V, Beta, G, State);
  HeadO := DeltaRuleTokens(Q.narrow(0, 1, 1), K.narrow(0, 1, 1), V.narrow(0, 1, 1),
                           Beta.narrow(0, 1, 1), G.narrow(0, 1, 1), HeadState);
  CheckTensor('O', ShapeText(HeadO), ValuesText(HeadO), O.narrow(0, 1, 1));
  CheckTensor('State', ShapeText(HeadState), ValuesText(HeadState), State.narrow(0, 1, 1));
end;

procedure TDeltaRuleTest.TestUT_DeltaRule_107;
var
  Q, K, V, Beta, G, O, State, RO, RS: TTensor;
begin
  RandomInputs(11, 3, 20, 4, 5, 0.1, Q, K, V, Beta, G);
  O := DeltaRuleTokens(Q, K, V, Beta, G, State);
  RefTokens(Q, K, V, Beta, G, RO, RS);
  AssertTrue('O diff', MaxAbsDiff(O, RO) <= 1e-5);
  AssertTrue('State diff', MaxAbsDiff(State, RS) <= 1e-5);
end;

procedure TDeltaRuleTest.TestUT_DeltaRule_108;
var
  Q, K, V, Beta, G, O, State, RO, RS: TTensor;
begin
  RandomInputs(12, 2, 65, 16, 16, 5.0, Q, K, V, Beta, G);
  O := DeltaRuleTokens(Q, K, V, Beta, G, State);
  RefTokens(Q, K, V, Beta, G, RO, RS);
  AssertTrue('O diff', MaxAbsDiff(O, RO) <= 1e-5);
  AssertTrue('State diff', MaxAbsDiff(State, RS) <= 1e-5);
end;

procedure TDeltaRuleTest.TestUT_DeltaRule_109;
var
  Q, K, V, Beta, G, O, State: TTensor;
begin
  Q := TTensor.zeros([2, 3, 4], kDouble);
  K := TTensor.zeros([2, 3, 4]);
  V := TTensor.zeros([2, 3, 5]);
  Beta := TTensor.zeros([2, 3]);
  G := TTensor.zeros([2, 3]);
  try
    O := DeltaRuleTokens(Q, K, V, Beta, G, State);
    Fail('no EDeltaRule');
  except
    on E: EDeltaRule do
      AssertEquals('Q must be kFloat (fp32)', E.Message);
  end;
end;

procedure TDeltaRuleTest.TestUT_DeltaRule_110;
var
  Q, K, V, Beta, G, O, State: TTensor;
begin
  Q := TTensor.zeros([2, 3, 4]);
  K := TTensor.zeros([2, 3, 4]);
  V := TTensor.zeros([2, 3, 5]);
  Beta := TTensor.zeros([2, 3]);
  G := TTensor.zeros([2, 3], kBFloat16);
  try
    O := DeltaRuleTokens(Q, K, V, Beta, G, State);
    Fail('no EDeltaRule');
  except
    on E: EDeltaRule do
      AssertEquals('G must be kFloat (fp32)', E.Message);
  end;
end;

procedure TDeltaRuleTest.TestUT_DeltaRule_111;
var
  Q, K, V, Beta, G, O, State: TTensor;
begin
  Q := TTensor.zeros([2, 3]);
  K := TTensor.zeros([2, 3, 4]);
  V := TTensor.zeros([2, 3, 5]);
  Beta := TTensor.zeros([2, 3]);
  G := TTensor.zeros([2, 3]);
  try
    O := DeltaRuleTokens(Q, K, V, Beta, G, State);
    Fail('no EDeltaRule');
  except
    on E: EDeltaRule do
      AssertEquals('Q must be [H, T, Dk], got [2, 3]', E.Message);
  end;
end;

procedure TDeltaRuleTest.TestUT_DeltaRule_112;
var
  Q, K, V, Beta, G, O, State: TTensor;
begin
  Q := TTensor.zeros([2, 0, 4]);
  K := TTensor.zeros([2, 0, 4]);
  V := TTensor.zeros([2, 0, 5]);
  Beta := TTensor.zeros([2, 0]);
  G := TTensor.zeros([2, 0]);
  try
    O := DeltaRuleTokens(Q, K, V, Beta, G, State);
    Fail('no EDeltaRule');
  except
    on E: EDeltaRule do
      AssertEquals('no tokens (T = 0)', E.Message);
  end;
end;

procedure TDeltaRuleTest.TestUT_DeltaRule_113;
var
  Q, K, V, Beta, G, O, State: TTensor;
begin
  Q := TTensor.zeros([2, 3, 4]);
  K := TTensor.zeros([2, 3, 3]);
  V := TTensor.zeros([2, 3, 5]);
  Beta := TTensor.zeros([2, 3]);
  G := TTensor.zeros([2, 3]);
  try
    O := DeltaRuleTokens(Q, K, V, Beta, G, State);
    Fail('no EDeltaRule');
  except
    on E: EDeltaRule do
      AssertEquals('K must be the shape of Q [2, 3, 4], got [2, 3, 3]', E.Message);
  end;
end;

procedure TDeltaRuleTest.TestUT_DeltaRule_114;
var
  Q, K, V, Beta, G, O, State: TTensor;
begin
  Q := TTensor.zeros([2, 3, 4]);
  K := TTensor.zeros([2, 3, 4]);
  V := TTensor.zeros([1, 3, 5]);
  Beta := TTensor.zeros([2, 3]);
  G := TTensor.zeros([2, 3]);
  try
    O := DeltaRuleTokens(Q, K, V, Beta, G, State);
    Fail('no EDeltaRule');
  except
    on E: EDeltaRule do
      AssertEquals('V must be [2, 3, Dv], got [1, 3, 5]', E.Message);
  end;
end;

procedure TDeltaRuleTest.TestUT_DeltaRule_115;
var
  Q, K, V, Beta, G, O, State: TTensor;
begin
  Q := TTensor.zeros([2, 3, 4]);
  K := TTensor.zeros([2, 3, 4]);
  V := TTensor.zeros([2, 2, 5]);
  Beta := TTensor.zeros([2, 3]);
  G := TTensor.zeros([2, 3]);
  try
    O := DeltaRuleTokens(Q, K, V, Beta, G, State);
    Fail('no EDeltaRule');
  except
    on E: EDeltaRule do
      AssertEquals('V must be [2, 3, Dv], got [2, 2, 5]', E.Message);
  end;
end;

procedure TDeltaRuleTest.TestUT_DeltaRule_116;
var
  Q, K, V, Beta, G, O, State: TTensor;
begin
  Q := TTensor.zeros([2, 3, 4]);
  K := TTensor.zeros([2, 3, 4]);
  V := TTensor.zeros([2, 3, 5]);
  Beta := TTensor.zeros([2, 4]);
  G := TTensor.zeros([2, 3]);
  try
    O := DeltaRuleTokens(Q, K, V, Beta, G, State);
    Fail('no EDeltaRule');
  except
    on E: EDeltaRule do
      AssertEquals('Beta must be [2, 3], got [2, 4]', E.Message);
  end;
end;

procedure TDeltaRuleTest.TestUT_DeltaRule_117;
var
  Q, K, V, Beta, G, O, State: TTensor;
begin
  Q := TTensor.zeros([2, 3, 4]);
  K := TTensor.zeros([2, 3, 4]);
  V := TTensor.zeros([2, 3, 5]);
  Beta := TTensor.zeros([2, 3]);
  G := TTensor.zeros([2]);
  try
    O := DeltaRuleTokens(Q, K, V, Beta, G, State);
    Fail('no EDeltaRule');
  except
    on E: EDeltaRule do
      AssertEquals('G must be [2, 3], got [2]', E.Message);
  end;
end;

{ Stage 2: DeltaRuleTokens kind check tests (118-120, added for coverage) }

procedure TDeltaRuleTest.TestUT_DeltaRule_118;
var
  Q, K, V, Beta, G, O, State: TTensor;
begin
  Q := TTensor.zeros([2, 3, 4]);
  K := TTensor.zeros([2, 3, 4], kDouble);
  V := TTensor.zeros([2, 3, 5]);
  Beta := TTensor.zeros([2, 3]);
  G := TTensor.zeros([2, 3]);
  try
    O := DeltaRuleTokens(Q, K, V, Beta, G, State);
    Fail('no EDeltaRule');
  except
    on E: EDeltaRule do
      AssertEquals('K must be kFloat (fp32)', E.Message);
  end;
end;

procedure TDeltaRuleTest.TestUT_DeltaRule_119;
var
  Q, K, V, Beta, G, O, State: TTensor;
begin
  Q := TTensor.zeros([2, 3, 4]);
  K := TTensor.zeros([2, 3, 4]);
  V := TTensor.zeros([2, 3, 5], kDouble);
  Beta := TTensor.zeros([2, 3]);
  G := TTensor.zeros([2, 3]);
  try
    O := DeltaRuleTokens(Q, K, V, Beta, G, State);
    Fail('no EDeltaRule');
  except
    on E: EDeltaRule do
      AssertEquals('V must be kFloat (fp32)', E.Message);
  end;
end;

procedure TDeltaRuleTest.TestUT_DeltaRule_120;
var
  Q, K, V, Beta, G, O, State: TTensor;
begin
  Q := TTensor.zeros([2, 3, 4]);
  K := TTensor.zeros([2, 3, 4]);
  V := TTensor.zeros([2, 3, 5]);
  Beta := TTensor.zeros([2, 3], kDouble);
  G := TTensor.zeros([2, 3]);
  try
    O := DeltaRuleTokens(Q, K, V, Beta, G, State);
    Fail('no EDeltaRule');
  except
    on E: EDeltaRule do
      AssertEquals('Beta must be kFloat (fp32)', E.Message);
  end;
end;

{ T09b1 stage 1: DeltaRuleChunked check tests (201-212) }

procedure TDeltaRuleTest.TestUT_DeltaRule_201;
var
  Q, K, V, Beta, G, O, State: TTensor;
begin
  Q := TTensor.zeros([2, 3, 4], kDouble);
  K := TTensor.zeros([2, 3, 4]);
  V := TTensor.zeros([2, 3, 5]);
  Beta := TTensor.zeros([2, 3]);
  G := TTensor.zeros([2, 3]);
  try
    O := DeltaRuleChunked(Q, K, V, Beta, G, State);
    Fail('no EDeltaRule');
  except
    on E: EDeltaRule do
      AssertEquals('Q must be kFloat (fp32)', E.Message);
  end;
end;

procedure TDeltaRuleTest.TestUT_DeltaRule_202;
var
  Q, K, V, Beta, G, O, State: TTensor;
begin
  Q := TTensor.zeros([2, 3, 4]);
  K := TTensor.zeros([2, 3, 4], kDouble);
  V := TTensor.zeros([2, 3, 5]);
  Beta := TTensor.zeros([2, 3]);
  G := TTensor.zeros([2, 3]);
  try
    O := DeltaRuleChunked(Q, K, V, Beta, G, State);
    Fail('no EDeltaRule');
  except
    on E: EDeltaRule do
      AssertEquals('K must be kFloat (fp32)', E.Message);
  end;
end;

procedure TDeltaRuleTest.TestUT_DeltaRule_203;
var
  Q, K, V, Beta, G, O, State: TTensor;
begin
  Q := TTensor.zeros([2, 3, 4]);
  K := TTensor.zeros([2, 3, 4]);
  V := TTensor.zeros([2, 3, 5], kDouble);
  Beta := TTensor.zeros([2, 3]);
  G := TTensor.zeros([2, 3]);
  try
    O := DeltaRuleChunked(Q, K, V, Beta, G, State);
    Fail('no EDeltaRule');
  except
    on E: EDeltaRule do
      AssertEquals('V must be kFloat (fp32)', E.Message);
  end;
end;

procedure TDeltaRuleTest.TestUT_DeltaRule_204;
var
  Q, K, V, Beta, G, O, State: TTensor;
begin
  Q := TTensor.zeros([2, 3, 4]);
  K := TTensor.zeros([2, 3, 4]);
  V := TTensor.zeros([2, 3, 5]);
  Beta := TTensor.zeros([2, 3], kDouble);
  G := TTensor.zeros([2, 3]);
  try
    O := DeltaRuleChunked(Q, K, V, Beta, G, State);
    Fail('no EDeltaRule');
  except
    on E: EDeltaRule do
      AssertEquals('Beta must be kFloat (fp32)', E.Message);
  end;
end;

procedure TDeltaRuleTest.TestUT_DeltaRule_205;
var
  Q, K, V, Beta, G, O, State: TTensor;
begin
  Q := TTensor.zeros([2, 3, 4]);
  K := TTensor.zeros([2, 3, 4]);
  V := TTensor.zeros([2, 3, 5]);
  Beta := TTensor.zeros([2, 3]);
  G := TTensor.zeros([2, 3], kBFloat16);
  try
    O := DeltaRuleChunked(Q, K, V, Beta, G, State);
    Fail('no EDeltaRule');
  except
    on E: EDeltaRule do
      AssertEquals('G must be kFloat (fp32)', E.Message);
  end;
end;

procedure TDeltaRuleTest.TestUT_DeltaRule_206;
var
  Q, K, V, Beta, G, O, State: TTensor;
begin
  Q := TTensor.zeros([2, 3]);
  K := TTensor.zeros([2, 3, 4]);
  V := TTensor.zeros([2, 3, 5]);
  Beta := TTensor.zeros([2, 3]);
  G := TTensor.zeros([2, 3]);
  try
    O := DeltaRuleChunked(Q, K, V, Beta, G, State);
    Fail('no EDeltaRule');
  except
    on E: EDeltaRule do
      AssertEquals('Q must be [H, T, Dk], got [2, 3]', E.Message);
  end;
end;

procedure TDeltaRuleTest.TestUT_DeltaRule_207;
var
  Q, K, V, Beta, G, O, State: TTensor;
begin
  Q := TTensor.zeros([2, 0, 4]);
  K := TTensor.zeros([2, 0, 4]);
  V := TTensor.zeros([2, 0, 5]);
  Beta := TTensor.zeros([2, 0]);
  G := TTensor.zeros([2, 0]);
  try
    O := DeltaRuleChunked(Q, K, V, Beta, G, State);
    Fail('no EDeltaRule');
  except
    on E: EDeltaRule do
      AssertEquals('no tokens (T = 0)', E.Message);
  end;
end;

procedure TDeltaRuleTest.TestUT_DeltaRule_208;
var
  Q, K, V, Beta, G, O, State: TTensor;
begin
  Q := TTensor.zeros([2, 3, 4]);
  K := TTensor.zeros([2, 3, 3]);
  V := TTensor.zeros([2, 3, 5]);
  Beta := TTensor.zeros([2, 3]);
  G := TTensor.zeros([2, 3]);
  try
    O := DeltaRuleChunked(Q, K, V, Beta, G, State);
    Fail('no EDeltaRule');
  except
    on E: EDeltaRule do
      AssertEquals('K must be the shape of Q [2, 3, 4], got [2, 3, 3]', E.Message);
  end;
end;

procedure TDeltaRuleTest.TestUT_DeltaRule_209;
var
  Q, K, V, Beta, G, O, State: TTensor;
begin
  Q := TTensor.zeros([2, 3, 4]);
  K := TTensor.zeros([2, 3, 4]);
  V := TTensor.zeros([2, 2, 5]);
  Beta := TTensor.zeros([2, 3]);
  G := TTensor.zeros([2, 3]);
  try
    O := DeltaRuleChunked(Q, K, V, Beta, G, State);
    Fail('no EDeltaRule');
  except
    on E: EDeltaRule do
      AssertEquals('V must be [2, 3, Dv], got [2, 2, 5]', E.Message);
  end;
end;

procedure TDeltaRuleTest.TestUT_DeltaRule_210;
var
  Q, K, V, Beta, G, O, State: TTensor;
begin
  Q := TTensor.zeros([2, 3, 4]);
  K := TTensor.zeros([2, 3, 4]);
  V := TTensor.zeros([2, 3, 5]);
  Beta := TTensor.zeros([2, 4]);
  G := TTensor.zeros([2, 3]);
  try
    O := DeltaRuleChunked(Q, K, V, Beta, G, State);
    Fail('no EDeltaRule');
  except
    on E: EDeltaRule do
      AssertEquals('Beta must be [2, 3], got [2, 4]', E.Message);
  end;
end;

procedure TDeltaRuleTest.TestUT_DeltaRule_211;
var
  Q, K, V, Beta, G, O, State: TTensor;
begin
  Q := TTensor.zeros([2, 3, 4]);
  K := TTensor.zeros([2, 3, 4]);
  V := TTensor.zeros([2, 3, 5]);
  Beta := TTensor.zeros([2, 3]);
  G := TTensor.zeros([2]);
  try
    O := DeltaRuleChunked(Q, K, V, Beta, G, State);
    Fail('no EDeltaRule');
  except
    on E: EDeltaRule do
      AssertEquals('G must be [2, 3], got [2]', E.Message);
  end;
end;

procedure TDeltaRuleTest.TestUT_DeltaRule_212;
var
  Q, K, V, Beta, G, O, State: TTensor;
begin
  Q := TTensor.zeros([2, 3, 4]);
  K := TTensor.zeros([2, 3, 4]);
  V := TTensor.zeros([2, 3, 5]);
  Beta := TTensor.zeros([2, 3]);
  G := TTensor.zeros([2, 3]);
  try
    O := DeltaRuleChunked(Q, K, V, Beta, G, State, 0);
    Fail('no EDeltaRule');
  except
    on E: EDeltaRule do
      AssertEquals('chunk size must be at least 1, got 0', E.Message);
  end;
end;

{ T09b2 stage 2: DeltaRuleChunked value tests (213-224) }

procedure TDeltaRuleTest.TestUT_DeltaRule_213;
var
  Q, K, V, Beta, G, O, State: TTensor;
begin
  Q := Make([1, 1, 1, 2], [1, 2, 2]);
  K := Make([1, 0, 1, 1], [1, 2, 2]);
  V := Make([2, 4, 3, 3], [1, 2, 2]);
  Beta := Make([0.5, 0.5], [1, 2]);
  G := Make([0, 0], [1, 2]);
  O := DeltaRuleChunked(Q, K, V, Beta, G, State);
  CheckTensor('O', '[1, 2, 2]', '1 2 4 3.5', O);
  CheckTensor('State', '[1, 2, 2]', '2 1 2.5 0.5', State);
end;

procedure TDeltaRuleTest.TestUT_DeltaRule_214;
var
  Q, K, V, Beta, G, O, State: TTensor;
begin
  Q := Make([1, 1, 1, 2], [1, 2, 2]);
  K := Make([1, 0, 1, 1], [1, 2, 2]);
  V := Make([2, 4, 3, 3], [1, 2, 2]);
  Beta := Make([0.5, 0.5], [1, 2]);
  G := Make([0, Ln(0.5)], [1, 2]);
  O := DeltaRuleChunked(Q, K, V, Beta, G, State);
  CheckTensor('O', '[1, 2, 2]', '1 2 4.25 4', O);
  CheckTensor('State', '[1, 2, 2]', '1.75 1.25 2 1', State);
end;

procedure TDeltaRuleTest.TestUT_DeltaRule_215;
var
  Q, K, V, Beta, G, O, State: TTensor;
begin
  RandomInputs(7, 3, 5, 4, 6, 1.0, Q, K, V, Beta, G);
  Beta := TTensor.zeros([3, 5]);
  O := DeltaRuleChunked(Q, K, V, Beta, G, State, 2);
  AssertEquals('O shape', '[3, 5, 6]', ShapeText(O));
  AssertEquals('State shape', '[3, 6, 4]', ShapeText(State));
  AssertTrue('O.abs.max is 0', O.abs.max.DoubleValue([]) <= 0);
  AssertTrue('State.abs.max is 0', State.abs.max.DoubleValue([]) <= 0);
end;

procedure TDeltaRuleTest.TestUT_DeltaRule_216;
var
  Q, K, V, Beta, G, O, State, TokO, TokS: TTensor;
begin
  RandomInputs(20, 2, 5, 4, 6, 1.0, Q, K, V, Beta, G);
  O := DeltaRuleChunked(Q, K, V, Beta, G, State, 2);
  TokO := DeltaRuleTokens(Q, K, V, Beta, G, TokS);
  AssertEquals('O shape', '[2, 5, 6]', ShapeText(O));
  AssertEquals('State shape', '[2, 6, 4]', ShapeText(State));
  AssertTrue('O diff', MaxAbsDiff(O, TokO) <= 1e-6);
  AssertTrue('State diff', MaxAbsDiff(State, TokS) <= 1e-6);
end;

procedure TDeltaRuleTest.TestUT_DeltaRule_217;
var
  Q, K, V, Beta, G, O, State, TokO, TokS: TTensor;
begin
  RandomInputs(16, 4, 50, 16, 16, 1.0, Q, K, V, Beta, G);
  O := DeltaRuleChunked(Q, K, V, Beta, G, State, 1);
  TokO := DeltaRuleTokens(Q, K, V, Beta, G, TokS);
  AssertTrue('O diff', MaxAbsDiff(O, TokO) <= 1e-6);
  AssertTrue('State diff', MaxAbsDiff(State, TokS) <= 1e-6);
end;

procedure TDeltaRuleTest.TestUT_DeltaRule_218;
var
  Q, K, V, Beta, G, O, State, TokO, TokS: TTensor;
begin
  RandomInputs(17, 4, 50, 16, 16, 1.0, Q, K, V, Beta, G);
  O := DeltaRuleChunked(Q, K, V, Beta, G, State, 7);
  TokO := DeltaRuleTokens(Q, K, V, Beta, G, TokS);
  AssertTrue('O diff', MaxAbsDiff(O, TokO) <= 1e-6);
  AssertTrue('State diff', MaxAbsDiff(State, TokS) <= 1e-6);
end;

procedure TDeltaRuleTest.TestUT_DeltaRule_219;
var
  Q, K, V, Beta, G, O, State, TokO, TokS: TTensor;
begin
  RandomInputs(101, 48, 1, 128, 128, 0.1, Q, K, V, Beta, G);
  O := DeltaRuleChunked(Q, K, V, Beta, G, State);
  TokO := DeltaRuleTokens(Q, K, V, Beta, G, TokS);
  AssertTrue('O diff', MaxAbsDiff(O, TokO) <= 1e-5);
  AssertTrue('State diff', MaxAbsDiff(State, TokS) <= 1e-5);
end;

procedure TDeltaRuleTest.TestUT_DeltaRule_220;
var
  Q, K, V, Beta, G, O, State, TokO, TokS: TTensor;
begin
  RandomInputs(164, 48, 64, 128, 128, 0.1, Q, K, V, Beta, G);
  O := DeltaRuleChunked(Q, K, V, Beta, G, State);
  TokO := DeltaRuleTokens(Q, K, V, Beta, G, TokS);
  AssertTrue('O diff', MaxAbsDiff(O, TokO) <= 1e-5);
  AssertTrue('State diff', MaxAbsDiff(State, TokS) <= 1e-5);
end;

procedure TDeltaRuleTest.TestUT_DeltaRule_221;
var
  Q, K, V, Beta, G, O, State, TokO, TokS: TTensor;
begin
  RandomInputs(165, 48, 65, 128, 128, 0.1, Q, K, V, Beta, G);
  O := DeltaRuleChunked(Q, K, V, Beta, G, State);
  TokO := DeltaRuleTokens(Q, K, V, Beta, G, TokS);
  AssertTrue('O diff', MaxAbsDiff(O, TokO) <= 1e-5);
  AssertTrue('State diff', MaxAbsDiff(State, TokS) <= 1e-5);
end;

procedure TDeltaRuleTest.TestUT_DeltaRule_222;
var
  Q, K, V, Beta, G, O, State, TokO, TokS: TTensor;
begin
  RandomInputs(14, 16, 200, 128, 128, 5.0, Q, K, V, Beta, G);
  O := DeltaRuleChunked(Q, K, V, Beta, G, State);
  TokO := DeltaRuleTokens(Q, K, V, Beta, G, TokS);
  AssertTrue('O diff', MaxAbsDiff(O, TokO) <= 1e-5);
  AssertTrue('State diff', MaxAbsDiff(State, TokS) <= 1e-5);
end;

procedure TDeltaRuleTest.TestUT_DeltaRule_223;
var
  Q, K, V, Beta, G, O1, O2, State1, State2: TTensor;
begin
  RandomInputs(21, 3, 70, 8, 8, 1.0, Q, K, V, Beta, G);
  O1 := DeltaRuleChunked(Q, K, V, Beta, G, State1);
  O2 := DeltaRuleChunked(Q, K, V, Beta, G, State2, 64);
  AssertTrue('O equal', MaxAbsDiff(O1, O2) <= 0);
  AssertTrue('State equal', MaxAbsDiff(State1, State2) <= 0);
end;

procedure TDeltaRuleTest.TestUT_DeltaRule_224;
var
  Q, K, V, Beta, G, W: TTensor;
  Q1, K1, V1, Beta1, G1: TTensor;
  Q2, K2, V2, Beta2, G2: TTensor;
  State1, State2, L1, L2: TTensor;
begin
  RandomInputs(22, 4, 70, 16, 16, 1.0, Q, K, V, Beta, G);
  ManualSeed(23);
  W := TTensor.randn([4, 70, 16]);
  Q1 := Q.detach.Copy.requires_grad_(True);
  K1 := K.detach.Copy.requires_grad_(True);
  V1 := V.detach.Copy.requires_grad_(True);
  Beta1 := Beta.detach.Copy.requires_grad_(True);
  G1 := G.detach.Copy.requires_grad_(True);
  Q2 := Q.detach.Copy.requires_grad_(True);
  K2 := K.detach.Copy.requires_grad_(True);
  V2 := V.detach.Copy.requires_grad_(True);
  Beta2 := Beta.detach.Copy.requires_grad_(True);
  G2 := G.detach.Copy.requires_grad_(True);
  L1 := (DeltaRuleTokens(Q1, K1, V1, Beta1, G1, State1) * W).sum(kNone);
  L1.Backward;
  L2 := (DeltaRuleChunked(Q2, K2, V2, Beta2, G2, State2, 16) * W).sum(kNone);
  L2.Backward;
  AssertTrue('Q grad diff', MaxAbsDiff(Q1.grad, Q2.grad) <= 1e-5);
  AssertTrue('K grad diff', MaxAbsDiff(K1.grad, K2.grad) <= 1e-5);
  AssertTrue('V grad diff', MaxAbsDiff(V1.grad, V2.grad) <= 1e-5);
  AssertTrue('Beta grad diff', MaxAbsDiff(Beta1.grad, Beta2.grad) <= 1e-5);
  AssertTrue('G grad diff', MaxAbsDiff(G1.grad, G2.grad) <= 1e-5);
end;

var
  App: TTestRunner;
begin
  RegisterTest(TDeltaRuleTest);
  App := TTestRunner.Create(nil);
  App.Initialize;
  App.Title := 'DeltaRule UT';
  App.Run;
  App.Free;
end.
