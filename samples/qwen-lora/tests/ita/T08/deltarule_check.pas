{ ITa driver of T08 (written by the harness owner; agents run it through
  tests/ita/T08-run.sh, they do not change it). Runs DeltaRuleTokens on the
  CPU at the sizes of the 27B model's linear blocks (48 value heads,
  Dk = Dv = 128) and compares with the scalar Double reference of
  tests/ut/DeltaRuleFixture.pas and with hand-computed values.
  Prints one line per case: case ID, measured value, limit, note
  (tab-separated); the run script turns value <= limit into OK. }
program deltarule_check;

{$mode objfpc}{$H+}

uses
  SysUtils, DateUtils, Math, Tch, DeltaRule, DeltaRuleFixture;

procedure Line(const AId: string; AValue, ALimit: Double; const ANote: string);
begin
  WriteLn(AId, #9, FloatToStrF(AValue, ffExponent, 3, 2), #9,
          FloatToStrF(ALimit, ffExponent, 3, 2), #9, ANote);
end;

function Make(const AValues: array of Double; const AShape: array of Int64): TTensor;
begin
  Result := TTensor.FromArray(AValues).reshape(AShape);
end;

procedure HandCase(const AId: string; AG1: Double; const AO, AState: array of Double; const ANote: string);
var
  O, S: TTensor;
begin
  O := DeltaRuleTokens(Make([1, 1, 1, 2], [1, 2, 2]), Make([1, 0, 1, 1], [1, 2, 2]),
                       Make([2, 4, 3, 3], [1, 2, 2]), Make([0.5, 0.5], [1, 2]),
                       Make([0, AG1], [1, 2]), S);
  Line(AId, Max(MaxAbsDiff(O, Make(AO, [1, 2, 2])), MaxAbsDiff(S, Make(AState, [1, 2, 2]))), 0,
       ANote + ', O ' + ValuesText(O) + ', S ' + ValuesText(S));
end;

procedure RefCase(const AId: string; ASeed, H, T: Int64; AGScale: Double; const ANote: string);
var
  Q, K, V, B, G, O, S, RO, RS: TTensor;
  T0: TDateTime;
  Ms: Int64;
begin
  RandomInputs(ASeed, H, T, 128, 128, AGScale, Q, K, V, B, G);
  T0 := Now;
  O := DeltaRuleTokens(Q, K, V, B, G, S);
  Ms := MilliSecondsBetween(Now, T0);
  RefTokens(Q, K, V, B, G, RO, RS);
  Line(AId, Max(MaxAbsDiff(O, RO), MaxAbsDiff(S, RS)), 1e-5,
       Format('%s, seed %d, H %d, T %d, O %s, %d ms', [ANote, ASeed, H, T, ShapeText(O), Ms]));
end;

var
  Q, K, V, B, G, O, S, RO, RS, P: TTensor;
  L: Integer;
  Worst: Double;
begin
  HandCase('ITa-T08-001', 0, [1, 2, 4, 3.5], [2, 1, 2.5, 0.5], 'hand case, two tokens');
  HandCase('ITa-T08-002', Ln(0.5), [1, 2, 4.25, 4], [1.75, 1.25, 2, 1], 'hand case, g = ln(1/2) on token 1');

  RandomInputs(31, 48, 200, 128, 128, 1.0, Q, K, V, B, G);
  O := DeltaRuleTokens(Q, K, V, TTensor.zeros([48, 200]), G, S);
  Line('ITa-T08-003', Max(O.abs.max.DoubleValue([]), S.abs.max.DoubleValue([])), 0,
       'beta = 0: max |O| and |State|, H 48, T 200');

  RefCase('ITa-T08-004', 13, 48, 200, 0.1, 'reference, g in [-0.1, 0]');
  RefCase('ITa-T08-005', 14, 16, 200, 5.0, 'reference, g in [-5, 0]');

  Worst := 0;
  for L in [1, 63, 64, 65, 200] do
  begin
    RandomInputs(100 + L, 48, L, 128, 128, 1.0, Q, K, V, B, G);
    O := DeltaRuleTokens(Q, K, V, B, G, S);
    RefTokens(Q, K, V, B, G, RO, RS);
    Worst := Max(Worst, Max(MaxAbsDiff(O, RO), MaxAbsDiff(S, RS)));
  end;
  Line('ITa-T08-006', Worst, 1e-5, 'reference, T = 1, 63, 64, 65, 200 (seeds 100 + T), H 48');

  RandomInputs(41, 48, 20, 128, 128, 1.0, Q, K, V, B, G);
  O := DeltaRuleTokens(Q, K, V, B, TTensor.zeros([48, 20]) + Double(NegInfinity), S);
  P := (V * B.unsqueeze(2)) * (K * Q).sum_dim_intlist(IntList([2]), True, kNone);
  Line('ITa-T08-007', MaxAbsDiff(O, P), 1e-6, 'g = -inf: o_t = beta_t v_t (k_t . q_t), H 48, T 20');
end.
