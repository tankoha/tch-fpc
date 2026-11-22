{ Test fixture of T08 (written by the harness owner; the agents use it, they
  do not change it): the random inputs of T08-facts "Unit tests" and a scalar
  reference of the gated delta rule in Double, independent of the unit
  DeltaRule, so that the UT compares with something outside the code under
  test. The ITa driver tests/ita/T08/deltarule_check.pas uses it too. }
unit DeltaRuleFixture;

{$mode objfpc}{$H+}

interface

uses
  SysUtils, Math, Tch;

type
  TSingleValues = array of Single;

{ Random inputs of the facts: ManualSeed(ASeed); Q = randn [H,T,Dk] / sqrt(Dk);
  K = randn [H,T,Dk], each row scaled to length 1 (x / sqrt(sum x^2 + 1e-6));
  V = randn [H,T,Dv]; Beta = rand [H,T]; G = rand [H,T] * (-AGScale).
  Drawn in this order; all kFloat on the CPU. }
procedure RandomInputs(ASeed: Int64; H, T, Dk, Dv: Int64; AGScale: Double;
  out Q, K, V, Beta, G: TTensor);

{ The rule token by token in Double, from a zero state:
    M := exp(g) M;  M := M + beta (v - M k) k^T;  o := M q   (per head, M [Dv, Dk]).
  Inputs as DeltaRuleTokens (kFloat [H,T,Dk], [H,T,Dk], [H,T,Dv], [H,T], [H,T]).
  O receives [H,T,Dv], S the final state [H,Dv,Dk]; both kFloat (rounded
  from Double once at the end). }
procedure RefTokens(const Q, K, V, Beta, G: TTensor; out O, S: TTensor);

{ The fp32 values of a tensor, row-major (any shape, kFloat). }
function Values(const T: TTensor): TSingleValues;

{ max |a - b| over all elements (same shape, kFloat). }
function MaxAbsDiff(const A, B: TTensor): Double;

{ The shape as text: '[2, 3, 4]'. }
function ShapeText(const T: TTensor): string;

{ The values as text, each with FloatToStr, separated by single spaces. }
function ValuesText(const T: TTensor): string;

implementation

procedure RandomInputs(ASeed: Int64; H, T, Dk, Dv: Int64; AGScale: Double;
  out Q, K, V, Beta, G: TTensor);
var
  R: TTensor;
begin
  ManualSeed(ASeed);
  Q := TTensor.randn([H, T, Dk]) * (1.0 / Sqrt(Dk));
  R := TTensor.randn([H, T, Dk]);
  K := R / (R * R).sum_dim_intlist(IntList([2]), True, kNone).add_scalar(1e-6).sqrt;
  V := TTensor.randn([H, T, Dv]);
  Beta := TTensor.rand([H, T]);
  G := TTensor.rand([H, T]) * (-AGScale);
end;

function Values(const T: TTensor): TSingleValues;
var
  C: TTensor;
  N: Int64;
begin
  C := T.contiguous;
  N := C.Numel;
  Result := nil;
  SetLength(Result, N);
  if N > 0 then
    C.CopyData(Result[0], N);
end;

procedure RefTokens(const Q, K, V, Beta, G: TTensor; out O, S: TTensor);
var
  QA, KA, VA, BA, GA: TSingleValues;
  OD, SD: array of Double;
  M: array of Double;
  H, T, Dk, Dv, hh, tt, i, j: Int64;
  e, p, d: Double;
begin
  H := Q.Shape[0]; T := Q.Shape[1]; Dk := Q.Shape[2]; Dv := V.Shape[2];
  QA := Values(Q); KA := Values(K); VA := Values(V); BA := Values(Beta); GA := Values(G);
  OD := nil; SD := nil; M := nil;
  SetLength(OD, H * T * Dv);
  SetLength(SD, H * Dv * Dk);
  SetLength(M, Dv * Dk);
  for hh := 0 to H - 1 do
  begin
    for i := 0 to High(M) do M[i] := 0;
    for tt := 0 to T - 1 do
    begin
      e := Exp(Double(GA[hh * T + tt]));
      for i := 0 to High(M) do M[i] := M[i] * e;
      for i := 0 to Dv - 1 do
      begin
        p := 0;
        for j := 0 to Dk - 1 do
          p := p + M[i * Dk + j] * KA[(hh * T + tt) * Dk + j];
        d := BA[hh * T + tt] * (VA[(hh * T + tt) * Dv + i] - p);
        for j := 0 to Dk - 1 do
          M[i * Dk + j] := M[i * Dk + j] + d * KA[(hh * T + tt) * Dk + j];
      end;
      for i := 0 to Dv - 1 do
      begin
        p := 0;
        for j := 0 to Dk - 1 do
          p := p + M[i * Dk + j] * QA[(hh * T + tt) * Dk + j];
        OD[(hh * T + tt) * Dv + i] := p;
      end;
    end;
    for i := 0 to High(M) do
      SD[hh * Dv * Dk + i] := M[i];
  end;
  O := TTensor.FromArray(OD).reshape([H, T, Dv]);
  S := TTensor.FromArray(SD).reshape([H, Dv, Dk]);
end;

function MaxAbsDiff(const A, B: TTensor): Double;
var
  X, Y: TSingleValues;
  I: Integer;
begin
  X := Values(A); Y := Values(B);
  if Length(X) <> Length(Y) then
    raise Exception.CreateFmt('MaxAbsDiff: %d against %d elements', [Length(X), Length(Y)]);
  Result := 0;
  for I := 0 to High(X) do
    Result := Max(Result, Abs(Double(X[I]) - Double(Y[I])));
end;

function ShapeText(const T: TTensor): string;
var
  S: TInt64Array;
  I: Integer;
begin
  S := T.Shape;
  Result := '[';
  for I := 0 to High(S) do
  begin
    if I > 0 then
      Result := Result + ', ';
    Result := Result + IntToStr(S[I]);
  end;
  Result := Result + ']';
end;

function ValuesText(const T: TTensor): string;
var
  A: TSingleValues;
  I: Integer;
begin
  A := Values(T);
  Result := '';
  for I := 0 to High(A) do
  begin
    if I > 0 then
      Result := Result + ' ';
    Result := Result + FloatToStr(A[I]);
  end;
end;

end.
