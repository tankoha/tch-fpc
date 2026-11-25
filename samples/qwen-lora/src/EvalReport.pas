{ EvalReport — the pure parts of the evaluation (T20), per DESIGN.md 8.17
  and tasks/T20-facts.md (authoritative): one judged reply per task,
  seed and thinking mode; per dialect the compile and run-pass counts, the
  verified aggregate, the paired flips with an exact two-sided sign test,
  the regression checks and the adoption rule.
  Interface by the harness owner (tasks/T20-facts.md); the bodies are
  written in T20b1. }
unit EvalReport;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Math;

const
  K_VERIFIED = 'objfpc,fpc-delphi,unleashed,oxygene';   { the aggregate's dialects }
  K_APPROX = 'delphi';                                  { reported apart }
  K_REGRESSION = 'regression';                          { the dialect of a regression task }

type
  { One judged reply. Compiled and Passed for a Pascal task (Passed: the
    program compiled, ran and printed the expected output); Passed alone
    for a regression task (its check). }
  TJudged = record
    Task, Dialect, Check: string;   { Check: '', 'tool-call' or 'contains' }
    Seed: Int64;
    Think: Boolean;
    Compiled, Passed: Boolean;
  end;
  TJudgedArray = array of TJudged;

  { Counts of one column (a dialect, the aggregate, or a regression check)
    of one model and thinking mode. }
  TCount = record
    N, Compiled, Passed: Integer;
  end;

  { The flips of one column between two models: B fail → pass, C pass →
    fail, of N paired replies. }
  TFlips = record
    N, B, C: Integer;
  end;

{ The counts of AJudged for thinking mode AThink and column AColumn: a
  dialect name, 'verified' (the dialects of K_VERIFIED together), or a
  regression check name ('tool-call', 'contains'). }
function CountOf(const AJudged: TJudgedArray; AThink: Boolean; const AColumn: string): TCount;

{ The flips of AColumn between ABefore and AAfter for AThink: replies
  paired by task and seed; a reply without a partner is not counted. }
function FlipsOf(const ABefore, AAfter: TJudgedArray; AThink: Boolean; const AColumn: string): TFlips;

{ The exact two-sided sign test of B against C:
  min(1, 2 · sum over k = 0..min(B, C) of C(B + C, k) / 2^(B + C)); 1 when B + C = 0. }
function SignTestP(AB, AC: Integer): Double;

{ The rate of ACount (Passed / N); NaN when N = 0. }
function PassRate(const ACount: TCount): Double;

{ The adoption rule of 8.17 for AThink, one line per condition
  ('yes'/'no'/'not measured' at the end), then the verdict line
  'adopt: yes', 'adopt: no' or 'adopt: undecided' (a condition not
  measured and none failed). ASpeedRatio: the after/before generation
  speed (H49, at least 0.7); NaN when not measured. }
function Adoption(const ABefore, AAfter: TJudgedArray; AThink: Boolean;
  ASpeedRatio: Double): TStringArray;

{ The dialects of AJudged (not the regression ones), ascending. }
function DialectsOf(const AJudged: TJudgedArray): TStringArray;

{ The first closed fenced block of AText: a line starting with three
  backquotes opens it, a line equal to three backquotes (trailing spaces
  ignored) closes it; ABlock gets the lines between, each ended by #10.
  False (ABlock '') when there is none. }
function FirstBlock(const AText: string; out ABlock: string): Boolean;

implementation

function InList(const AValue, AList: string): Boolean;
begin
  Result := Pos(',' + AValue + ',', ',' + AList + ',') > 0;
end;

function InColumn(const J: TJudged; const AColumn: string): Boolean;
begin
  if AColumn = 'verified' then
    Result := InList(J.Dialect, K_VERIFIED)
  else if J.Dialect = K_REGRESSION then
    Result := J.Check = AColumn
  else
    Result := J.Dialect = AColumn;
end;

function CountOf(const AJudged: TJudgedArray; AThink: Boolean; const AColumn: string): TCount;
var
  I: Integer;
begin
  Result := Default(TCount);
  for I := 0 to High(AJudged) do
    if (AJudged[I].Think = AThink) and InColumn(AJudged[I], AColumn) then
    begin
      Inc(Result.N);
      if AJudged[I].Compiled then
        Inc(Result.Compiled);
      if AJudged[I].Passed then
        Inc(Result.Passed);
    end;
end;

function FlipsOf(const ABefore, AAfter: TJudgedArray; AThink: Boolean; const AColumn: string): TFlips;
var
  I, J: Integer;
begin
  Result := Default(TFlips);
  for I := 0 to High(ABefore) do
    if (ABefore[I].Think = AThink) and InColumn(ABefore[I], AColumn) then
      for J := 0 to High(AAfter) do
        if (AAfter[J].Think = AThink) and (AAfter[J].Task = ABefore[I].Task) and
           (AAfter[J].Seed = ABefore[I].Seed) then
        begin
          Inc(Result.N);
          if not ABefore[I].Passed and AAfter[J].Passed then
            Inc(Result.B);
          if ABefore[I].Passed and not AAfter[J].Passed then
            Inc(Result.C);
          Break;
        end;
end;

function SignTestP(AB, AC: Integer): Double;
var
  N, K: Integer;
  LogC, S: Double;
begin
  N := AB + AC;
  if N = 0 then
    Exit(1);
  S := 0;
  LogC := 0;   { ln C(N, 0) }
  for K := 0 to Min(AB, AC) do
  begin
    if K > 0 then
      LogC := LogC + Ln(N - K + 1) - Ln(K);
    S := S + Exp(LogC - N * Ln(2));
  end;
  Result := Min(1, 2 * S);
end;

function PassRate(const ACount: TCount): Double;
begin
  if ACount.N = 0 then
    Result := NaN
  else
    Result := ACount.Passed / ACount.N;
end;

function DialectsOf(const AJudged: TJudgedArray): TStringArray;
var
  L: TStringList;
  I: Integer;
begin
  L := TStringList.Create;
  try
    L.Sorted := True;
    L.Duplicates := dupIgnore;
    for I := 0 to High(AJudged) do
      if AJudged[I].Dialect <> K_REGRESSION then
        L.Add(AJudged[I].Dialect);
    Result := L.ToStringArray;
  finally
    L.Free;
  end;
end;

{ A > B, False when either is NaN (an empty series). }
function Above(A, B: Double): Boolean;
begin
  Result := not IsNan(A) and not IsNan(B) and (A > B);
end;

function Adoption(const ABefore, AAfter: TJudgedArray; AThink: Boolean;
  ASpeedRatio: Double): TStringArray;
var
  L: TStringList;
  Rb, Ra: Double;
  F: TFlips;
  P: Double;
  D: TStringArray;
  I: Integer;
  Fall: string;
  Failed, Unmeasured: Boolean;

  procedure Cond(const AText: string; AOk: Boolean);
  begin
    L.Add(AText + ': ' + BoolToStr(AOk, 'yes', 'no'));
    if not AOk then
      Failed := True;
  end;

begin
  L := TStringList.Create;
  try
    Failed := False;
    Unmeasured := False;
    Rb := PassRate(CountOf(ABefore, AThink, 'verified'));
    Ra := PassRate(CountOf(AAfter, AThink, 'verified'));
    Cond(Format('verified aggregate rises (%.3f -> %.3f)', [Rb, Ra]), Above(Ra, Rb));
    F := FlipsOf(ABefore, AAfter, AThink, 'verified');
    P := SignTestP(F.B, F.C);
    Cond(Format('flips significant (%d fail->pass, %d pass->fail, p = %.4f)', [F.B, F.C, P]),
      (F.B > F.C) and (P < 0.05));
    Fall := '';
    D := DialectsOf(ABefore);
    for I := 0 to High(D) do
      if InList(D[I], K_VERIFIED) and
         Above(PassRate(CountOf(ABefore, AThink, D[I])) - 0.05, PassRate(CountOf(AAfter, AThink, D[I]))) then
        Fall := Fall + ' ' + D[I];
    Cond('no verified dialect falls by more than 5 points' + Fall, Fall = '');
    Cond(Format('tool-call check loses nothing (%d -> %d)', [CountOf(ABefore, AThink, 'tool-call').Passed,
      CountOf(AAfter, AThink, 'tool-call').Passed]),
      CountOf(AAfter, AThink, 'tool-call').Passed >= CountOf(ABefore, AThink, 'tool-call').Passed);
    if IsNan(ASpeedRatio) then
    begin
      L.Add('H49 speed at least 0.7 of before: not measured');
      Unmeasured := True;
    end
    else
      Cond(Format('H49 speed at least 0.7 of before (%.3f)', [ASpeedRatio]), ASpeedRatio >= 0.7);
    if Failed then
      L.Add('adopt: no')
    else if Unmeasured then
      L.Add('adopt: undecided')
    else
      L.Add('adopt: yes');
    Result := L.ToStringArray;
  finally
    L.Free;
  end;
end;

function FirstBlock(const AText: string; out ABlock: string): Boolean;
var
  L: TStringList;
  I: Integer;
  Inside: Boolean;
begin
  Result := False;
  ABlock := '';
  Inside := False;
  L := TStringList.Create;
  try
    L.Text := AText;
    for I := 0 to L.Count - 1 do
      if not Inside then
        Inside := Copy(L[I], 1, 3) = '```'
      else if TrimRight(L[I]) = '```' then
        Exit(True)
      else
        ABlock := ABlock + L[I] + #10;
    ABlock := '';
  finally
    L.Free;
  end;
end;

end.
