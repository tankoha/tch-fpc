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

function CountOf(const AJudged: TJudgedArray; AThink: Boolean; const AColumn: string): TCount;
begin
  Result := Default(TCount);
  raise Exception.Create('CountOf: not implemented yet');
end;

function FlipsOf(const ABefore, AAfter: TJudgedArray; AThink: Boolean; const AColumn: string): TFlips;
begin
  Result := Default(TFlips);
  raise Exception.Create('FlipsOf: not implemented yet');
end;

function SignTestP(AB, AC: Integer): Double;
begin
  Result := 0;
  raise Exception.Create('SignTestP: not implemented yet');
end;

function PassRate(const ACount: TCount): Double;
begin
  Result := 0;
  raise Exception.Create('PassRate: not implemented yet');
end;

function Adoption(const ABefore, AAfter: TJudgedArray; AThink: Boolean;
  ASpeedRatio: Double): TStringArray;
begin
  Result := nil;
  raise Exception.Create('Adoption: not implemented yet');
end;

function DialectsOf(const AJudged: TJudgedArray): TStringArray;
begin
  Result := nil;
  raise Exception.Create('DialectsOf: not implemented yet');
end;

function FirstBlock(const AText: string; out ABlock: string): Boolean;
begin
  Result := False;
  raise Exception.Create('FirstBlock: not implemented yet');
end;

end.
