{ UT program for src/EvalReport.pas: the rows UT-EvalReport-NNN of
  tests/ut/cases/EvalReport.cases.md. Pure functions; no files.
  Invoked with `--all --format=plain` (see .sdlc/ut-coverage.sh).

  Skeleton by the harness owner; the test methods are written in T20b1. }
program EvalReport_test;

{$mode objfpc}{$H+}

uses
  SysUtils, Classes, Math, EvalReport, fpcunit, testregistry, consoletestrunner;

type
  TEvalReportTest = class(TTestCase)
  published
  end;

{ ---- helpers (harness owner's; nobody changes them) ---- }

{ One judged reply; ACheck '' for a Pascal task. }
function J(const ATask, ADialect: string; ASeed: Int64; AThink, ACompiled, APassed: Boolean;
  const ACheck: string = ''): TJudged;
begin
  Result := Default(TJudged);
  Result.Task := ATask;
  Result.Dialect := ADialect;
  Result.Check := ACheck;
  Result.Seed := ASeed;
  Result.Think := AThink;
  Result.Compiled := ACompiled;
  Result.Passed := APassed;
end;

function Arr(const A: array of TJudged): TJudgedArray;
var
  I: Integer;
begin
  Result := nil;
  SetLength(Result, Length(A));
  for I := 0 to High(A) do
    Result[I] := A[I];
end;

{ AN Pascal replies of dialect ADialect, tasks <ADialect>-1..N, seed 1,
  thinking on, the first APassed passed (all compiled). }
function Series(const ADialect: string; AN, APassed: Integer): TJudgedArray;
var
  I: Integer;
begin
  Result := nil;
  SetLength(Result, AN);
  for I := 0 to AN - 1 do
    Result[I] := J(ADialect + '-' + IntToStr(I + 1), ADialect, 1, True, True, I < APassed);
end;

function Cat(const A, B: TJudgedArray): TJudgedArray;
var
  I: Integer;
begin
  Result := nil;
  SetLength(Result, Length(A) + Length(B));
  for I := 0 to High(A) do
    Result[I] := A[I];
  for I := 0 to High(B) do
    Result[Length(A) + I] := B[I];
end;

function Joined(const A: TStringArray): string;
begin
  Result := string.Join('|', A);
end;

var
  App: TTestRunner;
begin
  RegisterTest(TEvalReportTest);
  App := TTestRunner.Create(nil);
  App.Initialize;
  App.Title := 'EvalReport UT';
  App.Run;
  App.Free;
end.
