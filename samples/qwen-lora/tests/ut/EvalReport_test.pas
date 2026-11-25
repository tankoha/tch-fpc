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
    procedure TestUT_EvalReport_001;
    procedure TestUT_EvalReport_002;
    procedure TestUT_EvalReport_003;
    procedure TestUT_EvalReport_004;
    procedure TestUT_EvalReport_005;
    procedure TestUT_EvalReport_006;
    procedure TestUT_EvalReport_007;
    procedure TestUT_EvalReport_008;
    procedure TestUT_EvalReport_009;
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

{ ---- UT cases (bodies in TAssert scope) ---- }

procedure TEvalReportTest.TestUT_EvalReport_001;
var
  A: TJudgedArray;
  C: TCount;
begin
  A := Arr([J('a', 'objfpc', 1, True, True, True),
    J('b', 'objfpc', 1, True, True, False),
    J('c', 'objfpc', 1, True, False, False),
    J('d', 'fpc-delphi', 1, True, True, True),
    J('e', 'delphi', 1, True, True, True),
    J('f', 'unleashed', 1, True, False, False),
    J('g', 'oxygene', 1, True, True, True),
    J('a', 'objfpc', 1, False, True, True),
    J('t', 'regression', 1, True, True, True),
    J('u', 'regression', 1, True, True, True, 'tool-call'),
    J('u', 'regression', 1, True, True, False, 'tool-call'),
    J('w', 'regression', 1, True, True, True, 'contains')]);
  C := CountOf(A, True, 'objfpc');
  AssertEquals(3, C.N);
  AssertEquals(2, C.Compiled);
  AssertEquals(1, C.Passed);
  C := CountOf(A, True, 'verified');
  AssertEquals(6, C.N);
  AssertEquals(4, C.Compiled);
  AssertEquals(3, C.Passed);
  C := CountOf(A, True, 'delphi');
  AssertEquals(1, C.N);
  C := CountOf(A, True, 'tool-call');
  AssertEquals(2, C.N);
  AssertEquals(1, C.Passed);
  C := CountOf(A, True, 'contains');
  AssertEquals(1, C.N);
  C := CountOf(A, False, 'verified');
  AssertEquals(1, C.N);
  C := CountOf(A, True, 'pascal');
  AssertEquals(0, C.N);
end;

procedure TEvalReportTest.TestUT_EvalReport_002;
var
  B, A: TJudgedArray;
  F: TFlips;
begin
  B := Arr([J('a', 'objfpc', 1, True, False, False),
    J('a', 'objfpc', 2, True, True, True),
    J('b', 'objfpc', 1, True, True, True),
    J('c', 'objfpc', 1, True, False, False),
    J('d', 'objfpc', 1, True, False, False),
    J('a', 'objfpc', 1, False, False, False),
    J('x', 'delphi', 1, True, True, False)]);
  A := Arr([J('a', 'objfpc', 1, True, True, True),
    J('a', 'objfpc', 2, True, False, False),
    J('b', 'objfpc', 1, True, True, True),
    J('c', 'objfpc', 1, True, True, True),
    J('a', 'objfpc', 1, False, True, True),
    J('x', 'delphi', 1, True, True, True)]);
  F := FlipsOf(B, A, True, 'objfpc');
  AssertEquals(4, F.N);
  AssertEquals(2, F.B);
  AssertEquals(1, F.C);
  F := FlipsOf(B, A, True, 'verified');
  AssertEquals(4, F.N);
  AssertEquals(2, F.B);
  AssertEquals(1, F.C);
  F := FlipsOf(B, A, False, 'objfpc');
  AssertEquals(1, F.N);
  AssertEquals(1, F.B);
  F := FlipsOf(B, A, True, 'delphi');
  AssertEquals(1, F.N);
  AssertEquals(1, F.B);
end;

procedure TEvalReportTest.TestUT_EvalReport_003;
begin
  AssertEquals(1.0, SignTestP(0, 0));
  AssertEquals(0.0625, SignTestP(5, 0));
  AssertEquals(0.03125, SignTestP(0, 6));
  AssertEquals(158.0 / 4096.0, SignTestP(10, 2));
  AssertEquals(158.0 / 4096.0, SignTestP(2, 10));
  AssertEquals(1.0, SignTestP(3, 3));
  AssertEquals(2.0 * 41.0 / Power(2.0, 40.0), SignTestP(39, 1));
end;

procedure TEvalReportTest.TestUT_EvalReport_004;
var
  C: TCount;
begin
  C := Default(TCount);
  AssertTrue('PassRate(0/0) is NaN', IsNan(PassRate(C)));
  C.N := 4;
  C.Passed := 3;
  AssertEquals(0.75, PassRate(C));
end;

procedure TEvalReportTest.TestUT_EvalReport_005;
var
  B, A: TJudgedArray;
  L: TStringArray;
begin
  B := Cat(Series('objfpc', 10, 2),
    Arr([J('tc', 'regression', 1, True, False, True, 'tool-call')]));
  A := Cat(Series('objfpc', 10, 9),
    Arr([J('tc', 'regression', 1, True, False, True, 'tool-call')]));
  L := Adoption(B, A, True, 0.8);
  AssertEquals(6, Length(L));
  AssertEquals('verified aggregate rises (0.200 -> 0.900): yes', L[0]);
  AssertEquals('flips significant (7 fail->pass, 0 pass->fail, p = 0.0156): yes', L[1]);
  AssertEquals('no verified dialect falls by more than 5 points: yes', L[2]);
  AssertEquals('tool-call check loses nothing (1 -> 1): yes', L[3]);
  AssertEquals('H49 speed at least 0.7 of before (0.800): yes', L[4]);
  AssertEquals('adopt: yes', L[5]);
end;

procedure TEvalReportTest.TestUT_EvalReport_006;
var
  B, A: TJudgedArray;
  L: TStringArray;
begin
  B := Cat(Cat(Series('objfpc', 10, 2), Series('fpc-delphi', 10, 10)),
    Arr([J('tc', 'regression', 1, True, False, True, 'tool-call')]));
  A := Cat(Cat(Series('objfpc', 10, 10), Series('fpc-delphi', 10, 9)),
    Arr([J('tc', 'regression', 1, True, False, False, 'tool-call')]));
  L := Adoption(B, A, True, 0.65);
  AssertEquals('verified aggregate rises (0.600 -> 0.950): yes', L[0]);
  AssertEquals('flips significant (8 fail->pass, 1 pass->fail, p = 0.0391): yes', L[1]);
  AssertEquals('no verified dialect falls by more than 5 points fpc-delphi: no', L[2]);
  AssertEquals('tool-call check loses nothing (1 -> 0): no', L[3]);
  AssertEquals('H49 speed at least 0.7 of before (0.650): no', L[4]);
  AssertEquals('adopt: no', L[5]);
end;

procedure TEvalReportTest.TestUT_EvalReport_007;
var
  B, A: TJudgedArray;
  L: TStringArray;
begin
  B := Series('objfpc', 10, 2);
  A := Series('objfpc', 10, 9);
  L := Adoption(B, A, True, NaN);
  AssertEquals('flips significant (7 fail->pass, 0 pass->fail, p = 0.0156): yes', L[1]);
  AssertEquals('tool-call check loses nothing (0 -> 0): yes', L[3]);
  AssertEquals('H49 speed at least 0.7 of before: not measured', L[4]);
  AssertEquals('adopt: undecided', L[5]);
  L := Adoption(B, A, False, 1.0);
  AssertEquals('adopt: no', L[5]);
end;

procedure TEvalReportTest.TestUT_EvalReport_008;
begin
  AssertEquals('delphi|objfpc|unleashed',
    Joined(DialectsOf(Arr([J('u', 'unleashed', 1, True, True, True),
      J('o', 'objfpc', 1, True, True, True),
      J('r', 'regression', 1, True, False, True, 'tool-call'),
      J('d', 'delphi', 1, True, True, True),
      J('o2', 'objfpc', 1, True, True, True)]))));
  AssertEquals(0, Length(DialectsOf(nil)));
end;

procedure TEvalReportTest.TestUT_EvalReport_009;
var
  B: string;
begin
  AssertTrue(FirstBlock(
    'text' + #10 + '```' + #10 + 'a' + #10 + '  b' + #10 + '```  ' + #10 +
    '```' + #10 + 'c', B));
  AssertEquals('a' + #10 + '  b' + #10, B);
  AssertFalse(FirstBlock('```' + #10 + 'c', B));
  AssertEquals('', B);
  AssertFalse(FirstBlock('no block', B));
  AssertEquals('', B);
  AssertTrue(FirstBlock('```' + #10 + '```', B));
  AssertEquals('', B);
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
