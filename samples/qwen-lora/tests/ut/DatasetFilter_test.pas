{ UT program for src/DatasetFilter.pas: the rows UT-DatasetFilter-NNN of
  tests/ut/cases/DatasetFilter.cases.md. Pure functions; no files.
  Invoked with `--all --format=plain` (see .sdlc/ut-coverage.sh).

  Skeleton by the harness owner; the test methods are written in T19b1. }
program DatasetFilter_test;

{$mode objfpc}{$H+}

uses
  SysUtils, Classes, md5, fpjson, DatasetFilter, fpcunit, testregistry, consoletestrunner;

type
  TDatasetFilterTest = class(TTestCase)
  published
    procedure TestUT_DatasetFilter_001;
    procedure TestUT_DatasetFilter_002;
    procedure TestUT_DatasetFilter_003;
    procedure TestUT_DatasetFilter_004;
    procedure TestUT_DatasetFilter_005;
    procedure TestUT_DatasetFilter_006;
    procedure TestUT_DatasetFilter_007;
    procedure TestUT_DatasetFilter_008;
    procedure TestUT_DatasetFilter_009;
    procedure TestUT_DatasetFilter_010;
  end;

{ ---- helpers (harness owner's; nobody changes them) ---- }

{ A valid record line: id, dialect, task_type, prompt, answer as given;
  topic 't', language 'en', expected_stdout '1'#10, features [],
  generator {"model":"m"}. }
function RecLine(const AId, ADialect, ATaskType, APrompt, AAnswer: string): string;
var
  O: TJSONObject;
begin
  O := TJSONObject.Create;
  try
    O.Add('id', AId);
    O.Add('dialect', ADialect);
    O.Add('task_type', ATaskType);
    O.Add('topic', 't');
    O.Add('language', 'en');
    O.Add('prompt', APrompt);
    O.Add('answer', AAnswer);
    O.Add('expected_stdout', '1' + #10);
    O.Add('features', TJSONArray.Create);
    O.Add('generator', TJSONObject.Create(['model', 'm']));
    Result := O.AsJSON;
  finally
    O.Free;
  end;
end;

{ RecLine of a good record with key AKey replaced by the JSON text AJson
  ('' removes the key). }
function RecWith(const AKey, AJson: string): string;
var
  O: TJSONObject;
begin
  O := GetJSON(RecLine('r1', 'objfpc', 'write', 'p', '```pascal' + #10 + 'begin end.' + #10 + '```'), False) as TJSONObject;
  try
    O.Delete(AKey);
    if AJson <> '' then
      O.Add(AKey, GetJSON(AJson, False));
    Result := O.AsJSON;
  finally
    O.Free;
  end;
end;

{ ParseRecord's reason for ALine. }
function ReasonOf(const ALine: string): string;
var
  R: TRawRecord;
begin
  Result := ParseRecord(ALine, R);
end;

function Sh(const A: array of QWord): TShingles;
var
  I: Integer;
begin
  Result := nil;
  SetLength(Result, Length(A));
  for I := 0 to High(A) do
    Result[I] := A[I];
end;

function Joined(const A: TStringArray): string;
begin
  Result := string.Join('|', A);
end;

procedure TDatasetFilterTest.TestUT_DatasetFilter_001;
var
  R: TRawRecord;
begin
  AssertEquals('', ParseRecord(RecLine('a1', 'fpc-delphi', 'convert', 'Do it.',
    'Text' + #10 + '```pascal' + #10 + 'program P;' + #10 + 'begin end.' + #10 +
    '```' + #10), R));
  AssertEquals('a1', R.Id);
  AssertEquals('fpc-delphi', R.Dialect);
  AssertEquals('convert', R.TaskType);
  AssertEquals('Do it.', R.Prompt);
  AssertEquals('1' + #10, R.Expected);
  AssertEquals(1, Length(R.Blocks));
  AssertEquals('program P;' + #10 + 'begin end.' + #10, R.Blocks[0]);
end;

procedure TDatasetFilterTest.TestUT_DatasetFilter_002;
begin
  AssertEquals('schema: not a JSON object', ReasonOf('{"id": '));
  AssertEquals('schema: not a JSON object', ReasonOf('[1, 2]'));
  AssertEquals('schema: id', ReasonOf(RecWith('id', '')));
  AssertEquals('schema: id', ReasonOf(RecWith('id', '""')));
  AssertEquals('schema: id', ReasonOf(RecWith('id', '7')));
  AssertEquals('schema: dialect', ReasonOf(RecWith('dialect', '"turbo"')));
  AssertEquals('schema: dialect', ReasonOf(RecWith('dialect', '"OBJFPC"')));
  AssertEquals('schema: task_type', ReasonOf(RecWith('task_type', '"other"')));
  AssertEquals('schema: topic', ReasonOf(RecWith('topic', '')));
  AssertEquals('schema: language', ReasonOf(RecWith('language', '1')));
  AssertEquals('schema: prompt', ReasonOf(RecWith('prompt', '""')));
  AssertEquals('schema: answer', ReasonOf(RecWith('answer', '')));
  AssertEquals('schema: expected_stdout',
    ReasonOf(RecWith('expected_stdout', 'null')));
  AssertEquals('schema: features', ReasonOf(RecWith('features', '"a"')));
  AssertEquals('schema: generator', ReasonOf(RecWith('generator', '[]')));
  AssertEquals('schema: answer has no fenced block',
    ReasonOf(RecWith('answer', '"no fence"')));
  AssertEquals('', ReasonOf(RecWith('topic', '""')));
end;

procedure TDatasetFilterTest.TestUT_DatasetFilter_003;
var
  B: TStringArray;
begin
  B := ExtractBlocks('intro' + #10 + '```' + #10 + 'a' + #10 + '  b' + #10 +
    '```   ' + #10 + 'between' + #10 + '```' + #10 + 'c' + #10 + '```' + #10 +
    '```' + #10 + 'open');
  AssertEquals(2, Length(B));
  AssertEquals('a' + #10 + '  b' + #10, B[0]);
  AssertEquals('c' + #10, B[1]);
  AssertEquals(0, Length(ExtractBlocks('no blocks' + #10)));
  B := ExtractBlocks('```' + #10 + '```');
  AssertEquals(1, Length(B));
  AssertEquals('', B[0]);
end;

procedure TDatasetFilterTest.TestUT_DatasetFilter_004;
begin
  AssertEquals('program p; begin writeln(''A {b} // C''); end.',
    NormalizeSource('Program P; { note }' + #10 + '(* old *)Begin  // tail' +
    #10#9 + 'WriteLn(''A {b} // C'');' + #13#10 + 'End.'));
  AssertEquals('a b', NormalizeSource('  a{x}b  '));
  AssertEquals('''open', NormalizeSource('''open'));
  AssertEquals('x', NormalizeSource('x (* unclosed'));
  AssertEquals('', NormalizeSource(''));
end;

procedure TDatasetFilterTest.TestUT_DatasetFilter_005;
var
  B1, B2, B3, B4: TStringArray;
begin
  SetLength(B1, 2);
  B1[0] := 'Program P;' + #10 + 'begin end.' + #10;
  B1[1] := 'unit U; end.';
  SetLength(B2, 2);
  B2[0] := 'program   p; { c }' + #10#10 + 'BEGIN END.';
  B2[1] := 'Unit u;  end.';
  AssertEquals(SourceKey(B1), SourceKey(B2));
  AssertEquals(MD5Print(MD5String('program p; begin end.' + #10 + 'unit u; end.')),
    SourceKey(B1));
  SetLength(B2, 2);
  B2[0] := B1[0];
  B2[1] := 'unit v; end.';
  AssertTrue('row 005: the unit-v key must differ', SourceKey(B1) <> SourceKey(B2));
  SetLength(B3, 1);
  B3[0] := 'x := ''a'';';
  SetLength(B4, 1);
  B4[0] := 'x := ''A'';';
  AssertTrue('row 005: a vs A must differ', SourceKey(B3) <> SourceKey(B4));
end;

procedure TDatasetFilterTest.TestUT_DatasetFilter_006;
begin
  AssertEquals('foo_1|:|=|42|+|x|;', Joined(LexTokens('Foo_1 := 42+X;')));
  AssertEquals('a1b|7|_z', Joined(LexTokens(#9 + 'A1b' + #13#10 + ' 7 _Z ')));
  AssertEquals(0, Length(LexTokens('  ')));
end;

procedure TDatasetFilterTest.TestUT_DatasetFilter_007;
var
  T: TStringArray;
  S: TShingles;
begin
  SetLength(T, 4);
  T[0] := 'a'; T[1] := 'b'; T[2] := 'c'; T[3] := 'd';
  AssertEquals(0, Length(Shingles(T)));
  SetLength(T, 6);
  T[0] := 'a'; T[1] := 'b'; T[2] := 'c';
  T[3] := 'd'; T[4] := 'e'; T[5] := 'f';
  S := Shingles(T);
  AssertEquals(2, Length(S));
  AssertTrue('row 007: shingles must be ascending', S[0] < S[1]);
  SetLength(T, 7);
  T[0] := 'x'; T[1] := 'x'; T[2] := 'x'; T[3] := 'x';
  T[4] := 'x'; T[5] := 'x'; T[6] := 'y';
  AssertEquals(2, Length(Shingles(T)));
  SetLength(T, 6);
  T[0] := 'x'; T[1] := 'x'; T[2] := 'x'; T[3] := 'x';
  T[4] := 'x'; T[5] := 'x';
  AssertEquals(1, Length(Shingles(T)));
  SetLength(T, 5);
  T[0] := 'a'; T[1] := 'b'; T[2] := 'c'; T[3] := 'd'; T[4] := 'e';
  S := Shingles(T);
  AssertEquals(1, Length(S));
  AssertEquals(QWord($E22588B27BD8C58C), S[0]);
end;

procedure TDatasetFilterTest.TestUT_DatasetFilter_008;
var
  E: TShingles;
begin
  AssertEquals(1.0, Jaccard(Sh([1, 2, 3]), Sh([1, 2, 3])));
  AssertEquals(0.0, Jaccard(Sh([1, 2]), Sh([3, 4])));
  AssertEquals(0.5, Jaccard(Sh([1, 2, 3]), Sh([2, 3, 4])));
  AssertEquals(0.2, Jaccard(Sh([1, 5]), Sh([1, 2, 3, 4])));
  E := nil;
  AssertEquals(0.0, Jaccard(E, E));
  AssertEquals(0.0, Jaccard(Sh([1]), E));
end;

procedure TDatasetFilterTest.TestUT_DatasetFilter_009;
begin
  AssertEquals(0, ValidCount(0, 5));
  AssertEquals(0, ValidCount(9, 5));
  AssertEquals(1, ValidCount(10, 5));
  AssertEquals(1, ValidCount(29, 5));
  AssertEquals(2, ValidCount(30, 5));
  AssertEquals(0, ValidCount(1, 25));
  AssertEquals(1, ValidCount(2, 25));
  AssertEquals(2, ValidCount(6, 25));
  AssertEquals(5, ValidCount(5, 100));
  AssertEquals(0, ValidCount(5, 0));
end;

procedure TDatasetFilterTest.TestUT_DatasetFilter_010;
begin
  AssertEquals(MD5Print(MD5String('1:o1')), SplitKey(1, 'o1'));
  AssertEquals(MD5Print(MD5String('-3:Id')), SplitKey(-3, 'Id'));
  AssertEquals(32, Length(SplitKey(0, '')));
end;

var
  App: TTestRunner;
begin
  RegisterTest(TDatasetFilterTest);
  App := TTestRunner.Create(nil);
  App.Initialize;
  App.Title := 'DatasetFilter UT';
  App.Run;
  App.Free;
end.
