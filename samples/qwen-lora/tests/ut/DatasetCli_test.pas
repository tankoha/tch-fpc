{ UT program for src/DatasetCli.pas: the rows UT-DatasetCli-NNN of
  tests/ut/cases/DatasetCli.cases.md. Compiles and runs small objfpc
  programs with the stock fpc (as DialectVerify's UT); the tokenizer is
  the tiny one of tests/ut/QwenBlocksFixture.pas.
  Invoked with `--all --format=plain` (see .sdlc/ut-coverage.sh).

  Skeleton by the harness owner; the test methods are written in T19b2. }
program DatasetCli_test;

{$mode objfpc}{$H+}

uses
  SysUtils, Classes, fpjson, jsonparser, GgufFile, QwenTokenizer, DialectVerify, DatasetFilter,
  DatasetCli, QwenBlocksFixture, fpcunit, testregistry, consoletestrunner;

type
  TDatasetCliTest = class(TTestCase)
  published
    procedure TestUT_DatasetCli_101;
    procedure TestUT_DatasetCli_102;
    procedure TestUT_DatasetCli_103;
    procedure TestUT_DatasetCli_104;
    procedure TestUT_DatasetCli_105;
    procedure TestUT_DatasetCli_106;
    procedure TestUT_DatasetCli_107;
    procedure TestUT_DatasetCli_108;
    procedure TestUT_DatasetCli_109;
  end;

{ ---- helpers (harness owner's; nobody changes them) ---- }

var
  TmpDir, TinyPath: string;
  Env: TEnvRec;
  Fresh: Integer = 0;

const
  { three small objfpc programs and their output }
  PSum = 'program Sum;' + #10 + '{$mode objfpc}{$H+}' + #10 + 'var' + #10 + '  I, S: Integer;' + #10 +
    'begin' + #10 + '  S := 0;' + #10 + '  for I := 1 to 4 do' + #10 + '    S := S + I;' + #10 +
    '  WriteLn(S);' + #10 + 'end.' + #10;
  PSumOut = '10' + #10;
  PHello = 'program Hello;' + #10 + '{$mode objfpc}{$H+}' + #10 + 'begin' + #10 +
    '  WriteLn(''hello from a test program'');' + #10 + 'end.' + #10;
  PHelloOut = 'hello from a test program' + #10;
  PLoop = 'program Loop;' + #10 + '{$mode objfpc}{$H+}' + #10 + 'var' + #10 + '  K: Integer;' + #10 +
    'begin' + #10 + '  K := 3;' + #10 + '  repeat' + #10 + '    Write(K, '' '');' + #10 + '    Dec(K);' + #10 +
    '  until K = 0;' + #10 + '  WriteLn;' + #10 + 'end.' + #10;
  PLoopOut = '3 2 1 ' + #10;
  UTwice = 'unit TwiceU;' + #10 + '{$mode objfpc}{$H+}' + #10 + 'interface' + #10 +
    'function Twice(X: Integer): Integer;' + #10 + 'implementation' + #10 +
    'function Twice(X: Integer): Integer;' + #10 + 'begin' + #10 + '  Result := 2 * X;' + #10 +
    'end;' + #10 + 'end.' + #10;
  PTwice = 'program UseTwice;' + #10 + '{$mode objfpc}{$H+}' + #10 + 'uses' + #10 + '  TwiceU;' + #10 +
    'begin' + #10 + '  WriteLn(Twice(21));' + #10 + 'end.' + #10;

{ A fresh directory under TmpDir (created), ending in the delimiter. }
function NewDir: string;
begin
  Inc(Fresh);
  Result := TmpDir + 'd' + IntToStr(Fresh) + PathDelim;
  ForceDirectories(Result);
end;

function Fence(const ASrc: string): string;
begin
  Result := '```pascal' + #10 + ASrc + '```' + #10;
end;

{ A record line: dialect objfpc unless ADialect, task_type write, topic,
  language, features, generator {"model":"m"}; answer 'Code:' + the
  fenced sources. }
function Rec(const AId, APrompt: string; const ASrcs: array of string; const AExpected: string;
  const ADialect: string = 'objfpc'): string;
var
  O: TJSONObject;
  A: string;
  I: Integer;
begin
  A := 'Code:' + #10;
  for I := 0 to High(ASrcs) do
    A := A + Fence(ASrcs[I]);
  O := TJSONObject.Create;
  try
    O.Add('id', AId);
    O.Add('dialect', ADialect);
    O.Add('task_type', 'write');
    O.Add('topic', 't');
    O.Add('language', 'en');
    O.Add('features', TJSONArray.Create);
    O.Add('prompt', APrompt);
    O.Add('answer', A);
    O.Add('expected_stdout', AExpected);
    O.Add('generator', TJSONObject.Create(['model', 'm']));
    Result := O.AsJSON;
  finally
    O.Free;
  end;
end;

procedure SaveLines(const APath: string; const ALines: array of string);
var
  L: TStringList;
  I: Integer;
begin
  L := TStringList.Create;
  try
    for I := 0 to High(ALines) do
      L.Add(ALines[I]);
    L.SaveToFile(APath);
  finally
    L.Free;
  end;
end;

function FileText(const APath: string): string;
var
  L: TStringList;
begin
  L := TStringList.Create;
  try
    L.LoadFromFile(APath);
    Result := L.Text;
  finally
    L.Free;
  end;
end;

function LineCount(const APath: string): Integer;
var
  L: TStringList;
begin
  L := TStringList.Create;
  try
    L.LoadFromFile(APath);
    Result := L.Count;
  finally
    L.Free;
  end;
end;

{ The reasons of rejects.jsonl in ADir, joined by '|'. }
function Reasons(const ADir: string): string;
var
  L: TStringList;
  I: Integer;
  O: TJSONObject;
begin
  Result := '';
  L := TStringList.Create;
  try
    L.LoadFromFile(ADir + 'rejects.jsonl');
    for I := 0 to L.Count - 1 do
    begin
      O := GetJSON(L[I], False) as TJSONObject;
      try
        if I > 0 then
          Result := Result + '|';
        Result := Result + O.Strings['reason'];
      finally
        O.Free;
      end;
    end;
  finally
    L.Free;
  end;
end;

{ The tiny tokenizer's tokens of AText as a JSON array. }
function TokJson(T: TQwenTokenizer; const AText: string): string;
var
  A: TTokenArray;
  I: Integer;
begin
  A := T.Encode(AText);
  Result := '[';
  for I := 0 to High(A) do
  begin
    if I > 0 then
      Result := Result + ',';
    Result := Result + IntToStr(A[I]);
  end;
  Result := Result + ']';
end;

{ RunDatasetCLI with Env and the work root <TmpDir>work; AOut / AErr the
  lines joined by #10; the exit code. }
function RunCli(const AArgs: array of string; out AOut, AErr: string): Integer;
var
  O, E: TStringList;
begin
  O := TStringList.Create;
  E := TStringList.Create;
  try
    Result := RunDatasetCLI(AArgs, Env, TmpDir + 'work', O, E);
    AOut := string.Join(#10, O.ToStringArray);
    AErr := string.Join(#10, E.ToStringArray);
  finally
    O.Free;
    E.Free;
  end;
end;

function RawRec(const AId: string; const ABlocks: array of string; const AExpected: string;
  const ADialect: string = 'objfpc'): TRawRecord;
var
  I: Integer;
begin
  Result := Default(TRawRecord);
  Result.Id := AId;
  Result.Dialect := ADialect;
  Result.Expected := AExpected;
  SetLength(Result.Blocks, Length(ABlocks));
  for I := 0 to High(ABlocks) do
    Result.Blocks[I] := ABlocks[I];
end;

{ ---- test methods (rows UT-DatasetCli-101..109) ---- }

procedure TDatasetCliTest.TestUT_DatasetCli_101;
var
  C: TDatasetConfig;
begin
  C := DefaultDatasetConfig;
  AssertEquals(Int64(1), C.Seed);
  AssertEquals(5, C.ValidPercent);
  AssertEquals(10, C.Timeout);
  AssertEquals('v1', C.SpecVersion);
  AssertEquals('', C.Raw);
  AssertEquals('', C.OutDir);
  AssertEquals('', C.Model);
  AssertEquals('', C.RefTokens);
  AssertEquals(0, Length(C.Evals));
  AssertEquals(0, Length(C.LeakDirs));
end;

procedure TDatasetCliTest.TestUT_DatasetCli_102;
var
  C: TDatasetConfig;
begin
  AssertEquals('', ParseDatasetArgs(['--model', 'm.gguf', '--ref-tokens', 'r.jsonl',
    '--eval', 'e1', '--leak-dir', 'x', '--eval', 'e2', '--leak-dir', '/abs/y',
    '--seed', '-4', '--valid-percent', '0', '--timeout', '3600',
    '--spec-version', 'v2', 'raw.jsonl', 'out'], C));
  AssertEquals('m.gguf', C.Model);
  AssertEquals('r.jsonl', C.RefTokens);
  AssertEquals(2, Length(C.Evals));
  AssertEquals('e1', C.Evals[0]);
  AssertEquals('e2', C.Evals[1]);
  AssertEquals(2, Length(C.LeakDirs));
  AssertEquals('x', C.LeakDirs[0]);
  AssertEquals('/abs/y', C.LeakDirs[1]);
  AssertEquals(Int64(-4), C.Seed);
  AssertEquals(0, C.ValidPercent);
  AssertEquals(3600, C.Timeout);
  AssertEquals('v2', C.SpecVersion);
  AssertEquals('raw.jsonl', C.Raw);
  AssertEquals('out/', C.OutDir);
  AssertEquals('', ParseDatasetArgs(['r', 'o/'], C));
  AssertEquals('r', C.Raw);
  AssertEquals('o/', C.OutDir);
  AssertEquals(5, C.ValidPercent);
end;

procedure TDatasetCliTest.TestUT_DatasetCli_103;
var
  C: TDatasetConfig;
begin
  AssertEquals('--eval: value missing', ParseDatasetArgs(['--eval'], C));
  AssertEquals('--valid-percent: an integer in 0..100, got "101"',
    ParseDatasetArgs(['--valid-percent', '101'], C));
  AssertEquals('--valid-percent: an integer in 0..100, got "x"',
    ParseDatasetArgs(['--valid-percent', 'x'], C));
  AssertEquals('--timeout: an integer in 1..3600, got "0"',
    ParseDatasetArgs(['--timeout', '0'], C));
  AssertEquals('--seed: an integer, got "1.5"',
    ParseDatasetArgs(['--seed', '1.5'], C));
  AssertEquals('unknown option --Seed', ParseDatasetArgs(['--Seed', '1'], C));
  AssertEquals('expected RAW OUTDIR, got 1 arguments',
    ParseDatasetArgs(['only'], C));
  AssertEquals('expected RAW OUTDIR, got 3 arguments',
    ParseDatasetArgs(['a', 'b', 'c'], C));
  AssertEquals('--model and --ref-tokens go together',
    ParseDatasetArgs(['--model', 'm.gguf', 'raw.jsonl', 'out'], C));
  AssertEquals('--model and --ref-tokens go together',
    ParseDatasetArgs(['--ref-tokens', 'r.jsonl', 'raw.jsonl', 'out'], C));
end;

procedure TDatasetCliTest.TestUT_DatasetCli_104;
begin
  AssertEquals(1, MainBlock([UTwice, PTwice]));
  AssertEquals(0, MainBlock([PSum]));
  AssertEquals(-1, MainBlock([UTwice]));
  AssertEquals(-1, MainBlock([]));
  AssertEquals('TwiceU', UnitNameOf(UTwice));
  AssertEquals('Dotted.Name',
    UnitNameOf('{ c }' + #10 + 'UNIT Dotted.Name;' + #10 + 'end.'));
  AssertEquals('', UnitNameOf(PSum));
end;

procedure TDatasetCliTest.TestUT_DatasetCli_105;
const
  PSumQ = 'program Sum;' + #10 +
    '  {$mode objfpc}{$H+}' + #10 +
    'var' + #10 +
    '  S, I: Integer;' + #10 +
    'begin' + #10 +
    '  S := 0;' + #10 +
    '  I := 1;' + #10 +
    '  while (I <= 4) do' + #10 +
    '  begin' + #10 +
    '    S := S + Q;' + #10 +
    '    I := I + 1;' + #10 +
    '  end;' + #10 +
    '  WriteLn(S);' + #10 +
    'end.' + #10;
  PSumQExp = PSumQ;
var
  Cls, Outcome, Detail, Src: string;
begin
  Src := PSumQExp;
  Src := Copy(Src, 1, Pos('    S := S + Q;', PSumQExp) - 1) + #10 +
    '  { expect-error: Identifier not found "Q" }' + #10 +
    Copy(PSumQExp, Pos('    S := S + Q;', PSumQExp));
  AssertEquals(True, VerifyRecord(RawRec('a', [PSum], PSumOut), NewDir(), 10,
    Env, Cls, Outcome, Detail));
  AssertEquals('compiled-run', Cls);
  AssertEquals('accepted', Outcome);
  AssertEquals(True, VerifyRecord(RawRec('b', [UTwice, PTwice], '42' + #10),
    NewDir(), 10, Env, Cls, Outcome, Detail));
  AssertEquals(False, VerifyRecord(RawRec('c', [PSum], '11' + #10), NewDir(), 10,
    Env, Cls, Outcome, Detail));
  AssertEquals('wrong-output', Outcome);
  AssertEquals(False, VerifyRecord(RawRec('d', [PSumQ], PSumOut), NewDir(), 10,
    Env, Cls, Outcome, Detail));
  AssertEquals('compile-error', Outcome);
  AssertEquals(True, Pos('Identifier not found "Q"', Detail) > 0);
  AssertEquals(False, VerifyRecord(RawRec('e', [Src], PSumOut), NewDir(), 10,
    Env, Cls, Outcome, Detail));
  AssertEquals('accepted', Outcome);
  AssertEquals(False, VerifyRecord(RawRec('o', ['namespace N;' + #10 + 'end.' + #10],
    '', 'oxygene'), NewDir(), 10, Env, Cls, Outcome, Detail));
  AssertEquals('compiled-run', Cls);   { T22: Oxygene compiled with EBuild }
  AssertEquals('compile-error', Outcome);
  AssertEquals('E: Could not find a "Main" method in this project', Detail);
end;

procedure TDatasetCliTest.TestUT_DatasetCli_106;
const
  PHello2 = 'program Hello2;' + #10 +
    '  {$mode objfpc}{$H+}' + #10 +
    'begin' + #10 +
    '  WriteLn(''hello from a test program'');' + #10 +
    'end.' + #10;
var
  Dir, Out, Err, Card, Train, PWrong, PDup: string;
  L: TStringList;
begin
  Dir := NewDir();
  PWrong := 'program Wrong;' + #10 + '{$mode objfpc}{$H+}' + #10 + 'begin' + #10 +
    '  WriteLn(6 * 7);' + #10 + 'end.' + #10;
  PDup := 'program Sum;' + #10 + '{$mode objfpc}{$H+}' + #10 + 'var' + #10 +
    '  I, S: Integer;' + #10 + 'begin' + #10 + '  S := 0;' + #10 +
    '  for I := 1 to 4 do' + #10 + '    S := S + I;' + #10 + '  WriteLn(S);' + #10 +
    '  { a note }' + #10 + 'end.' + #10;
  SaveLines(Dir + 'raw.jsonl', [
    Rec('a', 'sum the first 4 numbers', [PSum], PSumOut),
    Rec('b', 'greet the user', [PHello], PHelloOut),
    '',
    Rec('c', 'count down from 3 to 1', [PLoop], PLoopOut),
    Rec('d', 'double the number 21', [UTwice, PTwice], '42' + #10),
    Rec('dup', 'sum the first 4 numbers again', [PDup], PSumOut),
    Rec('near', 'greet the user a bit differently', [PHello2], PHelloOut),
    Rec('wrong', 'multiply 6 and 7', [PWrong], '41' + #10)]);
  AssertEquals(0, RunCli(['--valid-percent', '50', '--seed', '2',
    Dir + 'raw.jsonl', Dir + 'out'], Out, Err));
  AssertEquals('4 of 7 records accepted (train 2, valid 2) into ' + Dir + 'out/', Out);
  AssertEquals('', Err);
  AssertEquals('duplicate: a|near-duplicate: b|verify: wrong-output',
    Reasons(Dir + 'out/'));
  AssertEquals(2, LineCount(Dir + 'out/train.jsonl'));
  AssertEquals(2, LineCount(Dir + 'out/valid.jsonl'));
  AssertEquals(4, LineCount(Dir + 'out/manifest.jsonl'));
  L := TStringList.Create;
  try
    L.LoadFromFile(Dir + 'out/rejects.jsonl');
    AssertEquals('{"line":6,"id":"dup","reason":"duplicate: a","detail":""}', L[0]);
  finally
    L.Free;
  end;
  Card := FileText(Dir + 'out/card.md');
  AssertEquals(True, Pos('Raw records: 7. Accepted: 4 (train 2, valid 2). ' +
    'Rejected: 3. Yield: 57.1 %.', Card) > 0);
  AssertEquals(True, Pos('| objfpc | write | compiled-run | 2 | 2 |', Card) > 0);
  AssertEquals(True, Pos('| verify | 1 |', Card) > 0);
  Train := FileText(Dir + 'out/train.jsonl');
  AssertEquals(True, Pos('"class" : "compiled-run"', Train) > 0);
end;

procedure TDatasetCliTest.TestUT_DatasetCli_107;
var
  Dir, Out, Err, PB, L1, L2: string;
  O: TJSONObject;
begin
  Dir := NewDir();
  PB := 'Write a program that greets the reader of this test once.';
  O := TJSONObject.Create;
  O.Add('id', 'e1');
  O.Add('prompt', PB);
  O.Add('reference', 'begin end.');
  L1 := O.AsJSON;
  O.Free;
  O := TJSONObject.Create;
  O.Add('id', 'e2');
  O.Add('prompt', 'add one to three');
  O.Add('reference', PSum);
  L2 := O.AsJSON;
  O.Free;
  SaveLines(Dir + 'e.jsonl', [L1, '', L2]);
  ForceDirectories(Dir + 'ldir/sub');
  SaveLines(Dir + 'ldir/sub/loop.pas', [PLoop]);
  SaveLines(Dir + 'raw.jsonl', [
    Rec('a', 'sum the first 4 numbers', [PSum], PSumOut),
    Rec('b', PB, [PHello], PHelloOut),
    Rec('c', 'count down from 3 to 1', [PLoop], PLoopOut),
    Rec('d', 'double the number 21', [UTwice, PTwice], '42' + #10)]);
  AssertEquals(0, RunCli(['--leak-dir', Dir + 'ldir', '--eval', Dir + 'e.jsonl',
    Dir + 'raw.jsonl', Dir + 'o'], Out, Err));
  AssertEquals('1 of 4 records accepted (train 1, valid 0) into ' + Dir + 'o/', Out);
  AssertEquals('leak-eval: e2|leak-eval: e1|leak-example: sub/loop.pas',
    Reasons(Dir + 'o/'));
end;

procedure TDatasetCliTest.TestUT_DatasetCli_108;
var
  Dir, Out, Err, PA, PB, LA, LB, Card: string;
  G: TGgufFile;
  Tok: TQwenTokenizer;
begin
  Dir := NewDir();
  PA := 'sum the first 4 numbers';
  PB := 'add one to three';
  G := TGgufFile.Open(TinyPath);
  Tok := TQwenTokenizer.Create(G);
  try
    LA := '{"id":"a","prompt":' + TokJson(Tok, PA) + ',"answer":' +
      TokJson(Tok, 'Code:' + #10 + Fence(PSum)) + '}';
    LB := '{"id":"b","prompt":' + TokJson(Tok, PB) + ',"answer":[1,2,3]}';
  finally
    Tok := nil;
    G := nil;
  end;
  SaveLines(Dir + 'ref.jsonl', [LA, LB, '', '{"answer":[],"id":"a","prompt":[]}']);
  SaveLines(Dir + 'raw.jsonl', [
    Rec('a', PA, [PSum], PSumOut),
    Rec('b', PB, [PHello], PHelloOut),
    Rec('c', 'count down from 3 to 1', [PLoop], PLoopOut)]);
  AssertEquals(0, RunCli(['--model', TinyPath, '--ref-tokens', Dir + 'ref.jsonl',
    Dir + 'raw.jsonl', Dir + 'o'], Out, Err));
  AssertEquals('tokenizer|tokenizer: no reference', Reasons(Dir + 'o/'));
  Card := FileText(Dir + 'o/card.md');
  AssertEquals(True, Pos('Tokenizer check: QwenTokenizer against ref.jsonl.',
    Card) > 0);
end;

procedure TDatasetCliTest.TestUT_DatasetCli_109;
var
  Dir, Out, Err: string;
begin
  Dir := NewDir();
  SaveLines(Dir + 'raw.jsonl', [
    'not json',
    Rec('a', 'sum the first 4 numbers', [PSum], PSumOut),
    Rec('a', 'greet the user', [PHello], PHelloOut),
    Rec('u', 'a helper unit only', [UTwice], ''),
    Rec('m', 'greet the user', [PHello], PHelloOut, 'fpc-delphi')]);
  AssertEquals(0, RunCli([Dir + 'raw.jsonl', Dir + 'o'], Out, Err));
  AssertEquals('schema: not a JSON object|duplicate-id|schema: answer has no program block|dialect: objfpc', Reasons(Dir + 'o/'));
  AssertEquals(2, RunCli(['--frob'], Out, Err));
  AssertEquals('unknown option --frob' + #10 + K_DATASET_USAGE, Err);
  AssertEquals('', Out);
  AssertEquals(1, RunCli([Dir + 'none.jsonl', Dir + 'o'], Out, Err));
  AssertEquals('no raw file ' + Dir + 'none.jsonl', Err);
  AssertEquals(1, RunCli(['--leak-dir', Dir + 'nodir', Dir + 'raw.jsonl', Dir + 'o'], Out, Err));
  AssertEquals('no leak directory ' + Dir + 'nodir', Err);
  AssertEquals(1, RunCli(['--eval', Dir + 'e.jsonl', Dir + 'raw.jsonl', Dir + 'o'], Out, Err));
  AssertEquals('no file ' + Dir + 'e.jsonl', Err);
end;

var
  App: TTestRunner;
begin
  TmpDir := GetTempDir + 'datasetcli-ut-' + IntToStr(GetProcessID) + PathDelim;
  ForceDirectories(TmpDir);
  TinyPath := TmpDir + 'tiny.gguf';
  WriteTinyQwenModel(TinyPath, 5, thTied, False);
  Env := ReadEnvRec('', '', GetEnvironmentVariable('HOME'));
  RegisterTest(TDatasetCliTest);
  App := TTestRunner.Create(nil);
  App.Initialize;
  App.Title := 'DatasetCli UT';
  App.Run;
  App.Free;
end.
