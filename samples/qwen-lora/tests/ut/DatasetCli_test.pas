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
