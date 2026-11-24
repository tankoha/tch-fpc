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
