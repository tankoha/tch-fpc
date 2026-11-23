{ UT program for src/TrainData.pas: the rows UT-TrainData-NNN of
  tests/ut/cases/TrainData.cases.md (stage 1 from 001: samples and the
  data file; stage 2 from 101: epoch order, batches, tensors). Runs on the
  CPU (.sdlc/ut-coverage.sh hides the GPUs). Invoked with `--all
  --format=plain` (see .sdlc/ut-coverage.sh).

  Skeleton by the harness owner; the test methods are written in T15b. }
program TrainData_test;

{$mode objfpc}{$H+}

uses
  SysUtils, Classes, Math, Tch, QwenTokenizer, QwenChat, TokenizerFixture, TrainData,
  fpcunit, testregistry, consoletestrunner;

const
  { The template's tags, built from pieces: never write them literally. }
  TagS = '<|' + 'im_start|>';
  TagE = '<|' + 'im_end|>';
  ThinkO = '<' + 'think>';
  ThinkC = '</' + 'think>';
  { A backslash, so that no JSON escape is written literally here. }
  BS = '\';
  { The four records of FixFile: a, b and d are admitted by FixClasses. }
  FixLines: array[0..3] of RawByteString = (
    '{"id":"a","dialect":"objfpc","prompt":"Q","answer":"A","class":"compiled-run"}',
    '{"id":"b","dialect":"delphi","prompt":"xyz","answer":"  12345  ","class":"fpc-approx","extra":[1,2]}',
    '{"id":"c","dialect":"oxygene","prompt":"p","answer":"q","class":"static-only"}',
    '{"id":"d","dialect":"objfpc","prompt":"' + #$E6#$97#$A5#$E6#$9C#$AC + '","answer":"' + #$C3#$A9 + '","class":"compiled-run"}');

type
  TTrainDataTest = class(TTestCase)
  public
    { The tiny tokenizer of T05 (TokenizerFixture), created once. }
    class function Tok: TQwenTokenizer; static;
    { Writes ALines (one per line) to the file AName under TmpDir; returns its path. }
    class function WriteLines(const AName: string; const ALines: array of RawByteString): string; static;
    { The set of FixLines with the classes compiled-run and fpc-approx and AMaxLen. }
    class function FixSet(AMaxLen: Integer): TTrainSet; static;
    { A set whose sample I has exactly ALens[I] tokens (each >= 36): id sI,
      prompt 'P', answer ALens[I] - 35 times 'z'. }
    class function LenSet(const ALens: array of Integer): TTrainSet; static;
    { The batches as text: indices separated by spaces, batches by '|'. }
    class function BatchText(const AB: TBatches): string; static;
    { The integers as text, separated by spaces. }
    class function IntsText(const A: array of Integer): string; static;
    { The tensor's values as Double, row-major. }
    class function Arr(const T: TTensor): TDoubleArray; static;
  published
  end;

var
  TmpDir: string;
  FTok: TQwenTokenizer = nil;
  FileNo: Integer = 0;

class function TTrainDataTest.Tok: TQwenTokenizer;
begin
  if FTok = nil then
    FTok := NewTinyTokenizer;
  Result := FTok;
end;

class function TTrainDataTest.WriteLines(const AName: string; const ALines: array of RawByteString): string;
var
  S: RawByteString;
  I: Integer;
begin
  S := '';
  for I := 0 to High(ALines) do
    S := S + ALines[I] + #10;
  Result := TmpDir + AName;
  WriteBytes(Result, S);
end;

class function TTrainDataTest.FixSet(AMaxLen: Integer): TTrainSet;
begin
  Result := TTrainSet.Create(Tok, WriteLines('fix.jsonl', FixLines),
    ['compiled-run', 'fpc-approx'], AMaxLen);
end;

class function TTrainDataTest.LenSet(const ALens: array of Integer): TTrainSet;
var
  L: array of RawByteString;
  I: Integer;
begin
  L := nil;
  SetLength(L, Length(ALens));
  for I := 0 to High(ALens) do
    L[I] := Format('{"id":"s%d","dialect":"objfpc","prompt":"P","answer":"%s","class":"compiled-run"}',
      [I, StringOfChar('z', ALens[I] - 35)]);
  Inc(FileNo);
  Result := TTrainSet.Create(Tok, WriteLines('len' + IntToStr(FileNo) + '.jsonl', L),
    ['compiled-run'], 100000);
end;

class function TTrainDataTest.BatchText(const AB: TBatches): string;
var
  I: Integer;
begin
  Result := '';
  for I := 0 to High(AB) do
  begin
    if I > 0 then
      Result := Result + '|';
    Result := Result + IntsText(AB[I]);
  end;
end;

class function TTrainDataTest.IntsText(const A: array of Integer): string;
var
  I: Integer;
begin
  Result := '';
  for I := 0 to High(A) do
  begin
    if I > 0 then
      Result := Result + ' ';
    Result := Result + IntToStr(A[I]);
  end;
end;

class function TTrainDataTest.Arr(const T: TTensor): TDoubleArray;
begin
  Result := T.detach.to_device(TDevice.Cpu, kDouble).reshape([-1]).ToDoubleArray;
end;

{ ETrainData from creating a set of ALines with AClasses and AMaxLen, message AMessage. }
procedure CheckLoad(T: TTestCase; const ALines: array of RawByteString;
  AMaxLen: Integer; const AMessage: string);
var
  S: TTrainSet;
begin
  S := nil;
  try
    try
      S := TTrainSet.Create(TTrainDataTest.Tok, TTrainDataTest.WriteLines('bad.jsonl', ALines),
        ['compiled-run'], AMaxLen);
      T.Fail('no ETrainData');
    except
      on E: ETrainData do
        T.AssertEquals(AMessage, E.Message);
    end;
  finally
    S.Free;
  end;
end;

{ Decode of the tokens of S in [AFrom, ATo). }
function Span(const S: TTrainSample; AFrom, ATo: Integer): RawByteString;
begin
  Result := TTrainDataTest.Tok.Decode(Copy(S.Tokens, AFrom, ATo - AFrom));
end;

{ Every sample of every batch once, and every batch within ABudget. }
procedure CheckBatches(T: TTestCase; S: TTrainSet; const AB: TBatches; ABudget: Integer);
var
  Seen: array of Integer;
  I, J, M: Integer;
begin
  Seen := nil;
  SetLength(Seen, S.Count);
  for I := 0 to High(AB) do
  begin
    M := 0;
    for J := 0 to High(AB[I]) do
    begin
      Inc(Seen[AB[I][J]]);
      M := Max(M, Length(S[AB[I][J]].Tokens));
    end;
    T.AssertTrue(Format('batch %d: %d x %d within %d', [I, Length(AB[I]), M, ABudget]),
      Length(AB[I]) * M <= ABudget);
    T.AssertTrue(Format('batch %d not empty', [I]), Length(AB[I]) > 0);
  end;
  for I := 0 to High(Seen) do
    T.AssertEquals(Format('sample %d once', [I]), 1, Seen[I]);
end;

procedure RemoveTree(const ADir: string);
var
  R: TSearchRec;
  D: string;
begin
  D := IncludeTrailingPathDelimiter(ADir);
  if FindFirst(D + '*', faAnyFile, R) = 0 then
  try
    repeat
      if (R.Name = '.') or (R.Name = '..') then
        Continue;
      if (R.Attr and faDirectory) <> 0 then
        RemoveTree(D + R.Name)
      else
        DeleteFile(D + R.Name);
    until FindNext(R) <> 0;
  finally
    FindClose(R);
  end;
  RemoveDir(ADir);
end;

var
  App: TTestRunner;
begin
  TmpDir := GetTempDir + 'traindata-ut-' + IntToStr(GetProcessID) + PathDelim;
  ForceDirectories(TmpDir);
  RegisterTest(TTrainDataTest);
  App := TTestRunner.Create(nil);
  App.Initialize;
  App.Title := 'TrainData UT';
  App.Run;
  App.Free;
  FTok.Free;
  RemoveTree(TmpDir);
end.
