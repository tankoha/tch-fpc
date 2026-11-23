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
    procedure TestUT_TrainData_001;
    procedure TestUT_TrainData_002;
    procedure TestUT_TrainData_003;
    procedure TestUT_TrainData_004;
    procedure TestUT_TrainData_005;
    procedure TestUT_TrainData_006;
    procedure TestUT_TrainData_007;
    procedure TestUT_TrainData_008;
    procedure TestUT_TrainData_009;
    procedure TestUT_TrainData_010;
    procedure TestUT_TrainData_011;
    procedure TestUT_TrainData_012;
    procedure TestUT_TrainData_013;
    procedure TestUT_TrainData_014;
    procedure TestUT_TrainData_101;
    procedure TestUT_TrainData_102;
    procedure TestUT_TrainData_103;
    procedure TestUT_TrainData_104;
    procedure TestUT_TrainData_105;
    procedure TestUT_TrainData_106;
    procedure TestUT_TrainData_107;
    procedure TestUT_TrainData_108;
    procedure TestUT_TrainData_109;
    procedure TestUT_TrainData_110;
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

{ UT-TrainData-001: EncodeSample, plain Q/A. }
procedure TTrainDataTest.TestUT_TrainData_001;
var
  T: TTokenArray;
  F, T2: Integer;
begin
  T := EncodeSample(Tok, 'Q', 'A', F, T2);
  AssertEquals(36, Length(T));
  AssertEquals(33, F);
  AssertEquals(35, T2);
  AssertEquals(TagS + 'user'#10'Q' + TagE + #10 + TagS + 'assistant'#10
    + ThinkO + #10#10 + ThinkC + #10#10 + 'A' + TagE + #10, Tok.Decode(T));
  AssertEquals('A' + TagE, Tok.Decode(Copy(T, F, T2 - F)));
end;

{ UT-TrainData-002: EncodeSample, trimmed answer. }
procedure TTrainDataTest.TestUT_TrainData_002;
var
  T: TTokenArray;
  F, T2: Integer;
begin
  T := EncodeSample(Tok, 'xyz', '  12345  ', F, T2);
  AssertEquals(42, Length(T));
  AssertEquals(35, F);
  AssertEquals(41, T2);
  AssertEquals('12345' + TagE, Tok.Decode(Copy(T, F, T2 - F)));
end;

{ UT-TrainData-003: EncodeSample, UTF-8 prompt/answer. }
procedure TTrainDataTest.TestUT_TrainData_003;
var
  T: TTokenArray;
  F, T2: Integer;
begin
  T := EncodeSample(Tok, #$E6#$97#$A5#$E6#$9C#$AC, #$C3#$A9, F, T2);
  AssertEquals(42, Length(T));
  AssertEquals(38, F);
  AssertEquals(41, T2);
  AssertEquals(#$C3#$A9 + TagE, Tok.Decode(Copy(T, F, T2 - F)));
end;

{ UT-TrainData-004: Create, both classes admitted. }
procedure TTrainDataTest.TestUT_TrainData_004;
var
  S: TTrainSet;
begin
  S := nil;
  try
    S := FixSet(100);
    AssertEquals(3, S.Count);
    AssertEquals(1, S.SkippedClass);
    AssertEquals(0, S.SkippedLong);
    AssertEquals('a b d', S[0].Id + ' ' + S[1].Id + ' ' + S[2].Id);
    AssertEquals('objfpc delphi objfpc',
      S[0].Dialect + ' ' + S[1].Dialect + ' ' + S[2].Dialect);
    AssertEquals('36 42 42',
      IntToStr(Length(S[0].Tokens)) + ' ' + IntToStr(Length(S[1].Tokens))
      + ' ' + IntToStr(Length(S[2].Tokens)));
    AssertEquals('33 35', IntToStr(S[0].LossFrom) + ' ' + IntToStr(S[0].LossTo));
    AssertEquals('35 41', IntToStr(S[1].LossFrom) + ' ' + IntToStr(S[1].LossTo));
    AssertEquals('38 41', IntToStr(S[2].LossFrom) + ' ' + IntToStr(S[2].LossTo));
    AssertEquals(42, S.MaxLength);
  finally
    S.Free;
  end;
end;

{ UT-TrainData-005: Create, drops samples over MaxLen. }
procedure TTrainDataTest.TestUT_TrainData_005;
var
  S41, S42: TTrainSet;
begin
  S41 := nil;
  S42 := nil;
  try
    S41 := FixSet(41);
    AssertEquals(1, S41.Count);
    AssertEquals('a', S41[0].Id);
    AssertEquals(2, S41.SkippedLong);
    AssertEquals(1, S41.SkippedClass);
    S42 := FixSet(42);
    AssertEquals(3, S42.Count);
  finally
    S41.Free;
    S42.Free;
  end;
end;

{ UT-TrainData-006: Create, blank lines + empty class list. }
procedure TTrainDataTest.TestUT_TrainData_006;
var
  L: array of RawByteString;
  S: TTrainSet;
begin
  L := ['' , FixLines[0], '   ', FixLines[1], ''];
  S := nil;
  try
    S := TTrainSet.Create(Tok, WriteLines('blank.jsonl', L),
      ['compiled-run', 'fpc-approx'], 100);
    AssertEquals(2, S.Count);
    AssertEquals('a b', S[0].Id + ' ' + S[1].Id);
  finally
    S.Free;
  end;
  S := nil;
  try
    S := TTrainSet.Create(Tok, WriteLines('noclass.jsonl', FixLines), [], 100);
    AssertEquals(0, S.Count);
    AssertEquals(4, S.SkippedClass);
    AssertEquals(0, S.MaxLength);
  finally
    S.Free;
  end;
end;

{ UT-TrainData-007: Create, missing answer/id/class. }
procedure TTrainDataTest.TestUT_TrainData_007;
var
  L: array of RawByteString;
begin
  L := [FixLines[0],
    '{"id":"a","dialect":"objfpc","prompt":"Q","class":"compiled-run"}'];
  CheckLoad(Self, L, 100, 'line 2: missing or bad answer');
  L := ['{"dialect":"objfpc","prompt":"Q","answer":"A","class":"compiled-run"}'];
  CheckLoad(Self, L, 100, 'line 1: missing or bad id');
  L := ['{"id":"a","dialect":"objfpc","prompt":"Q","answer":"A","class":1}'];
  CheckLoad(Self, L, 100, 'line 1: missing or bad class');
end;

{ UT-TrainData-008: Create, id/dialect/prompt not a string. }
procedure TTrainDataTest.TestUT_TrainData_008;
var
  L: array of RawByteString;
begin
  L := ['{"id":7,"dialect":"objfpc","prompt":"Q","answer":"A","class":"x"}'];
  CheckLoad(Self, L, 100, 'line 1: missing or bad id');
  L := ['{"id":"a","dialect":null,"prompt":"Q","answer":"A","class":"x"}'];
  CheckLoad(Self, L, 100, 'line 1: missing or bad dialect');
  L := ['{"id":"a","dialect":"objfpc","prompt":["p"],"answer":"A","class":"x"}'];
  CheckLoad(Self, L, 100, 'line 1: missing or bad prompt');
end;

{ UT-TrainData-009: Create, line not a JSON object. }
procedure TTrainDataTest.TestUT_TrainData_009;
var
  L: array of RawByteString;
begin
  L := ['{"id":'];
  CheckLoad(Self, L, 100, 'line 1: not a JSON object');
  L := ['' , '[1, 2]'];
  CheckLoad(Self, L, 100, 'line 2: not a JSON object');
end;

{ UT-TrainData-010: Create, duplicate id. }
procedure TTrainDataTest.TestUT_TrainData_010;
var
  L: array of RawByteString;
begin
  L := [FixLines[0], FixLines[2], FixLines[0]];
  CheckLoad(Self, L, 100, 'line 3: duplicate id a');
end;

{ UT-TrainData-011: Create, wide \u escape. }
procedure TTrainDataTest.TestUT_TrainData_011;
var
  L: array of RawByteString;
  S: TTrainSet;
begin
  L := ['{"id":"a","dialect":"objfpc","prompt":"Q","answer":"'
    + BS + 'u00e9","class":"compiled-run"}'];
  CheckLoad(Self, L, 100,
    'line 1: a ' + BS + 'u escape of a non-ASCII character; write UTF-8');
  L := ['{"id":"a","dialect":"objfpc","prompt":"Q","answer":"'
    + BS + 'u0041' + BS + BS + 'u00e9' + BS + 'n","class":"compiled-run"}'];
  S := nil;
  try
    S := TTrainSet.Create(Tok, WriteLines('wide.jsonl', L), ['compiled-run'], 100);
    AssertEquals(1, S.Count);
    AssertEquals('A' + BS + 'u00e9' + TagE, Span(S[0], S[0].LossFrom, S[0].LossTo));
  finally
    S.Free;
  end;
end;

{ UT-TrainData-012: Create, missing file + MaxLen 0. }
procedure TTrainDataTest.TestUT_TrainData_012;
begin
  try
    TTrainSet.Create(Tok, TmpDir + 'missing.jsonl', ['compiled-run'], 100);
    Fail('no ETrainData');
  except
    on E: ETrainData do
      AssertEquals('no data file ' + TmpDir + 'missing.jsonl', E.Message);
  end;
  try
    TTrainSet.Create(Tok, WriteLines('zero.jsonl', FixLines), ['compiled-run'], 0);
    Fail('no ETrainData');
  except
    on E: ETrainData do
      AssertEquals('max length must be positive, got 0', E.Message);
  end;
end;

{ UT-TrainData-013: Samples, out of range. }
procedure TTrainDataTest.TestUT_TrainData_013;
var
  S: TTrainSet;
begin
  S := nil;
  try
    S := FixSet(100);
    try
      S[3];
      Fail('no ETrainData');
    except
      on E: ETrainData do
        AssertEquals('sample 3 outside 0..2', E.Message);
    end;
    try
      S[-1];
      Fail('no ETrainData');
    except
      on E: ETrainData do
        AssertEquals('sample -1 outside 0..2', E.Message);
    end;
  finally
    S.Free;
  end;
end;

{ UT-TrainData-014: Samples, loss span decodes to answer. }
procedure TTrainDataTest.TestUT_TrainData_014;
var
  S: TTrainSet;
  I, Len: Integer;
  Z, E: string;
begin
  S := nil;
  try
    S := LenSet([36, 40, 77]);
    AssertEquals('36 40 77',
      IntToStr(Length(S[0].Tokens)) + ' ' + IntToStr(Length(S[1].Tokens))
      + ' ' + IntToStr(Length(S[2].Tokens)));
    for I := 0 to 2 do
    begin
      Len := Length(S[I].Tokens);
      Z := StringOfChar('z', Len - 35);
      AssertEquals(Z + TagE, Span(S[I], S[I].LossFrom, S[I].LossTo));
    end;
  finally
    S.Free;
  end;
  S := nil;
  try
    S := FixSet(100);
    AssertEquals('A' + TagE, Span(S[0], S[0].LossFrom, S[0].LossTo));
    AssertEquals('12345' + TagE, Span(S[1], S[1].LossFrom, S[1].LossTo));
    E := #$C3#$A9 + TagE;
    AssertEquals(E, Span(S[2], S[2].LossFrom, S[2].LossTo));
    for I := 0 to 2 do
      AssertEquals(#10, Span(S[I], S[I].LossTo, Length(S[I].Tokens)));
  finally
    S.Free;
  end;
end;

{ UT-TrainData-101: EpochOrder, permutation per (seed, epoch). }
procedure TTrainDataTest.TestUT_TrainData_101;
var
  A, C1, C2: TIntArray;
begin
  A := EpochOrder(10, 1, 0);
  AssertEquals('4 2 8 1 9 3 0 6 7 5', IntsText(A));
  AssertEquals('8 0 4 2 7 6 1 3 9 5', IntsText(EpochOrder(10, 1, 1)));
  C1 := EpochOrder(10, 2, 0);
  C2 := EpochOrder(10, 2, 0);
  AssertTrue('seed 2 equal twice', IntsText(C1) = IntsText(C2));
  AssertTrue('seed 2 differs from seed 1', IntsText(C1) <> IntsText(A));
  AssertEquals('', IntsText(EpochOrder(0, 1, 0)));
  AssertEquals('0', IntsText(EpochOrder(1, 1, 0)));
end;

{ UT-TrainData-102: EpochOrder, large N, negative seed. }
procedure TTrainDataTest.TestUT_TrainData_102;
var
  A: TIntArray;
  Seen: array of Boolean;
  I: Integer;
begin
  A := EpochOrder(1000, -5, 3);
  AssertEquals(1000, Length(A));
  Seen := nil;
  SetLength(Seen, 1000);
  for I := 0 to High(A) do
  begin
    AssertTrue('entry ' + IntToStr(I) + ' in 0..999', (A[I] >= 0) and (A[I] < 1000));
    AssertTrue('entry ' + IntToStr(I) + ' once', not Seen[A[I]]);
    Seen[A[I]] := True;
  end;
  for I := 0 to High(Seen) do
    AssertTrue('value ' + IntToStr(I) + ' present', Seen[I]);
  AssertTrue('differs from EpochOrder(1000, 0, 0)',
    IntsText(A) <> IntsText(EpochOrder(1000, 0, 0)));
end;

{ UT-TrainData-103: EpochBatches, pool split + shuffle. }
procedure TTrainDataTest.TestUT_TrainData_103;
var
  S: TTrainSet;
  O: TIntArray;
  B1, B2: TBatches;
  I, J, K: Integer;
  Found: Boolean;
begin
  S := nil;
  try
    S := LenSet([40, 50, 60, 70, 80, 90, 100, 110]);
    O := EpochOrder(8, 3, 0);
    AssertEquals('7 0 1 4 2 6 3 5', IntsText(O));
    B1 := EpochBatches(S, 3, 0, 4, 200);
    AssertEquals('7|4|0 1|2 3|5 6', BatchText(B1));
    CheckBatches(Self, S, B1, 200);
    B2 := EpochBatches(S, 3, 0, 8, 110);
    AssertEquals(7, Length(B2));
    Found := False;
    for I := 0 to High(B2) do
      for J := 0 to High(B2[I]) do
        for K := 0 to High(B2[I]) do
          if (B2[I][J] = 0) and (B2[I][K] = 1) then
            Found := True;
    AssertTrue('40 and 50 together', Found);
    CheckBatches(Self, S, B2, 110);
  finally
    S.Free;
  end;
end;

{ UT-TrainData-104: EpochBatches, 200 samples, one pool each. }
procedure TTrainDataTest.TestUT_TrainData_104;
var
  S: TTrainSet;
  L: array of Integer;
  Order, PosOf: TIntArray;
  B1, B2, B3: TBatches;
  P, I, J, K: Integer;
  K0, K1: Integer;
begin
  L := nil;
  SetLength(L, 200);
  for I := 0 to 199 do
    L[I] := 36 + (37 * I mod 265);
  S := nil;
  try
    S := LenSet(L);
    B1 := EpochBatches(S, 11, 2, 16, 512);
    CheckBatches(Self, S, B1, 512);
    Order := EpochOrder(200, 11, 2);
    PosOf := nil;
    SetLength(PosOf, 200);
    for P := 0 to High(Order) do
      PosOf[Order[P]] := P;
    K0 := -1;
    for I := 0 to High(B1) do
    begin
      K1 := -1;
      for J := 0 to High(B1[I]) do
      begin
        K := PosOf[B1[I][J]] div 16;
        if K1 < 0 then
          K1 := K
        else
          AssertEquals('batch ' + IntToStr(I) + ': within one pool', K1, K);
      end;
      if I = 0 then
        AssertEquals('first batch from pool 0', 0, K1)
      else
        AssertTrue('batch ' + IntToStr(I) + ': pools in order', K1 >= K0);
      K0 := K1;
    end;
    AssertEquals('last batch from pool 12', 12, K0);
    B2 := EpochBatches(S, 11, 2, 16, 512);
    AssertTrue('equal on a second call', BatchText(B1) = BatchText(B2));
    B3 := EpochBatches(S, 11, 3, 16, 512);
    AssertTrue('differs for epoch 3', BatchText(B1) <> BatchText(B3));
    B3 := EpochBatches(S, 11, 2, 1000, 2048);
    CheckBatches(Self, S, B3, 2048);
  finally
    S.Free;
  end;
end;

{ UT-TrainData-105: EpochBatches, pool 0 / budget below longest. }
procedure TTrainDataTest.TestUT_TrainData_105;
var
  S: TTrainSet;
begin
  S := nil;
  try
    S := LenSet([40, 110]);
    try
      EpochBatches(S, 3, 0, 0, 200);
      Fail('no ETrainData');
    except
      on E: ETrainData do
        AssertEquals('pool must be positive, got 0', E.Message);
    end;
    try
      EpochBatches(S, 3, 0, 4, 109);
      Fail('no ETrainData');
    except
      on E: ETrainData do
        AssertEquals('budget 109 is below the longest sample, 110', E.Message);
    end;
  finally
    S.Free;
  end;
end;

{ UT-TrainData-106: EpochBatches, empty set. }
procedure TTrainDataTest.TestUT_TrainData_106;
var
  S: TTrainSet;
  B: TBatches;
  L: array of RawByteString;
begin
  L := nil;
  S := nil;
  try
    S := TTrainSet.Create(Tok, WriteLines('empty.jsonl', L), ['compiled-run'], 100);
    B := EpochBatches(S, 1, 0, 4, 200);
    AssertEquals(0, Length(B));
    AssertEquals('', BatchText(B));
  finally
    S.Free;
  end;
end;

{ UT-TrainData-107: SkipBatches, by sample position. }
procedure TTrainDataTest.TestUT_TrainData_107;
var
  AB: TBatches;
begin
  AB := nil;
  SetLength(AB, 3);
  SetLength(AB[0], 2);
  AB[0][0] := 4;
  AB[0][1] := 1;
  SetLength(AB[1], 1);
  AB[1][0] := 0;
  SetLength(AB[2], 3);
  AB[2][0] := 2;
  AB[2][1] := 3;
  AB[2][2] := 5;
  AssertEquals('4 1|0|2 3 5', BatchText(SkipBatches(AB, 0)));
  AssertEquals('2 3 5', BatchText(SkipBatches(AB, 3)));
  AssertEquals('', BatchText(SkipBatches(AB, 6)));
  try
    SkipBatches(AB, 1);
    Fail('no ETrainData');
  except
    on E: ETrainData do
      AssertEquals('position 1 is not at the end of a batch', E.Message);
  end;
  try
    SkipBatches(AB, 7);
    Fail('no ETrainData');
  except
    on E: ETrainData do
      AssertEquals('position 7 is not at the end of a batch', E.Message);
  end;
  try
    SkipBatches(AB, -1);
    Fail('no ETrainData');
  except
    on E: ETrainData do
      AssertEquals('position -1 is negative', E.Message);
  end;
end;

{ UT-TrainData-108: BatchTensors, padded batch. }
procedure TTrainDataTest.TestUT_TrainData_108;
var
  S: TTrainSet;
  Ids, Tg, M: TTensor;
  SA, TA, MA: TDoubleArray;
  I: Integer;
begin
  S := nil;
  try
    S := FixSet(100);
    BatchTensors(S, [0, 2], 7, Ids, Tg, M);
    AssertEquals('shape B', 2, Integer(Ids.Shape[0]));
    AssertEquals('shape T', 42, Integer(Ids.Shape[1]));
    AssertTrue('ids kind int64', Ids.Kind = kInt64);
    AssertTrue('targets kind int64', Tg.Kind = kInt64);
    AssertTrue('mask kind float', M.Kind = kFloat);
    SA := Arr(Ids);
    TA := Arr(Tg);
    MA := Arr(M);
    for I := 0 to 35 do
      AssertEquals('row 0 id col ' + IntToStr(I), S[0].Tokens[I], Trunc(SA[I]));
    for I := 36 to 41 do
      AssertEquals('row 0 pad col ' + IntToStr(I), 7, Trunc(SA[I]));
    for I := 0 to 41 do
      AssertEquals('row 1 id col ' + IntToStr(I), S[2].Tokens[I], Trunc(SA[42 + I]));
    for I := 0 to 34 do
      AssertEquals('row 0 target col ' + IntToStr(I), S[0].Tokens[I + 1], Trunc(TA[I]));
    for I := 35 to 41 do
      AssertEquals('row 0 target pad col ' + IntToStr(I), 7, Trunc(TA[I]));
    for I := 0 to 40 do
      AssertEquals('row 1 target col ' + IntToStr(I), S[2].Tokens[I + 1], Trunc(TA[42 + I]));
    AssertEquals('row 1 last target', 7, Trunc(TA[83]));
    for I := 0 to 83 do
    begin
      if I in [32, 33, 79, 80, 81] then
        AssertEquals('mask col ' + IntToStr(I), 1.0, MA[I], 0.0)
      else
        AssertEquals('mask col ' + IntToStr(I), 0.0, MA[I], 0.0);
    end;
  finally
    S.Free;
  end;
end;

{ UT-TrainData-109: BatchTensors, masked targets decode to answer. }
procedure TTrainDataTest.TestUT_TrainData_109;
var
  S: TTrainSet;
  B: TBatches;
  Ids, Tg, M: TTensor;
  TgA, MA: TDoubleArray;
  Toks: array of Integer;
  I, J, K, TSize, Len: Integer;
  Z: string;
begin
  S := nil;
  try
    S := LenSet([36, 40, 50, 77, 120, 121]);
    B := EpochBatches(S, 9, 0, 6, 250);
    AssertTrue('at least one batch', Length(B) > 0);
    for I := 0 to High(B) do
    begin
      BatchTensors(S, B[I], 0, Ids, Tg, M);
      TSize := Integer(Ids.Shape[1]);
      TgA := Arr(Tg);
      MA := Arr(M);
      for J := 0 to High(B[I]) do
      begin
        Len := Length(S[B[I][J]].Tokens);
        Z := StringOfChar('z', Len - 35) + TagE;
        Toks := nil;
        for K := 0 to TSize - 1 do
          if MA[J * TSize + K] = 1 then
          begin
            SetLength(Toks, Length(Toks) + 1);
            Toks[High(Toks)] := Trunc(TgA[J * TSize + K]);
          end;
        AssertEquals('batch ' + IntToStr(I) + ' row ' + IntToStr(J)
          + ': masked targets', Z, Tok.Decode(Toks));
      end;
    end;
  finally
    S.Free;
  end;
end;

{ UT-TrainData-110: BatchTensors, empty batch / out-of-range sample. }
procedure TTrainDataTest.TestUT_TrainData_110;
var
  S: TTrainSet;
  Ids, Tg, M: TTensor;
  B: TBatch;
begin
  S := nil;
  try
    S := FixSet(100);
    try
      BatchTensors(S, nil, 7, Ids, Tg, M);
      Fail('no ETrainData');
    except
      on E: ETrainData do
        AssertEquals('empty batch', E.Message);
    end;
    B := [0, 3];
    try
      BatchTensors(S, B, 7, Ids, Tg, M);
      Fail('no ETrainData');
    except
      on E: ETrainData do
        AssertEquals('sample 3 outside 0..2', E.Message);
    end;
  finally
    S.Free;
  end;
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
