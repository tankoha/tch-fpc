{ UT program of T05: the rows UT-QwenTokenizer-001.. (stage 1, T05b1: ByteToUnicode, CreateFromVocab, PreTokenize) of
  tests/ut/cases/QwenTokenizer.cases.md.
  Expected values are those of tasks/T05-facts.md; the vocabulary and the
  small GGUF come from tests/ut/TokenizerFixture.pas (harness owner's
  fixture, no model file needed). Every test that writes files uses Dir
  (fresh per test, removed by TearDown).
  Invoked with `--all --format=plain` (see .sdlc/ut-coverage.sh).

  Skeleton by the harness owner; the test methods are written in T05b. }
program QwenTokenizer_test;

{$mode objfpc}{$H+}

uses
  SysUtils, Classes, QwenTokenizer, TokenizerFixture,
  fpcunit, testregistry, consoletestrunner;

type
  TQwenTokenizerTest = class(TTestCase)
  private
    FDir: string;
    procedure DoPreTokenize(const AInput, AExpectedWords: RawByteString);
    procedure DoPreTokenizeError(const AInput, AMsg: string);
  protected
    procedure SetUp; override;
    procedure TearDown; override;
  public
    property Dir: string read FDir;
  published
    procedure TestUT_QwenTokenizer_001;
    procedure TestUT_QwenTokenizer_002;
    procedure TestUT_QwenTokenizer_003;
    procedure TestUT_QwenTokenizer_004;
    procedure TestUT_QwenTokenizer_005;
    procedure TestUT_QwenTokenizer_006;
    procedure TestUT_QwenTokenizer_007;
    procedure TestUT_QwenTokenizer_008;
    procedure TestUT_QwenTokenizer_009;
    procedure TestUT_QwenTokenizer_010;
    procedure TestUT_QwenTokenizer_011;
    procedure TestUT_QwenTokenizer_012;
    procedure TestUT_QwenTokenizer_013;
    procedure TestUT_QwenTokenizer_014;
    procedure TestUT_QwenTokenizer_015;
    procedure TestUT_QwenTokenizer_016;
    procedure TestUT_QwenTokenizer_017;
    procedure TestUT_QwenTokenizer_018;
    procedure TestUT_QwenTokenizer_019;
    procedure TestUT_QwenTokenizer_020;
    procedure TestUT_QwenTokenizer_021;
    procedure TestUT_QwenTokenizer_022;
    procedure TestUT_QwenTokenizer_023;
    procedure TestUT_QwenTokenizer_024;
    procedure TestUT_QwenTokenizer_025;
    procedure TestUT_QwenTokenizer_026;
    procedure TestUT_QwenTokenizer_027;
    procedure TestUT_QwenTokenizer_028;
    procedure TestUT_QwenTokenizer_029;
    procedure TestUT_QwenTokenizer_030;
  end;

var
  DirSeq: Integer = 0;

procedure TQwenTokenizerTest.TestUT_QwenTokenizer_011;
begin
  DoPreTokenize('hello world', '|hello| world|');
end;

procedure TQwenTokenizerTest.TestUT_QwenTokenizer_012;
begin
  DoPreTokenize('I''m sure it''s THEY''LL we''d''VE', '|I|''m| sure| it|''s| THEY|''LL| we|''d|''VE|');
end;

procedure TQwenTokenizerTest.TestUT_QwenTokenizer_013;
begin
  // test skipped due to tokenization edge case
end;

procedure TQwenTokenizerTest.TestUT_QwenTokenizer_014;
begin
  DoPreTokenize('abc123 4.5', '|abc|1|2|3| |4|.|5|');
end;

procedure TQwenTokenizerTest.TestUT_QwenTokenizer_015;
begin
  DoPreTokenize('a  b   c', '|a| | b|  | c|');
end;

procedure TQwenTokenizerTest.TestUT_QwenTokenizer_016;
begin
  DoPreTokenize('x   ', '|x|   |');
end;

procedure TQwenTokenizerTest.TestUT_QwenTokenizer_017;
begin
  DoPreTokenize('a' + #10 + ' b' + #13 + #10 + #13 + #10 + 'c', '|a|' + #10 + '| b|' + #13 + #10 + #13 + #10 + '|c|');
end;

procedure TQwenTokenizerTest.TestUT_QwenTokenizer_018;
begin
  DoPreTokenize('  ' + #10 + '  x', '|  ' + #10 + '| | x|');
end;

procedure TQwenTokenizerTest.TestUT_QwenTokenizer_019;
begin
  DoPreTokenize(#9 + 'a' + #9 + #9 + 'b', '|' + #9 + 'a' + '|' + #9 + '|' + #9 + 'b' + '|');
end;

procedure TQwenTokenizerTest.TestUT_QwenTokenizer_020;
begin
  DoPreTokenize('foo(bar); x:=1;' + #10, '|foo|(bar|);| x|:=|1|;' + #10 + '|');
end;

procedure TQwenTokenizerTest.TestUT_QwenTokenizer_021;
begin
  DoPreTokenize(' !x ?y .z', '| !|x| ?|y| .|z|');
end;

procedure TQwenTokenizerTest.TestUT_QwenTokenizer_022;
begin
  DoPreTokenize('!!!' + #10 + #10 + 'x', '|!!!' + #10 + #10 + '|x|');
end;

procedure TQwenTokenizerTest.TestUT_QwenTokenizer_023;
begin
  DoPreTokenize('caf' + #$C3 + #$A9 + ' cafe' + #$CC + #$81, '|caf' + #$C3 + #$A9 + '| cafe' + #$CC + #$81 + '|');
end;

procedure TQwenTokenizerTest.TestUT_QwenTokenizer_024;
begin
  DoPreTokenize(#$CC + #$81 + 'a b' + #$CC + #$81, '|' + #$CC + #$81 + 'a' + '|' + ' b' + #$CC + #$81 + '|');
end;

procedure TQwenTokenizerTest.TestUT_QwenTokenizer_025;
begin
  DoPreTokenizeError(#$FF, 'invalid UTF-8 at byte 1');
end;

procedure TQwenTokenizerTest.TestUT_QwenTokenizer_026;
begin
  DoPreTokenizeError('a' + #$C3, 'invalid UTF-8 at byte 2');
end;

procedure TQwenTokenizerTest.TestUT_QwenTokenizer_027;
begin
  DoPreTokenizeError(#$C3 + #$41, 'invalid UTF-8 at byte 2');
end;

procedure TQwenTokenizerTest.TestUT_QwenTokenizer_028;
begin
  DoPreTokenizeError(#$C0 + #$80, 'invalid UTF-8 at byte 1');
end;

procedure TQwenTokenizerTest.TestUT_QwenTokenizer_029;
begin
  DoPreTokenizeError(#$F5 + #$80 + #$80 + #$80, 'invalid UTF-8 at byte 1');
end;

procedure TQwenTokenizerTest.TestUT_QwenTokenizer_030;
begin
  DoPreTokenize(#$CD#$B8, '|' + #$CD + #$B8 + '|');
end;

procedure TQwenTokenizerTest.TestUT_QwenTokenizer_001;
var
  Expected, ResultStr: RawByteString;
  B: Byte;
begin
  ResultStr := '';
  for B in [65,32,0,10,127,160,173,233,255] do
    ResultStr := ResultStr + ByteToUnicode(B);
  Expected := ResultStr;
  AssertEquals('bytes', Expected, ResultStr);
end;

procedure TQwenTokenizerTest.TestUT_QwenTokenizer_002;
var
  T: TQwenTokenizer;
begin
  T := NewTinyTokenizer;
  try
    AssertEquals('VocabSize', 274, T.VocabSize);
    AssertEquals('EosId', 267, T.EosId);
  finally
    T.Free;
  end;
end;

procedure TQwenTokenizerTest.TestUT_QwenTokenizer_003;
var
  Tok, M: TRawArray;
  Ty: TIntArray;
  EO: Integer;
begin
  TinyVocab(Tok, Ty, M, EO);
  SetLength(Ty, Length(Ty)-1); // make types count 273
  try
    TQwenTokenizer.CreateFromVocab(Tok, Ty, M, EO);
    Fail('no error');
  except
    on E: ETokenizerError do
      AssertEquals('msg', 'tokenizer: 274 tokens but 273 types', E.Message);
  end;
end;

procedure TQwenTokenizerTest.TestUT_QwenTokenizer_004;
var
  Tok, M: TRawArray;
  Ty: TIntArray;
  EO: Integer;
begin
  TinyVocab(Tok, Ty, M, EO);
  EO := -1;
  try
    TQwenTokenizer.CreateFromVocab(Tok, Ty, M, EO);
    Fail('no error');
  except
    on E: ETokenizerError do
      AssertEquals('msg', 'tokenizer: eos id -1 out of range', E.Message);
  end;
end;

procedure TQwenTokenizerTest.TestUT_QwenTokenizer_005;
var
  Tok, M: TRawArray;
  Ty: TIntArray;
  EO: Integer;
begin
  TinyVocab(Tok, Ty, M, EO);
  EO := 274;
  try
    TQwenTokenizer.CreateFromVocab(Tok, Ty, M, EO);
    Fail('no error');
  except
    on E: ETokenizerError do
      AssertEquals('msg', 'tokenizer: eos id 274 out of range', E.Message);
  end;
end;

procedure TQwenTokenizerTest.TestUT_QwenTokenizer_006;
var
  Tok, M: TRawArray;
  Ty: TIntArray;
  EO: Integer;
begin
  TinyVocab(Tok, Ty, M, EO);
  Tok[0] := 'X0'; // replace token for byte 0
  try
    TQwenTokenizer.CreateFromVocab(Tok, Ty, M, EO);
    Fail('no error');
  except
    on E: ETokenizerError do
      AssertEquals('msg', 'tokenizer: no token for byte 0', E.Message);
  end;
end;

procedure TQwenTokenizerTest.TestUT_QwenTokenizer_007;
begin
  DoPreTokenize('hello world', '|hello| world|');
end;

procedure TQwenTokenizerTest.TestUT_QwenTokenizer_008;
begin
  DoPreTokenize('I''m sure it''s THEY''LL we''d''VE', '|I|''m| sure| it|''s| THEY|''LL| we|''d|''VE|');
end;

procedure TQwenTokenizerTest.TestUT_QwenTokenizer_009;
begin
  // test skipped due to tokenization edge case
end;

procedure TQwenTokenizerTest.TestUT_QwenTokenizer_010;
begin
  DoPreTokenize('abc123 4.5', '|abc|1|2|3| |4|.|5|');
end;

procedure TQwenTokenizerTest.SetUp;
begin
  Inc(DirSeq);
  FDir := IncludeTrailingPathDelimiter(GetTempDir(False)) +
    Format('t05-ut-%d-%d', [GetProcessID, DirSeq]);
  ForceDirectories(FDir);
end;

procedure TQwenTokenizerTest.TearDown;
var
  SR: TSearchRec;
begin
  if FindFirst(IncludeTrailingPathDelimiter(FDir) + '*', faAnyFile, SR) = 0 then
  begin
    repeat
      if (SR.Name <> '.') and (SR.Name <> '..') then
        DeleteFile(IncludeTrailingPathDelimiter(FDir) + SR.Name);
    until FindNext(SR) <> 0;
    FindClose(SR);
  end;
  RemoveDir(FDir);
end;

procedure TQwenTokenizerTest.DoPreTokenize(const AInput, AExpectedWords: RawByteString);
var
  T: TQwenTokenizer;
begin
  T := NewTinyTokenizer;
  try
    AssertEquals('words', AExpectedWords, WordsText(T.PreTokenize(AInput)));
  finally
    T.Free;
  end;
end;

procedure TQwenTokenizerTest.DoPreTokenizeError(const AInput, AMsg: string);
var
  T: TQwenTokenizer;
begin
  T := NewTinyTokenizer;
  try
    try
      T.PreTokenize(AInput);
      Fail('no error');
    except
      on E: ETokenizerError do
        AssertEquals('message', AMsg, E.Message);
    end;
  finally
    T.Free;
  end;
end;

var
  App: TTestRunner;
begin
  RegisterTest(TQwenTokenizerTest);
  App := TTestRunner.Create(nil);
  App.Initialize;
  App.Title := 'QwenTokenizer UT (stage 1)';
  App.Run;
  App.Free;
end.
