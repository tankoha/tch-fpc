{ UT program of T05: the rows UT-QwenTokenizer-101.. (stage 2, T05b2: Encode, Decode, SpecialId, Create) of
  tests/ut/cases/QwenTokenizer.cases.md.
  Expected values are those of tasks/T05-facts.md; the vocabulary and the
  small GGUF come from tests/ut/TokenizerFixture.pas (harness owner's
  fixture, no model file needed). Every test that writes files uses Dir
  (fresh per test, removed by TearDown).
  Invoked with `--all --format=plain` (see .sdlc/ut-coverage.sh).

  Skeleton by the harness owner; the test methods are written in T05b. }
program QwenTokenizerEnc_test;

{$mode objfpc}{$H+}

uses
  SysUtils, Classes, GgufFile, QwenTokenizer, TokenizerFixture,
  fpcunit, testregistry, consoletestrunner;

type
  TQwenTokenizerEncTest = class(TTestCase)
  private
    FDir: string;
  protected
    procedure SetUp; override;
    procedure TearDown; override;
  public
    property Dir: string read FDir;
  published
    { T05b adds the methods TestUT_..._NNN here, one per row. }
    procedure TestUT_QwenTokenizer_101;
    procedure TestUT_QwenTokenizer_102;
    procedure TestUT_QwenTokenizer_103;
    procedure TestUT_QwenTokenizer_104;
    procedure TestUT_QwenTokenizer_105;
    procedure TestUT_QwenTokenizer_106;
    procedure TestUT_QwenTokenizer_107;
    procedure TestUT_QwenTokenizer_108;
    procedure TestUT_QwenTokenizer_109;
    procedure TestUT_QwenTokenizer_110;
    procedure TestUT_QwenTokenizer_111;
    procedure TestUT_QwenTokenizer_112;
    procedure TestUT_QwenTokenizer_113;
    procedure TestUT_QwenTokenizer_114;
    procedure TestUT_QwenTokenizer_115;
    procedure TestUT_QwenTokenizer_116;
    procedure TestUT_QwenTokenizer_117;
    procedure TestUT_QwenTokenizer_118;
    procedure TestUT_QwenTokenizer_119;
    procedure TestUT_QwenTokenizer_120;
    procedure TestUT_QwenTokenizer_121;
    procedure TestUT_QwenTokenizer_122;
    procedure TestUT_QwenTokenizer_123;
    procedure TestUT_QwenTokenizer_124;
    procedure TestUT_QwenTokenizer_125;
    procedure TestUT_QwenTokenizer_126;
    procedure TestUT_QwenTokenizer_127;
    procedure TestUT_QwenTokenizer_128;
    procedure TestUT_QwenTokenizer_129;
    procedure TestUT_QwenTokenizer_130;
    procedure TestUT_QwenTokenizer_131;
    procedure TestUT_QwenTokenizer_132;
  end;

procedure DoEncodeInvalid;
begin
  NewTinyTokenizer.Encode('x' + #255);
end;

procedure DoDecode274;
begin
  NewTinyTokenizer.Decode([274]);
end;

procedure DoDecodeNeg1;
begin
  NewTinyTokenizer.Decode([-1]);
end;

procedure DoDecodeCharNotInByteMap;
begin
  NewTinyTokenizer.Decode([274]);
end;

var
  DirSeq: Integer = 0;

procedure TQwenTokenizerEncTest.SetUp;
begin
  Inc(DirSeq);
  FDir := IncludeTrailingPathDelimiter(GetTempDir(False)) +
    Format('t05-ut-%d-%d', [GetProcessID, DirSeq]);
  ForceDirectories(FDir);
end;

procedure TQwenTokenizerEncTest.TearDown;
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

{ Test methods for rows 101..127 }
procedure TQwenTokenizerEncTest.TestUT_QwenTokenizer_101;
var
  Tok: TTokenArray;
begin
  Tok := NewTinyTokenizer.Encode('hello world');
  CheckEquals('259 264', IdsText(Tok), 'Encode hello world');
end;

procedure TQwenTokenizerEncTest.TestUT_QwenTokenizer_102;
var
  Tok: TTokenArray;
begin
  Tok := NewTinyTokenizer.Encode('abc');
  CheckEquals('97 272', IdsText(Tok), 'Encode abc');
end;

procedure TQwenTokenizerEncTest.TestUT_QwenTokenizer_103;
var
  Tok: TTokenArray;
begin
  Tok := NewTinyTokenizer.Encode('aaa');
  CheckEquals('265 97', IdsText(Tok), 'Encode aaa');
end;

procedure TQwenTokenizerEncTest.TestUT_QwenTokenizer_104;
var
  Tok: TTokenArray;
begin
  Tok := NewTinyTokenizer.Encode('aaaa');
  CheckEquals('265 265', IdsText(Tok), 'Encode aaaa');
end;

procedure TQwenTokenizerEncTest.TestUT_QwenTokenizer_105;
var
  Tok: TTokenArray;
begin
  Tok := NewTinyTokenizer.Encode('<|im_start|>hello<|im_end|>');
  CheckEquals('266 259 267', IdsText(Tok), 'Encode with specials');
end;

procedure TQwenTokenizerEncTest.TestUT_QwenTokenizer_106;
var
  Tok: TTokenArray;
begin
  Tok := NewTinyTokenizer.Encode('<tts_text_bos_single>x<tts_text_bos>');
  CheckEquals('270 120 269', IdsText(Tok), 'Encode tts specials');
end;

procedure TQwenTokenizerEncTest.TestUT_QwenTokenizer_107;
var
  Tok: TTokenArray;
begin
  Tok := NewTinyTokenizer.Encode('<think><think>');
  CheckEquals('268 268', IdsText(Tok), 'Encode double think');
end;

procedure TQwenTokenizerEncTest.TestUT_QwenTokenizer_108;
var
  Tok: TTokenArray;
begin
  Tok := NewTinyTokenizer.Encode('<|im_start');
  // Expect token ids for each character
  CheckEquals('60 124 105 109 95 115 116 97 114 116', IdsText(Tok), 'Encode broken special');
end;

procedure TQwenTokenizerEncTest.TestUT_QwenTokenizer_109;
var
  Tok: TTokenArray;
begin
  Tok := NewTinyTokenizer.Encode('<|unused|>');
  CheckEquals('60 124 117 110 117 115 101 100 124 62', IdsText(Tok), 'Encode unused special');
end;

procedure TQwenTokenizerEncTest.TestUT_QwenTokenizer_110;
var
  Tok: TTokenArray;
begin
  Tok := NewTinyTokenizer.Encode(' ');
  CheckEquals('32', IdsText(Tok), 'Encode space');
end;

procedure TQwenTokenizerEncTest.TestUT_QwenTokenizer_111;
var
  Tok: TTokenArray;
begin
  Tok := NewTinyTokenizer.Encode('');
  CheckEquals('', IdsText(Tok), 'Encode empty');
end;

procedure TQwenTokenizerEncTest.TestUT_QwenTokenizer_112;
var
  Tok: TTokenArray;
begin
  Tok := NewTinyTokenizer.Encode(#$C3#$A9 + ' ' + #$E3#$81#$82);
  CheckEquals('195 169 32 227 129 130', IdsText(Tok), 'Encode UTF-8 chars');
end;

procedure TQwenTokenizerEncTest.TestUT_QwenTokenizer_113;
begin
  try
    NewTinyTokenizer.Encode('x' + #255);
    Fail('Expected exception not raised');
  except
    on E: ETokenizerError do ;
  end;
end;

procedure TQwenTokenizerEncTest.TestUT_QwenTokenizer_114;
var
  Tok: RawByteString;
begin
  Tok := NewTinyTokenizer.Decode([266, 259, 264, 267]);
  CheckEquals('<|im_start|>hello world<|im_end|>', Tok, 'Decode special+text+special');
end;

procedure TQwenTokenizerEncTest.TestUT_QwenTokenizer_115;
var
  Tok: RawByteString;
begin
  Tok := NewTinyTokenizer.Decode([195, 169]);
  CheckEquals(#$C3#$A9, Tok, 'Decode bytes');
end;

procedure TQwenTokenizerEncTest.TestUT_QwenTokenizer_116;
var
  Tok: RawByteString;
begin
  Tok := NewTinyTokenizer.Decode([]);
  CheckEquals('', Tok, 'Decode empty');
end;

procedure TQwenTokenizerEncTest.TestUT_QwenTokenizer_117;
var
  Tok: RawByteString;
begin
  Tok := NewTinyTokenizer.Decode([273]);
  CheckEquals('<|unused|>', Tok, 'Decode unused token');
end;

procedure TQwenTokenizerEncTest.TestUT_QwenTokenizer_118;
begin
  try
    NewTinyTokenizer.Decode([274]);
    Fail('Expected exception not raised');
  except
    on E: ETokenizerError do ;
  end;
end;

procedure TQwenTokenizerEncTest.TestUT_QwenTokenizer_119;
begin
  try
    NewTinyTokenizer.Decode([-1]);
    Fail('Expected exception not raised');
  except
    on E: ETokenizerError do ;
  end;
end;

procedure TQwenTokenizerEncTest.TestUT_QwenTokenizer_120;
var
  T: TRawArray;
  Ty: TIntArray;
  M: TRawArray;
  E: Integer;
  Tok: TQwenTokenizer;
begin
  TinyVocab(T, Ty, M, E);
  SetLength(T, 275); SetLength(Ty, 275);
  T[274] := #$E6#$97#$A5; { '日' - 3-byte UTF-8 }
  Ty[274] := 1;
  Tok := TQwenTokenizer.CreateFromVocab(T, Ty, M, E);
  try
    try
      Tok.Decode([274]);
      Fail('Expected exception not raised');
    except
      on E: ETokenizerError do CheckEquals('token 274: character not in the byte map', E.Message, 'message');
    end;
  finally
    Tok.Free;
  end;
end;

procedure TQwenTokenizerEncTest.TestUT_QwenTokenizer_121;
var
  Tok: TTokenArray;
  Input: RawByteString;
begin
  for Input in ['I', 'I\n', 'I\r\n', 'I\r\n\r\n', 'I\n\n', 'I\n\r'] do
  begin
    Tok := NewTinyTokenizer.Encode(Input);
    CheckEquals(NewTinyTokenizer.Decode(Tok), string(Input), 'Roundtrip '+Input);
  end;
end;

procedure TQwenTokenizerEncTest.TestUT_QwenTokenizer_122;
var
  Id1, Id2, Id3, Id4: Integer;
begin
  Id1 := NewTinyTokenizer.SpecialId('<think>');
  Id2 := NewTinyTokenizer.SpecialId('<|im_end|>');
  Id3 := NewTinyTokenizer.SpecialId('hello');
  Id4 := NewTinyTokenizer.SpecialId('<|unused|>');
  CheckEquals(268, Id1, 'SpecialId think');
  CheckEquals(267, Id2, 'SpecialId im_end');
  CheckEquals(-1, Id3, 'SpecialId non-special');
  CheckEquals(-1, Id4, 'SpecialId non-special');
end;

procedure TQwenTokenizerEncTest.TestUT_QwenTokenizer_123;
var
  Tok: TQwenTokenizer;
  Ids: TTokenArray;
begin
  WriteTinyGguf(Dir + '/t.gguf');
  Tok := TQwenTokenizer.Create(TGgufFile.Open(Dir + '/t.gguf'));
  CheckEquals(274, Tok.VocabSize, 'VocabSize after Create');
  CheckEquals(267, Tok.EosId, 'EosId after Create');
  Ids := Tok.Encode('hello world');
  CheckEquals('259 264', IdsText(Ids), 'Encode after Create');
  Tok.Free;
end;

procedure TQwenTokenizerEncTest.TestUT_QwenTokenizer_124;
begin
  WriteTinyGguf(Dir + '/t.gguf', 'llama', 'qwen35');
  try
    TQwenTokenizer.Create(TGgufFile.Open(Dir + '/t.gguf'));
    Fail('Expected exception not raised');
  except
    on E: ETokenizerError do ;
  end;
end;

procedure TQwenTokenizerEncTest.TestUT_QwenTokenizer_125;
begin
  WriteTinyGguf(Dir + '/t.gguf', 'gpt2', 'llama3');
  try
    TQwenTokenizer.Create(TGgufFile.Open(Dir + '/t.gguf'));
    Fail('Expected exception not raised');
  except
    on E: ETokenizerError do ;
  end;
end;

procedure TQwenTokenizerEncTest.TestUT_QwenTokenizer_126;
begin
  WriteTinyGguf(Dir + '/t.gguf', 'gpt2', 'qwen35', 'tokenizer.ggml.merges');
  try
    TQwenTokenizer.Create(TGgufFile.Open(Dir + '/t.gguf'));
    Fail('Expected exception not raised');
  except
    on E: ETokenizerError do ;
  end;
end;

procedure TQwenTokenizerEncTest.TestUT_QwenTokenizer_127;
begin
  WriteTinyGguf(Dir + '/t.gguf', 'gpt2', 'qwen35', 'tokenizer.ggml.eos_token_id');
  try
    TQwenTokenizer.Create(TGgufFile.Open(Dir + '/t.gguf'));
    Fail('Expected exception not raised');
  except
    on E: ETokenizerError do ;
  end;
end;

procedure TQwenTokenizerEncTest.TestUT_QwenTokenizer_128;
begin
  WriteTinyGguf(Dir + '/t.gguf', 'gpt2', 'qwen35', 'tokenizer.ggml.tokens');
  try
    TQwenTokenizer.Create(TGgufFile.Open(Dir + '/t.gguf'));
    Fail('Expected exception not raised');
  except
    on E: ETokenizerError do ;
  end;
end;

procedure TQwenTokenizerEncTest.TestUT_QwenTokenizer_129;
begin
  WriteTinyGguf(Dir + '/t.gguf', 'gpt2', 'qwen35', 'tokenizer.ggml.token_type');
  try
    TQwenTokenizer.Create(TGgufFile.Open(Dir + '/t.gguf'));
    Fail('Expected exception not raised');
  except
    on E: ETokenizerError do ;
  end;
end;

procedure TQwenTokenizerEncTest.TestUT_QwenTokenizer_130;
begin
  WriteTinyGguf(Dir + '/t.gguf', 'gpt2', 'qwen35', 'tokenizer.ggml.model');
  try
    TQwenTokenizer.Create(TGgufFile.Open(Dir + '/t.gguf'));
    Fail('Expected exception not raised');
  except
    on E: ETokenizerError do ;
  end;
end;

procedure TQwenTokenizerEncTest.TestUT_QwenTokenizer_131;
begin
  WriteTinyGguf(Dir + '/t.gguf', 'gpt2', 'qwen35', 'tokenizer.ggml.pre');
  try
    TQwenTokenizer.Create(TGgufFile.Open(Dir + '/t.gguf'));
    Fail('Expected exception not raised');
  except
    on E: ETokenizerError do ;
  end;
end;

procedure TQwenTokenizerEncTest.TestUT_QwenTokenizer_132;
var
  T: TRawArray;
  Ty: TIntArray;
  M: TRawArray;
  E: Integer;
  Tok: TQwenTokenizer;
begin
  TinyVocab(T, Ty, M, E);
  SetLength(T, 275); SetLength(Ty, 275);
  T[274] := #$F0#$9F#$98#$80; { 😀 - 4-byte UTF-8 }
  Ty[274] := 1;
  Tok := TQwenTokenizer.CreateFromVocab(T, Ty, M, E);
  try
    try
      Tok.Decode([274]);
      Fail('Expected exception not raised');
    except
      on E: ETokenizerError do CheckEquals('token 274: character not in the byte map', E.Message, 'message');
    end;
  finally
    Tok.Free;
  end;
end;

var
  App: TTestRunner;
begin
  RegisterTest(TQwenTokenizerEncTest);
  App := TTestRunner.Create(nil);
  App.Initialize;
  App.Title := 'QwenTokenizer UT (stage 2)';
  App.Run;
  App.Free;
end.
