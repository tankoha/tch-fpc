{ UT program of T05: the rows UT-TokenizeCli-001.. (stage 3, T05b3: RunTokenizeCLI) of
  tests/ut/cases/TokenizeCli.cases.md.
  Expected values are those of tasks/T05-facts.md; the vocabulary and the
  small GGUF come from tests/ut/TokenizerFixture.pas (harness owner's
  fixture, no model file needed). Every test that writes files uses Dir
  (fresh per test, removed by TearDown).
  Invoked with `--all --format=plain` (see .sdlc/ut-coverage.sh).

  Skeleton by the harness owner; the test methods are written in T05b. }
program TokenizeCli_test;

{$mode objfpc}{$H+}

uses
  SysUtils, Classes, QwenTokenizer, TokenizeCli, TokenizerFixture,
  fpcunit, testregistry, consoletestrunner;

type
  TTokenizeCliTest = class(TTestCase)
  private
    FDir: string;
  protected
    procedure SetUp; override;
    procedure TearDown; override;
  public
    property Dir: string read FDir;
  published
    { T05b adds the methods TestUT_..._NNN here, one per row. }
    procedure TestUT_TokenizeCli_001;
    procedure TestUT_TokenizeCli_002;
    procedure TestUT_TokenizeCli_003;
    procedure TestUT_TokenizeCli_004;
    procedure TestUT_TokenizeCli_005;
    procedure TestUT_TokenizeCli_006;
    procedure TestUT_TokenizeCli_007;
    procedure TestUT_TokenizeCli_008;
    procedure TestUT_TokenizeCli_009;
    procedure TestUT_TokenizeCli_010;
    procedure TestUT_TokenizeCli_011;
    procedure TestUT_TokenizeCli_012;
    procedure TestUT_TokenizeCli_013;
    procedure TestUT_TokenizeCli_014;
    procedure TestUT_TokenizeCli_015;
    procedure TestUT_TokenizeCli_016;
    procedure TestUT_TokenizeCli_017;
    procedure TestUT_TokenizeCli_018;
    procedure TestUT_TokenizeCli_019;
    procedure TestUT_TokenizeCli_020;
  end;

var
  DirSeq: Integer = 0;

procedure TTokenizeCliTest.SetUp;
begin
  Inc(DirSeq);
  FDir := IncludeTrailingPathDelimiter(GetTempDir(False)) +
    Format('t05-ut-%d-%d', [GetProcessID, DirSeq]);
  ForceDirectories(FDir);
end;

procedure TTokenizeCliTest.TearDown;
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

{ --------- test implementations --------- }
procedure TTokenizeCliTest.TestUT_TokenizeCli_001;
var
  O, E: TStringList;
begin
  WriteTinyGguf(Dir + '/tiny.gguf');
  WriteBytes(Dir + '/a.txt', 'hello world<|im_end|>');
  O := TStringList.Create;
  E := TStringList.Create;
  try
    AssertEquals('exit', 0, RunTokenizeCLI([Dir + '/tiny.gguf', Dir + '/a.txt'], O, E));
    AssertEquals('out lines', 1, O.Count);
    AssertEquals('out', '259 264 267', O[0]);
    AssertEquals('err', 0, E.Count);
  finally
    O.Free;
    E.Free;
  end;
end;

procedure TTokenizeCliTest.TestUT_TokenizeCli_002;
var
  O, E: TStringList;
begin
  WriteTinyGguf(Dir + '/tiny.gguf');
  WriteBytes(Dir + '/a.txt', '');
  O := TStringList.Create;
  E := TStringList.Create;
  try
    AssertEquals('exit', 0, RunTokenizeCLI([Dir + '/tiny.gguf', Dir + '/a.txt'], O, E));
    AssertEquals('out lines', 1, O.Count);
    AssertEquals('out', '', O[0]);
    AssertEquals('err', 0, E.Count);
  finally
    O.Free;
    E.Free;
  end;
end;

procedure TTokenizeCliTest.TestUT_TokenizeCli_003;
var
  O, E: TStringList;
begin
  WriteTinyGguf(Dir + '/tiny.gguf');
  WriteBytes(Dir + '/a.txt', '{"id":"a","text":"hello"}' + #10 + '{"text":"aaa"}' + #10 + #10 + '{"id":"c","text":"<' + 'think>x"}' + #13);
  O := TStringList.Create;
  E := TStringList.Create;
  try
    AssertEquals('exit', 0, RunTokenizeCLI(['--jsonl', Dir + '/tiny.gguf', Dir + '/a.txt'], O, E));
    AssertEquals('out lines', 3, O.Count);
    AssertEquals('out1', 'a' + #9 + '259', O[0]);
    AssertEquals('out2', '2' + #9 + '265 97', O[1]);
    AssertEquals('out3', 'c' + #9 + '268 120', O[2]);
    AssertEquals('err', 0, E.Count);
  finally
    O.Free;
    E.Free;
  end;
end;

procedure TTokenizeCliTest.TestUT_TokenizeCli_004;
var
  O, E: TStringList;
begin
  WriteTinyGguf(Dir + '/tiny.gguf');
  WriteBytes(Dir + '/a.txt', '{"text":"é あ"}');
  O := TStringList.Create;
  E := TStringList.Create;
  try
    AssertEquals('exit', 0, RunTokenizeCLI(['--jsonl', Dir + '/tiny.gguf', Dir + '/a.txt'], O, E));
    AssertEquals('out lines', 1, O.Count);
    AssertEquals('out', '1' + #9 + '195 169 32 227 129 130', O[0]);
    AssertEquals('err', 0, E.Count);
  finally
    O.Free;
    E.Free;
  end;
end;

procedure TTokenizeCliTest.TestUT_TokenizeCli_005;
var
  O, E: TStringList;
begin
  WriteTinyGguf(Dir + '/tiny.gguf');
  WriteBytes(Dir + '/a.txt', '{"text":"' + #$C3#$A9 + ' ' + #$E3#$81#$82 + '"}');
  O := TStringList.Create;
  E := TStringList.Create;
  try
    AssertEquals('exit', 0, RunTokenizeCLI(['--jsonl', Dir + '/tiny.gguf', Dir + '/a.txt'], O, E));
    AssertEquals('out lines', 1, O.Count);
    AssertEquals('out', '1' + #9 + '195 169 32 227 129 130', O[0]);
    AssertEquals('err', 0, E.Count);
  finally
    O.Free;
    E.Free;
  end;
end;

procedure TTokenizeCliTest.TestUT_TokenizeCli_006;
var
  O, E: TStringList;
begin
  O := TStringList.Create;
  E := TStringList.Create;
  try
    AssertEquals('exit', 2, RunTokenizeCLI([], O, E));
    AssertEquals('out', 0, O.Count);
    AssertEquals('err lines', 1, E.Count);
    AssertEquals('err', 'Usage: qlora_tokenize [--jsonl] MODEL FILE', E[0]);
  finally
    O.Free;
    E.Free;
  end;
end;

procedure TTokenizeCliTest.TestUT_TokenizeCli_007;
var
  O, E: TStringList;
begin
  WriteTinyGguf(Dir + '/tiny.gguf');
  O := TStringList.Create;
  E := TStringList.Create;
  try
    AssertEquals('exit', 2, RunTokenizeCLI([Dir + '/tiny.gguf'], O, E));
    AssertEquals('out', 0, O.Count);
    AssertEquals('err lines', 1, E.Count);
    AssertEquals('err', 'Usage: qlora_tokenize [--jsonl] MODEL FILE', E[0]);
  finally
    O.Free;
    E.Free;
  end;
end;

procedure TTokenizeCliTest.TestUT_TokenizeCli_008;
var
  O, E: TStringList;
begin
  WriteTinyGguf(Dir + '/tiny.gguf');
  WriteBytes(Dir + '/a.txt', 'hello');
  O := TStringList.Create;
  E := TStringList.Create;
  try
    AssertEquals('exit', 2, RunTokenizeCLI([Dir + '/tiny.gguf', Dir + '/a.txt', 'x'], O, E));
    AssertEquals('out', 0, O.Count);
    AssertEquals('err lines', 1, E.Count);
    AssertEquals('err', 'Usage: qlora_tokenize [--jsonl] MODEL FILE', E[0]);
  finally
    O.Free;
    E.Free;
  end;
end;

procedure TTokenizeCliTest.TestUT_TokenizeCli_009;
var
  O, E: TStringList;
begin
  WriteTinyGguf(Dir + '/tiny.gguf');
  WriteBytes(Dir + '/a.txt', 'hello');
  O := TStringList.Create;
  E := TStringList.Create;
  try
    AssertEquals('exit', 2, RunTokenizeCLI(['--x', Dir + '/tiny.gguf', Dir + '/a.txt'], O, E));
    AssertEquals('out', 0, O.Count);
    AssertEquals('err lines', 1, E.Count);
    AssertEquals('err', 'Usage: qlora_tokenize [--jsonl] MODEL FILE', E[0]);
  finally
    O.Free;
    E.Free;
  end;
end;

procedure TTokenizeCliTest.TestUT_TokenizeCli_010;
var
  O, E: TStringList;
begin
  O := TStringList.Create;
  E := TStringList.Create;
  try
    AssertEquals('exit', 1, RunTokenizeCLI([Dir + '/missing.gguf', Dir + '/a.txt'], O, E));
    AssertEquals('out', 0, O.Count);
    AssertEquals('err lines', 1, E.Count);
    AssertTrue('err prefix', Pos('qlora_tokenize: ', E[0]) = 1);
    AssertTrue('err path', Pos(Dir + '/missing.gguf', E[0]) > 0);
  finally
    O.Free;
    E.Free;
  end;
end;

procedure TTokenizeCliTest.TestUT_TokenizeCli_011;
var
  O, E: TStringList;
begin
  WriteTinyGguf(Dir + '/tiny.gguf');
  O := TStringList.Create;
  E := TStringList.Create;
  try
    AssertEquals('exit', 1, RunTokenizeCLI([Dir + '/tiny.gguf', Dir + '/missing.txt'], O, E));
    AssertEquals('out', 0, O.Count);
    AssertEquals('err lines', 1, E.Count);
    AssertTrue('err path', Pos(Dir + '/missing.txt', E[0]) > 0);
  finally
    O.Free;
    E.Free;
  end;
end;

procedure TTokenizeCliTest.TestUT_TokenizeCli_012;
var
  O, E: TStringList;
begin
  WriteTinyGguf(Dir + '/m.gguf', 'llama');
  WriteBytes(Dir + '/a.txt', 'hello');
  O := TStringList.Create;
  E := TStringList.Create;
  try
    AssertEquals('exit', 1, RunTokenizeCLI([Dir + '/m.gguf', Dir + '/a.txt'], O, E));
    AssertEquals('out', 0, O.Count);
    AssertEquals('err lines', 1, E.Count);
    AssertEquals('err', 'qlora_tokenize: tokenizer: model llama, expected gpt2', E[0]);
  finally
    O.Free;
    E.Free;
  end;
end;

procedure TTokenizeCliTest.TestUT_TokenizeCli_013;
var
  O, E: TStringList;
begin
  WriteTinyGguf(Dir + '/m.gguf', 'gpt2', 'llama3');
  WriteBytes(Dir + '/a.txt', 'hello');
  O := TStringList.Create;
  E := TStringList.Create;
  try
    AssertEquals('exit', 1, RunTokenizeCLI([Dir + '/m.gguf', Dir + '/a.txt'], O, E));
    AssertEquals('out', 0, O.Count);
    AssertEquals('err lines', 1, E.Count);
    AssertEquals('err', 'qlora_tokenize: tokenizer: pre llama3, expected qwen35', E[0]);
  finally
    O.Free;
    E.Free;
  end;
end;

procedure TTokenizeCliTest.TestUT_TokenizeCli_014;
var
  O, E: TStringList;
begin
  WriteTinyGguf(Dir + '/m.gguf', 'gpt2', 'qwen35', 'tokenizer.ggml.merges');
  WriteBytes(Dir + '/a.txt', 'hello');
  O := TStringList.Create;
  E := TStringList.Create;
  try
    AssertEquals('exit', 1, RunTokenizeCLI([Dir + '/m.gguf', Dir + '/a.txt'], O, E));
    AssertEquals('out', 0, O.Count);
    AssertEquals('err lines', 1, E.Count);
    AssertEquals('err', 'qlora_tokenize: tokenizer: key tokenizer.ggml.merges missing', E[0]);
  finally
    O.Free;
    E.Free;
  end;
end;

procedure TTokenizeCliTest.TestUT_TokenizeCli_015;
var
  O, E: TStringList;
begin
  WriteTinyGguf(Dir + '/tiny.gguf');
  WriteBytes(Dir + '/a.txt', '{"text":"x"}' + #10 + 'not json');
  O := TStringList.Create;
  E := TStringList.Create;
  try
    AssertEquals('exit', 1, RunTokenizeCLI(['--jsonl', Dir + '/tiny.gguf', Dir + '/a.txt'], O, E));
    AssertEquals('out lines', 1, O.Count);
    AssertEquals('out', '1' + #9 + '120', O[0]);
    AssertEquals('err lines', 1, E.Count);
    AssertTrue('err prefix', Pos('qlora_tokenize: line 2: not JSON (', E[0]) = 1);
  finally
    O.Free;
    E.Free;
  end;
end;

procedure TTokenizeCliTest.TestUT_TokenizeCli_016;
var
  O, E: TStringList;
begin
  WriteTinyGguf(Dir + '/tiny.gguf');
  WriteBytes(Dir + '/a.txt', '{"text": 5}');
  O := TStringList.Create;
  E := TStringList.Create;
  try
    AssertEquals('exit', 1, RunTokenizeCLI(['--jsonl', Dir + '/tiny.gguf', Dir + '/a.txt'], O, E));
    AssertEquals('out', 0, O.Count);
    AssertEquals('err lines', 1, E.Count);
    AssertEquals('err', 'qlora_tokenize: line 1: no string "text"', E[0]);
  finally
    O.Free;
    E.Free;
  end;
end;

procedure TTokenizeCliTest.TestUT_TokenizeCli_017;
var
  O, E: TStringList;
begin
  WriteTinyGguf(Dir + '/tiny.gguf');
  WriteBytes(Dir + '/a.txt', '["text"]');
  O := TStringList.Create;
  E := TStringList.Create;
  try
    AssertEquals('exit', 1, RunTokenizeCLI(['--jsonl', Dir + '/tiny.gguf', Dir + '/a.txt'], O, E));
    AssertEquals('out', 0, O.Count);
    AssertEquals('err lines', 1, E.Count);
    AssertEquals('err', 'qlora_tokenize: line 1: no string "text"', E[0]);
  finally
    O.Free;
    E.Free;
  end;
end;

procedure TTokenizeCliTest.TestUT_TokenizeCli_018;
var
  O, E: TStringList;
begin
  WriteTinyGguf(Dir + '/tiny.gguf');
  WriteBytes(Dir + '/a.txt', 'ok' + #$FF);
  O := TStringList.Create;
  E := TStringList.Create;
  try
    AssertEquals('exit', 1, RunTokenizeCLI([Dir + '/tiny.gguf', Dir + '/a.txt'], O, E));
    AssertEquals('out', 0, O.Count);
    AssertEquals('err lines', 1, E.Count);
    AssertEquals('err', 'qlora_tokenize: invalid UTF-8 at byte 3', E[0]);
  finally
    O.Free;
    E.Free;
  end;
end;

procedure TTokenizeCliTest.TestUT_TokenizeCli_019;
var
  O, E: TStringList;
  OldDir: string;
begin
  WriteTinyGguf(Dir + '/tiny.gguf');
  WriteBytes(Dir + '/a.txt', 'hello world<|im_end|>');
  O := TStringList.Create;
  E := TStringList.Create;
  OldDir := GetCurrentDir;
  try
    SetCurrentDir(Dir);
    AssertEquals('exit', 0, RunTokenizeCLI(['tiny.gguf', 'a.txt'], O, E));
    AssertEquals('out lines', 1, O.Count);
    AssertEquals('out', '259 264 267', O[0]);
    AssertEquals('err', 0, E.Count);
  finally
    SetCurrentDir(OldDir);
    O.Free;
    E.Free;
  end;
end;

procedure TTokenizeCliTest.TestUT_TokenizeCli_020;
var
  O, E: TStringList;
begin
  WriteTinyGguf(Dir + '/tiny.gguf');
  WriteBytes(Dir + '/a.txt', 'hello');
  O := TStringList.Create;
  E := TStringList.Create;
  try
    AssertEquals('exit', 2, RunTokenizeCLI(['-x', Dir + '/a.txt'], O, E));
    AssertEquals('out', 0, O.Count);
    AssertEquals('err lines', 1, E.Count);
    AssertEquals('err', 'Usage: qlora_tokenize [--jsonl] MODEL FILE', E[0]);
  finally
    O.Free;
    E.Free;
  end;
end;

var
  App: TTestRunner;
begin
  RegisterTest(TTokenizeCliTest);
  App := TTestRunner.Create(nil);
  App.Initialize;
  App.Title := 'TokenizeCli UT';
  App.Run;
  App.Free;
end.
