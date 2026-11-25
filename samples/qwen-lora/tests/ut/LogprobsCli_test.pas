{ UT program for src/LogprobsCli.pas: the rows UT-LogprobsCli-NNN of
  tests/ut/cases/LogprobsCli.cases.md (stage 3 from 201). The tiny qwen35
  model with a tied head and the T05 UT tokenizer
  (WriteTinyQwenModel(.., thTied, False) of tests/ut/QwenBlocksFixture.pas)
  is written once at start (TinyPath), with the prompt file PromptPath.
  Runs on the CPU (.sdlc/ut-coverage.sh hides the GPUs).
  Invoked with `--all --format=plain` (see .sdlc/ut-coverage.sh).

  Skeleton by the harness owner; the test methods are written in T11b. 
  (T21 from 301: --compute; the methods are written in T21b.)}
program LogprobsCli_test;

{$mode objfpc}{$H+}

uses
  Classes, SysUtils, Tch, GgufFile, QwenTokenizer, QwenModel, LogprobsCli,
  TokenizerFixture, QwenBlocksFixture,
  fpcunit, testregistry, consoletestrunner;

type
  TLogprobsCliTest = class(TTestCase)
  public
    { Runs RunLogprobsCLI(AArgs); returns the exit code, fills AOut / AErr
      (created here, freed by the caller). }
    class function RunCli(const AArgs: array of string; out AOut, AErr: TStringList): Integer; static;
    { The expected output line for AId and AIds from TQwenModel itself
      (top ATop of LastLogProbs, through TopLine). }
    class function Want(const AId: string; const AIds: array of Integer; ATop: Integer): string; static;
  published
    procedure TestUT_LogprobsCli_201;
    procedure TestUT_LogprobsCli_202;
    procedure TestUT_LogprobsCli_203;
    procedure TestUT_LogprobsCli_204;
    procedure TestUT_LogprobsCli_205;
    procedure TestUT_LogprobsCli_206;
    procedure TestUT_LogprobsCli_207;
    procedure TestUT_LogprobsCli_208;
    procedure TestUT_LogprobsCli_209;
    procedure TestUT_LogprobsCli_210;
    procedure TestUT_LogprobsCli_211;
    procedure TestUT_LogprobsCli_212;
    procedure TestUT_LogprobsCli_213;
    procedure TestUT_LogprobsCli_214;
    procedure TestUT_LogprobsCli_215;
    procedure TestUT_LogprobsCli_216;
    procedure TestUT_LogprobsCli_217;
    procedure TestUT_LogprobsCli_301;
    procedure TestUT_LogprobsCli_302;
  end;

const
  Usage = 'Usage: qlora_logprobs [--top N] [--split S] [--compute K] MODEL FILE';

var
  { the tiny model and the prompt file, written before the tests }
  TinyPath, PromptPath, TmpPath: string;

class function TLogprobsCliTest.RunCli(const AArgs: array of string; out AOut, AErr: TStringList): Integer;
begin
  AOut := TStringList.Create;
  AErr := TStringList.Create;
  Result := RunLogprobsCLI(AArgs, AOut, AErr);
end;

class function TLogprobsCliTest.Want(const AId: string; const AIds: array of Integer; ATop: Integer): string;
var
  G: TGgufFile;
  M: TQwenModel;
  V, I: TTensor;
  Ids: array of Int64;
  Lp: array of Double;
  J: Integer;
begin
  G := TGgufFile.Open(TinyPath);
  try
    M := TQwenModel.Load(G, TDevice.Cpu, TDevice.Cpu, 0);
    try
      M.LastLogProbs(AIds).topk(ATop, 0, True, True, V, I);
      Ids := nil;
      Lp := nil;
      SetLength(Ids, ATop);
      SetLength(Lp, ATop);
      for J := 0 to ATop - 1 do
      begin
        Ids[J] := I.Int64Value([J]);
        Lp[J] := V.DoubleValue([J]);
      end;
      Result := TopLine(AId, Length(AIds), Ids, Lp);
    finally
      M.Free;
    end;
  finally
    G.Free;
  end;
end;

procedure WriteText(const APath, AText: string);
var
  L: TStringList;
begin
  L := TStringList.Create;
  try
    L.Text := AText;
    L.SaveToFile(APath);
  finally
    L.Free;
  end;
end;

{ usage errors: exit 2, the reason and the usage line }
procedure CheckUsage(T: TTestCase; const AArgs: array of string; const AReason: string);
var
  O, E: TStringList;
begin
  try
    T.AssertEquals(2, TLogprobsCliTest.RunCli(AArgs, O, E));
    T.AssertEquals(0, O.Count);
    T.AssertEquals(2, E.Count);
    T.AssertEquals('qlora_logprobs: ' + AReason, E[0]);
    T.AssertEquals(Usage, E[1]);
  finally
    O.Free;
    E.Free;
  end;
end;

{ runtime errors: exit 1, one message on stderr }
procedure CheckError(T: TTestCase; const AFile, AMessage: string);
var
  O, E: TStringList;
begin
  try
    T.AssertEquals(1, TLogprobsCliTest.RunCli([TinyPath, AFile], O, E));
    T.AssertEquals(1, E.Count);
    T.AssertEquals(AMessage, E[0]);
  finally
    O.Free;
    E.Free;
  end;
end;

{ how many times ASub occurs in AText }
function CountSub(AText, ASub: string): Integer;
var
  I: Integer;
begin
  Result := 0;
  I := Pos(ASub, AText);
  while I > 0 do
  begin
    Inc(Result);
    Delete(AText, 1, I + Length(ASub) - 1);
    I := Pos(ASub, AText);
  end;
end;

{ UT-LogprobsCli-201: TopLine, quoting and format (exact) }
procedure TLogprobsCliTest.TestUT_LogprobsCli_201;
var
  Ids: array of Int64;
  Lp: array of Double;
begin
  SetLength(Ids, 2);
  Ids[0] := 5;
  Ids[1] := 7;
  SetLength(Lp, 2);
  Lp[0] := -0.5;
  Lp[1] := -1.25;
  AssertEquals('{"id":"a\"b","n":3,"top":[{"id":5,"logprob":-0.5000000},{"id":7,"logprob":-1.2500000}]}',
    TopLine('a"b', 3, Ids, Lp));
end;

{ UT-LogprobsCli-202: no arguments }
procedure TLogprobsCliTest.TestUT_LogprobsCli_202;
var
  Args: array of string;
begin
  Args := nil;
  CheckUsage(Self, Args, 'MODEL and FILE expected');
end;

{ UT-LogprobsCli-203: --top without a value }
procedure TLogprobsCliTest.TestUT_LogprobsCli_203;
begin
  CheckUsage(Self, ['--top'], '--top needs a value');
end;

{ UT-LogprobsCli-204: --top 0, below range }
procedure TLogprobsCliTest.TestUT_LogprobsCli_204;
begin
  CheckUsage(Self, ['--top', '0', TinyPath, PromptPath], '--top must be 1..100, got 0');
end;

{ UT-LogprobsCli-205: --top 101, above range }
procedure TLogprobsCliTest.TestUT_LogprobsCli_205;
begin
  CheckUsage(Self, ['--top', '101', TinyPath, PromptPath], '--top must be 1..100, got 101');
end;

{ UT-LogprobsCli-206: --split x, not an integer }
procedure TLogprobsCliTest.TestUT_LogprobsCli_206;
begin
  CheckUsage(Self, ['--split', 'x', TinyPath, PromptPath], '--split must be 0..1000, got x');
end;

{ UT-LogprobsCli-207: --TOP (case) is unknown }
procedure TLogprobsCliTest.TestUT_LogprobsCli_207;
begin
  CheckUsage(Self, ['--TOP', '3', TinyPath, PromptPath], 'unknown option --TOP');
end;

{ UT-LogprobsCli-208: one positional argument }
procedure TLogprobsCliTest.TestUT_LogprobsCli_208;
begin
  CheckUsage(Self, [TinyPath], 'MODEL and FILE expected');
end;

{ UT-LogprobsCli-209: --top 3, token records }
procedure TLogprobsCliTest.TestUT_LogprobsCli_209;
var
  O, E: TStringList;
begin
  try
    AssertEquals(0, TLogprobsCliTest.RunCli(['--top', '3', TinyPath, PromptPath], O, E));
    AssertEquals(0, E.Count);
    AssertEquals(3, O.Count);
    AssertEquals(Want('a', [5, 17, 2], 3), O[0]);
    { the record without "id" gets its line number }
    AssertEquals(Want('2', [8], 3), O[1]);
  finally
    O.Free;
    E.Free;
  end;
end;

{ UT-LogprobsCli-210: text record, tokenized }
procedure TLogprobsCliTest.TestUT_LogprobsCli_210;
var
  O, E: TStringList;
  G: TGgufFile;
  Tok: TQwenTokenizer;
  Ids: array of Integer;
begin
  try
    AssertEquals(0, TLogprobsCliTest.RunCli(['--top', '3', TinyPath, PromptPath], O, E));
    G := TGgufFile.Open(TinyPath);
    try
      Tok := TQwenTokenizer.Create(G);
      try
        Ids := Tok.Encode('hello world');
        AssertEquals(Want('t', Ids, 3), O[2]);
      finally
        Tok.Free;
      end;
    finally
      G.Free;
    end;
  finally
    O.Free;
    E.Free;
  end;
end;

{ UT-LogprobsCli-211: default top (20) }
procedure TLogprobsCliTest.TestUT_LogprobsCli_211;
var
  O, E: TStringList;
  I: Integer;
begin
  try
    AssertEquals(0, TLogprobsCliTest.RunCli([TinyPath, PromptPath], O, E));
    AssertEquals(0, E.Count);
    AssertEquals(3, O.Count);
    for I := 0 to O.Count - 1 do
      AssertEquals(20, CountSub(O[I], '"logprob"'));
  finally
    O.Free;
    E.Free;
  end;
end;

{ UT-LogprobsCli-212: invalid JSON line }
procedure TLogprobsCliTest.TestUT_LogprobsCli_212;
begin
  WriteText(TmpPath, 'x');
  CheckError(Self, TmpPath, 'qlora_logprobs: line 1: not JSON (Error at line 1, Pos 1: Unexpected token (x) encountered.)');
end;

{ UT-LogprobsCli-213: JSON array, not an object }
procedure TLogprobsCliTest.TestUT_LogprobsCli_213;
begin
  WriteText(TmpPath, '[1, 2]');
  CheckError(Self, TmpPath, 'qlora_logprobs: line 1: not an object');
end;

{ UT-LogprobsCli-214: neither tokens nor text }
procedure TLogprobsCliTest.TestUT_LogprobsCli_214;
begin
  WriteText(TmpPath, '{"id": "x"}');
  CheckError(Self, TmpPath, 'qlora_logprobs: line 1: needs "tokens" (array) or "text" (string)');
end;

{ UT-LogprobsCli-215: missing model file }
procedure TLogprobsCliTest.TestUT_LogprobsCli_215;
var
  O, E: TStringList;
begin
  try
    AssertEquals(1, TLogprobsCliTest.RunCli([TinyPath + '.missing', PromptPath], O, E));
    AssertEquals(0, O.Count);
    AssertEquals(1, E.Count);
    AssertEquals('qlora_logprobs: Unable to open file "' + TinyPath + '.missing": No such file or directory', E[0]);
  finally
    O.Free;
    E.Free;
  end;
end;

{ UT-LogprobsCli-216: token id out of range }
procedure TLogprobsCliTest.TestUT_LogprobsCli_216;
begin
  WriteText(TmpPath, '{"tokens": [274]}');
  CheckError(Self, TmpPath, 'qlora_logprobs: token id 274 outside 0..273');
end;

{ UT-LogprobsCli-217: --split 1 without GPUs (the GPUs are hidden) }
procedure TLogprobsCliTest.TestUT_LogprobsCli_217;
var
  O, E: TStringList;
begin
  try
    AssertEquals(1, TLogprobsCliTest.RunCli(['--split', '1', TinyPath, PromptPath], O, E));
    AssertEquals(0, O.Count);
    AssertEquals(1, E.Count);
    AssertEquals('qlora_logprobs: No CUDA GPUs are available', E[0]);
  finally
    O.Free;
    E.Free;
  end;
end;

{ Like Want, but with the model's Compute set to kBFloat16 (T21). }
function Want16(const AId: string; const AIds: array of Integer; ATop: Integer): string;
var
  G: TGgufFile;
  M: TQwenModel;
  V, Idx: TTensor;
  Ids: array of Int64;
  Lp: array of Double;
  J: Integer;
begin
  G := TGgufFile.Open(TinyPath);
  try
    M := TQwenModel.Load(G, TDevice.Cpu, TDevice.Cpu, 0);
    try
      M.Compute := kBFloat16;
      M.LastLogProbs(AIds).topk(ATop, 0, True, True, V, Idx);
      Ids := nil;
      Lp := nil;
      SetLength(Ids, ATop);
      SetLength(Lp, ATop);
      for J := 0 to ATop - 1 do
      begin
        Ids[J] := Idx.Int64Value([J]);
        Lp[J] := V.DoubleValue([J]);
      end;
      Result := TopLine(AId, Length(AIds), Ids, Lp);
    finally
      M.Free;
    end;
  finally
    G.Free;
  end;
end;

{ UT-LogprobsCli-301: --compute bf16 and fp32 }
procedure TLogprobsCliTest.TestUT_LogprobsCli_301;
var
  O, E: TStringList;
begin
  try
    AssertEquals(0, TLogprobsCliTest.RunCli(['--compute', 'bf16', '--top', '3', TinyPath, PromptPath], O, E));
    AssertEquals(0, E.Count);
    AssertEquals(3, O.Count);
    AssertEquals('line 1 (bf16)', Want16('a', [5, 17, 2], 3), O[0]);
    AssertTrue('differs from the fp32 line', O[0] <> TLogprobsCliTest.Want('a', [5, 17, 2], 3));
  finally
    O.Free;
    E.Free;
  end;
  try
    AssertEquals(0, TLogprobsCliTest.RunCli(['--top', '3', '--compute', 'fp32', TinyPath, PromptPath], O, E));
    AssertEquals(0, E.Count);
    AssertEquals(3, O.Count);
    AssertEquals('line 1 (fp32)', TLogprobsCliTest.Want('a', [5, 17, 2], 3), O[0]);
  finally
    O.Free;
    E.Free;
  end;
end;

{ UT-LogprobsCli-302: --compute with no value and a bad value }
procedure TLogprobsCliTest.TestUT_LogprobsCli_302;
begin
  CheckUsage(Self, ['--compute'], '--compute needs a value');
  CheckUsage(Self, ['--compute', 'fp16', TinyPath, PromptPath], '--compute must be fp32 or bf16, got fp16');
end;

var
  App: TTestRunner;
  Base: string;
begin
  Base := GetTempDir + 'tiny-logprobs-ut-' + IntToStr(GetProcessID);
  TinyPath := Base + '.gguf';
  PromptPath := Base + '-prompts.jsonl';
  TmpPath := Base + '-tmp.jsonl';
  WriteTinyQwenModel(TinyPath, 5, thTied, False);
  WriteText(PromptPath, '{"id": "a", "tokens": [5, 17, 2]}'#10'{"tokens": [8]}'#10#10 +
    '{"id": "t", "text": "hello world"}');
  RegisterTest(TLogprobsCliTest);
  App := TTestRunner.Create(nil);
  App.Initialize;
  App.Title := 'LogprobsCli UT';
  App.Run;
  App.Free;
  DeleteFile(TinyPath);
  DeleteFile(PromptPath);
  DeleteFile(TmpPath);
end.
