{ UT program for src/LogprobsCli.pas: the rows UT-LogprobsCli-NNN of
  tests/ut/cases/LogprobsCli.cases.md (stage 3 from 201). The tiny qwen35
  model with a tied head and the T05 UT tokenizer
  (WriteTinyQwenModel(.., thTied, False) of tests/ut/QwenBlocksFixture.pas)
  is written once at start (TinyPath), with the prompt file PromptPath.
  Runs on the CPU (.sdlc/ut-coverage.sh hides the GPUs).
  Invoked with `--all --format=plain` (see .sdlc/ut-coverage.sh).

  Skeleton by the harness owner; the test methods are written in T11b. }
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
  end;

const
  Usage = 'Usage: qlora_logprobs [--top N] [--split S] MODEL FILE';

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
