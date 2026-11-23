{ UT program for src/TrainCli.pas: the rows UT-TrainCli-NNN of
  tests/ut/cases/TrainCli.cases.md (stage 2 from 101: the command line and
  the helpers; stage 3 from 201: runs). Runs on the CPU
  (.sdlc/ut-coverage.sh hides the GPUs) on the tiny model of
  tests/ut/QwenBlocksFixture.pas (which carries T05's tiny tokenizer).
  Invoked with `--all --format=plain` (see .sdlc/ut-coverage.sh).

  Skeleton by the harness owner; the test methods are written in T17b. }
program TrainCli_test;

{$mode objfpc}{$H+}

uses
  SysUtils, Classes, Math, Tch, TchNN, GgufFile, QwenTokenizer, QwenModel, Lora, LoraOptim, Checkpoint,
  TrainData, TrainStep, TrainCli, QwenBlocksFixture, fpcunit, testregistry, consoletestrunner;

type
  TTrainCliTest = class(TTestCase)
  public
    { The base arguments of a run: --valid ValidFile --epochs 2 --warmup 1
      --lr 1e-2 --budget 120 --pool 4 --batches-per-update 1 --ckpt-every 3
      --rank 4 --alpha 8, then AExtra, then TinyPath TrainFile AOut. }
    class function RunArgs(const AOut: string; const AExtra: array of string): TStringArray; static;
    { RunTrainCLI(AArgs) into AOut / AErr text (lines joined by #10). }
    class function Run(const AArgs: array of string; out AOut, AErr: string): Integer; static;
    { A fresh directory under TmpDir (not created), ending in the delimiter. }
    class function NewOut: string; static;
    { The update lines of AOut's train-log.tsv without the seconds column,
      joined by #10. }
    class function UpdateLines(const AOut: string): string; static;
    { The fields 1 to 3 (update, epoch, position) of every update line,
      'u e p' joined by '|'. }
    class function Steps(const AOut: string): string; static;
    { The update numbers of the ckpt lines, joined by spaces. }
    class function CkptUpdates(const AOut: string): string; static;
    { Every A, B and moment of checkpoint ADir, read into a fresh set of
      rank 4, scale 2 on a fresh tiny model, as Double; AState its state. }
    class function ReadBack(const ADir: string; out AState: TRunState): TDoubleArray; static;
    { Max |a - b|; Infinity for other lengths. }
    class function DiffArr(const A, B: TDoubleArray): Double; static;
  published
  end;

var
  TmpDir, TinyPath, TrainFile, ValidFile, OtherFile: string;
  OutNo: Integer = 0;

class function TTrainCliTest.RunArgs(const AOut: string; const AExtra: array of string): TStringArray;
const
  Base: array[0..19] of string = ('--valid', '', '--epochs', '2', '--warmup', '1', '--lr', '1e-2',
    '--budget', '120', '--pool', '4', '--batches-per-update', '1', '--ckpt-every', '3',
    '--rank', '4', '--alpha', '8');
var
  I, N: Integer;
begin
  Result := nil;
  SetLength(Result, Length(Base) + Length(AExtra) + 3);
  for I := 0 to High(Base) do
    Result[I] := Base[I];
  Result[1] := ValidFile;
  N := Length(Base);
  for I := 0 to High(AExtra) do
    Result[N + I] := AExtra[I];
  N := N + Length(AExtra);
  Result[N] := TinyPath;
  Result[N + 1] := TrainFile;
  Result[N + 2] := AOut;
end;

class function TTrainCliTest.Run(const AArgs: array of string; out AOut, AErr: string): Integer;
var
  O, E: TStringList;
begin
  O := TStringList.Create;
  E := TStringList.Create;
  try
    Result := RunTrainCLI(AArgs, O, E);
    AOut := string.Join(#10, O.ToStringArray);
    AErr := string.Join(#10, E.ToStringArray);
  finally
    O.Free;
    E.Free;
  end;
end;

class function TTrainCliTest.NewOut: string;
begin
  Inc(OutNo);
  Result := TmpDir + 'out' + IntToStr(OutNo) + PathDelim;
end;

function LogLines(const AOut: string): TStringList;
begin
  Result := TStringList.Create;
  if FileExists(AOut + 'train-log.tsv') then
    Result.LoadFromFile(AOut + 'train-log.tsv');
end;

class function TTrainCliTest.UpdateLines(const AOut: string): string;
var
  L: TStringList;
  I: Integer;
  F: TStringArray;
begin
  Result := '';
  L := LogLines(AOut);
  try
    for I := 0 to L.Count - 1 do
      if Copy(L[I], 1, 4) <> 'ckpt' then
      begin
        F := L[I].Split([#9]);
        SetLength(F, Min(Length(F), 7));
        Result := Result + string.Join(#9, F) + #10;
      end;
  finally
    L.Free;
  end;
end;

class function TTrainCliTest.Steps(const AOut: string): string;
var
  L: TStringList;
  I: Integer;
  F: TStringArray;
begin
  Result := '';
  L := LogLines(AOut);
  try
    for I := 0 to L.Count - 1 do
      if Copy(L[I], 1, 4) <> 'ckpt' then
      begin
        F := L[I].Split([#9]);
        if Result <> '' then
          Result := Result + '|';
        Result := Result + F[0] + ' ' + F[1] + ' ' + F[2];
      end;
  finally
    L.Free;
  end;
end;

class function TTrainCliTest.CkptUpdates(const AOut: string): string;
var
  L: TStringList;
  I: Integer;
begin
  Result := '';
  L := LogLines(AOut);
  try
    for I := 0 to L.Count - 1 do
      if Copy(L[I], 1, 4) = 'ckpt' then
      begin
        if Result <> '' then
          Result := Result + ' ';
        Result := Result + L[I].Split([#9])[1];
      end;
  finally
    L.Free;
  end;
end;

procedure AppendT(var R: TDoubleArray; const T: TTensor);
var
  V: TDoubleArray;
  N, I: Integer;
begin
  V := T.detach.to_device(TDevice.Cpu, kDouble).reshape([-1]).ToDoubleArray;
  N := Length(R);
  SetLength(R, N + Length(V));
  for I := 0 to High(V) do
    R[N + I] := V[I];
end;

class function TTrainCliTest.ReadBack(const ADir: string; out AState: TRunState): TDoubleArray;
var
  G: TGgufFile;
  M: TQwenModel;
  S: TLoraSet;
  O: TLoraAdamW;
  C: TCheckpoints;
  I, J: Integer;
  V: TNamedTensorArray;
begin
  Result := nil;
  G := TGgufFile.Open(TinyPath);
  M := TQwenModel.Load(G, TDevice.Cpu, TDevice.Cpu, 0);
  S := TLoraSet.Create(4, 2.0);
  O := nil;
  C := TCheckpoints.Create(ExtractFilePath(ExcludeTrailingPathDelimiter(ADir)), 1000);
  try
    M.AttachLora(S);
    O := TLoraAdamW.Create(S.Stores, DefaultAdamWConfig);
    AState := C.Load(ADir, S, O);
    for I := 0 to S.Count - 1 do
    begin
      AppendT(Result, S.Pair(I).A);
      AppendT(Result, S.Pair(I).B);
    end;
    for I := 0 to O.StoreCount - 1 do
    begin
      V := O.Moments(I).Variables;
      for J := 0 to High(V) do
        AppendT(Result, V[J].Tensor);
    end;
  finally
    C.Free;
    O.Free;
    M.Free;
    S.Free;
    G.Free;
  end;
end;

class function TTrainCliTest.DiffArr(const A, B: TDoubleArray): Double;
var
  I: Integer;
begin
  if Length(A) <> Length(B) then
    Exit(Infinity);
  Result := 0;
  for I := 0 to High(A) do
    Result := Max(Result, Abs(A[I] - B[I]));
end;

{ ParseTrainArgs of AArgs must give the error AMessage. }
procedure CheckParse(T: TTestCase; const AArgs: array of string; const AMessage: string);
var
  C: TTrainConfig;
begin
  T.AssertEquals(AMessage, ParseTrainArgs(AArgs, C));
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

procedure WriteData;
var
  L: TStringList;
  I: Integer;
begin
  L := TStringList.Create;
  try
    for I := 0 to 7 do
      L.Add(Format('{"id":"t%d","dialect":"objfpc","prompt":"hello %d","answer":"world %s","class":"compiled-run"}',
        [I, I, StringOfChar('a', I + 1)]));
    L.SaveToFile(TrainFile);
    L.Clear;
    L.Add('{"id":"v0","dialect":"objfpc","prompt":"hello","answer":"world aa","class":"compiled-run"}');
    L.SaveToFile(ValidFile);
    L.Clear;
    L.Add('{"id":"o0","dialect":"oxygene","prompt":"hello","answer":"world","class":"static-only"}');
    L.SaveToFile(OtherFile);
  finally
    L.Free;
  end;
end;

var
  App: TTestRunner;
begin
  TmpDir := GetTempDir + 'traincli-ut-' + IntToStr(GetProcessID) + PathDelim;
  ForceDirectories(TmpDir);
  TinyPath := TmpDir + 'tiny.gguf';
  TrainFile := TmpDir + 'train.jsonl';
  ValidFile := TmpDir + 'valid.jsonl';
  OtherFile := TmpDir + 'other.jsonl';
  WriteTinyQwenModel(TinyPath, 5, thTied, False);
  WriteData;
  RegisterTest(TTrainCliTest);
  App := TTestRunner.Create(nil);
  App.Initialize;
  App.Title := 'TrainCli UT';
  App.Run;
  App.Free;
  RemoveTree(TmpDir);
end.
