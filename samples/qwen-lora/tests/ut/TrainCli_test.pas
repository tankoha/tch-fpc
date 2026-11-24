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
    procedure TestUT_TrainCli_101;
    procedure TestUT_TrainCli_102;
    procedure TestUT_TrainCli_103;
    procedure TestUT_TrainCli_104;
    procedure TestUT_TrainCli_105;
    procedure TestUT_TrainCli_106;
    procedure TestUT_TrainCli_107;
    procedure TestUT_TrainCli_108;
    procedure TestUT_TrainCli_201;
    procedure TestUT_TrainCli_202;
    procedure TestUT_TrainCli_203;
    procedure TestUT_TrainCli_204;
    procedure TestUT_TrainCli_205;
    procedure TestUT_TrainCli_206;
    procedure TestUT_TrainCli_207;
    procedure TestUT_TrainCli_208;
    procedure TestUT_TrainCli_209;
    procedure TestUT_TrainCli_210;
    procedure TestUT_TrainCli_211;
    procedure TestUT_TrainCli_212;
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

procedure TTrainCliTest.TestUT_TrainCli_101;
var
  C: TTrainConfig;
  E: Double;
begin
  C := DefaultTrainConfig;
  AssertEquals('rank', 16, C.Rank);
  E := 32;
  AssertTrue('alpha', C.Alpha = E);
  E := 1e-4;
  AssertTrue('lr', C.Lr = E);
  E := 0.1;
  AssertTrue('min-lr-ratio', C.MinLrRatio = E);
  E := 1.0;
  AssertTrue('clip', C.Clip = E);
  AssertEquals('warmup', 50, C.Warmup);
  AssertEquals('epochs', 3, C.Epochs);
  AssertEquals('seed', 1, C.Seed);
  AssertEquals('pool', 64, C.Pool);
  AssertEquals('budget', 2048, C.Budget);
  AssertEquals('batches-per-update', 2, C.BatchesPerUpdate);
  AssertEquals('ckpt-every', 50, C.CkptEvery);
  AssertEquals('keep', 3, C.Keep);
  AssertEquals('slice', 256, C.Slice);
  AssertEquals('classes', 'compiled-run,fpc-approx', string.Join(',', C.Classes));
  AssertEquals('split', -1, C.Split);
  AssertTrue('max-updates', C.MaxUpdates = 0);
  AssertTrue('resume', not C.Resume);
  AssertEquals('Model', '', C.Model);
  AssertEquals('Train', '', C.Train);
  AssertEquals('OutDir', '', C.OutDir);
  AssertEquals('Valid', '', C.Valid);
  AssertEquals('BaseSha256', '', C.BaseSha256);
end;

procedure TTrainCliTest.TestUT_TrainCli_102;
var
  C: TTrainConfig;
begin
  AssertEquals('no error', '', ParseTrainArgs(['m.gguf', 't.jsonl', 'out'], C));
  AssertEquals('Model', 'm.gguf', C.Model);
  AssertEquals('Train', 't.jsonl', C.Train);
  AssertEquals('OutDir with the delimiter', 'out' + PathDelim, C.OutDir);
  AssertEquals('Valid', '', C.Valid);
  AssertEquals('BaseSha256', '', C.BaseSha256);
  AssertEquals('rank', 16, C.Rank);
  AssertTrue('resume', not C.Resume);
end;

procedure TTrainCliTest.TestUT_TrainCli_103;
var
  C: TTrainConfig;
  E: Double;
begin
  AssertEquals('no error', '', ParseTrainArgs(
    ['--rank', '8', '--alpha', '16', '--lr', '0.002', '--warmup', '5',
     '--min-lr-ratio', '0.2', '--epochs', '4', '--seed', '-7', '--pool', '16',
     '--budget', '512', '--batches-per-update', '3', '--clip', '0.5',
     '--ckpt-every', '10', '--keep', '2', '--slice', '64', '--classes', 'a,b,c',
     '--split', '12', '--base-sha256', 'abc', '--max-updates', '9', '--resume',
     '--valid', 'v.jsonl', 'm', 't', 'o/'], C));
  AssertEquals('rank', 8, C.Rank);
  E := 16.0;
  AssertTrue('alpha', C.Alpha = E);
  E := 0.002;
  AssertTrue('lr', C.Lr = E);
  AssertEquals('warmup', 5, C.Warmup);
  E := 0.2;
  AssertTrue('min-lr-ratio', C.MinLrRatio = E);
  AssertEquals('epochs', 4, C.Epochs);
  AssertEquals('seed', -7, C.Seed);
  AssertEquals('pool', 16, C.Pool);
  AssertEquals('budget', 512, C.Budget);
  AssertEquals('batches-per-update', 3, C.BatchesPerUpdate);
  E := 0.5;
  AssertTrue('clip', C.Clip = E);
  AssertEquals('ckpt-every', 10, C.CkptEvery);
  AssertEquals('keep', 2, C.Keep);
  AssertEquals('slice', 64, C.Slice);
  AssertEquals('classes', 'a,b,c', string.Join(',', C.Classes));
  AssertEquals('split', 12, C.Split);
  AssertEquals('base-sha256', 'abc', C.BaseSha256);
  AssertTrue('max-updates', C.MaxUpdates = 9);
  AssertTrue('resume', C.Resume);
  AssertEquals('valid', 'v.jsonl', C.Valid);
  AssertEquals('Model', 'm', C.Model);
  AssertEquals('Train', 't', C.Train);
  AssertEquals('OutDir o/ with one delimiter', 'o' + PathDelim, C.OutDir);
end;

procedure TTrainCliTest.TestUT_TrainCli_104;
begin
  CheckParse(Self, ['--frob', '1', 'm', 't', 'o'], 'unknown option --frob');
  CheckParse(Self, ['m', 't', 'o', '--rank'], '--rank: value missing');
  CheckParse(Self, ['--rank', '0', 'm', 't', 'o'], '--rank: an integer in 1..4096, got "0"');
  CheckParse(Self, ['--epochs', 'x', 'm', 't', 'o'], '--epochs: an integer in 1..1000, got "x"');
  CheckParse(Self, ['--lr', '-1', 'm', 't', 'o'], '--lr: a number >= 0, got "-1"');
  CheckParse(Self, ['--clip', 'nan', 'm', 't', 'o'], '--clip: a number >= 0, got "nan"');
  CheckParse(Self, ['m', 't'], 'expected MODEL TRAIN OUTDIR, got 2 arguments');
  CheckParse(Self, ['a', 'b', 'c', 'd'], 'expected MODEL TRAIN OUTDIR, got 4 arguments');
end;

procedure TTrainCliTest.TestUT_TrainCli_105;
const
  Hex = 'f38b791e7546f6dc9a0c6c2f1f8f4c31c55b90f74e00f0c41a93dbf0b9e8e2f1';
var
  Up: string;
begin
  AssertEquals('the 64 hex', Hex, ShaFromName('/usr/share/ollama/blobs/sha256-' + Hex));
  AssertEquals('63 hex', '', ShaFromName('sha256-' + Copy(Hex, 1, 63)));
  Up := UpCase(Hex);
  AssertEquals('upper case', '', ShaFromName('sha256-' + Up));
  AssertEquals('model.gguf', '', ShaFromName('model.gguf'));
  AssertEquals('a directory below', '', ShaFromName('sha256-' + Hex + '/x'));
end;

procedure TTrainCliTest.TestUT_TrainCli_106;
var
  C: TTrainConfig;
  F: string;
begin
  F := StringOfChar('f', 64);
  AssertEquals('no error', '', ParseTrainArgs(['d/sha256-' + F, 't', 'o'], C));
  AssertEquals('BaseSha256 from the model name', F, C.BaseSha256);
  AssertEquals('no error', '', ParseTrainArgs(['d/sha256-' + F, 't', 'o', '--base-sha256', 'given'], C));
  AssertEquals('BaseSha256 from --base-sha256', 'given', C.BaseSha256);
end;

procedure TTrainCliTest.TestUT_TrainCli_107;
var
  C0, C1: TTrainConfig;
  H0: string;
  L: TStringList;
  Train2: string;
begin
  AssertEquals('no error', '', ParseTrainArgs(['--valid', ValidFile, TinyPath, TrainFile, 'o'], C0));
  H0 := ConfigHash(C0);
  AssertEquals('32 characters', 32, Length(H0));
  AssertEquals('no error', '', ParseTrainArgs(
    ['--valid', ValidFile, TinyPath, TrainFile, 'o',
     '--max-updates', '3', '--resume', '--split', '1', '--ckpt-every', '7', '--keep', '9',
     '--base-sha256', 'z'], C1));
  AssertTrue('insensitive arguments do not change the hash', ConfigHash(C1) = H0);
  AssertEquals('no error', '', ParseTrainArgs(['--valid', ValidFile, TinyPath, TrainFile, 'other dir'], C1));
  AssertTrue('another output directory: same hash', ConfigHash(C1) = H0);
  AssertEquals('no error', '', ParseTrainArgs(['--valid', ValidFile, TinyPath, TrainFile, 'o', '--lr', '2e-4'], C1));
  AssertTrue('--lr 2e-4 changes the hash', ConfigHash(C1) <> H0);
  AssertEquals('no error', '', ParseTrainArgs(['--valid', ValidFile, TinyPath, TrainFile, 'o', '--seed', '2'], C1));
  AssertTrue('--seed 2 changes the hash', ConfigHash(C1) <> H0);
  AssertEquals('no error', '', ParseTrainArgs(['--valid', ValidFile, TinyPath, TrainFile, 'o', '--classes', 'x'], C1));
  AssertTrue('--classes x changes the hash', ConfigHash(C1) <> H0);
  AssertEquals('no error', '', ParseTrainArgs([TinyPath, TrainFile, 'o'], C1));
  AssertTrue('without --valid the hash changes', ConfigHash(C1) <> H0);
  Train2 := TmpDir + 'train-emptyline.jsonl';
  L := TStringList.Create;
  try
    L.LoadFromFile(TrainFile);
    L.Add('');
    L.SaveToFile(Train2);
  finally
    L.Free;
  end;
  AssertEquals('no error', '', ParseTrainArgs(['--valid', ValidFile, TinyPath, Train2, 'o'], C1));
  AssertTrue('a train file with one more (empty) line changes the hash', ConfigHash(C1) <> H0);
end;

procedure TTrainCliTest.TestUT_TrainCli_108;
var
  G: TGgufFile;
  Tok: TQwenTokenizer;
  S: TTrainSet;
  C1, C2, C3: TTrainConfig;
begin
  G := TGgufFile.Open(TinyPath);
  Tok := TQwenTokenizer.Create(G);
  S := TTrainSet.Create(Tok, TrainFile, ['compiled-run'], 120);
  try
    AssertEquals('4 batches in epoch 0', 4, Length(EpochBatches(S, 1, 0, 4, 120)));
    AssertEquals('4 batches in epoch 1', 4, Length(EpochBatches(S, 1, 1, 4, 120)));
    AssertEquals('no error', '', ParseTrainArgs(RunArgs(NewOut, []), C1));
    AssertTrue('TotalUpdates 8', TotalUpdates(S, C1) = 8);
    AssertEquals('no error', '', ParseTrainArgs(RunArgs(NewOut, ['--batches-per-update', '3']), C2));
    AssertTrue('TotalUpdates 4', TotalUpdates(S, C2) = 4);
    AssertEquals('no error', '',
      ParseTrainArgs(RunArgs(NewOut, ['--batches-per-update', '3', '--epochs', '5']), C3));
    AssertTrue('TotalUpdates 10', TotalUpdates(S, C3) = 10);
  finally
    S.Free;
    Tok.Free;
    G.Free;
  end;
end;

procedure TTrainCliTest.TestUT_TrainCli_201;
var
  O, Out, Err: string;
begin
  O := NewOut;
  AssertEquals('exit code', 0, Run(RunArgs(O, []), Out, Err));
  AssertEquals('out', 'done: 8 updates, best ' + O + 'ckpt' + PathDelim + 'ckpt-00000008', Out);
  AssertEquals('err', '', Err);
  AssertEquals('Steps', '1 0 2|2 0 4|3 0 6|4 0 8|5 1 2|6 1 4|7 1 6|8 1 8', Steps(O));
  AssertEquals('CkptUpdates', '3 4 6 8', CkptUpdates(O));
  AssertTrue('ckpt-00000004 kept', DirectoryExists(O + 'ckpt' + PathDelim + 'ckpt-00000004'));
  AssertTrue('ckpt-00000006 kept', DirectoryExists(O + 'ckpt' + PathDelim + 'ckpt-00000006'));
  AssertTrue('ckpt-00000008 kept', DirectoryExists(O + 'ckpt' + PathDelim + 'ckpt-00000008'));
  AssertTrue('ckpt-00000003 gone', not DirectoryExists(O + 'ckpt' + PathDelim + 'ckpt-00000003'));
end;

procedure TTrainCliTest.TestUT_TrainCli_202;
var
  O, Dir: string;
  Out, Err: string;
  C: TTrainConfig;
  St8, St4: TRunState;
  G: TGgufFile;
  Tok: TQwenTokenizer;
  M: TQwenModel;
  S: TLoraSet;
  Opt: TLoraAdamW;
  Ck: TCheckpoints;
  V: TTrainSet;
  Computed, Rel: Double;
begin
  O := NewOut;
  AssertEquals('exit code', 0, Run(RunArgs(O, []), Out, Err));
  AssertEquals('no error', '', ParseTrainArgs(RunArgs(O, []), C));
  Dir := O + 'ckpt' + PathDelim;
  St8 := ReadRunState(Dir + 'ckpt-00000008');
  AssertTrue('update 8', St8.Update = 8);
  AssertTrue('epoch 2', St8.Epoch = 2);
  AssertTrue('position 0', St8.Position = 0);
  AssertTrue('seed 1', St8.Seed = 1);
  AssertTrue('rank 4', St8.Rank = 4);
  AssertTrue('scale 2', St8.Scale = 2.0);
  AssertEquals('ConfigHash', ConfigHash(C), St8.ConfigHash);
  AssertEquals('base', '', St8.BaseSha256);
  St4 := ReadRunState(Dir + 'ckpt-00000004');
  AssertTrue('update 4', St4.Update = 4);
  AssertTrue('epoch 1', St4.Epoch = 1);
  AssertTrue('position 0', St4.Position = 0);
  G := TGgufFile.Open(TinyPath);
  M := TQwenModel.Load(G, TDevice.Cpu, TDevice.Cpu, 0);
  S := TLoraSet.Create(4, 2.0);
  Opt := nil;
  Ck := nil;
  V := nil;
  Tok := nil;
  try
    Tok := TQwenTokenizer.Create(G);
    V := TTrainSet.Create(Tok, C.Valid, C.Classes, C.Budget);
    M.AttachLora(S);
    Opt := TLoraAdamW.Create(S.Stores, DefaultAdamWConfig);
    Ck := TCheckpoints.Create(Dir, 1000);
    Ck.Load(Dir + 'ckpt-00000008', S, Opt);
    Computed := ValidationLoss(M, V, 256);
    Rel := Abs(St8.ValLoss - Computed) / Max(Abs(Computed), 1e-30);
    AssertTrue('ValLoss relative 1e-9 of the computed one', Rel <= 1e-9);
  finally
    Ck.Free;
    Opt.Free;
    V.Free;
    M.Free;
    S.Free;
    Tok.Free;
    G.Free;
  end;
end;

procedure TTrainCliTest.TestUT_TrainCli_203;
var
  O, Out, Err: string;
  L, U: TStringList;
  FS: TFormatSettings;
  F, F8: TStringArray;
  G: TGgufFile;
  Tok: TQwenTokenizer;
  M: TQwenModel;
  S: TTrainSet;
  B: TBatches;
  I: Integer;
  R: TSampleLoss;
  Loss1, Lr1, Lr8, Mean, Sum, Count, Rel: Double;
begin
  O := NewOut;
  AssertEquals('exit code', 0, Run(RunArgs(O, []), Out, Err));
  L := LogLines(O);
  U := TStringList.Create;
  try
    for I := 0 to L.Count - 1 do
      if Copy(L[I], 1, 4) <> 'ckpt' then
        U.Add(L[I]);
    AssertEquals('8 update lines', 8, U.Count);
    F := U[0].Split([#9]);
    F8 := U[7].Split([#9]);
    AssertEquals('the first update line has 8 fields', 8, Length(F));
    AssertEquals('the eighth update line has 8 fields', 8, Length(F8));
    FS := DefaultFormatSettings;
    FS.DecimalSeparator := '.';
    AssertTrue('the first loss parses', TryStrToFloat(F[3], Loss1, FS));
    AssertTrue('the lr of update 1 parses', TryStrToFloat(F[4], Lr1, FS));
    AssertTrue('the lr of update 8 parses', TryStrToFloat(F8[4], Lr8, FS));
    G := TGgufFile.Open(TinyPath);
    M := TQwenModel.Load(G, TDevice.Cpu, TDevice.Cpu, 0);
    S := nil;
    Tok := nil;
    try
      Tok := TQwenTokenizer.Create(G);
      S := TTrainSet.Create(Tok, TrainFile, DefaultTrainConfig.Classes, 120);
      B := EpochBatches(S, 1, 0, 4, 120);
      Sum := 0;
      Count := 0;
      for I := 0 to High(B[0]) do
      begin
        R := SampleLoss(M, S[B[0][I]], 256);
        Sum := Sum + R.Sum;
        Count := Count + R.Count;
      end;
      Mean := Sum / Count;
      Rel := Abs(Loss1 - Mean) / Max(Abs(Mean), 1e-30);
      AssertTrue('logged loss relative 1e-6 of the batch mean', Rel <= 1e-6);
      AssertTrue('lr of update 1 within 1e-15',
        Abs(Lr1 - ScheduleLR(1e-2, 1, 1, 8, 0.1)) <= 1e-15);
      AssertTrue('lr of update 8 within 1e-15',
        Abs(Lr8 - ScheduleLR(1e-2, 8, 1, 8, 0.1)) <= 1e-15);
    finally
      S.Free;
      M.Free;
      Tok.Free;
      G.Free;
    end;
  finally
    U.Free;
    L.Free;
  end;
end;

procedure TTrainCliTest.TestUT_TrainCli_204;
var
  A, B, Out, Err: string;
  StA, StB: TRunState;
  RA, RB: TDoubleArray;
begin
  A := NewOut;
  B := NewOut;
  AssertEquals('A exit code', 0, Run(RunArgs(A, []), Out, Err));
  AssertEquals('B first exit code', 0, Run(RunArgs(B, ['--max-updates', '3']), Out, Err));
  AssertEquals('B first out', 'stopped after update 3 of 8', Out);
  AssertEquals('B second exit code', 0, Run(RunArgs(B, ['--resume']), Out, Err));
  AssertEquals('B second out',
    'resumed from ' + B + 'ckpt' + PathDelim + 'ckpt-00000003 at update 3' + #10 +
    'done: 8 updates, best ' + B + 'ckpt' + PathDelim + 'ckpt-00000008', Out);
  AssertEquals('UpdateLines', UpdateLines(A), UpdateLines(B));
  RA := ReadBack(A + 'ckpt' + PathDelim + 'ckpt-00000008', StA);
  RB := ReadBack(B + 'ckpt' + PathDelim + 'ckpt-00000008', StB);
  AssertEquals('DiffArr', 0, DiffArr(RA, RB));
  AssertTrue('the same ValLoss', StA.ValLoss = StB.ValLoss);
end;

procedure TTrainCliTest.TestUT_TrainCli_205;
var
  A, B, Out, Err: string;
  StA, StB: TRunState;
  RA, RB: TDoubleArray;
begin
  A := NewOut;
  B := NewOut;
  AssertEquals('A exit code', 0, Run(RunArgs(A, []), Out, Err));
  AssertEquals('B first exit code', 0, Run(RunArgs(B, ['--max-updates', '4']), Out, Err));
  AssertEquals('B second exit code', 0, Run(RunArgs(B, ['--resume']), Out, Err));
  AssertEquals('UpdateLines', UpdateLines(A), UpdateLines(B));
  RA := ReadBack(A + 'ckpt' + PathDelim + 'ckpt-00000008', StA);
  RB := ReadBack(B + 'ckpt' + PathDelim + 'ckpt-00000008', StB);
  AssertEquals('DiffArr', 0, DiffArr(RA, RB));
end;

procedure TTrainCliTest.TestUT_TrainCli_206;
var
  A, F, Out, Err: string;
begin
  A := NewOut;
  F := NewOut;
  AssertEquals('first exit code', 0, Run(RunArgs(A, ['--max-updates', '3']), Out, Err));
  AssertEquals('config exit code', 1, Run(RunArgs(A, ['--resume', '--lr', '2e-2']), Out, Err));
  AssertEquals('config err',
    'the configuration differs from the checkpoint''s: ' + A + 'ckpt' + PathDelim + 'ckpt-00000003', Err);
  AssertEquals('base exit code', 1, Run(RunArgs(A, ['--resume', '--base-sha256', 'abc']), Out, Err));
  AssertEquals('base err',
    'the base model differs from the checkpoint''s: ' + A + 'ckpt' + PathDelim + 'ckpt-00000003', Err);
  AssertEquals('fresh exit code', 1, Run(RunArgs(F, ['--resume']), Out, Err));
  AssertEquals('fresh err', 'nothing to resume in ' + F + 'ckpt' + PathDelim, Err);
end;

procedure TTrainCliTest.TestUT_TrainCli_207;
var
  O, Out, Err: string;
  L: TStringList;
  I, N: Integer;
begin
  O := NewOut;
  AssertEquals('exit code', 3, Run(RunArgs(O, ['--lr', '1e30']), Out, Err));
  AssertEquals('err', 'non-finite loss at update 2; stopped without updating', Err);
  AssertEquals('Steps', '1 0 2|2 0 4', Steps(O));
  L := LogLines(O);
  try
    N := 0;
    for I := 0 to L.Count - 1 do
      if Copy(L[I], 1, 4) = 'ckpt' then
        Inc(N);
    AssertEquals('no ckpt line', 0, N);
  finally
    L.Free;
  end;
end;

procedure TTrainCliTest.TestUT_TrainCli_208;
var
  O, Out, Err: string;
  A: TStringArray;
begin
  O := NewOut;
  A := RunArgs(O, []);
  A[20] := TmpDir + 'none.gguf';
  AssertEquals('missing model exit code', 1, Run(A, Out, Err));
  AssertEquals('missing model err', 'no model file ' + TmpDir + 'none.gguf', Err);
  A := RunArgs(O, []);
  A[21] := TmpDir + 'none.jsonl';
  AssertEquals('missing data exit code', 1, Run(A, Out, Err));
  AssertEquals('missing data err', 'no data file ' + TmpDir + 'none.jsonl', Err);
  A := RunArgs(O, []);
  A[21] := OtherFile;
  AssertEquals('no sample exit code', 1, Run(A, Out, Err));
  AssertEquals('no sample err', 'no training sample in ' + OtherFile, Err);
  A := RunArgs(O, ['--warmup', '8']);
  AssertEquals('warm-up exit code', 1, Run(A, Out, Err));
  AssertEquals('warm-up err', 'warm-up 8 must be below the 8 updates', Err);
end;

procedure TTrainCliTest.TestUT_TrainCli_209;
var
  Out, Err: string;
begin
  AssertEquals('exit code', 2, Run(['--rank', '0', 'm', 't', 'o'], Out, Err));
  AssertEquals('err',
    '--rank: an integer in 1..4096, got "0"' + #10 +
    'usage: qlora_train [options] MODEL TRAIN OUTDIR (see the unit header)', Err);
  AssertEquals('out', '', Out);
end;

procedure TTrainCliTest.TestUT_TrainCli_210;
var
  O, Out, Err: string;
begin
  O := NewOut;
  AssertEquals('exit code', 0,
    Run(RunArgs(O, ['--max-updates', '100', '--batches-per-update', '3', '--keep', '1']), Out, Err));
  AssertEquals('out', 'done: 4 updates, best ' + O + 'ckpt' + PathDelim + 'ckpt-00000004', Out);
  AssertEquals('Steps', '1 0 6|2 0 8|3 1 6|4 1 8', Steps(O));
  AssertEquals('CkptUpdates', '2 3 4', CkptUpdates(O));
end;

procedure TTrainCliTest.TestUT_TrainCli_211;
var
  O, Out, Err: string;
  A, B: TStringArray;
  St: TRunState;
  L: TStringList;
  I: Integer;
  F: TStringArray;
begin
  O := NewOut;
  A := RunArgs(O, []);
  SetLength(B, Length(A) - 2);
  for I := 2 to High(A) do
    B[I - 2] := A[I];
  AssertEquals('exit code', 0, Run(B, Out, Err));
  AssertEquals('out', 'done: 8 updates, best ', Out);
  St := ReadRunState(O + 'ckpt' + PathDelim + 'ckpt-00000008');
  AssertTrue('ValLoss NaN', IsNan(St.ValLoss));
  L := LogLines(O);
  try
    for I := 0 to L.Count - 1 do
      if Copy(L[I], 1, 4) = 'ckpt' then
      begin
        F := L[I].Split([#9]);
        AssertEquals('the loss of the ckpt line', 'nan', F[2]);
      end;
  finally
    L.Free;
  end;
end;

procedure TTrainCliTest.TestUT_TrainCli_212;
var
  O, Out, Err: string;
  R: Integer;
begin
  O := NewOut;
  R := Run(RunArgs(O, ['--split', '1']), Out, Err);
  AssertEquals('exit code', 1, R);
  AssertEquals('err', 'No CUDA GPUs are available', Err);
  R := Run(RunArgs(O, ['--split', '3']), Out, Err);
  AssertEquals('exit code', 1, R);
  AssertEquals('err', 'split 3 outside 0..2', Err);
  AssertEquals('out', '', Out);
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
