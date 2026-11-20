{ UT for program qlora_inspect (T01c) — one published Test method per case
  row of tests/ut/cases/qlora_inspect.cases.md.
  Each case runs the program (compiled once with the system fpc into
  work/ut/qi-bin) with the project root as the working directory, so the
  test files under work/ut/inspect/ land in the project's git-ignored
  scratch. Captures stdout and stderr separately (the wrapper's contract:
  AOut -> stdout, AErr -> stderr) and pins exit code 0 / 1 / 2.
  Invoked with `--all --format=plain` (see .sdlc/ut-coverage.sh). }
program qlora_inspect_test;

{$mode objfpc}{$H+}

uses
  SysUtils, Classes, Process, fpcunit, testregistry, consoletestrunner;

const
  K_USAGE = 'Usage: qlora_inspect [--kv] [--tensors] FILE';
  K_DIR = 'work/ut/inspect';
  K_BIN = 'work/ut/qi-bin';
  K_TAB = #9;

var
  U_SEQ: Int64;
  RUN_SEQ: Int64; { unique suffix per RunBin call }
  FExe: string; { the qlora_inspect binary, built once and shared }

procedure ForceDir(const APath: string);
begin
  ForceDirectories(ExtractFileDir(APath));
end;

function NextName(const ABase: string): string;
begin
  Inc(U_SEQ);
  Result := ABase + '_' + IntToStr(U_SEQ) + '_' +
    IntToStr(GetTickCount64) + '.gguf';
end;

procedure WriteStream(const APath: string; S: TBytesStream);
begin
  { SaveToFile is this build's working stream->file idiom (GgufFile_test,
    30+ uses); F.Write(S.Memory, S.Size) leaks the object's own header area
    instead of the data (see work/ut/inspect/qi002_*.gguf garbage) }
  ForceDir(APath);
  S.SaveToFile(APath);
end;

type
  TGGUF = class
  public
    class procedure WU32(S: TBytesStream; V: LongWord);
    class procedure WU64(S: TBytesStream; V: QWord);
    class procedure WStr(S: TBytesStream; const A: string);
    class procedure WVal(S: TBytesStream; const AKey: string; AVT: LongWord;
      ARaw: QWord; AWidth: Integer);
    class procedure WTensor(S: TBytesStream; const AName: string; ANDims: LongWord;
      const ADims: array of QWord; ATType: LongWord; AOffset: QWord);
    class procedure WHeader(S: TBytesStream; AVersion: LongWord; ANTensors, ANKV: QWord);
    class procedure Pad(S: TBytesStream; ATarget: QWord);
  end;

class procedure TGGUF.WU32(S: TBytesStream; V: LongWord);
begin
  S.WriteBuffer(V, 4);
end;

class procedure TGGUF.WU64(S: TBytesStream; V: QWord);
begin
  S.WriteBuffer(V, 8);
end;

class procedure TGGUF.WStr(S: TBytesStream; const A: string);
var
  C: QWord;
begin
  C := Length(A);
  WU64(S, C);
  if C > 0 then
    S.WriteBuffer(A[1], LongInt(C));
end;

class procedure TGGUF.WVal(S: TBytesStream; const AKey: string; AVT: LongWord;
  ARaw: QWord; AWidth: Integer);
var
  Buf: array[0..7] of Byte;
begin
  WStr(S, AKey);
  WU32(S, AVT);
  Move(ARaw, Buf, AWidth);
  S.WriteBuffer(Buf, AWidth);
end;

class procedure TGGUF.WTensor(S: TBytesStream; const AName: string; ANDims: LongWord;
  const ADims: array of QWord; ATType: LongWord; AOffset: QWord);
var
  I: Integer;
begin
  WStr(S, AName);
  WU32(S, ANDims);
  for I := 0 to Integer(ANDims) - 1 do
    WU64(S, ADims[I]);
  WU32(S, ATType);
  WU64(S, AOffset);
end;

class procedure TGGUF.WHeader(S: TBytesStream; AVersion: LongWord; ANTensors, ANKV: QWord);
begin
  WU32(S, $46554747); { "GGUF" }
  WU32(S, AVersion);
  WU64(S, ANTensors);
  WU64(S, ANKV);
end;

class procedure TGGUF.Pad(S: TBytesStream; ATarget: QWord);
var
  I: QWord;
  B: Byte;
begin
  B := 0;
  for I := S.Size to ATarget - 1 do
    S.WriteBuffer(B, 1);
end;

{ --- process plumbing (same poll/drain shape as qlora_verify_test) --- }

function BuildProg: string;
var
  P: TProcess;
  S: TStringStream;
  Buf: array[1..4096] of Char;
  N: Integer;
  T0: QWord;
begin
  ForceDirectories(K_BIN);
  P := TProcess.Create(nil);
  S := TStringStream.Create('');
  try
    P.Executable := 'fpc';
    P.Parameters.Add('-Mobjfpc');
    P.Parameters.Add('-Sh');
    P.Parameters.Add('-vew');
    P.Parameters.Add('-l-');
    P.Parameters.Add('-Fu' + GetCurrentDir + '/src');
    P.Parameters.Add('-FU' + K_BIN);
    P.Parameters.Add('-FE' + K_BIN);
    P.Parameters.Add(GetCurrentDir + '/src/qlora_inspect.pas');
    P.Options := [poUsePipes, poStderrToOutPut];
    try
      P.Execute;
    except
      Exit('');
    end;
    T0 := GetTickCount64;
    while P.Running and (GetTickCount64 - T0 < QWord(120000)) do
      Sleep(50);
    if P.Running then
      P.Terminate(9);
    repeat
      N := P.Output.Read(Buf, SizeOf(Buf));
      if N > 0 then
        S.Write(Buf, N);
    until N <= 0;
    if (P.ExitCode = 0) and FileExists(K_BIN + '/qlora_inspect') then
      Result := K_BIN + '/qlora_inspect'
    else
      Result := '';
  finally
    P.Free;
    S.Free;
  end;
end;

function RunBin(const AArgs: array of string; out AOut, AErr: string): Integer;
var
  P: TProcess;
  I: Integer;
  T0: QWord;
  Cmd, OutF, ErrF: string;
  S: TStringStream;
  Sfx: Int64;
begin
  AOut := '';
  AErr := '';
  Inc(RUN_SEQ);
  Sfx := RUN_SEQ;
  OutF := K_BIN + '/run.out.' + IntToStr(Sfx);
  ErrF := K_BIN + '/run.err.' + IntToStr(Sfx);
  { sh -c with redirection: FPC's TProcess exposes no Error pipe in this
    build, and stdout/stderr must stay separate for the assertions below
    (same /bin/sh -c shape as qlora_verify_test 003) }
  Cmd := FExe;
  for I := Low(AArgs) to High(AArgs) do
    Cmd := Cmd + ' ' + AArgs[I];
  Cmd := Cmd + ' 1>' + OutF + ' 2>' + ErrF;
  P := TProcess.Create(nil);
  try
    P.Executable := '/bin/sh';
    P.Parameters.Add('-c');
    P.Parameters.Add(Cmd);
    try
      P.Execute;
      T0 := GetTickCount64;
      while P.Running and (GetTickCount64 - T0 < QWord(60000)) do
        Sleep(50);
      if P.Running then
        P.Terminate(9);
      Result := P.ExitCode;
    except
      Result := -1;
    end;
  finally
    P.Free;
  end;
  { TStringStream.DataString + LoadFromFile: the byte->string idiom this
    FPC build compiles cleanly (qlora_verify_test p.112/130; the direct
    string(array of Byte) conversion crashes on finalization, see
    work/ut/direct-check/probe6.pas) }
  if FileExists(OutF) then
  begin
    S := TStringStream.Create('');
    try
      S.LoadFromFile(OutF);
      AOut := S.DataString;
    finally
      S.Free;
    end;
  end;
  if FileExists(ErrF) then
  begin
    S := TStringStream.Create('');
    try
      S.LoadFromFile(ErrF);
      AErr := S.DataString;
    finally
      S.Free;
    end;
  end;
end;

type
  TTestQloraInspect = class(TTestCase)
  public
    function Exe: string;
  published
    procedure TestUT_qlora_inspect_001;
    procedure TestUT_qlora_inspect_002;
    procedure TestUT_qlora_inspect_003;
  end;

function TTestQloraInspect.Exe: string;
begin
  if FExe = '' then
    FExe := BuildProg;
  Result := FExe;
end;

{ UT-qlora_inspect-001: no arguments -> usage on stderr, nothing on stdout,
  exit 2 (pins the wrapper's AErr print loop). }
procedure TTestQloraInspect.TestUT_qlora_inspect_001;
var
  Out, Err: string;
begin
  AssertTrue(Exe <> '');
  AssertEquals(2, RunBin([], Out, Err));
  AssertEquals('', Out);
  AssertEquals(K_USAGE + #10, Err);
end;

{ UT-qlora_inspect-002: valid file, relative path, no flags (default) ->
  7 summary lines + the one type line on stdout, stderr empty, exit 0
  (pins the wrapper's AOut print loop). }
procedure TTestQloraInspect.TestUT_qlora_inspect_002;
var
  S: TBytesStream;
  P, Out, Err: string;
  B: Byte;
  I: Integer;
begin
  AssertTrue(Exe <> '');
  P := K_DIR + '/' + NextName('qi002');
  S := TBytesStream.Create;
  try
    TGGUF.WHeader(S, 3, 1, 0);
    TGGUF.WTensor(S, 'f0', 1, [4], 0, 0); { f32 [4]: 16 B }
    TGGUF.Pad(S, 64); { 24 + 34 = 58 after the directory; 58 -> 64 }
    B := 99;
    for I := 1 to 16 do
      S.WriteBuffer(B, 1); { the 16 B of f32 data: file size 80 }
    WriteStream(P, S);
  finally
    S.Free;
  end;
  AssertEquals(0, RunBin([P], Out, Err));
  AssertEquals('', Err);
  AssertEquals(
    'version' + K_TAB + '3' + #10 +
    'alignment' + K_TAB + '32' + #10 +
    'kv' + K_TAB + '0' + #10 +
    'tensors' + K_TAB + '1' + #10 +
    'data_offset' + K_TAB + '64' + #10 +
    'data_end' + K_TAB + '80' + #10 +
    'file_size' + K_TAB + '80' + #10 +
    'type' + K_TAB + 'f32' + K_TAB + '1' + #10, Out);
end;

{ UT-qlora_inspect-003: --kv, absolute path with uppercase extension,
  mixed-case keys 'KA' / 'kb' -> summary + 2 kv lines (uint8, uint32),
  no type lines, stderr empty, exit 0. }
procedure TTestQloraInspect.TestUT_qlora_inspect_003;
var
  S: TBytesStream;
  P, Out, Err: string;
begin
  AssertTrue(Exe <> '');
  P := K_DIR + '/' + NextName('qi003');
  P := ChangeFileExt(P, '.GGUF'); { uppercase extension on purpose }
  S := TBytesStream.Create;
  try
    TGGUF.WHeader(S, 3, 0, 2);
    TGGUF.WVal(S, 'KA', 0, 7, 1); { uint8 }
    TGGUF.WVal(S, 'kb', 4, 1234, 4); { uint32 }
    TGGUF.Pad(S, 64); { 24 + 15 + 18 = 57 after the directory; 57 -> 64 }
    WriteStream(P, S); { no tensors: file size = data_offset 64 }
  finally
    S.Free;
  end;
  P := GetCurrentDir + '/' + P; { absolute }
  AssertEquals(0, RunBin(['--kv', P], Out, Err));
  AssertEquals('', Err);
  AssertEquals(
    'version' + K_TAB + '3' + #10 +
    'alignment' + K_TAB + '32' + #10 +
    'kv' + K_TAB + '2' + #10 +
    'tensors' + K_TAB + '0' + #10 +
    'data_offset' + K_TAB + '64' + #10 +
    'data_end' + K_TAB + '64' + #10 +
    'file_size' + K_TAB + '64' + #10 +
    'kv' + K_TAB + 'KA' + K_TAB + 'uint8' + #10 +
    'kv' + K_TAB + 'kb' + K_TAB + 'uint32' + #10, Out);
end;

var
  App: TTestRunner;
begin
  ForceDirectories(K_DIR);
  ForceDirectories(K_BIN);
  RegisterTest(TTestQloraInspect);
  App := TTestRunner.Create(nil);
  App.Initialize;
  App.Title := 'qlora_inspect UT (T01c)';
  App.Run;
  App.Free;
end.
