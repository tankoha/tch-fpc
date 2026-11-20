{ UT program for src/InspectCli.pas -- T01c: rows UT-InspectCli-001..011 of
  tests/ut/cases/InspectCli.cases.md (012..017 are [廃止]: UT never needs the
  rehearsal model; the oracle comparison is ITa-T01-001..005).
  Builds its own small GGUF files (the writer is kept from
  tests/ut/GgufFile_test.pas) under work/ut/inspect/ with unique names;
  work/ is git-ignored and the gate wipes work/ut before running, so every
  file is written again on every run.  All behavior lines live in
  InspectCli.RunInspectCLI, measured in-process.  Invoked with
  `--all --format=plain` (see .sdlc/ut-coverage.sh). }
program InspectCli_test;

{$mode objfpc}{$H+}

uses
  SysUtils, Classes, GgufFile, InspectCli, fpcunit, testregistry, consoletestrunner;

const
  K_TAB = #9;
  K_DIR = 'work/ut/inspect';
  K_USAGE = 'Usage: qlora_inspect [--kv] [--tensors] FILE';

var
  U_SEQ: Integer; { unique-name counter, mktemp-style: never repeats }

function NextPath(const ABase: string): string;
begin
  Inc(U_SEQ);
  Result := K_DIR + '/' + ABase + '_' + IntToStr(U_SEQ) + '.gguf';
end;

type
  TInspectCliTest = class(TTestCase)
  public
    procedure WU32(S: TBytesStream; V: LongWord);
    procedure WU64(S: TBytesStream; V: QWord);
    procedure WB(S: TBytesStream; V: Byte);
    procedure WStr(S: TBytesStream; const A: string);
    procedure WVal(S: TBytesStream; const AKey: string; AVT: LongWord; ARaw: QWord; AWidth: Integer);
    procedure WValStr(S: TBytesStream; const AKey, AVal: string);
    procedure WArrInt(S: TBytesStream; const AKey: string; AElemVT: LongWord; const AVals: array of Int64);
    procedure WArrStr(S: TBytesStream; const AKey: string; const AVals: array of string);
    procedure WHeader(S: TBytesStream; AVersion: LongWord; ANTensors, ANKV: QWord);
    procedure WTensor(S: TBytesStream; const AName: string; ANDims: LongWord; const ADims: array of QWord; ATType: LongWord; AOffset: QWord);
    procedure PadTo(S: TBytesStream; ATarget: QWord);
    function MakePat(AN: Integer; ABase: Byte): TBytes;
  published
    procedure TestUT_InspectCli_001;
    procedure TestUT_InspectCli_002;
    procedure TestUT_InspectCli_003;
    procedure TestUT_InspectCli_004;
    procedure TestUT_InspectCli_005;
    procedure TestUT_InspectCli_006;
    procedure TestUT_InspectCli_007;
    procedure TestUT_InspectCli_008;
    procedure TestUT_InspectCli_009;
    procedure TestUT_InspectCli_010;
    procedure TestUT_InspectCli_011;
  end;

procedure TInspectCliTest.WU32(S: TBytesStream; V: LongWord);
begin
  S.WriteBuffer(V, 4);
end;

procedure TInspectCliTest.WU64(S: TBytesStream; V: QWord);
begin
  S.WriteBuffer(V, 8);
end;

procedure TInspectCliTest.WB(S: TBytesStream; V: Byte);
begin
  S.WriteBuffer(V, 1);
end;

{ A GGUF string: UInt64 number of bytes, then that many bytes. }
procedure TInspectCliTest.WStr(S: TBytesStream; const A: string);
var
  C: QWord;
begin
  C := Length(A);
  WU64(S, C);
  if C > 0 then
    S.WriteBuffer(A[1], LongInt(C));
end;

{ A scalar entry: key, value type (facts' wire codes 0..12), then
  AWidth bytes with the low twos-complement bytes of ARaw. }
procedure TInspectCliTest.WVal(S: TBytesStream; const AKey: string;
  AVT: LongWord; ARaw: QWord; AWidth: Integer);
var
  Buf: array[0..7] of Byte;
begin
  WStr(S, AKey);
  WU32(S, AVT);
  Move(ARaw, Buf, AWidth);
  S.WriteBuffer(Buf, AWidth);
end;

procedure TInspectCliTest.WValStr(S: TBytesStream; const AKey, AVal: string);
begin
  WStr(S, AKey);
  WU32(S, 8); { facts' wire order: 0..12, string = 8 }
  WStr(S, AVal);
end;

procedure TInspectCliTest.WArrInt(S: TBytesStream; const AKey: string;
  AElemVT: LongWord; const AVals: array of Int64);
var
  I, W: Integer;
  T: array[0..7] of Byte;
begin
  WStr(S, AKey);
  WU32(S, 9); { facts' wire order: array = 9 }
  WU32(S, AElemVT);
  WU64(S, Length(AVals));
  case AElemVT of
    0, 1, 7: { uint8, int8, bool }
      W := 1;
    2, 3: { uint16, int16 }
      W := 2;
    4, 5: { uint32, int32 }
      W := 4;
  else { uint64, int64 }
      W := 8;
  end;
  for I := Low(AVals) to High(AVals) do
  begin
    Move(AVals[I], T, W);
    S.WriteBuffer(T, W);
  end;
end;

procedure TInspectCliTest.WArrStr(S: TBytesStream; const AKey: string;
  const AVals: array of string);
var
  I: Integer;
begin
  WStr(S, AKey);
  WU32(S, 9); { facts' wire order: array = 9 }
  WU32(S, 8); { element type: string = 8 }
  WU64(S, Length(AVals));
  for I := Low(AVals) to High(AVals) do
    WStr(S, AVals[I]);
end;

procedure TInspectCliTest.WHeader(S: TBytesStream; AVersion: LongWord; ANTensors, ANKV: QWord);
begin
  WU32(S, $46554747); { "GGUF" }
  WU32(S, AVersion);
  WU64(S, ANTensors);
  WU64(S, ANKV);
end;

procedure TInspectCliTest.WTensor(S: TBytesStream; const AName: string; ANDims: LongWord;
  const ADims: array of QWord; ATType: LongWord; AOffset: QWord);
var
  I: Integer;
begin
  WStr(S, AName);
  WU32(S, ANDims);
  for I := Low(ADims) to High(ADims) do
    WU64(S, ADims[I]);
  WU32(S, ATType);
  WU64(S, AOffset);
end;

procedure TInspectCliTest.PadTo(S: TBytesStream; ATarget: QWord);
var
  I: QWord;
  B: Byte;
begin
  B := 0;
  for I := S.Size to ATarget - 1 do
    S.WriteBuffer(B, 1);
end;

function TInspectCliTest.MakePat(AN: Integer; ABase: Byte): TBytes;
var
  I: Integer;
  B: Byte;
begin
  SetLength(Result, AN);
  B := ABase;
  for I := 1 to AN do
  begin
    Result[I] := B;
    Inc(B);
  end;
end;

{ 001: valid file: the seven summary lines, no flags -- exact AOut, exit 0, AErr empty }
procedure TInspectCliTest.TestUT_InspectCli_001;
var
  S: TBytesStream;
  P: string;
  AOut, AErr: TStringList;
  D: TBytes;
begin
  P := NextPath('UT_InspectCli_001_a');
  S := TBytesStream.Create;
  try
    WHeader(S, 3, 1, 2);
    WVal(S, 'k1', 0, 1, 1);
    WValStr(S, 'k2', 'v');
    WTensor(S, 't1', 1, [4], 0, 0);
    PadTo(S, 96);
    D := MakePat(16, 1);
    S.WriteBuffer(D[0], 16);
    S.SaveToFile(P);
  finally
    S.Free;
  end;
  AOut := TStringList.Create;
  AErr := TStringList.Create;
  try
    AssertEquals(0, InspectCli.RunInspectCLI([P], AOut, AErr));
    AssertEquals(0, AErr.Count);
    AssertEquals(8, AOut.Count);
    AssertEquals('version' + K_TAB + '3', AOut[0]);
    AssertEquals('alignment' + K_TAB + '32', AOut[1]);
    AssertEquals('kv' + K_TAB + '2', AOut[2]);
    AssertEquals('tensors' + K_TAB + '1', AOut[3]);
    AssertEquals('data_offset' + K_TAB + '96', AOut[4]);
    AssertEquals('data_end' + K_TAB + '112', AOut[5]);
    AssertEquals('file_size' + K_TAB + '112', AOut[6]);
    AssertEquals('type' + K_TAB + 'f32' + K_TAB + '1', AOut[7]);
  finally
    AOut.Free;
    AErr.Free;
  end;
end;

{ 002: valid file, three types in file order q4_K, f16, f32 -- exactly three
  type lines ordered by id }
procedure TInspectCliTest.TestUT_InspectCli_002;
var
  S: TBytesStream;
  P: string;
  AOut, AErr: TStringList;
  D: TBytes;
begin
  P := NextPath('UT_InspectCli_002_a');
  S := TBytesStream.Create;
  try
    WHeader(S, 3, 3, 0);
    WTensor(S, 'a', 1, [256], 12, 0);      { q4_K [256]: 144 B }
    WTensor(S, 'b', 1, [8], 1, 160);       { f16 [8]: 16 B, after 144 -> 160 }
    WTensor(S, 'c', 1, [4], 0, 192);       { f32 [4]: 16 B, after 176 -> 192 }
    PadTo(S, 128); { 24 + 3x33 = 123 after the directory; 123 -> 128 }
    D := MakePat(144, 1);
    S.WriteBuffer(D[0], 144);
    PadTo(S, 288); { a: 144 -> 160 aligned absolute (128 + 160) }
    D := MakePat(16, 100);
    S.WriteBuffer(D[0], 16);
    PadTo(S, 320); { b: 16 -> 32 aligned (160 + 32 = 192) }
    D := MakePat(16, 200);
    S.WriteBuffer(D[0], 16);
    S.SaveToFile(P);
  finally
    S.Free;
  end;
  AOut := TStringList.Create;
  AErr := TStringList.Create;
  try
    AssertEquals(0, InspectCli.RunInspectCLI([P], AOut, AErr));
    AssertEquals(0, AErr.Count);
    AssertEquals(10, AOut.Count);
    AssertEquals('version' + K_TAB + '3', AOut[0]);
    AssertEquals('alignment' + K_TAB + '32', AOut[1]);
    AssertEquals('kv' + K_TAB + '0', AOut[2]);
    AssertEquals('tensors' + K_TAB + '3', AOut[3]);
    AssertEquals('data_offset' + K_TAB + '128', AOut[4]);
    AssertEquals('data_end' + K_TAB + '336', AOut[5]);
    AssertEquals('file_size' + K_TAB + '336', AOut[6]);
    AssertEquals('type' + K_TAB + 'f32' + K_TAB + '1', AOut[7]);
    AssertEquals('type' + K_TAB + 'f16' + K_TAB + '1', AOut[8]);
    AssertEquals('type' + K_TAB + 'q4_K' + K_TAB + '1', AOut[9]);
  finally
    AOut.Free;
    AErr.Free;
  end;
end;

{ 003: same file, no flags -- AOut = summary + type lines only }
procedure TInspectCliTest.TestUT_InspectCli_003;
var
  S: TBytesStream;
  P: string;
  AOut, AErr: TStringList;
  D: TBytes;
begin
  P := NextPath('UT_InspectCli_003_a');
  S := TBytesStream.Create;
  try
    WHeader(S, 3, 1, 2);
    WVal(S, 'k1', 0, 1, 1);
    WValStr(S, 'k2', 'v');
    WTensor(S, 't1', 1, [4], 0, 0);
    PadTo(S, 96);
    D := MakePat(16, 1);
    S.WriteBuffer(D[0], 16);
    S.SaveToFile(P);
  finally
    S.Free;
  end;
  AOut := TStringList.Create;
  AErr := TStringList.Create;
  try
    AssertEquals(0, InspectCli.RunInspectCLI([P], AOut, AErr));
    AssertEquals(0, AErr.Count);
    AssertEquals(8, AOut.Count);
    AssertEquals('version' + K_TAB + '3', AOut[0]);
    AssertEquals('alignment' + K_TAB + '32', AOut[1]);
    AssertEquals('kv' + K_TAB + '2', AOut[2]);
    AssertEquals('tensors' + K_TAB + '1', AOut[3]);
    AssertEquals('data_offset' + K_TAB + '96', AOut[4]);
    AssertEquals('data_end' + K_TAB + '112', AOut[5]);
    AssertEquals('file_size' + K_TAB + '112', AOut[6]);
    AssertEquals('type' + K_TAB + 'f32' + K_TAB + '1', AOut[7]);
  finally
    AOut.Free;
    AErr.Free;
  end;
end;

{ 004: --kv: one kv line per KV in file order -- exact AOut, rc 0 }
procedure TInspectCliTest.TestUT_InspectCli_004;
var
  S: TBytesStream;
  P: string;
  AOut, AErr: TStringList;
begin
  P := NextPath('UT_InspectCli_004_a');
  S := TBytesStream.Create;
  try
    WHeader(S, 3, 0, 4);
    WVal(S, 'K1', 0, 1, 1);
    WValStr(S, 'K2', 'v');
    WArrStr(S, 'K3', ['aa', 'bb']);
    WArrInt(S, 'K4', 5, [1, 2, 3]);
    PadTo(S, 160);
    S.SaveToFile(P);
  finally
    S.Free;
  end;
  AOut := TStringList.Create;
  AErr := TStringList.Create;
  try
    AssertEquals(0, InspectCli.RunInspectCLI(['--kv', P], AOut, AErr));
    AssertEquals(0, AErr.Count);
    AssertEquals(11, AOut.Count);
    AssertEquals('version' + K_TAB + '3', AOut[0]);
    AssertEquals('alignment' + K_TAB + '32', AOut[1]);
    AssertEquals('kv' + K_TAB + '4', AOut[2]);
    AssertEquals('tensors' + K_TAB + '0', AOut[3]);
    AssertEquals('data_offset' + K_TAB + '160', AOut[4]);
    AssertEquals('data_end' + K_TAB + '160', AOut[5]);
    AssertEquals('file_size' + K_TAB + '160', AOut[6]);
    AssertEquals('kv' + K_TAB + 'K1' + K_TAB + 'uint8', AOut[7]);
    AssertEquals('kv' + K_TAB + 'K2' + K_TAB + 'string', AOut[8]);
    AssertEquals('kv' + K_TAB + 'K3' + K_TAB + 'array<string>[2]', AOut[9]);
    AssertEquals('kv' + K_TAB + 'K4' + K_TAB + 'array<int32>[3]', AOut[10]);
  finally
    AOut.Free;
    AErr.Free;
  end;
end;

{ 005: --tensors: tensor lines, dims "256 2" -- exact AOut, rc 0 }
procedure TInspectCliTest.TestUT_InspectCli_005;
var
  S: TBytesStream;
  P: string;
  AOut, AErr: TStringList;
  D: TBytes;
begin
  P := NextPath('UT_InspectCli_005_a');
  S := TBytesStream.Create;
  try
    WHeader(S, 3, 2, 0);
    WTensor(S, 't1', 1, [4], 0, 0);
    WTensor(S, 't2', 2, [256, 2], 12, 32);
    PadTo(S, 128); { 24 + 34 + 42 = 100 after the directory; 100 -> 128 }
    D := MakePat(16, 1);
    S.WriteBuffer(D[0], 16);
    PadTo(S, 160); { t2 at data-absolute 32 (128 + 32) }
    D := MakePat(288, 7);
    S.WriteBuffer(D[0], 288);
    S.SaveToFile(P);
  finally
    S.Free;
  end;
  AOut := TStringList.Create;
  AErr := TStringList.Create;
  try
    AssertEquals(0, InspectCli.RunInspectCLI(['--tensors', P], AOut, AErr));
    AssertEquals(0, AErr.Count);
    AssertEquals(11, AOut.Count); { 7 summary + 2 type + 2 tensor lines (case 005's itemization) }
    AssertEquals('version' + K_TAB + '3', AOut[0]);
    AssertEquals('alignment' + K_TAB + '32', AOut[1]);
    AssertEquals('kv' + K_TAB + '0', AOut[2]);
    AssertEquals('tensors' + K_TAB + '2', AOut[3]);
    AssertEquals('data_offset' + K_TAB + '128', AOut[4]);
    AssertEquals('data_end' + K_TAB + '448', AOut[5]);
    AssertEquals('file_size' + K_TAB + '448', AOut[6]);
    AssertEquals('type' + K_TAB + 'f32' + K_TAB + '1', AOut[7]);
    AssertEquals('type' + K_TAB + 'q4_K' + K_TAB + '1', AOut[8]);
    AssertEquals('tensor' + K_TAB + 't1' + K_TAB + 'f32' + K_TAB + '4' + K_TAB + '0' + K_TAB + '16', AOut[9]);
    AssertEquals('tensor' + K_TAB + 't2' + K_TAB + 'q4_K' + K_TAB + '256 2' + K_TAB + '32' + K_TAB + '288', AOut[10]);
  finally
    AOut.Free;
    AErr.Free;
  end;
end;

{ 006: both flags: AOut order summary, type, kv, tensor -- exact AOut, rc 0 }
procedure TInspectCliTest.TestUT_InspectCli_006;
var
  S: TBytesStream;
  P: string;
  AOut, AErr: TStringList;
  D: TBytes;
begin
  P := NextPath('UT_InspectCli_006_a');
  S := TBytesStream.Create;
  try
    WHeader(S, 3, 2, 2);
    WVal(S, 'k1', 0, 1, 1);
    WValStr(S, 'k2', 'v');
    WTensor(S, 't1', 1, [4], 0, 0);
    WTensor(S, 't2', 1, [8], 0, 32);
    PadTo(S, 160);
    D := MakePat(16, 1);
    S.WriteBuffer(D[0], 16);
    PadTo(S, 192); { t2 at data-absolute 32 }
    D := MakePat(32, 50);
    S.WriteBuffer(D[0], 32);
    S.SaveToFile(P);
  finally
    S.Free;
  end;
  AOut := TStringList.Create;
  AErr := TStringList.Create;
  try
    AssertEquals(0, InspectCli.RunInspectCLI(['--kv', '--tensors', P], AOut, AErr));
    AssertEquals(0, AErr.Count);
    AssertEquals(12, AOut.Count);
    AssertEquals('version' + K_TAB + '3', AOut[0]);
    AssertEquals('alignment' + K_TAB + '32', AOut[1]);
    AssertEquals('kv' + K_TAB + '2', AOut[2]);
    AssertEquals('tensors' + K_TAB + '2', AOut[3]);
    AssertEquals('data_offset' + K_TAB + '160', AOut[4]);
    AssertEquals('data_end' + K_TAB + '224', AOut[5]);
    AssertEquals('file_size' + K_TAB + '224', AOut[6]);
    AssertEquals('type' + K_TAB + 'f32' + K_TAB + '2', AOut[7]);
    AssertEquals('kv' + K_TAB + 'k1' + K_TAB + 'uint8', AOut[8]);
    AssertEquals('kv' + K_TAB + 'k2' + K_TAB + 'string', AOut[9]);
    AssertEquals('tensor' + K_TAB + 't1' + K_TAB + 'f32' + K_TAB + '4' + K_TAB + '0' + K_TAB + '16', AOut[10]);
    AssertEquals('tensor' + K_TAB + 't2' + K_TAB + 'f32' + K_TAB + '8' + K_TAB + '32' + K_TAB + '32', AOut[11]);
  finally
    AOut.Free;
    AErr.Free;
  end;
end;

{ 007: no FILE: rc 2, usage exactly on AErr, AOut empty }
procedure TInspectCliTest.TestUT_InspectCli_007;
var
  AOut, AErr: TStringList;
begin
  AOut := TStringList.Create;
  AErr := TStringList.Create;
  try
    AssertEquals(2, InspectCli.RunInspectCLI(['--kv'], AOut, AErr));
    AssertEquals(0, AOut.Count);
    AssertEquals(1, AErr.Count);
    AssertEquals(K_USAGE, AErr[0]);
  finally
    AOut.Free;
    AErr.Free;
  end;
end;

{ 008: unrecognized option FILE: same as 007, the file is never opened }
procedure TInspectCliTest.TestUT_InspectCli_008;
var
  AOut, AErr: TStringList;
begin
  AOut := TStringList.Create;
  AErr := TStringList.Create;
  try
    AssertEquals(2, InspectCli.RunInspectCLI(['--bogus', K_DIR + '/no_such_file.gguf'], AOut, AErr));
    AssertEquals(0, AOut.Count);
    AssertEquals(1, AErr.Count);
    AssertEquals(K_USAGE, AErr[0]);
  finally
    AOut.Free;
    AErr.Free;
  end;
end;

{ 009: bad magic: rc 1, AErr = `error: ` + the EGgufError message, AOut empty }
procedure TInspectCliTest.TestUT_InspectCli_009;
var
  S: TBytesStream;
  P: string;
  AOut, AErr: TStringList;
  I: Integer;
begin
  P := NextPath('UT_InspectCli_009_a');
  S := TBytesStream.Create;
  try
    for I := 1 to 4 do
      WB(S, 88); { "XXXX", not "GGUF" }
    WU32(S, 2);
    WU64(S, 0);
    WU64(S, 0);
    S.SaveToFile(P);
  finally
    S.Free;
  end;
  AOut := TStringList.Create;
  AErr := TStringList.Create;
  try
    AssertEquals(1, InspectCli.RunInspectCLI([P], AOut, AErr));
    AssertEquals(0, AOut.Count);
    AssertEquals(1, AErr.Count);
    AssertEquals('error: gguf: bad magic at position 0 (expected "GGUF")', AErr[0]);
  finally
    AOut.Free;
    AErr.Free;
  end;
end;

{ 010: one extra byte after the last tensor: data_end < file_size -- rc 1,
  error line exact, AOut still written in full }
procedure TInspectCliTest.TestUT_InspectCli_010;
var
  S: TBytesStream;
  P: string;
  AOut, AErr: TStringList;
  D: TBytes;
  B: Byte;
begin
  P := NextPath('UT_InspectCli_010_a');
  S := TBytesStream.Create;
  try
    WHeader(S, 3, 1, 2);
    WVal(S, 'k1', 0, 1, 1);
    WValStr(S, 'k2', 'v');
    WTensor(S, 't1', 1, [4], 0, 0);
    PadTo(S, 96);
    D := MakePat(16, 1);
    S.WriteBuffer(D[0], 16);
    B := 255;
    S.WriteBuffer(B, 1); { trailing garbage byte: file_size = 113 > data_end 112 }
    S.SaveToFile(P);
  finally
    S.Free;
  end;
  AOut := TStringList.Create;
  AErr := TStringList.Create;
  try
    AssertEquals(1, InspectCli.RunInspectCLI([P], AOut, AErr));
    AssertEquals(1, AErr.Count);
    AssertEquals('error: data ends at 112, file size 113', AErr[0]);
    AssertEquals(8, AOut.Count);
    AssertEquals('version' + K_TAB + '3', AOut[0]);
    AssertEquals('alignment' + K_TAB + '32', AOut[1]);
    AssertEquals('kv' + K_TAB + '2', AOut[2]);
    AssertEquals('tensors' + K_TAB + '1', AOut[3]);
    AssertEquals('data_offset' + K_TAB + '96', AOut[4]);
    AssertEquals('data_end' + K_TAB + '112', AOut[5]);
    AssertEquals('file_size' + K_TAB + '113', AOut[6]);
    AssertEquals('type' + K_TAB + 'f32' + K_TAB + '1', AOut[7]);
  finally
    AOut.Free;
    AErr.Free;
  end;
end;

{ 011: 1 KV, 0 tensors: data_end = data_offset = file_size -- rc 0, no type lines }
procedure TInspectCliTest.TestUT_InspectCli_011;
var
  S: TBytesStream;
  P: string;
  AOut, AErr: TStringList;
begin
  P := NextPath('UT_InspectCli_011_a');
  S := TBytesStream.Create;
  try
    WHeader(S, 3, 0, 1);
    WVal(S, 'k', 0, 1, 1);
    PadTo(S, 64);
    S.SaveToFile(P);
  finally
    S.Free;
  end;
  AOut := TStringList.Create;
  AErr := TStringList.Create;
  try
    AssertEquals(0, InspectCli.RunInspectCLI([P], AOut, AErr));
    AssertEquals(0, AErr.Count);
    AssertEquals(7, AOut.Count);
    AssertEquals('version' + K_TAB + '3', AOut[0]);
    AssertEquals('alignment' + K_TAB + '32', AOut[1]);
    AssertEquals('kv' + K_TAB + '1', AOut[2]);
    AssertEquals('tensors' + K_TAB + '0', AOut[3]);
    AssertEquals('data_offset' + K_TAB + '64', AOut[4]);
    AssertEquals('data_end' + K_TAB + '64', AOut[5]);
    AssertEquals('file_size' + K_TAB + '64', AOut[6]);
  finally
    AOut.Free;
    AErr.Free;
  end;
end;

var
  App: TTestRunner;
begin
  ForceDirectories(K_DIR);
  Randomize;
  RegisterTest(TInspectCliTest);
  App := TTestRunner.Create(nil);
  App.Initialize;
  App.Title := 'InspectCli UT (T01c)';
  App.Run;
  App.Free;
end.
