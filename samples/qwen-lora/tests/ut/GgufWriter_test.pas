{ UT program for src/GgufWriter.pas: the rows UT-GgufWriter-NNN of
  tests/ut/cases/GgufWriter.cases.md (stage 1 from 001, stage 2 from 101).
  The sources are small GGUF files written by WriteSmallGguf below (the
  harness owner's helper, independent of the unit under test); the
  written files are read back with GgufFile (T01) and compared as bytes.
  Invoked with `--all --format=plain` (see .sdlc/ut-coverage.sh).

  Skeleton by the harness owner; the test methods are written in T04b. }
program GgufWriter_test;

{$mode objfpc}{$H+}

uses
  SysUtils, Classes, GgufFile, GgufWriter,
  fpcunit, testregistry, consoletestrunner;

type
  TGgufWriterTest = class(TTestCase)
  private
    FDir: string;
  protected
    procedure SetUp; override;
    procedure TearDown; override;
  public
    { Writes the small source file of T04-facts "Small source file" to
      APath: one key general.architecture = "test" (and, when AAlign64,
      general.alignment = 64), tensors a (f32 [3]), w (q4_K [256, 2]),
      b (f32 [5, 1]) unless ANoTensors; tensor data byte k of the data
      section (k counted from the data offset) is (k * 7 + 1) mod 256,
      padding bytes are 0. }
    class procedure WriteSmallGguf(const APath: string; AAlign64, ANoTensors: Boolean); static;
    { The whole file as bytes. }
    class function FileBytes(const APath: string): TBytes; static;
    { The fresh empty directory of this test (deleted after the test). }
    property Dir: string read FDir;
  published
    { T04b adds TestUT_GgufWriter_001.. (stage 1) and 101.. (stage 2). }
    procedure TestUT_GgufWriter_001;
    procedure TestUT_GgufWriter_002;
    procedure TestUT_GgufWriter_003;
    procedure TestUT_GgufWriter_004;
    procedure TestUT_GgufWriter_005;
    procedure TestUT_GgufWriter_006;
    procedure TestUT_GgufWriter_007;
    procedure TestUT_GgufWriter_008;
    procedure TestUT_GgufWriter_009;
    procedure TestUT_GgufWriter_010;
    procedure TestUT_GgufWriter_101;
    procedure TestUT_GgufWriter_102;
    procedure TestUT_GgufWriter_103;
    procedure TestUT_GgufWriter_104;
    procedure TestUT_GgufWriter_105;
    procedure TestUT_GgufWriter_106;
    procedure TestUT_GgufWriter_107;
    procedure TestUT_GgufWriter_108;
    procedure TestUT_GgufWriter_109;
    procedure TestUT_GgufWriter_110;
    procedure TestUT_GgufWriter_111;
    procedure TestUT_GgufWriter_112;
    procedure TestUT_GgufWriter_113;
    procedure TestUT_GgufWriter_114;
    procedure TestUT_GgufWriter_115;
    procedure TestUT_GgufWriter_116;
  end;

var
  DirSeq: Integer = 0;

procedure TGgufWriterTest.SetUp;
begin
  Inc(DirSeq);
  FDir := IncludeTrailingPathDelimiter(GetTempDir(False)) +
    Format('gguf-writer-test-%d-%d', [GetProcessID, DirSeq]);
  ForceDirectories(FDir);
end;

procedure TGgufWriterTest.TearDown;
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

class procedure TGgufWriterTest.WriteSmallGguf(const APath: string; AAlign64, ANoTensors: Boolean);
const
  Names: array[0..2] of RawByteString = ('a', 'w', 'b');
  NDims: array[0..2] of Cardinal = (1, 2, 2);
  Dims: array[0..2, 0..1] of QWord = ((3, 1), (256, 2), (5, 1));
  Types: array[0..2] of Cardinal = (0, 12, 0);
  Sizes: array[0..2] of QWord = (12, 288, 20);
var
  S: TFileStream;
  Align, Off, K, NT, NKV: QWord;
  I, J: Integer;
  U32: Cardinal;

  procedure W8(AV: QWord); begin S.WriteBuffer(AV, 8); end;
  procedure W4(AV: Cardinal); begin S.WriteBuffer(AV, 4); end;
  procedure WStr(const AText: RawByteString);
  begin
    W8(Length(AText));
    if AText <> '' then S.WriteBuffer(AText[1], Length(AText));
  end;
  procedure PadTo(AA: QWord);
  var Z: Byte = 0;
  begin
    while S.Position mod AA <> 0 do S.WriteBuffer(Z, 1);
  end;

begin
  if AAlign64 then Align := 64 else Align := 32;
  if ANoTensors then NT := 0 else NT := 3;
  if AAlign64 then NKV := 2 else NKV := 1;
  S := TFileStream.Create(APath, fmCreate);
  try
    S.WriteBuffer(PAnsiChar('GGUF')^, 4);
    W4(3); W8(NT); W8(NKV);
    WStr('general.architecture'); W4(8); WStr('test');
    if AAlign64 then
    begin
      WStr('general.alignment'); W4(4); U32 := 64; S.WriteBuffer(U32, 4);
    end;
    Off := 0;
    for I := 0 to Integer(NT) - 1 do
    begin
      WStr(Names[I]); W4(NDims[I]);
      for J := 0 to Integer(NDims[I]) - 1 do W8(Dims[I, J]);
      W4(Types[I]); W8(Off);
      Off := ((Off + Sizes[I] + Align - 1) div Align) * Align;
    end;
    PadTo(Align);
    K := 0;
    for I := 0 to Integer(NT) - 1 do
    begin
      for J := 0 to Integer(Sizes[I]) - 1 do
      begin
        U32 := (K * 7 + 1) mod 256;
        S.WriteBuffer(U32, 1);
        Inc(K);
      end;
      U32 := 0;
      while K mod Align <> 0 do
      begin
        S.WriteBuffer(U32, 1);
        Inc(K);
      end;
    end;
  finally
    S.Free;
  end;
end;

class function TGgufWriterTest.FileBytes(const APath: string): TBytes;
var
  S: TFileStream;
begin
  S := TFileStream.Create(APath, fmOpenRead or fmShareDenyNone);
  try
    SetLength(Result, S.Size);
    if S.Size > 0 then
      S.ReadBuffer(Result[0], S.Size);
  finally
    S.Free;
  end;
end;

{ Test methods 001-010 }

procedure TGgufWriterTest.TestUT_GgufWriter_001;
var
  SrcPath: string;
  Src: TGgufFile;
  Writer: TGgufWriter;
begin
  SrcPath := Dir + '/src.gguf';
  WriteSmallGguf(SrcPath, False, False); // align-32 source
  Src := TGgufFile.Open(SrcPath);
  Writer := TGgufWriter.Create(Src);
  try
    AssertEquals('TensorCount', 3, Writer.TensorCount);
    AssertEquals('Tensor[0] name', 'a', String(Writer.Tensor[0].Name));
    AssertEquals('Tensor[0] NDims', 1, Writer.Tensor[0].NDims);
    AssertEquals('Tensor[0] Dims[0]', QWord(3), Writer.Tensor[0].Dims[0]);
    AssertEquals('Tensor[0] TType', Cardinal(0), Writer.Tensor[0].TType);
    AssertEquals('Tensor[0] Offset', QWord(0), Writer.Tensor[0].Offset);
    AssertEquals('Tensor[0] NBytes', QWord(12), Writer.Tensor[0].NBytes);
    // w tensor
    AssertEquals('Tensor[1] name', 'w', String(Writer.Tensor[1].Name));
    AssertEquals('Tensor[1] NDims', 2, Writer.Tensor[1].NDims);
    AssertEquals('Tensor[1] Dims[0]', QWord(256), Writer.Tensor[1].Dims[0]);
    AssertEquals('Tensor[1] Dims[1]', QWord(2), Writer.Tensor[1].Dims[1]);
    AssertEquals('Tensor[1] TType', Cardinal(12), Writer.Tensor[1].TType);
    AssertEquals('Tensor[1] Offset', QWord(32), Writer.Tensor[1].Offset);
    AssertEquals('Tensor[1] NBytes', QWord(288), Writer.Tensor[1].NBytes);
    // b tensor
    AssertEquals('Tensor[2] name', 'b', String(Writer.Tensor[2].Name));
    AssertEquals('Tensor[2] NDims', 2, Writer.Tensor[2].NDims);
    AssertEquals('Tensor[2] Dims[0]', QWord(5), Writer.Tensor[2].Dims[0]);
    AssertEquals('Tensor[2] Dims[1]', QWord(1), Writer.Tensor[2].Dims[1]);
    AssertEquals('Tensor[2] TType', Cardinal(0), Writer.Tensor[2].TType);
    AssertEquals('Tensor[2] Offset', QWord(320), Writer.Tensor[2].Offset);
    AssertEquals('Tensor[2] NBytes', QWord(20), Writer.Tensor[2].NBytes);
    AssertEquals('DataOffset', QWord(192), Writer.DataOffset);
    AssertEquals('FileSize', QWord(544), Writer.FileSize);
  finally
    Writer.Free;
    Src.Free;
  end;
end;

procedure TGgufWriterTest.TestUT_GgufWriter_002;
var
  SrcPath: string;
  Src: TGgufFile;
  Writer: TGgufWriter;
begin
  SrcPath := Dir + '/src.gguf';
  WriteSmallGguf(SrcPath, True, False); // align-64 source
  Src := TGgufFile.Open(SrcPath);
  Writer := TGgufWriter.Create(Src);
  try
    AssertEquals('Tensor[0] Offset', QWord(0), Writer.Tensor[0].Offset);
    AssertEquals('Tensor[1] Offset', QWord(64), Writer.Tensor[1].Offset);
    AssertEquals('Tensor[2] Offset', QWord(384), Writer.Tensor[2].Offset);
    AssertEquals('DataOffset', QWord(256), Writer.DataOffset);
    AssertEquals('FileSize', QWord(704), Writer.FileSize);
  finally
    Writer.Free;
    Src.Free;
  end;
end;

procedure TGgufWriterTest.TestUT_GgufWriter_003;
var
  SrcPath: string;
  Src: TGgufFile;
  Writer: TGgufWriter;
begin
  SrcPath := Dir + '/src.gguf';
  WriteSmallGguf(SrcPath, False, True); // source without tensors
  Src := TGgufFile.Open(SrcPath);
  Writer := TGgufWriter.Create(Src);
  try
    AssertEquals('TensorCount', 0, Writer.TensorCount);
    AssertEquals('DataOffset', QWord(96), Writer.DataOffset);
    AssertEquals('FileSize', QWord(96), Writer.FileSize);
  finally
    Writer.Free;
    Src.Free;
  end;
end;

procedure TGgufWriterTest.TestUT_GgufWriter_004;
var
  SrcPath: string;
  Src: TGgufFile;
  Writer: TGgufWriter;
begin
  SrcPath := Dir + '/src.gguf';
  WriteSmallGguf(SrcPath, False, False);
  Src := TGgufFile.Open(SrcPath);
  Writer := TGgufWriter.Create(Src);
  try
    Writer.SetTensorType('w', 8); // Q8_0
    AssertEquals('w type after set', Cardinal(8), Writer.Tensor[1].TType);
    AssertEquals('w NBytes after set', QWord(544), Writer.Tensor[1].NBytes);
    AssertEquals('b offset after w retype', QWord(576), Writer.Tensor[2].Offset);
    AssertEquals('FileSize after w retype', QWord(800), Writer.FileSize);
  finally
    Writer.Free;
    Src.Free;
  end;
end;

procedure TGgufWriterTest.TestUT_GgufWriter_005;
var
  SrcPath: string;
  Src: TGgufFile;
  Writer: TGgufWriter;
begin
  SrcPath := Dir + '/src.gguf';
  WriteSmallGguf(SrcPath, False, False);
  Src := TGgufFile.Open(SrcPath);
  Writer := TGgufWriter.Create(Src);
  try
    Writer.SetTensorType('w', 8);
    Writer.SetTensorType('w', 12); // back to q4_K
    // verify back to original plan values
    AssertEquals('w type back', Cardinal(12), Writer.Tensor[1].TType);
    AssertEquals('TensorCount unchanged', 3, Writer.TensorCount);
    // offsets should match original
    AssertEquals('w Offset original', QWord(32), Writer.Tensor[1].Offset);
    AssertEquals('FileSize back to original', QWord(544), Writer.FileSize);
  finally
    Writer.Free;
    Src.Free;
  end;
end;

procedure TGgufWriterTest.TestUT_GgufWriter_006;
var
  SrcPath: string;
  Src: TGgufFile;
  Writer: TGgufWriter;
begin
  SrcPath := Dir + '/src.gguf';
  WriteSmallGguf(SrcPath, False, False);
  Src := TGgufFile.Open(SrcPath);
  Writer := TGgufWriter.Create(Src);
  try
    Writer.SetTensorType('a', 1); // f16
    // a bytes should be 6, offsets unchanged for w and b
    AssertEquals('a NBytes f16', QWord(6), Writer.Tensor[0].NBytes);
    AssertEquals('w Offset unchanged', QWord(32), Writer.Tensor[1].Offset);
    AssertEquals('b Offset unchanged', QWord(320), Writer.Tensor[2].Offset);
    AssertEquals('FileSize after a f16', QWord(544), Writer.FileSize);
  finally
    Writer.Free;
    Src.Free;
  end;
end;

procedure TGgufWriterTest.TestUT_GgufWriter_007;
var
  SrcPath: string;
  Src: TGgufFile;
  Writer: TGgufWriter;
begin
  SrcPath := Dir + '/src.gguf';
  WriteSmallGguf(SrcPath, False, False);
  Src := TGgufFile.Open(SrcPath);
  Writer := TGgufWriter.Create(Src);
  try
    try
      Writer.SetTensorType('x', 8);
      Fail('Expected EGgufError for unknown tensor name');
    except
      on E: EGgufError do
        AssertTrue('Error message contains x', Pos('x', E.Message) > 0);
    end;
  finally
    Writer.Free;
    Src.Free;
  end;
end;

procedure TGgufWriterTest.TestUT_GgufWriter_008;
var
  SrcPath: string;
  Src: TGgufFile;
  Writer: TGgufWriter;
begin
  SrcPath := Dir + '/src.gguf';
  WriteSmallGguf(SrcPath, False, False);
  Src := TGgufFile.Open(SrcPath);
  Writer := TGgufWriter.Create(Src);
  try
    try
      Writer.SetTensorType('W', 8); // upper case name
      Fail('Expected EGgufError for unknown tensor name case-sensitive');
    except
      on E: EGgufError do
        AssertTrue('Error message contains W', Pos('W', E.Message) > 0);
    end;
  finally
    Writer.Free;
    Src.Free;
  end;
end;

procedure TGgufWriterTest.TestUT_GgufWriter_009;
var
  SrcPath: string;
  Src: TGgufFile;
  Writer: TGgufWriter;
begin
  SrcPath := Dir + '/src.gguf';
  WriteSmallGguf(SrcPath, False, False);
  Src := TGgufFile.Open(SrcPath);
  Writer := TGgufWriter.Create(Src);
  try
    try
      Writer.SetTensorType('b', 8); // row length 5 not multiple of block 32
      Fail('Expected EGgufError for row length mismatch');
    except
      on E: EGgufError do
        AssertTrue('Error message about row length', Pos('row length', E.Message) > 0);
    end;
    // after exception, verify plan is same as row 001
    AssertEquals('TensorCount', 3, Writer.TensorCount);
    AssertEquals('Tensor[0] name', 'a', String(Writer.Tensor[0].Name));
    AssertEquals('Tensor[0] NDims', 1, Writer.Tensor[0].NDims);
    AssertEquals('Tensor[0] Dims[0]', QWord(3), Writer.Tensor[0].Dims[0]);
    AssertEquals('Tensor[0] TType', Cardinal(0), Writer.Tensor[0].TType);
    AssertEquals('Tensor[0] Offset', QWord(0), Writer.Tensor[0].Offset);
    AssertEquals('Tensor[0] NBytes', QWord(12), Writer.Tensor[0].NBytes);
    AssertEquals('Tensor[1] name', 'w', String(Writer.Tensor[1].Name));
    AssertEquals('Tensor[1] NDims', 2, Writer.Tensor[1].NDims);
    AssertEquals('Tensor[1] Dims[0]', QWord(256), Writer.Tensor[1].Dims[0]);
    AssertEquals('Tensor[1] Dims[1]', QWord(2), Writer.Tensor[1].Dims[1]);
    AssertEquals('Tensor[1] TType', Cardinal(12), Writer.Tensor[1].TType);
    AssertEquals('Tensor[1] Offset', QWord(32), Writer.Tensor[1].Offset);
    AssertEquals('Tensor[1] NBytes', QWord(288), Writer.Tensor[1].NBytes);
    AssertEquals('Tensor[2] name', 'b', String(Writer.Tensor[2].Name));
    AssertEquals('Tensor[2] NDims', 2, Writer.Tensor[2].NDims);
    AssertEquals('Tensor[2] Dims[0]', QWord(5), Writer.Tensor[2].Dims[0]);
    AssertEquals('Tensor[2] Dims[1]', QWord(1), Writer.Tensor[2].Dims[1]);
    AssertEquals('Tensor[2] TType', Cardinal(0), Writer.Tensor[2].TType);
    AssertEquals('Tensor[2] Offset', QWord(320), Writer.Tensor[2].Offset);
    AssertEquals('Tensor[2] NBytes', QWord(20), Writer.Tensor[2].NBytes);
    AssertEquals('DataOffset', QWord(192), Writer.DataOffset);
    AssertEquals('FileSize', QWord(544), Writer.FileSize);
  finally
    Writer.Free;
    Src.Free;
  end;
end;

procedure TGgufWriterTest.TestUT_GgufWriter_010;
var
  SrcPath: string;
  Src: TGgufFile;
  Writer: TGgufWriter;
begin
  SrcPath := Dir + '/src.gguf';
  WriteSmallGguf(SrcPath, False, False);
  Src := TGgufFile.Open(SrcPath);
  Writer := TGgufWriter.Create(Src);
  try
    try
      Writer.SetTensorType('w', 99); // unsupported type
      Fail('Expected EGgufError for unsupported type');
    except
      on E: EGgufError do
        AssertTrue('Error message contains type 99', Pos('99', E.Message) > 0);
    end;
    // after exception, verify plan is same as row 001
    AssertEquals('TensorCount', 3, Writer.TensorCount);
    AssertEquals('Tensor[0] name', 'a', String(Writer.Tensor[0].Name));
    AssertEquals('Tensor[0] NDims', 1, Writer.Tensor[0].NDims);
    AssertEquals('Tensor[0] Dims[0]', QWord(3), Writer.Tensor[0].Dims[0]);
    AssertEquals('Tensor[0] TType', Cardinal(0), Writer.Tensor[0].TType);
    AssertEquals('Tensor[0] Offset', QWord(0), Writer.Tensor[0].Offset);
    AssertEquals('Tensor[0] NBytes', QWord(12), Writer.Tensor[0].NBytes);
    AssertEquals('Tensor[1] name', 'w', String(Writer.Tensor[1].Name));
    AssertEquals('Tensor[1] NDims', 2, Writer.Tensor[1].NDims);
    AssertEquals('Tensor[1] Dims[0]', QWord(256), Writer.Tensor[1].Dims[0]);
    AssertEquals('Tensor[1] Dims[1]', QWord(2), Writer.Tensor[1].Dims[1]);
    AssertEquals('Tensor[1] TType', Cardinal(12), Writer.Tensor[1].TType);
    AssertEquals('Tensor[1] Offset', QWord(32), Writer.Tensor[1].Offset);
    AssertEquals('Tensor[1] NBytes', QWord(288), Writer.Tensor[1].NBytes);
    AssertEquals('Tensor[2] name', 'b', String(Writer.Tensor[2].Name));
    AssertEquals('Tensor[2] NDims', 2, Writer.Tensor[2].NDims);
    AssertEquals('Tensor[2] Dims[0]', QWord(5), Writer.Tensor[2].Dims[0]);
    AssertEquals('Tensor[2] Dims[1]', QWord(1), Writer.Tensor[2].Dims[1]);
    AssertEquals('Tensor[2] TType', Cardinal(0), Writer.Tensor[2].TType);
    AssertEquals('Tensor[2] Offset', QWord(320), Writer.Tensor[2].Offset);
    AssertEquals('Tensor[2] NBytes', QWord(20), Writer.Tensor[2].NBytes);
    AssertEquals('DataOffset', QWord(192), Writer.DataOffset);
    AssertEquals('FileSize', QWord(544), Writer.FileSize);
  finally
    Writer.Free;
    Src.Free;
  end;
end;

{--- Stage 2 UT methods ---}

procedure TGgufWriterTest.TestUT_GgufWriter_101;
var
  SrcPath, DstPath: string;
  SrcBytes, DstBytes, PaddingBytes: TBytes;
begin
  SrcPath := Dir + '/src.gguf';
  WriteSmallGguf(SrcPath, False, False);
  DstPath := Dir + '/copy.gguf';
  GgufCopy(SrcPath, DstPath);
  SrcBytes := FileBytes(SrcPath);
  DstBytes := FileBytes(DstPath);
  AssertEquals('copy size', Length(SrcBytes), Length(DstBytes));
  AssertTrue('bytes equal', CompareMem(@SrcBytes[0], @DstBytes[0], Length(SrcBytes)));
end;

procedure TGgufWriterTest.TestUT_GgufWriter_102;
var
  SrcPath, DstPath: string;
  SrcBytes, DstBytes: TBytes;
begin
  SrcPath := Dir + '/src.gguf';
  WriteSmallGguf(SrcPath, True, False);
  DstPath := Dir + '/copy.gguf';
  GgufCopy(SrcPath, DstPath);
  SrcBytes := FileBytes(SrcPath);
  DstBytes := FileBytes(DstPath);
  AssertEquals('copy size', Length(SrcBytes), Length(DstBytes));
  AssertTrue('bytes equal', CompareMem(@SrcBytes[0], @DstBytes[0], Length(SrcBytes)));
end;

procedure TGgufWriterTest.TestUT_GgufWriter_103;
var
  SrcPath, DstPath: string;
  SrcBytes, DstBytes: TBytes;
begin
  SrcPath := Dir + '/src.gguf';
  WriteSmallGguf(SrcPath, False, True);
  DstPath := Dir + '/copy.gguf';
  GgufCopy(SrcPath, DstPath);
  SrcBytes := FileBytes(SrcPath);
  DstBytes := FileBytes(DstPath);
  AssertEquals('copy size', Length(SrcBytes), Length(DstBytes));
  AssertTrue('bytes equal', CompareMem(@SrcBytes[0], @DstBytes[0], Length(SrcBytes)));
end;

procedure TGgufWriterTest.TestUT_GgufWriter_104;
var
  SrcPath, DstPath: string;
  PrevDir: string;
  SrcBytes, DstBytes: TBytes;
begin
  SrcPath := Dir + '/src.gguf';
  WriteSmallGguf(SrcPath, False, False);
  DstPath := Dir + '/copy.gguf';
  PrevDir := GetCurrentDir;
  try
    SetCurrentDir(Dir);
    GgufCopy('src.gguf', 'copy.gguf');
  finally
    SetCurrentDir(PrevDir);
  end;
  SrcBytes := FileBytes(SrcPath);
  DstBytes := FileBytes(DstPath);
  AssertEquals('copy size', Length(SrcBytes), Length(DstBytes));
  AssertTrue('bytes equal', CompareMem(@SrcBytes[0], @DstBytes[0], Length(SrcBytes)));
end;

procedure TGgufWriterTest.TestUT_GgufWriter_105;
var
  SrcPath, DstPath: string;
  Src: TGgufFile;
  Writer: TGgufWriter;
  I: Integer;
  Data, ReadBack, PaddingBytes: TBytes;
  Cur: TGgufTensorInfo;
  J: Integer;
  Expected: array of TBytes;
begin
  SrcPath := Dir + '/src.gguf';
  WriteSmallGguf(SrcPath, False, False);
  DstPath := Dir + '/copy.gguf';
  Src := TGgufFile.Open(SrcPath);
  Writer := TGgufWriter.Create(Src);
  try
    Writer.SetTensorType('w', 8); // Q8_0
    Writer.BeginWrite(DstPath);
    // generate and write patterned tensor bytes, then verify after reading back
    SetLength(Expected, Writer.TensorCount);
    for I := 0 to Writer.TensorCount - 1 do
    begin
      // determine size from writer's tensor info
      Cur := Writer.Tensor[I];
      SetLength(Data, Cur.NBytes);
      for J := 0 to Cur.NBytes - 1 do
        Data[J] := Byte((I * 3 + J + 5) mod 256);
      Expected[I] := Copy(Data);
      Writer.WriteTensor(Data);
    end;
    Writer.Finish;
    // verify written tensor data matches the generated pattern
    Src.Free; // close source
    Src := TGgufFile.Open(DstPath);
    for I := 0 to Writer.TensorCount - 1 do
    begin
      Src.ReadTensorBytes(I, ReadBack);
      AssertEquals('tensor data length', Length(Expected[I]), Length(ReadBack));
      AssertTrue('tensor bytes equal', CompareMem(@Expected[I][0], @ReadBack[0], Length(ReadBack)));
    end;
  finally
    Writer.Free;
    Src.Free;
  end;
  // verify output file properties
  Src := TGgufFile.Open(DstPath);
  try
    AssertEquals('TensorCount', 3, Src.TensorCount);
    AssertEquals('KVCount', 1, Src.KVCount);
    AssertEquals('DataOffset', QWord(192), Src.DataOffset);
    AssertEquals('general.architecture', 'test', String(Src.KV[Src.FindKey('general.architecture')].StrVal));
    // verify padding bytes are zero
    PaddingBytes := FileBytes(DstPath);
    for I := 0 to 19 do
      AssertEquals('padding zero', Byte(0), PaddingBytes[Integer(Src.DataOffset)+12+I]);
  finally
    Src.Free;
  end;
end;

procedure TGgufWriterTest.TestUT_GgufWriter_106;
var
  SrcPath: string;
  Src: TGgufFile;
  Writer: TGgufWriter;
begin
  SrcPath := Dir + '/src.gguf';
  WriteSmallGguf(SrcPath, False, False);
  Src := TGgufFile.Open(SrcPath);
  Writer := TGgufWriter.Create(Src);
  try
    Writer.BeginWrite(Dir + '/out.gguf');
    try
      Writer.BeginWrite(Dir + '/out2.gguf');
      Fail('Expected EGgufError for second BeginWrite');
    except
      on EGgufError do ; // expected
    end;
  finally
    Writer.Free;
    Src.Free;
  end;
end;

procedure TGgufWriterTest.TestUT_GgufWriter_107;
var
  SrcPath: string;
  Src: TGgufFile;
  Writer: TGgufWriter;
begin
  SrcPath := Dir + '/src.gguf';
  WriteSmallGguf(SrcPath, False, False);
  Src := TGgufFile.Open(SrcPath);
  Writer := TGgufWriter.Create(Src);
  try
    try
      Writer.BeginWrite('/nonexistent-dir/x.gguf');
      Fail('Expected EGgufError for bad path');
    except
      on E: EGgufError do
        AssertTrue('Path in message', Pos('/nonexistent-dir/x.gguf', E.Message) > 0);
    end;
  finally
    Writer.Free;
    Src.Free;
  end;
end;

procedure TGgufWriterTest.TestUT_GgufWriter_108;
var
  SrcPath: string;
  Src: TGgufFile;
  Writer: TGgufWriter;
begin
  SrcPath := Dir + '/src.gguf';
  WriteSmallGguf(SrcPath, False, False);
  Src := TGgufFile.Open(SrcPath);
  Writer := TGgufWriter.Create(Src);
  try
    try
      Writer.WriteTensor([]);
      Fail('Expected EGgufError for WriteTensor before BeginWrite');
    except
      on EGgufError do ; // expected
    end;
  finally
    Writer.Free;
    Src.Free;
  end;
end;

procedure TGgufWriterTest.TestUT_GgufWriter_109;
var
  SrcPath: string;
  Src: TGgufFile;
  Writer: TGgufWriter;
  BadBytes: TBytes;
  Data: TBytes;
begin
  SrcPath := Dir + '/src.gguf';
  WriteSmallGguf(SrcPath, False, False);
  Src := TGgufFile.Open(SrcPath);
  Writer := TGgufWriter.Create(Src);
  try
    Writer.BeginWrite(Dir + '/out.gguf');
    Src.ReadTensorBytes(0, Data);
    SetLength(BadBytes, 11);
    FillChar(BadBytes[0], 11, $AA);
    try
      Writer.WriteTensor(BadBytes); // 11 bytes for 'a', expected 12
    except
      on E: EGgufError do
      begin
        AssertTrue('Message contains a', Pos('a', E.Message) > 0);
        AssertTrue('Message contains 12', Pos('12', E.Message) > 0);
        AssertTrue('Message contains 11', Pos('11', E.Message) > 0);
      end;
    end;
  finally
    Writer.Free;
    Src.Free;
  end;
end;

procedure TGgufWriterTest.TestUT_GgufWriter_110;
var
  SrcPath: string;
  Src: TGgufFile;
  Writer: TGgufWriter;
  Data1, Data2, Data3: TBytes;
begin
  SrcPath := Dir + '/src.gguf';
  WriteSmallGguf(SrcPath, False, False);
  Src := TGgufFile.Open(SrcPath);
  Writer := TGgufWriter.Create(Src);
  try
    Writer.BeginWrite(Dir + '/out.gguf');
    Src.ReadTensorBytes(0, Data1);
    Writer.WriteTensor(Data1);
    Src.ReadTensorBytes(1, Data2);
    Writer.WriteTensor(Data2);
    Src.ReadTensorBytes(2, Data3);
    Writer.WriteTensor(Data3);
    try
      Writer.WriteTensor([]);
      Fail('Expected EGgufError for extra WriteTensor');
    except
      on EGgufError do ; // expected
    end;
  finally
    Writer.Free;
    Src.Free;
  end;
end;

procedure TGgufWriterTest.TestUT_GgufWriter_111;
var
  SrcPath: string;
  Src: TGgufFile;
  Writer: TGgufWriter;
begin
  SrcPath := Dir + '/src.gguf';
  WriteSmallGguf(SrcPath, False, False);
  Src := TGgufFile.Open(SrcPath);
  Writer := TGgufWriter.Create(Src);
  try
    try
      Writer.Finish;
      Fail('Expected EGgufError for Finish before BeginWrite');
    except
      on EGgufError do ; // expected
    end;
  finally
    Writer.Free;
  end;
end;

procedure TGgufWriterTest.TestUT_GgufWriter_112;
var
  SrcPath: string;
  Src: TGgufFile;
  Writer: TGgufWriter;
  Data: TBytes;
begin
  SrcPath := Dir + '/src.gguf';
  WriteSmallGguf(SrcPath, False, False);
  Src := TGgufFile.Open(SrcPath);
  Writer := TGgufWriter.Create(Src);
  try
    Writer.BeginWrite(Dir + '/out.gguf');
    Src.ReadTensorBytes(0, Data);
    Writer.WriteTensor(Data); // only one tensor
    try
      Writer.Finish;
      Fail('Expected EGgufError for incomplete tensors');
    except
      on E: EGgufError do
        AssertTrue('Message contains "1 of 3"', Pos('1 of 3', E.Message) > 0);
    end;
  finally
    Writer.Free;
    Src.Free;
  end;
end;

procedure TGgufWriterTest.TestUT_GgufWriter_113;
var
  SrcPath: string;
  Src: TGgufFile;
  Writer: TGgufWriter;
begin
  SrcPath := Dir + '/src.gguf';
  WriteSmallGguf(SrcPath, False, False);
  Src := TGgufFile.Open(SrcPath);
  Writer := TGgufWriter.Create(Src);
  try
    Writer.BeginWrite(Dir + '/out.gguf');
    try
      Writer.SetTensorType('a', 1);
      Fail('Expected EGgufError for SetTensorType after BeginWrite');
    except
      on EGgufError do ; // expected
    end;
  finally
    Writer.Free;
    Src.Free;
  end;
end;

procedure TGgufWriterTest.TestUT_GgufWriter_114;
var
  SrcPath: string;
  Src: TGgufFile;
  Writer: TGgufWriter;
  OutPath: string;
begin
  SrcPath := Dir + '/src.gguf';
  WriteSmallGguf(SrcPath, False, False);
  OutPath := Dir + '/out.gguf';
  Src := TGgufFile.Open(SrcPath);
  Writer := TGgufWriter.Create(Src);
  try
    Writer.BeginWrite(OutPath);
    // no Finish
  finally
    Writer.Free;
    Src.Free;
  end;
  AssertFalse('Output file should not exist', FileExists(OutPath));
end;

procedure TGgufWriterTest.TestUT_GgufWriter_115;
var
  DstPath: string;
begin
  DstPath := Dir + '/out.gguf';
  try
    GgufCopy('nonexistent.gguf', DstPath);
    Fail('Expected exception for missing source');
  except
    on E: Exception do ; // expected pass through
  end;
  AssertFalse('Destination should not be created', FileExists(DstPath));
end;

procedure TGgufWriterTest.TestUT_GgufWriter_116;
var
  SrcPath, DstPath: string;
  Src: TGgufFile;
  Writer: TGgufWriter;
  I: Integer;
  Data: TBytes;
begin
  SrcPath := Dir + '/src.gguf';
  WriteSmallGguf(SrcPath, False, False);
  DstPath := Dir + '/out.gguf';
  Src := TGgufFile.Open(SrcPath);
  Writer := TGgufWriter.Create(Src);
  try
    Writer.BeginWrite(DstPath);
    // write all tensors correctly
    for I := 0 to Writer.TensorCount - 1 do
    begin
      Src.ReadTensorBytes(I, Data);
      Writer.WriteTensor(Data);
    end;
    Writer.Finish;
    // second Finish should raise error (line 254)
    try
      Writer.Finish;
      Fail('Expected EGgufError for second Finish');
    except
      on EGgufError do ; // expected
    end;
    // WriteTensor after Finish should raise error (line 231)
    Src.ReadTensorBytes(0, Data);
    try
      Writer.WriteTensor(Data);
      Fail('Expected EGgufError for WriteTensor after Finish');
    except
      on EGgufError do ; // expected
    end;
  finally
    Writer.Free;
    Src.Free;
  end;
end;

var
  App: TTestRunner;
begin
  RegisterTest(TGgufWriterTest);
  App := TTestRunner.Create(nil);
  App.Initialize;
  App.Title := 'GgufWriter UT';
  App.Run;
  App.Free;
end.
