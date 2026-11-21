{ ITa driver of T04 (written by the harness owner; agents run it through
  tests/ita/T04-run.sh, they do not change it). Uses only the interfaces
  of GgufFile, GgufQuant and GgufWriter.
    writer_check copy SRC DST
      GgufCopy(SRC, DST).
    writer_check retype SRC DST NAMES
      TGgufWriter on SRC with SetTensorType(name, 8) for every line of
      NAMES; each retyped tensor is DequantizeBytes of its source bytes,
      then QuantizeQ8_0; every other tensor is its source bytes.
  Both modes print the planned layout on stdout, one line per tensor
  "index<TAB>name<TAB>type<TAB>offset<TAB>size", then "data_offset<TAB>N"
  and "file_size<TAB>N" (the format of work/oracle/T04/gguf_meta). }
program writer_check;

{$mode objfpc}{$H+}

uses
  Classes, SysUtils, GgufFile, GgufQuant, GgufWriter;

procedure PrintLayout(W: TGgufWriter);
var
  I: Integer;
  T: TGgufTensorInfo;
begin
  for I := 0 to W.TensorCount - 1 do
  begin
    T := W.Tensor[I];
    WriteLn(I, #9, T.Name, #9, T.TType, #9, T.Offset, #9, T.NBytes);
  end;
  WriteLn('data_offset'#9, W.DataOffset);
  WriteLn('file_size'#9, W.FileSize);
end;

procedure DoCopy(const ASrc, ADst: string);
var
  Src: TGgufFile;
  W: TGgufWriter;
begin
  Src := TGgufFile.Open(ASrc);
  try
    W := TGgufWriter.Create(Src);
    try
      PrintLayout(W);
    finally
      W.Free;
    end;
  finally
    Src.Free;
  end;
  GgufCopy(ASrc, ADst);
end;

procedure DoRetype(const ASrc, ADst, ANames: string);
var
  Src: TGgufFile;
  W: TGgufWriter;
  Names: TStringList;
  B: TBytes;
  T: TGgufTensorInfo;
  I: Integer;
  N: Int64;
begin
  Names := TStringList.Create;
  Src := TGgufFile.Open(ASrc);
  try
    Names.LoadFromFile(ANames);
    W := TGgufWriter.Create(Src);
    try
      for I := 0 to Names.Count - 1 do
        if Names[I] <> '' then
          W.SetTensorType(Names[I], 8);
      PrintLayout(W);
      W.BeginWrite(ADst);
      for I := 0 to Src.TensorCount - 1 do
      begin
        T := Src.Tensor[I];
        Src.ReadTensorBytes(I, B);
        if Names.IndexOf(T.Name) >= 0 then
        begin
          N := T.Dims[0] * T.Dims[1] * T.Dims[2] * T.Dims[3];
          B := QuantizeQ8_0(DequantizeBytes(T.TType, B, N));
        end;
        W.WriteTensor(B);
      end;
      W.Finish;
    finally
      W.Free;
    end;
  finally
    Src.Free;
    Names.Free;
  end;
end;

begin
  if (ParamCount = 3) and (ParamStr(1) = 'copy') then
    DoCopy(ParamStr(2), ParamStr(3))
  else if (ParamCount = 4) and (ParamStr(1) = 'retype') then
    DoRetype(ParamStr(2), ParamStr(3), ParamStr(4))
  else
  begin
    WriteLn(StdErr, 'usage: writer_check copy SRC DST | retype SRC DST NAMES');
    Halt(2);
  end;
end.
