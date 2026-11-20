{ InspectCLI (T01c) -- detailed design T01-detailed 2.12 / 3 / 4.20.

  Parses the command line, opens the GGUF file via GgufFile, and writes the
  inspect rows (summary, type, optional kv, optional tensor) as plain text
  into AOut / AErr.  Returns the exit code:
    0  inspected, data_end = file_size
    1  open failure (E.Message, or data_end <> file_size)
    2  usage error (no FILE, unrecognized option)
  All behavior lines live in this unit, measured in-process by
  tests/ut/InspectCli_test.pas (UT-InspectCli-001..011); the program
  src/qlora_inspect.pas (T01c step 2) only forwards. }
unit InspectCli;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, GgufFile;

{ AArgs: [--kv] [--tensors] FILE.  AOut / AErr are filled (caller-created,
  caller-owned); the Integer result is the exit code. }
function RunInspectCLI(const AArgs: array of string; AOut, AErr: TStrings): Integer;

implementation

const
  K_USAGE = 'Usage: qlora_inspect [--kv] [--tensors] FILE';

{ scalar type name, or array<elem>[count] for gvArray }
function KVTypeText(const AKV: TGgufKV): string;
begin
  if AKV.VType = gvArray then
    Result := 'array<' + GgufValueTypeName(AKV.ArrType) + '>['
      + IntToStr(Int64(AKV.Count)) + ']'
  else
    Result := GgufValueTypeName(AKV.VType);
end;

{ dim sizes in GGUF order, single spaces, e.g. "256 2" }
function DimsText(const AT: TGgufTensorInfo): string;
var
  I: Integer;
begin
  Result := '';
  for I := 0 to AT.NDims - 1 do
  begin
    if I > 0 then
      Result := Result + ' ';
    Result := Result + IntToStr(Int64(AT.Dims[I]));
  end;
end;

function RunInspectCLI(const AArgs: array of string; AOut, AErr: TStrings): Integer;
var
  I, J, N, CNT: Integer;
  Bad, Found: Boolean;
  AFile: string;
  OKv, OTensors: Boolean;
  VType: LongWord;
  F: TGgufFile;
  DataEnd: QWord;
  Types: array of LongWord;
begin
  { 1: parse }
  AFile := '';
  OKv := False;
  OTensors := False;
  Bad := False;
  for I := Low(AArgs) to High(AArgs) do
  begin
    if AArgs[I] = '--kv' then
      OKv := True
    else if AArgs[I] = '--tensors' then
      OTensors := True
    else if (Copy(AArgs[I], 1, 2) = '--') or (AFile <> '') then
    begin
      Bad := True;
      Break;
    end
    else
      AFile := AArgs[I];
  end;
  if (Bad) or (AFile = '') then
  begin
    AErr.Add(K_USAGE);
    Result := 2;
    Exit;
  end;
  { 2: open (a malformed GGUF raises EGgufError; nothing goes to AOut) }
  try
    F := TGgufFile.Open(AFile);
  except
    on E: EGgufError do
    begin
      AErr.Add('error: ' + E.Message);
      Result := 1;
      Exit;
    end;
  end;
  try
    { 3: data_end -- the GGUF data section ends after the last tensor, so
      the file is valid exactly when data_end equals file_size }
    if F.TensorCount = 0 then
      DataEnd := F.DataOffset
    else
      DataEnd := F.DataOffset + F.Tensor[F.TensorCount - 1].Offset
        + F.Tensor[F.TensorCount - 1].NBytes;
    { 4: summary lines }
    AOut.Add('version' + #9 + IntToStr(LongInt(F.Version)));
    AOut.Add('alignment' + #9 + IntToStr(Int64(F.Alignment)));
    AOut.Add('kv' + #9 + IntToStr(F.KVCount));
    AOut.Add('tensors' + #9 + IntToStr(F.TensorCount));
    AOut.Add('data_offset' + #9 + IntToStr(Int64(F.DataOffset)));
    AOut.Add('data_end' + #9 + IntToStr(Int64(DataEnd)));
    AOut.Add('file_size' + #9 + IntToStr(Int64(F.FileSize)));
    { 5: one type line per type that exists, ordered by id }
    SetLength(Types, F.TensorCount);
    N := 0;
    for I := 0 to F.TensorCount - 1 do
    begin
      Found := False;
      for J := 0 to N - 1 do
        if Types[J] = F.Tensor[I].TType then
          Found := True;
      if not Found then
      begin
        Types[N] := F.Tensor[I].TType;
        Inc(N);
      end;
    end;
    for I := 1 to N - 1 do
    begin
      VType := Types[I];
      J := I - 1;
      while (J >= 0) and (Types[J] > VType) do
      begin
        Types[J + 1] := Types[J];
        Dec(J);
      end;
      Types[J + 1] := VType;
    end;
    for I := 0 to N - 1 do
    begin
      CNT := 0;
      for J := 0 to F.TensorCount - 1 do
        if F.Tensor[J].TType = Types[I] then
          Inc(CNT);
      AOut.Add('type' + #9 + GgufTypeName(Types[I]) + #9 + IntToStr(CNT));
    end;
    { 6: --kv lines, file order }
    if OKv then
      for I := 0 to F.KVCount - 1 do
        AOut.Add('kv' + #9 + F.KV[I].Key + #9 + KVTypeText(F.KV[I]));
    { 7: --tensors lines, file order }
    if OTensors then
      for I := 0 to F.TensorCount - 1 do
        AOut.Add('tensor' + #9 + F.Tensor[I].Name + #9
          + GgufTypeName(F.Tensor[I].TType) + #9
          + DimsText(F.Tensor[I]) + #9 + IntToStr(Int64(F.Tensor[I].Offset))
          + #9 + IntToStr(Int64(F.Tensor[I].NBytes)));
    { 8: data_end vs file_size }
    if DataEnd <> F.FileSize then
    begin
      AErr.Add('error: data ends at ' + IntToStr(Int64(DataEnd))
        + ', file size ' + IntToStr(Int64(F.FileSize)));
      Result := 1;
    end
    else
      Result := 0;
    { 9: the TGgufFile is freed in finally }
  finally
    F.Free;
  end;
end;

end.
