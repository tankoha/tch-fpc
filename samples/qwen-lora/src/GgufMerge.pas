{ GgufMerge — the D3 writer (T18), per DESIGN.md 8.15 and
  tasks/T18-facts.md (authoritative): a base GGUF and a checkpoint's
  adapters become one GGUF whose adapted matrices hold W + s·B·A in Q8_0;
  the metadata is copied as bytes, every other tensor byte for byte.
  Interface by the harness owner (tasks/T18-facts.md); the bodies are
  written in T18b1 (MergeTargets, LoadAdapters, MergeTensor, WriteMerged)
  and T18b2 (RunMergeCLI). }
unit GgufMerge;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Tch, GgufFile, GgufQuant, GgufWriter, QwenBlocks, Lora,
  Checkpoint;

const
  GgufTypeQ8_0 = 8;

type
  EGgufMerge = class(Exception);

  { One adapted matrix: the adapter name (as TQwenModel.AttachLora gives
    it), the index of '<Name>.weight' in the base's directory, its
    dimensions (GGUF Dims[0] = InDim, Dims[1] = OutDim). }
  TMergeTarget = record
    Name: string;
    Tensor: Integer;
    InDim, OutDim: Int64;
  end;
  TMergeTargets = array of TMergeTarget;

{ The adapted matrices of AGguf: per trunk block 0..trunk-1 its kind's
  LoraTargets (linear: attn_qkv, attn_gate, ssm_out; full: attn_q,
  attn_k, attn_v, attn_output). Raises EGgufMerge when '<name>.weight'
  is missing or not two-dimensional. }
function MergeTargets(AGguf: TGgufFile): TMergeTargets;

{ A TLoraSet of AState's rank and scale with one CPU pair per target,
  loaded from ADir's lora.ot (the caller frees it). }
function LoadAdapters(const ATargets: TMergeTargets; const ADir: string;
  const AState: TRunState): TLoraSet;

{ The tensor of type ATType (AIn × AOut elements in ABytes) dequantised
  to fp32, plus AP.MergedDelta, quantised to Q8_0. Raises EGgufMerge when
  AP's dimensions differ from AIn, AOut. }
function MergeTensor(ATType: Cardinal; const ABytes: TBytes; AIn, AOut: Int64;
  AP: TLoraPair): TBytes;

{ Writes AGguf to AOutPath with the pairs of ASet merged (nil: no pair,
  a byte-identical copy). Written to AOutPath + '.part' and renamed; the
  part file is removed on failure. Returns the number of merged tensors. }
function WriteMerged(AGguf: TGgufFile; const ATargets: TMergeTargets; ASet: TLoraSet;
  const AOutPath: string): Integer;

{ qlora_merge [--base-sha256 H] BASE CKPT OUT; AOut / AErr caller-owned;
  the result is the exit code (0 done, 1 error, 2 usage). }
function RunMergeCLI(const AArgs: array of string; AOut, AErr: TStrings): Integer;

implementation

uses
  TrainCli;   { ShaFromName }

procedure AddTarget(AGguf: TGgufFile; const AName: string; var R: TMergeTargets);
var
  T, N: Integer;
  Info: TGgufTensorInfo;
begin
  T := AGguf.FindTensor(AName + '.weight');
  if T < 0 then
    raise EGgufMerge.CreateFmt('no tensor %s.weight', [AName]);
  Info := AGguf.Tensor[T];
  if Info.NDims <> 2 then
    raise EGgufMerge.CreateFmt('%s.weight has %d dimensions, expected 2', [AName, Info.NDims]);
  N := Length(R);
  SetLength(R, N + 1);
  R[N].Name := AName;
  R[N].Tensor := T;
  R[N].InDim := Info.Dims[0];
  R[N].OutDim := Info.Dims[1];
end;

function MergeTargets(AGguf: TGgufFile): TMergeTargets;
var
  D: TQwenDims;
  Trunk, B, K: Integer;
  P: string;
begin
  Result := nil;
  D := ReadQwenDims(AGguf);
  Trunk := D.Blocks;
  K := AGguf.FindKey('qwen35.nextn_predict_layers');
  if K >= 0 then
    Trunk := D.Blocks - AGguf.KV[K].IntVal;
  for B := 0 to Trunk - 1 do
  begin
    P := 'blk.' + IntToStr(B) + '.';
    if IsFullBlock(D, B) then
    begin
      AddTarget(AGguf, P + 'attn_q', Result);
      AddTarget(AGguf, P + 'attn_k', Result);
      AddTarget(AGguf, P + 'attn_v', Result);
      AddTarget(AGguf, P + 'attn_output', Result);
    end
    else
    begin
      AddTarget(AGguf, P + 'attn_qkv', Result);
      AddTarget(AGguf, P + 'attn_gate', Result);
      AddTarget(AGguf, P + 'ssm_out', Result);
    end;
  end;
end;

function LoadAdapters(const ATargets: TMergeTargets; const ADir: string;
  const AState: TRunState): TLoraSet;
var
  I: Integer;
begin
  Result := TLoraSet.Create(AState.Rank, AState.Scale);
  try
    for I := 0 to High(ATargets) do
      Result.Add(ATargets[I].Name, ATargets[I].InDim, ATargets[I].OutDim, TDevice.Cpu);
    Result.Load(IncludeTrailingPathDelimiter(ADir) + 'lora.ot');
  except
    Result.Free;
    raise;
  end;
end;

function MergeTensor(ATType: Cardinal; const ABytes: TBytes; AIn, AOut: Int64;
  AP: TLoraPair): TBytes;
var
  W: TSingleArray;
  NG: IInterface;
begin
  if (AP.InDim <> AIn) or (AP.OutDim <> AOut) then
    raise EGgufMerge.CreateFmt('%s: adapter %d -> %d, tensor %d -> %d',
      [AP.Name, AP.InDim, AP.OutDim, AIn, AOut]);
  NG := NoGradGuard;
  W := DequantizeBytes(ATType, ABytes, AIn * AOut);
  W := (TTensor.FromSingleArray(W).reshape([AOut, AIn]) + AP.MergedDelta).ToSingleArray;
  Result := QuantizeQ8_0(W);
end;

function WriteMerged(AGguf: TGgufFile; const ATargets: TMergeTargets; ASet: TLoraSet;
  const AOutPath: string): Integer;
var
  W: TGgufWriter;
  Pairs: array of TLoraPair;
  I: Integer;
  Part: string;
  B: TBytes;
  Info: TGgufTensorInfo;
begin
  Result := 0;
  Pairs := nil;
  SetLength(Pairs, AGguf.TensorCount);
  if ASet <> nil then
    for I := 0 to High(ATargets) do
    begin
      Pairs[ATargets[I].Tensor] := ASet.Find(ATargets[I].Name);
      if Pairs[ATargets[I].Tensor] = nil then
        raise EGgufMerge.CreateFmt('no adapter %s', [ATargets[I].Name]);
    end;
  Part := AOutPath + '.part';
  W := TGgufWriter.Create(AGguf);
  try
    for I := 0 to High(Pairs) do
      if Pairs[I] <> nil then
        W.SetTensorType(AGguf.Tensor[I].Name, GgufTypeQ8_0);
    W.BeginWrite(Part);
    for I := 0 to AGguf.TensorCount - 1 do
    begin
      AGguf.ReadTensorBytes(I, B);
      if Pairs[I] <> nil then
      begin
        Info := AGguf.Tensor[I];
        B := MergeTensor(Info.TType, B, Info.Dims[0], Info.Dims[1], Pairs[I]);
        Inc(Result);
      end;
      W.WriteTensor(B);
    end;
    W.Finish;
  finally
    W.Free;   { removes the part file when not finished }
  end;
  if not RenameFile(Part, AOutPath) then   { rename(2): replaces a file }
  begin
    DeleteFile(Part);
    raise EGgufMerge.CreateFmt('cannot rename %s to %s', [Part, AOutPath]);
  end;
end;

function RunMergeCLI(const AArgs: array of string; AOut, AErr: TStrings): Integer;
var
  Args: array of string;
  Sha, A: string;
  I, N: Integer;
  G: TGgufFile;
  St: TRunState;
  T: TMergeTargets;
  S: TLoraSet;
begin
  Args := nil;
  Sha := '';
  I := 0;
  while I <= High(AArgs) do
  begin
    A := AArgs[I];
    if A = '--base-sha256' then
    begin
      if I = High(AArgs) then
      begin
        AErr.Add('--base-sha256: value missing');
        Exit(2);
      end;
      Sha := AArgs[I + 1];
      Inc(I, 2);
      Continue;
    end;
    if (Length(A) > 1) and (A[1] = '-') then
    begin
      AErr.Add('unknown option ' + A);
      Exit(2);
    end;
    SetLength(Args, Length(Args) + 1);
    Args[High(Args)] := A;
    Inc(I);
  end;
  if Length(Args) <> 3 then
  begin
    AErr.Add(Format('expected BASE CKPT OUT, got %d arguments', [Length(Args)]));
    Exit(2);
  end;
  Result := 1;
  if not FileExists(Args[0]) then
  begin
    AErr.Add('no model file ' + Args[0]);
    Exit;
  end;
  if ExpandFileName(Args[2]) = ExpandFileName(Args[0]) then
  begin
    AErr.Add('the output must not be the base model');
    Exit;
  end;
  if Sha = '' then
    Sha := ShaFromName(Args[0]);
  G := nil;
  S := nil;
  try
    try
      St := ReadRunState(Args[1]);
      if (St.BaseSha256 <> '') and (Sha <> '') and (St.BaseSha256 <> Sha) then
      begin
        AErr.Add('the base model differs from the checkpoint''s: ' + Args[1]);
        Exit;
      end;
      G := TGgufFile.Open(Args[0]);
      T := MergeTargets(G);
      S := LoadAdapters(T, Args[1], St);
      N := WriteMerged(G, T, S, Args[2]);
      AOut.Add(Format('merged %d tensors (rank %d, scale %g) into %s',
        [N, St.Rank, St.Scale, Args[2]]));
      Result := 0;
    except
      on E: Exception do
        AErr.Add(E.Message);
    end;
  finally
    S.Free;
    G.Free;
  end;
end;

end.
