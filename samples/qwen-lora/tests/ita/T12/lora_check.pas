(* ITa driver of T12 (written by the harness owner; agents run it through
  tests/ita/T12-run.sh, they do not change it).
  Usage: lora_check MODEL PROMPTS.jsonl WORKDIR
  MODEL: the rehearsal GGUF; PROMPTS: work/oracle/T11/prompts.jsonl (token
  records); WORKDIR: room for the merged GGUF and the saved adapters.
  Rank 16, scale 2 (DESIGN 8.13), seed 12, on the CPU in fp32.
  Prints one line per case: case<TAB>value<TAB>limit<TAB>note; OK is
  decided by the runner (value <= limit).
  The oracles do not use the code under test for what they check:
  001 counts the targets from the tensor directory; 003 writes a GGUF whose
  target matrices are W + s*B*A computed here in Double from T02's scalar
  dequantisation and the pair's A and B, loads it without LoRA and
  compares; 004 compares the block-wise backward with one backward through
  the whole graph. *)
program lora_check;

{$mode objfpc}{$H+}

uses
  SysUtils, Classes, Math, fpjson, jsonparser, Tch, GgufFile, GgufQuant, GgufWriter,
  QwenBlocks, QwenModel, Recompute, Lora;

const
  Rank = 16;
  Scale = 2.0;
  Targets: array[0..6] of string = ('attn_qkv', 'attn_gate', 'ssm_out',
    'attn_q', 'attn_k', 'attn_v', 'attn_output');

type
  TIdArray = array of Integer;

var
  ModelPath, PromptPath, WorkDir: string;

procedure Emit(const ACase: string; AValue, ALimit: Double; const ANote: string);
begin
  WriteLn(ACase, #9, Format('%.1E', [AValue]), #9, Format('%.1E', [ALimit]), #9, ANote);
end;

function PromptIds(const AId: string): TIdArray;
var
  L: TStringList;
  I, J: Integer;
  O: TJSONObject;
  A: TJSONArray;
begin
  Result := nil;
  L := TStringList.Create;
  try
    L.LoadFromFile(PromptPath);
    for I := 0 to L.Count - 1 do
    begin
      if Trim(L[I]) = '' then
        Continue;
      O := GetJSON(L[I], False) as TJSONObject;
      try
        if (O.Get('id', '') = AId) and (O.Find('tokens') <> nil) then
        begin
          A := O.Arrays['tokens'];
          SetLength(Result, A.Count);
          for J := 0 to A.Count - 1 do
            Result[J] := A.Integers[J];
          Exit;
        end;
      finally
        O.Free;
      end;
    end;
  finally
    L.Free;
  end;
  raise Exception.Create('prompt not found: ' + AId);
end;

function MaxAbs(const T: TTensor): Double;
begin
  Result := T.abs.max.DoubleValue([]);
end;

function IsTarget(const AName: string; ATrunk: Integer): Boolean;
var
  P: TStringArray;
  I: Integer;
begin
  Result := False;
  P := AName.Split(['.']);
  if (Length(P) <> 4) or (P[0] <> 'blk') or (P[3] <> 'weight') then
    Exit;
  if StrToIntDef(P[1], -1) >= ATrunk then
    Exit;
  for I := 0 to High(Targets) do
    if P[2] = Targets[I] then
      Exit(True);
end;

{ 001: pairs and parameter count against the tensor directory }
procedure Case001(G: TGgufFile; M: TQwenModel; S: TLoraSet; ANumAdded: Integer);
var
  I, Want, Bad: Integer;
  Info: TGgufTensorInfo;
  Name: string;
  P: TLoraPair;
  Params: Int64;
begin
  Want := 0;
  Bad := 0;
  Params := 0;
  for I := 0 to G.TensorCount - 1 do
  begin
    Info := G.Tensor[I];
    if not IsTarget(Info.Name, M.TrunkBlocks) then
      Continue;
    Inc(Want);
    Name := Copy(Info.Name, 1, Length(Info.Name) - Length('.weight'));
    P := S.Find(Name);
    if (P = nil) or (P.InDim <> Int64(Info.Dims[0])) or (P.OutDim <> Int64(Info.Dims[1])) then
      Inc(Bad);
    Inc(Params, Rank * (Int64(Info.Dims[0]) + Int64(Info.Dims[1])));
  end;
  if (S.Count <> Want) or (ANumAdded <> Want) then
    Inc(Bad);
  if S.ParameterCount <> Params then
    Inc(Bad);
  Emit('ITa-T12-001', Bad, 0, Format('pairs %d (directory %d), parameters %d (directory %d), mismatches %d',
    [S.Count, Want, S.ParameterCount, Params, Bad]));
end;

function LastLp(M: TQwenModel; const AIds: TIdArray): TTensor;
begin
  Result := M.LastLogProbs(AIds);
end;

{ 002: B = 0 leaves the output unchanged }
procedure Case002(M, M0: TQwenModel);
const
  Ids: array[0..2] of string = ('pas-001', 'ja-065', 'chat-130');
var
  I: Integer;
  D: Double;
  P: TIdArray;
begin
  D := 0;
  for I := 0 to High(Ids) do
  begin
    P := PromptIds(Ids[I]);
    D := Max(D, MaxAbs(LastLp(M, P) - LastLp(M0, P)));
  end;
  Emit('ITa-T12-002', D, 0, 'B = 0: max |logprob with LoRA - without| over pas-001, ja-065, chat-130 (exact)');
end;

procedure RandomB(S: TLoraSet);
var
  G: IInterface;
  I: Integer;
begin
  G := NoGradGuard;
  for I := 0 to S.Count - 1 do
    S.Pair(I).B.CopyFrom(TTensor.randn([S.Pair(I).OutDim, Rank]) * 0.02);
end;

{ W + s*B*A in Double, as F32 bytes in the GGUF order (In fastest) }
function MergedBytes(G: TGgufFile; AIndex: Integer; P: TLoraPair): TBytes;
var
  Info: TGgufTensorInfo;
  Src: TBytes;
  W: TSingleArray;
  A, B: TDoubleArray;
  N, R, C, K: Int64;
  Acc: Double;
  F: TSingleArray;
begin
  Info := G.Tensor[AIndex];
  N := Int64(Info.Dims[0]) * Int64(Info.Dims[1]);
  G.ReadTensorBytes(AIndex, Src);
  W := DequantizeBytes(Info.TType, Src, N);
  A := P.A.detach.reshape([-1]).ToDoubleArray;
  B := P.B.detach.reshape([-1]).ToDoubleArray;
  F := nil;
  SetLength(F, N);
  for R := 0 to P.OutDim - 1 do
    for C := 0 to P.InDim - 1 do
    begin
      Acc := 0;
      for K := 0 to Rank - 1 do
        Acc := Acc + B[R * Rank + K] * A[K * P.InDim + C];
      F[R * P.InDim + C] := W[R * P.InDim + C] + Scale * Acc;
    end;
  Result := nil;
  SetLength(Result, N * 4);
  Move(F[0], Result[0], N * 4);
end;

{ 003: the adapted forward equals the base forward on merged weights }
procedure Case003(G: TGgufFile; M, M0: TQwenModel; S: TLoraSet);
const
  Ids: array[0..2] of string = ('pas-001', 'ja-065', 'chat-130');
var
  W: TGgufWriter;
  I, J: Integer;
  Name, Path: string;
  Bytes: TBytes;
  G2: TGgufFile;
  M2: TQwenModel;
  D, E: Double;
  P: TIdArray;
begin
  Path := WorkDir + '/merged.gguf';
  W := TGgufWriter.Create(G);
  try
    for I := 0 to G.TensorCount - 1 do
      if IsTarget(G.Tensor[I].Name, M.TrunkBlocks) then
        W.SetTensorType(G.Tensor[I].Name, 0);
    W.BeginWrite(Path);
    for I := 0 to G.TensorCount - 1 do
    begin
      Name := G.Tensor[I].Name;
      if IsTarget(Name, M.TrunkBlocks) then
        Bytes := MergedBytes(G, I, S.Find(Copy(Name, 1, Length(Name) - Length('.weight'))))
      else
        G.ReadTensorBytes(I, Bytes);
      W.WriteTensor(Bytes);
    end;
    W.Finish;
  finally
    W.Free;
  end;
  G2 := TGgufFile.Open(Path);
  M2 := TQwenModel.Load(G2, TDevice.Cpu, TDevice.Cpu, 0);
  try
    D := 0;
    E := 0;
    for J := 0 to High(Ids) do
    begin
      P := PromptIds(Ids[J]);
      D := Max(D, MaxAbs(LastLp(M, P) - LastLp(M2, P)));
      E := Max(E, MaxAbs(LastLp(M, P) - LastLp(M0, P)));
    end;
  finally
    M2.Free;
    G2.Free;
  end;
  DeleteFile(Path);
  Emit('ITa-T12-003', D, 1e-3, 'random B (0.02 N(0,1)): max |logprob with LoRA - base on W + s*B*A (Double)| over pas-001, ja-065, chat-130' +
    Format(' (the adapters move the logprobs by up to %.2f)', [E]));
end;

{ next-token log-likelihood of the prompt, summed }
function Loss(M: TQwenModel; const L: TTensor; const AIds: TIdArray): TTensor;
var
  T: array of Int64;
  I: Integer;
begin
  T := nil;
  SetLength(T, Length(AIds) - 1);
  for I := 1 to High(AIds) do
    T[I - 1] := AIds[I];
  Result := -L.narrow(0, 0, Length(AIds) - 1).log_softmax(-1)
    .gather(1, TTensor.FromArray(T).view([-1, 1])).sum(kNone);
end;

{ 004: gradients of the block-wise backward against one whole-graph backward }
procedure Case004(M: TQwenModel; S: TLoraSet);
var
  P: TIdArray;
  GA, GB: array of TTensor;
  I: Integer;
  Saved: TSavedInputs;
  H: TTensor;
  D: Double;
begin
  P := PromptIds('chat-065');
  Loss(M, M.Logits(M.Trunk(M.Embed(P))), P).Backward;
  GA := nil;
  GB := nil;
  SetLength(GA, S.Count);
  SetLength(GB, S.Count);
  for I := 0 to S.Count - 1 do
  begin
    GA[I] := S.Pair(I).A.grad.Copy;
    GB[I] := S.Pair(I).B.grad.Copy;
    S.Pair(I).A.ZeroGrad;
    S.Pair(I).B.ZeroGrad;
  end;
  Saved := TSavedInputs.Create(True);
  try
    H := RecomputeForward(M.Stages, M.Embed(P), Saved).detach.requires_grad_(True);
    Loss(M, M.Logits(H), P).Backward;
    RecomputeBackward(M.Stages, Saved, H.grad);
  finally
    Saved.Free;
  end;
  D := 0;
  for I := 0 to S.Count - 1 do
  begin
    D := Max(D, MaxAbs(S.Pair(I).A.grad - GA[I]) / Max(MaxAbs(GA[I]), 1e-30));
    D := Max(D, MaxAbs(S.Pair(I).B.grad - GB[I]) / Max(MaxAbs(GB[I]), 1e-30));
  end;
  Emit('ITa-T12-004', D, 1e-4, 'chat-065, random B: max over pairs of max|grad block-wise - whole graph| / max|whole graph| (A and B)');
end;

{ 005: Save, then Load into a fresh set on a fresh model }
procedure Case005(G: TGgufFile; S: TLoraSet);
var
  M3: TQwenModel;
  S3: TLoraSet;
  I: Integer;
  D: Double;
  Path: string;
begin
  Path := WorkDir + '/adapters.ot';
  S.Save(Path);
  M3 := TQwenModel.Load(G, TDevice.Cpu, TDevice.Cpu, 0);
  S3 := TLoraSet.Create(Rank, Scale);
  try
    M3.AttachLora(S3);
    S3.Load(Path);
    D := 0;
    for I := 0 to S.Count - 1 do
    begin
      D := Max(D, MaxAbs(S3.Pair(I).A - S.Pair(I).A));
      D := Max(D, MaxAbs(S3.Pair(I).B - S.Pair(I).B));
    end;
  finally
    M3.Free;
    S3.Free;
  end;
  DeleteFile(Path);
  Emit('ITa-T12-005', D, 0, 'Save then Load into a fresh set: max |difference| over every A and B (exact)');
end;

var
  G: TGgufFile;
  M, M0: TQwenModel;
  S: TLoraSet;
  N: Integer;
begin
  if ParamCount <> 3 then
  begin
    WriteLn(StdErr, 'usage: lora_check MODEL PROMPTS.jsonl WORKDIR');
    Halt(2);
  end;
  ModelPath := ParamStr(1);
  PromptPath := ParamStr(2);
  WorkDir := ParamStr(3);
  ManualSeed(12);
  G := TGgufFile.Open(ModelPath);
  M := TQwenModel.Load(G, TDevice.Cpu, TDevice.Cpu, 0);
  M0 := TQwenModel.Load(G, TDevice.Cpu, TDevice.Cpu, 0);
  S := TLoraSet.Create(Rank, Scale);
  try
    N := M.AttachLora(S);
    Case001(G, M, S, N);
    Case002(M, M0);
    RandomB(S);
    Case003(G, M, M0, S);
    M0.Free;
    M0 := nil;
    Case004(M, S);
    Case005(G, S);
  finally
    M.Free;
    M0.Free;
    S.Free;
    G.Free;
  end;
end.
