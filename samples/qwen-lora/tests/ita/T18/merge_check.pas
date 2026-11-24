(* ITa driver of T18 (written by the harness owner; agents run it through
  tests/ita/T18-run.sh, they do not change it).
  Usage: merge_check MODEL MERGED CKPT TRAIN.jsonl PROMPTS.jsonl ORACLE.jsonl WORKDIR
  MERGED: qlora_merge MODEL CKPT MERGED; PROMPTS: the H43 prompts of
  work/oracle/T11; ORACLE: the bundled llama-server on an all-F32 copy of
  the prototype's merged file (work/oracle/T18, DESIGN 8.15). WORKDIR gets
  identity.gguf and zero.gguf. Prints one line per case:
  case<TAB>value<TAB>limit<TAB>note; OK is decided by the runner (value <=
  limit). Case 006 (the merged file's bytes) is the runner's.
  The references do not use GgufMerge's MergeTensor: the round trip
  dequantises the base and adds the pair's MergedDelta here. *)
program merge_check;

{$mode objfpc}{$H+}

uses
  SysUtils, Classes, Math, fpjson, jsonparser, Tch, GgufFile, GgufQuant, QwenTokenizer,
  QwenModel, Lora, Checkpoint, TrainData, GgufMerge;

type
  TPrompt = record
    Id: string;
    Tokens: TTokenArray;
  end;
  TTop = record
    Id: string;
    TopIds: TInt64Array;
    TopLp: TDoubleArray;
  end;

var
  ModelPath, MergedPath, CkptDir, TrainPath, PromptPath, OraclePath, Work: string;
  GB, GM: TGgufFile;
  Tok: TQwenTokenizer;
  Prompts: array of TPrompt;
  Oracle: array of TTop;

procedure Emit(const ACase: string; AValue, ALimit: Double; const ANote: string);
begin
  WriteLn(ACase, #9, Format('%.3E', [AValue]), #9, Format('%.3E', [ALimit]), #9, ANote);
end;

function Lines(const APath: string): TStringList;
begin
  Result := TStringList.Create;
  Result.LoadFromFile(APath);
end;

procedure LoadPrompts;
var
  L: TStringList;
  I, J: Integer;
  O: TJSONObject;
  A: TJSONArray;
begin
  L := Lines(PromptPath);
  try
    for I := 0 to L.Count - 1 do
    begin
      if Trim(L[I]) = '' then
        Continue;
      O := GetJSON(L[I], False) as TJSONObject;
      try
        SetLength(Prompts, Length(Prompts) + 1);
        with Prompts[High(Prompts)] do
        begin
          Id := O.Strings['id'];
          if O.Find('tokens') <> nil then
          begin
            A := O.Arrays['tokens'];
            SetLength(Tokens, A.Count);
            for J := 0 to A.Count - 1 do
              Tokens[J] := A.Integers[J];
          end
          else
            Tokens := Tok.Encode(O.Strings['text']);
        end;
      finally
        O.Free;
      end;
    end;
  finally
    L.Free;
  end;
end;

procedure LoadOracle;
var
  L: TStringList;
  I, J: Integer;
  O: TJSONObject;
  A: TJSONArray;
begin
  L := Lines(OraclePath);
  try
    for I := 0 to L.Count - 1 do
    begin
      if Trim(L[I]) = '' then
        Continue;
      O := GetJSON(L[I], False) as TJSONObject;
      try
        SetLength(Oracle, Length(Oracle) + 1);
        with Oracle[High(Oracle)] do
        begin
          Id := O.Strings['id'];
          A := O.Arrays['top'];
          SetLength(TopIds, A.Count);
          SetLength(TopLp, A.Count);
          for J := 0 to A.Count - 1 do
          begin
            TopIds[J] := A.Objects[J].Int64s['id'];
            TopLp[J] := A.Objects[J].Floats['logprob'];
          end;
        end;
      finally
        O.Free;
      end;
    end;
  finally
    L.Free;
  end;
end;

function LastLogProbs(M: TQwenModel; const T: TTokenArray): TTensor;
var
  NG: IInterface;
begin
  NG := NoGradGuard;
  Result := M.Logits(M.Trunk(M.Embed(T))).select(0, Length(T) - 1).log_softmax(-1);
end;

function WholeLoss(M: TQwenModel; const S: TTrainSample): Double;
var
  NG: IInterface;
  L: TTensor;
  I: Integer;
begin
  NG := NoGradGuard;
  L := M.Logits(M.Trunk(M.Embed(S.Tokens))).log_softmax(-1);
  Result := 0;
  for I := S.LossFrom to S.LossTo - 1 do
    Result := Result - L.select(0, I - 1).select(0, S.Tokens[I]).DoubleValue([]);
end;

function MeanLoss(M: TQwenModel; ASet: TTrainSet): Double;
var
  I: Integer;
  Sum: Double;
  N: Int64;
begin
  Sum := 0;
  N := 0;
  for I := 0 to ASet.Count - 1 do
  begin
    Sum := Sum + WholeLoss(M, ASet[I]);
    Inc(N, ASet[I].LossTo - ASet[I].LossFrom);
  end;
  Result := Sum / N;
end;

{ The worst (over the prompts) mean |Δ logprob| over AA's top 10, AB
  against AA; ATop1 counts the prompts whose top-1 differs. }
function WorstMean(AA, AB: TQwenModel; out ATop1: Integer): Double;
var
  I, J: Integer;
  LA, LB: TTensor;
  Idx: TInt64Array;
  Sum: Double;
begin
  Result := 0;
  ATop1 := 0;
  for I := 0 to High(Prompts) do
  begin
    LA := LastLogProbs(AA, Prompts[I].Tokens);
    LB := LastLogProbs(AB, Prompts[I].Tokens);
    Idx := LA.argsort(0, True).narrow(0, 0, 10).ToInt64Array;
    Sum := 0;
    for J := 0 to 9 do
      Sum := Sum + Abs(LA.select(0, Idx[J]).DoubleValue([]) - LB.select(0, Idx[J]).DoubleValue([]));
    Result := Max(Result, Sum / 10);
    if LB.argmax(0, False).Int64Value([]) <> Idx[0] then
      Inc(ATop1);
  end;
end;

function SameFile(const A, B: string): Boolean;
var
  FA, FB: TFileStream;
  BA, BB: array[0..65535] of Byte;
  N: Integer;
begin
  FA := TFileStream.Create(A, fmOpenRead or fmShareDenyNone);
  FB := TFileStream.Create(B, fmOpenRead or fmShareDenyNone);
  try
    Result := FA.Size = FB.Size;
    while Result do
    begin
      N := FA.Read(BA, SizeOf(BA));
      if N = 0 then
        Break;
      Result := (FB.Read(BB, N) = N) and CompareMem(@BA, @BB, N);
    end;
  finally
    FA.Free;
    FB.Free;
  end;
end;

procedure CaseIdentity(const T: TMergeTargets);
var
  V: Double;
begin
  WriteMerged(GB, T, nil, Work + 'identity.gguf');
  if SameFile(ModelPath, Work + 'identity.gguf') then V := 0 else V := 1;
  Emit('ITa-T18-001', V, 0, 'WriteMerged without adapters against the base (0 when byte-identical)');
  DeleteFile(Work + 'identity.gguf');
end;

procedure CaseRoundTrip(const T: TMergeTargets; S: TLoraSet);
var
  I, J, K, Ti: Integer;
  B, BM: TBytes;
  W, Q: TSingleArray;
  NG: IInterface;
  D, Worst: Double;
  Bad: Integer;
begin
  NG := NoGradGuard;
  Worst := 0;
  Bad := 0;
  for I := 0 to High(T) do
  begin
    GB.ReadTensorBytes(T[I].Tensor, B);
    W := DequantizeBytes(GB.Tensor[T[I].Tensor].TType, B, T[I].InDim * T[I].OutDim);
    W := (TTensor.FromSingleArray(W).reshape([T[I].OutDim, T[I].InDim]) +
      S.Find(T[I].Name).MergedDelta).ToSingleArray;
    Ti := GM.FindTensor(T[I].Name + '.weight');
    if (Ti < 0) or (GM.Tensor[Ti].TType <> GgufTypeQ8_0) then
    begin
      Inc(Bad);
      Continue;
    end;
    GM.ReadTensorBytes(Ti, BM);
    Q := DequantizeBytes(GgufTypeQ8_0, BM, Length(W));
    for J := 0 to Length(W) div 32 - 1 do
    begin
      D := Fp16ToSingle(PWord(@BM[J * 34])^);
      for K := J * 32 to J * 32 + 31 do
        if D > 0 then
          Worst := Max(Worst, Abs(W[K] - Q[K]) / D)
        else if W[K] <> 0 then
          Worst := Infinity;
    end;
  end;
  Emit('ITa-T18-002', Worst + Bad * 99, 1,
    Format('%d tensors: max |merged - fp32 merge| / the block''s Q8_0 step (+ 99 per tensor missing or not Q8_0)', [Length(T)]));
end;

procedure CaseH47(MM: TQwenModel);
var
  I, J, K, Bad, Missing: Integer;
  L: TTensor;
  Worst: Double;
begin
  Worst := 0;
  Bad := 0;
  Missing := 0;
  for I := 0 to High(Prompts) do
  begin
    K := -1;
    for J := 0 to High(Oracle) do
      if Oracle[J].Id = Prompts[I].Id then
        K := J;
    if K < 0 then
    begin
      Inc(Missing);
      Continue;
    end;
    L := LastLogProbs(MM, Prompts[I].Tokens);
    for J := 0 to 9 do
      Worst := Max(Worst, Abs(L.select(0, Oracle[K].TopIds[J]).DoubleValue([]) - Oracle[K].TopLp[J]));
    if L.argmax(0, False).Int64Value([]) <> Oracle[K].TopIds[0] then
      Inc(Bad);
  end;
  Emit('ITa-T18-005', Worst + (Bad + Missing) * 99, 0.02,
    Format('%d prompts: max |Pascal on MERGED - llama-server on its F32 copy| over the reference''s top 10 (+ 99 per top-1 differing or prompt missing)',
      [Length(Prompts)]));
end;

var
  St: TRunState;
  T: TMergeTargets;
  S, Z, SA: TLoraSet;
  M0, MM, MZ: TQwenModel;
  GZ: TGgufFile;
  Data: TTrainSet;
  Floor, Fid, L0, LA, LM: Double;
  Top1F, Top1Z: Integer;
  NG: IInterface;
  I: Integer;
begin
  if ParamCount <> 7 then
  begin
    WriteLn(StdErr, 'usage: merge_check MODEL MERGED CKPT TRAIN.jsonl PROMPTS.jsonl ORACLE.jsonl WORKDIR');
    Halt(2);
  end;
  ModelPath := ParamStr(1);
  MergedPath := ParamStr(2);
  CkptDir := IncludeTrailingPathDelimiter(ParamStr(3));
  TrainPath := ParamStr(4);
  PromptPath := ParamStr(5);
  OraclePath := ParamStr(6);
  Work := IncludeTrailingPathDelimiter(ParamStr(7));
  GB := TGgufFile.Open(ModelPath);
  GM := TGgufFile.Open(MergedPath);
  Tok := TQwenTokenizer.Create(GB);
  LoadPrompts;
  LoadOracle;
  St := ReadRunState(CkptDir);
  T := MergeTargets(GB);
  S := LoadAdapters(T, CkptDir, St);
  CaseIdentity(T);
  CaseRoundTrip(T, S);
  { the retyping floor: the same targets merged with B = 0 }
  Z := LoadAdapters(T, CkptDir, St);
  NG := NoGradGuard;
  for I := 0 to Z.Count - 1 do
    Z.Pair(I).B.zero_;
  NG := nil;
  WriteMerged(GB, T, Z, Work + 'zero.gguf');
  GZ := TGgufFile.Open(Work + 'zero.gguf');
  M0 := TQwenModel.Load(GB, TDevice.Cpu, TDevice.Cpu, 0);
  MM := TQwenModel.Load(GM, TDevice.Cpu, TDevice.Cpu, 0);
  MZ := TQwenModel.Load(GZ, TDevice.Cpu, TDevice.Cpu, 0);
  Floor := WorstMean(M0, MZ, Top1Z);
  Data := TTrainSet.Create(Tok, TrainPath, ['compiled-run', 'fpc-approx'], 2048);
  L0 := MeanLoss(M0, Data);
  LM := MeanLoss(MM, Data);
  SA := TLoraSet.Create(St.Rank, St.Scale);
  M0.AttachLora(SA);
  SA.Load(CkptDir + 'lora.ot');
  LA := MeanLoss(M0, Data);
  Fid := WorstMean(M0, MM, Top1F);
  Emit('ITa-T18-003', Fid / Floor, 1.5,
    Format('worst mean |Δ logprob| over the top 10 of %d prompts: merged against the adapter %.4f, B = 0 merge against the base %.4f (merged / floor)',
      [Length(Prompts), Fid, Floor]));
  Emit('ITa-T18-004', 1 - (L0 - LM) / (L0 - LA), 0.1,
    Format('mean loss over TRAIN: base %.4f, adapter %.4f, merged %.4f (1 - gain kept)', [L0, LA, LM]));
  CaseH47(MM);
  DeleteFile(Work + 'zero.gguf');
end.
