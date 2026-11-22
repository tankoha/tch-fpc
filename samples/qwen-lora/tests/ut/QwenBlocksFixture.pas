{ T10 UT fixture (harness owner's): writes a tiny qwen35 GGUF with random
  weights (two trunk blocks: blk.0 linear, blk.1 full; full_attention_interval
  2), small enough for the scalar reference of QwenRefScalar.pas. Its key
  and value head counts differ (2 key heads, 4 value heads; 4 query heads,
  2 KV heads), so the tiling and the GQA mapping are exercised. Matrices
  are Q8_0 or F16, vectors F32. Not code under test. }
unit QwenBlocksFixture;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Math, GgufQuant;

const
  TinyHidden = 64;
  TinyFfn = 96;
  TinyHeadsQ = 4;
  TinyHeadsKV = 2;
  TinyHeadDim = 32;
  TinyRot = 16;
  TinySsmK = 2;
  TinySsmV = 4;
  TinySsmDim = 16;
  TinyConv = 4;

{ Writes the tiny model to APath (weights from RandSeed := ASeed). }
procedure WriteTinyQwen(const APath: string; ASeed: Integer);

implementation

type
  TTens = record
    Name: RawByteString;
    Dims: array of QWord;      { GGUF order: the first is the row length }
    TType: Cardinal;
    Data: TBytes;
  end;

var
  Tens: array of TTens;

procedure WU32(S: TStream; V: Cardinal); begin S.WriteBuffer(V, 4); end;
procedure WU64(S: TStream; V: QWord); begin S.WriteBuffer(V, 8); end;
procedure WF32(S: TStream; V: Single); begin S.WriteBuffer(V, 4); end;
procedure WStr(S: TStream; const V: RawByteString);
begin
  WU64(S, Length(V));
  if V <> '' then S.WriteBuffer(V[1], Length(V));
end;

procedure KvU32(S: TStream; const K: RawByteString; V: Cardinal);
begin WStr(S, K); WU32(S, 4); WU32(S, V); end;
procedure KvF32(S: TStream; const K: RawByteString; V: Single);
begin WStr(S, K); WU32(S, 6); WF32(S, V); end;
procedure KvStr(S: TStream; const K, V: RawByteString);
begin WStr(S, K); WU32(S, 8); WStr(S, V); end;

function Rand(N: Integer; Scale, Offset: Double): TSingleArray;
var
  I: Integer;
begin
  Result := nil;
  SetLength(Result, N);
  for I := 0 to N - 1 do
    Result[I] := Offset + Scale * RandG(0, 1);
end;

procedure Add(const AName: RawByteString; const ADims: array of QWord; ATType: Cardinal;
  const AValues: TSingleArray);
var
  N, I: Integer;
  B: TBytes;
  W: Word;
begin
  N := Length(Tens);
  SetLength(Tens, N + 1);
  Tens[N].Name := AName;
  SetLength(Tens[N].Dims, Length(ADims));
  for I := 0 to High(ADims) do
    Tens[N].Dims[I] := ADims[I];
  Tens[N].TType := ATType;
  case ATType of
    0: begin
         SetLength(B, Length(AValues) * 4);
         Move(AValues[0], B[0], Length(B));
       end;
    1: begin
         SetLength(B, Length(AValues) * 2);
         for I := 0 to High(AValues) do
         begin
           W := SingleToFp16(AValues[I]);
           Move(W, B[I * 2], 2);
         end;
       end;
    8: B := QuantizeQ8_0(AValues);
  end;
  Tens[N].Data := B;
end;

{ a matrix [Out, In] (GGUF dims In, Out) }
procedure Mat(const AName: RawByteString; AOut, AIn: Integer; ATType: Cardinal);
begin
  Add(AName, [AIn, AOut], ATType, Rand(AOut * AIn, 1 / Sqrt(AIn), 0));
end;

procedure Vec(const AName: RawByteString; const ADims: array of QWord; const AValues: TSingleArray);
begin
  Add(AName, ADims, 0, AValues);
end;

procedure WriteTinyQwen(const APath: string; ASeed: Integer);
const
  Align = 32;
var
  S: TFileStream;
  I, J: Integer;
  Off, Pos0: QWord;
  Pad: array[0..31] of Byte;
  A: TSingleArray;
  Qkv: Integer;
begin
  RandSeed := ASeed;
  Tens := nil;
  Qkv := 2 * TinySsmK * TinySsmDim + TinySsmV * TinySsmDim;
  { blk.0: linear }
  Vec('blk.0.attn_norm.weight', [TinyHidden], Rand(TinyHidden, 0.1, 1));
  Mat('blk.0.attn_qkv.weight', Qkv, TinyHidden, 8);
  Mat('blk.0.attn_gate.weight', TinySsmV * TinySsmDim, TinyHidden, 1);
  Mat('blk.0.ssm_beta.weight', TinySsmV, TinyHidden, 8);
  Mat('blk.0.ssm_alpha.weight', TinySsmV, TinyHidden, 8);
  Vec('blk.0.ssm_dt.bias', [TinySsmV], Rand(TinySsmV, 0.5, 0));
  A := Rand(TinySsmV, 0.5, 0);
  for J := 0 to High(A) do
    A[J] := -Exp(A[J]);
  Vec('blk.0.ssm_a', [TinySsmV], A);
  Vec('blk.0.ssm_conv1d.weight', [TinyConv, Qkv], Rand(Qkv * TinyConv, 0.5, 0));
  Vec('blk.0.ssm_norm.weight', [TinySsmDim], Rand(TinySsmDim, 0.1, 1));
  Mat('blk.0.ssm_out.weight', TinyHidden, TinySsmV * TinySsmDim, 8);
  Vec('blk.0.post_attention_norm.weight', [TinyHidden], Rand(TinyHidden, 0.1, 1));
  Mat('blk.0.ffn_gate.weight', TinyFfn, TinyHidden, 8);
  Mat('blk.0.ffn_up.weight', TinyFfn, TinyHidden, 1);
  Mat('blk.0.ffn_down.weight', TinyHidden, TinyFfn, 8);
  { blk.1: full }
  Vec('blk.1.attn_norm.weight', [TinyHidden], Rand(TinyHidden, 0.1, 1));
  Mat('blk.1.attn_q.weight', TinyHeadsQ * 2 * TinyHeadDim, TinyHidden, 8);
  Mat('blk.1.attn_k.weight', TinyHeadsKV * TinyHeadDim, TinyHidden, 8);
  Mat('blk.1.attn_v.weight', TinyHeadsKV * TinyHeadDim, TinyHidden, 1);
  Vec('blk.1.attn_q_norm.weight', [TinyHeadDim], Rand(TinyHeadDim, 0.1, 1));
  Vec('blk.1.attn_k_norm.weight', [TinyHeadDim], Rand(TinyHeadDim, 0.1, 1));
  Mat('blk.1.attn_output.weight', TinyHidden, TinyHeadsQ * TinyHeadDim, 8);
  Vec('blk.1.post_attention_norm.weight', [TinyHidden], Rand(TinyHidden, 0.1, 1));
  Mat('blk.1.ffn_gate.weight', TinyFfn, TinyHidden, 8);
  Mat('blk.1.ffn_up.weight', TinyFfn, TinyHidden, 8);
  Mat('blk.1.ffn_down.weight', TinyHidden, TinyFfn, 1);

  FillChar(Pad, SizeOf(Pad), 0);
  S := TFileStream.Create(APath, fmCreate);
  try
    S.WriteBuffer(PAnsiChar('GGUF')^, 4);
    WU32(S, 3);
    WU64(S, Length(Tens));
    WU64(S, 16);
    KvStr(S, 'general.architecture', 'qwen35');
    KvU32(S, 'qwen35.block_count', 2);
    KvU32(S, 'qwen35.embedding_length', TinyHidden);
    KvU32(S, 'qwen35.feed_forward_length', TinyFfn);
    KvU32(S, 'qwen35.attention.head_count', TinyHeadsQ);
    KvU32(S, 'qwen35.attention.head_count_kv', TinyHeadsKV);
    KvU32(S, 'qwen35.attention.key_length', TinyHeadDim);
    KvU32(S, 'qwen35.rope.dimension_count', TinyRot);
    KvF32(S, 'qwen35.rope.freq_base', 10000);
    KvF32(S, 'qwen35.attention.layer_norm_rms_epsilon', 1e-6);
    KvU32(S, 'qwen35.ssm.group_count', TinySsmK);
    KvU32(S, 'qwen35.ssm.time_step_rank', TinySsmV);
    KvU32(S, 'qwen35.ssm.state_size', TinySsmDim);
    KvU32(S, 'qwen35.ssm.conv_kernel', TinyConv);
    KvU32(S, 'qwen35.full_attention_interval', 2);
    KvU32(S, 'qwen35.attention.value_length', TinyHeadDim);
    Off := 0;
    for I := 0 to High(Tens) do
    begin
      WStr(S, Tens[I].Name);
      WU32(S, Length(Tens[I].Dims));
      for J := 0 to High(Tens[I].Dims) do
        WU64(S, Tens[I].Dims[J]);
      WU32(S, Tens[I].TType);
      WU64(S, Off);
      Off := (Off + QWord(Length(Tens[I].Data)) + Align - 1) div Align * Align;
    end;
    Pos0 := S.Position;
    if Pos0 mod Align <> 0 then
      S.WriteBuffer(Pad, Align - Pos0 mod Align);
    for I := 0 to High(Tens) do
    begin
      S.WriteBuffer(Tens[I].Data[0], Length(Tens[I].Data));
      if Length(Tens[I].Data) mod Align <> 0 then
        S.WriteBuffer(Pad, Align - Length(Tens[I].Data) mod Align);
    end;
  finally
    S.Free;
  end;
end;

end.
