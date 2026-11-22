{ T10 oracle (harness owner's): a deliberately slow scalar version of the
  qwen35 trunk blocks in Double, written from DESIGN 8.10 "Block
  skeleton", "Linear block" and "Full block" and independent of the tensor
  code: weights through the scalar GgufQuant (T02), plain loops for every
  step (convolution, L2 norm, key-head tiling, the delta rule token by
  token, RoPE, GQA, causal softmax). T11 adds RefLogits (embedding, trunk,
  output norm, head). Not code under test. }
unit QwenRefScalar;

{$mode objfpc}{$H+}

interface

uses
  SysUtils, Math, GgufFile, GgufQuant;

type
  TDoubleArray = array of Double;

  TRefDims = record
    Hidden, Ffn, HeadsQ, HeadsKV, HeadDim, Rot: Integer;
    RopeBase, Eps: Double;
    SsmHeadsK, SsmHeadsV, SsmDim, ConvKernel, FullInterval: Integer;
  end;

function RefDims(G: TGgufFile): TRefDims;
{ H: T rows of Hidden values (row-major). Returns block ABlock's output. }
function RefBlock(G: TGgufFile; ABlock: Integer; const D: TRefDims;
  const H: TDoubleArray; T: Integer): TDoubleArray;
{ RoPE on X: T rows of NHeads heads of HeadDim values, positions Pos0.. }
function RefRope(const X: TDoubleArray; T, NHeads, HeadDim, Rot: Integer;
  Base: Double; Pos0: Integer): TDoubleArray;
{ x / sqrt(mean(x^2) + Eps) * w over each run of Length(W) values }
function RefRms(const X: TDoubleArray; const W: TDoubleArray; Eps: Double): TDoubleArray;

{ T11: the whole model for AIds (T tokens): token_embd rows, the trunk
  blocks (block_count less qwen35.nextn_predict_layers when present),
  output_norm, then output.weight or, without it, token_embd (tied).
  Returns T rows of vocab logits. }
function RefLogits(G: TGgufFile; const D: TRefDims; const AIds: array of Integer): TDoubleArray;

implementation

function KeyIdx(G: TGgufFile; const K: RawByteString): Integer;
begin
  Result := G.FindKey(K);
  if Result < 0 then
    raise Exception.Create('missing key ' + K);
end;

function RefDims(G: TGgufFile): TRefDims;
begin
  Result.Hidden := G.KV[KeyIdx(G, 'qwen35.embedding_length')].IntVal;
  Result.Ffn := G.KV[KeyIdx(G, 'qwen35.feed_forward_length')].IntVal;
  Result.HeadsQ := G.KV[KeyIdx(G, 'qwen35.attention.head_count')].IntVal;
  Result.HeadsKV := G.KV[KeyIdx(G, 'qwen35.attention.head_count_kv')].IntVal;
  Result.HeadDim := G.KV[KeyIdx(G, 'qwen35.attention.key_length')].IntVal;
  Result.Rot := G.KV[KeyIdx(G, 'qwen35.rope.dimension_count')].IntVal;
  Result.RopeBase := G.KV[KeyIdx(G, 'qwen35.rope.freq_base')].FloatVal;
  Result.Eps := G.KV[KeyIdx(G, 'qwen35.attention.layer_norm_rms_epsilon')].FloatVal;
  Result.SsmHeadsK := G.KV[KeyIdx(G, 'qwen35.ssm.group_count')].IntVal;
  Result.SsmHeadsV := G.KV[KeyIdx(G, 'qwen35.ssm.time_step_rank')].IntVal;
  Result.SsmDim := G.KV[KeyIdx(G, 'qwen35.ssm.state_size')].IntVal;
  Result.ConvKernel := G.KV[KeyIdx(G, 'qwen35.ssm.conv_kernel')].IntVal;
  Result.FullInterval := G.KV[KeyIdx(G, 'qwen35.full_attention_interval')].IntVal;
end;

{ every value of a tensor, dequantised, in file (row-major libtorch) order }
function Values(G: TGgufFile; const AName: RawByteString; out Count: Int64): TDoubleArray;
var
  I, J: Integer;
  Info: TGgufTensorInfo;
  Bytes: TBytes;
  S: TSingleArray;
begin
  I := G.FindTensor(AName);
  if I < 0 then
    raise Exception.Create('missing tensor ' + AName);
  Info := G.Tensor[I];
  Count := 1;
  for J := 0 to Info.NDims - 1 do
    Count := Count * Int64(Info.Dims[J]);
  G.ReadTensorBytes(I, Bytes);
  S := DequantizeBytes(Info.TType, Bytes, Count);
  Result := nil;
  SetLength(Result, Count);
  for J := 0 to Count - 1 do
    Result[J] := S[J];
end;

{ y[t, o] = sum_i W[o, i] x[t, i]; W is [Out, In] }
function MatVec(G: TGgufFile; const AName: RawByteString; const X: TDoubleArray;
  T, InDim: Integer): TDoubleArray;
var
  W: TDoubleArray;
  N: Int64;
  OutDim, R, O, I: Integer;
  Acc: Double;
begin
  W := Values(G, AName, N);
  OutDim := N div InDim;
  Result := nil;
  SetLength(Result, T * OutDim);
  for R := 0 to T - 1 do
    for O := 0 to OutDim - 1 do
    begin
      Acc := 0;
      for I := 0 to InDim - 1 do
        Acc := Acc + W[O * InDim + I] * X[R * InDim + I];
      Result[R * OutDim + O] := Acc;
    end;
end;

function RefRms(const X: TDoubleArray; const W: TDoubleArray; Eps: Double): TDoubleArray;
var
  N, R, I: Integer;
  S: Double;
begin
  N := Length(W);
  Result := nil;
  SetLength(Result, Length(X));
  for R := 0 to Length(X) div N - 1 do
  begin
    S := 0;
    for I := 0 to N - 1 do
      S := S + Sqr(X[R * N + I]);
    S := 1 / Sqrt(S / N + Eps);
    for I := 0 to N - 1 do
      Result[R * N + I] := X[R * N + I] * S * W[I];
  end;
end;

function Silu(X: Double): Double;
begin
  Result := X / (1 + Exp(-X));
end;

function Sigmoid(X: Double): Double;
begin
  Result := 1 / (1 + Exp(-X));
end;

function Softplus(X: Double): Double;
begin
  if X > 20 then
    Result := X
  else
    Result := Ln(1 + Exp(X));
end;

function RefRope(const X: TDoubleArray; T, NHeads, HeadDim, Rot: Integer;
  Base: Double; Pos0: Integer): TDoubleArray;
var
  R, H, I, Half, P: Integer;
  A, X0, X1: Double;
begin
  Result := Copy(X);
  Half := Rot div 2;
  for R := 0 to T - 1 do
    for H := 0 to NHeads - 1 do
      for I := 0 to Half - 1 do
      begin
        P := (R * NHeads + H) * HeadDim;
        A := (Pos0 + R) * Power(Base, -2.0 * I / Rot);
        X0 := X[P + I];
        X1 := X[P + I + Half];
        Result[P + I] := X0 * Cos(A) - X1 * Sin(A);
        Result[P + I + Half] := X0 * Sin(A) + X1 * Cos(A);
      end;
end;

function LinearAttn(G: TGgufFile; const Pfx: RawByteString; const D: TRefDims;
  const X: TDoubleArray; T: Integer): TDoubleArray;
var
  Qkv, Z, Bz, Az, DtB, SsmA, Conv, SsmNorm, U, Q, K, V, M, O, Ot: TDoubleArray;
  N: Int64;
  C, Hk, Hv, Sd, R, Ch, I, J, Hd, Kh, A, B: Integer;
  S, Beta, Gt, Dlt, Pred: Double;
begin
  Hk := D.SsmHeadsK;
  Hv := D.SsmHeadsV;
  Sd := D.SsmDim;
  Qkv := MatVec(G, Pfx + 'attn_qkv.weight', X, T, D.Hidden);
  C := Length(Qkv) div T;
  Z := MatVec(G, Pfx + 'attn_gate.weight', X, T, D.Hidden);
  Bz := MatVec(G, Pfx + 'ssm_beta.weight', X, T, D.Hidden);
  Az := MatVec(G, Pfx + 'ssm_alpha.weight', X, T, D.Hidden);
  DtB := Values(G, Pfx + 'ssm_dt.bias', N);
  SsmA := Values(G, Pfx + 'ssm_a', N);
  Conv := Values(G, Pfx + 'ssm_conv1d.weight', N);   { [C, ConvKernel] }
  SsmNorm := Values(G, Pfx + 'ssm_norm.weight', N);
  { causal depthwise convolution, ConvKernel - 1 zeros in front, then SiLU }
  U := nil;
  SetLength(U, T * C);
  for R := 0 to T - 1 do
    for Ch := 0 to C - 1 do
    begin
      S := 0;
      for I := 0 to D.ConvKernel - 1 do
      begin
        J := R - (D.ConvKernel - 1) + I;
        if J >= 0 then
          S := S + Conv[Ch * D.ConvKernel + I] * Qkv[J * C + Ch];
      end;
      U[R * C + Ch] := Silu(S);
    end;
  { split; L2 norm of q and k per head }
  Q := nil; K := nil; V := nil;
  SetLength(Q, T * Hk * Sd);
  SetLength(K, T * Hk * Sd);
  SetLength(V, T * Hv * Sd);
  for R := 0 to T - 1 do
  begin
    for I := 0 to Hk * Sd - 1 do
    begin
      Q[R * Hk * Sd + I] := U[R * C + I];
      K[R * Hk * Sd + I] := U[R * C + Hk * Sd + I];
    end;
    for I := 0 to Hv * Sd - 1 do
      V[R * Hv * Sd + I] := U[R * C + 2 * Hk * Sd + I];
    for Hd := 0 to Hk - 1 do
    begin
      S := 0;
      for I := 0 to Sd - 1 do
        S := S + Sqr(Q[(R * Hk + Hd) * Sd + I]);
      S := 1 / Sqrt(S + D.Eps);
      for I := 0 to Sd - 1 do
        Q[(R * Hk + Hd) * Sd + I] := Q[(R * Hk + Hd) * Sd + I] * S / Sqrt(Sd);
      S := 0;
      for I := 0 to Sd - 1 do
        S := S + Sqr(K[(R * Hk + Hd) * Sd + I]);
      S := 1 / Sqrt(S + D.Eps);
      for I := 0 to Sd - 1 do
        K[(R * Hk + Hd) * Sd + I] := K[(R * Hk + Hd) * Sd + I] * S;
    end;
  end;
  { the gated delta rule per value head; value head j uses key head j mod Hk }
  O := nil;
  SetLength(O, T * Hv * Sd);
  M := nil;
  SetLength(M, Sd * Sd);   { M[a, b]: value a, key b }
  for Hd := 0 to Hv - 1 do
  begin
    Kh := Hd mod Hk;
    for I := 0 to High(M) do
      M[I] := 0;
    for R := 0 to T - 1 do
    begin
      Beta := Sigmoid(Bz[R * Hv + Hd]);
      Gt := Softplus(Az[R * Hv + Hd] + DtB[Hd]) * SsmA[Hd];
      for I := 0 to High(M) do
        M[I] := M[I] * Exp(Gt);
      for A := 0 to Sd - 1 do
      begin
        Pred := 0;
        for B := 0 to Sd - 1 do
          Pred := Pred + M[A * Sd + B] * K[(R * Hk + Kh) * Sd + B];
        Dlt := Beta * (V[(R * Hv + Hd) * Sd + A] - Pred);
        for B := 0 to Sd - 1 do
          M[A * Sd + B] := M[A * Sd + B] + Dlt * K[(R * Hk + Kh) * Sd + B];
      end;
      for A := 0 to Sd - 1 do
      begin
        S := 0;
        for B := 0 to Sd - 1 do
          S := S + M[A * Sd + B] * Q[(R * Hk + Kh) * Sd + B];
        O[(R * Hv + Hd) * Sd + A] := S;
      end;
    end;
  end;
  { rms over each head's Sd values with ssm_norm, times silu(z) }
  Ot := RefRms(O, SsmNorm, D.Eps);
  for I := 0 to High(Ot) do
    Ot[I] := Ot[I] * Silu(Z[I]);
  Result := MatVec(G, Pfx + 'ssm_out.weight', Ot, T, Hv * Sd);
end;

function FullAttn(G: TGgufFile; const Pfx: RawByteString; const D: TRefDims;
  const X: TDoubleArray; T: Integer): TDoubleArray;
var
  QG, Q, Gate, K, V, QNorm, KNorm, Att, Sc: TDoubleArray;
  N: Int64;
  Hq, Hkv, Dh, R, H, I, J, Kv: Integer;
  S, Mx, Sum: Double;
begin
  Hq := D.HeadsQ;
  Hkv := D.HeadsKV;
  Dh := D.HeadDim;
  QG := MatVec(G, Pfx + 'attn_q.weight', X, T, D.Hidden);   { [T, Hq, 2 Dh] }
  K := MatVec(G, Pfx + 'attn_k.weight', X, T, D.Hidden);
  V := MatVec(G, Pfx + 'attn_v.weight', X, T, D.Hidden);
  QNorm := Values(G, Pfx + 'attn_q_norm.weight', N);
  KNorm := Values(G, Pfx + 'attn_k_norm.weight', N);
  Q := nil; Gate := nil;
  SetLength(Q, T * Hq * Dh);
  SetLength(Gate, T * Hq * Dh);
  for R := 0 to T - 1 do
    for H := 0 to Hq - 1 do
      for I := 0 to Dh - 1 do
      begin
        Q[(R * Hq + H) * Dh + I] := QG[(R * Hq + H) * 2 * Dh + I];
        Gate[(R * Hq + H) * Dh + I] := QG[(R * Hq + H) * 2 * Dh + Dh + I];
      end;
  Q := RefRms(Q, QNorm, D.Eps);
  K := RefRms(K, KNorm, D.Eps);
  Q := RefRope(Q, T, Hq, Dh, D.Rot, D.RopeBase, 0);
  K := RefRope(K, T, Hkv, Dh, D.Rot, D.RopeBase, 0);
  Att := nil; Sc := nil;
  SetLength(Att, T * Hq * Dh);
  SetLength(Sc, T);
  for H := 0 to Hq - 1 do
  begin
    Kv := H div (Hq div Hkv);
    for R := 0 to T - 1 do
    begin
      Mx := -Infinity;
      for J := 0 to R do
      begin
        S := 0;
        for I := 0 to Dh - 1 do
          S := S + Q[(R * Hq + H) * Dh + I] * K[(J * Hkv + Kv) * Dh + I];
        Sc[J] := S / Sqrt(Dh);
        if Sc[J] > Mx then
          Mx := Sc[J];
      end;
      Sum := 0;
      for J := 0 to R do
      begin
        Sc[J] := Exp(Sc[J] - Mx);
        Sum := Sum + Sc[J];
      end;
      for I := 0 to Dh - 1 do
      begin
        S := 0;
        for J := 0 to R do
          S := S + Sc[J] / Sum * V[(J * Hkv + Kv) * Dh + I];
        Att[(R * Hq + H) * Dh + I] := S * Sigmoid(Gate[(R * Hq + H) * Dh + I]);
      end;
    end;
  end;
  Result := MatVec(G, Pfx + 'attn_output.weight', Att, T, Hq * Dh);
end;

function RefBlock(G: TGgufFile; ABlock: Integer; const D: TRefDims;
  const H: TDoubleArray; T: Integer): TDoubleArray;
var
  Pfx: RawByteString;
  Nrm, X, A, H1, Gt, Up, F: TDoubleArray;
  N: Int64;
  I: Integer;
begin
  Pfx := 'blk.' + IntToStr(ABlock) + '.';
  Nrm := Values(G, Pfx + 'attn_norm.weight', N);
  X := RefRms(H, Nrm, D.Eps);
  if (ABlock + 1) mod D.FullInterval = 0 then
    A := FullAttn(G, Pfx, D, X, T)
  else
    A := LinearAttn(G, Pfx, D, X, T);
  H1 := Copy(H);
  for I := 0 to High(H1) do
    H1[I] := H1[I] + A[I];
  Nrm := Values(G, Pfx + 'post_attention_norm.weight', N);
  X := RefRms(H1, Nrm, D.Eps);
  Gt := MatVec(G, Pfx + 'ffn_gate.weight', X, T, D.Hidden);
  Up := MatVec(G, Pfx + 'ffn_up.weight', X, T, D.Hidden);
  for I := 0 to High(Gt) do
    Gt[I] := Silu(Gt[I]) * Up[I];
  F := MatVec(G, Pfx + 'ffn_down.weight', Gt, T, D.Ffn);
  Result := H1;
  for I := 0 to High(Result) do
    Result[I] := Result[I] + F[I];
end;


function RefLogits(G: TGgufFile; const D: TRefDims; const AIds: array of Integer): TDoubleArray;
var
  E, H, W: TDoubleArray;
  N: Int64;
  T, I, J, Blocks: Integer;
  Head: RawByteString;
begin
  T := Length(AIds);
  E := Values(G, 'token_embd.weight', N);
  H := nil;
  SetLength(H, T * D.Hidden);
  for I := 0 to T - 1 do
    for J := 0 to D.Hidden - 1 do
      H[I * D.Hidden + J] := E[Int64(AIds[I]) * D.Hidden + J];
  Blocks := G.KV[KeyIdx(G, 'qwen35.block_count')].IntVal;
  if G.FindKey('qwen35.nextn_predict_layers') >= 0 then
    Blocks := Blocks - G.KV[G.FindKey('qwen35.nextn_predict_layers')].IntVal;
  for I := 0 to Blocks - 1 do
    H := RefBlock(G, I, D, H, T);
  W := Values(G, 'output_norm.weight', N);
  H := RefRms(H, W, D.Eps);
  if G.FindTensor('output.weight') >= 0 then
    Head := 'output.weight'
  else
    Head := 'token_embd.weight';
  Result := MatVec(G, Head, H, T, D.Hidden);
end;

end.
