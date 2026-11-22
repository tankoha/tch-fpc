{ T08: the gated delta rule of the Gated DeltaNet blocks, token by token
  (DESIGN 8.10, linear block step 7). The reference form of the rule; T09
  adds the chunked form and is checked against this one.

  Per head, state M [Dv, Dk] (value x key), zero at the sequence start;
  per token t:
    M := exp(g_t) * M
    d := beta_t * (v_t - M k_t)
    M := M + d k_t^T
    o_t := M q_t
  q is used as given (the caller scales it by 1/sqrt(Dk), DESIGN step 6).
  Everything is fp32 (kFloat), on the inputs' device.

  Interface by the harness owner (tasks/T08-facts.md); the bodies are
  written in T08b. }
unit DeltaRule;

{$mode objfpc}{$H+}

interface

uses
  SysUtils, Tch;

type
  EDeltaRule = class(Exception);

{ One token, all heads. M [H, Dv, Dk]; Q, K [H, Dk]; V [H, Dv]; Beta, G [H].
  Returns the new state [H, Dv, Dk]; O receives the output [H, Dv].
  Raises EDeltaRule (messages in the facts) for a kind or shape error. }
function DeltaStep(const M, Q, K, V, Beta, G: TTensor; out O: TTensor): TTensor;

{ A whole sequence from a zero state. Q, K [H, T, Dk]; V [H, T, Dv];
  Beta, G [H, T]; T >= 1. Returns O [H, T, Dv]; State receives the final
  state [H, Dv, Dk]. Raises EDeltaRule (messages in the facts). }
function DeltaRuleTokens(const Q, K, V, Beta, G: TTensor; out State: TTensor): TTensor;

{ T09: the same as DeltaRuleTokens in the chunked form (chunks of ChunkSize
  tokens, the sequence padded with zero tokens to a multiple of it; one
  triangular solve per chunk). G must be finite (-inf gives NaN here).
  Raises EDeltaRule (messages in tasks/T09-facts.md). }
function DeltaRuleChunked(const Q, K, V, Beta, G: TTensor; out State: TTensor;
  ChunkSize: Integer = 64): TTensor;

implementation

function ShapeStr(const T: TTensor): string;
var
  S: TInt64Array;
  I: Integer;
begin
  S := T.Shape;
  Result := '[';
  for I := 0 to High(S) do
  begin
    if I > 0 then
      Result := Result + ', ';
    Result := Result + IntToStr(S[I]);
  end;
  Result := Result + ']';
end;

function DeltaStep(const M, Q, K, V, Beta, G: TTensor; out O: TTensor): TTensor;
var
  H, Dv, Dk: Int64;
  Md, Pred, D: TTensor;
begin
  O := Default(TTensor);
  Result := Default(TTensor);
  { Kind checks, in order M, Q, K, V, Beta, G }
  if M.Kind <> kFloat then
    raise EDeltaRule.Create('M must be kFloat (fp32)');
  if Q.Kind <> kFloat then
    raise EDeltaRule.Create('Q must be kFloat (fp32)');
  if K.Kind <> kFloat then
    raise EDeltaRule.Create('K must be kFloat (fp32)');
  if V.Kind <> kFloat then
    raise EDeltaRule.Create('V must be kFloat (fp32)');
  if Beta.Kind <> kFloat then
    raise EDeltaRule.Create('Beta must be kFloat (fp32)');
  if G.Kind <> kFloat then
    raise EDeltaRule.Create('G must be kFloat (fp32)');
  { Shape checks }
  if Length(M.Shape) <> 3 then
    raise EDeltaRule.Create('M must be [H, Dv, Dk], got ' + ShapeStr(M));
  H := M.Shape[0];
  Dv := M.Shape[1];
  Dk := M.Shape[2];
  if (Length(Q.Shape) <> 2) or (Q.Shape[0] <> H) or (Q.Shape[1] <> Dk) then
    raise EDeltaRule.Create('Q must be [' + IntToStr(H) + ', ' + IntToStr(Dk) + '], got ' + ShapeStr(Q));
  if (Length(K.Shape) <> 2) or (K.Shape[0] <> H) or (K.Shape[1] <> Dk) then
    raise EDeltaRule.Create('K must be [' + IntToStr(H) + ', ' + IntToStr(Dk) + '], got ' + ShapeStr(K));
  if (Length(V.Shape) <> 2) or (V.Shape[0] <> H) or (V.Shape[1] <> Dv) then
    raise EDeltaRule.Create('V must be [' + IntToStr(H) + ', ' + IntToStr(Dv) + '], got ' + ShapeStr(V));
  if (Length(Beta.Shape) <> 1) or (Beta.Shape[0] <> H) then
    raise EDeltaRule.Create('Beta must be [' + IntToStr(H) + '], got ' + ShapeStr(Beta));
  if (Length(G.Shape) <> 1) or (G.Shape[0] <> H) then
    raise EDeltaRule.Create('G must be [' + IntToStr(H) + '], got ' + ShapeStr(G));
  { Implementation (five lines from facts) }
  Md := M * G.exp.view([H, 1, 1]);
  Pred := Md.matmul(K.unsqueeze(2)).squeeze_dim(2);          { [H, Dv] }
  D := (V - Pred) * Beta.unsqueeze(1);                        { [H, Dv] }
  Result := Md + D.unsqueeze(2).matmul(K.unsqueeze(1));       { [H, Dv, Dk] }
  O := Result.matmul(Q.unsqueeze(2)).squeeze_dim(2);          { [H, Dv] }
end;

function DeltaRuleTokens(const Q, K, V, Beta, G: TTensor; out State: TTensor): TTensor;
var
  H, T, Dk, Dv: Int64;
  I: Int64;
  M, Ot: TTensor;
  Outs: array of TTensor;
begin
  State := Default(TTensor);
  Result := Default(TTensor);
  { Kind checks, in order Q, K, V, Beta, G }
  if Q.Kind <> kFloat then
    raise EDeltaRule.Create('Q must be kFloat (fp32)');
  if K.Kind <> kFloat then
    raise EDeltaRule.Create('K must be kFloat (fp32)');
  if V.Kind <> kFloat then
    raise EDeltaRule.Create('V must be kFloat (fp32)');
  if Beta.Kind <> kFloat then
    raise EDeltaRule.Create('Beta must be kFloat (fp32)');
  if G.Kind <> kFloat then
    raise EDeltaRule.Create('G must be kFloat (fp32)');
  { Shape checks }
  if Length(Q.Shape) <> 3 then
    raise EDeltaRule.Create('Q must be [H, T, Dk], got ' + ShapeStr(Q));
  H := Q.Shape[0];
  T := Q.Shape[1];
  Dk := Q.Shape[2];
  if T = 0 then
    raise EDeltaRule.Create('no tokens (T = 0)');
  if not ((Length(K.Shape) = 3) and (K.Shape[0] = H) and (K.Shape[1] = T) and (K.Shape[2] = Dk)) then
    raise EDeltaRule.Create('K must be the shape of Q ' + ShapeStr(Q) + ', got ' + ShapeStr(K));
  if not ((Length(V.Shape) = 3) and (V.Shape[0] = H) and (V.Shape[1] = T)) then
    raise EDeltaRule.Create('V must be [' + IntToStr(H) + ', ' + IntToStr(T) + ', Dv], got ' + ShapeStr(V));
  Dv := V.Shape[2];
  if not ((Length(Beta.Shape) = 2) and (Beta.Shape[0] = H) and (Beta.Shape[1] = T)) then
    raise EDeltaRule.Create('Beta must be [' + IntToStr(H) + ', ' + IntToStr(T) + '], got ' + ShapeStr(Beta));
  if not ((Length(G.Shape) = 2) and (G.Shape[0] = H) and (G.Shape[1] = T)) then
    raise EDeltaRule.Create('G must be [' + IntToStr(H) + ', ' + IntToStr(T) + '], got ' + ShapeStr(G));
  { Implementation: loop calling DeltaStep once per token }
  M := TTensor.zeros([H, Dv, Dk], kFloat, Q.Device);
  SetLength(Outs, T);
  for I := 0 to T - 1 do
    M := DeltaStep(M, Q.select(1, I), K.select(1, I), V.select(1, I),
                   Beta.select(1, I), G.select(1, I), Outs[I]);
  Result := TTensor.stack(Outs, 1);
  State := M;
end;

function DeltaRuleChunked(const Q, K, V, Beta, G: TTensor; out State: TTensor;
  ChunkSize: Integer): TTensor;
var
  H, T, Dk, Dv, C, N, P, I: Int64;
  Qp, Kp, Vp, Bp, Gp, KB, VB, Gc, Decay, L, Eye, Tm, U, W, S: TTensor;
  Qi, Ki, Ui, Wi, Gi, Di, Ai, VNew, Last: TTensor;
  Outs: array of TTensor;
begin
  State := Default(TTensor);
  Result := Default(TTensor);
  { Kind checks, in order Q, K, V, Beta, G }
  if Q.Kind <> kFloat then
    raise EDeltaRule.Create('Q must be kFloat (fp32)');
  if K.Kind <> kFloat then
    raise EDeltaRule.Create('K must be kFloat (fp32)');
  if V.Kind <> kFloat then
    raise EDeltaRule.Create('V must be kFloat (fp32)');
  if Beta.Kind <> kFloat then
    raise EDeltaRule.Create('Beta must be kFloat (fp32)');
  if G.Kind <> kFloat then
    raise EDeltaRule.Create('G must be kFloat (fp32)');
  { Shape checks and the chunk size }
  if Length(Q.Shape) <> 3 then
    raise EDeltaRule.Create('Q must be [H, T, Dk], got ' + ShapeStr(Q));
  H := Q.Shape[0];
  T := Q.Shape[1];
  Dk := Q.Shape[2];
  if T = 0 then
    raise EDeltaRule.Create('no tokens (T = 0)');
  if not ((Length(K.Shape) = 3) and (K.Shape[0] = H) and (K.Shape[1] = T) and (K.Shape[2] = Dk)) then
    raise EDeltaRule.Create('K must be the shape of Q ' + ShapeStr(Q) + ', got ' + ShapeStr(K));
  if not ((Length(V.Shape) = 3) and (V.Shape[0] = H) and (V.Shape[1] = T)) then
    raise EDeltaRule.Create('V must be [' + IntToStr(H) + ', ' + IntToStr(T) + ', Dv], got ' + ShapeStr(V));
  Dv := V.Shape[2];
  if not ((Length(Beta.Shape) = 2) and (Beta.Shape[0] = H) and (Beta.Shape[1] = T)) then
    raise EDeltaRule.Create('Beta must be [' + IntToStr(H) + ', ' + IntToStr(T) + '], got ' + ShapeStr(Beta));
  if not ((Length(G.Shape) = 2) and (G.Shape[0] = H) and (G.Shape[1] = T)) then
    raise EDeltaRule.Create('G must be [' + IntToStr(H) + ', ' + IntToStr(T) + '], got ' + ShapeStr(G));
  if ChunkSize < 1 then
    raise EDeltaRule.Create('chunk size must be at least 1, got ' + IntToStr(ChunkSize));
  { Implementation (from tasks/T09-facts.md, as-is) }
  C := ChunkSize;
  N := (T + C - 1) div C;
  P := N * C - T;
  Qp := Q; Kp := K; Vp := V; Bp := Beta; Gp := G;
  if P > 0 then
  begin
    Qp := TTensor.cat([Q, TTensor.zeros([H, P, Dk], kFloat, Q.Device)], 1);
    Kp := TTensor.cat([K, TTensor.zeros([H, P, Dk], kFloat, Q.Device)], 1);
    Vp := TTensor.cat([V, TTensor.zeros([H, P, Dv], kFloat, Q.Device)], 1);
    Bp := TTensor.cat([Beta, TTensor.zeros([H, P], kFloat, Q.Device)], 1);
    Gp := TTensor.cat([G, TTensor.zeros([H, P], kFloat, Q.Device)], 1);
  end;
  KB := (Kp * Bp.unsqueeze(2)).reshape([H, N, C, Dk]);
  VB := (Vp * Bp.unsqueeze(2)).reshape([H, N, C, Dv]);
  Qp := Qp.reshape([H, N, C, Dk]);
  Kp := Kp.reshape([H, N, C, Dk]);
  Gc := Gp.reshape([H, N, C]).cumsum(2);
  Decay := (Gc.unsqueeze(3) - Gc.unsqueeze(2)).tril.exp.tril;
  L := (KB.matmul(Kp.transpose(2, 3)) * Decay).tril(-1);
  Eye := TTensor.eye(C, kFloat, Q.Device);
  Tm := (L + Eye).linalg_solve_triangular(Eye.expand([H, N, C, C]), False);
  U := Tm.matmul(VB);
  W := Tm.matmul(KB * Gc.exp.unsqueeze(3));
  S := TTensor.zeros([H, Dk, Dv], kFloat, Q.Device);
  SetLength(Outs, N);
  for I := 0 to N - 1 do
  begin
    Qi := Qp.select(1, I);
    Ki := Kp.select(1, I);
    Ui := U.select(1, I);
    Wi := W.select(1, I);
    Gi := Gc.select(1, I);
    Di := Decay.select(1, I);
    Ai := (Qi.matmul(Ki.transpose(1, 2)) * Di).tril;
    VNew := Ui - Wi.matmul(S);
    Outs[I] := (Qi * Gi.exp.unsqueeze(2)).matmul(S) + Ai.matmul(VNew);
    Last := Gi.narrow(1, C - 1, 1);
    S := S * Last.exp.unsqueeze(2)
         + (Ki * (Last - Gi).exp.unsqueeze(2)).transpose(1, 2).matmul(VNew);
  end;
  Result := TTensor.stack(Outs, 1).reshape([H, N * C, Dv]).narrow(1, 0, T);
  State := S.transpose(1, 2);
end;

end.
