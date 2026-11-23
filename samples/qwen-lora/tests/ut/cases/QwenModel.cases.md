# UT cases — unit `QwenModel` (T11)

Context (T11-facts): M is `LoadCpu(<path>, G)` — `TGgufFile.Open(<path>)`
then `TQwenModel.Load(G, TDevice.Cpu, TDevice.Cpu, 0)`; the caller frees M
then G. The files (all seed 5, written at start by the skeleton) are
`TiedPath` (thTied), `OwnPath` (thOwn: `output.weight` F16 274 × 64),
`NextPath` (thTied, NextN: `block_count` 3 and
`qwen35.nextn_predict_layers` 1), `NonePath` (thNone: no `token_embd`),
`BadEmbdPath` (thBadEmbd: `token_embd` of 32 columns) and `BadOutPath`
(thBadOutput: `output.weight` of 273 rows). `TinyModelVocab` = 274.
`ShapeText` comes from `tests/ut/DeltaRuleFixture.pas`. Error rows catch
the named exception class and AssertEquals the whole message; a row fails
if no exception comes. "Exact" means the exact values (difference 0).

Stage 1 (task T11b1): `TrunkBlockCount`, `TQwenModel.Load`, `Destroy`,
`SetHeadChunk`, `Embed`, `Stages`; rows UT-QwenModel-001 to 016.

| Case ID | Title | Target function | Input (short) | Expected result | Notes |
|---|---|---|---|---|---|
| UT-QwenModel-001 | Load, tied head | TQwenModel.Load | M on TiedPath | TrunkBlocks 2, Split 0, Vocab 274, Tied True, HeadChunk 8192, Dims.Hidden 64 | |
| UT-QwenModel-002 | Load, own head | TQwenModel.Load | M on OwnPath | Tied False, Vocab 274 | |
| UT-QwenModel-003 | Load, NextN file | TQwenModel.Load | M on NextPath | ReadQwenDims(G).Blocks 3; TrunkBlockCount(G, ReadQwenDims(G)) 2; M.TrunkBlocks 2 | |
| UT-QwenModel-004 | TrunkBlockCount, opened file | TrunkBlockCount | TiedPath opened, no model | 2 | |
| UT-QwenModel-005 | Load, split 3 | TQwenModel.Load | TQwenModel.Load(G, Cpu, Cpu, 3), G on TiedPath | EQwenModel `split 3 outside 0..2` | |
| UT-QwenModel-006 | Load, split -1 | TQwenModel.Load | the same, split -1 | EQwenModel `split -1 outside 0..2` | |
| UT-QwenModel-007 | Load, no token_embd | TQwenModel.Load | Load(.., 0) on NonePath | EPackedLinear `tensor not in the file: token_embd.weight` | |
| UT-QwenModel-008 | Load, bad embedding shape | TQwenModel.Load | Load(.., 0) on BadEmbdPath | EQwenModel `token_embd.weight must be [vocab, hidden]` | |
| UT-QwenModel-009 | Load, bad output shape | TQwenModel.Load | Load(.., 0) on BadOutPath | EQwenModel `output.weight must be [vocab, hidden]` | |
| UT-QwenModel-010 | Embed, exact rows | TQwenModel.Embed | M on TiedPath; E := M.Embed([3, 0, 273]) | ShapeText(E) `[3, 64]`; exact: equal to the rows 3, 0, 273 of DequantizeBytes(8, the bytes of token_embd.weight, 274 * 64) (G.ReadTensorBytes(G.FindTensor(..))) | |
| UT-QwenModel-011 | Embed, empty array | TQwenModel.Embed | M.Embed(<empty array>) | EQwenModel `no token ids` | |
| UT-QwenModel-012 | Embed, id 274 | TQwenModel.Embed | M.Embed([274]) | EQwenModel `token id 274 outside 0..273` | |
| UT-QwenModel-013 | Embed, id -1 | TQwenModel.Embed | M.Embed([5, -1]) | EQwenModel `token id -1 outside 0..273` | |
| UT-QwenModel-014 | Stages | TQwenModel.Stages | S := M.Stages | Length 2; S[0].ClassName TLinearBlock, S[1].ClassName TFullBlock; S[0].Device.ToString `cpu` | |
| UT-QwenModel-015 | Load, split 1 and 2 | TQwenModel.Load | Load(G, Cpu, Cpu, 1) then Load(G, Cpu, Cpu, 2) on one G | Split 1, then 2 | |
| UT-QwenModel-016 | HeadChunk 0 | TQwenModel.SetHeadChunk | M.HeadChunk := 0 | EQwenModel `head chunk must be positive, got 0` | |

Stage 2 (task T11b2): `Trunk`, `Logits`, `LastLogProbs`; rows
UT-QwenModel-101 to 110. Lp(R, t) is the log-softmax of row t of
`RefLogits` computed in the test in Double:
`R[t, j] - max - Ln(Σ exp(R[t, k] - max))`.

| Case ID | Title | Target function | Input (short) | Expected result | Notes |
|---|---|---|---|---|---|
| UT-QwenModel-101 | Logits vs scalar oracle, tied head | TQwenModel.Trunk / Logits | `LogitsDiff(TiedPath, Ids8)` | ≤ 1e-4 (measured 1.9e-5) | |
| UT-QwenModel-102 | Logits vs scalar oracle, own head | TQwenModel.Trunk / Logits | `LogitsDiff(OwnPath, Ids8)` | ≤ 1e-4 (measured 2.1e-6) | |
| UT-QwenModel-103 | Logits vs scalar oracle, NextN file | TQwenModel.Trunk / Logits | `LogitsDiff(NextPath, Ids8)` | ≤ 1e-4 (measured 1.9e-5) | |
| UT-QwenModel-104 | Logits vs oracle, T 70 tokens | TQwenModel.Trunk / Logits | `LogitsDiff(TiedPath, Ids)`, Ids[I] = `I * 37 mod 274`, I = 0..69 (T 70: two delta-rule chunks) | ≤ 1e-4 (measured 3.4e-5) | |
| UT-QwenModel-105 | HeadChunk 100: three chunks | TQwenModel.Logits / SetHeadChunk | M on `TiedPath`; H := `M.Trunk(M.Embed(Ids8))`; A := `M.Logits(H)`; `M.HeadChunk := 100`; B := `M.Logits(H)` (chunks of 100, 100, 74 rows) | HeadChunk 100; `ShapeText(B)` `[8, 274]`; `DiffArr(Arr(A), Arr(B))` ≤ 1e-5 (measured 3.8e-6) | |
| UT-QwenModel-106 | LastLogProbs vs oracle row 7 | TQwenModel.LastLogProbs | `LP := M.LastLogProbs(Ids8)` on `TiedPath` | `ShapeText(LP)` `[274]`; `LP.RequiresGrad` False; `Arr(LP)` vs Lp(RefLogits(G, RefDims(G), Ids8), 7) ≤ 1e-4 (measured 9.9e-6) | |
| UT-QwenModel-107 | Logits, H not [T, Hidden] | TQwenModel.Logits | `M.Logits(TTensor.zeros([3, 32]))` | EQwenModel `H must be [T, 64]` | |
| UT-QwenModel-108 | Logits, H 1-D | TQwenModel.Logits | `M.Logits(TTensor.zeros([64]))` | EQwenModel `H must be [T, 64]` | |
| UT-QwenModel-109 | Backward through the model | TQwenModel.Trunk / Logits | X := `M.Embed(Ids8).detach.Copy.requires_grad_(True)`; `M.Logits(M.Trunk(X)).sum(kNone).Backward` | `ShapeText(X.grad)` `[8, 64]`; `X.grad.isfinite.all.DoubleValue([]) = 1.0` | |
| UT-QwenModel-110 | LastLogProbs, one token | TQwenModel.LastLogProbs | `M.LastLogProbs([5])` (one token) | max over j of abs(LP[j] − Lp(RefLogits(G, RefDims(G), [5]), 0)[j]) ≤ 1e-4 (measured 1.2e-5) | |
| UT-QwenModel-201 | AttachLora, counts, names and attachment | TQwenModel.AttachLora | M := `LoadCpu(TiedPath, G)` (unless said otherwise); S := `TLoraSet.Create(4, 2.0)`; free M, then S, then G; `M.AttachLora(S)` | 7; Count 7; `PairNames(S)` = `TinyPairs`; ParameterCount 4864; block 0 (`M.Stages[0] as TQwenBlock`): `Target('attn_qkv').Lora = S.Pair(0)`, `Target('ssm_out').Lora = S.Pair(2)`, `Target('ssm_beta').Lora = nil`, `Target('ffn_gate').Lora = nil`; block 1 `Target('attn_output').Lora = S.Pair(6)` | added in T12b4 |
| UT-QwenModel-202 | AttachLora, output unchanged with B zero | TQwenModel.AttachLora | M0 := a second `TQwenModel.Load(G, Cpu, Cpu, 0)`; `M.AttachLora(S)` | `NoGradLogits(M)` equals `NoGradLogits(M0)` (exact) | added in T12b4 |
| UT-QwenModel-203 | AttachLora, random B changes the output | TQwenModel.AttachLora | as 202, then `RandomB(S)`; then under `NoGradGuard` H := `RecomputeForward(M.Stages, M.Embed(Ids8), Saved)` | `NoGradLogits(M)` vs `NoGradLogits(M0)` > 0.1 (measured 18.1); H equals `M.Trunk(M.Embed(Ids8))` (exact) | added in T12b4 |
| UT-QwenModel-204 | AttachLora, block-wise grads vs the whole graph | TQwenModel.AttachLora | `M.AttachLora(S)`; `RandomB(S)`; `M.Logits(M.Trunk(M.Embed(Ids8))).log_softmax(-1).sum(kNone).Backward`, keep every A.grad, B.grad, `ZeroGrad`; then the same loss block-wise (`RecomputeForward`, `Logits` on H detached with `requires_grad_`, `RecomputeBackward` with `H.grad`) | every pair's B.grad max \|·\| > 0; max \|grad block-wise − whole graph\| over all A and B ≤ 1e-6 (measured 0) | added in T12b4 |
| UT-QwenModel-205 | AttachLora twice on the same set | TQwenModel.AttachLora | `M.AttachLora(S)` twice | ELora `pair already present: blk.0.attn_qkv`; Count 7 | added in T12b4 |
| UT-QwenModel-206 | AttachLora, NextN file | TQwenModel.AttachLora | M := `LoadCpu(NextPath, G)`; `M.AttachLora(S)` | 7; `PairNames(S)` = `TinyPairs` (no MTP block) | added in T12b4 |
| UT-QwenModel-207 | AttachLora, split 1 | TQwenModel.AttachLora | `TQwenModel.Load(G, Cpu, Cpu, 1)` on `TiedPath`; `M.AttachLora(S)` | 7; `Length(S.Stores)` 1; `S.Stores[0].Count` 14 | added in T12b4 |
