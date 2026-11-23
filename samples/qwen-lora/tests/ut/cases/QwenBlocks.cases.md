# UT cases — unit `QwenBlocks` (T10)

Context (T10-facts): G is `TGgufFile.Open(TinyPath)`; D is
`ReadQwenDims(G)`, RD is `RefDims(G)`. Oracle: the slow scalar version
`tests/ut/QwenRefScalar.pas` (`RefDims`, `RefBlock`, `RefRope`, `RefRms`);
`Arr(T)` is a tensor's values as `TDoubleArray`; `MaxAbsDiff`, `ShapeText`,
`ValuesText` come from `tests/ut/DeltaRuleFixture.pas` (T05's
`tests/ut/TokenizerFixture.pas` has `WriteTinyGguf(Path)`, a GGUF without any
`qwen35` key). Error rows catch `EQwenBlock` and AssertEquals the whole
message; a row fails if no exception comes. A tolerance row passes when the
difference is ≤ the limit; "exact" means the exact strings or a difference
of 0.

Stage 2 (task T10b2): the helpers; rows UT-QwenBlocks-101 to 107.
Stage 3 (task T10b3): B := `TFullBlock.Load(G, 1, D, TDevice.Cpu)`; H for
T rows := `ManualSeed(40 + T); TTensor.randn([T, 64])`; rows UT-QwenBlocks-201
to 204.
Stage 4 (task T10b4): B := `TLinearBlock.Load(G, 0, D, TDevice.Cpu)`; H as
above; rows UT-QwenBlocks-211 to 215, plus UT-QwenBlocks-216 (added in
T10b4 for 100 % line coverage: the `KeyFloat` missing-key branch, a
metadata file that lacks `qwen35.rope.freq_base`).

| Case ID | Title | Target function | Input (short) | Expected result | Notes |
|---|---|---|---|---|---|
| UT-QwenBlocks-101 | ReadQwenDims, the tiny model | ReadQwenDims | `ReadQwenDims(G)` | Blocks 2, Hidden 64, Ffn 96, HeadsQ 4, HeadsKV 2, HeadDim 32, Rot 16, RopeBase 10000, Eps 1e-6 (delta 1e-12), SsmHeadsK 2, SsmHeadsV 4, SsmDim 16, ConvKernel 4, FullInterval 2 | |
| UT-QwenBlocks-102 | ReadQwenDims, metadata key missing | ReadQwenDims | `ReadQwenDims` of `WriteTinyGguf(GetTempDir + 'tiny-tok.gguf')` opened | `metadata key missing: qwen35.block_count` | the opened file |
| UT-QwenBlocks-103 | IsFullBlock, interval 4 | IsFullBlock | D with FullInterval 4; `IsFullBlock(D, I)` for I = 0..7 | F F F T F F F T | |
| UT-QwenBlocks-104 | RmsNorm, hand-computed | RmsNorm | `RmsNorm(TTensor.FromArray([3.0, 4.0]).view([1, 2]), TTensor.FromArray([1.0, 2.0]), 1e-6)` | `Arr` vs `RefRms([3.0, 4.0], [1.0, 2.0], 1e-6)` ≤ 1e-6 (measured 4.4e-8; values about 0.8485 2.2627) | |
| UT-QwenBlocks-105 | RopeText, positions 0..4 | RopeText | `ManualSeed(31)`; X := `TTensor.randn([5, 3, 32])`; `RopeText(X, 16, 10000)` | vs `RefRope(Arr(X), 5, 3, 32, 16, 10000, 0)` ≤ 1e-6 (measured 2.0e-7); exact: `narrow(2, 16, 16)` equals X's | |
| UT-QwenBlocks-106 | RopeText, Pos0 = 7 | RopeText | the X of 105; `RopeText(X, 16, 10000, 7)` | vs `RefRope(.., 7)` ≤ 1e-6 (measured 1.7e-7) | |
| UT-QwenBlocks-107 | RopeText, position 0 | RopeText | `RopeText(X.narrow(0, 0, 1), 16, 10000)` (position 0) | exact: equals `X.narrow(0, 0, 1)` | |
| UT-QwenBlocks-201 | TFullBlock, device and forward T=8 | TFullBlock / TQwenBlock.Forward | B; T 8 | `B.Device.ToString` `'cpu'`; `Arr(B.Forward(H))` vs `RefBlock(G, 1, RD, Arr(H), 8)` ≤ 1e-4 (measured 1.0e-6) | |
| UT-QwenBlocks-202 | TFullBlock, forward T=70 | TFullBlock / TQwenBlock.Forward | B; T 70 | ≤ 1e-4 (measured 1.3e-6) | |
| UT-QwenBlocks-203 | TFullBlock, causality | TFullBlock / TQwenBlock.Forward | B; H of T 8 (`ManualSeed(48)`); `B.Forward(H)` and `B.Forward(H.narrow(0, 0, 5))` | the first 5 rows agree ≤ 1e-5 (measured 8.3e-7: causal) | |
| UT-QwenBlocks-204 | TFullBlock, backward | TFullBlock / Backward | B; H of T 8 as a leaf (`detach.Copy.requires_grad_(True)`); `B.Forward(H).sum(kNone).Backward` | `ShapeText(H.grad)` `[8, 64]`; `H.grad.isfinite.all` true | |
| UT-QwenBlocks-211 | TLinearBlock, forward T=8 | TLinearBlock / TQwenBlock.Forward | B; T 8 | vs `RefBlock(G, 0, RD, Arr(H), 8)` ≤ 1e-4 (measured 1.2e-6) | |
| UT-QwenBlocks-212 | TLinearBlock, forward T=70 chunking | TLinearBlock / TQwenBlock.Forward | B; T 70 (two chunks of the delta rule) | ≤ 1e-4 (measured 6.2e-6) | |
| UT-QwenBlocks-213 | TLinearBlock, causality | TLinearBlock / TQwenBlock.Forward | B; as 203 | ≤ 1e-5 (measured 7.2e-7) | |
| UT-QwenBlocks-214 | TLinearBlock, backward | TLinearBlock / Backward | B; as 204 | `[8, 64]`, finite | |
| UT-QwenBlocks-215 | LoadBlock, class names | LoadBlock | `LoadBlock(G, 0, D, TDevice.Cpu)` and `LoadBlock(G, 1, ..)` | `ClassName` `'TLinearBlock'` and `'TFullBlock'` | |
| UT-QwenBlocks-216 | ReadQwenDims, float metadata key missing | ReadQwenDims / KeyFloat | `ReadQwenDims` of a GGUF carrying all the tiny-model keys except `qwen35.rope.freq_base` (`WriteQwenMetaNoRopeBase` of `tests/ut/QwenBlocksMetaFixture.pas`) | `metadata key missing: qwen35.rope.freq_base` | added in T10b4 for 100 % line coverage: the missing-key raise of `KeyFloat` (src/QwenBlocks.pas:94), the float-key counterpart of UT-QwenBlocks-102 |
| UT-QwenBlocks-301 | Target, block 0 (linear) | TLinearBlock.Target / Index | B := `TinyBlock(0, G)` (the caller frees the block, then G) | Index 0; `TargetText`: `attn_qkv 64 -> 128`, `attn_gate 64 -> 64`, `ssm_beta 64 -> 4`, `ssm_alpha 64 -> 4`, `ssm_out 64 -> 64`, `ffn_gate 64 -> 96`, `ffn_up 64 -> 96`, `ffn_down 96 -> 64` | added in T12b3 |
| UT-QwenBlocks-302 | Target, block 1 (full) | TFullBlock.Target / Index | B := `TinyBlock(1, G)` | Index 1; `attn_q 64 -> 256`, `attn_k 64 -> 64`, `attn_v 64 -> 64`, `attn_output 128 -> 64`, `ffn_gate 64 -> 96`, `ffn_up 64 -> 96`, `ffn_down 96 -> 64` | added in T12b3 |
| UT-QwenBlocks-303 | Target, name not of the kind | TLinearBlock.Target | block 0, `Target('attn_q')` | EQwenBlock `blk.0 has no matrix attn_q` | added in T12b3 |
| UT-QwenBlocks-304 | Target, upper-case name | TFullBlock.Target | block 1, `Target('ATTN_Q')` | EQwenBlock `blk.1 has no matrix ATTN_Q` | added in T12b3 |
| UT-QwenBlocks-305 | Target, name with .weight | TFullBlock.Target | block 1, `Target('attn_q.weight')` | EQwenBlock `blk.1 has no matrix attn_q.weight` | added in T12b3 |
| UT-QwenBlocks-306 | Target, empty name | TFullBlock.Target | block 1, `Target('')` | EQwenBlock `blk.1 has no matrix ` (the name empty) | added in T12b3 |
| UT-QwenBlocks-307 | LoraTargets of both kinds | TLinearBlock.LoraTargets / TFullBlock.LoraTargets | `string.Join(',', B.LoraTargets)` for blocks 0 and 1 | `attn_qkv,attn_gate,ssm_out`; `attn_q,attn_k,attn_v,attn_output` | added in T12b3 |
| UT-QwenBlocks-308 | Target returns the same matrix | TFullBlock.Target | block 1 | `Target('attn_v') = Target('attn_v')`; `Target('attn_v') <> Target('attn_k')` | added in T12b3 |
