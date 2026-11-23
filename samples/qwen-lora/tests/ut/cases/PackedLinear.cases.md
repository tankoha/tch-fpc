# UT cases — unit `PackedLinear` (T10)

Context (T10-facts): "the Q8_0 matrix" means V := 64 Singles `I / 10 - 3`
(I = 0..63), Bytes := `QuantizeQ8_0(V)` (unit `GgufQuant`, 68 bytes),
Raw := `TTensor.FromBytes(Bytes[0], [68], kUInt8)`, and the rows 001–011
use `L := TPackedLinear.Create(Raw, 8, 2, 32)`. G is
`TGgufFile.Open(TinyPath)` (the tiny model of `tests/ut/QwenBlocksFixture.pas`).
Error rows catch `EPackedLinear` and AssertEquals the whole message; a row
fails if no exception comes. "exact" means the exact strings or a difference
of 0. "zeros" is `TTensor.zeros` (kFloat unless a kind is given).

Stage 1 (task T10b1): all of `PackedLinear`; rows UT-PackedLinear-001 to 015.

| Case ID | Title | Target function | Input (short) | Expected result | Notes |
|---|---|---|---|---|---|
| UT-PackedLinear-001 | Create, the Q8_0 matrix | TPackedLinear.Create | the Q8_0 matrix | OutDim 2, InDim 32, TType 8, `Device.ToString` `'cpu'` | |
| UT-PackedLinear-002 | Weight of the Q8_0 matrix | TPackedLinear.Weight | `L.Weight` | shape `[2, 32]`; exact: equal to `DequantizeBytes(8, Bytes, 64)` | |
| UT-PackedLinear-003 | Forward, X [3, 32] | TPackedLinear.Forward | `L.Forward(X)`, X := `TTensor.FromArray([1.0, -2.0, 0.5]).view([3, 1]) * TTensor.ones([1, 32])` | shape `[3, 2]`; equal (≤ 1e-5, measured 0) to the loop Σ_k W[j, k] X[i, k] over the `DequantizeBytes` values | |
| UT-PackedLinear-004 | Forward, 3-D X | TPackedLinear.Forward | `L.Forward(TTensor.ones([2, 5, 32]))` | shape `[2, 5, 2]` | |
| UT-PackedLinear-005 | F32 case | TPackedLinear.Weight | F32: 6 Singles `I - 2.5` as 24 bytes (`Move`), `TPackedLinear.Create(bytes tensor, 0, 2, 3).Weight` | `[2, 3]` `-2.5 -1.5 -0.5 0.5 1.5 2.5` | |
| UT-PackedLinear-006 | Create, wrong kind (kFloat) | TPackedLinear.Create | `Create(TTensor.zeros([34]), 8, 1, 32)` (kFloat) | `the packed bytes must be a 1-D kUInt8 tensor, got [34]` | |
| UT-PackedLinear-007 | Create, 2-D raw | TPackedLinear.Create | `Create(TTensor.zeros([2, 34], kUInt8), 8, 1, 32)` | `the packed bytes must be a 1-D kUInt8 tensor, got [2, 34]` | |
| UT-PackedLinear-008 | Create, dimension 0 | TPackedLinear.Create | `Create(TTensor.zeros([34], kUInt8), 8, 0, 32)` | `dimensions must be positive, got 0 x 32` | |
| UT-PackedLinear-009 | Create, wrong byte count | TPackedLinear.Create | `Create(TTensor.zeros([34], kUInt8), 8, 2, 32)` | `[34] holds 34 bytes, 2 x 32 of type q8_0 need 68` | |
| UT-PackedLinear-010 | Forward, X not kFloat | TPackedLinear.Forward | `Create(zeros([34], kUInt8), 8, 1, 32).Forward(TTensor.zeros([3, 32], kDouble))` | `X must be kFloat (fp32)` | |
| UT-PackedLinear-011 | Forward, wrong last dim | TPackedLinear.Forward | the same `.Forward(TTensor.zeros([3, 31]))` | `X must end in 32, got [3, 31]` | |
| UT-PackedLinear-012 | LoadPacked, ffn_down | LoadPacked | `LoadPacked(G, 'blk.0.ffn_down.weight', TDevice.Cpu)` | OutDim 64, InDim 96, TType 8 | |
| UT-PackedLinear-013 | LoadDense, ssm_conv1d | LoadDense | `LoadDense(G, 'blk.0.ssm_conv1d.weight', TDevice.Cpu)` | shape `[128, 4]`, Kind kFloat | |
| UT-PackedLinear-014 | LoadPacked, tensor not found | LoadPacked / TensorRaw | `LoadPacked(G, 'blk.9.x', TDevice.Cpu)` | `tensor not in the file: blk.9.x` | |
| UT-PackedLinear-015 | LoadPacked, not a matrix | LoadPacked | `LoadPacked(G, 'blk.0.attn_norm.weight', TDevice.Cpu)` | `blk.0.attn_norm.weight is not a matrix (1 dimensions)` | |
| UT-PackedLinear-101 | Lora, default | TPackedLinear.Lora | `NewLinear` (L := `NewLinear`; S a CPU store; P := `NewPair(S, 8, 6)`) | `L.Lora = nil` | added in T12b3 |
| UT-PackedLinear-102 | Forward with a pair, B zero | TPackedLinear.Lora / Forward | Y0 := `L.Forward(FixX)`; `L.Lora := P` (B zero) | `L.Lora = P`; `L.Forward(FixX)` equals Y0 (exact) | added in T12b3 |
| UT-PackedLinear-103 | Forward, the delta vs the references | TPackedLinear.Forward | `ManualSeed(2)`; `SetB(P)`; `L.Lora := P`; without a graph | Y := `L.Forward(FixX)`: vs `FixX.matmul(L.Weight.t) + P.Delta(FixX)` ≤ 1e-6 (measured 0); vs `FixX.matmul((L.Weight + P.MergedDelta).t)` ≤ 1e-5 (measured 6.0e-8); vs `FixX.matmul(L.Weight.t)` > 0.1 (measured 0.147) | added in T12b3 |
| UT-PackedLinear-104 | SetLora, InDim differs | TPackedLinear.SetLora | `L.Lora := NewPair(S, 7, 6)` | EPackedLinear `LoRA pair q is 7 -> 6, the matrix 8 -> 6`; `L.Lora = nil` | added in T12b3 |
| UT-PackedLinear-105 | SetLora, OutDim differs | TPackedLinear.SetLora | `L.Lora := NewPair(S, 8, 5)` | EPackedLinear `LoRA pair q is 8 -> 5, the matrix 8 -> 6`; `L.Lora = nil` | added in T12b3 |
| UT-PackedLinear-106 | Lora := nil removes the pair | TPackedLinear.Lora | Y0 := `L.Forward(FixX)`; `SetB(P)`; `L.Lora := P`; `L.Lora := nil` | `L.Lora = nil`; `L.Forward(FixX)` equals Y0 (exact) | added in T12b3 |
| UT-PackedLinear-107 | Backward through Forward | TPackedLinear.Forward | `ManualSeed(2)`; `L.Lora := P` (B zero); `L.Forward(FixX).sum(kNone).Backward` | `ShapeText(P.B.grad)` `[6, 3]`; max \|B.grad\| > 0.1; max \|A.grad\| 0 (exact) | added in T12b3 |
