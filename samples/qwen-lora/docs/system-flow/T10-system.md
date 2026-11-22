# System flow — T10 (PackedLinear, QwenBlocks)

## Callers

### T11 (model initialisation and trunk execution)

T11's model initialisation opens the GGUF with `TGgufFile`, reads the dimensions with `ReadQwenDims`, and loads each trunk block with `LoadBlock(G, Block, Dims, Device)`. The blocks are stored as `TStage` instances.

T11 runs the trunk as `Recompute` stages (T13): each block's `Forward` is called within the recomputation framework, checkpointing activations to trade memory for compute. The full trunk is:

```pascal
for I := 0 to Dims.Blocks - 1 do
  H := Blocks[I].Forward(H);
```

where `Blocks[I]` is a `TQwenBlock` (either `TLinearBlock` or `TFullBlock`).

### T12 (LoRA)

T12 adds LoRA adapters to `TPackedLinear`. The `Forward` method is wrapped to compute:

```pascal
Result := X.matmul((Weight + LoraA.matmul(LoraB)).transpose(0, 1));
```

where `LoraA` and `LoraB` are low-rank fp32 matrices trained during fine-tuning. The base `Weight` is dequantised per call as before.

### T17 (inference and training loop)

T17 uses the trunk stages for both forward (inference) and backward (training with gradient checkpointing). The `Recompute` framework in T13 handles the backward pass by re-executing forward on the checkpointed inputs.

## Test callers

### UT: comparison with scalar reference

The unit tests use two fixtures:
- `tests/ut/QwenBlocksFixture.pas`: `WriteTinyQwen(Path, Seed)` writes a tiny model (hidden 64, FFN 96, 4 query / 2 KV heads of 32, SSM 2 key / 4 value heads of 16, kernel 4, FullInterval 2: blk.0 linear, blk.1 full)
- `tests/ut/QwenRefScalar.pas`: `RefBlock(G, Block, Dims, H, T)` computes a block's output in Double with plain loops (no tensor ops), as an oracle

Each test:
1. Writes the tiny model once (`TinyPath`)
2. Opens it with `TGgufFile.Open(TinyPath)`
3. Reads `D := ReadQwenDims(G)`, `RD := RefDims(G)`
4. Loads a block with the appropriate `Load` or `LoadBlock`
5. Generates input `H` with `ManualSeed(40 + T); TTensor.randn([T, 64])`
6. Calls `B.Forward(H)` and compares `Arr(Result)` with `RefBlock(G, Block, RD, Arr(H), T)` within a tolerance

The scalar reference is deliberately slow (Double, element-wise loops) but correct; the tensor version must match within 1e-4.

### ITa: rehearsal model blocks

The ITa driver `tests/ita/T10/blocks_check.pas`:
1. Opens the rehearsal GGUF (path from `QLORA_MODEL` environment variable)
2. Reads dimensions with `ReadQwenDims`
3. For each of the 24 blocks, loads with `LoadBlock`, runs with T=8 on random input, compares with the scalar version
4. Runs selected blocks with T=70 (longer sequences test chunking and GQA)
5. Chains blocks 0-3 and compares the chain output
6. Runs blocks 0-3 as `Recompute` stages and compares the input gradient with an ordinary backward

Result written to `<project root>/work/ita/T10/summary.tsv`.

## Dependencies

| Unit | Depends on |
|---|---|
| PackedLinear | Tch, TchNN (tensor ops), GgufFile (GGUF reading), GgufQuant (DequantizeT, GgufTensorBytes, GgufTypeName) |
| QwenBlocks | Tch, TchNN, GgufFile, PackedLinear, DeltaRuleChunked (T09) |

`DeltaRuleChunked` (T09) is the chunked delta rule recurrence used by `TLinearBlock`.

## Data flow

```
GGUF file
    |
   TGgufFile.Open
    |
    +-- ReadQwenDims --> TQwenDims
    |
    +-- LoadBlock(G, Block, Dims, Device)
           |
           +-- TensorRaw / LoadPacked / LoadDense
           |       |
           |       +-- DequantizeT (for dense)
           |       |
           |       +-- TPackedLinear.Create (raw bytes kept packed)
           |
           +-- TQwenBlock.Load (norms, FFN, attention weights)
                    |
                    +-- Forward(H)
                           |
                           +-- RmsNorm(H, FAttnNorm) --> Attn --> FFN --> output
```

In `Attn`:
- `TFullBlock`: Q, K, V linears -> RmsNorm on Q, K -> RoPE -> SDPA with GQA -> output linear with gate
- `TLinearBlock`: QKV, gate, beta, alpha linears -> convolution -> delta rule -> output linear with gate
