# System flow — unit `Lora` and LoRA hooks (T12)

## Callers

### T16: Checkpoint

T16's checkpoint saves and loads the LoRA set. The training run keeps one
`TLoraSet` with the adapter pairs. On save, it calls `TLoraSet.Save` with
the checkpoint path. On load, it creates a fresh set with the same rank and
scale, calls `AttachLora` to create the pairs, then `Load` to restore the
weights.

### T17: Training

T17's training attaches the LoRA set to the model and gives its stores'
variables to T14's `LoraOptim`. The flow:

1. Create `TLoraSet` with rank 16, scale 2.
2. Call `TQwenModel.AttachLora` to add one pair per LoRA target of every
   trunk block and attach them to the matrices.
3. For each store in `TLoraSet.Stores`, pass its trainable variables to the
   optimizer.

Only the LoRA parameters are trained; the base model weights stay frozen.

### T18: Merge

T18's merge reads `MergedDelta` from each pair and adds it to the base
weight. The flow:

1. Load the model and attach the trained LoRA set.
2. For each adapted matrix, compute `W + MergedDelta` and write it to a new
   GGUF.
3. The merged GGUF has no LoRA tensors; the adaptation is baked in.

## Unit tests

### Tiny models

The UT use the tiny models of `tests/ut/QwenBlocksFixture.pas`:
- Block 0: linear, Hidden 64, matrices `attn_qkv` (64 -> 128), `attn_gate`
  (64 -> 64), `ssm_out` (64 -> 64)
- Block 1: full, `attn_q` (64 -> 256), `attn_k` (64 -> 64), `attn_v` (64 ->
  64), `attn_output` (128 -> 64)

At rank 4, this gives 7 pairs and 4864 parameters.

### Test stages

- T12b1: `LoraPath`, `TLoraPair` (`Create`, `Delta`, `MergedDelta`); rows
  UT-Lora-001 to 016.
- T12b2: `TLoraSet` (all of it); rows UT-Lora-101 to 114.
- T12b3: `TPackedLinear.SetLora` and the two lines of `Forward`;
  `TQwenBlock.Index`, the three `Target` and two `LoraTargets`; rows
  UT-PackedLinear-101 to 107, UT-QwenBlocks-301 to 308.
- T12b4: `TQwenModel.AttachLora`; rows UT-QwenModel-201 to 207.

## ITa driver

The ITa driver `tests/ita/T12/lora_check.pas` runs on the rehearsal model
(0.8B with 78 LoRA pairs at rank 16). It tests:

1. `AttachLora` on the rehearsal model: pairs, dimensions, and
   `ParameterCount` against the tensor directory (ITa-T12-001).
2. B = 0: `LastLogProbs` with adapters against a model without them
   (ITa-T12-002).
3. B random: `LastLogProbs` with adapters against a base forward on a GGUF
   whose target matrices are W + s·B·A (ITa-T12-003).
4. Gradients: block-wise recomputation against one backward through the
   whole graph (ITa-T12-004).
5. Save and Load: every A and B preserved (ITa-T12-005).

The driver runs on the CPU from a fresh directory outside the project.
Write `<project root>` literally in the command and result file, not the
absolute path.
