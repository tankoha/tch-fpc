# External design — unit `Lora` and LoRA hooks (T12)

## Purpose

T12 adds LoRA (Low-Rank Adaptation) adapters to the Qwen model. The unit
`Lora` provides `TLoraPair` (one pair of low-rank matrices A and B for a
single linear layer) and `TLoraSet` (a collection of pairs, one per adapted
matrix, with one `TVarStore` per device). The existing units `PackedLinear`,
`QwenBlocks`, and `QwenModel` gain hooks to attach and use LoRA adapters.

All tensors are fp32 on the CPU. A `TLoraSet` keeps one `TVarStore` per
device, so the same code runs the two-GPU split of the 27B model.

## Forward with LoRA

For a linear layer with weight matrix W [OutDim, InDim], the forward with a
LoRA adapter is:

```
y = x·Wᵀ + s·((x·Aᵀ)·Bᵀ)
```

where:
- x is the input tensor [..., InDim]
- A is [Rank, InDim], initialised uniform in ±1/√InDim (kaiming_uniform
  with a = √5, as common LoRA code does)
- B is [OutDim, Rank], initialised to zero
- s is the scale (alpha / rank; alpha = 32 for rank 16 gives scale 2)

The zero initialisation of B means a fresh adapter leaves the model's
output unchanged: DESIGN 8.18's acceptance "zero B leaves the output
unchanged".

The merged delta, used for weight merging after training, is:

```
ΔW = s·B·A     [OutDim, InDim]
```

This matches the two-product form: DESIGN 8.18's acceptance "`MergedDelta`
equals the two-product form".

## Names

Names follow the GGUF convention. A pair is named after its matrix:
`blk.12.attn_qkv` for the QKV projection of block 12. The variables in the
store are `blk.12.attn_qkv.lora_a` and `blk.12.attn_qkv.lora_b`.

The function `LoraPath` splits a dot-separated name at the dots and builds a
path: `LoraPath(S.Root, 'blk.12.attn_qkv')` is `S.Root / 'blk' / '12' /
'attn_qkv'`.

## Targets per block kind

A linear block (TLinearBlock) adapts:
- `attn_qkv` (QKV projection)
- `attn_gate` (output gate)
- `ssm_out` (SSM output projection)

A full block (TFullBlock) adapts:
- `attn_q` (query)
- `attn_k` (key)
- `attn_v` (value)
- `attn_output` (attention output)

Only the trunk blocks are adapted, not the MTP block (T18 uses the trained
LoRA weights, not adapters at inference time).

On the rehearsal 0.8B (2 trunk blocks, one linear, one full) at rank 16:
78 pairs and 4 915 200 parameters (measured, ITa-T12-001).

## One store per device

A `TLoraSet` owns the pairs and the stores. When `Add` is called with a
device, the set finds or creates a `TVarStore` for that device. The same
set can hold pairs on different devices, which supports the two-GPU split
of the 27B model.

A matrix (`TPackedLinear`) holds a reference to its pair (`Lora` property)
without owning it. Free the model (or stop calling it) before the set.

## Save and Load

`Save` writes every pair as one tch archive (`.ot`, T07's `TVarStore.Save`
format) through a temporary store on the CPU, so a file does not depend on
the device map. Rank and scale are not in the file; T16's checkpoint keeps
the run's settings.

`Load` reads a file into an existing set: every pair must be in the file
with its shape, and A and B keep their handles. A missing variable raises
`ETorchError`; a shape mismatch raises `ETorchNativeError`.

## Errors

### ELora (unit `Lora`)

- `TLoraPair.Create`: dimensions must be positive (either below 1), before
  any variable is registered.
- `Delta`: X has no dimension or its last dimension is not InDim.
- `TLoraSet.Create`: rank must be positive.
- `Add`: pair already present (names compared literally).
- `Pair`: index outside 0..Count-1.
- From tch-fpc, passed on unchanged: empty segment in a name (ETorchError);
  Load of a file without one of the set's variables (ETorchError);
  shape mismatch (ETorchNativeError); missing file (ETorchNativeError).

### EPackedLinear (unit `PackedLinear`)

- `SetLora`: a non-nil pair's dimensions differ from the matrix's.

### EQwenBlock (unit `QwenBlocks`)

- `Target`: the block has no matrix by that name (names compared literally).

`AttachLora` raises what `Add` raises: a second call on the same set raises
"pair already present: blk.0.attn_qkv".
