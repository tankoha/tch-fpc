# T18 External Design: qlora_merge

## Purpose

`qlora_merge` writes a merged GGUF file that contains the adapted tensors from
a fine-tuning checkpoint combined with the base model.

## Command Line

```
qlora_merge [--base-sha256 H] BASE CKPT OUT
```

- `BASE`: path to the base GGUF model file
- `CKPT`: path to the checkpoint directory (contains `lora.ot` and `state.json`)
- `OUT`: path for the output merged GGUF file
- `--base-sha256 H`: optional sha256 hash of the base model (64 hex characters)

## Merged File Contents

### Metadata
- Copied byte-for-byte from the base GGUF (header and key-value pairs)

### Tensor Directory
- Rebuilt in the base's tensor order
- Offsets recomputed for the new tensor sizes
- Alignment preserved from the base

### Adapted Tensors
- Targets: per trunk block (excluding MTP block) the linear targets
  (`attn_qkv`, `attn_gate`, `ssm_out`) or full targets (`attn_q`, `attn_k`,
  `attn_v`, `attn_output`) depending on block kind
- Each adapted tensor: base tensor dequantised to fp32, plus `s·B·A` (the
  scaled LoRA delta), quantised to Q8_0
- Dimensions preserved from the base tensor

### Non-Target Tensors
- Copied byte-for-byte from the base GGUF

## Targets

The targets are taken from `TQwenModel.AttachLora`:
- For each trunk block (blocks minus the MTP block count from
  `qwen35.nextn_predict_layers`):
  - Linear block: `attn_qkv`, `attn_gate`, `ssm_out`
  - Full block: `attn_q`, `attn_k`, `attn_v`, `attn_output`
- Each target is `<block>.<name>.weight` in the GGUF tensor directory
- Dimensions: `Dims[0]` (in), `Dims[1]` (out) from the GGUF

## Base SHA256 Check

1. If `--base-sha256 H` is given, use H
2. Otherwise, extract from the base filename: `sha256-<64 hex>`
3. If both the checkpoint's `base_sha256` (from `state.json`) and the
   computed/given sha256 are known, they must be equal (literal comparison)
4. Mismatch: error, exit 1

## Part File

- Output is written to `<OUT>.part` first
- On success, renamed to `<OUT>` (rename(2) replaces an existing file)
- On failure, the part file is removed

## Exit Codes

| Code | Meaning |
|---|---|
| 0 | Success: merged file written |
| 1 | Error: file not found, sha256 mismatch, GGUF error, write error, etc. |
| 2 | Usage error: wrong number of arguments, unknown option, missing value |

## Error Conditions

### Usage Errors (exit 2)
- `--base-sha256: value missing`
- `unknown option <arg>`
- `expected BASE CKPT OUT, got <n> arguments`

### Runtime Errors (exit 1)
- `no model file <base>`
- `the output must not be the base model`
- `not a checkpoint: <ckpt>`
- `the base model differs from the checkpoint's: <ckpt>`
- Missing tensor in base: `no tensor <name>.weight`
- Wrong tensor dimensions: `<name>.weight has <n> dimensions, expected 2`
- Missing adapter file: `open file failed because of errno 2 on fopen: No such file or directory, file path: <dir>lora.ot`
- Missing adapter: `no adapter <name>`
- Dimension mismatch: `<name>: adapter <in> -> <out>, tensor <in> -> <out>`
- Write error: `TGgufWriter.BeginWrite: cannot create file <out>.part`
- Rename error: `cannot rename <out>.part to <out>`

## Units Used

- `GgufFile` (T01): reading the base GGUF
- `GgufQuant` (T02): dequantising, quantising to Q8_0
- `GgufWriter` (T04): writing the merged file
- `QwenBlocks` (T10): reading model dimensions, block kinds
- `Lora` (T12): LoRA pair set, merged delta
- `Checkpoint` (T16): reading run state, loading adapters
- `TrainCli` (T17): `ShaFromName`

## Fidelity Criterion

On the rehearsal model (M2):
- Merged file compared against forward with adapter
- Retyping floor (same targets with B = 0) at most 1.5×
- At least 90% of the loss gain kept
- H47: bundled `llama-server` on all-F32 copy vs Pascal forward, at most 0.02
