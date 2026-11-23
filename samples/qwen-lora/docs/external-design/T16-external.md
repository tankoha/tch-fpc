# External design — unit `Checkpoint` (T16)

## Purpose

T16 provides the unit `Checkpoint` for saving and restoring everything a
LoRA run needs to continue exactly: the adapters (T12's `TLoraSet`), the
AdamW moments (T14's `TLoraAdamW`) and the run state (`state.json`).
The code uses fp32 on the CPU; it has no dependency on the device map, so
the same code serves the two-GPU split of the 27B model.

The acceptance criterion is "exact resume" (DESIGN 8.14): ten updates in
one go and five + resume + five give identical parameters on the CPU.

## Layout of a checkpoint directory

A checkpoint directory `<root>/ckpt-<update, 8 digits>/` holds:
- `state.json`: the run state
- `lora.ot`: the adapters (`TLoraSet.Save`)
- `moments.ot`: the AdamW moments

Example: `<root>/ckpt-00000005/` contains the three files for update 5.

The directory name format guarantees lexical order equals update order.

## Save through a temporary directory and rename

A save writes `<root>/tmp-ckpt-<update>/` first, then renames it to the
final name. A directory named `ckpt-*` with a `state.json` is always
complete; an interrupted save leaves a `tmp-ckpt-*` entry.

Before saving, stale `tmp-ckpt-*` entries (directories and files) left by
earlier interrupted saves are removed.

## Retention rule

After each save:
1. Keep the last `Keep` checkpoints by update number (the constructor's
   `AKeep` parameter, minimum 1).
2. Keep the checkpoint with the lowest finite validation loss (the earlier
   on a tie).
3. Delete every other `ckpt-*` directory.

The default retention is 3 (DESIGN 8.14).

## Keys of `state.json`

`state.json` is written by `fpjson.FormatJSON` (exact round-trip of
numbers). Keys:

| Key | Type | Meaning |
|---|---|---|
| `update` | integer | optimiser updates done |
| `epoch` | integer | epoch in progress, from 0 |
| `position` | integer | samples of this epoch's order consumed |
| `seed` | integer | the run's shuffle seed |
| `rank` | integer | of the adapters (`TLoraSet.Rank`) |
| `scale` | float | of the adapters (`TLoraSet.Scale`) |
| `config_hash` | string | the trainer's configuration hash (printable ASCII) |
| `base_sha256` | string | sha256 of the base GGUF (printable ASCII) |
| `val_loss` | float or null | validation loss; NaN when not measured; non-finite written as `null` |

`config_hash` and `base_sha256` must be printable ASCII (code points
#32..#126); the save checks this. Unknown keys are ignored when reading.

`Save` takes rank and scale from the set; `Load` refuses a set of another
rank or scale.

## Why the moments are stored by variable name

T14's moment store `I` holds `m.<J>`, `v.<J>` for variable `J` of parameter
store `I` (insertion order). A per-store file would depend on the device
map: resuming with another split would load the moments of other matrices,
silently where the shapes agree (consecutive linear blocks).

The name-keyed archive avoids this dependency. Variables are stored as
`<variable>.adam_m` and `<variable>.adam_v` (e.g.
`blk.12.attn_qkv.lora_a.adam_m`). The variable name identifies the moment
uniquely regardless of the device map.

## Errors

`ECheckpoint` is raised for:
- Non-printable ASCII in `config_hash` or `base_sha256` (before writing).
- Invalid JSON or missing/wrong-type keys in `state.json` (when reading).
- A checkpoint directory without `state.json` (when loading).
- Invalid `Keep` value (< 1).
- Cannot create the root or temporary directory.
- Cannot rename the temporary directory to the final name.
- Rank/scale mismatch between the checkpoint and the set (when loading).
- Optimiser not matching the set's stores (wrong store count or moment
  count per store).

From tch-fpc, passed through unchanged:
- Missing `lora.ot` or `moments.ot` (`ETorchNativeError` with errno 2).
- Missing variable in a file (`ETorchError`).
- Shape mismatch in a copy (`ETorchNativeError`).
