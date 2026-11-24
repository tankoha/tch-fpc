# External design — program `qlora_train` (T17)

## Purpose

The program `qlora_train` fine-tunes a Qwen model with LoRA: it trains the
LoRA adapters on a data set, checkpoints the state, and produces a log for
monitoring. It is the harness owner's M2 acceptance test (overfitting 32
samples, validation loss falling on a small real set) and the user's
production trainer on the GPUs (M2g, M4).

## Command line

```
qlora_train [options] MODEL TRAIN OUTDIR
```

Required positional:
- `MODEL`: path to the base model (GGUF).
- `TRAIN`: path to the training data (JSONL, T15 format).
- `OUTDIR`: the output directory (created if missing; trailing delimiter optional).

Options (defaults in parentheses):
- `--rank N`: LoRA rank (16).
- `--alpha A`: LoRA alpha, combined with rank for scale (32).
- `--lr R`: base learning rate (1e-4).
- `--min-lr-ratio M`: minimum LR = M × lr (0.1).
- `--warmup N`: warm-up updates (50).
- `--epochs N`: number of epochs (3).
- `--seed N`: random seed for LoRA A matrices and epoch order (1).
- `--pool N`: samples per pool for batching (64).
- `--budget N`: token budget per batch (2048).
- `--batches-per-update N`: batches per gradient update (2).
- `--clip C`: gradient clipping norm (1.0).
- `--ckpt-every N`: checkpoint every N updates (50).
- `--keep N`: keep the last N checkpoints (3).
- `--slice N`: rows per loss slice (256).
- `--classes A,B,...`: admitted classes (`compiled-run,fpc-approx`).
- `--split N`: run on the two GPUs, trunk blocks `0..N-1` on `cuda:0`, the
  rest and the head on `cuda:1` (0..the trunk's block count, checked by
  `TQwenModel.Load`); without it, everything on the CPU (-1).
- `--max-updates N`: stop after N updates (0: no limit).
- `--resume`: continue from the latest checkpoint.
- `--valid FILE`: validation data (none).
- `--base-sha256 H`: base model's sha256 hash (from Ollama blob name if `sha256-<64 hex>`).

## Training schedule

- One sample per forward: the batch's samples are processed one after the other.
- One update: `--batches-per-update` consecutive batches, then:
  - Clip gradients: `ClipGlobalNorm(clip)`.
  - Step with scheduled LR: `ScheduleLR(lr, update, warmup, total, min-lr-ratio)`.
  - Zero gradients.
- Total updates: the sum over epochs of `ceil(batches / batches-per-update)`.
  Batches differ slightly between epochs (T15).

## Loss computation

- Each sample's loss is the summed cross-entropy of its loss tokens divided
  by the denominator (the loss tokens of the whole update).
- Slice loss: the loss rows are processed in slices of `--slice` rows; each
  slice's summed cross-entropy is divided by the denominator and sent
  backward independently.
- Non-finite loss: stops the run before the update, exit code 3.

## Checkpoints and resume

- Write to `OUTDIR/ckpt` after every `--ckpt-every` updates and at each
  epoch's end.
- At an epoch's end, the state is the next epoch's start (epoch + 1,
  position 0).
- Validation loss: computed when `--valid` is given; else NaN.
- `--resume` loads the latest checkpoint and checks:
  - Configuration hash (MD5 of train/valid contents and settings that shape
    the run; excludes `--max-updates`, `--resume`, `--split`, `--ckpt-every`,
    `--keep`, the output directory, and the base's sha256).
  - Base's sha256: from `--base-sha256` or the model file name (Ollama blob).
  - Refuses a mismatch with exit code 1.

## Log

- `OUTDIR/train-log.tsv`: tab-separated, appended line by line.
- Update line: `update epoch position loss lr norm tokens seconds`.
- Checkpoint line: `ckpt update val-loss dir`.
- The loss is the mean cross-entropy per loss token of the update.

## Exit codes

- 0: success (all epochs or `--max-updates` reached).
- 1: error (model not found, train file empty, warm-up >= total, resume
  mismatch, etc.).
- 2: usage error (bad option, wrong argument count).
- 3: non-finite loss (stopped without updating).

## Errors

In order:
- `--option: value missing`.
- `--option: an integer in <lo>..<hi>, got "<v>"`.
- `--option: a number >= 0, got "<v>"` (NaN and ±Inf refused).
- `unknown option <option>`.
- `expected MODEL TRAIN OUTDIR, got <n> arguments`.
- `no model file <model>`.
- Errors from the GGUF, tokenizer, and TrainData (no data file, no training
  sample).
- `warm-up <w> must be below the <n> updates`.
- Errors from `TQwenModel.Load` (no CUDA GPUs, split out of range).
- With `--resume`: `nothing to resume in <outdir>ckpt/`, `the configuration
  differs from the checkpoint's: <dir>`, `the base model differs from the
  checkpoint's: <dir>`.
- `non-finite loss at update <u>; stopped without updating` (exit 3).

## Outputs for T18

- The checkpoint directory `OUTDIR/ckpt` with the LoRA adapters, the
  optimizer state, and the validation loss.
- The best checkpoint's path in the exit message for T18's merge.

## Seeds

- `ManualSeed(seed)` before `AttachLora` (initialises the A matrices).
- Epoch order: TrainData's SplitMix64 from the same seed.
